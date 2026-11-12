"""SM120 register-P MMA with explicit cp.async ping-pong KV buffers."""
from functools import lru_cache
from .attention import _prefill_kernel as _reference_prefill_kernel


@lru_cache(maxsize=None)
def _prefill_kernel(b, sq, hq, hkv, d, capacity, pages, page, dtype,
                    qs, ks, vs, os, table_stride, has_lengths, paged, causal,
                    scale_log2, bm, bn, stages, threads, splits=1,
                    grouped=False, tma=False):
    """One query tile/head (prefill), or KV-head group/split (decode).

    Logical pages are translated inside the kernel. No dense gather or head
    expansion is performed. Only dense aligned inputs may select TMA.
    """
    import tilelang.cuda.language as T
    from tilelang.language import GemmWarpPolicy
    from .attention import _compile

    if (tma or grouped or splits != 1 or stages == 1 or bm < 64 or (paged and bn % page) or
            any(stride % 2 for strides in (ks, vs) for stride in strides[:-1])):
        return _reference_prefill_kernel(b, sq, hq, hkv, d, capacity, pages, page, dtype,
            qs, ks, vs, os, table_stride, has_lengths, paged, causal,
            scale_log2, bm, bn, stages, threads, splits, grouped, tma)
    group = hq // hkv
    qshape = (b, sq, hq, d)
    kshape = (pages, page, hkv, d) if paged else (b, capacity, hkv, d)
    lshape = (b,) if has_lengths else (1,)
    btshape = (b, capacity // page) if paged else (1,)
    btstrides = (table_stride, 1) if paged else (1,)
    pshape = (b, hq, splits, d) if splits > 1 else (1,)
    sshape = (b, hq, splits) if splits > 1 else (1,)
    grid_m = 1 if grouped else (sq + bm - 1) // bm
    grid_h = hkv if grouped else hq
    chunk = (capacity + splits - 1) // splits

    @T.prim_func
    def kernel(Q: T.StridedTensor(qshape, qs, dtype),
               K: T.StridedTensor(kshape, ks, dtype),
               V: T.StridedTensor(kshape, vs, dtype),
               Lengths: T.Tensor(lshape, "int32"),
               Table: T.StridedTensor(btshape, btstrides, "int32"),
               Out: T.StridedTensor(qshape, os, dtype),
               Partial: T.Tensor(pshape, "float32"),
               Maxima: T.Tensor(sshape, "float32"),
               Sums: T.Tensor(sshape, "float32")):
        with T.Kernel(grid_m, b * grid_h, splits, threads=threads) as (bx, bh, split):
            batch = bh // grid_h
            head = bh % grid_h
            kvhead = head if grouped else head // group
            length = Lengths[batch] if has_lengths else capacity
            lo = split * chunk
            stop = T.min(length, lo + chunk)
            if causal and not grouped:
                stop = T.max(0, T.min(stop, (bx + 1) * bm + length - sq))
            Qs = T.alloc_shared((bm, d), dtype)
            Ks = T.alloc_shared((2, bn, d), dtype)
            Vs = T.alloc_shared((2, bn, d), dtype)
            scores = T.alloc_fragment((bm, bn), "float32")
            # Keep P in registers; FullRow makes the row partition compatible
            # between softmax and the register/shared PV MMA.
            probs = T.alloc_fragment((bm, bn), dtype)
            acc = T.alloc_fragment((bm, d), "float32")
            maximum = T.alloc_fragment((bm,), "float32")
            updated = T.alloc_fragment((bm,), "float32")
            denom = T.alloc_fragment((bm,), "float32")
            rowsum = T.alloc_fragment((bm,), "float32")
            correction = T.alloc_fragment((bm,), "float32")
            T.clear(acc)
            T.clear(denom)
            T.fill(maximum, -T.infinity("float32"))
            if tma:
                # Explicit regions keep the tile on the sequence axis of
                # BSHD; scalar-address copy sugar would span the head axis.
                T.copy(Q[batch:batch + 1, bx * bm:(bx + 1) * bm, head:head + 1, 0:d],
                       Qs, prefer_instruction="tma")
            else:
                for i, z in T.Parallel(bm, d):
                    if grouped:
                        Qs[i, z] = T.if_then_else(i < group, Q[batch, 0, head * group + i, z], 0)
                    else:
                        Qs[i, z] = T.if_then_else(bx * bm + i < sq, Q[batch, bx * bm + i, head, z], 0)
            blocks = T.ceildiv(T.max(stop-lo,0),bn)
            if blocks > 0:
                if lo + bn <= stop:
                    if paged:
                        for pg in T.serial(bn // page):
                            physical = Table[batch, lo // page + pg]
                            T.async_copy(K[physical:physical+1, 0:page, kvhead:kvhead+1, 0:d],
                                         Ks[0:0+1, pg*page:(pg+1)*page, 0:d], annotations={"no_implicit_async_commit_wait": True})
                            T.async_copy(V[physical:physical+1, 0:page, kvhead:kvhead+1, 0:d],
                                         Vs[0:0+1, pg*page:(pg+1)*page, 0:d], annotations={"no_implicit_async_commit_wait": True})
                    else:
                        T.async_copy(K[batch:batch+1, lo:lo+bn, kvhead:kvhead+1, 0:d],
                                     Ks[0:0+1, 0:bn, 0:d], annotations={"no_implicit_async_commit_wait": True})
                        T.async_copy(V[batch:batch+1, lo:lo+bn, kvhead:kvhead+1, 0:d],
                                     Vs[0:0+1, 0:bn, 0:d], annotations={"no_implicit_async_commit_wait": True})
                    T.ptx_commit_group()
                else:
                    for j, z in T.Parallel(bn, d):
                        if lo + j < stop:
                            if paged:
                                physical = Table[batch, (lo+j)//page]
                                Ks[0,j,z] = K[physical,(lo+j)%page,kvhead,z]
                                Vs[0,j,z] = V[physical,(lo+j)%page,kvhead,z]
                            else:
                                Ks[0,j,z] = K[batch,lo+j,kvhead,z]
                                Vs[0,j,z] = V[batch,lo+j,kvhead,z]
                        else:
                            Ks[0,j,z] = 0
                            Vs[0,j,z] = 0
            for block in T.serial(blocks):
                slot = block % 2
                start = lo + block * bn
                T.ptx_wait_group(0)
                T.sync_threads()
                if block + 1 < blocks:
                    if (start+bn) + bn <= stop:
                        if paged:
                            for pg in T.serial(bn // page):
                                physical = Table[batch, (start+bn) // page + pg]
                                T.async_copy(K[physical:physical+1, 0:page, kvhead:kvhead+1, 0:d],
                                             Ks[((block+1)%2):((block+1)%2)+1, pg*page:(pg+1)*page, 0:d], annotations={"no_implicit_async_commit_wait": True})
                                T.async_copy(V[physical:physical+1, 0:page, kvhead:kvhead+1, 0:d],
                                             Vs[((block+1)%2):((block+1)%2)+1, pg*page:(pg+1)*page, 0:d], annotations={"no_implicit_async_commit_wait": True})
                        else:
                            T.async_copy(K[batch:batch+1, (start+bn):(start+bn)+bn, kvhead:kvhead+1, 0:d],
                                         Ks[((block+1)%2):((block+1)%2)+1, 0:bn, 0:d], annotations={"no_implicit_async_commit_wait": True})
                            T.async_copy(V[batch:batch+1, (start+bn):(start+bn)+bn, kvhead:kvhead+1, 0:d],
                                         Vs[((block+1)%2):((block+1)%2)+1, 0:bn, 0:d], annotations={"no_implicit_async_commit_wait": True})
                        T.ptx_commit_group()
                    else:
                        for j, z in T.Parallel(bn, d):
                            if (start+bn) + j < stop:
                                if paged:
                                    physical = Table[batch, ((start+bn)+j)//page]
                                    Ks[((block+1)%2),j,z] = K[physical,((start+bn)+j)%page,kvhead,z]
                                    Vs[((block+1)%2),j,z] = V[physical,((start+bn)+j)%page,kvhead,z]
                                else:
                                    Ks[((block+1)%2),j,z] = K[batch,(start+bn)+j,kvhead,z]
                                    Vs[((block+1)%2),j,z] = V[batch,(start+bn)+j,kvhead,z]
                            else:
                                Ks[((block+1)%2),j,z] = 0
                                Vs[((block+1)%2),j,z] = 0
                T.clear(scores)
                T.gemm(Qs, Ks[slot:slot+1,0:bn,0:d], scores, transpose_B=True, policy=GemmWarpPolicy.FullRow)
                # A uniform CTA test removes per-element causal/tail masks
                # from tiles wholly before the first query's causal boundary.
                if start + bn <= stop and (not causal or grouped or
                        start + bn <= bx * bm + length - sq + 1):
                    for i, j in T.Parallel(bm, bn):
                        scores[i, j] *= scale_log2
                else:
                    for i, j in T.Parallel(bm, bn):
                        valid = start + j < stop
                        if causal and not grouped:
                            valid = valid & (start + j <= bx * bm + i + length - sq)
                        scores[i, j] = T.if_then_else(valid, scores[i, j] * scale_log2, -T.infinity("float32"))
                T.reduce_max(scores, updated, dim=1, batch=2)
                for i in T.Parallel(bm):
                    updated[i] = T.max(maximum[i], updated[i])
                    correction[i] = T.exp2(maximum[i] - T.if_then_else(updated[i] == -T.infinity("float32"), 0, updated[i]))
                for i, j in T.Parallel(bm, bn):
                    scores[i, j] = T.exp2(scores[i, j] - T.if_then_else(updated[i] == -T.infinity("float32"), 0, updated[i]))
                T.reduce_sum(scores, rowsum, dim=1, batch=2)
                for i in T.Parallel(bm):
                    denom[i] = denom[i] * correction[i] + rowsum[i]
                    maximum[i] = updated[i]
                for i, z in T.Parallel(bm, d):
                    acc[i, z] *= correction[i]
                T.copy(scores, probs)
                T.gemm(probs, Vs[slot:slot+1,0:bn,0:d], acc, policy=GemmWarpPolicy.FullRow)
            for i, z in T.Parallel(bm, d):
                if grouped:
                    if i < group:
                        if splits > 1:
                            Partial[batch, head * group + i, split, z] = acc[i, z]
                        else:
                            Out[batch, 0, head * group + i, z] = acc[i, z] / T.max(denom[i], T.float32(1e-30))
                else:
                    if bx * bm + i < sq:
                        Out[batch, bx * bm + i, head, z] = acc[i, z] / T.max(denom[i], T.float32(1e-30))
            if splits > 1:
                for i in T.Parallel(bm):
                    if i < group:
                        Maxima[batch, head * group + i, split] = maximum[i]
                        Sums[batch, head * group + i, split] = denom[i]
    # The loop-top wait and CTA barrier publish Q/K/V and protect the two
    # KV slots from reuse until every warp has finished the preceding PV.
    # P stays in registers, so it needs no shared producer/consumer barrier.
    return _compile(kernel, tma=tma, manual_bounds=True, manual_sync=True)


