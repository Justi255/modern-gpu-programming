"""Decode SIMT candidate with tokens distributed across warps.

The public backend keeps the original vector kernel as a reference. This
factory has the same signature and split-state protocol, so the dispatcher can
select it without changing workspace ownership or the numerical contract.
"""
from functools import lru_cache


@lru_cache(maxsize=None)
def _decode_vector_kernel(b, hq, hkv, d, capacity, pages, page, dtype,
                          qs, ks, vs, os, table_stride, paged, scale_log2,
                          bn, threads, splits):
    """FP32 SIMT online softmax with one token stream per warp."""
    import tilelang.language as T
    from .attention import _compile

    group = hq // hkv
    warps = threads // 32
    kshape = (pages, page, hkv, d) if paged else (b, capacity, hkv, d)
    btshape = (b, capacity // page) if paged else (1,)
    btstrides = (table_stride, 1) if paged else (1,)
    pshape = (b, hq, splits, d) if splits > 1 else (1,)
    sshape = (b, hq, splits) if splits > 1 else (1,)
    chunk = (capacity + splits - 1) // splits

    @T.prim_func
    def kernel(Q: T.StridedTensor((b, hq, d), qs, dtype),
               K: T.StridedTensor(kshape, ks, dtype),
               V: T.StridedTensor(kshape, vs, dtype),
               Lengths: T.Tensor((b,), "int32"),
               Table: T.StridedTensor(btshape, btstrides, "int32"),
               Out: T.StridedTensor((b, hq, d), os, dtype),
               Partial: T.Tensor(pshape, "float32"),
               Maxima: T.Tensor(sshape, "float32"),
               Sums: T.Tensor(sshape, "float32")):
        with T.Kernel(hq, b, splits, threads=threads) as (h, batch, split):
            lo = split * chunk
            stop = T.min(Lengths[batch], lo + chunk)
            query = T.alloc_fragment((d,), "float32")
            values = T.alloc_fragment((bn, d), "float32")
            scores = T.alloc_fragment((bn,), "float32")
            p = T.alloc_fragment((bn,), "float32")
            acc = T.alloc_fragment((d,), "float32")
            weighted = T.alloc_fragment((d,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            updated = T.alloc_fragment((1,), "float32")
            denom = T.alloc_fragment((1,), "float32")
            psum = T.alloc_fragment((1,), "float32")
            alpha = T.alloc_fragment((1,), "float32")
            # Reducing D stays inside a warp. Reducing tokens combines the
            # independent warp streams once, instead of one CTA barrier per
            # token. The D-vectors are equal copies in all warps.
            T.annotate_layout({
                values: T.Fragment((bn, d),
                    forward_thread_fn=lambda j, z: (j % warps) * 32 + z % 32,
                    forward_index_fn=lambda j, z: (j // warps) * (d // 32) + z // 32),
                query: T.Fragment((d,), replicate=warps,
                    forward_thread_fn=lambda z, r: r * 32 + z % 32,
                    forward_index_fn=lambda z: z // 32),
                acc: T.Fragment((d,), replicate=warps,
                    forward_thread_fn=lambda z, r: r * 32 + z % 32,
                    forward_index_fn=lambda z: z // 32),
                weighted: T.Fragment((d,), replicate=warps,
                    forward_thread_fn=lambda z, r: r * 32 + z % 32,
                    forward_index_fn=lambda z: z // 32),
            })
            for z in T.Parallel(d):
                query[z] = Q[batch, h, z]
            T.clear(acc)
            T.clear(denom)
            T.fill(maximum, -T.infinity("float32"))
            for block in T.serial(T.ceildiv(T.max(stop - lo, 0), bn)):
                start = lo + block * bn
                for j, z in T.Parallel(bn, d):
                    if start + j < stop:
                        if paged:
                            physical = Table[batch, (start + j) // page]
                            values[j, z] = T.cast(K[physical, (start + j) % page, h // group, z], "float32") * query[z]
                        else:
                            values[j, z] = T.cast(K[batch, start + j, h // group, z], "float32") * query[z]
                    else:
                        values[j, z] = 0
                T.reduce_sum(values, scores, dim=1)
                for j in T.Parallel(bn):
                    scores[j] = T.if_then_else(start + j < stop, scores[j] * scale_log2, -T.infinity("float32"))
                T.reduce_max(scores, updated, dim=0)
                updated[0] = T.max(maximum[0], updated[0])
                alpha[0] = T.exp2(maximum[0] - T.if_then_else(updated[0] == -T.infinity("float32"), 0, updated[0]))
                maximum[0] = updated[0]
                for j in T.Parallel(bn):
                    p[j] = T.exp2(scores[j] - T.if_then_else(maximum[0] == -T.infinity("float32"), 0, maximum[0]))
                T.reduce_sum(p, psum, dim=0)
                for j, z in T.Parallel(bn, d):
                    if start + j < stop:
                        if paged:
                            physical = Table[batch, (start + j) // page]
                            values[j, z] = p[j] * T.cast(V[physical, (start + j) % page, h // group, z], "float32")
                        else:
                            values[j, z] = p[j] * T.cast(V[batch, start + j, h // group, z], "float32")
                    else:
                        values[j, z] = 0
                T.reduce_sum(values, weighted, dim=0)
                for z in T.Parallel(d):
                    acc[z] = acc[z] * alpha[0] + weighted[z]
                denom[0] = denom[0] * alpha[0] + psum[0]
            for z in T.Parallel(d):
                if splits > 1:
                    Partial[batch, h, split, z] = acc[z]
                else:
                    Out[batch, h, z] = acc[z] / T.max(denom[0], T.float32(1e-30))
            if splits > 1:
                Maxima[batch, h, split] = maximum[0]
                Sums[batch, h, split] = denom[0]
    return _compile(kernel)
