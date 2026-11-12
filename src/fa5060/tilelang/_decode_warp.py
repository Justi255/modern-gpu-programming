"""SIMT decode with independent online-softmax states in each warp.

Only the final per-CTA epilogue exchanges states through shared memory. The
factory accepts the original vector kernel arguments and writes its FP32 split
state protocol, including the unnormalized partial output.
"""
from functools import lru_cache


@lru_cache(maxsize=None)
def _decode_vector_kernel(b, hq, hkv, d, capacity, pages, page, dtype,
                          qs, ks, vs, os, table_stride, paged, scale_log2,
                          bn, threads, splits):
    """Accumulate independent warp states, then merge them stably once."""
    import tilelang.language as T
    from .attention import _compile

    group = hq // hkv
    warps = threads // 32
    tokens = bn // warps
    vector = d // 32
    kshape = (pages, page, hkv, d) if paged else (b, capacity, hkv, d)
    btshape = (b, capacity // page) if paged else (1,)
    btstrides = (table_stride, 1) if paged else (1,)
    pshape = (b, hq, splits, d) if splits > 1 else (1,)
    sshape = (b, hq, splits) if splits > 1 else (1,)
    chunk = (capacity + splits - 1) // splits
    aligned_pages = chunk % page == 0 and bn % page == 0
    physical_stride = page if aligned_pages else 1
    physical_entries = max(1, tokens // page) if aligned_pages else tokens

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
            values = T.alloc_fragment((warps, tokens, d), "float32")
            scores = T.alloc_fragment((warps, tokens), "float32")
            p = T.alloc_fragment((warps, tokens), "float32")
            acc = T.alloc_fragment((warps, d), "float32")
            weighted = T.alloc_fragment((warps, d), "float32")
            maximum = T.alloc_fragment((warps,), "float32")
            updated = T.alloc_fragment((warps,), "float32")
            denom = T.alloc_fragment((warps,), "float32")
            psum = T.alloc_fragment((warps,), "float32")
            alpha = T.alloc_fragment((warps,), "float32")
            warp_partial = T.alloc_shared((warps, d), "float32")
            warp_maxima = T.alloc_shared((warps,), "float32")
            warp_sums = T.alloc_shared((warps,), "float32")
            merged_max = T.alloc_local((1,), "float32")
            merged_sum = T.alloc_local((1,), "float32")
            merge_weight = T.alloc_local((1,), "float32")
            physical = T.alloc_fragment((warps, physical_entries), "int32")
            result = T.alloc_fragment((d,), "float32")
            # Contiguous D/32 elements per lane make D64/D128 loads eligible
            # for packed half2/half4 accesses. Tokens stay within their warp.
            T.annotate_layout({
                query: T.Fragment((d,), replicate=warps,
                    forward_thread_fn=lambda z, r: r * 32 + z // vector,
                    forward_index_fn=lambda z: z % vector),
                values: T.Fragment((warps, tokens, d),
                    forward_thread_fn=lambda w, j, z: w * 32 + z // vector,
                    forward_index_fn=lambda w, j, z: j * vector + z % vector),
                scores: T.Fragment((warps, tokens), replicate=32,
                    forward_thread_fn=lambda w, j, r: w * 32 + r,
                    forward_index_fn=lambda w, j: j),
                p: T.Fragment((warps, tokens), replicate=32,
                    forward_thread_fn=lambda w, j, r: w * 32 + r,
                    forward_index_fn=lambda w, j: j),
                physical: T.Fragment((warps, physical_entries), replicate=32,
                    forward_thread_fn=lambda w, j, r: w * 32 + r,
                    forward_index_fn=lambda w, j: j),
                acc: T.Fragment((warps, d),
                    forward_thread_fn=lambda w, z: w * 32 + z // vector,
                    forward_index_fn=lambda w, z: z % vector),
                weighted: T.Fragment((warps, d),
                    forward_thread_fn=lambda w, z: w * 32 + z // vector,
                    forward_index_fn=lambda w, z: z % vector),
                maximum: T.Fragment((warps,), replicate=32,
                    forward_thread_fn=lambda w, r: w * 32 + r,
                    forward_index_fn=lambda w: 0),
                updated: T.Fragment((warps,), replicate=32,
                    forward_thread_fn=lambda w, r: w * 32 + r,
                    forward_index_fn=lambda w: 0),
                denom: T.Fragment((warps,), replicate=32,
                    forward_thread_fn=lambda w, r: w * 32 + r,
                    forward_index_fn=lambda w: 0),
                psum: T.Fragment((warps,), replicate=32,
                    forward_thread_fn=lambda w, r: w * 32 + r,
                    forward_index_fn=lambda w: 0),
                alpha: T.Fragment((warps,), replicate=32,
                    forward_thread_fn=lambda w, r: w * 32 + r,
                    forward_index_fn=lambda w: 0),
                result: T.Fragment((d,),
                    forward_thread_fn=lambda z: z // vector,
                    forward_index_fn=lambda z: z % vector),
            })
            for z in T.Parallel(d):
                query[z] = Q[batch, h, z]
            T.clear(acc)
            T.clear(denom)
            T.fill(maximum, -T.infinity("float32"))
            for block in T.serial(T.ceildiv(T.max(stop - lo, 0), bn)):
                start = lo + block * bn
                if paged:
                    # A materialized register avoids the compiler repeating
                    # the page-table load in every K/V bounds check and cast.
                    # Aligned tiles need one register per physical page;
                    # arbitrary split alignment uses a register per token.
                    for w, j in T.Parallel(warps, physical_entries):
                        position = start + w * tokens + j * physical_stride
                        if position < stop:
                            physical[w, j] = Table[batch, position // page]
                        else:
                            physical[w, j] = 0
                for w, j, z in T.Parallel(warps, tokens, d):
                    position = start + w * tokens + j
                    if position < stop:
                        if paged:
                            values[w, j, z] = T.cast(K[physical[w, j // physical_stride], position % page, h // group, z], "float32") * query[z]
                        else:
                            values[w, j, z] = T.cast(K[batch, position, h // group, z], "float32") * query[z]
                    else:
                        values[w, j, z] = 0
                T.reduce_sum(values, scores, dim=2)
                for w, j in T.Parallel(warps, tokens):
                    scores[w, j] = T.if_then_else(start + w * tokens + j < stop,
                        scores[w, j] * scale_log2, -T.infinity("float32"))
                T.reduce_max(scores, updated, dim=1)
                for w in T.Parallel(warps):
                    updated[w] = T.max(maximum[w], updated[w])
                    alpha[w] = T.exp2(maximum[w] - T.if_then_else(updated[w] == -T.infinity("float32"), 0, updated[w]))
                    maximum[w] = updated[w]
                for w, j in T.Parallel(warps, tokens):
                    p[w, j] = T.exp2(scores[w, j] - T.if_then_else(maximum[w] == -T.infinity("float32"), 0, maximum[w]))
                T.reduce_sum(p, psum, dim=1)
                for w, j, z in T.Parallel(warps, tokens, d):
                    position = start + w * tokens + j
                    if position < stop:
                        if paged:
                            values[w, j, z] = p[w, j] * T.cast(V[physical[w, j // physical_stride], position % page, h // group, z], "float32")
                        else:
                            values[w, j, z] = p[w, j] * T.cast(V[batch, position, h // group, z], "float32")
                    else:
                        values[w, j, z] = 0
                T.reduce_sum(values, weighted, dim=1)
                for w, z in T.Parallel(warps, d):
                    acc[w, z] = acc[w, z] * alpha[w] + weighted[w, z]
                for w in T.Parallel(warps):
                    denom[w] = denom[w] * alpha[w] + psum[w]
            T.copy(acc, warp_partial)
            T.copy(maximum, warp_maxima)
            T.copy(denom, warp_sums)
            T.sync_threads()
            # Four/eight warp states are tiny. Serial scalar reductions avoid
            # CTA collectives and are replicated in each output lane. Empty
            # warps get zero weight even when every warp is empty.
            merged_max[0] = -T.infinity("float32")
            for w in T.serial(warps):
                merged_max[0] = T.max(merged_max[0], warp_maxima[w])
            merged_sum[0] = 0
            for w in T.serial(warps):
                merge_weight[0] = T.exp2(warp_maxima[w] - T.if_then_else(merged_max[0] == -T.infinity("float32"), 0, merged_max[0]))
                merged_sum[0] += merge_weight[0] * warp_sums[w]
            for z in T.Parallel(d):
                result[z] = 0
                for w in T.serial(warps):
                    result[z] += T.exp2(warp_maxima[w] - T.if_then_else(merged_max[0] == -T.infinity("float32"), 0, merged_max[0])) * warp_partial[w, z]
            for z in T.Parallel(d):
                if splits > 1:
                    Partial[batch, h, split, z] = result[z]
                else:
                    Out[batch, h, z] = result[z] / T.max(merged_sum[0], T.float32(1e-30))
            if splits > 1:
                if T.get_thread_binding() == 0:
                    Maxima[batch, h, split] = merged_max[0]
                    Sums[batch, h, split] = merged_sum[0]
    return _compile(kernel)
