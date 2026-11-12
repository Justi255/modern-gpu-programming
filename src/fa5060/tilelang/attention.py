"""SM120 TileLang attention: native paging, online softmax and FP32 split states.

PyTorch supplies tensors and allocations only. Kernel factories are cached by
shape, stride, dtype and configuration; compilation precedes graph capture.
"""
from functools import lru_cache
import math

import torch
from .._workspace import check_workspace_disjoint

from ._contracts import (_validate, _output, _configuration, _decode_parameters,
                         _scale, DecodeWorkspace, decode_workspace)


@lru_cache(maxsize=None)
def _dummy(device, floating=False):
    return torch.empty((1,), dtype=torch.float32 if floating else torch.int32, device=device)


def _compile(prim, *, tma=False, manual_bounds=False, manual_sync=False, scalar_memory=False):
    import tilelang
    passes = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
    if scalar_memory:
        # Torch views may begin at a two-byte storage offset. The compiler's
        # default tensor alignment is stronger, so use scalar global traffic
        # and synchronous copies for this metadata-selected fallback.
        passes[tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE] = True
        passes[tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY] = False
        passes[tilelang.PassConfigKey.TIR_USE_ASYNC_COPY] = False
    if manual_bounds:
        # These factories predicate logical tails explicitly. Live page indices
        # obey the public metadata contract, so duplicating their range checks
        # at every K/V use only repeats address-dependent loads.
        passes[tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS] = True
    if manual_sync:
        # Only factories with explicit producer/consumer and buffer-reuse
        # barriers may opt out of automatic shared-memory synchronization.
        passes[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = True
    if tma:
        # 0.1.15's pipeline planner otherwise rewrites even explicit TMA
        # preferences into cp.async; a dynamic KV trip count then produces
        # an invalid nonconstant wait_group immediate. Keep the staged loop
        # and let each explicit TMA copy lower with its own mbarrier/wait.
        passes[tilelang.PassConfigKey.TIR_USE_ASYNC_COPY] = False
    return tilelang.compile(prim, target={"kind": "cuda", "arch": "sm_120"},
        execution_backend="tvm_ffi", pass_configs=passes)


@lru_cache(maxsize=None)
def _zero_output(shape, strides, dtype, scalar_memory=False):
    """Empty KV has no softmax reductions or tensor-map descriptors."""
    import tilelang.language as T
    b, sq, hq, d = shape
    count = b * sq * hq * d
    @T.prim_func
    def kernel(Out: T.StridedTensor(shape, strides, dtype)):
        with T.Kernel(T.ceildiv(count, 256), threads=128) as block:
            for i in T.Parallel(256):
                index = block * 256 + i
                if index < count:
                    Out[index // (sq * hq * d), index // (hq * d) % sq,
                        index // d % hq, index % d] = 0
    return _compile(kernel, scalar_memory=scalar_memory)


@lru_cache(maxsize=None)
def _prefill_kernel(b, sq, hq, hkv, d, capacity, pages, page, dtype,
                    qs, ks, vs, os, table_stride, has_lengths, paged, causal,
                    scale_log2, bm, bn, stages, threads, splits=1,
                    grouped=False, tma=False, scalar_memory=False):
    """One query tile/head (prefill), or KV-head group/split (decode).

    Logical pages are translated inside the kernel. No dense gather or head
    expansion is performed. Only dense aligned inputs may select TMA.
    """
    import tilelang.cuda.language as T

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
            Ks = T.alloc_shared((bn, d), dtype)
            Vs = T.alloc_shared((bn, d), dtype)
            scores = T.alloc_fragment((bm, bn), "float32")
            # The two GEMMs can require different register layouts. Shared
            # staging makes this conversion explicit and avoids forcing the
            # score accumulator into the second GEMM's A-fragment layout.
            probs = T.alloc_shared((bm, bn), dtype)
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
            for block in T.Pipelined(T.ceildiv(T.max(stop - lo, 0), bn), num_stages=stages):
                start = lo + block * bn
                if tma:
                    T.copy(K[batch:batch + 1, start:start + bn, kvhead:kvhead + 1, 0:d],
                           Ks, prefer_instruction="tma")
                    # Scores for non-live keys are masked after QK. Values
                    # must instead be zeroed before PV: 0*NaN is still NaN.
                    # A predicated V copy avoids modifying a TMA destination
                    # twice in the pipeline and needs no second V buffer.
                    for j, z in T.Parallel(bn, d):
                        if start + j < stop:
                            Vs[j, z] = V[batch, start + j, kvhead, z]
                        else:
                            Vs[j, z] = 0
                else:
                    for j, z in T.Parallel(bn, d):
                        if start + j < stop:
                            if paged:
                                physical = Table[batch, (start + j) // page]
                                Ks[j, z] = K[physical, (start + j) % page, kvhead, z]
                                Vs[j, z] = V[physical, (start + j) % page, kvhead, z]
                            else:
                                Ks[j, z] = K[batch, start + j, kvhead, z]
                                Vs[j, z] = V[batch, start + j, kvhead, z]
                        else:
                            Ks[j, z] = 0
                            Vs[j, z] = 0
                T.clear(scores)
                T.gemm(Qs, Ks, scores, transpose_B=True)
                for i, j in T.Parallel(bm, bn):
                    valid = start + j < stop
                    if causal and not grouped:
                        valid = valid & (start + j <= bx * bm + i + length - sq)
                    scores[i, j] = T.if_then_else(valid, scores[i, j] * scale_log2, -T.infinity("float32"))
                T.reduce_max(scores, updated, dim=1)
                for i in T.Parallel(bm):
                    updated[i] = T.max(maximum[i], updated[i])
                    correction[i] = T.exp2(maximum[i] - T.if_then_else(updated[i] == -T.infinity("float32"), 0, updated[i]))
                for i, j in T.Parallel(bm, bn):
                    scores[i, j] = T.exp2(scores[i, j] - T.if_then_else(updated[i] == -T.infinity("float32"), 0, updated[i]))
                T.reduce_sum(scores, rowsum, dim=1)
                for i in T.Parallel(bm):
                    denom[i] = denom[i] * correction[i] + rowsum[i]
                    maximum[i] = updated[i]
                for i, z in T.Parallel(bm, d):
                    acc[i, z] *= correction[i]
                T.copy(scores, probs)
                T.gemm(probs, Vs, acc)
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
    return _compile(kernel, tma=tma, scalar_memory=scalar_memory)


@lru_cache(maxsize=None)
def _decode_vector_kernel(b, hq, hkv, d, capacity, pages, page, dtype,
                          qs, ks, vs, os, table_stride, paged, scale_log2,
                          bn, threads, splits, scalar_memory=False):
    """FP32 SIMT dot/reduction; no Tensor Core padding for a single query."""
    import tilelang.language as T
    group = hq // hkv
    kshape = (pages, page, hkv, d) if paged else (b, capacity, hkv, d)
    btshape = (b, capacity // page) if paged else (1,)
    btstrides = (table_stride, 1) if paged else (1,)
    pshape = (b, hq, splits, d) if splits > 1 else (1,)
    sshape = (b, hq, splits) if splits > 1 else (1,)
    chunk = (capacity + splits - 1) // splits

    @T.prim_func
    def kernel(Q: T.StridedTensor((b, hq, d), qs, dtype),
               K: T.StridedTensor(kshape, ks, dtype), V: T.StridedTensor(kshape, vs, dtype),
               Lengths: T.Tensor((b,), "int32"),
               Table: T.StridedTensor(btshape, btstrides, "int32"),
               Out: T.StridedTensor((b, hq, d), os, dtype),
               Partial: T.Tensor(pshape, "float32"), Maxima: T.Tensor(sshape, "float32"),
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
                alpha[0] = T.exp2(maximum[0] - updated[0])
                maximum[0] = updated[0]
                for j in T.Parallel(bn):
                    p[j] = T.exp2(scores[j] - maximum[0])
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
    return _compile(kernel, scalar_memory=scalar_memory)


def _decode_grouped_kernel(*args, **kwargs):
    """Use the shared online-softmax Tensor Core implementation for GQA."""
    return _prefill_kernel(*args, grouped=True, **kwargs)


@lru_cache(maxsize=None)
def _merge_splits(b, hq, d, splits, dtype, os, scalar_memory=False):
    """Merge unnormalized FP32 states without exp(-inf - -inf)."""
    import tilelang.language as T
    @T.prim_func
    def kernel(Partial: T.Tensor((b, hq, splits, d), "float32"),
               Maxima: T.Tensor((b, hq, splits), "float32"),
               Sums: T.Tensor((b, hq, splits), "float32"),
               Out: T.StridedTensor((b, hq, d), os, dtype)):
        with T.Kernel(hq, b, threads=128) as (h, batch):
            maximum = T.alloc_fragment((splits,), "float32")
            global_max = T.alloc_fragment((1,), "float32")
            weights = T.alloc_fragment((splits,), "float32")
            normalizer = T.alloc_fragment((1,), "float32")
            values = T.alloc_fragment((splits, d), "float32")
            result = T.alloc_fragment((d,), "float32")
            for s in T.Parallel(splits):
                maximum[s] = Maxima[batch, h, s]
            T.reduce_max(maximum, global_max, dim=0)
            for s in T.Parallel(splits):
                maximum[s] = T.exp2(maximum[s] - T.if_then_else(global_max[0] == -T.infinity("float32"), 0, global_max[0]))
                weights[s] = maximum[s] * Sums[batch, h, s]
            T.reduce_sum(weights, normalizer, dim=0)
            for s, z in T.Parallel(splits, d):
                values[s, z] = maximum[s] * Partial[batch, h, s, z]
            T.reduce_sum(values, result, dim=0)
            for z in T.Parallel(d):
                Out[batch, h, z] = result[z] / T.max(normalizer[0], T.float32(1e-30))
    return _compile(kernel, scalar_memory=scalar_memory)


def prefill(q, k, v, *, causal=True, scale=None, kv_lens=None, block_table=None,
            out=None, config=None):
    """BSHD inference attention, right-aligned causal, native dense/paged KV."""
    b, hq, hkv, d, capacity = _validate(q, k, v, kv_lens, block_table, 4)
    c = _configuration(config)
    if c['implementation'] == 'optimized':
        explicit = config or {}
        if 'block_m' not in explicit:
            c['block_m'] = 64
        if 'block_n' not in explicit:
            c['block_n'] = 32 if d == 128 else 64
    output = _output(q, k, v, out)
    scalar_memory = any(x.data_ptr() % 16 for x in (q, k, v, output))
    if q.numel() == 0:
        return output
    tma = c["loader"] == "tma"
    if tma:
        if block_table is not None or d not in (64, 128):
            raise ValueError("TMA requires dense KV and D=64/128")
        if any(not x.is_contiguous() or x.data_ptr() % 16 for x in (q, k, v)):
            raise ValueError("TMA inputs must be contiguous and 16-byte aligned")
    if 4*c["block_n"]*d*c["num_stages"] + 2*c["block_m"]*d > 96*1024:
        raise ValueError("configuration exceeds SM120 shared memory budget")
    with torch.cuda.device(q.device):
        if capacity == 0:
            _zero_output(tuple(output.shape), tuple(output.stride()), str(q.dtype).split(".")[-1], scalar_memory)(output)
            return output
        ints = _dummy(q.device)
        floats = _dummy(q.device, True)
        factory = _prefill_kernel
        if (c['implementation'] == 'optimized' and not tma and not scalar_memory and
                c['block_m'] >= 16*c['num_warps']):
            if c['num_stages'] >= 2:
                from ._prefill_qasync import _prefill_kernel as factory
            else:
                from ._prefill_qregister import _prefill_kernel as factory
        extra = {'scalar_memory': True} if scalar_memory else {}
        kernel = factory(b, q.shape[1], hq, hkv, d, capacity, k.shape[0],
            k.shape[1], str(q.dtype).split(".")[-1], tuple(q.stride()), tuple(k.stride()),
            tuple(v.stride()), tuple(output.stride()), 0 if block_table is None else block_table.stride(0),
            kv_lens is not None, block_table is not None, bool(causal), _scale(d, scale),
            c["block_m"], c["block_n"], c["num_stages"], c["num_warps"]*32, tma=tma, **extra)
        kernel(q, k, v, ints if kv_lens is None else kv_lens,
               ints if block_table is None else block_table, output, floats, floats, floats)
    return output


def decode(q, k, v, *, kv_lens, block_table=None, scale=None, out=None,
           workspace=None, num_splits=None, config=None):
    """BHD decode; all split states and merge arithmetic are FP32."""
    b, hq, hkv, d, capacity = _validate(q, k, v, kv_lens, block_table, 3)
    c, mode, splits = _decode_parameters(q, k, block_table, num_splits, config)
    output = _output(q, k, v, out)
    scalar_memory = any(x.data_ptr() % 16 for x in (q, k, v, output))
    if not b:
        return output
    with torch.cuda.device(q.device):
        if capacity == 0:
            output4 = output.unsqueeze(1)
            _zero_output(tuple(output4.shape), tuple(output4.stride()), str(q.dtype).split(".")[-1], scalar_memory)(output4)
            return output
        if splits > 1:
            if workspace is None:
                workspace = decode_workspace(q, k, block_table=block_table, num_splits=splits, config=c)
            shape = (b, hq, splits)
            if (not isinstance(workspace, DecodeWorkspace) or workspace.num_splits != splits or
                workspace.partial.shape != (*shape, d) or workspace.maxima.shape != shape or workspace.sums.shape != shape):
                raise ValueError("workspace shape/split count does not match this decode")
            for x in (workspace.partial, workspace.maxima, workspace.sums):
                if x.device != q.device or x.dtype != torch.float32 or not x.is_contiguous():
                    raise ValueError("workspace must contain contiguous CUDA FP32 tensors")
            check_workspace_disjoint(workspace, (q, k, v, kv_lens, block_table, output))
            partial, maxima, sums = workspace.partial, workspace.maxima, workspace.sums
            scalar_memory = scalar_memory or any(x.data_ptr() % 16 for x in (partial, maxima, sums))
        else:
            partial = maxima = sums = _dummy(q.device, True)
        table = _dummy(q.device) if block_table is None else block_table
        dtype = str(q.dtype).split(".")[-1]
        ts = 0 if block_table is None else block_table.stride(0)
        if mode == "grouped":
            bm = max(16, 1 << (hq // hkv - 1).bit_length())
            q4, o4 = q.unsqueeze(1), output.unsqueeze(1)
            factory = _decode_grouped_kernel
            extra = {'scalar_memory': True} if scalar_memory else {}
            if c['implementation'] == 'optimized' and not scalar_memory:
                from ._decode_pipeline import _prefill_kernel as factory
                extra = {'grouped': True, 'segmented_reduce': True}
            kernel = factory(b, 1, hq, hkv, d, capacity, k.shape[0], k.shape[1],
                dtype, tuple(q4.stride()), tuple(k.stride()), tuple(v.stride()), tuple(o4.stride()),
                ts, True, block_table is not None, False, _scale(d, scale),
                bm, c["block_n"], c["num_stages"], c["num_warps"]*32, splits=splits, **extra)
            kernel(q4, k, v, kv_lens, table, o4, partial, maxima, sums)
        else:
            factory = _decode_vector_kernel
            if c['implementation'] == 'optimized' and not scalar_memory:
                from ._decode_warp import _decode_vector_kernel as factory
            extra = {'scalar_memory': True} if scalar_memory else {}
            kernel = factory(b, hq, hkv, d, capacity, k.shape[0], k.shape[1],
                dtype, tuple(q.stride()), tuple(k.stride()), tuple(v.stride()), tuple(output.stride()),
                ts, block_table is not None, _scale(d, scale), c["block_n"], c["num_warps"]*32, splits, **extra)
            kernel(q, k, v, kv_lens, table, output, partial, maxima, sums)
        if splits > 1:
            _merge_splits(b, hq, d, splits, dtype, tuple(output.stride()), scalar_memory)(partial, maxima, sums, output)
    return output
