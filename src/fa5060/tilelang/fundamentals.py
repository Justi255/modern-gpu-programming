"""Standalone TileLang primitives and three warp-MMA GEMM schedules.

PyTorch supplies host metadata, CUDA device context, and output allocation;
every arithmetic/data-movement operation runs in the TileLang kernel below.
Kernels target RTX 5060's SM120 and use the TVM FFI execution backend.
"""

from collections.abc import Mapping
from functools import lru_cache

import torch

__all__ = ["vector_add", "row_softmax", "transpose", "matmul"]

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MATMUL_DTYPES = (torch.float16, torch.bfloat16)
_TARGET = {"kind": "cuda", "arch": "sm_120"}


def _check_tensor(x, name, ndim, dtypes=_FLOAT_DTYPES):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {tuple(x.shape)}")
    if x.dtype not in dtypes:
        raise TypeError(f"{name} dtype must be one of {dtypes}, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError(f"{name} must be on a CUDA device")
    if x.layout != torch.strided:
        raise ValueError(f"{name} must have strided layout")


def _check_pair(x, y):
    if x.device != y.device:
        raise ValueError("inputs must be on the same CUDA device")
    if x.dtype != y.dtype:
        raise TypeError("inputs must have the same dtype")


def _check_columns(x, name):
    if x.stride(-1) != 1:
        raise ValueError(f"{name} last dimension must be contiguous (stride 1)")


def _dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def _compile(kernel):
    # Importing/validating this module on CPU does not require TileLang.
    import tilelang

    # SM120 supports warp MMA, but not Hopper's setmaxnreg reconfiguration.
    # Keep the requested CTA warp count and software-pipeline stages rather
    # than letting long K loops become producer/consumer warp groups.
    return tilelang.compile(kernel, target=_TARGET, execution_backend="tvm_ffi",
                            pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})


@lru_cache(maxsize=128)
def _vector_add_kernel(n, dtype, block=1024):
    """One CTA per vector tile; explicit tail guards and FP32 arithmetic."""
    import tilelang.language as T

    @T.prim_func
    def kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype), Out: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, block), threads=128) as tile:
            for i in T.Parallel(block):
                offset = tile * block + i
                if offset < n:
                    Out[offset] = X[offset].astype("float32") + Y[offset].astype("float32")

    return _compile(kernel)


def vector_add(x, y):
    """Add equal contiguous 1-D CUDA FP16/BF16/FP32 tensors in FP32.

    The fresh result uses the input dtype. Empty inputs are supported.
    Inputs must share shape, dtype, and CUDA device. No autograd is supplied.
    """
    _check_tensor(x, "x", 1)
    _check_tensor(y, "y", 1)
    _check_pair(x, y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous")
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if x.numel():
        with torch.cuda.device(x.device):
            _vector_add_kernel(x.numel(), _dtype_name(x.dtype))(x, y, out)
    return out


@lru_cache(maxsize=128)
def _row_softmax_kernel(rows, cols, stride_x, dtype, block, threads):
    """One CTA per row; bounded online normalizer for very wide rows."""
    import tilelang.language as T

    @T.prim_func
    def kernel(X: T.StridedTensor((rows, cols), (stride_x, 1), dtype), Out: T.Tensor((rows, cols), dtype)):
        with T.Kernel(rows, threads=threads) as row:
            values = T.alloc_fragment((block,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            denominator = T.alloc_fragment((1,), "float32")
            if cols <= block:
                for j in T.Parallel(block):
                    values[j] = T.if_then_else(j < cols, X[row, j].astype("float32"), -T.infinity("float32"))
                T.reduce_max(values, maximum, dim=0)
                for j in T.Parallel(block):
                    values[j] = T.exp(values[j] - maximum[0])
                T.reduce_sum(values, denominator, dim=0)
                for j in T.Parallel(block):
                    if j < cols:
                        Out[row, j] = values[j] / denominator[0]
            else:
                chunk_maximum = T.alloc_fragment((1,), "float32")
                chunk_sum = T.alloc_fragment((1,), "float32")
                previous_maximum = T.alloc_fragment((1,), "float32")
                T.fill(maximum, -T.infinity("float32"))
                T.clear(denominator)
                for tile in T.serial(T.ceildiv(cols, block)):
                    for j in T.Parallel(block):
                        col = tile * block + j
                        values[j] = T.if_then_else(col < cols, X[row, col].astype("float32"), -T.infinity("float32"))
                    T.reduce_max(values, chunk_maximum, dim=0)
                    previous_maximum[0] = maximum[0]
                    maximum[0] = T.max(maximum[0], chunk_maximum[0])
                    for j in T.Parallel(block):
                        values[j] = T.exp(values[j] - maximum[0])
                    T.reduce_sum(values, chunk_sum, dim=0)
                    denominator[0] = denominator[0] * T.exp(previous_maximum[0] - maximum[0]) + chunk_sum[0]
                # The complete row normalizer is now known: reload and write.
                for tile in T.serial(T.ceildiv(cols, block)):
                    for j in T.Parallel(block):
                        col = tile * block + j
                        if col < cols:
                            Out[row, col] = T.exp(X[row, col].astype("float32") - maximum[0]) / denominator[0]

    return _compile(kernel)


def row_softmax(x):
    """Stable column softmax for 2-D CUDA FP16/BF16/FP32 inputs.

    Reductions and exponentials use FP32. The last stride must be 1; sliced
    and broadcast outer rows are supported. Empty axes return an empty
    output. Widths above 32768 use two passes and an online normalizer.
    Stability assumes finite logits; NaN/Inf follow max/exp arithmetic.
    """
    _check_tensor(x, "x", 2)
    _check_columns(x, "x")
    rows, cols = x.shape
    out = torch.empty((rows, cols), device=x.device, dtype=x.dtype)
    if rows and cols:
        block = min(1 << (cols - 1).bit_length(), 32768)
        threads = 128 if block <= 2048 else 256 if block <= 8192 else 512
        with torch.cuda.device(x.device):
            _row_softmax_kernel(rows, cols, x.stride(0), _dtype_name(x.dtype), block, threads)(x, out)
    return out


@lru_cache(maxsize=128)
def _transpose_kernel(rows, cols, stride_x, dtype, tile=32):
    """Stage a square tile in shared memory and exchange read/write axes."""
    import tilelang.language as T

    @T.prim_func
    def kernel(X: T.StridedTensor((rows, cols), (stride_x, 1), dtype), Out: T.Tensor((cols, rows), dtype)):
        with T.Kernel(T.ceildiv(rows, tile), T.ceildiv(cols, tile), threads=128) as (bx, by):
            shared = T.alloc_shared((tile, tile), dtype)
            T.copy(X[bx * tile, by * tile], shared)
            for i, j in T.Parallel(tile, tile):
                row, col = bx * tile + i, by * tile + j
                if row < rows and col < cols:
                    Out[col, row] = shared[i, j]

    return _compile(kernel)


def transpose(x):
    """Materialize a contiguous 2-D CUDA FP16/BF16/FP32 transpose.

    Supports rectangular tails, empty axes, and arbitrary outer row strides.
    The input's last dimension must have stride 1.
    """
    _check_tensor(x, "x", 2)
    _check_columns(x, "x")
    rows, cols = x.shape
    out = torch.empty((cols, rows), device=x.device, dtype=x.dtype)
    if rows and cols:
        with torch.cuda.device(x.device):
            _transpose_kernel(rows, cols, x.stride(0), _dtype_name(x.dtype))(x, out)
    return out


@lru_cache(maxsize=128)
def _matmul_tiled_kernel(m, n, k, stride_a, stride_b, dtype, tile_m, tile_n, tile_k, num_warps, num_stages):
    """Baseline: one output tile per CTA and an ordinary serial K loop."""
    import tilelang.language as T

    # A minimum 32x32 physical tile allows even 16x16 logical tiles to use
    # four/eight warp MMA groups. Only the requested logical tile is stored.
    compute_m, compute_n = max(tile_m, 32), max(tile_n, 32)

    @T.prim_func
    def kernel(
        A: T.StridedTensor((m, k), (stride_a, 1), dtype),
        B: T.StridedTensor((k, n), (stride_b, 1), dtype),
        C: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(T.ceildiv(m, tile_m) * T.ceildiv(n, tile_n), threads=num_warps * 32) as tile:
            a_shared = T.alloc_shared((compute_m, tile_k), dtype)
            b_shared = T.alloc_shared((tile_k, compute_n), dtype)
            accumulator = T.alloc_fragment((compute_m, compute_n), "float32")
            by, bx = tile // T.ceildiv(n, tile_n), tile % T.ceildiv(n, tile_n)
            T.clear(accumulator)
            for ko in T.serial(T.ceildiv(k, tile_k)):
                T.copy(A[by * tile_m, ko * tile_k], a_shared)
                T.copy(B[ko * tile_k, bx * tile_n], b_shared)
                # SM120 + fragment accumulator selects warp mma.sync.
                T.gemm(a_shared, b_shared, accumulator)
            for i, j in T.Parallel(compute_m, compute_n):
                if i < tile_m and j < tile_n and by * tile_m + i < m and bx * tile_n + j < n:
                    C[by * tile_m + i, bx * tile_n + j] = accumulator[i, j]

    return _compile(kernel)


@lru_cache(maxsize=128)
def _matmul_pipelined_kernel(m, n, k, stride_a, stride_b, dtype, tile_m, tile_n, tile_k, num_warps, num_stages):
    """Software pipeline: stage future K tiles while warp MMA accumulates."""
    import tilelang.language as T

    compute_m, compute_n = max(tile_m, 32), max(tile_n, 32)

    @T.prim_func
    def kernel(
        A: T.StridedTensor((m, k), (stride_a, 1), dtype),
        B: T.StridedTensor((k, n), (stride_b, 1), dtype),
        C: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(T.ceildiv(m, tile_m) * T.ceildiv(n, tile_n), threads=num_warps * 32) as tile:
            a_shared = T.alloc_shared((compute_m, tile_k), dtype)
            b_shared = T.alloc_shared((tile_k, compute_n), dtype)
            accumulator = T.alloc_fragment((compute_m, compute_n), "float32")
            by, bx = tile // T.ceildiv(n, tile_n), tile % T.ceildiv(n, tile_n)
            T.clear(accumulator)
            for ko in T.Pipelined(T.ceildiv(k, tile_k), num_stages=num_stages):
                T.copy(A[by * tile_m, ko * tile_k], a_shared)
                T.copy(B[ko * tile_k, bx * tile_n], b_shared)
                T.gemm(a_shared, b_shared, accumulator)
            for i, j in T.Parallel(compute_m, compute_n):
                if i < tile_m and j < tile_n and by * tile_m + i < m and bx * tile_n + j < n:
                    C[by * tile_m + i, bx * tile_n + j] = accumulator[i, j]

    return _compile(kernel)


@lru_cache(maxsize=128)
def _matmul_persistent_kernel(m, n, k, stride_a, stride_b, dtype, tile_m, tile_n, tile_k, num_warps, num_stages, programs):
    """Bound the CTA grid; each CTA advances through independent C tiles."""
    import tilelang.language as T

    compute_m, compute_n = max(tile_m, 32), max(tile_n, 32)

    @T.prim_func
    def kernel(
        A: T.StridedTensor((m, k), (stride_a, 1), dtype),
        B: T.StridedTensor((k, n), (stride_b, 1), dtype),
        C: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(programs, threads=num_warps * 32) as program:
            a_shared = T.alloc_shared((compute_m, tile_k), dtype)
            b_shared = T.alloc_shared((tile_k, compute_n), dtype)
            accumulator = T.alloc_fragment((compute_m, compute_n), "float32")
            for tile in T.serial(program, T.ceildiv(m, tile_m) * T.ceildiv(n, tile_n), programs):
                by, bx = tile // T.ceildiv(n, tile_n), tile % T.ceildiv(n, tile_n)
                T.clear(accumulator)
                for ko in T.Pipelined(T.ceildiv(k, tile_k), num_stages=num_stages):
                    T.copy(A[by * tile_m, ko * tile_k], a_shared)
                    T.copy(B[ko * tile_k, bx * tile_n], b_shared)
                    T.gemm(a_shared, b_shared, accumulator)
                for i, j in T.Parallel(compute_m, compute_n):
                    if i < tile_m and j < tile_n and by * tile_m + i < m and bx * tile_n + j < n:
                        C[by * tile_m + i, bx * tile_n + j] = accumulator[i, j]

    return _compile(kernel)


def _matmul_config(config, variant):
    result = {"tile_m": 32, "tile_n": 64, "tile_k": 32,
              "num_warps": 4, "num_stages": 1 if variant == "tiled" else 3}
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or None")
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown, key=str)}")
        result.update(config)
    allowed = {"tile_m": (16, 32, 64, 128), "tile_n": (16, 32, 64, 128),
               "tile_k": (16, 32, 64, 128), "num_warps": (4, 8), "num_stages": (1, 2, 3, 4)}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, int) or value not in allowed[key]:
            raise ValueError(f"config {key} must be one of {allowed[key]}, got {value!r}")
    bytes_per_stage = 2 * result["tile_k"] * (result["tile_m"] + result["tile_n"])
    if bytes_per_stage * result["num_stages"] > 80 * 1024:
        raise ValueError("config tiles and num_stages exceed the 80-KiB teaching operand budget")
    return result


def matmul(a, b, *, variant="tiled", config=None):
    """Compute CUDA FP16/BF16 ``a[M,K] @ b[K,N]`` with FP32 warp MMA.

    Outer strides may vary; last strides must be 1. Arbitrary M/N/K tails
    and zero axes are valid. Output is freshly allocated in the input dtype.
    ``variant`` chooses tiled, pipelined, or persistent. Persistent launches
    at most one CTA per SM and advances through C tiles; residency/speedups
    are not guaranteed. No autograd backward rule is supplied.

    ``config`` matches the Triton teaching API: tile_m/tile_n/tile_k each
    accept 16,32,64,128; num_warps accepts 4 or 8; num_stages accepts 1..4.
    Defaults are 32/64/32, 4 warps, and 1 stage for tiled or 3 otherwise.
    Tiled retains a serial K loop. Physical MMA tiles are padded to at least
    32x32 when a logical tile is smaller. The 80-KiB operand preflight bound
    is conservative; compiler/device resource failures are never replaced
    with library GEMM.
    """
    if variant not in ("tiled", "pipelined", "persistent"):
        raise ValueError("variant must be 'tiled', 'pipelined', or 'persistent'")
    settings = _matmul_config(config, variant)
    _check_tensor(a, "a", 2, _MATMUL_DTYPES)
    _check_tensor(b, "b", 2, _MATMUL_DTYPES)
    _check_pair(a, b)
    _check_columns(a, "a")
    _check_columns(b, "b")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a and b inner dimensions must match")
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if not m or not n:
        return out
    arguments = (m, n, k, a.stride(0), b.stride(0), _dtype_name(a.dtype),
                 settings["tile_m"], settings["tile_n"], settings["tile_k"],
                 settings["num_warps"], settings["num_stages"])
    with torch.cuda.device(a.device):
        if variant == "tiled":
            kernel = _matmul_tiled_kernel(*arguments)
        elif variant == "pipelined":
            kernel = _matmul_pipelined_kernel(*arguments)
        else:
            num_tiles = ((m + settings["tile_m"] - 1) // settings["tile_m"]) * ((n + settings["tile_n"] - 1) // settings["tile_n"])
            programs = min(num_tiles, torch.cuda.get_device_properties(a.device).multi_processor_count)
            kernel = _matmul_persistent_kernel(*arguments, programs)
        kernel(a, b, out)
    return out
