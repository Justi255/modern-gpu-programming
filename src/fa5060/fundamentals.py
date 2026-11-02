"""Readable Triton primitives and three distinct GEMM schedules.

这些函数是教材中的真实 GPU 实现；PyTorch 只负责校验和分配输出。
Every operation allocates a new tensor, preserves its input dtype, and has no
autograd rule.  CUDA is required: there is deliberately no compute fallback.
"""

from collections.abc import Mapping

import torch
import triton
import triton.language as tl

__all__ = ["vector_add", "row_softmax", "transpose", "matmul"]

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MATMUL_DTYPES = (torch.float16, torch.bfloat16)


def _check_tensor(x, name, ndim, dtypes=_FLOAT_DTYPES):
    """Check cheap host metadata; never copy an unsupported input silently."""
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


@triton.jit
def _vector_add_kernel(X, Y, Out, N, BLOCK: tl.constexpr):
    # 一个 program 对应一个 block；mask 是尾部合法性，而不是整除假设。
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    x = tl.load(X + offsets, mask=valid, other=0).to(tl.float32)
    y = tl.load(Y + offsets, mask=valid, other=0).to(tl.float32)
    tl.store(Out + offsets, x + y, mask=valid)


def vector_add(x, y):
    """Add same-shape contiguous 1-D CUDA tensors in FP16, BF16, or FP32.

    Arithmetic uses fp32 and the result is rounded to the input dtype.  Empty
    vectors are supported.  Inputs must share both dtype and CUDA device.
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
            _vector_add_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, out, x.numel(), BLOCK=1024)
    return out


@triton.jit
def _row_softmax_kernel(X, Out, COLS: tl.constexpr, STRIDE_X: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    if COLS <= BLOCK:
        # fp32 的 max/exp/sum，先减最大值，避免大正 logits 使 exp 溢出。
        values = tl.load(X + row * STRIDE_X + offsets, mask=offsets < COLS, other=-float("inf"))
        values = values.to(tl.float32)
        numerator = tl.exp(values - tl.max(values, axis=0))
        probabilities = numerator / tl.sum(numerator, axis=0)
        tl.store(Out + row * COLS + offsets, probabilities, mask=offsets < COLS)
    else:
        # Very wide rows: online normalizer bounds register usage.  A second
        # pass writes the result after the complete row's normalizer is known.
        maximum = tl.full((), -float("inf"), tl.float32)
        denominator = tl.full((), 0.0, tl.float32)
        for start in range(tl.cdiv(COLS, BLOCK)):
            cols = start * BLOCK + offsets
            values = tl.load(X + row * STRIDE_X + cols, mask=cols < COLS, other=-float("inf"))
            values = values.to(tl.float32)
            next_maximum = tl.maximum(maximum, tl.max(values, axis=0))
            denominator = denominator * tl.exp(maximum - next_maximum) + tl.sum(
                tl.exp(values - next_maximum), axis=0
            )
            maximum = next_maximum
        for start in range(tl.cdiv(COLS, BLOCK)):
            cols = start * BLOCK + offsets
            values = tl.load(X + row * STRIDE_X + cols, mask=cols < COLS, other=-float("inf"))
            probabilities = tl.exp(values.to(tl.float32) - maximum) / denominator
            tl.store(Out + row * COLS + cols, probabilities, mask=cols < COLS)


def row_softmax(x):
    """Numerically stable softmax over columns of a 2-D CUDA tensor.

    FP16, BF16, and FP32 are supported.  Only the last dimension must have
    stride 1; sliced or broadcast outer rows are valid.  Empty dimensions
    return an empty result.  For widths above 32768, an online normalizer and
    two passes bound per-program register use.  IEEE NaN/Inf inputs propagate
    according to max/exp arithmetic; stability assumes finite input logits.
    """
    _check_tensor(x, "x", 2)
    _check_columns(x, "x")
    rows, cols = x.shape
    out = torch.empty((rows, cols), device=x.device, dtype=x.dtype)
    if rows and cols:
        block = min(triton.next_power_of_2(cols), 32768)
        num_warps = 4 if block <= 2048 else 8 if block <= 8192 else 16
        with torch.cuda.device(x.device):
            _row_softmax_kernel[(rows,)](x, out, cols, x.stride(0), block, num_warps=num_warps)
    return out


@triton.jit
def _transpose_kernel(X, Out, ROWS, COLS, STRIDE_X: tl.constexpr, TILE: tl.constexpr):
    # 二维 tile 对调读写地址。Triton 会插入需要的 layout conversion；
    # Python 层没有调用 contiguous()/transpose() 来代替这次搬运。
    rows = tl.program_id(0) * TILE + tl.arange(0, TILE)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    valid = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
    values = tl.load(X + rows[:, None] * STRIDE_X + cols[None, :], mask=valid, other=0)
    tl.store(Out + cols[None, :] * ROWS + rows[:, None], values, mask=valid)


def transpose(x):
    """Materialize a contiguous transpose of a 2-D FP16/BF16/FP32 CUDA tensor.

    Rectangular shapes, partial 32-by-32 tiles, empty axes, and noncontiguous
    outer row strides are supported.  The input's last dimension needs stride
    1.  This is a data-movement kernel, rather than a zero-copy tensor view.
    """
    _check_tensor(x, "x", 2)
    _check_columns(x, "x")
    rows, cols = x.shape
    out = torch.empty((cols, rows), device=x.device, dtype=x.dtype)
    if rows and cols:
        with torch.cuda.device(x.device):
            _transpose_kernel[(triton.cdiv(rows, 32), triton.cdiv(cols, 32))](
                x, out, rows, cols, x.stride(0), TILE=32, num_warps=4
            )
    return out


@triton.jit
def _matmul_tiled_kernel(
    A, B, C, M, N, K, STRIDE_A: tl.constexpr, STRIDE_B: tl.constexpr,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
):
    """Baseline: one output tile per program; ordinary serial K-tile loop."""
    tile = tl.program_id(0)
    tiles_n = tl.cdiv(N, TILE_N)
    rows = (tile // tiles_n) * TILE_M + tl.arange(0, TILE_M)
    cols = (tile % tiles_n) * TILE_N + tl.arange(0, TILE_N)
    ks = tl.arange(0, TILE_K)
    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k_tile in range(tl.cdiv(K, TILE_K)):
        k = k_tile * TILE_K + ks
        a = tl.load(A + rows[:, None] * STRIDE_A + k[None, :],
                    mask=(rows[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(B + k[:, None] * STRIDE_B + cols[None, :],
                    mask=(k[:, None] < K) & (cols[None, :] < N), other=0)
        accumulator = tl.dot(a, b, accumulator)
    tl.store(C + rows[:, None] * N + cols[None, :], accumulator,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _matmul_pipelined_kernel(
    A, B, C, M, N, K, STRIDE_A: tl.constexpr, STRIDE_B: tl.constexpr,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr, STAGES: tl.constexpr,
):
    """Software pipeline: explicitly stage future K tiles while doing dots."""
    tile = tl.program_id(0)
    tiles_n = tl.cdiv(N, TILE_N)
    rows = (tile // tiles_n) * TILE_M + tl.arange(0, TILE_M)
    cols = (tile % tiles_n) * TILE_N + tl.arange(0, TILE_N)
    ks = tl.arange(0, TILE_K)
    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    # num_stages belongs to this loop.  It asks the compiler to overlap
    # global-to-shared loads with Tensor Core work; no async claim is assumed
    # without inspecting generated code and measuring on the actual device.
    for k_tile in tl.range(0, tl.cdiv(K, TILE_K), num_stages=STAGES):
        k = k_tile * TILE_K + ks
        a = tl.load(A + rows[:, None] * STRIDE_A + k[None, :],
                    mask=(rows[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(B + k[:, None] * STRIDE_B + cols[None, :],
                    mask=(k[:, None] < K) & (cols[None, :] < N), other=0)
        accumulator = tl.dot(a, b, accumulator)
    tl.store(C + rows[:, None] * N + cols[None, :], accumulator,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _matmul_persistent_kernel(
    A, B, C, M, N, K, STRIDE_A: tl.constexpr, STRIDE_B: tl.constexpr,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    STAGES: tl.constexpr, PROGRAMS: tl.constexpr,
):
    """A bounded grid reuses each program for several independent output tiles."""
    tiles_m = tl.cdiv(M, TILE_M)
    tiles_n = tl.cdiv(N, TILE_N)
    local_rows = tl.arange(0, TILE_M)
    local_cols = tl.arange(0, TILE_N)
    ks = tl.arange(0, TILE_K)
    # 一开始最多一个 program/SM；每次加 PROGRAMS，各输出 tile 恰好写一次。
    # 这是显式的持久调度教学实现，不依赖动态工作队列或全局原子计数器。
    for tile in tl.range(tl.program_id(0), tiles_m * tiles_n, PROGRAMS):
        rows = (tile // tiles_n) * TILE_M + local_rows
        cols = (tile % tiles_n) * TILE_N + local_cols
        accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for k_tile in tl.range(0, tl.cdiv(K, TILE_K), num_stages=STAGES):
            k = k_tile * TILE_K + ks
            a = tl.load(A + rows[:, None] * STRIDE_A + k[None, :],
                        mask=(rows[:, None] < M) & (k[None, :] < K), other=0)
            b = tl.load(B + k[:, None] * STRIDE_B + cols[None, :],
                        mask=(k[:, None] < K) & (cols[None, :] < N), other=0)
            accumulator = tl.dot(a, b, accumulator)
        tl.store(C + rows[:, None] * N + cols[None, :], accumulator,
                 mask=(rows[:, None] < M) & (cols[None, :] < N))


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
               "tile_k": (16, 32, 64, 128), "num_warps": (4, 8),
               "num_stages": (1, 2, 3, 4)}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, int) or value not in allowed[key]:
            raise ValueError(f"config {key} must be one of {allowed[key]}, got {value!r}")
    # Conservative operand-storage budget for SM120's 99-KiB block limit.
    # This is a preflight bound, not a claim about the compiler's exact layout.
    stages = result["num_stages"]
    bytes_per_stage = 2 * result["tile_k"] * (result["tile_m"] + result["tile_n"])
    if bytes_per_stage * stages > 80 * 1024:
        raise ValueError("config tiles and num_stages exceed the 80-KiB teaching operand budget")
    return result


def matmul(a, b, *, variant="tiled", config=None):
    """Compute ``a[M,K] @ b[K,N]`` with a selected teaching Triton kernel.

    Inputs are 2-D CUDA tensors with equal FP16 or BF16 dtype and device;
    outer strides may differ from their logical row widths, but each last
    dimension must have stride 1.  The accumulator is fp32, with one cast to
    the input dtype on store.  Arbitrary M/N/K tails and zero axes are valid.

    ``variant`` selects ``'tiled'`` (one output tile per program),
    ``'pipelined'`` (a staged K loop), or ``'persistent'`` (at most one program
    per SM, each advancing through output tiles with a staged K loop).
    Persistence controls the grid and does not guarantee physical residency
    or a speedup.  These kernels do not provide an autograd backward rule.

    ``config`` is an optional mapping with these integer keys:

    * ``tile_m``, ``tile_n``, ``tile_k``: one of 16, 32, 64, 128; defaults
      32, 64, 32.  These are block sizes, not required input divisibility.
    * ``num_warps``: 4 (default) or 8, passed to Triton's launch compiler.
    * ``num_stages``: 1..4; default 1 for tiled, 3 otherwise.  Sets the
      compiler launch option and the pipelined/persistent K-loop hint.
      A tiled kernel keeps its ordinary K loop even when this is changed.

    Configurations above a conservative 80-KiB staged operand budget are
    rejected.  Compilation resources can still depend on compiler/device;
    a rejected or failed configuration is never replaced by library GEMM.
    """
    # Schedule/configuration checks use host values only, so invalid tuning
    # requests can be diagnosed even on a documentation builder without CUDA.
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
    tile_m, tile_n, tile_k = (settings[key] for key in ("tile_m", "tile_n", "tile_k"))
    stages = settings["num_stages"]
    num_tiles = triton.cdiv(m, tile_m) * triton.cdiv(n, tile_n)
    arguments = (a, b, out, m, n, k, a.stride(0), b.stride(0), tile_m, tile_n, tile_k)
    launch = {"num_warps": settings["num_warps"], "num_stages": stages}
    with torch.cuda.device(a.device):
        if variant == "tiled":
            _matmul_tiled_kernel[(num_tiles,)](*arguments, **launch)
        elif variant == "pipelined":
            _matmul_pipelined_kernel[(num_tiles,)](*arguments, STAGES=stages, **launch)
        else:
            programs = min(num_tiles, torch.cuda.get_device_properties(a.device).multi_processor_count)
            _matmul_persistent_kernel[(programs,)](*arguments, STAGES=stages, PROGRAMS=programs, **launch)
    return out
