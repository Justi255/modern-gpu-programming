"""Optional dense SM120 prefill using host-encoded TMA tensor descriptors.

合法布局是连续的 BSHD FP16/BF16，D 为 64 或 128。Q/K/V 的实际数据
指针必须按 16 字节对齐；连续 BSHD 的行距 H*D*2 字节和 head 偏移
h*D*2 字节也因而满足 TMA 的 16 字节条件。连续但有一个 half 元素偏移
的 view 不满足此条件。描述符把 Q 看作 [B*Sq,Hq*D]，KV 看作
[B*capacity,Hkv*D]，每次搬运 [BM,D] 或 [BN,D]，并显式掩掉跨 batch
边界、live length 之外的 tile 尾部。

No paged/strided-input fallback is performed. Unsupported layouts raise
ValueError so the benchmark can record an unsupported candidate. Output uses
ordinary masked stores and can have noncontiguous outer strides. This module
uses warp-level MMA/register accumulators, with no tcgen05, TMEM, or SM100
warp specialization. The public entry point is the internal root-module hook
``prefill_tma(q,k,v,out,causal,scale,kv_lens,block_table,c)``.
"""

from collections.abc import Mapping
import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

__all__ = ["prefill_tma"]


@triton.jit
def _prefill_tma_kernel(
    QDesc, KDesc, VDesc, Lengths, Out,
    O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr,
    SQ: tl.constexpr, HQ: tl.constexpr, GROUP: tl.constexpr,
    D: tl.constexpr, CAPACITY: tl.constexpr, HAS_LENGTHS: tl.constexpr,
    CAUSAL: tl.constexpr, SCALE_LOG2: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr,
):
    block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // HQ, batch_head % HQ
    kv_head = head // GROUP
    length = tl.load(Lengths + batch) if HAS_LENGTHS else CAPACITY
    rows = block * BM + tl.arange(0, BM)
    local_keys = tl.arange(0, BN)
    dims = tl.arange(0, D)

    # A descriptor's first dimension combines all batches.  Its zero padding
    # protects the allocation boundary; the explicit row mask protects each
    # logical batch boundary when Sq is not divisible by BM.
    query = QDesc.load([batch * SQ + block * BM, head * D])
    query = tl.where(rows[:, None] < SQ, query, 0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, D), tl.float32)

    stop = length
    if CAUSAL:
        stop = tl.maximum(0, tl.minimum(length, (block + 1) * BM + length - SQ))
    for start in tl.range(0, stop, BN, num_stages=STAGES):
        positions = start + local_keys
        live_keys = positions < length
        key = KDesc.load([batch * CAPACITY + start, kv_head * D])
        # TMA loads rectangular tiles, while a live cache can stop inside one.
        # Mask before dot: invalid storage may contain NaNs, including values
        # read from the next batch by a tile straddling the capacity boundary.
        key = tl.where(live_keys[:, None], key, 0)
        scores = tl.dot(query, tl.trans(key)) * SCALE_LOG2
        valid = live_keys[None, :] & (rows[:, None] < SQ)
        if CAUSAL:
            valid = valid & (positions[None, :] <= rows[:, None] + length - SQ)
        scores = tl.where(valid, scores, -float("inf"))

        updated = tl.maximum(maximum, tl.max(scores, 1))
        # All-masked rows keep m=-inf and l=0; use a finite subtraction base
        # so exp2(-inf - -inf) never produces a NaN.
        safe = tl.where(updated == -float("inf"), 0.0, updated)
        correction = tl.exp2(maximum - safe)
        probabilities = tl.exp2(scores - safe[:, None])
        value = VDesc.load([batch * CAPACITY + start, kv_head * D])
        value = tl.where(live_keys[:, None], value, 0)
        accumulator *= correction[:, None]
        accumulator = tl.dot(probabilities.to(query.dtype), value, accumulator)
        denominator = denominator * correction + tl.sum(probabilities, 1)
        maximum = updated

    result = accumulator / tl.where(denominator > 0.0, denominator, 1.0)[:, None]
    output_offsets = batch * O0 + rows[:, None] * O1 + head * O2 + dims[None, :]
    tl.store(Out + output_offsets, result, mask=rows[:, None] < SQ)


@triton.jit
def _zero_output_kernel(
    Out, ELEMENTS, SQ: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr,
    O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr, BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    dim = index % D
    head = (index // D) % HQ
    row = (index // (D * HQ)) % SQ
    batch = index // (D * HQ * SQ)
    offsets = batch * O0 + row * O1 + head * O2 + dim
    tl.store(Out + offsets, 0.0, mask=index < ELEMENTS)


def _tma_configuration(c, d):
    if not isinstance(c, Mapping):
        raise ValueError("TMA config must be a mapping")
    config = {"block_m": 32, "block_n": 64, "num_warps": 4,
              "num_stages": 2, "loader": "tma", "mode": "auto"}
    if set(c) - set(config):
        raise ValueError("unknown TMA config keys")
    config.update(c)
    choices = {"block_m": (32, 64, 128), "block_n": (32, 64, 128),
               "num_warps": (4, 8), "num_stages": (1, 2, 3, 4)}
    for name, allowed in choices.items():
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
            raise ValueError(f"unsupported TMA config {name}={value!r}; allowed {allowed}")
    if config["loader"] != "tma":
        raise ValueError("TMA config loader must be 'tma'")
    # Q + staged K/V operands, plus conservative barrier/layout scratch.
    shared_budget = 2 * config["block_m"] * d + 4 * config["block_n"] * d * config["num_stages"] + 1024
    if shared_budget > 96 * 1024:
        raise ValueError("TMA config exceeds the conservative 96-KiB shared memory budget")
    return config


def prefill_tma(q, k, v, out, causal, scale, kv_lens, block_table, c):
    """Fill ``out`` using dense contiguous BSHD TMA loads and return ``out``.

    ``q`` is [B,Sq,Hq,D], ``k/v`` are [B,capacity,Hkv,D]. Equal FP16/BF16
    dtype/device, D=64/128, Hq divisible by Hkv, contiguous inputs, 16-byte
    aligned data pointers, and row strides divisible by 16 bytes are required.
    Head offsets must also be 16-byte aligned (automatic for supported D).
    Block choices are BM/BN=32/64/128, stages=1..4, warps=4/8, subject to a
    conservative <=96-KiB shared-memory check including 1 KiB scratch reserve.

    ``kv_lens`` is optional contiguous CUDA int32 [B], with caller-guaranteed
    values 0<=length<=capacity. Lengths stay on the GPU. Causal query positions
    are right aligned by each live length. Entirely masked rows return zero;
    fp32 online maxima/sums/accumulators are used and probabilities are cast
    to query dtype before the second dot. Empty capacity uses a Triton zero
    store because CUDA tensor maps require positive descriptor dimensions.

    Output must match Q shape/device/dtype, have last stride 1, and not alias
    any input; its outer strides may be noncontiguous. No autograd backward,
    paged storage, implicit copies, pointer-loader fallback, or host reads of
    live metadata are provided. Unsupported cases raise ValueError.
    """
    if block_table is not None:
        raise ValueError("TMA prefill supports dense KV only; paged storage is unsupported")
    for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out)):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise ValueError(f"TMA {name} must be a four-dimensional Tensor")
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("TMA prefill requires FP16 or BF16 tensors")
        if tensor.dtype != q.dtype or tensor.device != q.device:
            raise ValueError("TMA q/k/v/out must share dtype and device")
    batch, sq, hq, d = q.shape
    if d not in (64, 128):
        raise ValueError("TMA prefill requires D=64 or 128")
    hkv = k.shape[2]
    if hq <= 0 or hkv <= 0 or hq % hkv or k.shape[0] != batch or k.shape[3] != d or k.shape != v.shape:
        raise ValueError("TMA dense shapes/heads do not match; Hq must be divisible by Hkv")
    if out.shape != q.shape or out.stride(-1) != 1:
        raise ValueError("TMA output must match Q shape with contiguous last dimension")
    from .attention import _check_output_layout
    _check_output_layout(out)
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not tensor.is_contiguous():
            raise ValueError(f"TMA {name} must be contiguous BSHD")
        if tensor.data_ptr() % 16 or (tensor.shape[2] * d * tensor.element_size()) % 16:
            raise ValueError(f"TMA {name} base and row stride must be 16-byte aligned")
    config = _tma_configuration(c, d)
    scale_value = d ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value):
        raise ValueError("TMA scale must be finite")
    if q.device.type != "cuda":
        raise ValueError("TMA prefill requires CUDA tensors")
    if any(x.requires_grad for x in (q, k, v)) and torch.is_grad_enabled():
        raise ValueError("TMA prefill is inference only; use torch.no_grad")
    if out.numel() and out.untyped_storage().data_ptr() in {
        tensor.untyped_storage().data_ptr() for tensor in (q, k, v) if tensor.numel()
    }:
        raise ValueError("TMA output must not alias inputs")
    if kv_lens is not None and (
        not isinstance(kv_lens, torch.Tensor) or kv_lens.shape != (batch,) or
        kv_lens.dtype != torch.int32 or kv_lens.device != q.device or not kv_lens.is_contiguous()
    ):
        raise ValueError("TMA kv_lens must be contiguous CUDA int32 [B]")
    if not q.numel():
        return out

    bm, bn = config["block_m"], config["block_n"]
    capacity = k.shape[1]
    with torch.cuda.device(q.device):
        if capacity == 0:
            _zero_output_kernel[(triton.cdiv(q.numel(), 256),)](
                out, q.numel(), sq, hq, d, *out.stride()[:3], BLOCK=256
            )
            return out
        q_descriptor = TensorDescriptor(q, [batch * sq, hq * d], [hq * d, 1], [bm, d])
        k_descriptor = TensorDescriptor(k, [batch * capacity, hkv * d], [hkv * d, 1], [bn, d])
        v_descriptor = TensorDescriptor(v, [batch * capacity, hkv * d], [hkv * d, 1], [bn, d])
        _prefill_tma_kernel[(triton.cdiv(sq, bm), batch * hq)](
            q_descriptor, k_descriptor, v_descriptor, q if kv_lens is None else kv_lens, out,
            *out.stride()[:3], sq, hq, hq // hkv, d, capacity, kv_lens is not None,
            bool(causal), scale_value * math.log2(math.e), bm, bn, config["num_stages"],
            num_warps=config["num_warps"], num_stages=config["num_stages"],
        )
    return out
