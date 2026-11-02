"""Readable, chunked attention reference; deliberately excluded from timing."""

import math
import torch


@torch.no_grad()
def attention_reference(q, k, v, *, causal=True, kv_lens=None,
                        block_table=None, scale=None, chunk_size=128):
    """BSHD prefill or BHD decode, with right-aligned causal positions.

    Query chunks bound the score allocation to Hq * chunk_size * KV length.
    CPU synchronization of lengths/page tables is intentional in this reference.
    Inputs retain their original floating-point values; arithmetic is FP32 (FP64
    for FP64 inputs). This is not an implementation used as a timed baseline.
    """
    decode = q.ndim == 3
    if q.ndim not in (3, 4) or k.ndim != 4 or k.shape != v.shape:
        raise ValueError('expected BSHD/BHD queries and matching four-dimensional KV')
    query = q[:, None] if decode else q
    batch, sq, hq, d = query.shape
    hkv = k.shape[2]
    if hkv <= 0 or hq % hkv or k.shape[-1] != d or chunk_size <= 0:
        raise ValueError('invalid head dimensions or chunk size')
    capacity = k.shape[1] if block_table is None else block_table.shape[1]*k.shape[1]
    if block_table is not None and kv_lens is None:
        raise ValueError('paged KV requires kv_lens')
    lengths = [capacity]*batch if kv_lens is None else kv_lens.cpu().tolist()
    if len(lengths) != batch or any(x < 0 or x > capacity for x in lengths):
        raise ValueError('invalid KV lengths')
    scale = d**-.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError('scale must be finite')
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    result = torch.zeros_like(query)
    head_index = torch.arange(hq, device=q.device)//(hq//hkv)
    for b, length in enumerate(lengths):
        if length == 0:
            continue
        if block_table is None:
            kb, vb = k[b, :length], v[b, :length]
        else:
            pages = block_table[b, :math.ceil(length/k.shape[1])].long()
            if bool((pages < 0).any()) or bool((pages >= k.shape[0]).any()):
                raise ValueError('invalid live page index')
            kb = k.index_select(0, pages).reshape(-1, hkv, d)[:length]
            vb = v.index_select(0, pages).reshape(-1, hkv, d)[:length]
        kh = kb[:, head_index].permute(1, 2, 0).to(compute_dtype)
        vh = vb[:, head_index].permute(1, 0, 2).to(compute_dtype)
        for start in range(0, sq, chunk_size):
            end = min(start+chunk_size, sq)
            qh = query[b, start:end].transpose(0, 1).to(compute_dtype)
            scores = (qh @ kh)*scale
            if causal and not decode:
                positions = torch.arange(start, end, device=q.device)+length-sq
                valid = torch.arange(length, device=q.device)[None, :] <= positions[:, None]
                scores.masked_fill_(~valid[None], -float('inf'))
            # Entirely masked rows have defined zero output, not NaN.
            probabilities = scores.softmax(-1).nan_to_num(0.)
            result[b, start:end] = (probabilities @ vh).transpose(0, 1).to(q.dtype)
    return result[:, 0] if decode else result
