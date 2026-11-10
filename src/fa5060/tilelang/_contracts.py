"""Host-only contracts and scratch for the standalone TileLang backend.

No Triton/TileLang compiler import; input metadata stays on the host while
live cache lengths and page indices are read only by device kernels.
"""
import math
from dataclasses import dataclass
import torch

def _next_power_of_2(n):
    return 1 << max(0, int(n - 1).bit_length())

def _validate(q,k,v,kv_lens,block_table,rank):
    for name,x in [('q',q),('k',k),('v',v)]:
        if not isinstance(x,torch.Tensor) or x.device.type != 'cuda':
            raise ValueError(f'{name} must be a CUDA tensor')
        if x.dtype not in (torch.float16,torch.bfloat16):
            raise ValueError('attention requires FP16 or BF16')
        if x.device != q.device or x.dtype != q.dtype:
            raise ValueError('q/k/v must have the same device and dtype')
        if x.stride(-1) != 1:
            raise ValueError('the last dimension must be contiguous')
        if x.requires_grad and torch.is_grad_enabled():
            raise ValueError('inference only; use torch.no_grad for requires_grad inputs')
    if q.ndim != rank or k.ndim != 4 or v.shape != k.shape:
        raise ValueError('invalid query/cache dimensions')
    b,hq,d=q.shape[0],q.shape[-2],q.shape[-1]
    hkv=k.shape[2]
    if d not in (32,64,128) or hq<=0 or hkv<=0 or hq%hkv or k.shape[3]!=d:
        raise ValueError('head dimensions must match; D=32/64/128 and Hq divisible by Hkv')
    paged=block_table is not None
    if paged:
        if k.shape[1] not in (16,32,64):
            raise ValueError('page size must be 16, 32, or 64')
        if (block_table.ndim != 2 or block_table.shape[0]!=b or
            block_table.dtype!=torch.int32 or block_table.device!=q.device or block_table.stride(1)!=1):
            raise ValueError('block_table must be a CUDA int32 [B,max_pages] tensor')
        capacity=block_table.shape[1]*k.shape[1]
    else:
        if k.shape[0]!=b:
            raise ValueError('dense KV batch size must match Q')
        capacity=k.shape[1]
    if kv_lens is None and (paged or rank==3):
        raise ValueError('decode/paged KV requires kv_lens')
    if kv_lens is not None and (kv_lens.shape!=(b,) or kv_lens.dtype!=torch.int32 or
                                 kv_lens.device!=q.device or not kv_lens.is_contiguous()):
        raise ValueError('kv_lens must be a contiguous CUDA int32 [B] tensor')
    # Metadata values are a caller contract: 0 <= length <= capacity; all live
    # page indices are in range. Reading them on the host would break graphs.
    return b,hq,hkv,d,capacity

def _check_output_layout(out):
    """Accept separated strided axes; reject overlapping writable views.

    This metadata-only sufficient check supports permutations and stepped
    outer slices without a device synchronization. Exotic interleaved
    as_strided layouts that cannot be proved disjoint are rejected too.
    """
    if not out.numel():
        return
    span = 1
    for stride, size in sorted((s, n) for s, n in zip(out.stride(), out.shape) if n > 1):
        if stride < span:
            raise ValueError('out layout must have nonoverlapping, separated strides')
        span += (size - 1) * stride

def _output(q,k,v,out):
    if out is None:
        return torch.empty_like(q, memory_format=torch.contiguous_format)
    if out.shape!=q.shape or out.device!=q.device or out.dtype!=q.dtype or out.stride(-1)!=1:
        raise ValueError('out must match Q shape/device/dtype and contiguous last dimension')
    _check_output_layout(out)
    if out.untyped_storage().data_ptr() in {x.untyped_storage().data_ptr() for x in (q,k,v)}:
        raise ValueError('out must not alias the inputs')
    return out

def _configuration(config,decode=False):
    result={'block_m':32,'block_n':64,'num_warps':4,'num_stages':2,
            'loader':'pointer','mode':'auto'}
    if config:
        if set(config)-set(result):
            raise ValueError(f'unknown attention config keys: {set(config)-set(result)}')
        result.update(config)
    if (result['block_m'] not in (16,32,64,128) or result['block_n'] not in (32,64,128,256) or
        result['num_warps'] not in (4,8) or result['num_stages'] not in (1,2,3,4)):
        raise ValueError('invalid tile/warp/pipeline configuration')
    if result['mode'] not in ('auto','vector','grouped'):
        raise ValueError('invalid decode mode')
    if result['loader'] not in ('pointer','tma'):
        raise ValueError('invalid loader')
    if decode and result['loader']!='pointer':
        raise ValueError('decode supports the pointer loader')
    return result

def _scale(d,scale):
    value=d**-.5 if scale is None else float(scale)
    if not math.isfinite(value):
        raise ValueError('scale must be finite')
    return value*math.log2(math.e)

@dataclass(frozen=True)
class DecodeWorkspace:
    """Per-call scratch. Concurrent streams need distinct workspaces."""
    partial: torch.Tensor
    maxima: torch.Tensor
    sums: torch.Tensor
    num_splits: int

def _decode_parameters(q,k,block_table,num_splits,config):
    c=_configuration(config,decode=True)
    group=q.shape[1]//k.shape[2]
    capacity=k.shape[1] if block_table is None else block_table.shape[1]*k.shape[1]
    mode=c['mode']
    if mode=='auto':
        mode='grouped' if group>=4 and capacity>=1024 else 'vector'
    if num_splits is None:
        blocks=q.shape[0]*(k.shape[2] if mode=='grouped' else q.shape[1])
        desired=_next_power_of_2(max(1,math.ceil(104/max(1,blocks))))
        max_splits=2**max(0,int(math.log2(max(1,capacity//256))))
        num_splits=min(32,desired,max_splits) if capacity>=1024 else 1
    if not isinstance(num_splits,int) or num_splits<1 or num_splits>64 or num_splits&(num_splits-1):
        raise ValueError('num_splits must be a power of two in [1,64]')
    return c,mode,num_splits

def decode_workspace(q,k,*,block_table=None,num_splits=None,config=None):
    """Allocate FP32 (unnormalized output, row max, row sum) split states.

    Allocate before graph capture. Pass the same config/split count to decode.
    Cache capacity rather than live GPU lengths determines scheduling.
    """
    if q.ndim!=3 or k.ndim!=4 or q.device.type!='cuda' or k.device!=q.device:
        raise ValueError('expected CUDA BHD queries and four-dimensional cache')
    _,_,splits=_decode_parameters(q,k,block_table,num_splits,config)
    shape=(q.shape[0],q.shape[1],splits)
    return DecodeWorkspace(torch.empty((*shape,q.shape[2]),device=q.device,dtype=torch.float32),
                           torch.empty(shape,device=q.device,dtype=torch.float32),
                           torch.empty(shape,device=q.device,dtype=torch.float32),splits)
