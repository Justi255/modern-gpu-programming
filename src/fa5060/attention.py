"""SM120 inference attention, written directly in Triton.

The implementation uses warp-level MMA and register accumulators, not the
SM100-only tcgen05/TMEM path. Cache lengths and page indices stay on the GPU.
The caller supplies valid cache metadata and populates KV including the current
decode token. Kernels do not mutate the cache and do not implement autograd.
"""

from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _kv_bases(BT, b, n, length, head,
              K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
              V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
              TABLE_STRIDE: tl.constexpr, PAGE: tl.constexpr, PAGED: tl.constexpr):
    if PAGED:
        pages = tl.load(BT+b*TABLE_STRIDE+n//PAGE, n < length, other=0)
        kb = pages*K0+(n % PAGE)*K1+head*K2
        vb = pages*V0+(n % PAGE)*V1+head*V2
    else:
        kb = b*K0+n*K1+head*K2
        vb = b*V0+n*V1+head*V2
    return kb, vb


@triton.jit
def _prefill_kernel(Q, K, V, Lengths, BT, O,
                    Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
                    K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                    V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                    O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr,
                    TABLE_STRIDE: tl.constexpr, SQ: tl.constexpr,
                    HQ: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
                    CAPACITY: tl.constexpr, PAGE: tl.constexpr,
                    HAS_LENGTHS: tl.constexpr, PAGED: tl.constexpr,
                    CAUSAL: tl.constexpr, SCALE_LOG2: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh//HQ, bh % HQ
    length = tl.load(Lengths+b) if HAS_LENGTHS else CAPACITY
    m = block*BM+tl.arange(0, BM)
    n = tl.arange(0, BN)
    d = tl.arange(0, D)
    q = tl.load(Q+b*Q0+m[:, None]*Q1+h*Q2+d[None, :], m[:, None] < SQ, other=0)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, D), tl.float32)
    stop = length
    if CAUSAL:
        stop = tl.maximum(0, tl.minimum(length, (block+1)*BM+length-SQ))
    for start in tl.range(0, stop, BN, num_stages=STAGES):
        positions = start+n
        kb, vb = _kv_bases(BT,b,positions,length,h//GROUP,
                          K0,K1,K2,V0,V1,V2,TABLE_STRIDE,PAGE,PAGED)
        key = tl.load(K+kb[None, :]+d[:, None], positions[None, :] < length, other=0)
        scores = tl.dot(q, key)*SCALE_LOG2
        valid = positions[None, :] < length
        if CAUSAL:
            valid = valid & (positions[None, :] <= m[:, None]+length-SQ)
        scores = tl.where(valid, scores, -float('inf'))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(new_maximum == -float('inf'), 0., new_maximum)
        correction = tl.exp2(maximum-safe_maximum)
        probabilities = tl.exp2(scores-safe_maximum[:, None])
        value = tl.load(V+vb[:, None]+d[None, :], positions[:, None] < length, other=0)
        accumulator *= correction[:, None]
        accumulator = tl.dot(probabilities.to(q.dtype), value, accumulator)
        denominator = denominator*correction+tl.sum(probabilities, 1)
        maximum = new_maximum
    output = accumulator/tl.where(denominator > 0., denominator, 1.)[:, None]
    tl.store(O+b*O0+m[:, None]*O1+h*O2+d[None, :], output, m[:, None] < SQ)


@triton.jit
def _decode_vector_kernel(Q,K,V,Lengths,BT,O,Partial,Maxima,Sums,
                          Q0:tl.constexpr,Q1:tl.constexpr,
                          K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,
                          V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,
                          O0:tl.constexpr,O1:tl.constexpr,
                          TABLE_STRIDE:tl.constexpr,HQ:tl.constexpr,GROUP:tl.constexpr,
                          D:tl.constexpr,CAPACITY:tl.constexpr,PAGE:tl.constexpr,
                          PAGED:tl.constexpr,SPLITS:tl.constexpr,
                          SCALE_LOG2:tl.constexpr,BN:tl.constexpr):
    h,b,split=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    d=tl.arange(0,D)
    n=tl.arange(0,BN)
    query=tl.load(Q+b*Q0+h*Q1+d).to(tl.float32)
    length=tl.load(Lengths+b)
    chunk=tl.cdiv(CAPACITY,SPLITS)
    lo=split*chunk
    stop=tl.minimum(lo+chunk,length)
    maximum=tl.full((),-float('inf'),tl.float32)
    denominator=tl.full((),0.,tl.float32)
    accumulator=tl.zeros((D,),tl.float32)
    for start in range(lo,stop,BN):
        positions=start+n
        kb,vb=_kv_bases(BT,b,positions,length,h//GROUP,
                        K0,K1,K2,V0,V1,V2,TABLE_STRIDE,PAGE,PAGED)
        valid=(positions<stop)
        key=tl.load(K+kb[:,None]+d[None,:],valid[:,None],other=0).to(tl.float32)
        scores=tl.sum(key*query[None,:],1)*SCALE_LOG2
        scores=tl.where(valid,scores,-float('inf'))
        updated=tl.maximum(maximum,tl.max(scores,0))
        safe=tl.where(updated == -float('inf'),0.,updated)
        correction=tl.exp2(maximum-safe)
        p=tl.exp2(scores-safe)
        value=tl.load(V+vb[:,None]+d[None,:],valid[:,None],other=0).to(tl.float32)
        accumulator=accumulator*correction+tl.sum(p[:,None]*value,0)
        denominator=denominator*correction+tl.sum(p,0)
        maximum=updated
    if SPLITS == 1:
        tl.store(O+b*O0+h*O1+d,accumulator/tl.where(denominator>0,denominator,1.))
    else:
        state=(b*HQ+h)*SPLITS+split
        tl.store(Partial+state*D+d,accumulator)
        tl.store(Maxima+state,maximum)
        tl.store(Sums+state,denominator)


@triton.jit
def _decode_grouped_kernel(Q,K,V,Lengths,BT,O,Partial,Maxima,Sums,
                           Q0:tl.constexpr,Q1:tl.constexpr,
                           K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,
                           V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,
                           O0:tl.constexpr,O1:tl.constexpr,
                           TABLE_STRIDE:tl.constexpr,HQ:tl.constexpr,GROUP:tl.constexpr,
                           D:tl.constexpr,CAPACITY:tl.constexpr,PAGE:tl.constexpr,
                           PAGED:tl.constexpr,SPLITS:tl.constexpr,
                           SCALE_LOG2:tl.constexpr,BN:tl.constexpr,
                           GM:tl.constexpr,STAGES:tl.constexpr):
    kvhead,b,split=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    g=tl.arange(0,GM)
    heads=kvhead*GROUP+g
    d=tl.arange(0,D)
    n=tl.arange(0,BN)
    query=tl.load(Q+b*Q0+heads[:,None]*Q1+d[None,:],g[:,None]<GROUP,other=0)
    length=tl.load(Lengths+b)
    chunk=tl.cdiv(CAPACITY,SPLITS)
    lo=split*chunk
    stop=tl.minimum(length,lo+chunk)
    maximum=tl.full((GM,),-float('inf'),tl.float32)
    denominator=tl.zeros((GM,),tl.float32)
    accumulator=tl.zeros((GM,D),tl.float32)
    for start in tl.range(lo,stop,BN,num_stages=STAGES):
        positions=start+n
        kb,vb=_kv_bases(BT,b,positions,length,kvhead,
                        K0,K1,K2,V0,V1,V2,TABLE_STRIDE,PAGE,PAGED)
        key=tl.load(K+kb[None,:]+d[:,None],positions[None,:]<stop,other=0)
        scores=tl.dot(query,key)*SCALE_LOG2
        scores=tl.where(positions[None,:]<stop,scores,-float('inf'))
        updated=tl.maximum(maximum,tl.max(scores,1))
        safe=tl.where(updated == -float('inf'),0.,updated)
        correction=tl.exp2(maximum-safe)
        p=tl.exp2(scores-safe[:,None])
        value=tl.load(V+vb[:,None]+d[None,:],positions[:,None]<stop,other=0)
        accumulator*=correction[:,None]
        accumulator=tl.dot(p.to(query.dtype),value,accumulator)
        denominator=denominator*correction+tl.sum(p,1)
        maximum=updated
    if SPLITS == 1:
        output=accumulator/tl.where(denominator>0,denominator,1.)[:,None]
        tl.store(O+b*O0+heads[:,None]*O1+d[None,:],output,g[:,None]<GROUP)
    else:
        states=(b*HQ+heads)*SPLITS+split
        tl.store(Partial+states[:,None]*D+d[None,:],accumulator,g[:,None]<GROUP)
        tl.store(Maxima+states,maximum,g<GROUP)
        tl.store(Sums+states,denominator,g<GROUP)


@triton.jit
def _merge_splits(Partial,Maxima,Sums,O,O0:tl.constexpr,O1:tl.constexpr,
                  HQ:tl.constexpr,D:tl.constexpr,SPLITS:tl.constexpr):
    h,b=tl.program_id(0),tl.program_id(1)
    splits=tl.arange(0,SPLITS)
    d=tl.arange(0,D)
    states=(b*HQ+h)*SPLITS+splits
    maxima=tl.load(Maxima+states)
    sums=tl.load(Sums+states)
    global_max=tl.max(maxima,0)
    safe=tl.where(global_max == -float('inf'),0.,global_max)
    corrections=tl.exp2(maxima-safe)
    denominator=tl.sum(sums*corrections,0)
    partial=tl.load(Partial+states[:,None]*D+d[None,:])
    result=tl.sum(partial*corrections[:,None],0)/tl.where(denominator>0,denominator,1.)
    tl.store(O+b*O0+h*O1+d,result)


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


def prefill(q,k,v,*,causal=True,scale=None,kv_lens=None,block_table=None,out=None,config=None):
    """Exact attention in BSHD layout; causal positions are right aligned.

    Dense KV is [B,capacity,Hkv,D]; paged KV is [pages,page_size,Hkv,D].
    GQA maps head h to h//(Hq/Hkv). Output is zero for entirely masked rows.
    `config` selects block_m/block_n/num_warps/num_stages/loader explicitly;
    there is no implicit online autotuning or external-library fallback.
    """
    b,hq,hkv,d,capacity=_validate(q,k,v,kv_lens,block_table,4)
    c=_configuration(config)
    output=_output(q,k,v,out)
    if q.numel()==0:
        return output
    if c['loader']=='tma':
        return _prefill_tma(q,k,v,output,causal,scale,kv_lens,block_table,c)
    if 4*c['block_n']*d*c['num_stages']+2*c['block_m']*d>96*1024:
        raise ValueError('configuration exceeds conservative SM120 shared memory budget')
    table_stride=0 if block_table is None else block_table.stride(0)
    _prefill_kernel[(triton.cdiv(q.shape[1],c['block_m']),b*hq)](
        q,k,v,q if kv_lens is None else kv_lens,q if block_table is None else block_table,output,
        *q.stride()[:3],*k.stride()[:3],*v.stride()[:3],*output.stride()[:3],
        table_stride,q.shape[1],hq,hq//hkv,d,capacity,k.shape[1],
        kv_lens is not None,block_table is not None,bool(causal),_scale(d,scale),
        c['block_m'],c['block_n'],c['num_stages'],num_warps=c['num_warps'],num_stages=c['num_stages'])
    return output


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
        desired=triton.next_power_of_2(max(1,math.ceil(104/max(1,blocks))))
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


def decode(q,k,v,*,kv_lens,block_table=None,scale=None,out=None,workspace=None,num_splits=None,config=None):
    """BHD decode from an already populated cache, including the current token.

    `mode` is vector, grouped (Tensor Core GQA), or auto. Split-KV uses a
    separate stable merge kernel; all stages are included in benchmark timing.
    Supply `out` and `decode_workspace(...)` for reusable allocations/graphs.
    """
    b,hq,hkv,d,capacity=_validate(q,k,v,kv_lens,block_table,3)
    c,mode,splits=_decode_parameters(q,k,block_table,num_splits,config)
    output=_output(q,k,v,out)
    if b==0:
        return output
    if splits>1:
        if workspace is None:
            workspace=decode_workspace(q,k,block_table=block_table,num_splits=splits,config=c)
        shape=(b,hq,splits)
        if (not isinstance(workspace,DecodeWorkspace) or workspace.num_splits!=splits or
            workspace.partial.shape!=(*shape,d) or workspace.maxima.shape!=shape or workspace.sums.shape!=shape):
            raise ValueError('workspace shape/split count does not match this decode')
        for x in (workspace.partial,workspace.maxima,workspace.sums):
            if x.device!=q.device or x.dtype!=torch.float32 or not x.is_contiguous():
                raise ValueError('workspace must contain contiguous CUDA FP32 tensors')
        partial,maxima,sums=workspace.partial,workspace.maxima,workspace.sums
    else:
        partial=maxima=sums=output
    args=(q,k,v,kv_lens,q if block_table is None else block_table,output,partial,maxima,sums,
          *q.stride()[:2],*k.stride()[:3],*v.stride()[:3],*output.stride()[:2],
          0 if block_table is None else block_table.stride(0),hq,hq//hkv,d,capacity,k.shape[1],
          block_table is not None,splits,_scale(d,scale),c['block_n'])
    if mode=='grouped':
        gm=max(16,triton.next_power_of_2(hq//hkv))
        if 4*c['block_n']*d*c['num_stages']+2*gm*d>96*1024:
            raise ValueError('configuration exceeds SM120 shared memory budget')
        _decode_grouped_kernel[(hkv,b,splits)](*args,gm,c['num_stages'],
            num_warps=c['num_warps'],num_stages=c['num_stages'])
    else:
        _decode_vector_kernel[(hq,b,splits)](*args,num_warps=c['num_warps'],num_stages=1)
    if splits>1:
        _merge_splits[(hq,b)](partial,maxima,sums,output,*output.stride()[:2],hq,d,splits,num_warps=4)
    return output


def _prefill_tma(q,k,v,out,causal,scale,kv_lens,block_table,c):
    from .attention_tma import prefill_tma

    return prefill_tma(q,k,v,out,causal,scale,kv_lens,block_table,c)
