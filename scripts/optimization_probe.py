"""Small same-input experiments, never a substitute for the frozen full suite.

CUDA Graph timing includes the entire operator and decode merge. Unlike the
release benchmark, this diagnostic measures GPU time only and does no eager
API timing. Variant selection is confined to this process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', action='append', required=True)
    p.add_argument('--variant', choices=['original', 'simt', 'warp', 'fullrow', 'register', 'qregister', 'async', 'pipeline'], default='original')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--external', action='store_true')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--block-n', type=int, default=64)
    p.add_argument('--warps', type=int, default=4)
    p.add_argument('--tiles', help='prefill diagnostic tiles M:N:stages, separated by commas')
    p.add_argument('--fast-math', action='store_true')
    p.add_argument('--manual-bounds', action='store_true',
                   help='use explicit kernel predicates without extra compiler bounds legalization')
    p.add_argument('--modes', default='vector,grouped')
    p.add_argument('--splits', help='comma-separated decode split counts')
    args = p.parse_args()
    if args.output.exists():
        p.error('output already exists')
    import torch
    assert torch.cuda.is_available(), 'This experiment requires a visible CUDA GPU'
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fa5060.workloads import full_suite, make_inputs
    from fa5060.reference import attention_reference
    from fa5060.benchmark import _provider, _validate_output, _event_round, atomic_write_json
    from fa5060.probe import collect_environment, _nvidia_smi
    import fa5060.tilelang.attention as tl
    if args.fast_math or args.manual_bounds:
        import tilelang
        def compile_fast(prim, *, tma=False, manual_bounds=False, manual_sync=False, scalar_memory=False):
            passes = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
                      tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: args.fast_math,
                      tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: args.manual_bounds or manual_bounds,
                      tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: manual_sync}
            if scalar_memory:
                passes[tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE] = True
                passes[tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY] = False
                passes[tilelang.PassConfigKey.TIR_USE_ASYNC_COPY] = False
            if tma:
                passes[tilelang.PassConfigKey.TIR_USE_ASYNC_COPY] = False
            return tilelang.compile(prim, target={'kind':'cuda','arch':'sm_120'},
                                    execution_backend='tvm_ffi', pass_configs=passes)
        tl._compile = compile_fast
    if args.variant in ('simt', 'warp'):
        from importlib import import_module
        _decode_vector_kernel = import_module('fa5060.tilelang._decode_' + args.variant)._decode_vector_kernel
        tl._decode_vector_kernel = _decode_vector_kernel
    elif args.variant in ('fullrow', 'register', 'qregister', 'async'):
        from importlib import import_module
        _prefill_kernel = import_module('fa5060.tilelang._prefill_' + ('mma' if args.variant == 'fullrow' else args.variant))._prefill_kernel
        tl._prefill_kernel = _prefill_kernel
        def grouped_kernel(*args, **kwargs):
            return _prefill_kernel(*args, grouped=True, **kwargs)
        tl._decode_grouped_kernel = grouped_kernel
    elif args.variant == 'pipeline':
        from fa5060.tilelang._decode_pipeline import _prefill_kernel
        def grouped_kernel(*args, **kwargs):
            return _prefill_kernel(*args, grouped=True, **kwargs)
        tl._decode_grouped_kernel = grouped_kernel
    cases = {c.case_id: c for c in full_suite()}
    report = {'purpose': 'diagnostic subset; no full-suite performance claim',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'variant': args.variant, 'fast_math': args.fast_math,
              'manual_bounds': args.manual_bounds, 'environment': collect_environment(),
              'sources': {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in sorted(Path('src/fa5060').rglob('*.py'))},
              'cases': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case_id in args.case:
        case = cases[case_id]
        inputs = make_inputs(case)
        reference = attention_reference(inputs.q, inputs.k, inputs.v, causal=case.causal,
            kv_lens=inputs.kv_lens, block_table=inputs.block_table)
        configs = [('triton', 'ours', None)]
        if case.phase == 'decode':
            split_choices = [int(s) for s in args.splits.split(',')] if args.splits else ((1, 4) if args.quick else (1, 4, 16, 32))
            for splits in split_choices:
                for mode in args.modes.split(','):
                    configs.append(('tilelang', 'ours', {'config': {'mode': mode,
                        'block_n': args.block_n, 'num_warps': args.warps,
                        'num_stages': 1}, 'num_splits': splits}))
        else:
            tiles = [(32,64,2),(64,64,2)] if args.quick else [(32,64,2),(64,64,1),(64,64,2),(64,128,1),(64,128,2),(128,64,1),(128,64,2)]
            if args.tiles:
                tiles = [tuple(map(int, t.split(':'))) for t in args.tiles.split(',')]
            for bm,bn,stages in tiles:
                configs.append(('tilelang','ours',{'config': {'block_m':bm,
                    'block_n':bn,'num_stages':stages},'num_splits':None}))
        if args.external:
            configs += [('external', name, None) for name in
                        (['fa4','torch_flash','torch_cudnn','flashinfer_fa2','flashinfer_fmha_v2']
                         if case.layout=='dense' else ['flashinfer_fa2','flashinfer_fmha_v2'])]
        record = {'case': case.to_dict(), 'trials': []}
        for backend,name,choice in configs:
            trial = {'backend':backend, 'provider':name,'choice':choice}
            start = time.perf_counter()
            try:
                provider = _provider(case, inputs, name, choice,
                                     backend='triton' if backend=='external' else backend)
                trial['validation'] = _validate_output(provider.graph_call(), reference, case.dtype)
                torch.cuda.synchronize()
                trial['first_call_wall_ms'] = (time.perf_counter()-start)*1000
                for _ in range(4): provider.graph_call()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                count = 128 if case.phase == 'decode' else 16
                with torch.cuda.graph(graph,stream=stream):
                    for _ in range(count): result = provider.graph_call()
                torch.cuda.current_stream().wait_stream(stream)
                graph.replay()
                torch.cuda.synchronize()
                _validate_output(result,reference,case.dtype)
                samples = [_event_round(graph.replay,1)[0]/count for _ in range(7)]
                trial.update(status='ok', gpu_us_rounds=samples,
                    gpu_us_p50=statistics.median(samples), graph_calls_per_replay=count,
                    metadata=provider.metadata, telemetry=_nvidia_smi())
                del graph, result, provider
            except Exception as exc:
                trial.update(status='error',reason=f'{type(exc).__name__}: {exc}')
            record['trials'].append(trial)
            print(case_id,backend,name,choice,trial.get('gpu_us_p50',trial.get('reason')),flush=True)
            atomic_write_json(args.output, {**report,'cases':report['cases']+[record]})
        report['cases'].append(record)
        del reference,inputs
        atomic_write_json(args.output,report)


if __name__ == '__main__':
    main()
