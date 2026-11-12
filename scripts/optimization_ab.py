"""Interleaved same-input A/B experiments; this subset cannot establish SOTA."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics


def _fast_compile(prim, *, tma=False, manual_bounds=False, manual_sync=False, scalar_memory=False):
    import tilelang
    key = tilelang.PassConfigKey
    passes = {key.TL_DISABLE_WARP_SPECIALIZED: True, key.TL_ENABLE_FAST_MATH: True,
              key.TL_DISABLE_SAFE_MEMORY_ACCESS: manual_bounds,
              key.TL_DISABLE_THREAD_STORAGE_SYNC: manual_sync}
    if tma or scalar_memory:
        passes[key.TIR_USE_ASYNC_COPY] = False
    if scalar_memory:
        passes[key.TIR_DISABLE_VECTORIZE] = True
        passes[key.TL_ENABLE_ASYNC_COPY] = False
    return tilelang.compile(prim, target={'kind':'cuda', 'arch':'sm_120'},
                            execution_backend='tvm_ffi', pass_configs=passes)


@contextmanager
def factory_variant(api, variant, *, fast_math=False):
    original = api._prefill_kernel, api._decode_grouped_kernel
    original_compile = getattr(api, '_compile', None)
    candidate = None
    try:
        if variant in ('register', 'qregister', 'async', 'qasync'):
            from importlib import import_module
            _prefill_kernel = import_module('fa5060.tilelang._prefill_' + variant)._prefill_kernel
            candidate = _prefill_kernel
            _prefill_kernel.cache_clear()
            api._prefill_kernel = _prefill_kernel
        elif variant in ('pipeline', 'pipeline_scalar', 'pipeline_segmented'):
            from fa5060.tilelang._decode_pipeline import _prefill_kernel
            candidate = _prefill_kernel
            _prefill_kernel.cache_clear()
            api._decode_grouped_kernel = lambda *a, **kw: _prefill_kernel(
                *a, grouped=True, batched_reduce=variant != 'pipeline_scalar',
                segmented_reduce=variant == 'pipeline_segmented', **kw)
        elif variant not in ('reference', 'external'):
            raise ValueError(f'unknown variant {variant}')
        if fast_math:
            # Kernel caches include the factory arguments, not this temporary
            # diagnostic compiler hook. Drop the memory cache before lowering.
            clear = getattr(api._prefill_kernel, 'cache_clear', None)
            if clear: clear()
            api._compile = _fast_compile
        yield
    finally:
        if fast_math:
            clear = getattr(candidate or api._prefill_kernel, 'cache_clear', None)
            if clear: clear()
        api._prefill_kernel, api._decode_grouped_kernel = original
        if fast_math:
            api._compile = original_compile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, action='append')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--trials', type=Path,
                        help='JSON mapping phase to explicit variant/config/num_splits trials')
    args = parser.parse_args()
    if args.output.exists(): parser.error('output exists')
    import torch
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fa5060.workloads import full_suite, make_inputs
    from fa5060.reference import attention_reference
    from fa5060.benchmark import _provider, _validate_output, _event_round, atomic_write_json, PROVIDER_NAMES
    from fa5060.probe import collect_environment, _nvidia_smi
    import fa5060.tilelang.attention as api
    report = {'purpose': __doc__, 'full_suite_target_met': False,
        'environment': collect_environment(), 'seed': 20261102,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sources': {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in sorted(Path('src/fa5060').rglob('*.py'))}, 'cases': []}
    cases = {c.case_id: c for c in full_suite()}
    unknown = set(args.case) - cases.keys()
    if unknown:
        parser.error(f'unknown frozen case IDs: {sorted(unknown)}')
    trial_spec = json.loads(args.trials.read_text()) if args.trials else None
    if trial_spec is not None:
        report['trial_specification'] = trial_spec
    for case_id in args.case:
        case = cases[case_id]
        data = make_inputs(case)
        ref = attention_reference(data.q, data.k, data.v, causal=case.causal,
                                  kv_lens=data.kv_lens, block_table=data.block_table)
        if case.phase == 'decode':
            own = [('reference', {'mode':'grouped','block_n':64,'num_stages':1}, 16),
                   ('pipeline', {'mode':'grouped','block_n':32,'num_stages':2}, 32)]
        else:
            own = [('reference', {'block_m':64,'block_n':64,'num_stages':2}, None),
                   ('register', {'block_m':64,'block_n':32 if case.head_dim==128 else 64,
                                 'num_stages':1}, None)]
        if trial_spec is not None:
            own = [(t['variant'], t['config'], t['num_splits'], t.get('fast_math', False))
                   for t in trial_spec[case.phase]]
        else:
            own = [(v,cfg,s,False) for v,cfg,s in own]
        requests = [(v,'ours',{'config':cfg,'num_splits':s}, fast) for v,cfg,s,fast in own]
        requests += [('external', n, None, False) for n in PROVIDER_NAMES if n != 'ours']
        records, graphs = [], []
        for variant, name, choice, fast in requests:
            record = {'variant':variant,'provider':name,'choice':choice,'fast_math':fast}
            try:
                with factory_variant(api, variant, fast_math=fast):
                    provider = _provider(case,data,name,choice,backend='tilelang')
                    _validate_output(provider.graph_call(),ref,case.dtype)
                    for _ in range(3): provider.graph_call()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    calls = 512 if case.phase=='decode' else 16
                    with torch.cuda.graph(graph):
                        for _ in range(calls): output = provider.graph_call()
                    graph.replay()
                    torch.cuda.synchronize()
                    record['validation'] = _validate_output(output,ref,case.dtype)
                record.update(status='ok', metadata=provider.metadata, gpu_us_rounds=[],
                              graph_calls_per_replay=calls)
                graphs.append((record, graph, calls, output, provider))
            except Exception as exc:
                record.update(status='error',reason=f'{type(exc).__name__}: {exc}')
            records.append(record)
        for round_index in range(9):
            ordered = graphs if round_index % 2 == 0 else list(reversed(graphs))
            for rec, graph, calls, _, _ in ordered:
                rec['gpu_us_rounds'].append(_event_round(graph.replay,1)[0]/calls)
                rec.setdefault('telemetry_rounds', []).append(_nvidia_smi())
        for rec, _, _, _, _ in graphs:
            rec['gpu_us_p50'] = statistics.median(rec['gpu_us_rounds'])
            print(case_id,rec['variant'],rec['provider'],rec['gpu_us_p50'],flush=True)
        record = {'case':case.to_dict(),'trials':records,'telemetry_after':_nvidia_smi(),
                  'order':'forward in even rounds; reverse in odd rounds; one repeated-operator graph per trial'}
        report['cases'].append(record)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        atomic_write_json(args.output,report)
        del graphs, data, ref


if __name__ == '__main__': main()
