"""CPU regressions for candidate swapping in interleaved measurements."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _context():
    path = Path(__file__).parents[1] / 'scripts' / 'optimization_ab.py'
    spec = importlib.util.spec_from_file_location('optimization_ab_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.factory_variant


@pytest.mark.parametrize('variant', ['reference', 'external'])
def test_unmodified_providers_are_allowed(variant):
    prefill, decode = object(), object()
    api = SimpleNamespace(_prefill_kernel=prefill, _decode_grouped_kernel=decode)
    with _context()(api, variant):
        assert api._prefill_kernel is prefill
        assert api._decode_grouped_kernel is decode
    assert api._prefill_kernel is prefill
    assert api._decode_grouped_kernel is decode


def test_unknown_variant_fails_without_mutating_factories():
    prefill, decode = object(), object()
    api = SimpleNamespace(_prefill_kernel=prefill, _decode_grouped_kernel=decode)
    with pytest.raises(ValueError, match='unknown variant'):
        with _context()(api, 'typo'):
            pass
    assert api._prefill_kernel is prefill
    assert api._decode_grouped_kernel is decode


def test_diagnostic_compile_hook_cannot_leave_fast_kernel_in_factory_cache():
    from functools import lru_cache
    @lru_cache(None)
    def factory(shape):
        return object()
    original_compile = object()
    api = SimpleNamespace(_prefill_kernel=factory, _decode_grouped_kernel=object(),
                          _compile=original_compile)
    factory(1)
    with _context()(api, 'reference', fast_math=True):
        assert factory.cache_info().currsize == 0
        assert api._compile is not original_compile
        factory(1)
    assert api._compile is original_compile
    assert factory.cache_info().currsize == 0


def test_optimized_configuration_is_explicit_and_checked():
    pytest.importorskip('torch')
    from fa5060.tilelang._contracts import _configuration
    assert _configuration(None)['implementation'] == 'reference'
    assert _configuration({'implementation': 'optimized'})['implementation'] == 'optimized'
    with pytest.raises(ValueError, match='implementation'):
        _configuration({'implementation': 'typo'})


def test_explicit_benchmark_choice_is_forwarded_and_recorded(tmp_path):
    from fa5060.benchmark import run_benchmark, PROVIDER_NAMES
    from fa5060.workloads import AttentionCase
    choice = {'config': {'implementation': 'optimized'}, 'num_splits': None}
    case = AttentionCase('decode', 1, 512, 64, 8, 'fp16', 'paged')
    def runner(case, *, rounds, tune, backend, ours_choice):
        assert ours_choice == choice and not tune and backend == 'tilelang'
        return {**case.to_dict(), 'providers': {name: {'status': 'unsupported',
                'reason': 'CPU stub'} for name in PROVIDER_NAMES}}
    report = run_benchmark([case], output=tmp_path/'report.json', environment={},
                           backend='tilelang', ours_choice=choice, case_runner=runner)
    assert report['configuration_mode'] == 'explicit_configuration'
    assert report['ours_choice'] == choice
    with pytest.raises(ValueError, match='mutually exclusive'):
        run_benchmark([], output=tmp_path/'bad.json', environment={}, tune=True,
                      ours_choice=choice)
