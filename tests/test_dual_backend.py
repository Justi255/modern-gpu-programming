"""CPU contracts for backend evidence; no compiler or GPU is needed."""

import copy
import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from fa5060 import benchmark, experiments
from fa5060.workloads import AttentionCase


def _case():
    return AttentionCase("decode", 1, 512, 64, 8, "fp16", "paged")


def _ok(us):
    return {"status": "ok", "validated": True, "gpu_us_p50": us}


def _record(case, us=10):
    return {**case.to_dict(), "seed": 20261102, "providers": {
        "ours": _ok(us), "flashinfer_fa2": _ok(20),
        **{name: {"status": "unsupported", "reason": "not supported"}
           for name in ("torch_flash", "torch_cudnn", "torch_efficient", "fa4", "flashinfer_fmha_v2")},
    }}


def _environment():
    return {"cuda_build": "13.0", "cuda_available": True,
            "python": {"version": "3.13.9", "implementation": "CPython", "compiler": "GCC 14"},
            "platform": {"system": "Linux", "release": "test", "machine": "x86_64"},
            "packages": {"torch": "2.10", "triton": "3.6", "tilelang": "0.1.7",
                         "apache-tvm-ffi": "0.1.4", "flashinfer-python": "0.7.2"},
            "gpus": [{"index": 0, "name": "test GPU", "compute_capability": [12, 0],
                      "sm_count": 26, "total_memory_bytes": 8000000000}],
            "nvidia_smi": {"devices": [{"index": 0, "driver_version": "595.99", "power_draw_watts": 20}]},
            "compiler": {"nvcc": {"output": "release 13.1"}, "ptxas": {"output": "release 13.1"}}}


def _report(backend, us):
    return {"schema_version": 1, "backend": backend, "suite": "smoke", "rounds": 5,
            "tune": True, "seed": 20261102, "environment": _environment(),
            "timing": {"graph_boundary": "complete operator"}, "cases": [_record(_case(), us)]}


def test_backend_report_preserves_legacy_runner_and_labels_tilelang_runner(tmp_path):
    def legacy_runner(case, *, rounds, tune):
        return _record(case)

    report = benchmark.run_benchmark([_case()], output=tmp_path / "old.json", environment={},
                                     case_runner=legacy_runner)
    assert report["backend"] == "triton"
    assert benchmark.report_backend({"schema_version": 1}) == "triton"

    def tilelang_runner(case, *, rounds, tune, backend):
        return {**_record(case), "backend": backend}

    report = benchmark.run_benchmark([_case()], output=tmp_path / "new.json", environment=_environment(),
                                     backend="tilelang", case_runner=tilelang_runner)
    assert report["backend"] == report["cases"][0]["backend"] == "tilelang"
    assert report["backend_packages"]["tilelang"] == "0.1.7"
    assert report["configuration_mode"] == "public_default"


def test_invalid_backend_fails_before_checkpoint_or_gpu_import(tmp_path):
    target = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="backend"):
        benchmark.run_benchmark([], output=target, environment={}, backend="cuda")
    with pytest.raises(ValueError, match="backend"):
        experiments.run("basics", backend="cuda")
    assert not target.exists()


def test_tilelang_provider_uses_native_pages_and_captures_split_merge(monkeypatch):
    torch = pytest.importorskip("torch")
    from fa5060.workloads import make_inputs

    case = _case()
    inputs = make_inputs(case, device="cpu")
    captured = []

    def decode_workspace(q, k, **kwargs):
        assert k is inputs.k and kwargs["block_table"] is inputs.block_table
        return object()

    def decode(q, k, v, **kwargs):
        assert q is inputs.q and k is inputs.k and v is inputs.v
        assert kwargs["block_table"] is inputs.block_table
        captured.append(kwargs)
        return kwargs.get("out", q)

    monkeypatch.setitem(sys.modules, "fa5060.tilelang", SimpleNamespace(
        decode=decode, decode_workspace=decode_workspace, prefill=None))
    provider = benchmark._ours_provider(case, inputs, backend="tilelang")
    assert provider.graph_call().shape == inputs.q.shape
    assert provider.eager_call().shape == inputs.q.shape
    assert captured[0]["workspace"] is not None and "workspace" not in captured[1]
    assert provider.metadata["backend"] == "tilelang"
    assert provider.metadata["native_paged"] is True
    assert provider.metadata["graph_boundary"] == "complete operator including all decode splits and merge"


def test_disclosed_tilelang_grid_preserves_failures_and_uses_backend_configs(monkeypatch):
    case = _case()
    grid = benchmark.tuning_candidates(case, backend="tilelang")
    assert grid[0] == {"config": None, "num_splits": None}
    assert {c["num_splits"] for c in grid} >= {None, 1, 4, 16}

    def trial(case, inputs, reference, name, rounds, choice=None, backend="triton"):
        assert backend == "tilelang"
        if choice == grid[0]:
            return {"status": "incorrect", "validated": False, "reason": "bad result"}
        return _ok(10 if choice == grid[1] else 20)

    monkeypatch.setattr(benchmark, "_run_provider", trial)
    result = benchmark._run_ours(case, None, None, 5, True, backend="tilelang")
    assert result["selected_config"] == grid[1]
    assert result["tuning"]["candidates"][0]["status"] == "incorrect"
    assert result["tuning"]["correctness_failures"] == 1


def test_first_call_compile_cost_is_saved_outside_steady_state(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    provider = benchmark.Provider(lambda: torch.ones(1), lambda: torch.ones(1), {"backend": "tilelang"})
    monkeypatch.setattr(benchmark, "_provider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(benchmark, "_time_provider", lambda *args: {"gpu_us_p50": 2, "warmup_wall_ms": 3})
    result = benchmark._run_provider(_case(), None, torch.ones(1), "ours", 5, backend="tilelang")
    assert result["status"] == "ok"
    assert result["startup"]["prepare_wall_ms"] >= 0
    assert result["startup"]["graph_first_call_wall_ms"] >= 0
    assert result["startup"]["eager_first_call_wall_ms"] >= 0
    assert result["gpu_us_p50"] == 2 and result["warmup_wall_ms"] == 3


def test_both_backend_clis_and_imports_work_without_torch_or_compilers():
    code = '''
import importlib.abc, importlib, sys
class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'tilelang', 'tvm', 'tvm_ffi'}:
            raise AssertionError('GPU import attempted: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
module = importlib.import_module(sys.argv[1])
module.main(['--help'])
'''
    for module in ("fa5060.benchmark", "fa5060.experiments"):
        result = subprocess.run([sys.executable, "-c", code, module], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "--backend {triton,tilelang}" in result.stdout


def _comparison_module():
    return importlib.import_module("scripts.compare_backends")


def test_pair_comparison_joins_case_ids_and_separates_external_baselines():
    comparison = _comparison_module()
    triton, tilelang = _report("triton", 10), _report("tilelang", 5)
    tilelang["environment"]["timestamp_utc"] = "later observation"
    tilelang["environment"]["nvidia_smi"]["devices"][0]["power_draw_watts"] = 35
    summary = comparison.compare_reports(triton, tilelang)
    row = summary["cases"][0]
    assert row["tilelang_over_triton_latency"] == .5
    assert row["tilelang_speedup_vs_triton"] == 2
    assert row["tilelang_speedup_vs_fastest_external"] == 4
    assert row["triton_speedup_vs_fastest_external"] == 2
    assert summary["phases"]["decode"]["geometric_mean_tilelang_speedup_vs_triton"] == 2
    legacy = copy.deepcopy(triton)
    del legacy["backend"]
    assert comparison.compare_reports(legacy, tilelang)["triton_backend"] == "triton"


@pytest.mark.parametrize("change", ["package", "driver", "gpu", "shape", "seed", "mode", "missing", "duplicate", "rounds", "timing"])
def test_pair_comparison_rejects_unfair_or_ambiguous_evidence(change):
    triton, tilelang = _report("triton", 10), _report("tilelang", 5)
    if change == "package":
        tilelang["environment"]["packages"]["torch"] = "different"
    elif change == "driver":
        tilelang["environment"]["nvidia_smi"]["devices"][0]["driver_version"] = "different"
    elif change == "gpu":
        tilelang["environment"]["gpus"][0]["sm_count"] = 100
    elif change == "shape":
        tilelang["cases"][0]["shape"]["q"] = [1, 32, 128]
    elif change == "seed":
        tilelang["cases"][0]["seed"] = 123
    elif change == "mode":
        tilelang["tune"] = False
    elif change == "missing":
        tilelang["cases"] = []
    elif change == "duplicate":
        tilelang["cases"].append(tilelang["cases"][0])
    elif change == "rounds":
        tilelang["rounds"] = 1
    else:
        tilelang["timing"]["graph_boundary"] = "excludes merge"
    with pytest.raises(ValueError):
        _comparison_module().compare_reports(triton, tilelang)


def test_pair_cli_writes_three_reviewable_artifacts_without_changing_inputs(tmp_path):
    comparison = _comparison_module()
    inputs = []
    for backend, us in (("triton", 10), ("tilelang", 5)):
        path = tmp_path / f"{backend}.json"
        path.write_text(json.dumps(_report(backend, us)))
        inputs.append(path)
    before = [path.read_bytes() for path in inputs]
    output = tmp_path / "evidence"
    assert comparison.main(["--triton", str(inputs[0]), "--tilelang", str(inputs[1]),
                            "--output-dir", str(output)]) == 0
    assert [path.read_bytes() for path in inputs] == before
    assert sorted(path.name for path in output.iterdir()) == ["comparison.csv", "comparison.json", "comparison.md"]
    assert "TileLang / Triton" in (output / "comparison.md").read_text()
    assert json.loads((output / "comparison.json").read_text())["sources"]["triton"]["sha256"]


def test_teaching_record_attributes_selected_backend_and_first_call(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(experiments, "_time_cuda", lambda *args: {"median_ms": 1.25})
    result = experiments._record(
        torch, _environment(), name="vector_add", inputs={"length": 2},
        operation=lambda: torch.tensor([3., 5.]), expected=torch.tensor([3., 5.]),
        rtol=0, atol=0, reference="literal CPU check", backend="tilelang")
    assert result["backend"] == result["provider"]["backend"] == "tilelang"
    assert result["provider"]["package_versions"] == {"tilelang": "0.1.7", "apache-tvm-ffi": "0.1.4"}
    assert result["startup"]["first_call_wall_ms"] >= 0
    assert result["latency_ms"] == 1.25


def test_pair_comparison_joins_in_different_order_and_preserves_missing_timings():
    triton, tilelang = _report("triton", 10), _report("tilelang", 5)
    second = AttentionCase("decode", 1, 512, 128, 8, "bf16", "paged")
    triton["cases"].append(_record(second, 20))
    failed = _record(second, 0)
    failed["providers"]["ours"] = {"status": "error", "validated": False, "reason": "compile failed"}
    tilelang["cases"].insert(0, failed)
    result = _comparison_module().compare_reports(triton, tilelang)
    assert result["cases"][0]["tilelang_speedup_vs_triton"] == 2
    assert result["cases"][1]["tilelang_speedup_vs_triton"] is None
    assert result["phases"]["decode"]["comparable_cases"] == 1
    assert result["phases"]["decode"]["unavailable_case_ids"] == [second.case_id]
    assert result["phases"]["decode"]["all_backend_outputs_correct"] is False


def test_pair_comparison_rejects_missing_workload_identity():
    triton, tilelang = _report("triton", 10), _report("tilelang", 5)
    del tilelang["cases"][0]["shape"]
    with pytest.raises(ValueError, match="workload"):
        _comparison_module().compare_reports(triton, tilelang)


def test_first_call_validation_releases_allocated_outputs_before_timing(monkeypatch):
    import weakref
    torch = pytest.importorskip("torch")
    outputs = []

    def call():
        output = torch.ones(1)
        outputs.append(weakref.ref(output))
        return output

    def time_provider(*args):
        assert all(output() is None for output in outputs)
        return {"gpu_us_p50": 2}

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(benchmark, "_provider", lambda *args, **kwargs: benchmark.Provider(call, call, {}))
    monkeypatch.setattr(benchmark, "_time_provider", time_provider)
    result = benchmark._run_provider(_case(), None, torch.ones(1), "ours", 5)
    assert result["status"] == "ok", result.get("reason")


def test_external_summary_requires_complete_baselines_and_correctness_per_backend():
    comparison = _comparison_module()
    triton, tilelang = _report("triton", 10), _report("tilelang", 5)
    second = AttentionCase("decode", 1, 512, 128, 8, "bf16", "paged")
    triton["cases"].append(_record(second, 10))
    tilelang["cases"].append(_record(second, 20))
    tilelang["cases"][1]["providers"]["flashinfer_fmha_v2"] = {
        "status": "error", "reason": "native provider failed"}
    summary = comparison.compare_reports(triton, tilelang)
    phase = summary["phases"]["decode"]
    assert summary["cases"][1]["tilelang_speedup_vs_fastest_external"] == 1
    assert phase["triton_baseline_coverage_complete"] is True
    assert phase["triton_baseline_coverage_count"] == 2
    assert phase["triton_baseline_coverage_invalid_case_ids"] == []
    assert phase["tilelang_baseline_coverage_complete"] is False
    assert phase["tilelang_baseline_coverage_count"] == 1
    assert phase["tilelang_baseline_coverage_invalid_case_ids"] == [second.case_id]
    assert phase["tilelang_external_comparable_cases"] == 1
    assert phase["tilelang_geometric_mean_speedup_vs_fastest_external"] == 4
    assert phase["triton_geometric_mean_speedup_vs_fastest_external"] == 2

    # Even covered rows cannot qualify if any recorded numerical check failed.
    tilelang["cases"][1]["providers"]["flashinfer_fmha_v2"] = {
        "status": "unsupported", "reason": "not supported"}
    tilelang["cases"][1]["providers"]["ours"]["tuning"] = {"correctness_failures": 1}
    summary = comparison.compare_reports(triton, tilelang)
    phase = summary["phases"]["decode"]
    assert phase["tilelang_baseline_coverage_complete"] is True
    assert phase["tilelang_external_comparable_cases"] == 1
    assert phase["tilelang_external_invalid_case_ids"] == [second.case_id]
    assert phase["tilelang_geometric_mean_speedup_vs_fastest_external"] == 4


def test_complete_frozen_ids_do_not_imply_complete_external_baseline_coverage():
    from fa5060.workloads import full_suite
    comparison = _comparison_module()
    triton, tilelang = _report("triton", 10), _report("tilelang", 5)
    for report, ours_us in ((triton, 10), (tilelang, 5)):
        report["suite"] = "full"
        report["cases"] = [_record(case, ours_us) for case in full_suite()]
        for record in report["cases"]:
            record["providers"]["flashinfer_fmha_v2"] = {
                "status": "error", "reason": "native provider failed"}
    summary = comparison.compare_reports(triton, tilelang)
    assert summary["frozen_suite_complete"] is True
    assert len(summary["cases"]) == 312
    assert all(row["tilelang_speedup_vs_fastest_external"] == 4 for row in summary["cases"])
    for phase, count in (("prefill", 144), ("decode", 168)):
        for backend in ("triton", "tilelang"):
            values = summary["phases"][phase]
            assert values[f"{backend}_baseline_coverage_complete"] is False
            assert values[f"{backend}_baseline_coverage_count"] == 0
            assert len(values[f"{backend}_baseline_coverage_invalid_case_ids"]) == count
            assert values[f"{backend}_external_comparable_cases"] == 0
            assert values[f"{backend}_geometric_mean_speedup_vs_fastest_external"] is None
    markdown = comparison.comparison_markdown(summary)
    assert "外部基线覆盖不完整" in markdown
    assert "0 / 144" in markdown and "0 / 168" in markdown
    assert "冻结集合完整：是" not in markdown


def test_cli_hashes_the_same_input_snapshot_it_compares(tmp_path, monkeypatch):
    import hashlib
    comparison = _comparison_module()
    paths, snapshots = {}, {}
    for backend, us in (("triton", 10), ("tilelang", 5)):
        paths[backend] = tmp_path / f"{backend}.json"
        snapshots[backend] = json.dumps(_report(backend, us)).encode("utf-8")
        paths[backend].write_bytes(snapshots[backend])
    compare = comparison.compare_reports

    def update_checkpoint_after_comparison(triton, tilelang):
        result = compare(triton, tilelang)
        paths["tilelang"].write_text(json.dumps(_report("tilelang", 100)))
        return result

    monkeypatch.setattr(comparison, "compare_reports", update_checkpoint_after_comparison)
    output = tmp_path / "evidence"
    assert comparison.main(["--triton", str(paths["triton"]), "--tilelang", str(paths["tilelang"]),
                            "--output-dir", str(output)]) == 0
    summary = json.loads((output / "comparison.json").read_text())
    assert summary["cases"][0]["tilelang_gpu_us"] == 5
    for backend in ("triton", "tilelang"):
        assert summary["sources"][backend]["sha256"] == hashlib.sha256(snapshots[backend]).hexdigest()
