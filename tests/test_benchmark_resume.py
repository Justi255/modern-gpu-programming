"""Resume evidence contracts run entirely with CPU fixtures and injected runners."""

import copy
import hashlib
import json

import pytest

from fa5060 import benchmark
from fa5060.workloads import smoke_suite


def _environment():
    return {
        "timestamp_utc": "first segment", "cuda_build": "13.0", "cuda_available": True,
        "python": {"version": "3.13", "implementation": "CPython", "compiler": "GCC"},
        "platform": {"system": "Linux", "release": "test", "machine": "x86_64"},
        "packages": {"torch": "2.12.1", "triton": "3.7.1", "tilelang": "0.1.15",
                     "apache-tvm-ffi": "0.1.12"},
        "gpus": [{"index": 0, "name": "fixture", "compute_capability": [12, 0],
                  "sm_count": 26, "total_memory_bytes": 8000000000}],
        "nvidia_smi": {"devices": [{"index": 0, "driver_version": "595.99",
                                    "temperature_celsius": 40, "sm_clock_mhz": 1000}]},
        "compiler": {"nvcc": {"output": "13.1"}, "ptxas": {"output": "13.1"}},
    }


def _row(case, *, backend="tilelang"):
    return {**case.to_dict(), "seed": 20261102, "backend": backend, "providers": {
        name: {"status": "ok", "validated": True, "gpu_us_p50": 10 if name == "ours" else 20}
        for name in benchmark.PROVIDER_NAMES}, "extra_evidence": {"keep": [1, 2, 3]}}


@pytest.fixture
def checkpoint(tmp_path):
    cases = smoke_suite()
    source = tmp_path / "original.json"

    def runner(case, **kwargs):
        if case == cases[2]:
            raise KeyboardInterrupt
        return _row(case)

    with pytest.raises(KeyboardInterrupt):
        benchmark.run_benchmark(cases, output=source, environment=_environment(),
                                rounds=5, tune=True, backend="tilelang", case_runner=runner)
    # Nonstandard whitespace makes an accidental load-and-rewrite observable.
    report = json.loads(source.read_bytes())
    source.write_bytes((json.dumps(report, indent=1) + "\n\n").encode())
    return source, cases


def _resume(source, cases, output, *, environment=None, runner=None, **kwargs):
    return benchmark.run_benchmark(
        cases, output=output, environment=_environment() if environment is None else environment,
        rounds=5, tune=True, backend="tilelang", resume_from=source,
        case_runner=_row if runner is None else runner, **kwargs)


def test_resume_preserves_raw_source_and_rows_and_only_executes_remaining(checkpoint, tmp_path):
    source, cases = checkpoint
    snapshot = source.read_bytes()
    original = json.loads(snapshot)
    fresh = copy.deepcopy(_environment())
    fresh["timestamp_utc"] = "resumed segment"
    fresh["python"]["executable"] = "/different/path/python"
    fresh["nvidia_smi"]["devices"][0].update(temperature_celsius=70, sm_clock_mhz=2000)
    calls = []

    def runner(case, *, rounds, tune, backend):
        assert rounds == 5 and tune is True and backend == "tilelang"
        calls.append(case.case_id)
        return _row(case)

    output = tmp_path / "completed.json"
    report = _resume(source, cases, output, environment=fresh, runner=runner)
    assert source.read_bytes() == snapshot
    assert report == json.loads(output.read_bytes())
    assert report["cases"][:2] == original["cases"]
    assert calls == [case.case_id for case in cases[2:]]
    assert report["environment"] == original["environment"]
    assert report["resume"]["source_sha256"] == hashlib.sha256(snapshot).hexdigest()
    assert report["resume"]["source_path"] == str(source)
    assert report["resume"]["initial_case_count"] == 2
    assert report["resume"]["environment"] == fresh
    assert report["resume"]["segmented"] is True
    assert len(report["cases"]) == 8


@pytest.mark.parametrize("change", [
    "schema_version", "backend", "suite", "rounds", "tune", "seed", "timing", "mode",
    "duplicate", "order", "shape", "missing_shape", "case_seed", "case_backend",
    "missing_provider", "invalid_provider", "manifest", "package", "driver", "gpu",
    "missing_environment", "no_cuda", "timing_type", "shape_type", "case_seed_type",
])
def test_resume_rejects_incompatible_evidence_before_writing_or_executing(checkpoint, tmp_path, change):
    source, cases = checkpoint
    report = json.loads(source.read_bytes())
    if change in ("schema_version", "rounds", "seed"):
        report[change] += 1
    elif change in ("backend", "suite"):
        report[change] = "triton" if change == "backend" else "full"
    elif change == "tune":
        report["tune"] = False
    elif change == "timing":
        report["timing"]["round_target_ms"] = 20
    elif change == "mode":
        report["configuration_mode"] = "public_default"
    elif change == "duplicate":
        report["cases"][1] = copy.deepcopy(report["cases"][0])
    elif change == "order":
        report["cases"].reverse()
    elif change == "shape":
        report["cases"][0]["shape"]["q"][1] = 1
    elif change == "missing_shape":
        del report["cases"][0]["shape"]
    elif change == "case_seed":
        report["cases"][0]["seed"] += 1
    elif change == "case_backend":
        report["cases"][0]["backend"] = "triton"
    elif change == "missing_provider":
        del report["cases"][0]["providers"]["ours"]
    elif change == "invalid_provider":
        report["cases"][0]["providers"]["ours"]["status"] = "unknown"
        report["summary"] = benchmark.summarize_results(report["cases"], cases, suite="smoke")
    elif change == "manifest":
        report["summary"]["prefill"]["expected_cases"] -= 1
    elif change == "package":
        report["environment"]["packages"]["torch"] = "other"
    elif change == "driver":
        report["environment"]["nvidia_smi"]["devices"][0]["driver_version"] = "other"
    elif change == "gpu":
        report["environment"]["gpus"][0]["sm_count"] = 1
    elif change == "missing_environment":
        del report["environment"]["compiler"]
    elif change == "no_cuda":
        report["environment"]["cuda_available"] = False
    elif change == "timing_type":
        report["timing"]["warmup_calls"] = 3.0
    elif change == "shape_type":
        report["cases"][0]["shape"]["q"][0] = True
    elif change == "case_seed_type":
        report["cases"][0]["seed"] = 20261102.0
    source.write_text(json.dumps(report))
    snapshot = source.read_bytes()
    output = tmp_path / "rejected.json"

    def forbidden(case, **kwargs):
        pytest.fail("incompatible checkpoint executed a case")

    with pytest.raises(ValueError):
        _resume(source, cases, output, runner=forbidden)
    assert source.read_bytes() == snapshot
    assert not output.exists()


@pytest.mark.parametrize("destination", ["source", "existing", "symlink", "hardlink"])
def test_resume_requires_new_output_and_preserves_colliding_files(checkpoint, tmp_path, destination):
    source, cases = checkpoint
    snapshot = source.read_bytes()
    output = source if destination == "source" else tmp_path / "occupied.json"
    if destination == "existing":
        output.write_bytes(b"untouched output")
    elif destination == "symlink":
        output.symlink_to(source)
    elif destination == "hardlink":
        output.hardlink_to(source)
    output_snapshot = output.read_bytes()
    with pytest.raises(ValueError, match="output"):
        _resume(source, cases, output)
    assert source.read_bytes() == snapshot
    assert output.read_bytes() == output_snapshot


def test_resume_rejects_complete_checkpoint_and_nonfrozen_manifest(checkpoint, tmp_path):
    source, cases = checkpoint
    with pytest.raises(ValueError, match="frozen"):
        _resume(source, tuple(reversed(cases)), tmp_path / "reordered.json")
    report = json.loads(source.read_bytes())
    report["cases"] = [_row(case) for case in cases]
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="complete"):
        _resume(source, cases, tmp_path / "complete.json")
    assert not (tmp_path / "reordered.json").exists()
    assert not (tmp_path / "complete.json").exists()


def test_resumed_checkpoint_still_survives_next_case_interruption(checkpoint, tmp_path):
    source, cases = checkpoint
    snapshot = source.read_bytes()
    output = tmp_path / "interrupted.json"

    def runner(case, **kwargs):
        if case == cases[3]:
            raise KeyboardInterrupt
        return _row(case)

    with pytest.raises(KeyboardInterrupt):
        _resume(source, cases, output, runner=runner)
    saved = json.loads(output.read_bytes())
    assert len(saved["cases"]) == 3
    assert saved["resume"]["initial_case_count"] == 2
    assert saved["cases"][:2] == json.loads(snapshot)["cases"]
    assert source.read_bytes() == snapshot

    next_output = tmp_path / "after-second-interruption.json"
    finished = _resume(output, cases, next_output,
                       runner=lambda case, **kwargs: _row(case))
    assert len(finished["cases"]) == 8
    assert finished["cases"][:3] == saved["cases"]
    assert finished["resume"]["initial_case_count"] == 3
    assert finished["resume"]["previous"] == saved["resume"]
    assert source.read_bytes() == snapshot
    assert json.loads(output.read_bytes()) == saved


def test_resume_cli_continues_checkpoint(checkpoint, tmp_path, monkeypatch, capsys):
    from fa5060 import probe

    source, _ = checkpoint
    monkeypatch.setattr(probe, "collect_environment", _environment)
    monkeypatch.setattr(benchmark, "benchmark_case", lambda case, **kwargs: _row(case))
    output = tmp_path / "cli.json"
    assert benchmark.main(["--backend", "tilelang", "--tune", "--resume-from", str(source),
                           "--output", str(output)]) == 0
    assert len(json.loads(output.read_bytes())["cases"]) == 8
    assert "[8/8]" in capsys.readouterr().err
