"""CPU-only contracts for reproducible environment and experiment records."""

import importlib
import json
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe():
    return importlib.import_module("fa5060.probe")


def _no_commands(command):
    return {"available": False, "output": None, "returncode": None, "error": "not installed"}


def test_import_and_help_do_not_import_torch():
    script = (
        "import sys; sys.modules['torch'] = None; "
        "import fa5060.probe, fa5060.experiments; "
        "fa5060.experiments.main(['--help'])"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "basics" in result.stdout and "attention" in result.stdout


def test_cpu_absence_is_explicit_json_and_does_not_dump_secrets(probe, monkeypatch):
    def missing_torch():
        raise ImportError("torch is absent")

    monkeypatch.setattr(probe, "_load_torch", missing_torch)
    monkeypatch.setattr(probe, "_command_output", _no_commands)
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-copy-this-secret")
    result = probe.collect_environment()
    encoded = json.dumps(result, allow_nan=False)
    assert result["schema_version"] == 1
    assert result["cuda_available"] is False
    assert result["cuda_build"] is None
    assert result["gpus"] == []
    assert result["gpu_unavailable_reason"]
    assert "python" in result and "platform" in result and "packages" in result
    assert datetime.fromisoformat(result["timestamp_utc"]).utcoffset().total_seconds() == 0
    assert "do-not-copy-this-secret" not in encoded
    assert "OPENAI_API_KEY" not in encoded


def test_runtime_properties_preserve_observed_values_and_missing_fields(probe, monkeypatch):
    properties = SimpleNamespace(
        name="Observed test GPU", major=12, minor=0, multi_processor_count=26,
        total_memory=8 * 2**30, L2_cache_size=3 * 2**20,
        shared_memory_per_block=48 * 1024,
        shared_memory_per_block_optin=99 * 1024,
        shared_memory_per_multiprocessor=100 * 1024,
    )
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                             get_device_properties=lambda index: properties),
    )
    monkeypatch.setattr(probe, "_load_torch", lambda: fake_torch)
    monkeypatch.setattr(probe, "_command_output", _no_commands)
    result = probe.collect_environment()
    gpu = result["gpus"][0]
    assert result["cuda_available"] is True
    assert result["cuda_build"] == "13.0"
    assert result["gpu_unavailable_reason"] is None
    assert gpu["name"] == "Observed test GPU"
    assert gpu["compute_capability"] == [12, 0]
    assert gpu["sm_count"] == 26
    assert gpu["total_memory_bytes"] == 8 * 2**30
    assert gpu["l2_cache_bytes"] == 3 * 2**20
    assert gpu["shared_memory_per_block_optin_bytes"] == 99 * 1024
    assert gpu["warp_size"] is None
    json.dumps(result, allow_nan=False)


def test_runtime_failure_is_reported_without_fabricating_gpu(probe, monkeypatch):
    def fail():
        raise RuntimeError("CUDA driver unavailable")

    fake_torch = SimpleNamespace(version=SimpleNamespace(cuda="13.0"),
                                 cuda=SimpleNamespace(is_available=fail))
    monkeypatch.setattr(probe, "_load_torch", lambda: fake_torch)
    monkeypatch.setattr(probe, "_command_output", _no_commands)
    result = probe.collect_environment()
    assert result["cuda_available"] is False
    assert result["gpus"] == []
    assert "driver unavailable" in result["gpu_unavailable_reason"]
    assert result["cuda_build"] == "13.0"


def test_optional_smi_metrics_parse_na_as_null(probe, monkeypatch):
    fake_torch = SimpleNamespace(version=SimpleNamespace(cuda=None),
                                 cuda=SimpleNamespace(is_available=lambda: False))

    def command_output(command):
        if command[0] == "nvidia-smi":
            return {"available": True, "returncode": 0, "error": None,
                    "output": "0, Test GPU, 595.99.02, P8, 7.5, [N/A], 210, 405, 42\n"}
        return _no_commands(command)

    monkeypatch.setattr(probe, "_load_torch", lambda: fake_torch)
    monkeypatch.setattr(probe, "_command_output", command_output)
    gpu = probe.collect_environment()["nvidia_smi"]["devices"][0]
    assert gpu["driver_version"] == "595.99.02"
    assert gpu["power_draw_watts"] == 7.5
    assert gpu["power_limit_watts"] is None
    assert gpu["temperature_celsius"] == 42


def test_cli_creates_parent_and_writes_exact_json(probe, monkeypatch, tmp_path, capsys):
    expected = {"schema_version": 1, "cuda_available": False, "gpus": []}
    monkeypatch.setattr(probe, "collect_environment", lambda: expected)
    destination = tmp_path / "nested" / "environment.json"
    assert probe.main(["--output", str(destination)]) == 0
    assert json.loads(destination.read_text()) == expected
    assert destination.read_text().endswith("\n")
    assert str(destination) in capsys.readouterr().out


def test_experiments_require_cuda_before_creating_results(monkeypatch, tmp_path, capsys):
    experiments = importlib.import_module("fa5060.experiments")

    def unavailable():
        raise RuntimeError("CUDA GPU is required for experiments")

    monkeypatch.setattr(experiments, "_require_cuda", unavailable)
    destination = tmp_path / "should-not-exist.json"
    assert experiments.main(["basics", "--output", str(destination)]) == 2
    assert not destination.exists()
    assert "CUDA" in capsys.readouterr().err


def test_bad_numerical_result_is_rejected_before_timing(monkeypatch):
    torch = pytest.importorskip("torch")
    experiments = importlib.import_module("fa5060.experiments")
    timing_calls = []
    monkeypatch.setattr(experiments, "_time_cuda", lambda *args, **kwargs: timing_calls.append(True))
    with pytest.raises(AssertionError):
        experiments._record(
            torch, {"packages": {}}, name="deliberately-wrong", inputs={},
            operation=lambda: torch.tensor([1.0]), expected=torch.tensor([0.0]),
            rtol=0, atol=0, reference="CPU test",
        )
    assert timing_calls == []


def test_nonfinite_numerical_result_is_rejected(monkeypatch):
    torch = pytest.importorskip("torch")
    experiments = importlib.import_module("fa5060.experiments")
    with pytest.raises(AssertionError, match="NaN or Inf"):
        experiments._check(torch, torch.tensor([float("nan")]), torch.tensor([0.0]), rtol=0, atol=0)
