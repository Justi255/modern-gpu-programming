"""CPU checks for the independent, auditable FMHAv2 precision experiment."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "fmha_fp32acc.py"


def _script():
    assert SCRIPT.is_file(), "the independent precision-mode script is missing"
    spec = importlib.util.spec_from_file_location("fmha_fp32acc_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _installed_root():
    root = Path(__file__).parents[1] / ".venv-tilelang/lib/python3.13/site-packages/flashinfer"
    if not root.exists():
        pytest.skip("the pinned FlashInfer source package is unavailable")
    return root


def test_help_runs_without_site_packages_or_gpu_imports():
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert all(command in result.stdout for command in ("generate", "build", "validate"))


def test_runner_patch_changes_only_the_two_precision_assignments():
    module = _script()
    original = (_installed_root() / "data/csrc/fmha_v2_run.cu").read_text()
    patched = module.patch_runner(original)
    restored = patched.replace("Data_type acc_type = DATA_TYPE_FP32;", "Data_type acc_type =\n"
        "      (data_type == DATA_TYPE_BF16 || data_type == DATA_TYPE_E4M3) ? DATA_TYPE_FP32 : data_type;")
    restored = restored.replace("bool force_fp32_acc = true;",
        "bool force_fp32_acc = (data_type == DATA_TYPE_BF16 || data_type == DATA_TYPE_E4M3);")
    assert restored == original
    with pytest.raises(ValueError, match="precision assignments"):
        module.patch_runner(patched)


def test_generate_uses_real_generator_and_keeps_native_evidence(tmp_path):
    module = _script()
    before = set(sys.modules)
    root = _installed_root()
    destination = tmp_path / "independent"
    manifest = module.generate(destination, root)
    assert not {"torch", "flashinfer"} & (set(sys.modules) - before)
    assert manifest["gpu_validation"] == "not_run"
    assert set(manifest["layouts"]) == {"CONTIGUOUS_Q_KV", "Q_PAGED_KV_NHD"}
    runner = root / "data/csrc/fmha_v2_run.cu"
    assert manifest["native_sources"]["data/csrc/fmha_v2_run.cu"] == hashlib.sha256(runner.read_bytes()).hexdigest()
    assert (destination / "native/data/csrc/fmha_v2_run.cu").read_bytes() == runner.read_bytes()
    assert (destination / "runner.patch").read_text().count("\n+") == 3  # header + two assignments
    for layout, config in manifest["layouts"].items():
        assert config["uri"].startswith("fa5060_fmha_v2_fp32acc_")
        assert "trtllm_fmha_v2_" not in config["uri"]
        specs = config["specs"]
        assert [spec["head_size"] for spec in specs] == [64, 128]
        assert all(spec["dtype"] == "fp16_fp32" and spec["output_dtype"] == "fp16" for spec in specs)
        cu_sources = [destination / source for source in config["sources"]]
        assert len(cu_sources) == 4
        generated_kernels = cu_sources[:2]
        assert all("Ampere_hmma_fp32_traits" in path.read_text() for path in generated_kernels)
        assert all("Ampere_hmma_fp16_traits" not in path.read_text() for path in generated_kernels)
        registry = (destination / config["directory"] / "fmha_v2_api.h").read_text()
        assert "!force_fp32_acc" not in registry
        assert registry.count("&& force_fp32_acc") == 2
        assert "DATA_TYPE_FP16" in registry
    assert module.verify_manifest(destination) == manifest
    with pytest.raises(FileExistsError):
        module.generate(destination, root)


def test_manifest_rejects_changed_generated_source_before_build(tmp_path):
    module = _script()
    destination = tmp_path / "independent"
    manifest = module.generate(destination, _installed_root())
    source = destination / manifest["layouts"]["CONTIGUOUS_Q_KV"]["sources"][0]
    source.write_text(source.read_text() + "\n// altered\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.verify_manifest(destination)


def test_full_precision_selection_covers_both_phases_and_layouts():
    module = _script()
    cases = module.select_cases("full-fp16", None, None)
    assert len(cases) == 156
    assert {case.dtype for case in cases} == {"fp16"}
    assert {case.phase for case in cases} == {"prefill", "decode"}
    assert {case.layout for case in cases} == {"dense", "paged"}
    with pytest.raises(ValueError, match="FP16"):
        module.select_cases("full-fp16", ["prefill-b1-l512-hq32-hkv32-d64-bf16-dense"], None)


def test_native_failure_selection_preserves_frozen_case_identity():
    from fa5060.workloads import AttentionCase
    module = _script()
    failed = AttentionCase("prefill", 1, 512, 128, 32, "fp16", "dense")
    passed = AttentionCase("decode", 1, 512, 64, 1, "fp16", "paged")
    report = {"cases": [
        {**failed.to_dict(), "providers": {"flashinfer_fmha_v2": {"status": "incorrect"}}},
        {**passed.to_dict(), "providers": {"flashinfer_fmha_v2": {"status": "ok"}}}]}
    assert module.select_cases("native-failures", None, report) == [failed]


def test_diagnostics_retain_evaluator_failure_and_worst_coordinates():
    torch = pytest.importorskip("torch")
    module = _script()
    expected = torch.zeros((1, 2, 2, 4), dtype=torch.float16)
    actual = expected.clone()
    actual[0, 1, 1, 2] = 0.02
    result = module.output_diagnostics(actual, expected)
    assert result["status"] == "incorrect"
    assert result["elementwise_failure_count"] == 1
    assert result["worst_absolute_error_element"]["index"] == [0, 1, 1, 2]
    assert result["worst_absolute_error_element"]["query"] == 1
    assert result["worst_absolute_error_element"]["head"] == 1
    assert "reference mismatch" in result["validation_error"]
    assert module.output_diagnostics(expected, expected)["status"] == "ok"


@pytest.mark.parametrize("layout", ["dense", "paged"])
def test_corrected_adapter_preserves_native_layout_and_run_abi(layout):
    torch = pytest.importorskip("torch")
    from fa5060.workloads import AttentionCase, make_inputs
    module = _script()
    case = AttentionCase("prefill", 1, 4, 64, 1, "fp16", layout, heads_q=2)
    inputs = make_inputs(case, device="cpu")
    received = []

    # The real external C++ ABI is unavailable on CPU. This strict endpoint
    # checks the adapter boundary, without emulating attention or GPU execution.
    def run(q, k, v, out, workspace, workspace_bytes, block_table, page_size,
            lengths, cum_q, cum_kv, input_layout, max_q, max_kv, batch, mask,
            softmax_scale, bmm1_scale, bmm2_scale, window, chunk, alibi,
            softcap, skip, lse, sinks):
        assert q.dtype == out.dtype == torch.float16
        assert workspace_bytes == workspace.numel() * workspace.element_size()
        assert (max_q, max_kv, batch, mask) == (4, 4, 1, "causal")
        assert (softmax_scale, bmm1_scale, bmm2_scale) == (1.0, .125, 1.0)
        assert (window, chunk, alibi, softcap, skip, lse, sinks) == (-1, 0, False, 0.0, 0.0, None, None)
        assert lengths is inputs.kv_lens
        assert cum_q.tolist() == cum_kv.tolist() == [0, 4]
        if layout == "dense":
            assert input_layout == "contiguous_q_kv" and page_size == 0
            assert block_table is None
            assert v is k  # public native API uses the packed KV tensor twice
            assert torch.equal(k[:, 0], inputs.k.view(4, 1, 64))
            assert torch.equal(k[:, 1], inputs.v.view(4, 1, 64))
        else:
            assert input_layout == "q_paged_kv_nhd" and page_size == 16
            assert k is inputs.k and v is inputs.v
            assert torch.equal(block_table[:, 0], inputs.block_table)
            assert torch.equal(block_table[:, 1], inputs.block_table + 1)
        out.zero_()
        received.append(out)

    provider = module.corrected_provider(case, inputs, SimpleNamespace(run=run))
    first = provider.graph_call()
    second = provider.graph_call()
    eager = provider.eager_call()
    assert first.shape == second.shape == eager.shape == inputs.q.shape
    assert first.data_ptr() == second.data_ptr()
    assert eager.data_ptr() != first.data_ptr()
    assert len(received) == 3
