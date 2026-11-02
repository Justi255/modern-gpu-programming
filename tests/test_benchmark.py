"""Statistics use literal timings, independently of any CUDA availability."""

import json
import math

import pytest

from fa5060.benchmark import (CorrectnessError, _flashinfer_provider, _graph_batch_size,
                              _sdpa_views, _validate_output,
                              atomic_write_json, evaluate_case, run_benchmark,
                              summarize_results, tuning_candidates)
from fa5060.workloads import AttentionCase, make_inputs


def _ok(gpu, eager=100):
    return {"status": "ok", "validated": True, "gpu_us_p50": gpu,
            "eager_us_p50": eager}


def test_flashinfer_workspace_covers_long_prefill_partial_outputs():
    from fa5060.benchmark import _flashinfer_workspace_bytes

    short = AttentionCase('prefill', 1, 512, 64, 32, 'fp16', 'dense')
    long = AttentionCase('prefill', 4, 8192, 128, 32, 'fp16', 'paged')
    assert _flashinfer_workspace_bytes(short) == 128 * 1024**2
    required = long.batch * long.query_len * long.heads_q * (long.head_dim + 1) * 4
    assert _flashinfer_workspace_bytes(long) > required
    assert _flashinfer_workspace_bytes(long) <= 1100 * 1024**2


def _record(case, ours, flash, fa4, infer):
    return {**case.to_dict(), "providers": {
        "ours": _ok(ours), "torch_flash": _ok(flash),
        "torch_cudnn": {"status": "unsupported", "reason": "not compiled"},
        "torch_efficient": {"status": "unsupported", "reason": "not compiled"},
        "fa4": _ok(fa4), "flashinfer_fa2": _ok(infer),
        "flashinfer_fmha_v2": {"status": "unsupported", "reason": "not compiled"},
    }}


def test_fastest_baseline_uses_validated_gpu_times_and_never_eager_or_missing():
    case = AttentionCase("prefill", 1, 512, 64, 32, "fp16", "dense")
    record = _record(case, 10, 30, 20, 40)
    record["providers"]["flashinfer_fmha_v2"] = {
        "status": "incorrect", "validated": False, "gpu_us_p50": 1,
        "eager_us_p50": 1,
    }
    result = evaluate_case(record)
    assert result["fastest_baseline"] == "fa4"
    assert result["speedup_vs_fastest"] == 2
    assert result["mandatory_baselines_covered"] is True
    assert result["correctness_passed"] is False


def test_phase_geometric_mean_and_worst_have_hand_checked_values():
    first = AttentionCase("prefill", 1, 512, 64, 32, "fp16", "dense")
    second = AttentionCase("prefill", 1, 512, 128, 32, "fp16", "dense")
    summary = summarize_results([
        _record(first, 10, 30, 20, 40),  # speedup = 2
        _record(second, 20, 20, 30, 40),  # speedup = 1
    ], [first, second])
    phase = summary["prefill"]
    assert phase["geometric_mean_speedup"] == pytest.approx(math.sqrt(2))
    assert phase["worst_speedup"] == 1
    assert phase["expected_cases"] == phase["completed_cases"] == 2
    assert phase["target_met"] is True
    assert summary["decode"]["target_met"] is False


@pytest.mark.parametrize("breakage", ["missing_case", "missing_provider", "missing_ours",
                                      "failed", "unvalidated", "nonfinite", "duplicate",
                                      "missing_optional_record", "changed_shape", "optional_error"])
def test_summary_fails_closed_when_evidence_is_incomplete(breakage):
    first = AttentionCase("prefill", 1, 512, 64, 32, "fp16", "dense")
    second = AttentionCase("prefill", 1, 512, 128, 32, "fp16", "dense")
    records = [_record(first, 10, 30, 20, 40), _record(second, 10, 30, 20, 40)]
    if breakage == "missing_case":
        records.pop()
    elif breakage == "missing_provider":
        del records[0]["providers"]["fa4"]
    elif breakage == "missing_ours":
        del records[0]["providers"]["ours"]
    elif breakage == "failed":
        records[0]["providers"]["ours"]["status"] = "error"
    elif breakage == "unvalidated":
        records[0]["providers"]["fa4"]["validated"] = False
    elif breakage == "nonfinite":
        records[0]["providers"]["fa4"]["gpu_us_p50"] = float("nan")
    elif breakage == "missing_optional_record":
        del records[0]["providers"]["torch_efficient"]
    elif breakage == "changed_shape":
        records[0]["shape"]["q"] = [1, 1, 32, 64]
    elif breakage == "optional_error":
        records[0]["providers"]["flashinfer_fmha_v2"] = {
            "status": "error", "reason": "unexpected native adapter failure"}
    else:
        records.append(records[0])
    summary = summarize_results(records, [first, second])
    assert summary["prefill"]["target_met"] is False
    assert summary["target_met"] is False
    assert summary["prefill"]["coverage_complete"] is False


def test_native_paged_baseline_is_required_and_dense_only_providers_do_not_substitute():
    case = AttentionCase("decode", 1, 512, 64, 8, "bf16", "paged")
    record = _record(case, 1, 100, 100, 2)
    record["providers"]["flashinfer_fa2"] = {"status": "unsupported", "reason": "missing"}
    result = evaluate_case(record)
    assert result["mandatory_baselines_covered"] is False
    assert result["speedup_vs_fastest"] is None
    assert result["fastest_baseline"] is None


def test_worst_case_threshold_and_smoke_scope_block_a_target_claim():
    first = AttentionCase("decode", 1, 512, 64, 32, "fp16", "dense")
    second = AttentionCase("decode", 1, 512, 128, 32, "fp16", "dense")
    records = [_record(first, 10, 40, 40, 40), _record(second, 10, 9, 20, 30)]
    summary = summarize_results(records, [first, second])
    assert summary["decode"]["geometric_mean_speedup"] == pytest.approx(math.sqrt(3.6))
    assert summary["decode"]["worst_speedup"] == .9
    assert summary["decode"]["target_met"] is False
    records[1] = _record(second, 10, 20, 30, 40)
    assert summarize_results(records, [first, second], suite="smoke")["decode"]["target_met"] is False


def test_atomic_json_preserves_last_complete_checkpoint_on_invalid_numbers(tmp_path):
    output = tmp_path / "nested" / "result.json"
    atomic_write_json(output, {"cases": [{"id": "first"}], "summary": {"target_met": False}})
    assert json.loads(output.read_text())["cases"] == [{"id": "first"}]
    with pytest.raises(ValueError):
        atomic_write_json(output, {"bad": float("nan")})
    assert json.loads(output.read_text())["cases"] == [{"id": "first"}]
    assert sorted(p.name for p in output.parent.iterdir()) == ["result.json"]


def test_case_checkpoint_survives_interruption_during_the_next_case(tmp_path):
    first = AttentionCase("prefill", 1, 512, 64, 32, "fp16", "dense")
    second = AttentionCase("prefill", 1, 512, 128, 32, "fp16", "dense")
    output = tmp_path / "result.json"

    def runner(case, *, rounds, tune):
        if case == second:
            raise KeyboardInterrupt
        assert rounds == 5 and tune is False
        return _record(case, 10, 30, 20, 40)

    with pytest.raises(KeyboardInterrupt):
        run_benchmark([first, second], output=output, environment={"test": True},
                      suite="full", case_runner=runner)
    saved = json.loads(output.read_text())
    assert saved["schema_version"] == 1
    assert saved["environment"] == {"test": True}
    assert saved["rounds"] == 5
    assert [case["id"] for case in saved["cases"]] == [first.case_id]
    assert saved["summary"]["prefill"]["missing_cases"] == [second.case_id]
    assert saved["summary"]["target_met"] is False


def test_tuning_grid_records_explicit_configs_without_changing_the_workload():
    dense = AttentionCase("prefill", 1, 512, 64, 8, "fp16", "dense")
    paged = AttentionCase("prefill", 1, 512, 64, 8, "fp16", "paged")
    decode = AttentionCase("decode", 1, 512, 64, 8, "fp16", "paged")
    assert tuning_candidates(dense)[0] == {"config": None, "num_splits": None}
    assert any(c["config"] and c["config"].get("loader") == "tma"
               for c in tuning_candidates(dense))
    assert all(not c["config"] or c["config"].get("loader") != "tma"
               for c in tuning_candidates(paged))
    assert {c["num_splits"] for c in tuning_candidates(decode)} >= {None, 1, 4, 16}
    assert dense.to_dict()["shape"]["q"] == [1, 512, 32, 64]


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_torch_sdpa_inputs_preserve_native_gqa_storage_without_contiguous_copies(phase):
    torch = pytest.importorskip("torch")
    case = AttentionCase(phase, 2, 32, 32, 1, "fp16", "dense")
    inputs = make_inputs(case, device="cpu")
    q, k, v = _sdpa_views(case, inputs)
    assert q.shape == (2, 32, 32 if phase == "prefill" else 1, 32)
    assert k.shape == v.shape == (2, 1, 32, 32)
    assert q.data_ptr() == inputs.q.data_ptr()
    assert k.data_ptr() == inputs.k.data_ptr()
    assert v.data_ptr() == inputs.v.data_ptr()
    assert k.shape[1] == 1  # Expanding native GQA to 32 heads would change this.


def test_output_validation_records_errors_and_rejects_wrong_or_nonfinite_results():
    torch = pytest.importorskip("torch")
    reference = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    good = torch.tensor([0.001, 1.001, -0.999], dtype=torch.float16)
    metrics = _validate_output(good, reference, "fp16")
    assert metrics["validated"] is True
    assert metrics["max_abs_error"] < .002
    for wrong in (torch.tensor([0., 1., 4.]), torch.tensor([0., 1., float("nan")]),
                  torch.zeros((1, 3))):
        with pytest.raises(CorrectnessError):
            _validate_output(wrong, reference, "fp16")


def test_output_validation_rejects_zero_answers_when_decode_values_are_small():
    torch = pytest.importorskip("torch")
    reference = torch.tensor([.001, -.002, .001])
    for dtype in ("fp16", "bf16"):
        with pytest.raises(CorrectnessError):
            _validate_output(torch.zeros(3), reference, dtype)


@pytest.mark.parametrize("trial_us,expected", [(1, 2048), (10, 1000), (90, 128),
                                              (1000, 10), (20000, 1)])
def test_repeated_graph_batch_limits_host_gaps_and_caps_capture_cost(trial_us, expected):
    assert _graph_batch_size(trial_us) == expected


@pytest.mark.gpu
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("dtype,dim,heads", [("fp16", 64, 32), ("bf16", 128, 8)])
def test_flashinfer_dense_fmha_uses_a_working_sm120_native_layout(phase, dtype, dim, heads):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("targets the SM120 native-layout regression")
    pytest.importorskip("flashinfer")
    from fa5060.reference import attention_reference

    case = AttentionCase(phase, 1, 512, dim, heads, dtype, "dense")
    with torch.inference_mode():
        inputs = make_inputs(case)
        reference = attention_reference(inputs.q, inputs.k, inputs.v, causal=case.causal,
                                        kv_lens=inputs.kv_lens)
        provider = _flashinfer_provider(case, inputs, "fmha_v2")
        _validate_output(provider.graph_call(), reference, dtype)
        _validate_output(provider.eager_call(), reference, dtype)
        assert provider.metadata["packing_timed"] is True
