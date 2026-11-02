"""CPU checks for the benchmark's frozen workload and input semantics."""

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from fa5060.workloads import AttentionCase, full_suite, make_inputs, smoke_suite


def test_full_suite_keeps_all_312_preregistered_cases():
    cases = full_suite()
    assert len(cases) == len({case.case_id for case in cases}) == 312
    assert sum(case.phase == "prefill" for case in cases) == 144
    assert sum(case.phase == "decode" for case in cases) == 168
    assert {(c.batch, c.kv_len) for c in cases if c.phase == "prefill"} == {
        (1, 512), (1, 2048), (1, 8192), (4, 512), (4, 2048), (4, 8192)
    }
    assert {(c.batch, c.kv_len) for c in cases if c.phase == "decode"} == {
        (1, 512), (1, 8192), (1, 32768), (4, 2048), (4, 8192),
        (16, 512), (16, 2048)
    }
    assert {(c.head_dim, c.heads_kv, c.dtype, c.layout) for c in cases} == {
        (d, h, dtype, layout) for d in (64, 128) for h in (32, 8, 1)
        for dtype in ("fp16", "bf16") for layout in ("dense", "paged")
    }
    assert all(c.heads_q == 32 and c.page_size == 16 for c in cases)
    assert all(c.causal == (c.phase == "prefill") for c in cases)


def test_case_json_round_trip_is_immutable_and_has_explicit_shapes():
    case = AttentionCase("decode", 4, 2048, 128, 8, "bf16", "paged")
    encoded = json.loads(json.dumps(case.to_dict()))
    assert AttentionCase.from_dict(encoded) == case
    assert encoded["shape"] == {
        "q": [4, 32, 128], "k": [512, 16, 8, 128],
        "v": [512, 16, 8, 128], "kv_lens": [2048, 2048, 2048, 2048],
        "block_table": [4, 128], "heads_q": 32, "heads_kv": 8,
        "head_dim": 128, "batch": 4, "query_len": 1, "kv_len": 2048,
    }
    with pytest.raises(FrozenInstanceError):
        case.kv_len = 1
    with pytest.raises(ValueError):
        AttentionCase("decode", 1, 512, 64, 3, "fp16", "dense")
    with pytest.raises(ValueError):
        AttentionCase.from_dict({**encoded, "causal": True})


def test_smoke_is_a_stable_small_subset_covering_each_phase_dtype_and_layout():
    smoke = smoke_suite()
    assert len(smoke) == 8
    assert set(smoke) <= set(full_suite())
    assert {(c.phase, c.dtype, c.layout) for c in smoke} == {
        (phase, dtype, layout) for phase in ("prefill", "decode")
        for dtype in ("fp16", "bf16") for layout in ("dense", "paged")
    }
    assert smoke == smoke_suite()
    assert {case.heads_kv for case in smoke} == {32, 8, 1}


@pytest.mark.parametrize("layout", ["dense", "paged"])
def test_inputs_are_seeded_and_paged_storage_is_a_real_permutation(layout):
    torch = pytest.importorskip("torch")
    case = AttentionCase("prefill", 2, 32, 32, 1, "fp16", layout)
    first = make_inputs(case, device="cpu", seed=17)
    second = make_inputs(case, device="cpu", seed=17)
    assert torch.equal(first.q, second.q)
    assert torch.equal(first.k, second.k)
    assert torch.equal(first.v, second.v)
    assert first.q.shape == (2, 32, 32, 32)
    assert first.k.dtype == torch.float16
    assert first.kv_lens.tolist() == [32, 32]
    if layout == "paged":
        assert first.k.shape == (4, 16, 1, 32)
        assert first.k.is_contiguous() and first.v.is_contiguous()
        assert first.k.untyped_storage().data_ptr() == first.v.untyped_storage().data_ptr()
        assert first.block_table.dtype == torch.int32
        assert sorted(first.block_table.flatten().tolist()) == [0, 1, 2, 3]
        assert first.block_table.flatten().tolist() != [0, 1, 2, 3]
    else:
        assert first.k.shape == (2, 32, 1, 32)
        assert first.block_table is None
    assert not torch.equal(first.q, make_inputs(case, device="cpu", seed=18).q)
