"""TMA loader contracts plus GPU numerical checks (root runs the latter)."""

from importlib import import_module

import pytest

torch = pytest.importorskip("torch")


def _config(**updates):
    result = {"block_m": 32, "block_n": 64, "num_warps": 4,
              "num_stages": 2, "loader": "tma", "mode": "auto"}
    result.update(updates)
    return result


def _cpu_inputs(d=64):
    q = torch.empty((2, 37, 4, d), dtype=torch.float16)
    k = torch.empty((2, 97, 2, d), dtype=q.dtype)
    return q, k, torch.empty_like(k), torch.empty_like(q)


def test_tma_requires_cuda_without_compute_fallback():
    tma = import_module("fa5060.attention_tma")
    q, k, v, out = _cpu_inputs()
    with pytest.raises(ValueError, match="CUDA"):
        tma.prefill_tma(q, k, v, out, True, None, None, None, _config())


def test_tma_explicitly_rejects_paged_storage():
    tma = import_module("fa5060.attention_tma")
    q, k, v, out = _cpu_inputs()
    with pytest.raises(ValueError, match="dense|paged"):
        tma.prefill_tma(q, k, v, out, True, None, None, torch.zeros(2, 3, dtype=torch.int32), _config())


@pytest.mark.parametrize("name", ["q", "k", "v"])
def test_tma_rejects_noncontiguous_bshd_before_cuda(name):
    tma = import_module("fa5060.attention_tma")
    q, k, v, out = _cpu_inputs()
    values = {"q": q, "k": k, "v": v}
    x = values[name]
    values[name] = torch.empty((x.shape[0], x.shape[1] * 2, *x.shape[2:]), dtype=x.dtype)[:, ::2]
    with pytest.raises(ValueError, match="contiguous"):
        tma.prefill_tma(values["q"], values["k"], values["v"], out, True, None, None, None, _config())


def test_tma_rejects_misaligned_contiguous_view():
    tma = import_module("fa5060.attention_tma")
    q, k, v, out = _cpu_inputs()
    q = torch.empty(q.numel() + 1, dtype=q.dtype)[1:].view(q.shape)
    assert q.is_contiguous()
    with pytest.raises(ValueError, match="16-byte"):
        tma.prefill_tma(q, k, v, out, True, None, None, None, _config())


@pytest.mark.parametrize("d", [32, 96])
def test_tma_rejects_head_dimensions_outside_64_128(d):
    tma = import_module("fa5060.attention_tma")
    q, k, v, out = _cpu_inputs(d)
    with pytest.raises(ValueError, match="64|128"):
        tma.prefill_tma(q, k, v, out, True, None, None, None, _config())


@pytest.mark.parametrize("updates", [
    {"block_m": 16}, {"block_n": 256}, {"num_warps": 3},
    {"num_stages": 0}, {"num_stages": 5}, {"block_m": True},
    {"block_m": 128, "block_n": 128, "num_stages": 4},
])
def test_tma_rejects_illegal_config_and_memory_budget_before_cuda(updates):
    tma = import_module("fa5060.attention_tma")
    q, k, v, out = _cpu_inputs(128)
    with pytest.raises(ValueError, match="config|memory|budget"):
        tma.prefill_tma(q, k, v, out, True, None, None, None, _config(**updates))


def test_tma_rejects_nonfinite_scale_before_cuda():
    tma = import_module("fa5060.attention_tma")
    q, k, v, out = _cpu_inputs()
    with pytest.raises(ValueError, match="finite"):
        tma.prefill_tma(q, k, v, out, True, float("nan"), None, None, _config())


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("a CUDA GPU is required")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("this optional loader is validated for SM120")
    return torch.device("cuda")


def test_output_rejects_internal_overlap_before_launch():
    from fa5060.attention import _output

    q = torch.empty((2, 7, 4, 64), dtype=torch.float16)
    k, v = torch.empty_like(q), torch.empty_like(q)
    out = torch.empty((2, 1, 4, 64), dtype=q.dtype).expand_as(q)
    with pytest.raises(ValueError, match="overlap|layout"):
        _output(q, k, v, out)
    from fa5060.attention_tma import prefill_tma
    with pytest.raises(ValueError, match="overlap|layout"):
        prefill_tma(q, k, v, out, True, None, None, None, _config())


def test_output_accepts_nonoverlapping_outer_strides():
    from fa5060.attention import _output

    q = torch.empty((2, 7, 4, 64), dtype=torch.float16)
    k, v = torch.empty_like(q), torch.empty_like(q)
    out = torch.empty((7, 4, 4, 64), dtype=q.dtype).transpose(0, 1)[::2]
    assert out.shape == q.shape
    assert _output(q, k, v, out) is out


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("hq,hkv", [(4, 4), (4, 2), (4, 1)])
def test_tma_prefill_right_alignment_variable_lengths_and_zero(cuda_device, dtype, d, causal, hq, hkv):
    from fa5060.attention_tma import prefill_tma
    from fa5060.reference import attention_reference

    torch.manual_seed(27)
    q = torch.randn((3, 37, hq, d), device=cuda_device, dtype=dtype)
    k = torch.randn((3, 97, hkv, d), device=cuda_device, dtype=dtype)
    v = torch.randn_like(k)
    # Invalid cache values must not contaminate zero-probability contributions.
    k[1, 13:] = float("nan")
    v[1, 13:] = float("nan")
    k[2] = float("nan")
    v[2] = float("nan")
    lengths = torch.tensor([97, 13, 0], device=cuda_device, dtype=torch.int32)
    out = torch.full_like(q, float("nan"))
    actual = prefill_tma(q, k, v, out, causal, None, lengths, None, _config())
    assert actual is out
    assert bool(torch.isfinite(actual).all())
    expected = attention_reference(q, k, v, causal=causal, kv_lens=lengths)
    tol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    assert torch.count_nonzero(actual[2]) == 0


@pytest.mark.gpu
@pytest.mark.parametrize("block_m,block_n,stages", [(32, 32, 1), (64, 64, 2), (128, 32, 4), (32, 128, 2)])
def test_tma_tiles_full_capacity_custom_scale_and_strided_output(cuda_device, block_m, block_n, stages):
    from fa5060.attention_tma import prefill_tma
    from fa5060.reference import attention_reference

    q = torch.randn((2, 71, 4, 64), device=cuda_device, dtype=torch.float16)
    k = torch.randn((2, 133, 2, 64), device=cuda_device, dtype=q.dtype)
    v = torch.randn_like(k)
    out = torch.empty((2, 142, 4, 64), device=cuda_device, dtype=q.dtype)[:, ::2]
    config = _config(block_m=block_m, block_n=block_n, num_stages=stages)
    actual = prefill_tma(q, k, v, out, True, .17, None, None, config)
    torch.testing.assert_close(actual, attention_reference(q, k, v, scale=.17), atol=3e-3, rtol=3e-3)


@pytest.mark.gpu
@pytest.mark.parametrize("capacity", [0, 17])
def test_tma_empty_capacity_or_all_masked_prefix_is_zero(cuda_device, capacity):
    from fa5060.attention_tma import prefill_tma

    q = torch.randn((2, 37, 4, 64), device=cuda_device, dtype=torch.float16)
    k = torch.randn((2, capacity, 2, 64), device=cuda_device, dtype=q.dtype)
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    lengths = torch.zeros(2, device=cuda_device, dtype=torch.int32)
    prefill_tma(q, k, v, out, True, None, lengths, None, _config())
    assert torch.count_nonzero(out) == 0


@pytest.mark.gpu
def test_tma_repeated_calls_read_changed_gpu_lengths(cuda_device):
    from fa5060.attention_tma import prefill_tma
    from fa5060.reference import attention_reference

    q = torch.randn((2, 37, 4, 64), device=cuda_device, dtype=torch.float16)
    k = torch.randn((2, 97, 2, 64), device=cuda_device, dtype=q.dtype)
    v = torch.randn_like(k)
    lengths = torch.full((2,), 97, device=cuda_device, dtype=torch.int32)
    out = torch.empty_like(q)
    prefill_tma(q, k, v, out, True, 0., lengths, None, _config())
    lengths.copy_(torch.tensor([7, 0], device=cuda_device, dtype=torch.int32))
    prefill_tma(q, k, v, out, True, 0., lengths, None, _config())
    torch.testing.assert_close(out, attention_reference(q, k, v, kv_lens=lengths, scale=0.), atol=3e-3, rtol=3e-3)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_public_prefill_dispatches_to_tma_loader(cuda_device, dtype):
    from fa5060 import prefill
    from fa5060.reference import attention_reference

    q = torch.randn((2, 37, 4, 128), device=cuda_device, dtype=dtype)
    k = torch.randn((2, 97, 2, 128), device=cuda_device, dtype=dtype)
    v = torch.randn_like(k)
    lengths = torch.tensor([97, 0], device=cuda_device, dtype=torch.int32)
    out = torch.empty_like(q)
    actual = prefill(q, k, v, kv_lens=lengths, out=out, config={"loader": "tma"})
    assert actual is out
    expected = attention_reference(q, k, v, kv_lens=lengths)
    tol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tma_graph_replay_reads_changed_lengths(cuda_device, dtype):
    from fa5060 import prefill
    from fa5060.reference import attention_reference

    q = torch.randn((2, 37, 4, 64), device=cuda_device, dtype=dtype)
    k = torch.randn((2, 97, 2, 64), device=cuda_device, dtype=dtype)
    v = torch.randn_like(k)
    lengths = torch.tensor([97, 61], device=cuda_device, dtype=torch.int32)
    out = torch.empty_like(q)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            prefill(q, k, v, kv_lens=lengths, out=out, config={"loader": "tma"})
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        prefill(q, k, v, kv_lens=lengths, out=out, config={"loader": "tma"})
    lengths.copy_(torch.tensor([13, 0], device=cuda_device, dtype=torch.int32))
    graph.replay()
    expected = attention_reference(q, k, v, kv_lens=lengths)
    tol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)
    assert torch.count_nonzero(out[1]) == 0
