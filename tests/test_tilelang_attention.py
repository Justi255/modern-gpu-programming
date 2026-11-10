"""TileLang must obey the same attention contracts as the Triton backend."""
import importlib

import pytest

torch = pytest.importorskip("torch")


def test_cpu_inputs_rejected_without_importing_compiler():
    api = importlib.import_module("fa5060.tilelang")
    q = torch.zeros(1, 3, 2, 32, dtype=torch.float16)
    with pytest.raises(ValueError, match="CUDA"):
        api.prefill(q, q, q)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_prefill_prefix_tail_and_masked_rows(dtype, causal):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    from fa5060.tilelang import prefill
    from fa5060.reference import attention_reference

    q = torch.randn(2, 37, 4, 64, device="cuda", dtype=dtype)
    k = torch.randn(2, 194, 2, 64, device="cuda", dtype=dtype)[:, ::2]
    v = torch.randn_like(k)
    lengths = torch.tensor([97, 13], device="cuda", dtype=torch.int32)
    expected = attention_reference(q, k, v, causal=causal, kv_lens=lengths)
    tol = .003 if dtype == torch.float16 else .02
    torch.testing.assert_close(prefill(q, k, v, causal=causal, kv_lens=lengths),
                               expected, atol=tol, rtol=tol)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["vector", "grouped"])
def test_decode_paged_split_empty_and_graph(mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    from fa5060.tilelang import decode, decode_workspace
    from fa5060.reference import attention_reference

    q = torch.randn(2, 4, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(8, 16, 2, 64, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    table = torch.tensor([[5, 1, 3], [2, 6, -1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([35, 0], device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    config = {"mode": mode}
    scratch = decode_workspace(q, k, block_table=table, num_splits=4, config=config)
    def call():
        return decode(q, k, v, kv_lens=lengths, block_table=table, out=out,
                      workspace=scratch, num_splits=4, config=config)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    lengths.copy_(torch.tensor([7, 21], device="cuda", dtype=torch.int32))
    table[0, 0] = 4
    graph.replay()
    torch.testing.assert_close(out, attention_reference(q, k, v, kv_lens=lengths,
                              block_table=table), atol=.003, rtol=.003)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("capacity", [0, 97])
def test_dense_tma_variable_lengths_graph_and_empty(dtype, d, capacity):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    from fa5060.tilelang import prefill
    from fa5060.reference import attention_reference

    q = torch.randn(2, 37, 4, d, device="cuda", dtype=dtype)
    k = torch.randn(2, capacity, 2, d, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    lengths = torch.tensor([min(83, capacity), min(13, capacity)], device="cuda", dtype=torch.int32)
    k[0, min(83, capacity):] = float("nan")
    v[0, min(83, capacity):] = float("nan")
    k[1, min(13, capacity):] = float("nan")
    v[1, min(13, capacity):] = float("nan")
    out = torch.empty((2, 74, 4, d), device="cuda", dtype=dtype)[:, ::2]
    config = {"loader": "tma"}
    prefill(q, k, v, kv_lens=lengths, out=out, config=config)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prefill(q, k, v, kv_lens=lengths, out=out, config=config)
    lengths[0] = min(7, capacity)
    graph.replay()
    tol = .003 if dtype == torch.float16 else .02
    torch.testing.assert_close(out, attention_reference(q, k, v, kv_lens=lengths),
                               atol=tol, rtol=tol)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stages,causal", [(2, True), (3, True), (4, True), (3, False)])
def test_dense_tma_long_pipeline_barrier_reuse_and_graph_lengths(dtype, stages, causal):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    from fa5060.tilelang import prefill
    from fa5060.reference import attention_reference

    torch.manual_seed(4100 + stages)
    capacity, d = 513, 64
    q = torch.randn(2, 37, 4, d, device="cuda", dtype=dtype)
    k = torch.randn(2, capacity, 2, d, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    batch = torch.arange(2, device="cuda", dtype=torch.float32)[:, None, None, None]
    q_head = torch.arange(4, device="cuda", dtype=torch.float32)[None, None, :, None]
    kv_head = torch.arange(2, device="cuda", dtype=torch.float32)[None, None, :, None]
    token = torch.arange(capacity, device="cuda", dtype=torch.float32)[None, :, None, None] / capacity
    q.add_(.1 * batch + .05 * q_head)
    k.mul_(.5).add_(.2 * batch + .1 * kv_head + .25 * token)
    v.mul_(.25).add_(3 * batch + kv_head + token)
    lengths = torch.tensor([513, 449], device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    config = {"loader": "tma", "block_m": 32, "block_n": 64, "num_stages": stages}

    def call():
        return prefill(q, k, v, causal=causal, kv_lens=lengths, out=out, config=config)

    def check():
        tol = .003 if dtype == torch.float16 else .02
        torch.testing.assert_close(out, attention_reference(q, k, v, causal=causal,
                                   kv_lens=lengths), atol=tol, rtol=tol)

    # Nine KV iterations reuse every 2/3/4-stage mbarrier slot across phases.
    # Distinct batches, heads and token values expose incorrect TMA regions
    # and stale tiles when a barrier ring is reused.
    call()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for live_lengths in ([193, 511], [0, 321], [513, 449]):
        lengths.copy_(torch.tensor(live_lengths, device="cuda", dtype=torch.int32))
        graph.replay()
        check()


@pytest.mark.gpu
def test_small_prefill_against_float64():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    from fa5060.tilelang import prefill
    q = torch.randn(1, 7, 4, 32, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 13, 2, 32, dtype=q.dtype, device="cuda")
    v = torch.randn_like(k)
    q64 = q.double().transpose(1, 2)
    k64 = k.double().repeat_interleave(2, dim=2).transpose(1, 2)
    v64 = v.double().repeat_interleave(2, dim=2).transpose(1, 2)
    logits = q64 @ k64.transpose(-1, -2) / (32 ** .5)
    mask = torch.arange(13, device="cuda")[None, :] <= torch.arange(7, device="cuda")[:, None] + 6
    expected = (logits.masked_fill(~mask, -float("inf")).softmax(-1) @ v64).transpose(1, 2)
    torch.testing.assert_close(prefill(q, k, v).double(), expected, atol=.003, rtol=.003)


@pytest.mark.gpu
@pytest.mark.parametrize("d", [32, 128])
@pytest.mark.parametrize("mode", ["vector", "grouped"])
def test_decode_dimensions_and_custom_blocks(d, mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    from fa5060.tilelang import decode
    from fa5060.reference import attention_reference
    q = torch.randn(2, 8, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(2, 193, 1, d, dtype=q.dtype, device="cuda")
    v = torch.randn_like(k)
    lengths = torch.tensor([193, 17], device="cuda", dtype=torch.int32)
    out = torch.empty((2, 16, d), dtype=q.dtype, device="cuda")[:, ::2]
    result = decode(q, k, v, kv_lens=lengths, num_splits=4, out=out,
                    config={"mode": mode, "block_n": 32, "num_stages": 1})
    torch.testing.assert_close(result, attention_reference(q, k, v, kv_lens=lengths), atol=.02, rtol=.02)
