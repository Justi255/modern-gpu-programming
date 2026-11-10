from dataclasses import replace
from importlib import import_module

import pytest
torch = pytest.importorskip('torch')

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU required')

@pytest.fixture(params=["triton", "tilelang"])
def api(request):
    return import_module("fa5060.attention" if request.param == "triton" else "fa5060.tilelang.attention")



def test_cpu_is_rejected_explicitly(api):
    prefill = api.prefill

    q = torch.zeros(1, 3, 2, 32, dtype=torch.float16)
    with pytest.raises(ValueError, match='CUDA'):
        prefill(q, q, q)


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('hkv', [4, 2, 1])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('d', [32, 64, 128])
def test_prefill_odd_tail_prefix_and_outer_stride(api, dtype, hkv, causal, d):
    prefill = api.prefill
    from fa5060.reference import attention_reference

    torch.manual_seed(11)
    q = torch.randn(2, 37, 4, d, device='cuda', dtype=dtype)
    k = torch.randn(2, 194, hkv, d, device='cuda', dtype=dtype)[:, ::2]
    v = torch.randn_like(k)
    lens = torch.tensor([97, 13], dtype=torch.int32, device='cuda')
    expected = attention_reference(q, k, v, causal=causal, kv_lens=lens)
    actual = prefill(q, k, v, causal=causal, kv_lens=lens)
    tol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('page', [16, 32, 64])
@pytest.mark.parametrize('splits', [1, 4])
@pytest.mark.parametrize('mode', ['vector', 'grouped'])
def test_decode_paged_tail_and_empty(api, dtype, page, splits, mode):
    decode = api.decode
    from fa5060.reference import attention_reference

    torch.manual_seed(12)
    q = torch.randn(3, 8, 64, device='cuda', dtype=dtype)
    k = torch.randn(12, page, 2, 64, device='cuda', dtype=dtype)
    v = torch.randn_like(k)
    table = torch.tensor([[6, 2, 8], [3, 0, -1], [-1, -1, -1]], dtype=torch.int32, device='cuda')
    lens = torch.tensor([2*page+3, page-1, 0], dtype=torch.int32, device='cuda')
    expected = attention_reference(q, k, v, kv_lens=lens, block_table=table)
    actual = decode(q, k, v, kv_lens=lens, block_table=table, num_splits=splits, config={'mode': mode})
    tol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


@GPU
@pytest.mark.gpu
def test_prefill_paged_right_alignment_and_scale(api):
    prefill = api.prefill
    from fa5060.reference import attention_reference

    torch.manual_seed(13)
    q = torch.randn(2, 23, 8, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(8, 16, 2, 64, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    table = torch.tensor([[4, 1, 6], [3, 5, -1]], device='cuda', dtype=torch.int32)
    lens = torch.tensor([35, 9], device='cuda', dtype=torch.int32)
    expected = attention_reference(q, k, v, kv_lens=lens, block_table=table, scale=.17)
    actual = prefill(q, k, v, kv_lens=lens, block_table=table, scale=.17)
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)


@GPU
@pytest.mark.gpu
def test_graph_replay_reads_changed_lengths_and_workspace(api):
    decode = api.decode
    decode_workspace = api.decode_workspace
    from fa5060.reference import attention_reference

    torch.manual_seed(14)
    q = torch.randn(2, 4, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(2, 193, 1, 64, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    lens = torch.tensor([193, 109], device='cuda', dtype=torch.int32)
    out = torch.empty_like(q)
    workspace = decode_workspace(q, k, num_splits=4)
    for _ in range(3):
        decode(q, k, v, kv_lens=lens, out=out, workspace=workspace, num_splits=4)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode(q, k, v, kv_lens=lens, out=out, workspace=workspace, num_splits=4)
    lens.copy_(torch.tensor([7, 0], device='cuda', dtype=torch.int32))
    graph.replay()
    torch.testing.assert_close(out, attention_reference(q,k,v,kv_lens=lens), atol=3e-3, rtol=3e-3)


@GPU
@pytest.mark.gpu
def test_large_logits_and_zero_scale_are_finite(api):
    prefill = api.prefill
    from fa5060.reference import attention_reference

    torch.manual_seed(15)
    q = torch.randn(1, 17, 2, 64, device='cuda', dtype=torch.float16)*40
    k = torch.randn(1, 43, 1, 64, device='cuda', dtype=q.dtype)*40
    v = torch.randn_like(k)
    for scale in (0., .125):
        actual = prefill(q, k, v, scale=scale)
        assert bool(actual.isfinite().all())
        torch.testing.assert_close(actual, attention_reference(q,k,v,scale=scale), atol=3e-3, rtol=3e-3)


@GPU
@pytest.mark.gpu
def test_decode_rejects_aliased_workspace_before_launch(api):
    q = torch.randn(1, 2, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 17, 1, 64, device='cuda', dtype=q.dtype)
    lens = torch.tensor([17], device='cuda', dtype=torch.int32)
    scratch = api.decode_workspace(q, k, num_splits=4)
    scratch = replace(scratch, sums=scratch.maxima)
    with pytest.raises(ValueError, match='overlap'):
        api.decode(q, k, k, kv_lens=lens, workspace=scratch, num_splits=4)

    # Half-precision output and FP32 scratch can alias the same allocation.
    scratch = api.decode_workspace(q, k, num_splits=4)
    out = scratch.partial.view(torch.float16).flatten()[:q.numel()].view_as(q)
    with pytest.raises(ValueError, match='overlap'):
        api.decode(q, k, k, kv_lens=lens, out=out, workspace=scratch, num_splits=4)
