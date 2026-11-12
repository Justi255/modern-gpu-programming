"""Ragged and graph semantics for the measured optimization candidates."""
import pytest

torch = pytest.importorskip('torch')
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA GPU required')]


@pytest.mark.parametrize('variant', ['register', 'qregister', 'async', 'qasync', 'pipeline'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('d', [64, 128])
def test_prefill_candidate_dynamic_pages_and_poisoned_tail(monkeypatch, variant, dtype, d):
    from importlib import import_module
    import fa5060.tilelang.attention as api
    from fa5060.reference import attention_reference
    if variant == 'pipeline':
        from functools import partial
        factory = partial(import_module('fa5060.tilelang._decode_pipeline')._prefill_kernel, grouped=False)
    else:
        factory = import_module('fa5060.tilelang._prefill_' + variant)._prefill_kernel
    monkeypatch.setattr(api, '_prefill_kernel', factory)
    torch.manual_seed(6111)
    q = torch.randn(2, 146, 8, d, device='cuda', dtype=dtype)[:, ::2]
    k = torch.randn(40, 16, 2, d, device='cuda', dtype=dtype)
    v = torch.randn_like(k)
    # Capacity 208, with a random physical mapping and one dead tail page.
    table = torch.randperm(38, device='cuda')[:26].int().reshape(2, 13)
    table[:, -1] = 39
    k[39] = float('nan')
    v[39] = float('nan')
    lengths = torch.tensor([191, 17], device='cuda', dtype=torch.int32)
    out = torch.empty(2, 146, 8, d, device='cuda', dtype=dtype)[:, ::2]
    config = {'block_m': 64, 'block_n': 32, 'num_stages': 2}
    def call():
        return api.prefill(q, k, v, kv_lens=lengths, block_table=table,
                           config=config, out=out)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): call()
    tol = .003 if dtype == torch.float16 else .02
    for live in ([191, 17], [7, 161], [0, 97]):
        lengths.copy_(torch.tensor(live, device='cuda', dtype=torch.int32))
        table[0, 0] = (table[0, 0] + 1) % 38
        graph.replay()
        expected = attention_reference(q, k, v, kv_lens=lengths, block_table=table)
        torch.testing.assert_close(out, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize('page', [16, 32, 64])
@pytest.mark.parametrize('d', [32, 64, 128])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_warp_decode_empty_splits_and_unaligned_split_boundary(monkeypatch, page, d, dtype):
    import fa5060.tilelang.attention as api
    from fa5060.tilelang._decode_warp import _decode_vector_kernel
    from fa5060.reference import attention_reference
    monkeypatch.setattr(api, '_decode_vector_kernel', _decode_vector_kernel)
    q = torch.randn(3, 16, d, device='cuda', dtype=dtype)[:, ::2]
    k = torch.randn(12, page, 2, d, device='cuda', dtype=dtype)
    v = torch.randn_like(k)
    table = torch.tensor([[6,2,8], [3,0,-1], [-1,-1,-1]], device='cuda', dtype=torch.int32)
    lengths = torch.tensor([2*page+3, page-1, 0], device='cuda', dtype=torch.int32)
    config = {'mode':'vector', 'block_n':32}
    scratch = api.decode_workspace(q, k, block_table=table, num_splits=8, config=config)
    out = torch.empty(3, 16, d, device='cuda', dtype=dtype)[:, ::2]
    def call():
        return api.decode(q, k, v, kv_lens=lengths, block_table=table,
            num_splits=8, config=config, workspace=scratch, out=out)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): call()
    lengths[0] = 7
    table[0, 0] = 5
    graph.replay()
    tol = .003 if dtype == torch.float16 else .02
    torch.testing.assert_close(out, attention_reference(q, k, v, kv_lens=lengths,
        block_table=table), atol=tol, rtol=tol)


@pytest.mark.parametrize('page', [16, 32, 64])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('d', [32, 64, 128])
@pytest.mark.parametrize('segmented', [False, True])
def test_pipeline_grouped_decode_graph_and_boundary_lengths(monkeypatch, page, dtype, d, segmented):
    import fa5060.tilelang.attention as api
    from fa5060.tilelang._decode_pipeline import _prefill_kernel
    from fa5060.reference import attention_reference
    monkeypatch.setattr(api, '_decode_grouped_kernel',
                        lambda *a, **kw: _prefill_kernel(*a, grouped=True,
                                                       segmented_reduce=segmented, **kw))
    torch.manual_seed(6112)
    q = torch.randn(3, 16, d, device='cuda', dtype=dtype)[:, ::2]
    k = torch.randn(50, page, 4, d, device='cuda', dtype=dtype)[:, :, ::2]
    v = torch.randn_like(k)
    table = torch.randperm(48, device='cuda').int().reshape(3, 16)
    lengths = torch.tensor([16*page, 33, 0], device='cuda', dtype=torch.int32)
    config = {'mode': 'grouped', 'block_n': 32, 'num_stages': 2}
    scratch = api.decode_workspace(q, k, block_table=table, num_splits=4, config=config)
    out = torch.empty(3, 16, d, device='cuda', dtype=dtype)[:, ::2]
    def call():
        return api.decode(q, k, v, kv_lens=lengths, block_table=table,
                          num_splits=4, config=config, workspace=scratch, out=out)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): call()
    tol = .003 if dtype == torch.float16 else .02
    for live in ([16*page, 33, 0], [31, 32, 33], [63, 64, 65], [0, 1, 16*page-1]):
        lengths.copy_(torch.tensor(live, device='cuda', dtype=torch.int32))
        table[0, 0] = (table[0, 0] + 1) % 48
        graph.replay()
        expected = attention_reference(q, k, v, kv_lens=lengths, block_table=table)
        torch.testing.assert_close(out, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize('implementation', ['reference', 'optimized'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_public_unaligned_storage_views_and_split_workspace(implementation, dtype):
    from fa5060.tilelang import prefill, decode, decode_workspace
    from fa5060.reference import attention_reference
    def shifted(shape, kind=dtype):
        count = 1
        for size in shape: count *= size
        return torch.randn(count+1, device='cuda', dtype=kind)[1:].view(shape)
    q = shifted((1, 65, 8, 64))
    k, v = shifted((1, 129, 2, 64)), shifted((1, 129, 2, 64))
    lengths = torch.tensor([129], device='cuda', dtype=torch.int32)
    config = {'implementation': implementation, 'block_m':64, 'block_n':32,
              'num_stages':2, 'mode':'grouped'}
    out = shifted(tuple(q.shape))
    prefill(q, k, v, kv_lens=lengths, out=out, config=config)
    tol = .003 if dtype == torch.float16 else .02
    torch.testing.assert_close(out, attention_reference(q,k,v,kv_lens=lengths), atol=tol, rtol=tol)
    qd, od = q[:, 0], shifted((1, 8, 64))
    scratch = decode_workspace(qd,k,num_splits=4,config=config)
    scratch = type(scratch)(shifted(tuple(scratch.partial.shape), torch.float32),
                            shifted(tuple(scratch.maxima.shape), torch.float32),
                            shifted(tuple(scratch.sums.shape), torch.float32), 4)
    def call():
        return decode(qd,k,v,kv_lens=lengths,workspace=scratch,out=od,num_splits=4,config=config)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): call()
    lengths.zero_()
    graph.replay()
    torch.testing.assert_close(od, torch.zeros_like(od), atol=0, rtol=0)
    lengths.fill_(129)
    graph.replay()
    torch.testing.assert_close(od, attention_reference(qd,k,v,kv_lens=lengths), atol=tol, rtol=tol)
