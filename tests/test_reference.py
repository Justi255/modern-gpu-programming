import pytest
torch = pytest.importorskip('torch')


def test_right_aligned_causal_and_gqa():
    from fa5060.reference import attention_reference

    q = torch.zeros(1, 2, 4, 8)
    k = torch.zeros(1, 4, 2, 8)
    v = torch.arange(4.0)[None, :, None, None].expand_as(k).clone()
    result = attention_reference(q, k, v)
    torch.testing.assert_close(result[:, 0], torch.ones(1, 4, 8))
    torch.testing.assert_close(result[:, 1], torch.full((1, 4, 8), 1.5))


def test_paged_reference_ignores_unused_and_tail_slots():
    from fa5060.reference import attention_reference

    q = torch.zeros(2, 4, 8)
    k = torch.zeros(3, 2, 1, 8)
    v = torch.tensor([20., 21., 10., 11., 30., 999.]).view(3, 2, 1, 1).expand_as(k)
    table = torch.tensor([[1, 2], [0, -1]], dtype=torch.int32)
    result = attention_reference(q, k, v, kv_lens=torch.tensor([3, 0]), block_table=table)
    torch.testing.assert_close(result[0], torch.full((4, 8), 17.))
    torch.testing.assert_close(result[1], torch.zeros(4, 8))


def test_reference_empty_masked_queries_are_zero():
    from fa5060.reference import attention_reference

    q = torch.zeros(1, 3, 1, 4)
    k = torch.zeros(1, 1, 1, 4)
    v = torch.full_like(k, 7.)
    result = attention_reference(q, k, v)
    torch.testing.assert_close(result[0, :2], torch.zeros(2, 1, 4))
    torch.testing.assert_close(result[0, 2], torch.full((1, 4), 7.))


def test_reference_scale_and_small_chunk_match_explicit_math():
    from fa5060.reference import attention_reference

    torch.manual_seed(4)
    q, k, v = [torch.randn(1, 7, 2, 8, dtype=torch.float64) for _ in range(3)]
    expected = ((q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)) * 0.3).softmax(-1) @ v.transpose(1, 2)
    actual = attention_reference(q, k, v, causal=False, scale=0.3, chunk_size=2)
    torch.testing.assert_close(actual, expected.transpose(1, 2), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('length', [-1, 5])
def test_reference_rejects_invalid_lengths(length):
    from fa5060.reference import attention_reference

    q = torch.zeros(1, 1, 2, 8)
    k = torch.zeros(1, 4, 1, 8)
    with pytest.raises(ValueError):
        attention_reference(q, k, k, kv_lens=torch.tensor([length]))
