"""Contracts and numerical checks for the teaching kernels.

CPU validation tests are independent of CUDA.  GPU tests deliberately compare
against fp32 PyTorch arithmetic, including masked tails and strided rows.
"""

from importlib import import_module

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def fundamentals():
    return import_module("fa5060.fundamentals")


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("a CUDA GPU is required")
    return torch.device("cuda")


@pytest.mark.parametrize("function", ["vector_add", "row_softmax", "transpose", "matmul"])
def test_requires_tensor_arguments(function):
    fundamentals = import_module("fa5060.fundamentals")
    args = (None, None) if function in {"vector_add", "matmul"} else (None,)
    with pytest.raises(TypeError, match="Tensor"):
        getattr(fundamentals, function)(*args)


@pytest.mark.parametrize("function", ["vector_add", "row_softmax", "transpose", "matmul"])
def test_rejects_cpu_inputs_without_computing(function):
    fundamentals = import_module("fa5060.fundamentals")
    shape = (7,) if function == "vector_add" else (3, 3)
    x = torch.ones(shape, dtype=torch.float16)
    args = (x, x) if function in {"vector_add", "matmul"} else (x,)
    with pytest.raises(ValueError, match="CUDA"):
        getattr(fundamentals, function)(*args)


@pytest.mark.parametrize("function", ["vector_add", "row_softmax", "transpose", "matmul"])
def test_rejects_wrong_rank(function):
    fundamentals = import_module("fa5060.fundamentals")
    # Even on CPU, the shape contract can be checked before launching anything.
    x = torch.ones(2, 3, 4, dtype=torch.float16)
    args = (x, x) if function in {"vector_add", "matmul"} else (x,)
    with pytest.raises(ValueError, match="dimensional"):
        getattr(fundamentals, function)(*args)


@pytest.mark.parametrize("function", ["vector_add", "row_softmax", "transpose", "matmul"])
def test_rejects_integer_inputs(function):
    fundamentals = import_module("fa5060.fundamentals")
    shape = (7,) if function == "vector_add" else (3, 3)
    x = torch.ones(shape, dtype=torch.int32)
    args = (x, x) if function in {"vector_add", "matmul"} else (x,)
    with pytest.raises(TypeError, match="dtype"):
        getattr(fundamentals, function)(*args)


@pytest.mark.parametrize("config", [
    {"tile_m": 17}, {"tile_n": 0}, {"tile_k": 8}, {"num_warps": 3},
    {"num_stages": 0}, {"tile_m": True}, {"unknown": 32},
    {"tile_m": 128, "tile_n": 128, "tile_k": 128, "num_stages": 4},
    [("tile_m", 32)],
])
def test_matmul_rejects_bad_config_without_cuda(config):
    fundamentals = import_module("fa5060.fundamentals")
    a = torch.ones((3, 4), dtype=torch.float16)
    b = torch.ones((4, 5), dtype=torch.float16)
    with pytest.raises((TypeError, ValueError), match="config"):
        fundamentals.matmul(a, b, config=config)


def test_matmul_rejects_unknown_schedule_without_cuda():
    fundamentals = import_module("fa5060.fundamentals")
    a = torch.ones((3, 4), dtype=torch.float16)
    b = torch.ones((4, 5), dtype=torch.float16)
    with pytest.raises(ValueError, match="variant"):
        fundamentals.matmul(a, b, variant="unknown")


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [0, 1, 1025, 100003])
def test_vector_add_masked_tail(fundamentals, cuda_device, dtype, length):
    x = torch.randn(length, device=cuda_device, dtype=dtype)
    y = torch.randn_like(x)
    actual = fundamentals.vector_add(x, y)
    assert actual.shape == x.shape
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, (x.float() + y.float()).to(dtype), rtol=0, atol=0)


@pytest.mark.gpu
def test_vector_add_rejects_shape_dtype_and_stride_mismatches(fundamentals, cuda_device):
    x = torch.ones(6, device=cuda_device)
    with pytest.raises(ValueError, match="shape"):
        fundamentals.vector_add(x, x[:3])
    with pytest.raises(TypeError, match="dtype"):
        fundamentals.vector_add(x, x.half())
    with pytest.raises(ValueError, match="contiguous"):
        fundamentals.vector_add(x[::2], x[::2])


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1), (7, 33), (17, 513), (3, 4097), (0, 7), (5, 0)])
def test_row_softmax_strided_rows_and_tail(fundamentals, cuda_device, dtype, shape):
    rows, cols = shape
    x = torch.randn(rows * 2, cols, device=cuda_device, dtype=dtype)[::2]
    actual = fundamentals.row_softmax(x)
    expected = torch.softmax(x.float(), dim=-1).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=2e-3 if dtype != torch.bfloat16 else 1e-2, atol=2e-4)


@pytest.mark.gpu
def test_row_softmax_does_not_overflow_large_logits(fundamentals, cuda_device):
    x = torch.tensor([[10000.0, 9999.0, -10000.0], [-10000.0, -10000.0, -10001.0]], device=cuda_device)
    actual = fundamentals.row_softmax(x)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.softmax(x, dim=-1), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_row_softmax_wide_online_normalizer_and_broadcast_rows(fundamentals, cuda_device):
    x = torch.randn((1, 65537), device=cuda_device, dtype=torch.float32).expand(3, -1)
    actual = fundamentals.row_softmax(x)
    torch.testing.assert_close(actual, torch.softmax(x, dim=-1), rtol=1e-5, atol=1e-7)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 32), (37, 71), (1, 65), (0, 5), (5, 0)])
def test_transpose_strided_rows_and_rectangular_tails(fundamentals, cuda_device, dtype, shape):
    rows, cols = shape
    x = torch.randn(rows * 2, cols, device=cuda_device, dtype=dtype)[::2]
    actual = fundamentals.transpose(x)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, x.t(), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("function", ["row_softmax", "transpose"])
def test_row_kernels_reject_noncontiguous_columns(fundamentals, cuda_device, function):
    x = torch.ones((3, 8), device=cuda_device)[:, ::2]
    with pytest.raises(ValueError, match="last dimension"):
        getattr(fundamentals, function)(x)


@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["tiled", "pipelined", "persistent"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1, 1), (37, 53, 29), (65, 129, 97), (0, 7, 3), (7, 0, 3), (5, 7, 0)])
def test_matmul_variants_mask_all_three_tails(fundamentals, cuda_device, variant, dtype, shape):
    m, n, k = shape
    a = torch.randn((m * 2, k), dtype=dtype, device=cuda_device)[::2]
    b = torch.randn((k * 2, n), dtype=dtype, device=cuda_device)[::2]
    actual = fundamentals.matmul(a, b, variant=variant)
    expected = (a.float() @ b.float()).to(dtype)
    assert actual.shape == (m, n)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=1e-2 if dtype == torch.bfloat16 else 2e-3, atol=3e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["tiled", "pipelined", "persistent"])
def test_matmul_explicit_tile_config_and_persistent_multiple_tiles(fundamentals, cuda_device, variant):
    # 221 tiles exceeds the 26-SM target: persistent programs must advance tiles.
    a = torch.randn((257, 77), dtype=torch.float16, device=cuda_device)
    b = torch.randn((77, 385), dtype=torch.float16, device=cuda_device)
    config = {"tile_m": 16, "tile_n": 32, "tile_k": 32, "num_warps": 4, "num_stages": 2}
    actual = fundamentals.matmul(a, b, variant=variant, config=config)
    torch.testing.assert_close(actual, (a.float() @ b.float()).half(), rtol=2e-3, atol=3e-2)


@pytest.mark.gpu
def test_matmul_rejects_invalid_contracts(fundamentals, cuda_device):
    a = torch.ones((3, 4), dtype=torch.float16, device=cuda_device)
    b = torch.ones((4, 5), dtype=torch.float16, device=cuda_device)
    with pytest.raises(ValueError, match="inner dimensions"):
        fundamentals.matmul(a, b[:3])
    with pytest.raises(ValueError, match="last dimension"):
        fundamentals.matmul(a[:, ::2], b[::2])
    with pytest.raises(TypeError, match="dtype"):
        fundamentals.matmul(a.float(), b.float())
    with pytest.raises(TypeError, match="dtype"):
        fundamentals.matmul(a, b.bfloat16())
    with pytest.raises(ValueError, match="variant"):
        fundamentals.matmul(a, b, variant="unknown")
