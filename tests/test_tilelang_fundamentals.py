"""Public TileLang contracts and GPU numerics, independently of Triton."""

from importlib import import_module
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def fundamentals():
    return import_module("fa5060.tilelang.fundamentals")


@pytest.fixture
def cuda_device():
    pytest.importorskip("tilelang")
    if not torch.cuda.is_available():
        pytest.skip("a CUDA GPU is required")
    return torch.device("cuda")


def test_package_import_is_lazy_without_gpu_dependencies():
    # A CPU documentation process must not load either GPU compiler or torch.
    source = """
import importlib.abc
import sys
class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'tilelang'}:
            raise AssertionError('unexpected eager import: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
import fa5060.tilelang as backend
assert {'vector_add', 'row_softmax', 'transpose', 'matmul', 'prefill', 'decode', 'decode_workspace'} <= set(backend.__all__)
try:
    backend.not_a_kernel
except AttributeError:
    pass
else:
    raise AssertionError('unknown names must raise AttributeError')
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-c", source], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("function", ["vector_add", "row_softmax", "transpose", "matmul"])
def test_host_contracts_do_not_require_a_gpu_compiler(fundamentals, function):
    args = (None, None) if function in {"vector_add", "matmul"} else (None,)
    with pytest.raises(TypeError, match="Tensor"):
        getattr(fundamentals, function)(*args)
    shape = (7,) if function == "vector_add" else (3, 3)
    for dtype, error, match in [
        (torch.float16, ValueError, "CUDA"),
        (torch.int32, TypeError, "dtype"),
    ]:
        x = torch.ones(shape, dtype=dtype)
        args = (x, x) if function in {"vector_add", "matmul"} else (x,)
        with pytest.raises(error, match=match):
            getattr(fundamentals, function)(*args)
    x = torch.ones((2, 3, 4), dtype=torch.float16)
    args = (x, x) if function in {"vector_add", "matmul"} else (x,)
    with pytest.raises(ValueError, match="dimensional"):
        getattr(fundamentals, function)(*args)


@pytest.mark.parametrize("config", [
    {"tile_m": 17}, {"tile_n": 0}, {"tile_k": 8}, {"num_warps": 3},
    {"num_stages": 0}, {"tile_m": True}, {"unknown": 32},
    {"tile_m": 128, "tile_n": 128, "tile_k": 128, "num_stages": 4},
    [("tile_m", 32)],
])
def test_matmul_rejects_invalid_config_before_cuda(fundamentals, config):
    a, b = torch.ones((3, 4), dtype=torch.float16), torch.ones((4, 5), dtype=torch.float16)
    with pytest.raises((TypeError, ValueError), match="config"):
        fundamentals.matmul(a, b, config=config)


def test_matmul_rejects_unknown_variant_before_cuda(fundamentals):
    with pytest.raises(ValueError, match="variant"):
        fundamentals.matmul(None, None, variant="unknown")


@pytest.mark.parametrize("variant", ["pipelined", "persistent"])
def test_long_matmul_pipeline_avoids_unsupported_sm120_warp_specialization(fundamentals, monkeypatch, variant):
    # Use the actual lowering/codegen passes without invoking nvcc or a GPU.
    # Long K loops trigger 0.1.15's warp-specialization heuristic; its
    # setmaxnreg CTA reconfiguration is unsupported on SM120.
    tilelang = pytest.importorskip("tilelang")

    def lower_without_device_compilation(prim, **options):
        target = tilelang.tvm.target.Target(options["target"])
        with target, tilelang.tvm.transform.PassContext(config=options.get("pass_configs", {})):
            return tilelang.lower(prim, target=target, enable_host_codegen=False, enable_device_compile=False)

    monkeypatch.setattr(tilelang, "compile", lower_without_device_compilation)
    factory = getattr(fundamentals, f"_matmul_{variant}_kernel")
    arguments = (512, 512, 512, 512, 512, "float16", 32, 64, 32, 4, 3)
    factory.cache_clear()
    try:
        artifact = factory(*arguments, 26) if variant == "persistent" else factory(*arguments)
        source = artifact.kernel_source
        assert "mma_sync" in source
        assert "warpgroup_reg_alloc" not in source
        assert "warpgroup_reg_dealloc" not in source
        assert "__launch_bounds__(128" in source
    finally:
        factory.cache_clear()


# Reuse the exact numerical/shape contract suite of the Triton backend. Pytest
# injects this module's TileLang fixture into those imported test functions;
# future additions to the shared cases do not need another mirrored body.
from test_fundamentals import (  # noqa: E402, F401
    test_matmul_explicit_tile_config_and_persistent_multiple_tiles,
    test_matmul_rejects_invalid_contracts,
    test_matmul_variants_mask_all_three_tails,
    test_row_kernels_reject_noncontiguous_columns,
    test_row_softmax_does_not_overflow_large_logits,
    test_row_softmax_strided_rows_and_tail,
    test_row_softmax_wide_online_normalizer_and_broadcast_rows,
    test_transpose_strided_rows_and_rectangular_tails,
    test_vector_add_masked_tail,
    test_vector_add_rejects_shape_dtype_and_stride_mismatches,
)
