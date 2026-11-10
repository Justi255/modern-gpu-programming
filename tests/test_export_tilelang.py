"""CPU checks for compiler-evidence provenance and failure handling."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_tilelang.py"
spec = importlib.util.spec_from_file_location("export_tilelang", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class Target:
    # TVM Target exposes attributes through an FFI Map, not as .arch.
    attrs = {"arch": "sm_120"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __str__(self):
        return "cuda -arch=sm_120"


class Kernel:
    execution_backend = "tvm_ffi"
    target = Target()
    pass_configs = {"tl.disable_warp_specialized": True}
    compile_flags = None

    def get_kernel_source(self):
        return "// captured generated CUDA\n"

    def _get_ptx(self, verbose):
        return ".version 9.0\n.target sm_120\n.address_size 64\n"


def capture(kernel=None):
    return {"kernel": kernel or Kernel(), "module": "test", "factory": "_merge_splits",
            "compile_keywords": {}, "original_tir": "# captured original TIR\n"}


def test_help_requires_only_standard_library(tmp_path):
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--cuobjdump" in result.stdout
    assert "decode_grouped_split" in result.stdout


def test_capture_snapshots_original_and_restores_hook_after_failure():
    prim = SimpleNamespace(source="original", script=lambda: prim.source)

    def original(value, *, fail=False):
        if fail:
            raise RuntimeError("compile failed")
        value.source = "changed during lowering"
        return "compiled"

    module = SimpleNamespace(__name__="fake", _compile=original)
    with export.capture_compiles([module]) as compiled:
        assert module._compile(prim) == "compiled"
    assert module._compile is original
    assert compiled[0]["original_tir"] == "original"
    with pytest.raises(RuntimeError, match="compile failed"):
        with export.capture_compiles([module]):
            module._compile(prim, fail=True)
    assert module._compile is original


def test_resource_report_is_hashed_and_temporary_binary_deleted(tmp_path, monkeypatch):
    binary_paths = []

    def compile_cuda(source, *, target_format, arch, options, path_target, verbose):
        assert source == Kernel().get_kernel_source()
        assert target_format == "cubin" and arch == "sm_120"
        path = Path(path_target)
        path.write_bytes(b"temporary-cubin")
        binary_paths.append(path)
        return bytearray(path.read_bytes())

    def resource_command(command):
        assert Path(command[-1]).is_file()
        assert command[-2] == "--dump-resource-usage"
        return "Function kernel:\n REG:32 STACK:0 SHARED:1024\n"

    monkeypatch.setattr(export, "checked_text", resource_command)
    nvcc = SimpleNamespace(default_compile_options=lambda flags: ["-std=c++20"],
                           compile_cuda=compile_cuda)
    result = export.export_kernel(capture(), tmp_path, nvcc, Path("/fake/cuobjdump"))
    # A successful export must also survive the manifest's JSON boundary.
    export.write_json(tmp_path / "manifest.json", result)
    assert json.loads((tmp_path / "manifest.json").read_text())["target"] == "cuda -arch=sm_120"
    assert result["temporary_cubin_retained"] is False
    assert result["temporary_cubin_sha256"] == hashlib.sha256(b"temporary-cubin").hexdigest()
    assert all(not path.exists() for path in binary_paths)
    assert {path.suffix for path in tmp_path.iterdir()} == {".cu", ".py", ".ptx", ".txt", ".json"}
    for item in result["artifacts"].values():
        assert hashlib.sha256((tmp_path / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert "not extracted" in result["resource_provenance"]


def test_ptx_compiler_failure_propagates_without_fake_ptx(tmp_path):
    class FailedKernel(Kernel):
        def _get_ptx(self, verbose):
            raise RuntimeError("real compiler diagnostic")

    with pytest.raises(RuntimeError, match="real compiler diagnostic"):
        export.export_kernel(capture(FailedKernel()), tmp_path, None, None)
    assert not (tmp_path / "kernel.ptx").exists()


def test_wrong_ptx_target_is_rejected(tmp_path):
    class WrongTarget(Kernel):
        def _get_ptx(self, verbose):
            return ".version 9.0\n.target sm_100\n"

    with pytest.raises(RuntimeError, match="did not declare"):
        export.export_kernel(capture(WrongTarget()), tmp_path, None, None)
    assert not (tmp_path / "kernel.ptx").exists()


def test_nonzero_tool_status_is_not_resource_evidence():
    with pytest.raises(RuntimeError, match="missing real resources"):
        export.checked_text([sys.executable, "-c", "import sys; sys.stderr.write('missing real resources'); sys.exit(3)"])
