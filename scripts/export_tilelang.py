"""Export auditable SM120 TileLang compilation evidence from public wrappers.

The GPU/compiler imports happen after argument parsing, so --help works on a
CPU-only documentation install. Actual export requires TileLang 0.1.15,
apache-tvm-ffi 0.1.12, a CUDA toolkit, cuobjdump, and an SM120 device.

Each selected public operation is compiled and launched once. This is a source
and compiler-evidence tool, not a correctness test or benchmark. Binary files
are temporary and removed; the output contains text and JSON only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "gemm_tiled", "gemm_pipelined", "gemm_persistent",
    "prefill_pointer", "prefill_tma", "decode_vector_split", "decode_grouped_split",
)
PINNED_PACKAGES = {"tilelang": "0.1.15", "apache-tvm-ffi": "0.1.12"}
SOURCE_PATHS = (
    "scripts/export_tilelang.py", "pyproject.toml",
    "src/fa5060/tilelang/__init__.py", "src/fa5060/tilelang/fundamentals.py",
    "src/fa5060/tilelang/attention.py", "src/fa5060/tilelang/_contracts.py",
    "src/fa5060/_workspace.py", "src/fa5060/probe.py",
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, default=Path("results/tilelang/compiler"),
                        help="new or empty directory for text artifacts and manifest.json")
    result.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES),
                        help="public operations to capture; each split decode also exports its merge")
    result.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    result.add_argument("--device", type=int, default=0, help="CUDA device index; capability must be 12.0")
    result.add_argument("--seed", type=int, default=20261105)
    result.add_argument("--cuobjdump", type=Path, help="override cuobjdump; defaults to the selected NVCC's toolkit")
    return result


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_hashes():
    return {name: sha256((ROOT / name).read_bytes()) for name in SOURCE_PATHS}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def checked_text(command):
    result = subprocess.run([str(value) for value in command], capture_output=True,
                            text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command!r}\n"
                           f"{result.stdout}\n{result.stderr}")
    combined = result.stdout + result.stderr
    if not combined.strip():
        raise RuntimeError(f"command returned no evidence: {command!r}")
    return combined


def packages():
    values = {}
    for name in (*PINNED_PACKAGES, "torch", "triton", "numpy"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    for name, expected in PINNED_PACKAGES.items():
        if values[name] != expected:
            raise RuntimeError(f"{name} must be {expected}; found {values[name]!r}. "
                               "Private export interfaces are pinned; adapt and review before changing the version.")
    return values


def tool_path(value, name, sibling=None):
    candidates = [value] if value is not None else []
    if value is None:
        if sibling is not None:
            candidates.append(sibling.parent / name)
        resolved = shutil.which(name)
        if resolved:
            candidates.append(Path(resolved))
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        if not candidate.is_absolute() and candidate.parent == Path("."):
            found = shutil.which(str(candidate))
            if found:
                candidate = Path(found)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(f"{name} is required; check CUDA_HOME/PATH or --cuobjdump")


@contextmanager
def capture_compiles(modules):
    """Hook the project's two compile boundaries, preserving real execution."""
    captured = []
    originals = []

    def hook(original, module_name):
        def compile_and_capture(prim, *args, **kwargs):
            # Snapshot before lowering can transform/annotate the PrimFunc.
            original_tir = prim.script()
            frame = inspect.currentframe().f_back
            factory = frame.f_code.co_name
            del frame
            kernel = original(prim, *args, **kwargs)
            captured.append({"module": module_name, "factory": factory,
                             "compile_keywords": kwargs, "original_tir": original_tir,
                             "kernel": kernel})
            return kernel
        return compile_and_capture

    try:
        for module in modules:
            original = module._compile
            originals.append((module, original))
            module._compile = hook(original, module.__name__)
        yield captured
    finally:
        for module, original in originals:
            module._compile = original


def clear_factory_caches(modules):
    # Otherwise a previous case's merge could bypass the compile hook. Disk
    # cache hits remain valid: get_kernel_source/_get_ptx also work without an
    # in-memory kernel.artifact in the pinned release.
    for module in modules:
        for name, value in vars(module).items():
            clear = getattr(value, "cache_clear", None)
            if name.startswith("_") and callable(clear):
                clear()


def tensor_metadata(value):
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device)}


def run_case(name, torch, fundamentals, attention, dtype, device):
    """Use public wrappers, never bind their private factory signatures."""
    if name.startswith("gemm_"):
        variant = name.removeprefix("gemm_")
        config = {"tile_m": 32, "tile_n": 64, "tile_k": 32,
                  "num_warps": 4, "num_stages": 1 if variant == "tiled" else 3}
        a = torch.randn((512, 1024), dtype=dtype, device=device)
        b = torch.randn((1024, 512), dtype=dtype, device=device)
        output = fundamentals.matmul(a, b, variant=variant, config=config)
        return {"public_api": "fa5060.tilelang.matmul", "variant": variant,
                "config": config, "inputs": {"a": tensor_metadata(a), "b": tensor_metadata(b)},
                "output": tensor_metadata(output), "expected_compilations": 1}
    if name.startswith("prefill_"):
        loader = name.removeprefix("prefill_")
        config = {"loader": loader, "block_m": 64 if loader == "tma" else 32,
                  "block_n": 64, "num_warps": 4, "num_stages": 2}
        q = torch.randn((1, 128, 8, 64), dtype=dtype, device=device)
        k = torch.randn((1, 256, 2, 64), dtype=dtype, device=device)
        v = torch.randn_like(k)
        lengths = torch.tensor([193], dtype=torch.int32, device=device)
        output = torch.empty_like(q)
        attention.prefill(q, k, v, kv_lens=lengths, causal=True, out=output, config=config)
        return {"public_api": "fa5060.tilelang.prefill", "config": config,
                "causal": "right_aligned", "kv_lens": [193], "layout": "dense",
                "inputs": {"q": tensor_metadata(q), "k": tensor_metadata(k), "v": tensor_metadata(v)},
                "output": tensor_metadata(output), "expected_compilations": 1}
    mode = "vector" if name == "decode_vector_split" else "grouped"
    config = {"mode": mode, "loader": "pointer", "block_n": 64,
              "num_warps": 4, "num_stages": 1 if mode == "vector" else 2}
    q = torch.randn((2, 8, 64), dtype=dtype, device=device)
    k = torch.randn((2, 1024, 2, 64), dtype=dtype, device=device)
    v = torch.randn_like(k)
    lengths = torch.tensor([1024, 193], dtype=torch.int32, device=device)
    output = torch.empty_like(q)
    workspace = attention.decode_workspace(q, k, num_splits=4, config=config)
    attention.decode(q, k, v, kv_lens=lengths, out=output, workspace=workspace,
                     num_splits=4, config=config)
    return {"public_api": "fa5060.tilelang.decode", "config": config, "num_splits": 4,
            "causal": False, "kv_lens": [1024, 193], "layout": "dense",
            "inputs": {"q": tensor_metadata(q), "k": tensor_metadata(k), "v": tensor_metadata(v)},
            "output": tensor_metadata(output), "expected_compilations": 2,
            "workspace": {key: tensor_metadata(getattr(workspace, key))
                          for key in ("partial", "maxima", "sums")}}


def text_artifact(directory, name, content):
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"empty or non-text export: {name}")
    encoded = content.encode("utf-8")
    (directory / name).write_bytes(encoded)
    return {"path": name, "sha256": sha256(encoded), "bytes": len(encoded)}


def export_kernel(capture, directory, nvcc, cuobjdump):
    kernel = capture["kernel"]
    if kernel.execution_backend != "tvm_ffi" or str(kernel.target.attrs.get("arch")) != "sm_120":
        raise RuntimeError(f"unexpected backend/target: {kernel.execution_backend}, {kernel.target}")
    cuda = kernel.get_kernel_source()
    artifacts = {
        "cuda": text_artifact(directory, "kernel.cu", cuda),
        "original_tir": text_artifact(directory, "original_tir.py", capture["original_tir"]),
    }
    # Unlike export_ptx(), _get_ptx() propagates compiler failures in 0.1.15.
    # It recompiles the actual generated CUDA under kernel.target, using the
    # release's default_compile_options plus kernel.compile_flags.
    ptx = kernel._get_ptx(verbose=True)
    target = re.search(r"(?m)^\s*\.target\s+([^\n]+)", ptx)
    if not target or not re.search(r"\bsm_120\b", target.group(1)):
        raise RuntimeError(f"PTX did not declare the requested sm_120 target: {target}")
    artifacts["ptx"] = text_artifact(directory, "kernel.ptx", ptx)
    options = nvcc.default_compile_options(kernel.compile_flags)
    with tempfile.TemporaryDirectory(prefix="fa5060-tilelang-export-") as temporary:
        binary = Path(temporary) / "kernel.cubin"
        # Reassemble the same generated source for resources. This is not an
        # extraction of the loaded FFI module and is labeled accordingly.
        with kernel.target:
            cubin = nvcc.compile_cuda(cuda, target_format="cubin", arch="sm_120",
                                      options=options, path_target=str(binary), verbose=True)
        if not binary.is_file() or not cubin:
            raise RuntimeError("NVCC produced no CUBIN for resource inspection")
        if binary.read_bytes() != bytes(cubin):
            raise RuntimeError("NVCC returned bytes that differ from the inspected CUBIN")
        resource_command = [str(cuobjdump), "--dump-resource-usage", str(binary)]
        resources = checked_text(resource_command)
        if not re.search(r"\bREG\s*:|REGISTERS|Function\s*:", resources):
            raise RuntimeError("cuobjdump output contains no per-kernel resource evidence")
        artifacts["resources"] = text_artifact(directory, "cuobjdump-resources.txt", resources)
        binary_sha = sha256(bytes(cubin))
    record = {key: capture[key] for key in ("module", "factory", "compile_keywords")}
    record.update({"target": str(kernel.target), "execution_backend": kernel.execution_backend,
                   "pass_configs": {str(k): v for k, v in kernel.pass_configs.items()},
                   "compile_flags": kernel.compile_flags, "nvcc_options": options,
                   "artifacts": artifacts,
                   "resource_provenance": "cuobjdump on a temporary CUBIN recompiled from this generated CUDA; not extracted from the loaded FFI module",
                   "temporary_cubin_sha256": binary_sha, "temporary_cubin_retained": False,
                   "resource_command": resource_command,
                   "jit_kernel_source_sha256": sha256(Path(inspect.getfile(type(kernel))).read_bytes()),
                   "cuda_resource_fields": "No n_regs/n_spills getattr values are treated as measured CUDA resources.",
                   "ptx_observations": {"mma_sync_present": "mma.sync" in ptx,
                                        "tma_load_present": "cp.async.bulk.tensor" in ptx,
                                        "tcgen05_present": "tcgen05" in ptx}})
    return record


def execute(args):
    if args.device < 0:
        raise ValueError("--device must be nonnegative")
    if len(set(args.cases)) != len(args.cases):
        raise ValueError("duplicate --cases are not allowed")
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"choose a new or empty --output-dir: {output}")
    versions = packages()  # Verify the pinned private interface before imports.
    import torch
    from tilelang.contrib import nvcc
    from fa5060.tilelang import fundamentals, attention
    from fa5060.probe import collect_environment

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(args.device) != (12, 0):
        raise RuntimeError("export requires a visible device with compute capability 12.0 (SM120)")
    nvcc_path = tool_path(Path(nvcc.get_nvcc_compiler()), "nvcc")
    cuobjdump = tool_path(args.cuobjdump, "cuobjdump", nvcc_path)
    ptxas = tool_path(None, "ptxas", nvcc_path)
    props = torch.cuda.get_device_properties(args.device)
    snapshot = source_hashes()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "compiler evidence; selected public operations launch once, without numerical or timing claims",
        "target": {"kind": "cuda", "arch": "sm_120", "compute_capability": [12, 0]},
        "device": {"index": args.device, "name": props.name, "sm_count": props.multi_processor_count,
                   "total_memory_bytes": props.total_memory},
        "packages": versions, "python": sys.version, "platform": platform.platform(),
        "torch_cuda_build": torch.version.cuda, "source_sha256": snapshot,
        "environment": collect_environment(),
        "reference_tilelang_source_commit": "a35f8ddf45eba16c21211ec8822d56ce5363036f",
        "reference_commit_note": "Reference attribution, not an assertion of the installed wheel's VCS commit; installed provenance is recorded by the environment probe.",
        "private_interfaces": ["project fundamentals._compile/attention._compile hooks and factory cache_clear",
                               "TileLang 0.1.15 JITKernel._get_ptx, compile_flags/pass_configs",
                               "tilelang.contrib.nvcc.default_compile_options/compile_cuda"],
        "tools": {"nvcc": {"path": str(nvcc_path), "version": checked_text([nvcc_path, "--version"])},
                  "ptxas": {"path": str(ptxas), "version": checked_text([ptxas, "--version"])},
                  "cuobjdump": {"path": str(cuobjdump), "version": checked_text([cuobjdump, "--version"])}},
        "seed": args.seed, "dtype": args.dtype, "requested_cases": args.cases,
        "binary_policy": "Only text/JSON are published; resource CUBINs are temporary and deleted.",
        "cases": [],
    }
    manifest["git_commit"] = checked_text(["git", "-C", ROOT, "rev-parse", "HEAD"]).strip()
    manifest["installed_export_source_sha256"] = {
        "tilelang.contrib.nvcc": sha256(Path(inspect.getfile(nvcc)).read_bytes()),
    }
    write_json(output / "manifest.json", manifest)
    modules = (fundamentals, attention)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    current = None
    try:
        with torch.cuda.device(args.device):
            torch.manual_seed(args.seed)
            for current in args.cases:
                clear_factory_caches(modules)
                with capture_compiles(modules) as captures:
                    case = run_case(current, torch, fundamentals, attention, dtype, f"cuda:{args.device}")
                    torch.cuda.synchronize(args.device)
                if len(captures) != case["expected_compilations"]:
                    raise RuntimeError(f"{current}: expected {case['expected_compilations']} compiled kernels, captured {len(captures)}")
                case.update({"name": current, "launch_completed": True, "correctness_validated": False,
                             "kernels": []})
                manifest["cases"].append(case)
                for index, capture in enumerate(captures):
                    role = "merge" if capture["factory"] == "_merge_splits" else "main"
                    directory = output / current / role
                    directory.mkdir(parents=True)
                    record = export_kernel(capture, directory, nvcc, cuobjdump)
                    record["role"] = role
                    record["compile_order"] = index
                    record["directory"] = directory.relative_to(output).as_posix()
                    case["kernels"].append(record)
                write_json(output / "manifest.json", manifest)
        if source_hashes() != snapshot:
            raise RuntimeError("project sources changed during export; rerun against a fixed source snapshot")
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update({"status": "failed", "failed_case": current,
                         "error": {"type": type(exc).__name__, "message": str(exc)}})
        write_json(output / "manifest.json", manifest)
        raise
    print(f"Exported {sum(len(c['kernels']) for c in manifest['cases'])} kernels to {output}")
    print(f"Manifest SHA256: {sha256((output / 'manifest.json').read_bytes())}")
    return 0


def main(argv=None):
    return execute(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
