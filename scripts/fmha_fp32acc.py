"""Auditable SM120 FMHAv2 FP16-input/FP32-accumulation experiment.

generate is CPU-only; build and validate explicitly use CUDA. This is a separate
precision mode, not the native flashinfer_fmha_v2 benchmark provider. Original
reports, installed packages, the reference, and validation thresholds are never
modified. Run --help or a subcommand's --help for commands.
"""
import argparse
import ast
from dataclasses import asdict, replace
import difflib
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


LAYOUTS = {"CONTIGUOUS_Q_KV": "CONTIGUOUS_Q_KV", "Q_PAGED_KV_NHD": "Q_PAGED_KV"}
ROOT = Path(__file__).resolve().parents[1]
NATIVE_REPORT = ROOT / "results/tilelang/full-completed.json"
EXPECTED_VERSION = "0.7.2.post1"
ACC_ASSIGNMENT = ("Data_type acc_type =\n"
    "      (data_type == DATA_TYPE_BF16 || data_type == DATA_TYPE_E4M3) ? DATA_TYPE_FP32 : data_type;")
FORCE_ASSIGNMENT = "bool force_fp32_acc = (data_type == DATA_TYPE_BF16 || data_type == DATA_TYPE_E4M3);"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    """Replace only our own new report; never an input report."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def installed_root():
    return Path(importlib.metadata.distribution("flashinfer-python").locate_file("flashinfer")).resolve()


def patch_runner(original):
    """Fail closed if upstream no longer has precisely the inspected code."""
    if original.count(ACC_ASSIGNMENT) != 1 or original.count(FORCE_ASSIGNMENT) != 1:
        raise ValueError("native runner precision assignments differ from the audited version")
    return original.replace(ACC_ASSIGNMENT, "Data_type acc_type = DATA_TYPE_FP32;").replace(
        FORCE_ASSIGNMENT, "bool force_fp32_acc = true;")


def _load_generator(package):
    """Execute installed generator functions without importing FlashInfer/CUDA.

    Only three integration imports are removed from an in-memory AST. The
    installed .utils, factory, validator, encoder, templates and code emitters
    remain unchanged. No module named flashinfer is injected or monkeypatched.
    This import adaptation is recorded in manifest.json.
    """
    source_dir = package / "jit/attention/fmha_v2"
    namespace = "_fa5060_fmha_" + hashlib.sha256(str(package).encode()).hexdigest()[:12]
    private_package = ModuleType(namespace)
    private_package.__path__ = [str(source_dir)]
    sys.modules[namespace] = private_package
    utils_spec = importlib.util.spec_from_file_location(namespace + ".utils", source_dir / "utils.py")
    utils = importlib.util.module_from_spec(utils_spec)
    sys.modules[utils_spec.name] = utils
    utils_spec.loader.exec_module(utils)
    source = source_dir / "fmha_library.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    removed = []
    retained = []
    expected = {(3, None, ("env",)), (4, "compilation_context", ("CompilationContext",)),
                (3, "utils", ("write_if_different",))}
    for node in tree.body:
        key = (node.level, node.module, tuple(alias.name for alias in node.names)) if isinstance(node, ast.ImportFrom) else None
        if key in expected:
            removed.append(key)
        else:
            retained.append(node)
    if set(removed) != expected or len(removed) != 3:
        raise ValueError("installed generator integration imports differ from the audited version")
    tree.body = retained
    generator = ModuleType(namespace + ".fmha_library")
    generator.__package__ = namespace
    generator.__file__ = str(source)
    generator.jit_env = SimpleNamespace(FLASHINFER_CSRC_DIR=package / "data/csrc")
    generator.CompilationContext = object  # annotation only; no instance created
    sys.modules[generator.__name__] = generator  # dataclasses resolves its module
    exec(compile(tree, str(source), "exec"), generator.__dict__)
    return generator, utils


def _safe_directory(directory, package):
    directory = Path(directory).resolve()
    protected = (package.resolve(), Path.home() / ".cache/flashinfer")
    if any(directory == path or path in directory.parents for path in protected):
        raise ValueError("experiment directory must be outside the installed package and native cache")
    return directory


def generate(directory, package=None):
    """Write all four installed-generator kernel variants and their provenance."""
    package = (Path(package) if package is not None else installed_root()).resolve()
    directory = _safe_directory(directory, package)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    generator, utils = _load_generator(package)
    native_paths = ["jit/attention/fmha_v2/utils.py", "jit/attention/fmha_v2/fmha_library.py",
                    "data/csrc/fmha_v2_run.cu", "data/csrc/fmha_v2_jit_binding.cu",
                    "jit/core.py", "jit/attention/modules.py", "compilation_context.py", "prefill.py"]
    # Snapshot all FMHAv2 headers/templates used by the generated kernels.
    native_paths += [path.relative_to(package).as_posix()
                     for path in sorted((package / "data/csrc/fmha_v2").rglob("*"))
                     if path.is_file() and path.suffix not in (".pyc",)]
    native_hashes = {name: sha256(package / name) for name in native_paths}
    original = (package / "data/csrc/fmha_v2_run.cu").read_text()
    patched = patch_runner(original)
    patch = "".join(difflib.unified_diff(original.splitlines(keepends=True),
        patched.splitlines(keepends=True), fromfile="native/fmha_v2_run.cu", tofile="fp32acc/fmha_v2_run.cu"))
    # The identity includes every native FMHAv2 header/template and our script.
    identity = hashlib.sha256(json.dumps({"native": native_hashes,
        "script": sha256(__file__), "mode": "fp16_input_fp32_acc_fp16_output_sm120_d64_d128"},
        sort_keys=True).encode()).hexdigest()[:16]
    directory.mkdir(parents=True)
    for name in native_paths:
        destination = directory / "native" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((package / name).read_bytes())
    (directory / "runner.patch").write_text(patch)
    layouts = {}
    for public_layout, enum_name in LAYOUTS.items():
        uri = f"fa5060_fmha_v2_fp32acc_{public_layout.lower()}_{identity}"
        relative = Path("generated") / uri
        output = directory / relative
        output.mkdir(parents=True)
        specs_names = []
        sources = []
        for dim in (64, 128):
            native_spec = generator.generate_kernel_spec(sm=120, head_size=dim,
                dtype="fp16", output_dtype="fp16", alibi=False, return_softmax=False,
                enable_attn_logit_softcapping=False, enable_skip_softmax=False,
                input_layout=getattr(utils.InputLayout, enum_name))
            spec = replace(native_spec, dtype="fp16_fp32")
            if not generator.is_kernel_spec_valid(spec):
                raise ValueError(f"installed generator rejects {public_layout} D{dim}")
            filename, launch_name, kernel_name = utils.encode_name(spec)
            code = generator.get_kernel_code(spec, kernel_name, launch_name)
            if code is None or "Ampere_hmma_fp32_traits" not in code:
                raise ValueError(f"missing FP32-accumulation kernel for {public_layout} D{dim}")
            (output / filename).write_text(code)
            sources.append((relative / filename).as_posix())
            specs_names.append((spec, filename, launch_name, kernel_name))
        registry = generator.get_api_code(specs_names)
        if "!force_fp32_acc" in registry or registry.count("&& force_fp32_acc") != 2:
            raise ValueError("generated registry has unexpected precision dispatch")
        (output / "fmha_v2_api.h").write_text(registry)
        (output / "fmha_v2_run.cu").write_text(patched)
        (output / "fmha_v2_jit_binding.cu").write_bytes((package / "data/csrc/fmha_v2_jit_binding.cu").read_bytes())
        sources += [(relative / name).as_posix() for name in ("fmha_v2_run.cu", "fmha_v2_jit_binding.cu")]
        layouts[public_layout] = {"uri": uri, "directory": relative.as_posix(),
            "sources": sources, "specs": [asdict(item[0]) for item in specs_names]}
    manifest = {"schema_version": 1, "purpose": "independent FMHAv2 precision diagnostic; no SOTA qualification",
        "precision_mode": "FP16 inputs and output; FP32 accumulation and BMM1/softmax scale encoding",
        "gpu_validation": "not_run", "flashinfer_expected_version": EXPECTED_VERSION,
        "flashinfer_package": str(package), "script_sha256": sha256(__file__),
        "native_sources": native_hashes, "layouts": layouts,
        "cpu_import_adaptation": "Only env, CompilationContext and write_if_different integration imports removed from in-memory AST; pure generator functions unchanged.",
        "runner_change": "Only acc_type=DATA_TYPE_FP32 and force_fp32_acc=true. Original native comments are retained verbatim; runner.patch is authoritative.",
        "cache_workspace_base": str(directory / "cache"),
        "generated_files": {path.relative_to(directory).as_posix(): sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}}
    _json(directory / "manifest.json", manifest)
    return manifest


def verify_manifest(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["generated_files"].items():
        path = (directory / name).resolve()
        if directory not in path.parents or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"SHA256 mismatch: {name}")
    for name, expected in manifest["native_sources"].items():
        path = Path(manifest["flashinfer_package"]) / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"native source SHA256 mismatch: {name}")
    return manifest


def build(directory, selected_layouts=None):
    """CUDA entry point. No FlashInfer import is allowed before cache isolation."""
    directory = Path(directory).resolve()
    manifest = verify_manifest(directory)
    workspace = str(directory / "cache")
    if "flashinfer" in sys.modules:
        from flashinfer.jit import env as existing_env
        if existing_env.FLASHINFER_BASE_DIR.resolve() != Path(workspace):
            raise RuntimeError("FlashInfer already imported with another cache; run this command in a fresh process")
    os.environ["FLASHINFER_WORKSPACE_BASE"] = workspace
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = "12.0"
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("the independent build requires the actual SM120 GPU")
    import flashinfer
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec
    if flashinfer.__version__ != EXPECTED_VERSION:
        raise RuntimeError(f"expected FlashInfer {EXPECTED_VERSION}, got {flashinfer.__version__}")
    if Path(flashinfer.__file__).resolve().parent != Path(manifest["flashinfer_package"]):
        raise RuntimeError("build imports a different FlashInfer package than generation")
    if jit_env.FLASHINFER_BASE_DIR.resolve() != Path(workspace):
        raise RuntimeError("FlashInfer cache isolation failed")
    target_archs = current_compilation_context.TARGET_CUDA_ARCHS
    if len(target_archs) != 1 or next(iter(target_archs))[0] != 12:
        raise RuntimeError(f"expected only SM120 compiler target, got {target_archs}")
    modules = {}
    records = {}
    for layout in (selected_layouts or LAYOUTS):
        config = manifest["layouts"][layout]
        generated = directory / config["directory"]
        # Include-path order and flags mirror installed gen_fmha_v2_module.
        fmha_source = directory / "native/data/csrc/fmha_v2"
        flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
        flags.extend([f"-I{fmha_source}", f"-I{generated}", f"-I{fmha_source}",
                      f"-I{jit_env.FLASHINFER_INCLUDE_DIR}", "-Wno-deprecated-gpu-targets"])
        spec = gen_jit_spec(config["uri"], [directory / source for source in config["sources"]],
                            extra_cuda_cflags=flags)  # native default use_fast_math=True
        modules[layout] = spec.build_and_load()
        library = spec.get_library_path()
        records[layout] = {"uri": config["uri"], "extra_cuda_cflags": flags,
            "use_fast_math": True, "library_path": str(library),
            "library_sha256": sha256(library), "gpu_validation": "not_run"}
    _json(directory / "build.json", {"layouts": records, "cache_workspace_base": workspace,
        "torch": torch.__version__, "flashinfer": flashinfer.__version__,
        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "manifest_sha256": sha256(directory / "manifest.json")})
    return modules


def select_cases(suite, case_ids, native_report):
    from fa5060.workloads import AttentionCase, full_suite
    all_cases = {case.case_id: case for case in full_suite()}
    if case_ids:
        cases = [all_cases[case_id] for case_id in case_ids]
    elif suite == "full-fp16":
        cases = [case for case in all_cases.values() if case.dtype == "fp16"]
    else:
        if native_report is None:
            raise ValueError("native-failures requires --native-report")
        cases = [AttentionCase.from_dict(row) for row in native_report["cases"]
                 if row["providers"].get("flashinfer_fmha_v2", {}).get("status") == "incorrect"]
    if not cases or any(case.dtype != "fp16" or case.head_dim not in (64, 128) for case in cases):
        raise ValueError("select at least one frozen FP16 D64/D128 case; BF16 is not a precision-mode input")
    return cases


def corrected_provider(case, inputs, module):
    """Exact installed module.run ABI; native-layout preparation mirrors adapter."""
    import torch
    from fa5060.benchmark import Provider, _flashinfer_workspace_bytes
    if case.dtype != "fp16" or inputs.q.dtype != torch.float16:
        raise ValueError("the independent precision mode accepts FP16 only")
    q = inputs.q.reshape(-1, case.heads_q, case.head_dim)
    out = torch.empty_like(q)
    workspace = torch.empty(_flashinfer_workspace_bytes(case), device=q.device, dtype=torch.uint8)
    qo = (torch.arange(case.batch + 1, dtype=torch.int32) * case.query_len).to(q.device)
    kv = (torch.arange(case.batch + 1, dtype=torch.int32) * case.kv_len).to(q.device)
    table = None
    if case.layout == "dense":
        layout = "contiguous_q_kv"
        k = inputs.k.view(-1, case.heads_kv, case.head_dim)
        v = inputs.v.view(-1, case.heads_kv, case.head_dim)
        page_size = 0
    else:
        layout = "q_paged_kv_nhd"
        k, v = inputs.k, inputs.v
        if (k.untyped_storage().data_ptr() != v.untyped_storage().data_ptr()
                or v.data_ptr() - k.data_ptr() != k.numel() * k.element_size()):
            raise ValueError("separate paged K/V must share one block-aligned allocation")
        table = torch.stack((inputs.block_table, inputs.block_table + k.shape[0]), dim=1)
        page_size = case.page_size

    def call(preallocate):
        # Dense packing stays inside both eager and graph call boundaries.
        key = torch.stack((k, v), dim=1) if case.layout == "dense" else k
        value = key if case.layout == "dense" else v
        result = out if preallocate else torch.empty_like(q)
        module.run(q, key, value, result, workspace, workspace.numel() * workspace.element_size(),
            table, page_size, inputs.kv_lens, qo, kv, layout, case.query_len, case.kv_len,
            case.batch, "causal" if case.causal else "padding", 1.0, case.head_dim ** -.5,
            1.0, -1, 0, False, 0.0, 0.0, None, None)
        return result.reshape(inputs.q.shape)

    return Provider(lambda: call(True), lambda: call(False), {
        "api": "private FP32-accumulation FMHAv2 module.run", "precision_mode": "FP16/FP32acc/FP16",
        "input_layout": layout.upper(), "preallocated_output": True,
        "packing_timed": case.layout == "dense", "native_paged": case.layout == "paged",
        "block_table_format": "[B,2,M] K/V offsets relative to K" if table is not None else None})


def output_diagnostics(actual, expected):
    """Call unchanged frozen evaluator, retaining its exception plus diagnostics."""
    import torch
    from fa5060.benchmark import CorrectnessError, _validate_output
    result = {}
    try:
        result.update(_validate_output(actual, expected, "fp16"))
        result["status"] = "ok"
    except CorrectnessError as exc:
        result.update(status="incorrect", validation_error=str(exc))
    if not isinstance(actual, torch.Tensor) or tuple(actual.shape) != tuple(expected.shape):
        return result
    # Supplemental element count uses the same frozen FP16 elementwise check.
    # It never replaces the evaluator or determines acceptance itself.
    a, b = actual.reshape(-1), expected.reshape(-1)
    failure_count = 0
    worst_error, worst_index = -1.0, 0
    squared_error = squared_reference = 0.0
    nonfinite = 0
    for start in range(0, expected.numel(), 1 << 20):
        got, want = a[start:start + (1 << 20)].float(), b[start:start + (1 << 20)].float()
        error = (got - want).abs()
        finite = torch.isfinite(got) & torch.isfinite(want)
        nonfinite += int((~finite).sum().item())
        failure_count += int(((error > .003 + .003 * want.abs()) | ~finite).sum().item())
        value, index = error.nan_to_num(nan=float("inf")).max(dim=0)
        if value.item() > worst_error:
            worst_error, worst_index = value.item(), start + index.item()
        squared_error += error.square().sum().item()
        squared_reference += want.square().sum().item()
    shape = tuple(expected.shape)
    coordinate = []
    remainder = worst_index
    for dimension in reversed(shape):
        coordinate.append(remainder % dimension)
        remainder //= dimension
    coordinate.reverse()
    result.update(elementwise_failure_count=failure_count, nonfinite_count=nonfinite,
        supplemental_max_abs_error=worst_error if math.isfinite(worst_error) else None,
        supplemental_normalized_rms_error=(math.sqrt(squared_error / max(squared_reference, 1e-30))
            if math.isfinite(squared_error) and math.isfinite(squared_reference) else None),
        worst_absolute_error_element={"index": coordinate, "query": coordinate[1] if len(shape) == 4 else 0,
            "head": coordinate[-2], "actual": float(a[worst_index].item()) if nonfinite == 0 else None,
            "reference": float(b[worst_index].item()) if nonfinite == 0 else None})
    return result


def _observe(provider, reference):
    import torch
    observations = {}
    for name, call in (("eager", provider.eager_call), ("preallocated", provider.graph_call)):
        observations[name] = output_diagnostics(call(), reference)
    for _ in range(3):
        provider.graph_call()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = provider.graph_call()
    torch.cuda.current_stream().wait_stream(stream)
    # Capture records work. Validate its first actual execution, then a second
    # replay, poisoning the destination so warmup values cannot hide no writes.
    captured.fill_(float("nan"))
    graph.replay()
    observations["captured"] = output_diagnostics(captured, reference)
    captured.fill_(float("nan"))
    graph.replay()
    observations["replayed"] = output_diagnostics(captured, reference)
    torch.cuda.synchronize()
    return observations


def validate(directory, output, suite="native-failures", case_ids=None, native_report_path=NATIVE_REPORT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    native_report_path = Path(native_report_path).resolve()
    native_report = json.loads(native_report_path.read_text()) if native_report_path.exists() else None
    cases = select_cases(suite, case_ids, native_report)
    layouts = list(dict.fromkeys("CONTIGUOUS_Q_KV" if case.layout == "dense" else "Q_PAGED_KV_NHD" for case in cases))
    modules = build(directory, layouts)
    import torch
    from fa5060.benchmark import _flashinfer_provider
    from fa5060.reference import attention_reference
    from fa5060.workloads import make_inputs
    torch.backends.cuda.matmul.allow_tf32 = False
    native_rows = {row["id"]: row for row in native_report["cases"]} if native_report else {}
    report = {"purpose": "independent FP32-accumulation correctness comparison; not a qualification report",
        "manifest": str(Path(directory).resolve() / "manifest.json"),
        "manifest_sha256": sha256(Path(directory) / "manifest.json"),
        "suite": suite, "status": "running", "gpu_validation": "in_progress", "cases": [],
        "native_report": str(native_report_path) if native_report else None,
        "native_report_sha256": sha256(native_report_path) if native_report else None}
    output.parent.mkdir(parents=True, exist_ok=True)
    _json(output, report)
    for case in cases:
        original_row = native_rows.get(case.case_id, {})
        seed = original_row.get("seed", 20261102)
        row = {**case.to_dict(), "seed": seed,
               "original_native_observation": original_row.get("providers", {}).get("flashinfer_fmha_v2"),
               "observations": {}}
        try:
            inputs = make_inputs(case, seed=seed)
            reference = attention_reference(inputs.q, inputs.k, inputs.v, causal=case.causal,
                kv_lens=inputs.kv_lens, block_table=inputs.block_table)
            layout = "CONTIGUOUS_Q_KV" if case.layout == "dense" else "Q_PAGED_KV_NHD"
            for mode in ("native", "fp32acc"):
                try:
                    provider = (_flashinfer_provider(case, inputs, "fmha_v2") if mode == "native"
                                else corrected_provider(case, inputs, modules[layout]))
                    row["observations"][mode] = {"metadata": provider.metadata,
                                                "outputs": _observe(provider, reference)}
                    del provider
                except Exception as exc:
                    row["observations"][mode] = {"status": "error", "error": repr(exc)}
            del reference, inputs
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
        report["cases"].append(row)
        _json(output, report)
        print(case.case_id, json.dumps(row["observations"].get("fp32acc", {})), flush=True)
        torch.cuda.empty_cache()
    passed = all(row.get("observations", {}).get("fp32acc", {}).get("outputs")
        and all(item["status"] == "ok" for item in row["observations"]["fp32acc"]["outputs"].values())
        for row in report["cases"])
    report.update(status="completed", gpu_validation="passed_selected_cases" if passed else "failed_selected_cases",
                  all_corrected_outputs_pass=bool(passed))
    _json(output, report)
    return bool(passed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("generate", help="CPU only: generate sources, source snapshots, patch and SHA manifest")
    generation.add_argument("--directory", type=Path, required=True, help="new independent directory (must not exist)")
    generation.add_argument("--flashinfer-root", type=Path, help="installed flashinfer package directory; no import needed")
    compilation = commands.add_parser("build", help="GPU: compile/load both layouts into the directory's private cache")
    compilation.add_argument("--directory", type=Path, required=True)
    validation = commands.add_parser("validate", help="GPU: compare native and FP32acc on identical frozen inputs")
    validation.add_argument("--directory", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True, help="new standalone report; input reports cannot be overwritten")
    validation.add_argument("--native-report", type=Path, default=NATIVE_REPORT)
    validation.add_argument("--suite", choices=("native-failures", "full-fp16"), default="native-failures")
    validation.add_argument("--case", action="append", help="frozen case ID; repeat for a diagnostic subset")
    args = parser.parse_args(argv)
    # Allows running the repository script without an editable project install.
    sys.path.insert(0, str(ROOT / "src"))
    if args.command == "generate":
        manifest = generate(args.directory, args.flashinfer_root)
        print(json.dumps({"directory": str(args.directory.resolve()), "gpu_validation": "not_run",
                          "uris": [item["uri"] for item in manifest["layouts"].values()]}, indent=2))
    elif args.command == "build":
        build(args.directory)
        print("Independent modules built; numerical GPU validation has not been run.")
    else:
        return 0 if validate(args.directory, args.output, args.suite, args.case, args.native_report) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
