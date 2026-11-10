"""Native attention comparisons with crash-safe evidence and fail-closed claims.

CPU-only consumers can import the workload/statistics code without importing
PyTorch, Triton, FlashInfer, or FA4. CUDA providers are imported on demand.
"""

import argparse
from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
from typing import Any, Callable

from .workloads import AttentionCase, AttentionInputs, full_suite, make_inputs, smoke_suite


SCHEMA_VERSION = 1
BACKENDS = ("triton", "tilelang")
PROVIDER_NAMES = ("ours", "torch_flash", "torch_cudnn", "torch_efficient", "fa4",
                  "flashinfer_fa2", "flashinfer_fmha_v2")


class UnsupportedProvider(RuntimeError):
    """An explicitly unavailable native API or unsupported semantic layout."""


class CorrectnessError(RuntimeError):
    """A provider returned a numerically incorrect result before timing."""


@dataclass
class Provider:
    graph_call: Callable
    eager_call: Callable
    metadata: dict[str, Any]


def _check_backend(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    return backend


def report_backend(report: dict[str, Any]) -> str:
    """Schema-1 reports predating backend selection used Triton."""
    return _check_backend(report.get("backend", "triton"))


def environment_key(environment):
    """Observed software/hardware identity, excluding time-varying telemetry.

    Timestamp, power, temperature and clocks remain in each raw observation.
    Executable paths may differ while the runtime/package versions match.
    This helper deliberately imports no GPU package.
    """
    required = ("python", "platform", "packages", "cuda_build", "cuda_available", "gpus", "compiler", "nvidia_smi")
    missing = [name for name in required if name not in environment]
    if missing:
        raise ValueError(f"missing environment keys: {missing}")
    python = environment["python"]
    if not environment["gpus"] or not environment["cuda_available"]:
        raise ValueError("paired evidence requires observed CUDA devices")
    if not python.get("version") or not environment["packages"].get("torch"):
        raise ValueError("paired evidence requires Python and Torch versions")
    drivers = [{"index": device.get("index"), "driver_version": device.get("driver_version")}
               for device in environment["nvidia_smi"].get("devices", [])]
    if not drivers or any(not device["driver_version"] for device in drivers):
        raise ValueError("paired evidence requires observed driver versions")
    return {"python": {name: python.get(name) for name in ("version", "implementation", "compiler")},
            "platform": environment["platform"], "packages": environment["packages"],
            "cuda_build": environment["cuda_build"], "gpus": environment["gpus"],
            "compiler": environment["compiler"], "drivers": drivers}


def _backend_packages(packages, backend):
    names = ("triton",) if backend == "triton" else ("tilelang", "apache-tvm-ffi")
    return {name: packages.get(name) for name in names}


def _installed_backend_packages(backend):
    packages = {}
    for name in _backend_packages({}, backend):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def _sdpa_views(case: AttentionCase, inputs: AttentionInputs):
    """BSHD -> BHSD views; the KV head dimension is never expanded."""
    q = inputs.q if case.phase == "prefill" else inputs.q.unsqueeze(1)
    return q.transpose(1, 2), inputs.k.transpose(1, 2), inputs.v.transpose(1, 2)


def _torch_provider(case: AttentionCase, inputs: AttentionInputs, name: str) -> Provider:
    if case.layout != "dense":
        raise UnsupportedProvider("forced Torch SDPA has no native paged-KV public API")
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    backend_name = {"torch_flash": "FLASH_ATTENTION", "torch_cudnn": "CUDNN_ATTENTION",
                    "torch_efficient": "EFFICIENT_ATTENTION"}[name]
    backend = getattr(SDPBackend, backend_name, None)
    if backend is None:
        raise UnsupportedProvider(f"installed Torch has no SDPBackend.{backend_name}")
    q, k, v = _sdpa_views(case, inputs)
    checks = {"torch_flash": "can_use_flash_attention",
              "torch_cudnn": "can_use_cudnn_attention",
              "torch_efficient": "can_use_efficient_attention"}
    checker = getattr(torch.backends.cuda, checks[name], None)
    params_type = getattr(torch.backends.cuda, "SDPAParams", None)
    if checker is not None and params_type is not None:
        params = params_type(q, k, v, None, 0.0, case.causal, case.heads_q != case.heads_kv)
        if not checker(params):
            raise UnsupportedProvider(f"Torch {checks[name]} rejected the exact strided/GQA inputs; "
                                      f"backend={backend_name}, case={case.case_id}")

    def call():
        with sdpa_kernel(backends=[backend]):
            result = scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                                   is_causal=case.causal,
                                                   enable_gqa=case.heads_q != case.heads_kv)
        result = result.transpose(1, 2)
        return result if case.phase == "prefill" else result.squeeze(1)

    return Provider(call, call, {
        "api": "torch.nn.functional.scaled_dot_product_attention",
        "forced_backend": backend_name, "q_layout": "BHSD view of BSHD",
        "kv_layout": "BHSD view of BSHD", "gqa_expansion": False,
        "input_copies": False, "preallocated_output": False,
    })


def _ours_provider(case: AttentionCase, inputs: AttentionInputs,
                   choice: dict[str, Any] | None = None, *, backend: str = "triton") -> Provider:
    _check_backend(backend)
    import torch

    module = importlib.import_module(".attention" if backend == "triton" else ".tilelang", __package__)
    decode, decode_workspace, prefill = module.decode, module.decode_workspace, module.prefill

    choice = choice or {"config": None, "num_splits": None}
    config, splits = choice["config"], choice["num_splits"]
    out = torch.empty_like(inputs.q)
    kwargs = {"kv_lens": inputs.kv_lens, "block_table": inputs.block_table,
              "config": config}
    if case.phase == "prefill":
        def graph_call():
            return prefill(inputs.q, inputs.k, inputs.v, causal=case.causal, out=out, **kwargs)

        def eager_call():
            return prefill(inputs.q, inputs.k, inputs.v, causal=case.causal, **kwargs)
    else:
        workspace = decode_workspace(inputs.q, inputs.k, block_table=inputs.block_table,
                                     num_splits=splits, config=config)

        def graph_call():
            return decode(inputs.q, inputs.k, inputs.v, out=out, workspace=workspace,
                          num_splits=splits, **kwargs)

        def eager_call():
            return decode(inputs.q, inputs.k, inputs.v, num_splits=splits, **kwargs)

    return Provider(graph_call, eager_call, {
        "api": f"{module.__name__}.{case.phase}" if hasattr(module, "__name__") else f"fa5060.{backend}.{case.phase}",
        "backend": backend, "package_versions": _installed_backend_packages(backend),
        "preallocated_output": True,
        "preallocated_workspace": case.phase == "decode", "config": config,
        "num_splits": splits, "native_paged": case.layout == "paged",
        "eager_allocates_output": True, "eager_allocates_workspace": case.phase == "decode",
        "graph_boundary": "complete operator including all decode splits and merge",
    })


def _fa4_provider(case: AttentionCase, inputs: AttentionInputs) -> Provider:
    if case.layout != "dense":
        raise UnsupportedProvider("FA4 flash_attn_func adapter has no verified native page-table API")
    import torch
    from flash_attn.cute.interface import flash_attn_func

    signature = inspect.signature(flash_attn_func)
    parameters = signature.parameters
    if "causal" not in parameters:
        raise UnsupportedProvider("FA4 public flash_attn_func has no causal keyword")
    q = inputs.q if case.phase == "prefill" else inputs.q.unsqueeze(1)
    kwargs = {"causal": case.causal}
    out = torch.empty_like(q) if "out" in parameters else None

    def call(preallocate):
        result = flash_attn_func(q, inputs.k, inputs.v,
                                **({**kwargs, "out": out} if preallocate and out is not None else kwargs))
        if isinstance(result, tuple):
            result = result[0]
        return result if case.phase == "prefill" else result.squeeze(1)

    return Provider(lambda: call(True), lambda: call(False), {
        "api": "flash_attn.cute.interface.flash_attn_func", "signature": str(signature),
        "q_layout": "BSHD", "kv_layout": "BSHD", "input_copies": False,
        "gqa_expansion": False, "preallocated_output": out is not None,
    })


def _flashinfer_workspace_bytes(case: AttentionCase) -> int:
    # Graph-enabled FA2 prefill reserves FP32 partial values AND sums. A
    # fixed 128 MiB fits values but not sums for B1/S8192/H32/D128. Reserve
    # two complete partial-output sets plus padding; small calls keep 128 MiB.
    partial = case.batch * case.query_len * case.heads_q * (case.head_dim + 1) * 4
    return max(128 * 1024**2, 2 * partial + 8 * 1024**2)


def _flashinfer_provider(case: AttentionCase, inputs: AttentionInputs, backend: str) -> Provider:
    """Use FI public native ragged/paged wrappers and SM120 FMHAv2."""
    import torch
    import flashinfer

    workspace = torch.empty(_flashinfer_workspace_bytes(case), device=inputs.q.device, dtype=torch.uint8)
    q = inputs.q.reshape(-1, case.heads_q, case.head_dim)
    out = torch.empty_like(q)
    qo_cpu = torch.arange(case.batch + 1, dtype=torch.int32) * case.query_len
    kv_cpu = torch.arange(case.batch + 1, dtype=torch.int32) * case.kv_len
    qo_gpu, kv_gpu = qo_cpu.to(inputs.q.device), kv_cpu.to(inputs.q.device)
    common = {"num_qo_heads": case.heads_q, "num_kv_heads": case.heads_kv,
              "q_data_type": inputs.q.dtype, "kv_data_type": inputs.k.dtype,
              "sm_scale": case.head_dim ** -.5}
    metadata = {
        "backend": backend, "workspace_bytes": workspace.numel(), "planning_timed": False,
        "input_copies": False, "gqa_expansion": False, "preallocated_output": True,
        "native_paged": case.layout == "paged", "kv_layout": "NHD",
    }
    if backend == "fmha_v2":
        if case.layout == "dense":
            # Standard SM120 kernels exclude SEPARATE_Q_K_V (the Python
            # entry point accepts it without a working D64/128 kernel).
            # The public ragged wrapper uses CONTIGUOUS_Q_KV and stacks KV
            # inside run(). Keep that conversion in BOTH timing boundaries.
            cls = flashinfer.BatchPrefillWithRaggedKVCacheWrapper
            wrapper = cls(workspace, kv_layout="NHD", use_cuda_graph=True,
                          qo_indptr_buf=qo_gpu, kv_indptr_buf=kv_gpu, backend="fmha_v2")
            wrapper.plan(qo_cpu, kv_cpu, head_dim_qk=case.head_dim, causal=case.causal,
                         **common)
            k_view = inputs.k.view(-1, case.heads_kv, case.head_dim)
            v_view = inputs.v.view(-1, case.heads_kv, case.head_dim)

            def call(preallocate):
                result = wrapper.run(q, k_view, v_view, **({"out": out} if preallocate else {}))
                return result.reshape(inputs.q.shape)

            metadata.update(
                api=f"flashinfer.{cls.__name__}.run", input_layout="CONTIGUOUS_Q_KV",
                input_copies=True, packing_timed=True,
                packing="public FlashInfer run() internally torch.stack((k,v), dim=1)",
                constructor_signature=str(inspect.signature(cls)),
                run_signature=str(inspect.signature(wrapper.run)),
                resolved_backend=str(getattr(wrapper, "_backend", "fmha_v2")),
                decode_entry="one-query native FMHAv2" if case.phase == "decode" else None,
            )
            return Provider(lambda: call(True), lambda: call(False), metadata)

        from flashinfer.prefill import trtllm_fmha_v2_prefill

        signature = inspect.signature(trtllm_fmha_v2_prefill)
        kwargs = {
            "workspace_buffer": workspace, "seq_lens": inputs.kv_lens,
            "max_q_len": case.query_len, "max_kv_len": case.kv_len,
            "bmm1_scale": case.head_dim ** -.5, "bmm2_scale": 1.0,
            "batch_size": case.batch, "cum_seq_lens_q": qo_gpu,
            "cum_seq_lens_kv": kv_gpu, "mask_mode": "causal" if case.causal else "padding",
        }
        if inputs.k.untyped_storage().data_ptr() != inputs.v.untyped_storage().data_ptr():
            raise UnsupportedProvider("FMHAv2 separate native pages require shared K/V allocation")
        offsets = torch.stack((inputs.block_table,
                               inputs.block_table + inputs.k.shape[0]), dim=1)
        args = (q, (inputs.k, inputs.v))
        kwargs.update(input_layout="Q_PAGED_KV_NHD", block_tables=offsets)
        metadata["block_table_format"] = "[B,2,M] native K/V block offsets; static preprocessing"

        def call(preallocate):
            result = trtllm_fmha_v2_prefill(args, **kwargs,
                                           **({"out": out} if preallocate else {}))
            if isinstance(result, tuple):
                result = result[0]
            return result.reshape(inputs.q.shape)

        metadata.update(api="flashinfer.prefill.trtllm_fmha_v2_prefill", signature=str(signature),
                        input_layout=kwargs["input_layout"],
                        decode_entry="one-query native FMHAv2" if case.phase == "decode" else None)
        return Provider(lambda: call(True), lambda: call(False), metadata)

    if backend != "fa2":
        raise UnsupportedProvider(f"no explicit adapter for FlashInfer backend {backend}")
    if case.phase == "prefill" and case.layout == "dense":
        cls = flashinfer.BatchPrefillWithRaggedKVCacheWrapper
        wrapper = cls(workspace, kv_layout="NHD", use_cuda_graph=True,
                      qo_indptr_buf=qo_gpu, kv_indptr_buf=kv_gpu, backend="fa2")
        wrapper.plan(qo_cpu, kv_cpu, head_dim_qk=case.head_dim, causal=case.causal,
                     uniform_q_len=case.query_len, **common)
        k = inputs.k.reshape(-1, case.heads_kv, case.head_dim)
        v = inputs.v.reshape(-1, case.heads_kv, case.head_dim)

        def call(preallocate):
            result = wrapper.run(q, k, v, **({"out": out} if preallocate else {}))
            return result.reshape(inputs.q.shape)
    else:
        pages_per_sequence = (case.kv_len + case.page_size - 1) // case.page_size
        page_indptr_cpu = torch.arange(case.batch + 1, dtype=torch.int32) * pages_per_sequence
        page_indptr = page_indptr_cpu.to(inputs.q.device)
        last_cpu = torch.full((case.batch,), (case.kv_len - 1) % case.page_size + 1,
                              dtype=torch.int32)
        last = last_cpu.to(inputs.q.device)
        if case.layout == "paged":
            indices = inputs.block_table.reshape(-1)
            k, v = inputs.k, inputs.v
        else:
            if case.kv_len % case.page_size:
                raise UnsupportedProvider("zero-copy dense decode page view requires page-aligned length")
            indices = torch.arange(case.batch * pages_per_sequence, dtype=torch.int32,
                                   device=inputs.q.device)
            k = inputs.k.view(-1, case.page_size, case.heads_kv, case.head_dim)
            v = inputs.v.view(-1, case.page_size, case.heads_kv, case.head_dim)
            metadata["dense_storage_view"] = "zero-copy page16 view for native decode wrapper"
        if case.phase == "prefill":
            cls = flashinfer.BatchPrefillWithPagedKVCacheWrapper
            wrapper = cls(workspace, kv_layout="NHD", use_cuda_graph=True, backend="fa2",
                          qo_indptr_buf=qo_gpu, paged_kv_indptr_buf=page_indptr,
                          paged_kv_indices_buf=indices, paged_kv_last_page_len_buf=last)
            wrapper.plan(qo_cpu, page_indptr_cpu, indices, last_cpu,
                         head_dim_qk=case.head_dim, page_size=case.page_size,
                         causal=case.causal, uniform_q_len=case.query_len, **common)
        else:
            cls = flashinfer.BatchDecodeWithPagedKVCacheWrapper
            wrapper = cls(workspace, kv_layout="NHD", use_cuda_graph=True, backend="fa2",
                          use_tensor_cores=True, paged_kv_indptr_buffer=page_indptr,
                          paged_kv_indices_buffer=indices, paged_kv_last_page_len_buffer=last)
            wrapper.plan(page_indptr_cpu, indices, last_cpu, head_dim=case.head_dim,
                         page_size=case.page_size, **common)

        def call(preallocate):
            result = wrapper.run(q, (k, v), **({"out": out} if preallocate else {}))
            return result.reshape(inputs.q.shape)

    metadata.update(api=f"flashinfer.{cls.__name__}.run",
                    constructor_signature=str(inspect.signature(cls)),
                    run_signature=str(inspect.signature(wrapper.run)),
                    resolved_backend=str(getattr(wrapper, "_backend", "fa2")))
    return Provider(lambda: call(True), lambda: call(False), metadata)


def _provider(case: AttentionCase, inputs: AttentionInputs, name: str,
              choice: dict[str, Any] | None = None, *, backend: str = "triton") -> Provider:
    if name == "ours":
        return _ours_provider(case, inputs, choice, backend=backend)
    if name.startswith("torch_"):
        return _torch_provider(case, inputs, name)
    if name == "fa4":
        return _fa4_provider(case, inputs)
    return _flashinfer_provider(case, inputs, name.removeprefix("flashinfer_"))


def _validate_output(actual, expected, dtype: str) -> dict[str, Any]:
    """Chunked fp32 checks keep validation memory bounded for long prefill."""
    import torch

    if not isinstance(actual, torch.Tensor) or tuple(actual.shape) != tuple(expected.shape):
        raise CorrectnessError(f"output shape {getattr(actual, 'shape', None)} != {tuple(expected.shape)}")
    atol, rtol = (.003, .003) if dtype == "fp16" else (.02, .02)
    actual_flat, expected_flat = actual.reshape(-1), expected.reshape(-1)
    max_abs, max_relative, valid = 0.0, 0.0, True
    squared_error, squared_reference = 0.0, 0.0
    for start in range(0, expected.numel(), 1 << 20):
        got = actual_flat[start:start + (1 << 20)].float()
        want = expected_flat[start:start + (1 << 20)].float()
        if not bool(torch.isfinite(got).all()) or not bool(torch.isfinite(want).all()):
            raise CorrectnessError("provider or reference contains nonfinite output")
        difference = (got - want).abs()
        max_abs = max(max_abs, difference.max().item() if difference.numel() else 0.0)
        max_relative = max(max_relative, (difference / want.abs().clamp_min(atol)).max().item()
                           if difference.numel() else 0.0)
        squared_error += difference.square().sum().item()
        squared_reference += want.square().sum().item()
        valid = valid and bool(torch.all(difference <= atol + rtol * want.abs()))
    normalized_rms = math.sqrt(squared_error / max(squared_reference, 1e-30))
    rms_tolerance = .02 if dtype == "fp16" else .05
    if not valid or normalized_rms > rms_tolerance:
        raise CorrectnessError(f"reference mismatch: max_abs={max_abs:.8g}, "
                               f"max_relative_atol_floor={max_relative:.8g}, "
                               f"normalized_rms={normalized_rms:.8g}, "
                               f"atol={atol}, rtol={rtol}, rms_tolerance={rms_tolerance}")
    return {"validated": True, "max_abs_error": max_abs,
            "max_relative_error_atol_floor": max_relative, "atol": atol, "rtol": rtol,
            "normalized_rms_error": normalized_rms, "normalized_rms_tolerance": rms_tolerance}


def _event_round(call: Callable, count: int) -> tuple[float, float]:
    import torch

    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter_ns()
    start.record()
    for _ in range(count):
        call()
    stop.record()
    stop.synchronize()
    wall_us = (time.perf_counter_ns() - wall_start) / 1000 / count
    return start.elapsed_time(stop) * 1000 / count, wall_us


def _graph_batch_size(trial_us: float) -> int:
    """Target 10 ms, amortize short-call host gaps, and cap graph size."""
    count = math.ceil(10000 / max(trial_us, .001))
    if trial_us <= 100:
        count = max(128, count)
    return max(1, min(2048, count))


def _time_provider(provider: Provider, rounds: int, reference, dtype: str) -> dict[str, Any]:
    """Capture and replay the complete operator; separately time its eager API."""
    import torch

    torch.cuda.synchronize()
    warmup_start = time.perf_counter_ns()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            provider.graph_call()
            provider.eager_call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    warmup_wall_ms = (time.perf_counter_ns() - warmup_start) / 1e6
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_output = provider.graph_call()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    _validate_output(captured_output, reference, dtype)
    graph_trial, _ = _event_round(graph.replay, 3)
    eager_trial, _ = _event_round(provider.eager_call, 3)
    graph_count = _graph_batch_size(graph_trial)
    eager_count = max(1, min(10000, math.ceil(10000 / max(eager_trial, .001))))
    del graph, captured_output
    # Capture the complete operator repeatedly in ONE graph. Replaying a
    # one-operator graph in a Python loop can starve a microsecond kernel and
    # report host-submission gaps as if they were GPU execution time.
    graph = torch.cuda.CUDAGraph()
    captured_output = None
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(graph_count):
            # Release prior allocation-only API outputs before the next call;
            # graph-pool storage can then be reused in stream order.
            captured_output = None
            captured_output = provider.graph_call()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    _validate_output(captured_output, reference, dtype)
    graph_samples, eager_samples, wall_samples = [], [], []
    for round_index in range(rounds):
        # Alternate the two measurement modes to limit fixed thermal ordering.
        if round_index % 2:
            eager_us, wall_us = _event_round(provider.eager_call, eager_count)
            graph_us, _ = _event_round(graph.replay, 1)
        else:
            graph_us, _ = _event_round(graph.replay, 1)
            eager_us, wall_us = _event_round(provider.eager_call, eager_count)
        graph_samples.append(graph_us / graph_count)
        eager_samples.append(eager_us)
        wall_samples.append(wall_us)
    # Keep the captured output alive until the final replay has completed.
    del captured_output
    return {"gpu_us_p50": statistics.median(graph_samples),
            "warmup_wall_ms": warmup_wall_ms, "warmup_calls_per_api": 3,
            "eager_us_p50": statistics.median(eager_samples),
            "eager_wall_us_p50": statistics.median(wall_samples),
            "gpu_us_rounds": graph_samples, "eager_us_rounds": eager_samples,
            "eager_wall_us_rounds": wall_samples,
            "graph_replays_per_round": 1, "graph_operator_calls_per_replay": graph_count,
            "eager_calls_per_round": eager_count,
            "graph_timing": "single replay of repeated complete-operator capture, divided by call count"}


def _failure(exc: Exception, stage: str, metadata=None) -> dict[str, Any]:
    if isinstance(exc, CorrectnessError):
        status = "incorrect"
    elif isinstance(exc, (UnsupportedProvider, ImportError, ModuleNotFoundError, NotImplementedError)):
        status = "unsupported"
    else:
        status = "error"
    return {"status": status, "validated": False, "stage": stage,
            "reason": f"{type(exc).__name__}: {exc}"[:6000],
            "gpu_us_p50": None, "eager_us_p50": None, "metadata": metadata or {}}


def _first_call(call):
    """Observe synchronized first-call wall cost, including any cold JIT."""
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    output = call()
    torch.cuda.synchronize()
    return output, (time.perf_counter_ns() - start) / 1e6


def _run_provider(case, inputs, reference, name, rounds, choice=None, *, backend="triton"):
    provider, stage = None, "prepare"
    startup = {"first_call_scope": "synchronized wall time; includes JIT on cache miss; excluded from steady-state samples"}
    try:
        prepare_start = time.perf_counter_ns()
        provider = _provider(case, inputs, name, choice, backend=backend)
        startup["prepare_wall_ms"] = (time.perf_counter_ns() - prepare_start) / 1e6
        stage = "correctness"
        actual, startup["graph_first_call_wall_ms"] = _first_call(provider.graph_call)
        validation = _validate_output(actual, reference, case.dtype)
        del actual
        # Allocating public APIs may have a different path: check both paths.
        actual, startup["eager_first_call_wall_ms"] = _first_call(provider.eager_call)
        _validate_output(actual, reference, case.dtype)
        del actual
        stage = "timing"
        times = _time_provider(provider, rounds, reference, case.dtype)
        return {"status": "ok", **validation, **times, "startup": startup, "metadata": provider.metadata}
    except Exception as exc:
        return {**_failure(exc, stage, provider.metadata if provider else None), "startup": startup}


def _run_ours(case, inputs, reference, rounds, tune, *, backend="triton"):
    if not tune:
        result = _run_provider(case, inputs, reference, "ours", rounds, backend=backend)
        result["selected_config"] = {"config": None, "num_splits": None}
        result["configuration_mode"] = "public_default"
        return result
    trials = []
    for choice in tuning_candidates(case, backend=backend):
        result = _run_provider(case, inputs, reference, "ours", 1, choice, backend=backend)
        trials.append({"choice": choice, **result})
    valid = [trial for trial in trials if _valid_timing(trial)]
    if valid:
        winner = min(valid, key=lambda trial: trial["gpu_us_p50"])
        result = _run_provider(case, inputs, reference, "ours", rounds, winner["choice"], backend=backend)
        result["selected_config"] = winner["choice"]
    else:
        result = _failure(RuntimeError("all disclosed tuning candidates failed"), "tuning")
        result["selected_config"] = None
    result["tuning"] = {
        "backend": backend,
        "strategy": "offline per-case fixed-grid exploration followed by separate measurement",
        "selection_metric": "minimum validated trial gpu_us_p50", "trial_rounds": 1,
        "candidates": trials,
        "correctness_failures": sum(trial["status"] == "incorrect" for trial in trials),
    }
    result["configuration_mode"] = "offline_tuned"
    return result


def benchmark_case(case: AttentionCase, *, rounds: int = 5, tune: bool = False,
                   backend: str = "triton") -> dict[str, Any]:
    """Validate against blockwise fp32 reference before timing any provider."""
    _check_backend(backend)
    record = {**case.to_dict(), "backend": backend, "seed": 20261102, "providers": {}}
    try:
        import torch
        from .reference import attention_reference

        if not torch.cuda.is_available():
            raise UnsupportedProvider("torch.cuda.is_available() is false")
        with torch.inference_mode():
            inputs = make_inputs(case)
            # Reference matmuls must retain fp32 accuracy rather than TF32.
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                reference = attention_reference(inputs.q, inputs.k, inputs.v, causal=case.causal,
                                                kv_lens=inputs.kv_lens,
                                                block_table=inputs.block_table)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.cuda.synchronize()
            names = list(PROVIDER_NAMES)
            # One provider's graph/workspace is alive at a time. The random
            # order is deterministic, disclosed, and spreads thermal ordering.
            random.Random(20261102 + sum(map(ord, case.case_id))).shuffle(names)
            record["provider_execution_order"] = names
            record["reference"] = {"api": "fa5060.reference.attention_reference",
                                   "algorithm": "blockwise fp32; no full Sq*Sk scores",
                                   "tf32": False}
            for name in names:
                if name == "ours":
                    result = _run_ours(case, inputs, reference, rounds, tune, backend=backend)
                else:
                    result = _run_provider(case, inputs, reference, name, rounds)
                record["providers"][name] = result
    except Exception as exc:
        for name in PROVIDER_NAMES:
            record["providers"].setdefault(name, _failure(exc, "inputs_or_reference"))
    return record


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Replace a checkpoint only after strict JSON serialization and fsync."""
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _valid_timing(provider: dict[str, Any]) -> bool:
    timing = provider.get("gpu_us_p50")
    return (provider.get("status") == "ok" and provider.get("validated") is True
            and isinstance(timing, (int, float)) and not isinstance(timing, bool)
            and math.isfinite(timing) and timing > 0)


def _baseline_groups(layout: str) -> dict[str, tuple[str, ...]]:
    groups = {"flashinfer_native": ("flashinfer_fa2", "flashinfer_fmha_v2")}
    if layout == "dense":
        groups = {"torch_forced_sdpa": ("torch_flash", "torch_cudnn", "torch_efficient"),
                  "fa4_native": ("fa4",), **groups}
    return groups


def evaluate_case(record: dict[str, Any]) -> dict[str, Any]:
    """Choose the fastest correct native baseline; missing providers never win."""
    providers = record.get("providers", {})
    layout = record.get("layout")
    groups = _baseline_groups(layout)
    covered = {name: any(_valid_timing(providers.get(p, {})) for p in names)
               for name, names in groups.items()}
    eligible_names = {p for names in groups.values() for p in names}
    baselines = {name: p["gpu_us_p50"] for name, p in providers.items()
                 if name in eligible_names and _valid_timing(p)}
    fastest = min(baselines, key=baselines.get) if baselines else None
    ours_valid = _valid_timing(providers.get("ours", {}))
    failures = [name for name, p in providers.items()
                if p.get("status") == "incorrect" or p.get("tuning", {}).get("correctness_failures", 0)]
    missing_providers = sorted(set(PROVIDER_NAMES) - set(providers))
    errors = [name for name, p in providers.items() if p.get("status") == "error"]
    invalid_records = [name for name, p in providers.items()
                       if p.get("status") not in ("ok", "unsupported", "error", "incorrect")
                       or (p.get("status") == "ok" and not _valid_timing(p))
                       or (p.get("status") == "unsupported" and not p.get("reason"))]
    all_covered = (bool(covered) and all(covered.values())
                   and not (missing_providers or errors or invalid_records))
    return {
        "fastest_baseline": fastest,
        "speedup_vs_fastest": (baselines[fastest] / providers["ours"]["gpu_us_p50"]
                               if fastest is not None and ours_valid else None),
        "mandatory_baseline_groups": {name: list(names) for name, names in groups.items()},
        "baseline_coverage": covered,
        "mandatory_baselines_covered": all_covered,
        "correctness_passed": ours_valid and not failures,
        "incorrect_providers": failures,
        "missing_provider_records": missing_providers,
        "errored_providers": errors,
        "invalid_provider_records": invalid_records,
    }


def summarize_results(records: list[dict[str, Any]], expected_cases,
                      *, suite: str = "full") -> dict[str, Any]:
    """Compute phase ratios without dropping absent, duplicated or invalid cases."""
    expected_cases = tuple(expected_cases)
    expected_ids = {case.case_id for case in expected_cases}
    seen_ids = [record.get("id") for record in records]
    unexpected = [value for value in seen_ids if value not in expected_ids]
    summary = {"suite": suite, "unexpected_cases": unexpected,
               "thresholds": {"geometric_mean_speedup": 1.10, "worst_speedup": .95}}
    for phase in ("prefill", "decode"):
        expected = [case for case in expected_cases if case.phase == phase]
        phase_ids = {case.case_id for case in expected}
        selected = [r for r in records if r.get("id") in phase_ids]
        evaluated = [(r, evaluate_case(r)) for r in selected]
        missing = sorted(phase_ids - set(seen_ids))
        duplicates = sorted({value for value in phase_ids if seen_ids.count(value) > 1})
        malformed = []
        expected_by_id = {case.case_id: case for case in expected}
        for record in selected:
            try:
                if AttentionCase.from_dict(record) != expected_by_id[record["id"]]:
                    malformed.append(record.get("id"))
            except (ValueError, KeyError, TypeError):
                malformed.append(record.get("id"))
        uncovered = [r.get("id") for r, e in evaluated
                     if not e["mandatory_baselines_covered"] or not e["correctness_passed"]]
        uncovered += malformed
        ratios = [e["speedup_vs_fastest"] for _, e in evaluated
                  if e["speedup_vs_fastest"] is not None]
        complete = bool(expected) and not (missing or duplicates or uncovered or unexpected)
        complete = complete and len(ratios) == len(expected)
        geometric = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios)) if ratios else None
        worst = min(ratios) if ratios else None
        summary[phase] = {
            "expected_cases": len(expected), "completed_cases": len(selected),
            "comparable_cases": len(ratios), "missing_cases": missing,
            "duplicate_cases": duplicates, "uncovered_or_incorrect_cases": uncovered,
            "coverage_complete": complete, "geometric_mean_speedup": geometric,
            "worst_speedup": worst,
            "target_met": bool(suite == "full" and complete and geometric >= 1.10 and worst >= .95),
        }
    active = [phase for phase in ("prefill", "decode") if summary[phase]["expected_cases"]]
    summary["target_met"] = bool(active) and all(summary[phase]["target_met"] for phase in active)
    return summary


def tuning_candidates(case: AttentionCase, *, backend: str = "triton") -> tuple[dict[str, Any], ...]:
    """Small, disclosed offline search; every attempt remains in the result."""
    _check_backend(backend)
    candidates = [{"config": None, "num_splits": None}]
    if case.phase == "prefill":
        for block_m, block_n, stages in ((32, 64, 1), (64, 64, 2), (32, 128, 2), (64, 128, 2)):
            candidates.append({"config": {"block_m": block_m, "block_n": block_n,
                                          "num_warps": 4, "num_stages": stages,
                                          "loader": "pointer"}, "num_splits": None})
        if case.layout == "dense":
            candidates.append({"config": {"block_m": 64, "block_n": 64,
                                          "num_warps": 4, "num_stages": 2,
                                          "loader": "tma"}, "num_splits": None})
    else:
        for splits in (1, 4, 16):
            for mode in ("vector", "grouped"):
                candidates.append({"config": {"block_n": 64, "num_warps": 4,
                                              "num_stages": 1, "mode": mode},
                                   "num_splits": splits})
    return tuple(candidates)


def _same_json(left, right):
    """Compare serialized contracts without Python's bool/int/float coercion."""
    return (json.dumps(left, sort_keys=True, allow_nan=False)
            == json.dumps(right, sort_keys=True, allow_nan=False))


def _resume_report(source, output, expected, cases):
    """Read and validate an immutable checkpoint before touching the new output."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists() or output.is_symlink():
        raise ValueError("resume output must be a different, new path")
    frozen = full_suite() if expected["suite"] == "full" else smoke_suite()
    manifests = (frozen, *(tuple(case for case in frozen if case.phase == phase)
                           for phase in ("prefill", "decode")))
    if expected["suite"] not in ("full", "smoke") or cases not in manifests:
        raise ValueError("resume requires the frozen suite order (all cases or one phase)")
    snapshot = source.read_bytes()
    try:
        report = json.loads(snapshot)
        # Strict serialization also rejects nonfinite JSON constants accepted by json.loads.
        json.dumps(report, allow_nan=False)
        for name in ("schema_version", "backend", "backend_packages", "configuration_mode",
                     "suite", "rounds", "tune", "seed", "timing"):
            if name not in report or not _same_json(report[name], expected[name]):
                raise ValueError(f"resume run contract differs: {name}")
        fresh_key = environment_key(expected["environment"])
        if not _same_json(environment_key(report["environment"]), fresh_key):
            raise ValueError("resume software/hardware environment keys differ")
        # A later interruption may itself be resumed; retain the earlier segment's provenance.
        if "resume" in report and not _same_json(environment_key(report["resume"]["environment"]), fresh_key):
            raise ValueError("resume segment software/hardware environment keys differ")
        records = report["cases"]
        if not isinstance(records, list):
            raise ValueError("resume cases must be a list")
        if len(records) >= len(cases):
            raise ValueError("resume checkpoint is complete or has excess cases")
        seen = set()
        for case, record in zip(cases, records):
            identity = case.to_dict()
            if record["id"] in seen:
                raise ValueError(f"duplicate resume case ID: {record['id']}")
            seen.add(record["id"])
            if any(name not in record or not _same_json(record[name], value) for name, value in identity.items()):
                raise ValueError(f"resume case order/shape differs: {case.case_id}")
            if (not _same_json(record.get("seed"), expected["seed"])
                    or record.get("backend") != expected["backend"]):
                raise ValueError(f"resume case seed/backend differs: {case.case_id}")
            providers = record["providers"]
            if (not isinstance(providers, dict) or set(providers) != set(PROVIDER_NAMES)
                    or any(not isinstance(provider, dict) for provider in providers.values())):
                raise ValueError(f"incomplete resume provider records: {case.case_id}")
            if evaluate_case(record)["invalid_provider_records"]:
                raise ValueError(f"invalid resume provider records: {case.case_id}")
        if report["summary"] != summarize_results(records, cases, suite=expected["suite"]):
            raise ValueError("resume summary disagrees with the frozen manifest or completed rows")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("malformed resume checkpoint") from exc
    provenance = {
        "source_path": str(source), "source_sha256": hashlib.sha256(snapshot).hexdigest(),
        "initial_case_count": len(records), "environment": expected["environment"], "segmented": True,
    }
    if "resume" in report:
        provenance["previous"] = report["resume"]
    report["resume"] = provenance
    return report


def run_benchmark(cases, *, output: str | Path, environment: dict[str, Any],
                  suite: str = "smoke", rounds: int = 5, tune: bool = False,
                  case_runner: Callable | None = None, backend: str = "triton",
                  resume_from: str | Path | None = None) -> dict[str, Any]:
    """Checkpoint each completed case; optionally continue into a new output file."""
    _check_backend(backend)
    if rounds < 1:
        raise ValueError("rounds must be positive")
    cases = tuple(cases)
    runner = benchmark_case if case_runner is None else case_runner
    parameters = inspect.signature(runner).parameters
    accepts_backend = ("backend" in parameters or
                       any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()))
    if not accepts_backend and backend != "triton":
        raise ValueError("a custom case_runner must accept backend for a TileLang report")
    report = {
        "schema_version": SCHEMA_VERSION, "environment": environment,
        "backend": backend, "backend_packages": _backend_packages(environment.get("packages", {}), backend),
        "configuration_mode": "offline_tuned" if tune else "public_default",
        "suite": suite, "rounds": rounds, "tune": tune, "seed": 20261102,
        "timing": {
            "gpu_us_p50": "median of per-round CUDA-event single repeated-operator graph replay / batch count",
            "eager_us_p50": "median of per-round public eager API CUDA-event means",
            "eager_wall_us_p50": "median of per-round synchronized public eager API wall means",
            "round_target_ms": 10, "warmup_calls": 3,
            "graph_batch_cap": 2048, "short_operator_min_graph_batch": 128,
            "graph_boundary": "complete operator including all decode splits and merge; planning outside",
            "eager_boundary": "public operator including output/workspace allocation where API permits",
            "startup": "provider preparation, first calls including JIT, and warmup saved separately; excluded from latency samples",
        },
        "cases": [], "summary": summarize_results([], cases, suite=suite),
    }
    if resume_from is not None:
        report = _resume_report(resume_from, output, report, cases)
        # Reserve a new destination exclusively; a concurrent creator must not be overwritten.
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        try:
            with Path(output).open("x", encoding="utf-8"):
                pass
        except FileExistsError as exc:
            raise ValueError("resume output must be a new path") from exc
    atomic_write_json(output, report)
    for case in cases[len(report["cases"]):]:
        kwargs = {"rounds": rounds, "tune": tune}
        if accepts_backend:
            kwargs["backend"] = backend
        record = runner(case, **kwargs)
        if record.get("backend", backend) != backend:
            raise ValueError("case_runner returned a different backend")
        record["backend"] = backend
        record.update(evaluate_case(record))
        report["cases"].append(record)
        report["summary"] = summarize_results(report["cases"], cases, suite=suite)
        atomic_write_json(output, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--backend", choices=BACKENDS, default="triton")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path,
                        help="continue a compatible checkpoint into a different, new output file")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--tune", action="store_true",
                        help="disclosed offline per-case fixed-grid search; keeps failed candidates")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    from .probe import collect_environment

    cases = full_suite() if args.suite == "full" else smoke_suite()
    cases = tuple(case for case in cases if args.phase == "all" or case.phase == args.phase)
    case_numbers = {case.case_id: index + 1 for index, case in enumerate(cases)}

    def runner(case, *, rounds, tune, backend):
        result = benchmark_case(case, rounds=rounds, tune=tune, backend=backend)
        states = ", ".join(f"{name}={provider['status']}" for name, provider in result["providers"].items())
        print(f"[{case_numbers[case.case_id]}/{len(cases)}] {case.case_id}: {states}",
              file=sys.stderr, flush=True)
        return result

    report = run_benchmark(cases, output=args.output, environment=collect_environment(),
                           suite=args.suite, rounds=args.rounds, tune=args.tune, case_runner=runner,
                           backend=args.backend, resume_from=args.resume_from)
    print(json.dumps({"output": str(args.output), "summary": report["summary"]},
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
