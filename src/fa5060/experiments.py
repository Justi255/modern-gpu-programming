"""Small, reproducible teaching experiments; GPU work starts only in run().

These are learning workloads, not an exhaustive performance leaderboard.
Every candidate is checked against an independent numerical reference before
its latency is sampled.  Imports and ``--help`` remain usable without torch.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from statistics import median
import sys

from .probe import _write_json, collect_environment

SEED = 20261102
REPETITIONS = 10
WARMUP = 3


def _require_cuda():
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch and a CUDA GPU are required for experiments") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("A visible CUDA GPU is required for experiments")
    return torch


def _time_cuda(torch, operation, repetitions=REPETITIONS, warmup=WARMUP):
    """Median event time of eager calls, after compilation and warmup.

    Allocation in a public wrapper remains part of the call.  Attention
    outputs/workspaces are preallocated explicitly; this scope is saved in
    the record.  This helper does not claim isolated instruction latency.
    """
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repetitions)
    ]
    samples = []
    for start, end in events:
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del result
    return {
        "clock": "CUDA events on the current stream", "mode": "eager",
        "warmup": warmup, "repetitions": repetitions,
        "summary": "median", "median_ms": float(median(samples)),
        "samples_ms": samples,
    }


def _check(torch, actual, expected, *, rtol, atol):
    if not bool(torch.isfinite(actual).all().item()):
        raise AssertionError("candidate contains NaN or Inf")
    if not bool(torch.isfinite(expected).all().item()):
        raise AssertionError("reference contains NaN or Inf")
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)
    difference = actual.float() - expected.float()
    maximum = float(difference.abs().max().item()) if difference.numel() else 0.0
    numerator = float(difference.square().mean().sqrt().item()) if difference.numel() else 0.0
    denominator = float(expected.float().square().mean().sqrt().item()) if expected.numel() else 0.0
    return {"passed": True, "rtol": rtol, "atol": atol, "max_error": maximum,
            "rms_relative_error": numerator / max(denominator, 1e-12)}


def _record(torch, environment, *, name, inputs, operation, expected, rtol, atol,
            reference, baseline=None, timing_scope="public call; output allocation included"):
    actual = operation()
    correctness = _check(torch, actual, expected, rtol=rtol, atol=atol)
    baseline_correctness = None
    if baseline is not None:
        baseline_correctness = _check(torch, baseline(), expected, rtol=rtol, atol=atol)
    # Validation and .item() synchronization above occur before any timer starts.
    timing = _time_cuda(torch, operation)
    baseline_timing = _time_cuda(torch, baseline) if baseline is not None else None
    packages = environment["packages"]
    return {
        "name": name, "inputs": inputs,
        "provider": {"name": "fa5060/Triton",
                     "version": packages.get("modern-gpu-programming-5060"),
                     "triton_version": packages.get("triton")},
        "reference": {"name": reference, "torch_version": packages.get("torch")},
        "correctness": correctness, "max_error": correctness["max_error"],
        "latency_ms": timing["median_ms"], "timing": timing,
        "timing_scope": timing_scope,
        "baseline_provider": ({"name": "PyTorch", "version": packages.get("torch")}
                              if baseline is not None else None),
        "baseline_correctness": baseline_correctness,
        "baseline_latency_ms": baseline_timing["median_ms"] if baseline_timing else None,
        "baseline_timing": baseline_timing,
    }


def _basics(torch, environment):
    from .fundamentals import row_softmax, transpose, vector_add

    records = []
    for length in (100003, 1 << 20):
        x = torch.randn(length, device="cuda", dtype=torch.float32)
        y = torch.randn_like(x)
        records.append(_record(
            torch, environment, name="vector_add", inputs={"length": length, "dtype": "float32"},
            operation=lambda: vector_add(x, y), expected=x + y, rtol=0, atol=0,
            reference="torch.add float32", baseline=lambda: x + y,
        ))
    for rows, cols in ((17, 781), (128, 1024)):
        # Strided outer rows exercise stride handling as well as the tail mask.
        storage = torch.randn((rows * 2, cols), device="cuda", dtype=torch.float32) * 4
        x = storage[::2]
        records.append(_record(
            torch, environment, name="row_softmax",
            inputs={"shape": list(x.shape), "stride": list(x.stride()), "dtype": "float32"},
            operation=lambda: row_softmax(x), expected=torch.softmax(x, dim=-1),
            rtol=2e-5, atol=2e-6, reference="torch.softmax float32",
            baseline=lambda: torch.softmax(x, dim=-1),
        ))
    for rows, cols in ((37, 53), (1024, 1024)):
        x = torch.randn((rows * 2, cols), device="cuda", dtype=torch.float32)[::2]
        records.append(_record(
            torch, environment, name="transpose",
            inputs={"shape": list(x.shape), "stride": list(x.stride()), "dtype": "float32"},
            operation=lambda: transpose(x), expected=x.T.contiguous(), rtol=0, atol=0,
            reference="torch.Tensor.T.contiguous", baseline=lambda: x.T.contiguous(),
        ))
    return records


def _gemm(torch, environment):
    from .fundamentals import matmul

    records = []
    for dtype in (torch.float16, torch.bfloat16):
        for m, n, k in ((65, 79, 97), (512, 512, 512)):
            a = torch.randn((m, k), device="cuda", dtype=dtype)
            b = torch.randn((k, n), device="cuda", dtype=dtype)
            expected = a.float() @ b.float()
            tolerance = 2e-2 if dtype == torch.float16 else 5e-2
            for variant in ("tiled", "pipelined", "persistent"):
                records.append(_record(
                    torch, environment, name=f"matmul/{variant}",
                    inputs={"m": m, "n": n, "k": k, "dtype": str(dtype).removeprefix("torch.")},
                    operation=lambda variant=variant: matmul(a, b, variant=variant),
                    expected=expected, rtol=tolerance, atol=tolerance,
                    reference="torch.matmul fp32 inputs, float32_matmul_precision=highest",
                    baseline=lambda: torch.matmul(a, b),
                ))
    return records


def _attention(torch, environment):
    from .attention import decode, decode_workspace, prefill
    from .reference import attention_reference

    records = []
    batch, queries, hq, hkv, capacity, page = 3, 17, 4, 2, 80, 16
    lengths = [79, 53, 0]
    kv_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    pages_per_sequence = capacity // page
    for dtype, dimension in ((torch.float16, 64), (torch.bfloat16, 128)):
        q = torch.randn((batch, queries, hq, dimension), device="cuda", dtype=dtype)
        dense_k = torch.randn((batch, capacity, hkv, dimension), device="cuda", dtype=dtype)
        dense_v = torch.randn_like(dense_k)
        table = torch.randperm(batch * pages_per_sequence, device="cuda", dtype=torch.int64)
        table = table.reshape(batch, pages_per_sequence).to(torch.int32)
        paged_k = torch.empty((batch * pages_per_sequence, page, hkv, dimension), device="cuda", dtype=dtype)
        paged_v = torch.empty_like(paged_k)
        # Pack once outside timing.  Nonidentity pages catch accidental dense addressing.
        physical = table.to(torch.long).reshape(-1)
        paged_k[physical] = dense_k.reshape(-1, page, hkv, dimension)
        paged_v[physical] = dense_v.reshape(-1, page, hkv, dimension)
        tolerance = 1e-2 if dtype == torch.float16 else 3e-2
        for layout in ("dense", "paged"):
            k, v = (dense_k, dense_v) if layout == "dense" else (paged_k, paged_v)
            block_table = None if layout == "dense" else table
            saved_k, saved_v = k.clone(), v.clone()
            for phase in ("prefill", "decode"):
                query = q if phase == "prefill" else q[:, -1].contiguous()
                output = torch.empty_like(query)
                expected = attention_reference(
                    query, k, v, causal=(phase == "prefill"), kv_lens=kv_lens,
                    block_table=block_table,
                )
                if phase == "prefill":
                    operation = lambda: prefill(
                        query, k, v, causal=True, kv_lens=kv_lens,
                        block_table=block_table, out=output,
                    )
                else:
                    workspace = decode_workspace(query, k, block_table=block_table, num_splits=2)
                    operation = lambda: decode(
                        query, k, v, kv_lens=kv_lens, block_table=block_table,
                        out=output, workspace=workspace, num_splits=2,
                    )
                records.append(_record(
                    torch, environment, name=f"attention/{phase}/{layout}",
                    inputs={"q_shape": list(query.shape), "kv_shape": list(k.shape),
                            "dtype": str(dtype).removeprefix("torch."),
                            "layout": layout, "kv_lens": lengths,
                            "page_size": page if layout == "paged" else None,
                            "block_table_shape": list(table.shape) if layout == "paged" else None,
                            "causal": phase == "prefill", "num_splits": 2 if phase == "decode" else None},
                    operation=operation, expected=expected, rtol=tolerance, atol=tolerance,
                    reference="fa5060.reference.attention_reference blockwise fp32",
                    timing_scope="eager attention call; preallocated output and decode workspace; no cache update",
                ))
                torch.testing.assert_close(k, saved_k, rtol=0, atol=0)
                torch.testing.assert_close(v, saved_v, rtol=0, atol=0)
    return records


def run(suite):
    """Run one suite and return only measured, numerically validated records."""
    if suite not in {"basics", "gemm", "attention"}:
        raise ValueError("suite must be basics, gemm, or attention")
    torch = _require_cuda()
    environment = collect_environment()
    torch.manual_seed(SEED)
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad():
            runner = {"basics": _basics, "gemm": _gemm, "attention": _attention}[suite]
            records = runner(torch, environment)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    return {
        "schema_version": 1, "suite": suite, "seed": SEED,
        "environment": environment, "records": records,
        "notes": [
            "Numerical checks precede all timing; latency_ms is a median of 10 CUDA-event samples.",
            "Teaching experiments are not an exhaustive benchmark or SOTA claim.",
            "FP32 numerical references are separate from same-dtype PyTorch timing baselines.",
            "Attention has a numerical reference only; no native paged performance baseline is claimed.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run verified GPU teaching experiments")
    parser.add_argument("suite", choices=("basics", "gemm", "attention"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.suite)
        _write_json(args.output, result)
    except (ImportError, RuntimeError, ValueError, AssertionError) as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 2
    print(f"{args.suite}: {len(result['records'])} verified records written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
