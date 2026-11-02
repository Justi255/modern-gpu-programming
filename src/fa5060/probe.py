"""JSON environment records without importing GPU packages at module import.

Only named package versions, platform fields, CUDA device properties, and a
small allowlist of compiler/device commands are collected.  Environment
variables, credentials, process listings, and sys.path are never recorded.
"""

from __future__ import annotations

import argparse
import csv
import importlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _load_torch():
    return importlib.import_module("torch")


def _package_version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _command_output(command):
    """Run an allowlisted caller-supplied argv, with bounded wait and output."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "output": None, "returncode": None,
                "error": f"{type(exc).__name__}: {exc}"}
    output = result.stdout.strip()[:8192]
    error = result.stderr.strip()[:2048] if result.returncode else None
    if result.returncode and not error:
        error = f"command exited with status {result.returncode}"
    return {"available": True, "output": output, "returncode": result.returncode,
            "error": error}


def _number(value):
    try:
        result = float(value.strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return int(result) if result.is_integer() else result


def _nvidia_smi():
    fields = (
        "index,name,driver_version,pstate,power.draw,power.limit,"
        "clocks.current.sm,clocks.current.memory,temperature.gpu"
    )
    command = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    result = _command_output(command)
    devices = []
    if result["returncode"] == 0 and result["output"]:
        for row in csv.reader(result["output"].splitlines(), skipinitialspace=True):
            if len(row) != 9:
                continue
            devices.append({
                "index": _number(row[0]), "name": row[1].strip(),
                "driver_version": row[2].strip(), "performance_state": row[3].strip(),
                "power_draw_watts": _number(row[4]), "power_limit_watts": _number(row[5]),
                "sm_clock_mhz": _number(row[6]), "memory_clock_mhz": _number(row[7]),
                "temperature_celsius": _number(row[8]),
            })
    return {"available": result["available"], "devices": devices, "error": result["error"]}


def _integer_property(properties, *names):
    for name in names:
        value = getattr(properties, name, None)
        if value is not None:
            return int(value)
    return None


def collect_environment():
    """Return a JSON-serializable observation, including explicit GPU absence.

    This function may query the CUDA runtime, but never launches a kernel.
    Missing optional tools/properties produce null fields rather than guessed
    hardware values.  The device count, memory, and SM count come from runtime
    properties, not from a table of product names.
    """
    packages = {
        name: _package_version(name)
        for name in ("modern-gpu-programming-5060", "numpy", "torch", "triton",
                     "flashinfer-python", "flash-attn-4", "flash-attn")
    }
    record = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(), "sys_version": sys.version,
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(), "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "description": platform.platform(),
        },
        "packages": packages,
        "cuda_build": None, "cuda_available": False, "gpus": [],
        "gpu_unavailable_reason": None,
    }
    try:
        torch = _load_torch()
        record["cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                record["gpus"].append({
                    "index": index, "name": str(properties.name),
                    "compute_capability": [int(properties.major), int(properties.minor)],
                    "sm_count": int(properties.multi_processor_count),
                    "total_memory_bytes": int(properties.total_memory),
                    "l2_cache_bytes": _integer_property(properties, "L2_cache_size", "l2_cache_size"),
                    "shared_memory_per_block_bytes": _integer_property(properties, "shared_memory_per_block"),
                    "shared_memory_per_block_optin_bytes": _integer_property(properties, "shared_memory_per_block_optin"),
                    "shared_memory_per_sm_bytes": _integer_property(properties, "shared_memory_per_multiprocessor"),
                    "warp_size": _integer_property(properties, "warp_size"),
                    "max_threads_per_block": _integer_property(properties, "max_threads_per_block"),
                })
            record["cuda_available"] = bool(record["gpus"])
            if not record["gpus"]:
                record["gpu_unavailable_reason"] = "CUDA reports no visible devices"
        else:
            record["gpu_unavailable_reason"] = "torch.cuda.is_available() is false"
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        record["gpus"] = []
        record["cuda_available"] = False
        record["gpu_unavailable_reason"] = f"{type(exc).__name__}: {exc}"

    record["nvidia_smi"] = _nvidia_smi()
    record["compiler"] = {
        "nvcc": _command_output(["nvcc", "--version"]),
        "ptxas": _command_output(["ptxas", "--version"]),
        "triton_version": packages["triton"],
    }
    return record


def _write_json(destination, record):
    """Serialize before touching the destination; reject NaN/Inf in records."""
    encoded = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record reproducible GPU/software environment JSON")
    parser.add_argument("--output", type=Path, default=Path("results/environment.json"))
    args = parser.parse_args(argv)
    _write_json(args.output, collect_environment())
    print(f"Environment written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
