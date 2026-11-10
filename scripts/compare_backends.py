"""Compare two fresh attention reports without importing a GPU package.

Use the same environment and workload for both runs, then run:
python scripts/compare_backends.py --triton results/tilelang/paired-triton.json \
    --tilelang results/tilelang/full-tuned.json --output-dir results/tilelang

The output keeps the direct backend ratio separate from each run's fastest
correct native external baseline. Failed cases remain visible with null ratios.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from fa5060.benchmark import (_valid_timing, atomic_write_json, environment_key,
                             evaluate_case, report_backend)
from fa5060.workloads import AttentionCase, full_suite, smoke_suite


def _indexed_cases(report):
    indexed = {}
    for record in report.get("cases", []):
        if any(name not in record for name in ("id", "shape", "causal", "query_len", "seed")):
            raise ValueError("case is missing serialized workload identity")
        case = AttentionCase.from_dict(record)
        if case.case_id in indexed:
            raise ValueError(f"duplicate case ID: {case.case_id}")
        if record.get("backend", report_backend(report)) != report_backend(report):
            raise ValueError(f"case backend disagrees with report: {case.case_id}")
        indexed[case.case_id] = record
    if not indexed:
        raise ValueError("paired reports must contain cases")
    return indexed


def _geometric(values):
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else None


def compare_reports(triton, tilelang):
    """Join by case ID and reject ambiguous or incompatible run contracts."""
    if report_backend(triton) != "triton" or report_backend(tilelang) != "tilelang":
        raise ValueError("inputs must be a Triton report and a TileLang report")
    if triton.get("schema_version") != 1 or tilelang.get("schema_version") != 1:
        raise ValueError("only schema-1 reports are supported")
    for name in ("suite", "rounds", "tune", "seed", "timing"):
        if name not in triton or name not in tilelang or triton[name] != tilelang[name]:
            raise ValueError(f"run contract differs: {name}")
    mode = "offline_tuned" if triton["tune"] else "public_default"
    for report in (triton, tilelang):
        if report.get("configuration_mode", mode) != mode:
            raise ValueError("configuration mode disagrees with tune flag")
    triton_environment = environment_key(triton["environment"])
    if triton_environment != environment_key(tilelang["environment"]):
        raise ValueError("software/hardware environment keys differ; rerun both backends in the same environment")
    left, right = _indexed_cases(triton), _indexed_cases(tilelang)
    if left.keys() != right.keys():
        raise ValueError("paired reports have different case IDs")
    rows = []
    for case_id, a in left.items():
        b = right[case_id]
        case = AttentionCase.from_dict(a)
        if case != AttentionCase.from_dict(b) or a.get("seed") != b.get("seed"):
            raise ValueError(f"workload/seed differs for {case_id}")
        if a.get("seed") != triton["seed"]:
            raise ValueError(f"case seed disagrees with report for {case_id}")
        a_eval, b_eval = evaluate_case(a), evaluate_case(b)
        a_ours, b_ours = a.get("providers", {}).get("ours", {}), b.get("providers", {}).get("ours", {})
        a_us = a_ours["gpu_us_p50"] if _valid_timing(a_ours) else None
        b_us = b_ours["gpu_us_p50"] if _valid_timing(b_ours) else None
        comparable = a_us is not None and b_us is not None
        row = {name: a[name] for name in ("id", "phase", "batch", "kv_len", "heads_q", "heads_kv", "head_dim", "dtype", "layout")}
        row.update(
            triton_status=a_ours.get("status", "missing"), tilelang_status=b_ours.get("status", "missing"),
            triton_gpu_us=a_us, tilelang_gpu_us=b_us,
            tilelang_over_triton_latency=b_us / a_us if comparable else None,
            tilelang_speedup_vs_triton=a_us / b_us if comparable else None,
            triton_fastest_external=a_eval["fastest_baseline"],
            tilelang_fastest_external=b_eval["fastest_baseline"],
            triton_speedup_vs_fastest_external=a_eval["speedup_vs_fastest"],
            tilelang_speedup_vs_fastest_external=b_eval["speedup_vs_fastest"],
            triton_correctness_passed=a_eval["correctness_passed"],
            tilelang_correctness_passed=b_eval["correctness_passed"],
            triton_baseline_coverage=a_eval["mandatory_baselines_covered"],
            tilelang_baseline_coverage=b_eval["mandatory_baselines_covered"],
        )
        rows.append(row)
    phases = {}
    for phase in ("prefill", "decode"):
        selected = [row for row in rows if row["phase"] == phase]
        ratios = [row["tilelang_speedup_vs_triton"] for row in selected if row["tilelang_speedup_vs_triton"] is not None]
        phases[phase] = {
            "case_count": len(selected), "comparable_cases": len(ratios),
            "unavailable_case_ids": [row["id"] for row in selected if row["tilelang_speedup_vs_triton"] is None],
            "all_backend_outputs_correct": bool(selected) and all(
                row["triton_correctness_passed"] and row["tilelang_correctness_passed"] for row in selected),
            "geometric_mean_tilelang_speedup_vs_triton": _geometric(ratios),
            "worst_tilelang_speedup_vs_triton": min(ratios) if ratios else None,
        }
        for backend in ("triton", "tilelang"):
            covered = [row for row in selected if row[f"{backend}_baseline_coverage"]]
            qualified = [row for row in covered if row[f"{backend}_correctness_passed"]
                         and row[f"{backend}_speedup_vs_fastest_external"] is not None]
            values = [row[f"{backend}_speedup_vs_fastest_external"] for row in qualified]
            phases[phase][f"{backend}_baseline_coverage_complete"] = bool(selected) and len(covered) == len(selected)
            phases[phase][f"{backend}_baseline_coverage_count"] = len(covered)
            phases[phase][f"{backend}_baseline_coverage_invalid_case_ids"] = [
                row["id"] for row in selected if not row[f"{backend}_baseline_coverage"]]
            phases[phase][f"{backend}_external_comparable_cases"] = len(values)
            qualified_ids = {row["id"] for row in qualified}
            phases[phase][f"{backend}_external_invalid_case_ids"] = [
                row["id"] for row in selected if row["id"] not in qualified_ids]
            phases[phase][f"{backend}_geometric_mean_speedup_vs_fastest_external"] = _geometric(values)
    expected = full_suite() if triton["suite"] == "full" else smoke_suite()
    frozen_complete = set(left) == {case.case_id for case in expected}
    return {
        "schema_version": 1, "triton_backend": "triton", "tilelang_backend": "tilelang",
        "suite": triton["suite"], "rounds": triton["rounds"], "configuration_mode": mode,
        "environment_key": triton_environment, "frozen_suite_complete": frozen_complete,
        "ratio_definition": {"tilelang_over_triton_latency": "TileLang GPU microseconds / Triton GPU microseconds; lower is faster",
                             "tilelang_speedup_vs_triton": "Triton GPU microseconds / TileLang GPU microseconds; higher is faster",
                             "speedup_vs_fastest_external": "fastest validated native external provider GPU microseconds / selected backend GPU microseconds"},
        "phases": phases, "cases": rows,
        "notes": ["Direct backend ratios and external-provider ratios are separate comparisons.",
                  "Geometric means use available valid pairs; unavailable case IDs and counts remain explicit.",
                  "External geometric means include only correct rows with complete mandatory baseline coverage; raw per-case external ratios are retained even when coverage is incomplete.",
                  "Public-default calls and offline per-case tuning are distinct modes; this comparison does not load an online tuning table.",
                  "Matching environment keys does not establish exclusive GPU access or fixed thermal/power conditions; inspect original telemetry."],
    }


def _number(value):
    return f"{value:.3f}" if value is not None else "unavailable"


def comparison_markdown(summary):
    text = ["### TileLang 与 Triton 的配对测量", "",
            f"配置模式：`{summary['configuration_mode']}`；每项 {summary['rounds']} 轮。"
            f"按 case ID 匹配 {len(summary['cases'])} 项；冻结清单匹配：{'是' if summary['frozen_suite_complete'] else '否'}。"
            "清单匹配只说明 case ID 齐全，正确性和外部基线覆盖另行检查。", "",
            "TileLang / Triton 是两种后端的延迟比，越小越快；直接加速是 Triton / TileLang，越大越快。"
            "外部基线几何平均仅纳入强制基线覆盖完整且全部数值检查通过的项，"
            "分别取每次运行中最快的正确、兼容原生 provider，与后端间比值分开统计。", "",
            "| Phase | 有效配对 / 全部 | 两后端正确性完整 | TileLang / Triton 几何平均延迟比 | TileLang 对 Triton 几何平均加速 | 最差直接加速 | Triton 对最快外部基线加速 | TileLang 对最快外部基线加速 |",
            "|---|---:|---|---:|---:|---:|---:|---:|"]
    for phase, values in summary["phases"].items():
        speedup = values["geometric_mean_tilelang_speedup_vs_triton"]
        text.append(f"| {phase} | {values['comparable_cases']} / {values['case_count']} | "
                    f"{'是' if values['all_backend_outputs_correct'] else '否'} | "
                    f"{_number(1 / speedup if speedup else None)} | {_number(speedup)} | "
                    f"{_number(values['worst_tilelang_speedup_vs_triton'])} | "
                    f"{_number(values['triton_geometric_mean_speedup_vs_fastest_external'])} | "
                    f"{_number(values['tilelang_geometric_mean_speedup_vs_fastest_external'])} |")
    text.extend(["", "| Phase | 后端 | 强制基线覆盖 / 全部 | 基线覆盖完整 | 外部几何平均有效项 / 全部 |",
                 "|---|---|---:|---|---:|"])
    incomplete = []
    for phase, values in summary["phases"].items():
        if not values["case_count"]:
            continue
        for backend in ("triton", "tilelang"):
            complete = values[f"{backend}_baseline_coverage_complete"]
            text.append(f"| {phase} | {backend} | {values[f'{backend}_baseline_coverage_count']} / {values['case_count']} | "
                        f"{'是' if complete else '否'} | {values[f'{backend}_external_comparable_cases']} / {values['case_count']} |")
            if not complete:
                incomplete.append(f"`{backend}/{phase}`")
    if incomplete:
        text.extend(["", "**外部基线覆盖不完整**：" + "、".join(incomplete) + "。"
                     "外部基线几何平均只描述表中合格子集，不能作为完整冻结集合的性能结论；"
                     "缺失覆盖和不合格项的 case ID 保留在 JSON，逐项原始比值保留在 JSON/CSV。"])
    text.extend(["", "失败和缺失项保留在 JSON/CSV 中，几何平均只使用有效配对；有效项数不足时不能据此声称完整集合更快。"
                 "逐元素 FP16 0.003 / BF16 0.02 和归一化 RMS 合同保持不变。图计时包含全部 split 与 merge，"
                 "准备、首次调用（含缓存缺失时的 JIT）、预热单独记录，不计入稳定态延迟。", "",
                 "同一软件和硬件键并不代表设备独占、锁频或相同温度；功耗、时钟和温度应检查两份原始记录及遥测。", ""])
    for backend, source in summary.get("sources", {}).items():
        text.append(f"- `{backend}` 原始文件：`{source['path']}`；SHA256 `{source['sha256']}`。")
    return "\n".join(text) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton", type=Path, required=True)
    parser.add_argument("--tilelang", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tilelang"))
    args = parser.parse_args(argv)
    paths = {"triton": args.triton, "tilelang": args.tilelang}
    snapshots = {name: path.read_bytes() for name, path in paths.items()}
    reports = {name: json.loads(snapshot) for name, snapshot in snapshots.items()}
    summary = compare_reports(reports["triton"], reports["tilelang"])
    summary["sources"] = {name: {"path": str(path), "sha256": hashlib.sha256(snapshots[name]).hexdigest()}
                          for name, path in paths.items()}
    destinations = [args.output_dir / f"comparison.{suffix}" for suffix in ("json", "csv", "md")]
    if any(destination.resolve() == source.resolve() for destination in destinations for source in paths.values()):
        raise ValueError("comparison outputs must not overwrite raw input reports")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destinations[0], summary)
    with destinations[1].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["cases"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["cases"])
    destinations[2].write_text(comparison_markdown(summary), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "cases": len(summary["cases"]),
                      "frozen_suite_complete": summary["frozen_suite_complete"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
