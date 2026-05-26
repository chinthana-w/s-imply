"""Summarize streamed b17 coverage CSV artifacts.

The full b17 benchmark can run for days, so this script is intentionally useful
for both partial and completed per-fault CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

RESULT_NAMES = {
    "0": "UNTESTABLE",
    "1": "SUCCESS",
    "2": "TIMEOUT",
    "3": "BACKTRACK_LIMIT",
    "": "NOT_RUN",
    "None": "NOT_RUN",
}

SLOW_CLASSIC_HEADER = (
    "| Fault index | Gate | Fault val | Classic result | Classic backtracks | "
    "Classic recursive calls | Classic time s | AI ok | AI time s |"
)
BACKTRACK_HEADER = (
    "| Fault index | Gate | Fault val | Classic result | Classic backtracks | "
    "Classic time s | AI ok | AI time s |"
)


def _truth(value: str | None) -> bool:
    return str(value).lower() == "true"


def _num(value: str | None, default: float = 0.0) -> float:
    if value in (None, "", "None"):
        return default
    return float(value)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _time_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"total": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "total": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _fmt_time_stats(label: str, stats: dict[str, float]) -> str:
    return (
        f"| {label} | {stats['total']:.2f} | {stats['mean']:.4f} | "
        f"{stats['median']:.4f} | {stats['p90']:.4f} | "
        f"{stats['p99']:.4f} | {stats['max']:.4f} |"
    )


def _classic_result_name(row: dict[str, str]) -> str:
    result_code = str(row.get("classic_result_code", ""))
    return RESULT_NAMES.get(result_code, result_code)


def summarize(csv_path: Path, expected_total: int) -> tuple[dict, str]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    total_seen = len(rows)
    ai_ok = [row for row in rows if _truth(row.get("ok"))]
    classic_ok = [row for row in rows if _truth(row.get("classic_ok"))]
    classic_results = Counter(_classic_result_name(row) for row in rows)
    target = 0.80
    target_denominator = len(classic_ok)
    target_required = math.ceil(target * target_denominator) if target_denominator else 0
    target_observed = len(ai_ok) / target_denominator if target_denominator else 0.0

    ai_total = [_num(row.get("time_s")) for row in rows]
    ai_precheck_solve = [_num(row.get("ai_precheck_solve_time_s")) for row in rows]
    ai_precheck_sim = [_num(row.get("ai_precheck_sim_time_s")) for row in rows]
    ai_hint_solve = [_num(row.get("ai_hint_solve_time_s")) for row in rows]
    ai_podem_search = [_num(row.get("ai_podem_search_time_s")) for row in rows]
    classic_total = [_num(row.get("classic_time_s")) for row in rows]

    top_ai_slow = sorted(rows, key=lambda row: _num(row.get("time_s")), reverse=True)[:20]
    top_classic_slow = sorted(
        rows,
        key=lambda row: _num(row.get("classic_time_s")),
        reverse=True,
    )[:20]
    top_classic_backtracks = sorted(
        rows,
        key=lambda row: (_num(row.get("classic_backtracks")), _num(row.get("classic_time_s"))),
        reverse=True,
    )[:20]

    summary = {
        "csv": str(csv_path),
        "expected_total": expected_total,
        "processed_faults": total_seen,
        "progress": total_seen / expected_total if expected_total else 0.0,
        "ai_succeeded": len(ai_ok),
        "ai_coverage_seen": len(ai_ok) / total_seen if total_seen else 0.0,
        "ai_coverage_expected_total": len(ai_ok) / expected_total if expected_total else 0.0,
        "classic_succeeded": len(classic_ok),
        "classic_coverage_seen": len(classic_ok) / total_seen if total_seen else 0.0,
        "target": {
            "coverage": target,
            "denominator": "classic_succeeded",
            "denominator_count": target_denominator,
            "required_faults": target_required,
            "observed_faults": len(ai_ok),
            "observed_coverage": target_observed,
            "passed": len(ai_ok) >= target_required if target_denominator else False,
        },
        "classic_results": dict(classic_results),
        "timing": {
            "ai_total": _time_summary(ai_total),
            "ai_precheck_solve": _time_summary(ai_precheck_solve),
            "ai_precheck_sim": _time_summary(ai_precheck_sim),
            "ai_hint_solve": _time_summary(ai_hint_solve),
            "ai_podem_search": _time_summary(ai_podem_search),
            "classic_total": _time_summary(classic_total),
        },
    }

    lines = [
        "# b17 Full-Fault Coverage Timing Summary",
        "",
        f"- CSV: `{csv_path}`",
        f"- Processed faults: `{total_seen}/{expected_total}` ({summary['progress']:.2%})",
        (
            "- AI/system-mode coverage over processed faults: "
            f"`{len(ai_ok)}/{total_seen}` = `{summary['ai_coverage_seen']:.4%}`"
        ),
        (
            "- AI/system-mode coverage lower bound over all expected faults: "
            f"`{len(ai_ok)}/{expected_total}` = "
            f"`{summary['ai_coverage_expected_total']:.4%}`"
        ),
        (
            "- Classic coverage over processed faults: "
            f"`{len(classic_ok)}/{total_seen}` = "
            f"`{summary['classic_coverage_seen']:.4%}`"
        ),
        (
            "- Target metric: AI/system mode must cover `80.0000%` of faults covered "
            f"by classic PODEM; observed `{len(ai_ok)}/{target_denominator}` = "
            f"`{target_observed:.4%}`, required `{target_required}`, "
            f"pass=`{summary['target']['passed']}`"
        ),
        "",
        "## Classic Result Codes",
        "",
        "| Result | Count |",
        "|---|---:|",
    ]
    for result, count in sorted(classic_results.items()):
        lines.append(f"| {result} | {count} |")

    lines.extend(
        [
            "",
            "## Timing Breakdown",
            "",
            "| Segment | Total s | Mean s | Median s | P90 s | P99 s | Max s |",
            "|---|---:|---:|---:|---:|---:|---:|",
            _fmt_time_stats("AI total per fault", summary["timing"]["ai_total"]),
            _fmt_time_stats("AI precheck solver", summary["timing"]["ai_precheck_solve"]),
            _fmt_time_stats("AI precheck simulation", summary["timing"]["ai_precheck_sim"]),
            _fmt_time_stats("AI hint solver", summary["timing"]["ai_hint_solve"]),
            _fmt_time_stats("AI-guided PODEM search", summary["timing"]["ai_podem_search"]),
            _fmt_time_stats("Classic total per fault", summary["timing"]["classic_total"]),
            "",
            "## Slowest AI Faults",
            "",
            "| Fault index | Gate | Fault val | AI ok | AI time s | Classic ok | Classic time s |",
            "|---:|---:|---:|:---|---:|:---|---:|",
        ]
    )
    for row in top_ai_slow:
        lines.append(
            f"| {row.get('fault_index')} | {row.get('gate_id')} | {row.get('fault_val')} | "
            f"{row.get('ok')} | {_num(row.get('time_s')):.4f} | "
            f"{row.get('classic_ok')} | {_num(row.get('classic_time_s')):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Slowest Classic Faults",
            "",
            SLOW_CLASSIC_HEADER,
            "|---:|---:|---:|---|---:|---:|---:|:---|---:|",
        ]
    )
    for row in top_classic_slow:
        result = _classic_result_name(row)
        lines.append(
            f"| {row.get('fault_index')} | {row.get('gate_id')} | {row.get('fault_val')} | "
            f"{result} | {_num(row.get('classic_backtracks')):.0f} | "
            f"{_num(row.get('classic_recursive_calls')):.0f} | "
            f"{_num(row.get('classic_time_s')):.4f} | {row.get('ok')} | "
            f"{_num(row.get('time_s')):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Highest Classic Backtrack Counts",
            "",
            BACKTRACK_HEADER,
            "|---:|---:|---:|---|---:|---:|:---|---:|",
        ]
    )
    for row in top_classic_backtracks:
        result = _classic_result_name(row)
        lines.append(
            f"| {row.get('fault_index')} | {row.get('gate_id')} | {row.get('fault_val')} | "
            f"{result} | {_num(row.get('classic_backtracks')):.0f} | "
            f"{_num(row.get('classic_time_s')):.4f} | {row.get('ok')} | "
            f"{_num(row.get('time_s')):.4f} |"
        )

    return summary, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize b17 full-fault coverage CSV")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--expected-total", type=int, default=64458)
    args = parser.parse_args()

    summary, markdown = summarize(args.csv, args.expected_total)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
