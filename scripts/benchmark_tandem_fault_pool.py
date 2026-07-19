"""Benchmark a fault pool with a classic baseline and a classic/AI tandem pass.

Phase 1 runs classic PODEM with a 10 second per-fault budget. Phase 2 runs
classic PODEM and AI-guided PODEM independently with equal 5 second budgets.
The tandem result succeeds if either method succeeds, and its solve time is the
minimum successful time. CSV checkpoints make long runs resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from scripts.benchmark_itc99_gate import ImprovedHintBacktracer
from src.atpg.ai_podem import AiPodemConfig, HierarchicalReconvSolver, ModelPairPredictor
from src.atpg.logic_sim_three import fault_is_at_po, logic_sim, reset_gates
from src.atpg.podem import (
    BACKTRACK_LIMIT,
    SUCCESS,
    TIMEOUT,
    UNTESTABLE,
    get_statistics,
    initialize,
    podem,
    reset_statistics,
    simple_backtrace,
)
from src.util.io import parse_bench_file
from src.util.struct import Fault, GateType, LogicValue

BASELINE_FIELDS = [
    "fault_index",
    "bench",
    "gate_id",
    "fault_val",
    "classic10_ok",
    "classic10_result_code",
    "classic10_backtracks",
    "classic10_recursive_calls",
    "classic10_backtrace_count",
    "classic10_backtrace_hops",
    "classic10_time_s",
]

TANDEM_FIELDS = [
    "fault_index",
    "bench",
    "gate_id",
    "fault_val",
    "classic5_ok",
    "classic5_result_code",
    "classic5_backtracks",
    "classic5_recursive_calls",
    "classic5_time_s",
    "ai5_ok",
    "ai5_result_code",
    "ai5_backtracks",
    "ai5_recursive_calls",
    "ai5_time_s",
    "ai5_precheck_success",
    "ai5_has_reconv_pairs",
    "ai5_error",
    "tandem_ok",
    "tandem_timeout",
    "tandem_winner",
    "tandem_solve_time_s",
]


class _OverallAiTimeout(TimeoutError):
    pass


def _git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args], check=False, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_fault_pool(path: Path) -> tuple[list[tuple[str, Fault]], dict]:
    with path.open() as handle:
        payload = json.load(handle)
    default_bench = payload.get("bench")
    fault_entries = [
        (
            item.get("bench") or default_bench,
            Fault(int(item["gate_id"]), LogicValue(int(item["fault_val"]))),
        )
        for item in payload["faults"]
    ]
    missing = [index for index, (bench, _) in enumerate(fault_entries) if not bench]
    if missing:
        raise ValueError(f"fault pool has no circuit path for fault indices {missing[:10]}")
    return fault_entries, payload


def _eta(elapsed: float, completed: int, total: int) -> str:
    if completed <= 0:
        return "unknown"
    remaining = max(0.0, elapsed / completed * (total - completed))
    hours, remainder = divmod(int(remaining), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress(
    phase: str,
    started: float,
    completed: int,
    total: int,
    succeeded: int,
    resume_from: int = 0,
) -> None:
    elapsed = time.monotonic() - started
    completed_this_run = completed - resume_from
    remaining_this_run = total - resume_from
    rate = completed_this_run / elapsed if elapsed > 0 else 0.0
    print(
        f"[{phase}] progress={completed}/{total} ({completed / total:.2%}) "
        f"solved={succeeded} coverage={succeeded / max(1, completed):.2%} "
        f"rate={rate:.3f} faults/s "
        f"eta={_eta(elapsed, completed_this_run, remaining_this_run)}",
        flush=True,
    )


def _run_classic(circuit: list, total_gates: int, fault: Fault, timeout: float, cap: int) -> dict:
    reset_gates(circuit, total_gates)
    reset_statistics()
    started = time.monotonic()
    result = podem(
        circuit,
        fault,
        total_gates,
        backtrace_func=simple_backtrace,
        timeout=timeout,
        max_backtracks=cap,
    )
    elapsed = time.monotonic() - started
    stats = get_statistics()
    return {
        "ok": int(result) == SUCCESS,
        "result_code": int(result),
        "backtracks": int(stats.get("backtrack_count", 0)),
        "recursive_calls": int(stats.get("total_recursive_calls", 0)),
        "backtrace_count": int(stats.get("backtrace_count", 0)),
        "backtrace_hops": int(stats.get("backtrace_hops", 0)),
        "time_s": elapsed,
    }


def _direct_detection(
    circuit: list,
    total_gates: int,
    fault: Fault,
    assignment: dict[int, LogicValue] | None,
) -> bool:
    if not assignment:
        return False
    assigned = 0
    for gate_id, value in assignment.items():
        gate_id = int(gate_id)
        if 0 <= gate_id < len(circuit) and circuit[gate_id].type == GateType.INPT:
            circuit[gate_id].val = LogicValue(value)
            assigned += 1
    if assigned == 0:
        return False
    logic_sim(circuit, total_gates, fault)
    return fault_is_at_po(circuit, total_gates)


def _run_ai_body(
    circuit: list,
    total_gates: int,
    fault: Fault,
    solver: HierarchicalReconvSolver,
    timeout: float,
    cap: int,
    seed: int,
) -> dict:
    started = time.monotonic()
    activation = LogicValue.ONE if fault.value == LogicValue.ZERO else LogicValue.ZERO
    pairs = solver._collect_and_sort_pairs(int(fault.gate_id))
    solver.pair_cache[int(fault.gate_id)] = pairs
    has_pairs = bool(pairs)
    reset_gates(circuit, total_gates)
    assignment = solver.solve(int(fault.gate_id), activation, seed=seed) if has_pairs else None
    precheck_success = _direct_detection(circuit, total_gates, fault, assignment)
    if precheck_success:
        return {
            "ok": True,
            "result_code": SUCCESS,
            "backtracks": 0,
            "recursive_calls": 0,
            "time_s": time.monotonic() - started,
            "precheck_success": True,
            "has_reconv_pairs": has_pairs,
            "error": "",
        }

    remaining = max(0.001, timeout - (time.monotonic() - started))
    reset_gates(circuit, total_gates)
    reset_statistics()
    backtracer = ImprovedHintBacktracer(assignment or {}, no_fallback=True)
    result = podem(
        circuit,
        fault,
        total_gates,
        backtrace_func=backtracer,
        timeout=remaining,
        max_backtracks=cap,
    )
    stats = get_statistics()
    return {
        "ok": int(result) == SUCCESS,
        "result_code": int(result),
        "backtracks": int(stats.get("backtrack_count", 0)),
        "recursive_calls": int(stats.get("total_recursive_calls", 0)),
        "time_s": time.monotonic() - started,
        "precheck_success": False,
        "has_reconv_pairs": has_pairs,
        "error": "",
    }


def _run_ai(
    circuit: list,
    total_gates: int,
    fault: Fault,
    solver: HierarchicalReconvSolver,
    timeout: float,
    cap: int,
    seed: int,
) -> dict:
    started = time.monotonic()

    def alarm_handler(signum, frame):
        raise _OverallAiTimeout(f"AI mode exceeded its {timeout:.3f}s total budget")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return _run_ai_body(circuit, total_gates, fault, solver, timeout, cap, seed)
    except _OverallAiTimeout as exc:
        stats = get_statistics()
        return {
            "ok": False,
            "result_code": TIMEOUT,
            "backtracks": int(stats.get("backtrack_count", 0)),
            "recursive_calls": int(stats.get("total_recursive_calls", 0)),
            "time_s": time.monotonic() - started,
            "precheck_success": False,
            "has_reconv_pairs": bool(solver.pair_cache.get(int(fault.gate_id), [])),
            "error": str(exc),
        }
    except Exception as exc:
        stats = get_statistics()
        return {
            "ok": False,
            "result_code": UNTESTABLE,
            "backtracks": int(stats.get("backtrack_count", 0)),
            "recursive_calls": int(stats.get("total_recursive_calls", 0)),
            "time_s": time.monotonic() - started,
            "precheck_success": False,
            "has_reconv_pairs": bool(solver.pair_cache.get(int(fault.gate_id), [])),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _float(row: dict, key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else math.nan


def _bool(row: dict, key: str) -> bool:
    return str(row.get(key, "")).lower() in {"1", "true", "yes"}


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _timing(values: list[float]) -> dict:
    clean = [value for value in values if not math.isnan(value)]
    return {
        "count": len(clean),
        "total_s": round(sum(clean), 6),
        "mean_s": round(statistics.mean(clean), 6) if clean else None,
        "median_s": round(statistics.median(clean), 6) if clean else None,
        "p95_s": round(_percentile(clean, 0.95), 6) if clean else None,
        "max_s": round(max(clean), 6) if clean else None,
    }


def _summarize(baseline: list[dict], tandem: list[dict], args: argparse.Namespace) -> dict:
    baseline_by_index = {int(row["fault_index"]): row for row in baseline}
    base_ok = {idx for idx, row in baseline_by_index.items() if _bool(row, "classic10_ok")}
    classic_ok = {int(row["fault_index"]) for row in tandem if _bool(row, "classic5_ok")}
    ai_ok = {int(row["fault_index"]) for row in tandem if _bool(row, "ai5_ok")}
    union = classic_ok | ai_ok
    both = classic_ok & ai_ok
    ai_only = ai_ok - classic_ok
    classic_only = classic_ok - ai_ok
    neither = set(range(len(tandem))) - union
    ai_wins = sum(row.get("tandem_winner") == "ai" for row in tandem)
    classic_wins = sum(row.get("tandem_winner") == "classic" for row in tandem)
    ties = sum(row.get("tandem_winner") == "tie" for row in tandem)
    benches = sorted({row["bench"] for row in tandem})
    per_circuit = []
    for bench in benches:
        base_rows = [row for row in baseline if row["bench"] == bench]
        tandem_rows = [row for row in tandem if row["bench"] == bench]
        classic_count = sum(_bool(row, "classic5_ok") for row in tandem_rows)
        ai_count = sum(_bool(row, "ai5_ok") for row in tandem_rows)
        union_count = sum(_bool(row, "tandem_ok") for row in tandem_rows)
        ai_only_count = sum(
            _bool(row, "ai5_ok") and not _bool(row, "classic5_ok") for row in tandem_rows
        )
        per_circuit.append(
            {
                "bench": bench,
                "faults": len(tandem_rows),
                "classic10_solved": sum(_bool(row, "classic10_ok") for row in base_rows),
                "classic5_solved": classic_count,
                "ai5_solved": ai_count,
                "tandem_solved": union_count,
                "tandem_coverage": union_count / max(1, len(tandem_rows)),
                "ai_only_solved": ai_only_count,
            }
        )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": len(baseline) == len(tandem) == args.total_faults,
        "total_faults": args.total_faults,
        "budgets": {
            "baseline_classic_s": args.baseline_timeout,
            "tandem_classic_s": args.tandem_timeout,
            "tandem_ai_s": args.tandem_timeout,
        },
        "baseline": {
            "solved": len(base_ok),
            "failed": len(baseline) - len(base_ok),
            "coverage": len(base_ok) / max(1, len(baseline)),
            "backtracks_total": sum(int(row["classic10_backtracks"]) for row in baseline),
            "timing_all": _timing([_float(row, "classic10_time_s") for row in baseline]),
            "timing_solved": _timing(
                [_float(row, "classic10_time_s") for row in baseline if _bool(row, "classic10_ok")]
            ),
        },
        "tandem": {
            "classic_solved": len(classic_ok),
            "ai_solved": len(ai_ok),
            "union_solved": len(union),
            "union_failed": len(neither),
            "union_coverage": len(union) / max(1, len(tandem)),
            "both_solved": len(both),
            "ai_only_solved": len(ai_only),
            "classic_only_solved": len(classic_only),
            "ai_wins": ai_wins,
            "classic_wins": classic_wins,
            "ties": ties,
            "ai_incremental_coverage_points": len(ai_only) / max(1, len(tandem)),
            "ai_incremental_relative_to_classic": len(ai_only) / max(1, len(classic_ok)),
            "classic_backtracks_total": sum(int(row["classic5_backtracks"]) for row in tandem),
            "ai_search_backtracks_total": sum(int(row["ai5_backtracks"]) for row in tandem),
            "classic_timing_all": _timing([_float(row, "classic5_time_s") for row in tandem]),
            "ai_timing_all": _timing([_float(row, "ai5_time_s") for row in tandem]),
            "portfolio_solve_timing": _timing(
                [_float(row, "tandem_solve_time_s") for row in tandem if _bool(row, "tandem_ok")]
            ),
        },
        "cross_budget": {
            "tandem_union_also_in_baseline": len(union & base_ok),
            "tandem_union_not_in_baseline": len(union - base_ok),
            "baseline_not_in_tandem_union": len(base_ok - union),
            "both_fail": len(set(range(len(tandem))) - (union | base_ok)),
        },
        "per_circuit": per_circuit,
        "result_code_labels": {
            str(SUCCESS): "SUCCESS",
            str(UNTESTABLE): "UNTESTABLE",
            str(TIMEOUT): "TIMEOUT",
            str(BACKTRACK_LIMIT): "BACKTRACK_LIMIT",
        },
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _write_report(path: Path, summary: dict, args: argparse.Namespace) -> None:
    baseline = summary["baseline"]
    tandem = summary["tandem"]
    cross = summary["cross_budget"]
    total = summary["total_faults"]
    ai_only = tandem["ai_only_solved"]
    lines = [
        "# Equal-Budget Classic and AI Tandem ATPG Benchmark",
        "",
        "## Executive summary",
        "",
        f"This benchmark evaluates **{total:,} identical faults** in two controlled phases. "
        f"The baseline gives classic PODEM {args.baseline_timeout:g} seconds per fault. The "
        f"tandem phase gives classic and AI-guided PODEM {args.tandem_timeout:g} seconds each, "
        "records both outcomes, and uses the faster successful method as portfolio solve time.",
        "",
        f"The equal-budget tandem solved **{tandem['union_solved']:,}/{total:,} "
        f"({tandem['union_coverage']:.2%})**. AI uniquely solved **{ai_only:,}** faults that "
        f"classic did not solve in its equal {args.tandem_timeout:g}-second budget, adding "
        f"**{tandem['ai_incremental_coverage_points']:.2%} absolute coverage**. On faults solved "
        f"by both, AI was the faster successful path for **{tandem['ai_wins']:,}** faults.",
        "",
        "## Experimental design",
        "",
        "```mermaid",
        "flowchart TD",
        "    P[Same complete fault pool] --> B[Phase 1: Classic PODEM\\n10 s per fault]",
        "    P --> T[Phase 2: Equal-budget tandem]",
        "    T --> C[Classic PODEM\\n5 s per fault]",
        "    T --> A[AI-guided PODEM\\n5 s total per fault]",
        "    C --> M{Either succeeds?}",
        "    A --> M",
        "    M -->|Yes| S[Portfolio success\\nmin successful time]",
        "    M -->|No| X[Tandem timeout/failure]",
        "```",
        "",
        "The two tandem methods run as independent attempts on reset circuit state. Both are "
        "always measured; a success by one method does not suppress the other measurement. "
        "AI's five-second cap includes topology lookup, model inference, direct-pattern "
        "simulation, and any AI-guided PODEM search.",
        "",
        "## Coverage comparison",
        "",
        "| Configuration | Solved | Failed | Coverage |",
        "|---|---:|---:|---:|",
        f"| Classic baseline ({args.baseline_timeout:g} s) | {baseline['solved']:,} | "
        f"{baseline['failed']:,} | {baseline['coverage']:.2%} |",
        f"| Classic equal-budget ({args.tandem_timeout:g} s) | "
        f"{tandem['classic_solved']:,} | {total - tandem['classic_solved']:,} | "
        f"{tandem['classic_solved'] / total:.2%} |",
        f"| AI equal-budget ({args.tandem_timeout:g} s) | {tandem['ai_solved']:,} | "
        f"{total - tandem['ai_solved']:,} | {tandem['ai_solved'] / total:.2%} |",
        f"| Tandem union | {tandem['union_solved']:,} | {tandem['union_failed']:,} | "
        f"{tandem['union_coverage']:.2%} |",
        "",
        "## Per-circuit coverage",
        "",
        "Every locally available standard ITC99 circuit contributes faults to the pool. This "
        "table prevents a large circuit from hiding weak or missing circuit coverage.",
        "",
        "| Circuit | Pool faults | Classic 10 s | Classic 5 s | AI 5 s | Tandem | AI only |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {Path(row['bench']).stem} | {row['faults']:,} | "
            f"{row['classic10_solved']:,} | {row['classic5_solved']:,} | "
            f"{row['ai5_solved']:,} | {row['tandem_solved']:,} "
            f"({row['tandem_coverage']:.2%}) | {row['ai_only_solved']:,} |"
            for row in summary["per_circuit"]
        ],
        "",
        "| Equal-budget overlap | Faults | Meaning |",
        "|---|---:|---|",
        f"| Both solve | {tandem['both_solved']:,} | Redundant coverage; latency competition |",
        f"| AI only | {ai_only:,} | Direct complementary contribution from AI |",
        f"| Classic only | {tandem['classic_only_solved']:,} | Classic remains essential |",
        f"| Neither | {tandem['union_failed']:,} | Tandem timeout/failure |",
        "",
        "## Timing and winner analysis",
        "",
        "| Metric | Classic 5 s | AI 5 s | Tandem chosen solve time |",
        "|---|---:|---:|---:|",
        f"| Mean (s) | {_fmt(tandem['classic_timing_all']['mean_s'])} | "
        f"{_fmt(tandem['ai_timing_all']['mean_s'])} | "
        f"{_fmt(tandem['portfolio_solve_timing']['mean_s'])} |",
        f"| Median (s) | {_fmt(tandem['classic_timing_all']['median_s'])} | "
        f"{_fmt(tandem['ai_timing_all']['median_s'])} | "
        f"{_fmt(tandem['portfolio_solve_timing']['median_s'])} |",
        f"| P95 (s) | {_fmt(tandem['classic_timing_all']['p95_s'])} | "
        f"{_fmt(tandem['ai_timing_all']['p95_s'])} | "
        f"{_fmt(tandem['portfolio_solve_timing']['p95_s'])} |",
        "",
        f"AI supplied the minimum successful solve time on **{tandem['ai_wins']:,}** faults; "
        f"classic won on **{tandem['classic_wins']:,}**, with **{tandem['ties']:,}** ties.",
        "",
        "## Backtracking",
        "",
        "| Measurement | Total | Interpretation |",
        "|---|---:|---|",
        f"| Classic 10 s backtracks | {baseline['backtracks_total']:,} | Baseline search work |",
        f"| Classic 5 s backtracks | {tandem['classic_backtracks_total']:,} | "
        "Equal-budget classic work |",
        f"| AI-guided search backtracks | {tandem['ai_search_backtracks_total']:,} | "
        "PODEM search after AI inference; direct AI successes use zero |",
        "",
        "The AI counter is deliberately labeled *AI-guided search backtracks*: model inference "
        "is not a backtracking procedure, so treating it as an ordinary PODEM backtrack count "
        "would overstate comparability.",
        "",
        "## Relation to the 10-second classic baseline",
        "",
        "| Cross-budget category | Faults |",
        "|---|---:|",
        f"| Tandem union also solved by classic 10 s | "
        f"{cross['tandem_union_also_in_baseline']:,} |",
        f"| Tandem union solved, classic 10 s did not | "
        f"{cross['tandem_union_not_in_baseline']:,} |",
        f"| Classic 10 s solved, tandem union did not | "
        f"{cross['baseline_not_in_tandem_union']:,} |",
        f"| Failed in all measured configurations | {cross['both_fail']:,} |",
        "",
        "## Why AI is a valid contribution",
        "",
        "1. **Controlled marginal coverage.** AI-only solves use the same faults and the same "
        "per-method timeout as classic, so they directly measure complementary capability.",
        "2. **Portfolio latency.** Taking the minimum successful time is operationally valid "
        "when both solvers can be dispatched together; AI wins reduce response latency even "
        "when classic would eventually solve the same fault.",
        "3. **Different search bias.** Classic uses deterministic structural heuristics, while "
        "AI supplies learned reconvergence-aware assignments. Unique solves demonstrate that "
        "the learned bias reaches useful regions classic misses under the same time cap.",
        "4. **Failure containment.** The portfolio never loses a classic success merely because "
        "AI fails: independent state and union semantics preserve either result.",
        "5. **Auditable evidence.** Per-fault raw result codes, wall times, backtracks, winner, "
        "and checkpoint data support reproduction and alternative aggregation.",
        "",
        "## Limitations and interpretation",
        "",
        "- This establishes contribution on this fixed fault pool and checkpoint; it does not "
        "by itself prove generalization to unrelated circuits.",
        "- Sequential wall-clock execution emulates two independent equal-budget workers. The "
        "reported portfolio latency is the minimum method time; actual parallel deployment "
        "also incurs scheduler and hardware contention overhead.",
        "- `tandem_timeout` means neither method produced a test within its budget. Raw result "
        "codes remain available to distinguish algorithmic timeout, backtrack limit, and "
        "untestable/error outcomes.",
        "- OS scheduling and accelerator state introduce timing noise. Repeated trials are "
        "appropriate before making small latency-difference claims.",
        "",
        "## Reproduction artifacts",
        "",
        f"- Fault pool: `{args.fault_list}`",
        f"- Model: `{args.model}`",
        f"- Baseline CSV: `{args.baseline_csv}`",
        f"- Tandem CSV: `{args.tandem_csv}`",
        f"- Machine-readable summary: `{args.summary_out}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines))
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classic baseline and equal-budget AI tandem")
    parser.add_argument("--fault-list", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--baseline-timeout", type=float, default=10.0)
    parser.add_argument("--tandem-timeout", type=float, default=5.0)
    parser.add_argument("--max-backtracks", type=int, default=5000)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--candidate-seed-base", type=int, default=20260504)
    parser.add_argument("--max-confidence-retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--limit-faults",
        type=int,
        default=0,
        help="Run only the first N pool rows for a smoke test; 0 runs the complete pool.",
    )
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    args.baseline_csv = str(out_dir / "classic10_per_fault.csv")
    args.tandem_csv = str(out_dir / "classic5_ai5_tandem_per_fault.csv")
    args.summary_out = str(out_dir / "summary.json")
    args.report_out = str(out_dir / "detailed_report.md")
    args.manifest_out = str(out_dir / "manifest.json")

    fault_entries, pool_meta = _load_fault_pool(Path(args.fault_list))
    if args.limit_faults:
        if args.limit_faults < 1:
            raise ValueError("--limit-faults must be positive")
        fault_entries = fault_entries[: args.limit_faults]
    args.total_faults = len(fault_entries)

    baseline_path = Path(args.baseline_csv)
    baseline_rows = _load_csv(baseline_path)
    if len(baseline_rows) > len(fault_entries):
        raise RuntimeError("baseline checkpoint contains more rows than the fault pool")
    print(
        f"[setup] faults={len(fault_entries)} circuits="
        f"{len({bench for bench, _ in fault_entries})} baseline_resume={len(baseline_rows)} "
        f"tandem_budget={args.tandem_timeout:g}s+{args.tandem_timeout:g}s",
        flush=True,
    )

    phase_started = time.monotonic()
    baseline_solved = sum(_bool(row, "classic10_ok") for row in baseline_rows)
    current_bench = ""
    circuit = []
    total_gates = 0
    phase_start_index = len(baseline_rows)
    for idx in range(phase_start_index, len(fault_entries)):
        bench_path, fault = fault_entries[idx]
        if bench_path != current_bench:
            current_bench = bench_path
            circuit, total_gates = parse_bench_file(bench_path)
            initialize(circuit, total_gates)
            print(f"[classic10] circuit={bench_path}", flush=True)
        result = _run_classic(
            circuit, total_gates, fault, args.baseline_timeout, args.max_backtracks
        )
        baseline_solved += int(result["ok"])
        baseline_rows.append(
            {
                "fault_index": idx,
                "bench": bench_path,
                "gate_id": int(fault.gate_id),
                "fault_val": int(fault.value),
                "classic10_ok": result["ok"],
                "classic10_result_code": result["result_code"],
                "classic10_backtracks": result["backtracks"],
                "classic10_recursive_calls": result["recursive_calls"],
                "classic10_backtrace_count": result["backtrace_count"],
                "classic10_backtrace_hops": result["backtrace_hops"],
                "classic10_time_s": round(result["time_s"], 6),
            }
        )
        completed = idx + 1
        if completed % args.checkpoint_every == 0:
            _write_csv(baseline_path, BASELINE_FIELDS, baseline_rows)
        if completed % args.progress_every == 0:
            _progress(
                "classic10",
                phase_started,
                completed,
                len(fault_entries),
                baseline_solved,
                phase_start_index,
            )
    _write_csv(baseline_path, BASELINE_FIELDS, baseline_rows)
    print("[classic10] complete; loading AI model for tandem phase", flush=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AiPodemConfig(
        model_path=args.model,
        device=device,
        enable_ai_activation=True,
        enable_ai_propagation=False,
        verbose=False,
        no_fallback=True,
        candidate_count=args.candidate_count,
        candidate_seed_base=args.candidate_seed_base,
        max_confidence_retries=args.max_confidence_retries,
    )
    tandem_path = Path(args.tandem_csv)
    tandem_rows = _load_csv(tandem_path)
    if len(tandem_rows) > len(fault_entries):
        raise RuntimeError("tandem checkpoint contains more rows than the fault pool")
    phase_start_index = len(tandem_rows)
    phase_started = time.monotonic()
    tandem_solved = sum(_bool(row, "tandem_ok") for row in tandem_rows)
    current_bench = ""
    predictor = None
    solver = None
    preloaded_model = None
    for idx in range(phase_start_index, len(fault_entries)):
        bench_path, fault = fault_entries[idx]
        if bench_path != current_bench:
            current_bench = bench_path
            circuit, total_gates = parse_bench_file(bench_path)
            initialize(circuit, total_gates)
            predictor = ModelPairPredictor(
                circuit,
                bench_path,
                config,
                pre_loaded_model=preloaded_model,
            )
            if preloaded_model is None:
                preloaded_model = predictor.model
            solver = HierarchicalReconvSolver(circuit, predictor, circuit_path=bench_path)
            print(f"[classic5+ai5] circuit={bench_path}", flush=True)
        assert solver is not None
        classic = _run_classic(
            circuit, total_gates, fault, args.tandem_timeout, args.max_backtracks
        )
        ai = _run_ai(
            circuit,
            total_gates,
            fault,
            solver,
            args.tandem_timeout,
            args.max_backtracks,
            args.candidate_seed_base + idx,
        )
        tandem_ok = bool(classic["ok"] or ai["ok"])
        if classic["ok"] and ai["ok"]:
            delta = classic["time_s"] - ai["time_s"]
            winner = "tie" if abs(delta) < 1e-6 else ("ai" if delta > 0 else "classic")
            solve_time = min(classic["time_s"], ai["time_s"])
        elif classic["ok"]:
            winner = "classic"
            solve_time = classic["time_s"]
        elif ai["ok"]:
            winner = "ai"
            solve_time = ai["time_s"]
        else:
            winner = "none"
            solve_time = None
        tandem_solved += int(tandem_ok)
        tandem_rows.append(
            {
                "fault_index": idx,
                "bench": bench_path,
                "gate_id": int(fault.gate_id),
                "fault_val": int(fault.value),
                "classic5_ok": classic["ok"],
                "classic5_result_code": classic["result_code"],
                "classic5_backtracks": classic["backtracks"],
                "classic5_recursive_calls": classic["recursive_calls"],
                "classic5_time_s": round(classic["time_s"], 6),
                "ai5_ok": ai["ok"],
                "ai5_result_code": ai["result_code"],
                "ai5_backtracks": ai["backtracks"],
                "ai5_recursive_calls": ai["recursive_calls"],
                "ai5_time_s": round(ai["time_s"], 6),
                "ai5_precheck_success": ai["precheck_success"],
                "ai5_has_reconv_pairs": ai["has_reconv_pairs"],
                "ai5_error": ai["error"],
                "tandem_ok": tandem_ok,
                "tandem_timeout": not tandem_ok,
                "tandem_winner": winner,
                "tandem_solve_time_s": round(solve_time, 6) if solve_time is not None else "",
            }
        )
        completed = idx + 1
        if completed % args.checkpoint_every == 0:
            _write_csv(tandem_path, TANDEM_FIELDS, tandem_rows)
        if completed % args.progress_every == 0:
            _progress(
                "classic5+ai5",
                phase_started,
                completed,
                len(fault_entries),
                tandem_solved,
                phase_start_index,
            )
    _write_csv(tandem_path, TANDEM_FIELDS, tandem_rows)

    summary = _summarize(baseline_rows, tandem_rows, args)
    summary.update(
        {
            "run_id": args.run_id,
            "fault_list": args.fault_list,
            "benches": sorted({bench for bench, _ in fault_entries}),
            "model": args.model,
            "device": device,
            "pool_meta": {
                key: pool_meta.get(key)
                for key in (
                    "selected_faults",
                    "seed",
                    "selection_sha256",
                    "total_candidate_reconvergent_faults",
                )
            },
        }
    )
    _write_json(Path(args.summary_out), summary)
    _write_report(Path(args.report_out), summary, args)
    _write_json(
        Path(args.manifest_out),
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "command": [sys.executable, "-m", "scripts.benchmark_tandem_fault_pool", *sys.argv[1:]],
            "cwd": os.getcwd(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": _git_value(["rev-parse", "HEAD"]),
            "git_status_short": _git_value(["status", "--short"]),
            "outputs": [
                args.baseline_csv,
                args.tandem_csv,
                args.summary_out,
                args.report_out,
                args.manifest_out,
            ],
        },
    )
    print(
        f"[complete] tandem={summary['tandem']['union_solved']}/{len(fault_entries)} "
        f"ai_only={summary['tandem']['ai_only_solved']} report={args.report_out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
