"""
Benchmark AI-PODEM (no fallback) vs Vanilla PODEM on all ISCAS85 circuits.

For each circuit:
  1. Identify reconvergent faults (solver returns a non-None assignment).
  2. Run AI-PODEM with no_fallback=True; record which faults succeed.
  3. Run Vanilla PODEM on exactly that matched set.
  4. Compare backtrack counts and wall-clock time.

Results are saved per-circuit to --out_dir as JSON, then compiled into a
Markdown report at --report.

Usage:
    python scripts/benchmark_iscas85_nofallback.py \\
        --model checkpoints/supervised_v4/best_model.pth \\
        --bench_dir data/bench/ISCAS85 \\
        --out_dir /tmp/iscas85_bench \\
        --report docs/benchmark_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.setrecursionlimit(200000)

import torch

from src.atpg.ai_podem import AiPodemConfig, HierarchicalReconvSolver, ModelPairPredictor, ai_podem
from src.atpg.logic_sim_three import reset_gates
from src.atpg.podem import (
    get_all_faults,
    get_statistics,
    initialize,
    podem,
    reset_statistics,
    simple_backtrace,
)
from src.util.io import parse_bench_file


def benchmark_circuit(bench_path: str, model_path: str, device: str, max_backtracks: int = 5000) -> dict:
    name = os.path.basename(bench_path).replace(".bench", "")
    print(f"\n[{name}] Parsing circuit...", flush=True)
    circuit, total_gates = parse_bench_file(bench_path)
    faults = get_all_faults(circuit, total_gates)
    print(f"[{name}] {len(faults)} faults total", flush=True)

    cfg = AiPodemConfig(model_path=model_path, device=device)
    pred = ModelPairPredictor(circuit, bench_path, cfg)
    solver = HierarchicalReconvSolver(circuit, pred)

    # Step 1: identify reconvergent faults via topology only (no model inference)
    print(f"[{name}] Identifying reconvergent faults (topology)...", flush=True)
    reconv_faults = []
    for f in faults:
        # _collect_and_sort_pairs uses BFS only — no model inference
        pairs = solver._collect_and_sort_pairs(f.gate_id)
        if pairs:
            reconv_faults.append(f)
    print(f"[{name}] {len(reconv_faults)}/{len(faults)} reconvergent", flush=True)

    if not reconv_faults:
        return {
            "circuit": name,
            "total_faults": len(faults),
            "reconv_faults": 0,
            "ai_succeeded": 0,
            "ai_failed": 0,
            "ai_bt": 0,
            "ai_time": 0.0,
            "van_bt": 0,
            "van_time": 0.0,
        }

    # Step 2: AI no_fallback — record successes
    print(f"[{name}] Running AI-PODEM (no fallback)...", flush=True)
    ai_succeeded_faults = []
    ai_bt_list = []
    ai_t_list = []
    ai_failed_count = 0

    for i, f in enumerate(reconv_faults):
        reset_statistics()
        t0 = time.time()
        r = ai_podem(
            circuit, f, total_gates,
            model_path=model_path,
            circuit_path=bench_path,
            predictor=pred,
            solver=solver,
            no_fallback=True,
            max_backtracks=max_backtracks,
        )
        elapsed = time.time() - t0
        bt = get_statistics().get("backtrack_count", 0)
        if r:
            ai_succeeded_faults.append(f)
            ai_bt_list.append(bt)
            ai_t_list.append(elapsed)
        else:
            ai_failed_count += 1
        if (i + 1) % 100 == 0:
            total_so_far = len(ai_succeeded_faults) + ai_failed_count
            print(
                f"[{name}]  AI progress: {i+1}/{len(reconv_faults)}  "
                f"succ={len(ai_succeeded_faults)}  fail={ai_failed_count}  "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    print(
        f"[{name}] AI done: {len(ai_succeeded_faults)} succeeded, "
        f"{ai_failed_count} failed",
        flush=True,
    )

    if not ai_succeeded_faults:
        return {
            "circuit": name,
            "total_faults": len(faults),
            "reconv_faults": len(reconv_faults),
            "ai_succeeded": 0,
            "ai_failed": ai_failed_count,
            "ai_bt": 0,
            "ai_time": 0.0,
            "van_bt": 0,
            "van_time": 0.0,
        }

    # Step 3: Vanilla PODEM on exactly the matched set
    print(f"[{name}] Running Vanilla PODEM on {len(ai_succeeded_faults)} matched faults...", flush=True)
    initialize(circuit, total_gates)
    van_bt_total = 0
    van_t_total = 0.0
    for f in ai_succeeded_faults:
        reset_gates(circuit, total_gates)
        reset_statistics()
        t0 = time.time()
        podem(circuit, f, total_gates, backtrace_func=simple_backtrace, timeout=30.0)
        van_t_total += time.time() - t0
        van_bt_total += get_statistics().get("backtrack_count", 0)

    ai_bt_total = sum(ai_bt_list)
    ai_t_total = sum(ai_t_list)

    bt_reduction = (1 - ai_bt_total / max(van_bt_total, 1)) * 100
    speedup = van_t_total / max(ai_t_total, 0.001)

    print(
        f"[{name}] Vanilla bt={van_bt_total}  AI bt={ai_bt_total}  "
        f"reduction={bt_reduction:.1f}%  speedup={speedup:.2f}x",
        flush=True,
    )

    return {
        "circuit": name,
        "total_faults": len(faults),
        "reconv_faults": len(reconv_faults),
        "ai_succeeded": len(ai_succeeded_faults),
        "ai_failed": ai_failed_count,
        "ai_bt": ai_bt_total,
        "ai_time": round(ai_t_total, 2),
        "van_bt": van_bt_total,
        "van_time": round(van_t_total, 2),
        "bt_reduction_pct": round(bt_reduction, 1),
        "speedup": round(speedup, 2),
        "max_backtracks": max_backtracks,
    }


def write_report(results: list[dict], model_path: str, report_path: str) -> None:
    import datetime

    lines = []
    lines.append("# AI-PODEM Benchmark Report — ISCAS85")
    lines.append("")
    lines.append(f"**Model:** `{model_path}`  ")
    lines.append(f"**Date:** {datetime.date.today()}  ")
    lines.append("**Mode:** no-fallback (AI-only, no vanilla retry)  ")
    lines.append(f"**Max backtracks per attempt:** {results[0].get('max_backtracks', 500)}  ")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    # Aggregate totals
    total_faults = sum(r["total_faults"] for r in results)
    total_reconv = sum(r["reconv_faults"] for r in results)
    total_ai_succ = sum(r["ai_succeeded"] for r in results)
    total_ai_fail = sum(r["ai_failed"] for r in results)
    total_ai_bt = sum(r["ai_bt"] for r in results)
    total_van_bt = sum(r["van_bt"] for r in results)
    total_ai_t = sum(r["ai_time"] for r in results)
    total_van_t = sum(r["van_time"] for r in results)

    overall_bt_red = (1 - total_ai_bt / max(total_van_bt, 1)) * 100
    overall_speedup = total_van_t / max(total_ai_t, 0.001)
    ai_coverage = total_ai_succ / max(total_reconv, 1) * 100

    lines.append(f"- **Total faults:** {total_faults:,}")
    lines.append(f"- **Reconvergent faults:** {total_reconv:,} ({total_reconv/total_faults*100:.1f}% of all faults)")
    lines.append(f"- **AI-PODEM coverage** (no fallback, on reconv faults): {total_ai_succ:,}/{total_reconv:,} = **{ai_coverage:.1f}%**")
    lines.append(f"- **Backtrack reduction** (on matched faults): **{overall_bt_red:.1f}%**")
    lines.append(f"- **Speedup** (on matched faults): **{overall_speedup:.2f}×**")
    lines.append("")

    lines.append("## Per-Circuit Results")
    lines.append("")
    lines.append("### Coverage (no fallback)")
    lines.append("")
    lines.append("| Circuit | Total Faults | Reconv Faults | AI Succeeded | AI Failed | AI Coverage |")
    lines.append("|---------|-------------|---------------|--------------|-----------|-------------|")
    for r in results:
        cov = r["ai_succeeded"] / max(r["reconv_faults"], 1) * 100
        lines.append(
            f"| {r['circuit']} | {r['total_faults']:,} | {r['reconv_faults']:,} "
            f"| {r['ai_succeeded']:,} | {r['ai_failed']:,} | {cov:.1f}% |"
        )
    # totals row
    cov_tot = total_ai_succ / max(total_reconv, 1) * 100
    lines.append(
        f"| **TOTAL** | **{total_faults:,}** | **{total_reconv:,}** "
        f"| **{total_ai_succ:,}** | **{total_ai_fail:,}** | **{cov_tot:.1f}%** |"
    )
    lines.append("")

    lines.append("### Backtrack Comparison (on matched faults only)")
    lines.append("")
    lines.append("| Circuit | Matched Faults | Vanilla BT | AI BT | BT Reduction | Speedup |")
    lines.append("|---------|---------------|-----------|-------|--------------|---------|")
    for r in results:
        if r["ai_succeeded"] == 0:
            lines.append(
                f"| {r['circuit']} | 0 | — | — | — | — |"
            )
        else:
            lines.append(
                f"| {r['circuit']} | {r['ai_succeeded']:,} "
                f"| {r['van_bt']:,} | {r['ai_bt']:,} "
                f"| {r['bt_reduction_pct']:+.1f}% | {r['speedup']:.2f}× |"
            )
    lines.append(
        f"| **TOTAL** | **{total_ai_succ:,}** "
        f"| **{total_van_bt:,}** | **{total_ai_bt:,}** "
        f"| **{overall_bt_red:+.1f}%** | **{overall_speedup:.2f}×** |"
    )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- *Reconvergent fault*: a fault where the AI solver finds a non-trivial reconvergent path structure and returns an assignment.")
    lines.append("- *AI Succeeded*: AI-PODEM solved the fault within `max_backtracks=5000` using only the AI-generated assignment (no vanilla PODEM fallback).")
    lines.append("- *Matched faults*: the exact subset of faults where AI succeeded — vanilla is evaluated on the same set for a fair comparison.")
    lines.append("- *BT Reduction*: `(1 - AI_backtracks / Vanilla_backtracks) × 100%` on the matched set.")
    lines.append("")

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/supervised_v4/best_model.pth")
    parser.add_argument("--bench_dir", default="data/bench/ISCAS85")
    parser.add_argument("--out_dir", default="/tmp/iscas85_bench")
    parser.add_argument("--report", default="docs/benchmark_report.md")
    parser.add_argument("--circuits", nargs="*", help="Subset of circuits to run (e.g. c17 c432)")
    parser.add_argument("--max_backtracks", type=int, default=500,
                        help="Max PODEM backtracks per AI attempt (default 500 for fast failure)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Model: {args.model}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    bench_files = sorted(f for f in os.listdir(args.bench_dir) if f.endswith(".bench"))
    if args.circuits:
        bench_files = [f for f in bench_files if f.replace(".bench", "") in args.circuits]

    results = []
    for bench_file in bench_files:
        name = bench_file.replace(".bench", "")
        cache_path = os.path.join(args.out_dir, f"{name}.json")

        # Resume from cached result if available
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                r = json.load(f)
            print(f"[{name}] Loaded from cache: {cache_path}", flush=True)
            results.append(r)
            continue

        bench_path = os.path.join(args.bench_dir, bench_file)
        try:
            r = benchmark_circuit(bench_path, args.model, device, args.max_backtracks)
        except Exception as e:
            import traceback
            print(f"[{name}] ERROR: {e}", flush=True)
            traceback.print_exc()
            continue

        with open(cache_path, "w") as f:
            json.dump(r, f, indent=2)
        results.append(r)

        # Write partial report after each circuit
        write_report(results, args.model, args.report)

    write_report(results, args.model, args.report)


if __name__ == "__main__":
    main()
