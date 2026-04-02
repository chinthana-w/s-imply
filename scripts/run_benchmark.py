"""
Full ISCAS85 benchmark: Vanilla PODEM vs AI-PODEM.

Usage:
    conda activate deepgate
    python scripts/run_benchmark.py \
        --model checkpoints/supervised_v4/best_model.pth \
        --bench_dir data/bench/ISCAS85 \
        --timeout 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def run_vanilla(circuit, total_gates, faults, timeout):
    succ = 0
    backtracks = 0
    t0 = time.time()
    initialize(circuit, total_gates)
    for fault in faults:
        reset_gates(circuit, total_gates)
        reset_statistics()
        if podem(circuit, fault, total_gates, backtrace_func=simple_backtrace, timeout=timeout):
            succ += 1
        stats = get_statistics()
        backtracks += stats.get("backtrack_count", 0)
    elapsed = time.time() - t0
    return succ, backtracks, elapsed


def run_ai(circuit, total_gates, faults, bench_path, model_path, device, timeout):
    config = AiPodemConfig(model_path=model_path, device=device)
    predictor = ModelPairPredictor(circuit, bench_path, config)
    solver = HierarchicalReconvSolver(circuit, predictor)
    succ = 0
    backtracks = 0
    t0 = time.time()
    for fault in faults:
        reset_statistics()
        # ai_podem calls initialize + reset_gates internally
        if ai_podem(
            circuit,
            fault,
            total_gates,
            model_path=model_path,
            circuit_path=bench_path,
            predictor=predictor,
            solver=solver,
        ):
            succ += 1
        stats = get_statistics()
        backtracks += stats.get("backtrack_count", 0)
    elapsed = time.time() - t0
    return succ, backtracks, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--bench_dir", default="data/bench/ISCAS85")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-fault timeout (s)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Bench: {args.bench_dir}")
    print(f"Timeout: {args.timeout}s/fault\n")

    bench_files = sorted(
        f for f in os.listdir(args.bench_dir) if f.endswith(".bench")
    )

    header = f"{'Circuit':<12} {'Faults':>6}  {'Vanilla%':>8}  {'AI%':>8}  {'BT-Van':>8}  {'BT-AI':>8}  {'T-Van(s)':>9}  {'T-AI(s)':>8}"
    print(header)
    print("-" * len(header))

    total_faults = total_van_succ = total_ai_succ = 0
    total_van_bt = total_ai_bt = 0
    total_van_t = total_ai_t = 0.0

    for bench_file in bench_files:
        bench_path = os.path.join(args.bench_dir, bench_file)
        circuit, total_gates = parse_bench_file(bench_path)
        faults = get_all_faults(circuit, total_gates)
        n = len(faults)

        van_succ, van_bt, van_t = run_vanilla(circuit, total_gates, faults, args.timeout)
        try:
            ai_succ, ai_bt, ai_t = run_ai(
                circuit, total_gates, faults, bench_path, args.model, device, args.timeout
            )
        except Exception as e:
            print(f"{bench_file:<12}  ERROR in AI: {e}")
            import traceback
            traceback.print_exc()
            continue

        name = bench_file.replace(".bench", "")
        van_pct = van_succ / n * 100
        ai_pct = ai_succ / n * 100
        print(
            f"{name:<12} {n:>6}  {van_pct:>7.1f}%  {ai_pct:>7.1f}%  "
            f"{van_bt:>8d}  {ai_bt:>8d}  {van_t:>9.1f}  {ai_t:>8.1f}",
            flush=True,
        )

        total_faults += n
        total_van_succ += van_succ
        total_ai_succ += ai_succ
        total_van_bt += van_bt
        total_ai_bt += ai_bt
        total_van_t += van_t
        total_ai_t += ai_t

    print("-" * len(header))
    print(
        f"{'TOTAL':<12} {total_faults:>6}  "
        f"{total_van_succ/total_faults*100:>7.1f}%  "
        f"{total_ai_succ/total_faults*100:>7.1f}%  "
        f"{total_van_bt:>8d}  {total_ai_bt:>8d}  "
        f"{total_van_t:>9.1f}  {total_ai_t:>8.1f}"
    )


if __name__ == "__main__":
    main()
