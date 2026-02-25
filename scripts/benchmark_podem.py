"""
Benchmark: Classic PODEM vs AI-PODEM on a single fault.

Each run parses a fresh copy of the circuit to avoid state pollution across repeats.

Usage:
    conda run -n deepgate python -m scripts.benchmark_podem \\
        --bench data/bench/ISCAS85/c432.bench \\
        --model checkpoints/reconv_max_occupancy/best_model.pth \\
        --gate 259 --sa 1 --repeats 3
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.atpg.ai_podem import (
    AiPodemConfig,
    HierarchicalReconvSolver,
    ModelPairPredictor,
    ai_podem,
)
from src.atpg.podem import get_statistics, podem, reset_statistics
from src.util.io import parse_bench_file
from src.util.struct import Fault, LogicValue


@contextlib.contextmanager
def _silence():
    """Suppress all stdout and stderr inside the block."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _fresh_circuit(bench_path: str):
    """Return a freshly parsed (or deep-copied) circuit each call to avoid state pollution."""
    return parse_bench_file(bench_path)


def run_classic_podem(bench_path: str, fault: Fault) -> tuple[bool, float, int]:
    """Classic PODEM on a fresh circuit copy, timed, silent."""
    import src.atpg.podem as podem_mod

    circuit, total_gates = _fresh_circuit(bench_path)
    # Force full re-init (SCOAP, topological order) for this circuit copy
    podem_mod.scoap_calculated = False
    reset_statistics()

    t0 = time.perf_counter()
    with _silence():
        result = podem(circuit, fault, total_gates)
    elapsed = time.perf_counter() - t0
    stats = get_statistics()
    return bool(result), elapsed, stats["backtrack_count"]


def run_ai_podem(
    bench_path: str,
    fault: Fault,
    circuit_path: str,
    device: str,
    model_path: str,
) -> tuple[bool, float, int]:
    """AI-PODEM (AI activation, classic PODEM propagation) on a fresh circuit, timed, silent."""
    circuit, total_gates = _fresh_circuit(bench_path)

    config = AiPodemConfig(
        model_path=model_path,
        device=device,
        enable_ai_activation=True,
        enable_ai_propagation=False,
    )

    with _silence():
        predictor = ModelPairPredictor(circuit, circuit_path, config)
    solver = HierarchicalReconvSolver(circuit, predictor, verbose=False)
    reset_statistics()

    t0 = time.perf_counter()
    with _silence():
        result = ai_podem(
            circuit,
            fault,
            total_gates,
            predictor=predictor,
            solver=solver,
            enable_ai_activation=True,
            enable_ai_propagation=False,
            verbose=False,
        )
    elapsed = time.perf_counter() - t0
    stats = get_statistics()
    return bool(result), elapsed, stats["backtrack_count"]


def main():
    parser = argparse.ArgumentParser(description="Benchmark Classic PODEM vs AI-PODEM")
    parser.add_argument(
        "--bench",
        type=str,
        default="data/bench/ISCAS85/c432.bench",
        help="Path to .bench file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/reconv_max_occupancy/best_model.pth",
        help="Path to AI model checkpoint",
    )
    parser.add_argument("--gate", type=int, default=329, help="Fault gate ID")
    parser.add_argument("--sa", type=int, default=1, choices=[0, 1], help="Stuck-at value (0 or 1)")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed repeats per method")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SA0 → LogicValue.D (good machine output is 1, D)
    # SA1 → LogicValue.DB (good machine output is 0, D̄)
    fault_val = LogicValue.DB if args.sa == 1 else LogicValue.D
    fault = Fault(args.gate, fault_val)

    print(f"Circuit  : {args.bench}")
    print(f"Fault    : Gate {args.gate}  stuck-at-{args.sa}  ({fault_val})")
    print(f"Device   : {device}")
    print(f"Repeats  : {args.repeats}")
    print()

    # ------------------------------------------------------------------ #
    #  Timed runs  (each run gets a fresh circuit to eliminate state bugs)
    # ------------------------------------------------------------------ #
    classic_times: list[float] = []
    classic_ok: list[bool] = []
    classic_bt: list[int] = []
    ai_times: list[float] = []
    ai_ok: list[bool] = []
    ai_bt: list[int] = []

    for r in range(args.repeats):
        ok_c, t_c, bt_c = run_classic_podem(args.bench, fault)
        classic_times.append(t_c)
        classic_ok.append(ok_c)
        classic_bt.append(bt_c)

        ok_a, t_a, bt_a = run_ai_podem(args.bench, fault, args.bench, device, args.model)
        ai_times.append(t_a)
        ai_ok.append(ok_a)
        ai_bt.append(bt_a)

        print(
            f"  Run {r + 1}/{args.repeats}"
            f"  |  Classic PODEM : {'OK  ' if ok_c else 'FAIL'}  {t_c:.4f}s  BT: {bt_c}"
            f"  |  AI-PODEM : {'OK  ' if ok_a else 'FAIL'}  {t_a:.4f}s  BT: {bt_a}"
        )

    # ------------------------------------------------------------------ #
    #  Summary
    # ------------------------------------------------------------------ #
    avg_c = sum(classic_times) / len(classic_times)
    avg_a = sum(ai_times) / len(ai_times)
    avg_bt_c = sum(classic_bt) / len(classic_bt)
    avg_bt_a = sum(ai_bt) / len(ai_bt)
    speedup = avg_c / avg_a if avg_a > 0 else float("inf")

    c_pass = sum(classic_ok)
    a_pass = sum(ai_ok)

    print()
    print("=" * 72)
    print(f"{'Method':<20} {'Pass':>6} {'Avg time (s)':>14} {'Avg BT':>10}  {'vs Classic':>10}")
    print("-" * 72)
    print(
        f"{'Classic PODEM':<20} {c_pass}/{args.repeats:>3} {avg_c:>14.4f} "
        f"{avg_bt_c:>10.1f}  {'1.00×':>10}"
    )
    print(
        f"{'AI-PODEM':<20} {a_pass}/{args.repeats:>3} {avg_a:>14.4f} "
        f"{avg_bt_a:>10.1f}  {speedup:>9.2f}×"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
