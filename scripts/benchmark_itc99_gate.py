"""Benchmark AI-PODEM on the deterministic ITC99 gate subset.

This is the cheap held-out gate before running the full ITC99 benchmark.  It
never builds training data and never feeds results back into training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.atpg.ai_podem import AiPodemConfig, HierarchicalReconvSolver, ModelPairPredictor, ai_podem
from src.atpg.logic_sim_three import reset_gates
from src.atpg.podem import get_all_faults, initialize
from src.util.io import parse_bench_file
from src.util.struct import Fault, LogicValue


def _load_gate_faults(path: str) -> tuple[str, list[Fault], dict]:
    with open(path) as f:
        payload = json.load(f)
    faults = [
        Fault(int(item["gate_id"]), LogicValue(int(item["fault_val"])))
        for item in payload["faults"]
    ]
    return payload["bench"], faults, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the ITC99 10% gate subset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fault-list", default="data/bench/ITC99/b17_gate_10pct_faults.json")
    parser.add_argument("--out", default="docs/itc99_gate_report.json")
    parser.add_argument("--max-backtracks", type=int, default=5000)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--candidate-seed-base", type=int, default=20260504)
    parser.add_argument("--full", action="store_true", help="Ignore fault-list and run all faults")
    args = parser.parse_args()

    if args.full:
        with open(args.fault_list) as f:
            payload = json.load(f)
        bench_path = payload["bench"]
        circuit, total_gates = parse_bench_file(bench_path)
        faults = get_all_faults(circuit, total_gates)
        gate_meta = {"bench": bench_path, "selected_faults": len(faults), "full": True}
    else:
        bench_path, faults, gate_meta = _load_gate_faults(args.fault_list)
        circuit, total_gates = parse_bench_file(bench_path)

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
    )
    predictor = ModelPairPredictor(circuit, bench_path, config)
    solver = HierarchicalReconvSolver(circuit, predictor)

    succeeded = 0
    failed = 0
    total_time = 0.0
    per_fault = []

    initialize(circuit, total_gates)
    for idx, fault in enumerate(faults):
        reset_gates(circuit, total_gates)
        t0 = time.time()
        ok = ai_podem(
            circuit,
            fault,
            total_gates,
            predictor=predictor,
            solver=solver,
            enable_ai_activation=True,
            enable_ai_propagation=False,
            no_fallback=True,
            max_backtracks=args.max_backtracks,
            seed=args.candidate_seed_base + idx,
        )
        elapsed = time.time() - t0
        total_time += elapsed
        succeeded += int(ok)
        failed += int(not ok)
        per_fault.append({
            "gate_id": int(fault.gate_id),
            "fault_val": int(fault.value),
            "ok": bool(ok),
            "time_s": round(elapsed, 4),
        })
        if (idx + 1) % 100 == 0:
            print(
                f"ITC99 gate progress {idx + 1}/{len(faults)} "
                f"coverage={succeeded / (idx + 1):.2%}",
                flush=True,
            )

    report = {
        "model": args.model,
        "bench": bench_path,
        "fault_list": args.fault_list,
        "gate_meta": gate_meta,
        "candidate_count": args.candidate_count,
        "candidate_seed_base": args.candidate_seed_base,
        "max_backtracks": args.max_backtracks,
        "total": len(faults),
        "succeeded": succeeded,
        "failed": failed,
        "coverage": succeeded / max(1, len(faults)),
        "total_time_s": round(total_time, 2),
        "per_fault": per_fault,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp_path = args.out + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, args.out)
    print(
        f"ITC99 gate coverage: {succeeded}/{len(faults)} "
        f"= {succeeded / max(1, len(faults)):.2%}; wrote {args.out}"
    )


if __name__ == "__main__":
    main()
