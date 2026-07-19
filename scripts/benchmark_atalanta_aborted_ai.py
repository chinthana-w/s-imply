"""Run checkpointed AI-guided PODEM on a multi-circuit Atalanta-aborted pool."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from scripts.benchmark_tandem_fault_pool import _run_ai
from src.atpg.ai_podem import AiPodemConfig, HierarchicalReconvSolver, ModelPairPredictor
from src.atpg.podem import initialize
from src.util.io import parse_bench_file
from src.util.struct import Fault, LogicValue


FIELDS = (
    "fault_index",
    "circuit",
    "bench",
    "gate_id",
    "stuck_at",
    "fault_val",
    "atalanta_backtracks",
    "atalanta_backtracks_min",
    "atalanta_backtracks_max",
    "atalanta_time_s_max",
    "ai_ok",
    "ai_result_code",
    "ai_backtracks",
    "ai_recursive_calls",
    "ai_time_s",
    "ai_precheck_success",
    "ai_has_reconv_pairs",
    "ai_error",
    "backtrack_reduction",
    "backtrack_reduction_pct",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(row: dict, key: str) -> bool:
    return str(row.get(key, "")).lower() in {"1", "true", "yes"}


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "total": 0, "mean": None, "median": None, "p95": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _summary(rows: list[dict], total: int, args: argparse.Namespace) -> dict:
    completed = len(rows)
    solved = sum(_bool(row, "ai_ok") for row in rows)
    successful = [row for row in rows if _bool(row, "ai_ok")]
    source_bt = [int(row["atalanta_backtracks"]) for row in rows]
    ai_bt = [int(row["ai_backtracks"]) for row in rows]
    successful_ai_bt = [int(row["ai_backtracks"]) for row in successful]
    successful_source_bt = [int(row["atalanta_backtracks"]) for row in successful]
    per_circuit = defaultdict(lambda: {"attempted": 0, "solved": 0, "ai_backtracks": 0})
    for row in rows:
        item = per_circuit[row["circuit"]]
        item["attempted"] += 1
        item["solved"] += int(_bool(row, "ai_ok"))
        item["ai_backtracks"] += int(row["ai_backtracks"])
    for item in per_circuit.values():
        item["coverage"] = item["solved"] / max(1, item["attempted"])
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "fault_pool": args.fault_list,
        "model": args.model,
        "device": args.device,
        "timeout_s": args.timeout,
        "max_backtracks": args.max_backtracks,
        "total_faults": total,
        "completed": completed,
        "remaining": total - completed,
        "ai_solved": solved,
        "ai_failed": completed - solved,
        "ai_coverage_completed": solved / max(1, completed),
        "precheck_solved": sum(_bool(row, "ai_precheck_success") for row in rows),
        "faults_with_reconv_pairs": sum(_bool(row, "ai_has_reconv_pairs") for row in rows),
        "atalanta_backtracks_all_completed": _distribution(source_bt),
        "ai_backtracks_all_completed": _distribution(ai_bt),
        "atalanta_backtracks_on_ai_success": _distribution(successful_source_bt),
        "ai_backtracks_on_ai_success": _distribution(successful_ai_bt),
        "backtrack_reduction_on_ai_success": (
            1 - sum(successful_ai_bt) / sum(successful_source_bt)
            if sum(successful_source_bt)
            else None
        ),
        "ai_successes_with_fewer_backtracks": sum(
            int(row["ai_backtracks"]) < int(row["atalanta_backtracks"])
            for row in successful
        ),
        "result_codes": dict(Counter(row["ai_result_code"] for row in rows)),
        "per_circuit": dict(sorted(per_circuit.items())),
    }


def _eta(elapsed: float, done: int, total: int) -> str:
    if done <= 0:
        return "unknown"
    seconds = int(max(0, elapsed / done * (total - done)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault-list", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--max-backtracks", type=int, default=100000)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--candidate-seed-base", type=int, default=20260716)
    parser.add_argument("--max-confidence-retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--limit-faults", type=int, default=0)
    parser.add_argument("--run-id", default="atalanta_aborted_ai")
    args = parser.parse_args()

    with Path(args.fault_list).open() as handle:
        pool = json.load(handle)
    faults = pool["faults"][: args.limit_faults or None]
    out_dir = Path(args.out_dir)
    csv_path = out_dir / "ai_per_fault.csv"
    summary_path = out_dir / "summary.json"
    rows = _load_rows(csv_path)
    if len(rows) > len(faults):
        raise RuntimeError("checkpoint has more rows than the selected fault pool")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
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

    print(
        f"[setup] faults={len(faults)} resume={len(rows)} timeout={args.timeout:g}s "
        f"max_backtracks={args.max_backtracks} device={device}",
        flush=True,
    )
    started = time.monotonic()
    resume = len(rows)
    current_bench = ""
    predictor = solver = preloaded_model = None
    for index in range(resume, len(faults)):
        item = faults[index]
        bench = item["bench"]
        if bench != current_bench:
            current_bench = bench
            circuit, total_gates = parse_bench_file(bench)
            initialize(circuit, total_gates)
            predictor = ModelPairPredictor(
                circuit, bench, config, pre_loaded_model=preloaded_model
            )
            if preloaded_model is None:
                preloaded_model = predictor.model
            solver = HierarchicalReconvSolver(circuit, predictor, circuit_path=bench)
            print(f"[ai] circuit={item['circuit']} bench={bench}", flush=True)
        assert solver is not None
        fault = Fault(int(item["gate_id"]), LogicValue(int(item["fault_val"])))
        result = _run_ai(
            circuit,
            total_gates,
            fault,
            solver,
            args.timeout,
            args.max_backtracks,
            args.candidate_seed_base + index,
        )
        source_backtracks = int(item["atalanta_backtracks_representative"])
        reduction = source_backtracks - int(result["backtracks"])
        rows.append(
            {
                "fault_index": index,
                "circuit": item["circuit"],
                "bench": bench,
                "gate_id": item["gate_id"],
                "stuck_at": item["stuck_at"],
                "fault_val": item["fault_val"],
                "atalanta_backtracks": source_backtracks,
                "atalanta_backtracks_min": item["atalanta_backtracks_min"],
                "atalanta_backtracks_max": item["atalanta_backtracks_max"],
                "atalanta_time_s_max": item["atalanta_time_s_max"] or "",
                "ai_ok": result["ok"],
                "ai_result_code": result["result_code"],
                "ai_backtracks": result["backtracks"],
                "ai_recursive_calls": result["recursive_calls"],
                "ai_time_s": round(result["time_s"], 6),
                "ai_precheck_success": result["precheck_success"],
                "ai_has_reconv_pairs": result["has_reconv_pairs"],
                "ai_error": result["error"],
                "backtrack_reduction": reduction,
                "backtrack_reduction_pct": reduction / source_backtracks,
            }
        )
        completed = index + 1
        if completed % args.checkpoint_every == 0:
            _write_csv(csv_path, rows)
            _write_json(summary_path, _summary(rows, len(faults), args))
        if completed % args.progress_every == 0:
            elapsed = time.monotonic() - started
            solved = sum(_bool(row, "ai_ok") for row in rows)
            print(
                f"[ai] progress={completed}/{len(faults)} ({completed/len(faults):.2%}) "
                f"solved={solved} coverage={solved/completed:.2%} "
                f"eta={_eta(elapsed, completed-resume, len(faults)-resume)}",
                flush=True,
            )
    _write_csv(csv_path, rows)
    _write_json(summary_path, _summary(rows, len(faults), args))
    print(f"[complete] rows={len(rows)} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
