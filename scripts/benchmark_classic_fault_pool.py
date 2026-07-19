"""Benchmark classic PODEM on a JSON fault pool."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.atpg.logic_sim_three import reset_gates
from src.atpg.podem import (
    BACKTRACK_LIMIT,
    SUCCESS,
    TIMEOUT,
    get_statistics,
    initialize,
    podem,
    reset_statistics,
    simple_backtrace,
)
from src.util.io import parse_bench_file
from src.util.struct import Fault, LogicValue


def _git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fault_index",
        "gate_id",
        "fault_val",
        "classic_ok",
        "classic_result_code",
        "classic_backtracks",
        "classic_recursive_calls",
        "classic_backtrace_count",
        "classic_backtrace_hops",
        "classic_time_s",
    ]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _load_fault_pool(path: Path) -> tuple[str, list[Fault], dict]:
    with path.open() as f:
        payload = json.load(f)
    faults = [
        Fault(int(item["gate_id"]), LogicValue(int(item["fault_val"])))
        for item in payload["faults"]
    ]
    return payload["bench"], faults, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark classic PODEM on a fault pool")
    parser.add_argument("--fault-list", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--classic-timeout", type=float, default=5.0)
    parser.add_argument("--max-backtracks", type=int, default=5000)
    parser.add_argument("--start-from-fault", type=int, default=0)
    parser.add_argument("--stop-before-fault", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    fault_list = Path(args.fault_list)
    out_path = Path(args.out)
    csv_path = Path(args.csv_out)
    manifest_path = Path(args.manifest_out) if args.manifest_out else None

    bench_path, faults, pool_meta = _load_fault_pool(fault_list)
    start_from = max(0, args.start_from_fault)
    stop_before = int(args.stop_before_fault or len(faults))
    if stop_before > len(faults):
        stop_before = len(faults)
    if stop_before <= start_from:
        raise ValueError("--stop-before-fault must be greater than --start-from-fault")

    circuit, total_gates = parse_bench_file(bench_path)
    initialize(circuit, total_gates)

    rows: list[dict] = []
    succeeded = 0
    result_counts: dict[int, int] = {}
    total_time = 0.0
    total_backtracks = 0
    total_recursive_calls = 0

    for idx in range(start_from, stop_before):
        fault = faults[idx]
        reset_gates(circuit, total_gates)
        reset_statistics()
        start = time.time()
        result = podem(
            circuit,
            fault,
            total_gates,
            backtrace_func=simple_backtrace,
            timeout=args.classic_timeout,
            max_backtracks=args.max_backtracks,
        )
        elapsed = time.time() - start
        stats = get_statistics()
        ok = int(result) == SUCCESS
        succeeded += int(ok)
        result_counts[int(result)] = result_counts.get(int(result), 0) + 1
        total_time += elapsed
        total_backtracks += int(stats.get("backtrack_count", 0))
        total_recursive_calls += int(stats.get("total_recursive_calls", 0))
        rows.append(
            {
                "fault_index": idx,
                "gate_id": int(fault.gate_id),
                "fault_val": int(fault.value),
                "classic_ok": ok,
                "classic_result_code": int(result),
                "classic_backtracks": int(stats.get("backtrack_count", 0)),
                "classic_recursive_calls": int(stats.get("total_recursive_calls", 0)),
                "classic_backtrace_count": int(stats.get("backtrace_count", 0)),
                "classic_backtrace_hops": int(stats.get("backtrace_hops", 0)),
                "classic_time_s": round(elapsed, 6),
            }
        )

        attempted = idx - start_from + 1
        if args.checkpoint_every > 0 and attempted % args.checkpoint_every == 0:
            _write_csv(csv_path, rows)
        if args.progress_every > 0 and attempted % args.progress_every == 0:
            print(
                f"classic progress {idx + 1}/{len(faults)} "
                f"shard_attempted={attempted}/{stop_before - start_from} "
                f"coverage={succeeded / attempted:.2%}",
                flush=True,
            )

    attempted = len(rows)
    coverage = succeeded / attempted if attempted else 0.0
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "command": [sys.executable, "-m", "scripts.benchmark_classic_fault_pool", *sys.argv[1:]],
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_status_short": _git_value(["status", "--short"]),
        "fault_list": str(fault_list),
        "bench": bench_path,
        "pool_meta": {
            "selected_faults": pool_meta.get("selected_faults"),
            "seed": pool_meta.get("seed"),
            "selection_sha256": pool_meta.get("selection_sha256"),
            "total_candidate_reconvergent_faults": pool_meta.get(
                "total_candidate_reconvergent_faults"
            ),
        },
        "start_from_fault": start_from,
        "stop_before_fault": stop_before,
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": attempted - succeeded,
        "coverage": coverage,
        "classic_timeout": args.classic_timeout,
        "max_backtracks": args.max_backtracks,
        "result_code_counts": {str(k): v for k, v in sorted(result_counts.items())},
        "result_code_labels": {
            str(SUCCESS): "SUCCESS",
            str(TIMEOUT): "TIMEOUT",
            str(BACKTRACK_LIMIT): "BACKTRACK_LIMIT",
            "0": "UNTESTABLE",
        },
        "classic_backtracks_total": total_backtracks,
        "classic_recursive_calls_total": total_recursive_calls,
        "classic_time_s": round(total_time, 6),
        "per_fault": rows,
    }
    _write_csv(csv_path, rows)
    _write_json(out_path, report)
    if manifest_path:
        _write_json(
            manifest_path,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_id": args.run_id,
                "command": report["command"],
                "git_commit": report["git_commit"],
                "git_status_short": report["git_status_short"],
                "outputs": [str(out_path), str(csv_path), str(manifest_path)],
            },
        )
    print(
        f"classic coverage {succeeded}/{attempted} = {coverage:.2%}; "
        f"wrote {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
