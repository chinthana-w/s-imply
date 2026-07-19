"""Collect ITC99 stuck-at faults that exceed an Atalanta wall-clock limit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import subprocess
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.atpg.podem import get_all_faults
from src.util.io import parse_bench_file

FIELDNAMES = (
    "fault_index",
    "bench",
    "source_fault_index",
    "gate_id",
    "fault_val",
    "stuck_at",
    "status",
    "elapsed_s",
    "return_code",
)


@dataclass(frozen=True)
class FaultJob:
    fault_index: int
    bench: str
    source_fault_index: int
    gate_id: int
    fault_val: int
    stuck_at: int


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def enumerate_faults(
    bench_dir: Path, bench_names: set[str] | None = None
) -> tuple[list[FaultJob], dict[str, int]]:
    jobs: list[FaultJob] = []
    counts: dict[str, int] = {}
    benches = sorted(bench_dir.glob("*.bench"))
    if bench_names:
        benches = [bench for bench in benches if bench.stem in bench_names]
        found = {bench.stem for bench in benches}
        missing = sorted(bench_names - found)
        if missing:
            raise FileNotFoundError(f"requested benchmark circuits not found: {missing}")
    for bench in benches:
        circuit, total_gates = parse_bench_file(str(bench))
        faults = get_all_faults(circuit, total_gates)
        counts[str(bench)] = len(faults)
        for source_index, fault in enumerate(faults):
            fault_val = int(fault.value)
            if fault_val not in (3, 4):
                raise ValueError(f"unexpected five-valued fault encoding: {fault_val}")
            jobs.append(
                FaultJob(
                    fault_index=len(jobs),
                    bench=str(bench),
                    source_fault_index=source_index,
                    gate_id=int(fault.gate_id),
                    fault_val=fault_val,
                    stuck_at=0 if fault_val == 3 else 1,
                )
            )
    return jobs, counts


def terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=0.5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_fault(job: FaultJob, atalanta: Path, timeout_s: float) -> dict:
    start = time.monotonic()
    with tempfile.NamedTemporaryFile(mode="w", prefix="atalanta-fault-", suffix=".txt") as fault:
        fault.write(f"{job.gate_id} /{job.stuck_at}\n")
        fault.flush()
        command = [
            str(atalanta),
            "-r",
            "0",
            "-N",
            "-b",
            "100000000",
            "-B",
            "0",
            "-s",
            "1",
            "-f",
            fault.name,
            "-t",
            "/dev/null",
            job.bench,
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_s)
            status = "completed" if return_code == 0 else "error"
        except subprocess.TimeoutExpired:
            terminate_process(process)
            return_code = 124
            status = "timeout"
    return {
        **asdict(job),
        "status": status,
        "elapsed_s": round(time.monotonic() - start, 6),
        "return_code": return_code,
    }


def load_completed(path: Path) -> tuple[set[int], list[dict]]:
    if not path.exists():
        return set(), []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["fault_index"]) for row in rows}, rows


def write_hard_faults(
    path: Path,
    rows: list[dict],
    counts: dict[str, int],
    timeout_s: float,
    limit: int,
    exhausted: bool,
) -> None:
    hard = sorted(
        (row for row in rows if row["status"] == "timeout"),
        key=lambda row: int(row["fault_index"]),
    )[:limit]
    faults = [
        {
            "fault_index": int(row["fault_index"]),
            "bench": row["bench"],
            "source_fault_index": int(row["source_fault_index"]),
            "gate_id": int(row["gate_id"]),
            "fault_val": int(row["fault_val"]),
            "stuck_at": int(row["stuck_at"]),
            "elapsed_s": float(row["elapsed_s"]),
        }
        for row in hard
    ]
    atomic_json(
        path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": "scripts/collect_atalanta_hard_faults.py",
            "atpg": "Atalanta 2.0",
            "timeout_s_per_fault": timeout_s,
            "definition": "wall-clock timeout with Atalanta FAN backtrack limit 100000000",
            "source_fault_counts": counts,
            "source_faults_total": sum(counts.values()),
            "faults_attempted": len(rows),
            "requested_timeout_faults": limit,
            "collected_timeout_faults": len(faults),
            "source_exhausted": exhausted,
            "faults": faults,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", default="data/bench/ITC99_all_numeric")
    parser.add_argument(
        "--bench",
        action="append",
        dest="benches",
        help="limit the scan to this circuit stem; may be repeated",
    )
    parser.add_argument("--atalanta", default="tools/atalanta/atalanta")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--largest-first",
        action="store_true",
        help="scan circuits with the largest fault populations first",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        help="deterministically shuffle faults within each selected circuit",
    )
    args = parser.parse_args()

    atalanta = Path(args.atalanta).resolve()
    if not atalanta.is_file():
        raise FileNotFoundError(f"Atalanta binary not found: {atalanta}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "attempts.csv"
    hard_path = output_dir / "hard_faults.json"

    jobs, counts = enumerate_faults(
        Path(args.bench_dir), set(args.benches) if args.benches else None
    )
    completed, existing_rows = load_completed(results_path)
    timeout_count = sum(row["status"] == "timeout" for row in existing_rows)
    mode = "a" if results_path.exists() else "w"
    started = time.monotonic()

    with results_path.open(mode, newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()

        remaining_jobs = [job for job in jobs if job.fault_index not in completed]
        if args.largest_first:
            remaining_jobs.sort(
                key=lambda job: (-counts[job.bench], job.fault_index)
            )
        if args.shuffle_seed is not None:
            rng = random.Random(args.shuffle_seed)
            grouped: dict[str, list[FaultJob]] = {}
            circuit_order: list[str] = []
            for job in remaining_jobs:
                if job.bench not in grouped:
                    grouped[job.bench] = []
                    circuit_order.append(job.bench)
                grouped[job.bench].append(job)
            remaining_jobs = []
            for bench in circuit_order:
                rng.shuffle(grouped[bench])
                remaining_jobs.extend(grouped[bench])
        pending_jobs = iter(remaining_jobs)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            while len(futures) < args.workers:
                try:
                    job = next(pending_jobs)
                except StopIteration:
                    break
                futures[executor.submit(run_fault, job, atalanta, args.timeout)] = job

            new_rows = 0
            while futures and timeout_count < args.limit:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    job = futures.pop(future)
                    row = future.result()
                    writer.writerow(row)
                    existing_rows.append(row)
                    completed.add(job.fault_index)
                    new_rows += 1
                    timeout_count += row["status"] == "timeout"
                    if timeout_count < args.limit:
                        try:
                            next_job = next(pending_jobs)
                        except StopIteration:
                            continue
                        futures[
                            executor.submit(run_fault, next_job, atalanta, args.timeout)
                        ] = next_job
                if new_rows % args.checkpoint_every < len(done):
                    handle.flush()
                    write_hard_faults(
                        hard_path,
                        existing_rows,
                        counts,
                        args.timeout,
                        args.limit,
                        len(completed) == len(jobs),
                    )
                    elapsed = time.monotonic() - started
                    print(
                        f"attempted={len(completed)}/{len(jobs)} "
                        f"timeouts={timeout_count}/{args.limit} elapsed_s={elapsed:.1f}",
                        flush=True,
                    )

            for future in futures:
                future.cancel()

    exhausted = len(completed) == len(jobs)
    write_hard_faults(
        hard_path, existing_rows, counts, args.timeout, args.limit, exhausted
    )
    print(
        f"done attempted={len(completed)}/{len(jobs)} "
        f"timeouts={min(timeout_count, args.limit)}/{args.limit} exhausted={exhausted}",
        flush=True,
    )


if __name__ == "__main__":
    main()
