"""Create a circuit-balanced random fault pool across numeric ITC99 BENCH files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from src.atpg.podem import get_all_faults
from src.util.io import parse_bench_file


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def _balanced_quotas(capacities: dict[str, int], pool_size: int) -> dict[str, int]:
    if pool_size > sum(capacities.values()):
        raise ValueError("pool size exceeds the total available fault population")
    quotas = {bench: 0 for bench in capacities}
    remaining = pool_size
    active = set(capacities)
    while remaining and active:
        share = max(1, remaining // len(active))
        progressed = False
        for bench in sorted(active):
            available = capacities[bench] - quotas[bench]
            take = min(share, available, remaining)
            quotas[bench] += take
            remaining -= take
            progressed |= take > 0
        active = {bench for bench in active if quotas[bench] < capacities[bench]}
        if not progressed:
            break
    if remaining:
        raise RuntimeError(f"failed to allocate {remaining} requested faults")
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pool-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    bench_paths = sorted(Path(args.bench_dir).glob("*.bench"))
    if not bench_paths:
        raise RuntimeError(f"no .bench files found in {args.bench_dir}")

    populations = {}
    faults_by_bench = {}
    for path in bench_paths:
        circuit, total_gates = parse_bench_file(str(path))
        faults = get_all_faults(circuit, total_gates)
        populations[str(path)] = len(faults)
        faults_by_bench[str(path)] = faults

    quotas = _balanced_quotas(populations, args.pool_size)
    rng = random.Random(args.seed)
    selected = []
    for bench in sorted(faults_by_bench):
        faults = faults_by_bench[bench]
        indices = rng.sample(range(len(faults)), quotas[bench])
        indices.sort()
        for source_index in indices:
            fault = faults[source_index]
            selected.append(
                {
                    "bench": bench,
                    "source_fault_index": source_index,
                    "gate_id": int(fault.gate_id),
                    "fault_val": int(fault.value),
                }
            )
    for pool_index, item in enumerate(selected):
        item["pool_index"] = pool_index

    encoded = json.dumps(selected, sort_keys=True).encode()
    payload = {
        "bench": None,
        "benches": [str(path) for path in bench_paths],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Circuit-balanced random stuck-at fault pool spanning every locally materialized "
            "standard ITC99 full-scan circuit."
        ),
        "selection_method": "equal allocation with capacity-aware redistribution",
        "seed": args.seed,
        "requested_pool_size": args.pool_size,
        "selected_faults": len(selected),
        "source_fault_counts": populations,
        "selected_counts_by_bench": quotas,
        "all_circuits_represented": all(quotas[path] > 0 for path in populations),
        "selection_sha256": hashlib.sha256(encoded).hexdigest(),
        "faults": selected,
    }
    _write_json(Path(args.output), payload)
    print(f"Wrote {len(selected)} faults across {len(bench_paths)} circuits")
    for bench, count in quotas.items():
        print(f"{Path(bench).name}: {count}/{populations[bench]}")
    print(f"selection_sha256={payload['selection_sha256']}")


if __name__ == "__main__":
    main()
