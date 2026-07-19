"""Create a random reconvergent-only fault pool from available ITC99 benches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.atpg.podem import get_all_faults
from src.atpg.recursive_reconv_solver import HierarchicalReconvSolver, ReconvPairPredictor
from src.util.io import parse_bench_file
from src.util.struct import LogicValue


class _TopologyOnlyPredictor(ReconvPairPredictor):
    """Placeholder predictor; pool selection only calls topology collection."""

    def predict(
        self,
        pair_info: dict,
        constraints: dict[int, LogicValue],
        seed: int | None = None,
    ) -> list[dict[int, LogicValue]]:
        return []


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _bench_paths(bench_dir: Path) -> list[Path]:
    return sorted(path for path in bench_dir.glob("*.bench") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select random reconvergent faults from all ITC99 benches present locally"
    )
    parser.add_argument("--bench-dir", default="data/bench/ITC99")
    parser.add_argument(
        "--model",
        default="",
        help="Accepted for provenance compatibility; not loaded during topology filtering.",
    )
    parser.add_argument("--output", default="data/bench/ITC99/reconv_pool_10000.json")
    parser.add_argument("--pool-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument(
        "--require-all-circuits",
        action="store_true",
        help="Guarantee at least one selected reconvergent fault from every input circuit.",
    )
    args = parser.parse_args()

    bench_paths = _bench_paths(Path(args.bench_dir))
    if not bench_paths:
        raise RuntimeError(f"No .bench files found in {args.bench_dir}")

    candidates: list[dict] = []
    source_counts: list[dict] = []
    for bench_path in bench_paths:
        bench_name = str(bench_path)
        circuit, total_gates = parse_bench_file(bench_name)
        faults = get_all_faults(circuit, total_gates)
        solver = HierarchicalReconvSolver(circuit, _TopologyOnlyPredictor())

        reconv_count = 0
        gate_has_reconv: dict[int, bool] = {}
        for fault_index, fault in enumerate(faults):
            gate_id = int(fault.gate_id)
            has_pairs = gate_has_reconv.get(gate_id)
            if has_pairs is None:
                pairs = solver._collect_and_sort_pairs(gate_id)
                has_pairs = bool(pairs)
                gate_has_reconv[gate_id] = has_pairs
                solver.pair_cache.clear()
                del pairs
                if fault_index % 500 == 0:
                    gc.collect()
            if not has_pairs:
                if args.progress_every > 0 and (fault_index + 1) % args.progress_every == 0:
                    print(
                        f"{bench_name}: scanned {fault_index + 1}/{len(faults)} "
                        f"reconv={reconv_count}",
                        flush=True,
                    )
                continue
            reconv_count += 1
            candidates.append(
                {
                    "bench": bench_name,
                    "index": int(fault_index),
                    "gate_id": int(fault.gate_id),
                    "fault_val": int(fault.value),
                }
            )
            if args.progress_every > 0 and (fault_index + 1) % args.progress_every == 0:
                print(
                    f"{bench_name}: scanned {fault_index + 1}/{len(faults)} reconv={reconv_count}",
                    flush=True,
                )

        source_counts.append(
            {
                "bench": bench_name,
                "total_faults": len(faults),
                "reconvergent_faults": reconv_count,
                "non_reconvergent_faults": len(faults) - reconv_count,
            }
        )
        print(f"{bench_name}: {reconv_count}/{len(faults)} reconvergent faults", flush=True)

    if len(candidates) < args.pool_size:
        raise RuntimeError(
            f"Requested {args.pool_size} faults, but only found {len(candidates)} "
            "reconvergent ITC99 faults"
        )

    rng = random.Random(args.seed)
    selected: list[dict] = []
    if args.require_all_circuits:
        by_bench: dict[str, list[dict]] = {}
        for item in candidates:
            by_bench.setdefault(item["bench"], []).append(item)
        missing = [str(path) for path in bench_paths if str(path) not in by_bench]
        if missing:
            raise RuntimeError(
                "--require-all-circuits cannot be satisfied because these circuits have no "
                f"reconvergent candidates: {missing}"
            )
        selected.extend(rng.choice(items) for items in by_bench.values())
        selected_keys = {(item["bench"], item["index"], item["fault_val"]) for item in selected}
        remaining = [
            item
            for item in candidates
            if (item["bench"], item["index"], item["fault_val"]) not in selected_keys
        ]
        selected.extend(rng.sample(remaining, args.pool_size - len(selected)))
    else:
        selected = rng.sample(candidates, args.pool_size)
    selected.sort(key=lambda item: (item["bench"], item["index"], item["fault_val"]))
    for pool_index, item in enumerate(selected):
        item["pool_index"] = pool_index

    selected_benches = sorted({item["bench"] for item in selected})
    selected_counts = {
        bench: sum(item["bench"] == bench for item in selected) for bench in selected_benches
    }

    encoded = json.dumps(selected, sort_keys=True).encode()
    payload = {
        "bench": selected_benches[0] if len(selected_benches) == 1 else None,
        "benches": selected_benches,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Random reconvergent-only fault pool from all ITC99 .bench files present "
            "in this checkout."
        ),
        "seed": args.seed,
        "model": args.model,
        "requested_pool_size": args.pool_size,
        "selected_faults": len(selected),
        "total_candidate_reconvergent_faults": len(candidates),
        "itc99_bench_files": [str(path) for path in bench_paths],
        "source_counts": source_counts,
        "selected_counts_by_bench": selected_counts,
        "require_all_circuits": args.require_all_circuits,
        "selection_sha256": hashlib.sha256(encoded).hexdigest(),
        "faults": selected,
    }
    _write_json(Path(args.output), payload)
    print(f"Wrote {len(selected)} faults to {args.output}")
    print(f"selection_sha256={payload['selection_sha256']}")


if __name__ == "__main__":
    main()
