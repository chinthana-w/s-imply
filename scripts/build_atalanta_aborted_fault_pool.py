"""Build a deduplicated AI-PODEM pool from Atalanta CSV rows marked aborted."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PREFERRED_BENCH_DIRS = (
    Path("data/bench/ITC99_all_numeric"),
    Path("data/bench/ISCAS85"),
    Path("data/bench/iscas89"),
)


def _circuit_name(path: Path) -> str:
    stem = path.stem
    if stem == "b19_123":
        return "b19"
    return re.sub(
        r"_(?:10000|100000|500000)(?:_combined|_\d+|_\d+_old)?$", "", stem
    )


def _bench_map() -> dict[str, Path]:
    benches: dict[str, Path] = {}
    for directory in PREFERRED_BENCH_DIRS:
        for path in sorted(directory.glob("*.bench")):
            benches.setdefault(path.stem, path)
    b15 = Path("data/bench/ITC99_all_numeric/b15_1.bench")
    if b15.exists():
        benches["b15"] = b15
    return benches


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/atalanta_hdf"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benches = _bench_map()
    observations: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    excluded = Counter()
    source_files = []

    for path in sorted(args.input_dir.glob("*.csv")):
        circuit = _circuit_name(path)
        source_files.append(str(path))
        with path.open(newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                if row.get("status") != "aborted":
                    continue
                if circuit not in benches:
                    excluded[circuit] += 1
                    continue
                gate_text, stuck_text = row["fault"].split()
                key = (circuit, int(gate_text), int(stuck_text.removeprefix("/")))
                observations[key].append(
                    {
                        "file": str(path),
                        "row": row_number,
                        "backtracks": int(row["backtracks"]),
                        "time_s": (
                            float(row["time_sec"]) if row.get("time_sec") else None
                        ),
                        "return_code": int(row["returncode"]),
                    }
                )

    faults = []
    for index, ((circuit, gate_id, stuck_at), source_rows) in enumerate(
        sorted(observations.items())
    ):
        backtracks = [row["backtracks"] for row in source_rows]
        times = [row["time_s"] for row in source_rows if row["time_s"] is not None]
        faults.append(
            {
                "fault_index": index,
                "bench": str(benches[circuit]),
                "circuit": circuit,
                "gate_id": gate_id,
                "fault_val": 3 if stuck_at == 0 else 4,
                "stuck_at": stuck_at,
                "atalanta_observation_count": len(source_rows),
                "atalanta_backtracks_min": min(backtracks),
                "atalanta_backtracks_max": max(backtracks),
                "atalanta_backtracks_representative": max(backtracks),
                "atalanta_time_s_min": min(times) if times else None,
                "atalanta_time_s_max": max(times) if times else None,
                "atalanta_sources": source_rows,
            }
        )

    circuit_counts = Counter(item["circuit"] for item in faults)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/build_atalanta_aborted_fault_pool.py",
        "definition": (
            "Unique (circuit, gate, stuck-at) faults with status=aborted in at least "
            "one data/atalanta_hdf CSV and a compatible local BENCH circuit."
        ),
        "deduplication": "Repeated observations are retained under atalanta_sources.",
        "atalanta_backtracks_representative": (
            "Maximum recorded abort backtracks across repeated source observations."
        ),
        "source_files": source_files,
        "source_aborted_observations_included": sum(
            len(rows) for rows in observations.values()
        ),
        "duplicate_observations_removed": sum(
            max(0, len(rows) - 1) for rows in observations.values()
        ),
        "excluded_aborted_observations_no_local_bench": dict(sorted(excluded.items())),
        "total_faults": len(faults),
        "circuit_counts": dict(sorted(circuit_counts.items())),
        "faults": faults,
    }
    _atomic_json(args.output, payload)
    print(
        f"wrote {len(faults)} unique aborted faults across {len(circuit_counts)} "
        f"circuits to {args.output}"
    )
    print(
        f"included observations={payload['source_aborted_observations_included']} "
        f"deduplicated={payload['duplicate_observations_removed']} "
        f"excluded_no_bench={sum(excluded.values())}"
    )


if __name__ == "__main__":
    main()
