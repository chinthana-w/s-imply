"""Create a deterministic ITC99 gate fault subset.

The selected JSON is intentionally small metadata: it stores only fault gate IDs
and fault values, not circuits or patterns.  ITC99 remains held out; this file is
used only to gate whether a checkpoint is worth a full ITC99 benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.atpg.podem import get_all_faults
from src.util.io import parse_bench_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Select deterministic ITC99 gate faults")
    parser.add_argument("--bench", default="data/bench/ITC99/b17.bench")
    parser.add_argument("--output", default="data/bench/ITC99/b17_gate_10pct_faults.json")
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260504)
    args = parser.parse_args()

    circuit, total_gates = parse_bench_file(args.bench)
    faults = get_all_faults(circuit, total_gates)
    n_select = max(1, int(len(faults) * args.fraction))

    rng = random.Random(args.seed)
    indices = list(range(len(faults)))
    rng.shuffle(indices)
    selected_indices = sorted(indices[:n_select])
    selected = [
        {
            "index": int(i),
            "gate_id": int(faults[i].gate_id),
            "fault_val": int(faults[i].value),
        }
        for i in selected_indices
    ]

    payload = {
        "bench": args.bench,
        "total_faults": len(faults),
        "fraction": args.fraction,
        "seed": args.seed,
        "selected_faults": len(selected),
        "faults": selected,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tmp_path = args.output + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, args.output)

    print(
        f"Wrote {len(selected)}/{len(faults)} faults "
        f"({args.fraction:.1%}) to {args.output}"
    )


if __name__ == "__main__":
    main()
