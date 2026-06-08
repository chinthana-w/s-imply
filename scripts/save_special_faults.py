"""Script to extract faults where classic PODEM fails or is slower than AI-guided PODEM."""

from __future__ import annotations

import csv
import json
import os


def main() -> None:
    csv_path = (
        "docs/session_reports/codex_20260601_b17_reconv_full_process_guard_gpu/"
        "b17_full_per_fault.csv"
    )
    out_path = "data/bench/ITC99/b17_classic_fail_or_slower_faults.json"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    special_faults = []
    print(f"Reading from {csv_path}...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse row fields
            ok = row["ok"].strip() == "True"
            classic_ok_str = row["classic_ok"].strip()
            classic_ok = classic_ok_str == "True"

            # Check if AI succeeded
            if not ok:
                continue

            time_s = float(row["time_s"])
            classic_time_s_str = row["classic_time_s"].strip()

            # Classic failed if classic_ok is False
            classic_failed = classic_ok_str == "False" or not classic_ok

            # Classic was slower if classic_time_s > time_s
            classic_slower = False
            if not classic_failed and classic_time_s_str:
                classic_slower = float(classic_time_s_str) > time_s

            if classic_failed or classic_slower:
                special_faults.append(
                    {
                        "gate_id": int(row["gate_id"]),
                        "fault_val": int(row["fault_val"]),
                        "classic_ok": classic_ok,
                        "time_s": time_s,
                        "classic_time_s": (
                            float(classic_time_s_str) if classic_time_s_str else None
                        ),
                    }
                )

    # Wrap in payload
    payload = {
        "bench": "data/bench/ITC99/b17.bench",
        "total_faults": len(special_faults),
        "faults": [
            {
                "index": idx,
                "gate_id": item["gate_id"],
                "fault_val": item["fault_val"],
            }
            for idx, item in enumerate(special_faults)
        ],
    }

    # Write payload
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Saved {len(special_faults)} special faults to {out_path}")


if __name__ == "__main__":
    main()
