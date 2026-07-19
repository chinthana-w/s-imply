"""Append cumulative Atalanta attempt-time metrics without touching the runner."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def summarize(attempts_path: Path) -> tuple[int, int, float]:
    attempted = 0
    timeouts = 0
    total_time = 0.0
    with attempts_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            attempted += 1
            timeouts += row["status"] == "timeout"
            total_time += float(row["elapsed_s"])
    average = total_time / attempted if attempted else 0.0
    return attempted, timeouts, average


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--watch-pid", required=True, type=int)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()

    while process_exists(args.watch_pid):
        attempted, timeouts, average = summarize(args.attempts)
        line = (
            f"metrics attempted={attempted} timeouts={timeouts} "
            f"average_attempt_time_s={average:.6f}"
        )
        with args.log.open("a", buffering=1) as handle:
            handle.write(line + "\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
