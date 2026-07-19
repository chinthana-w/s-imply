"""Generate a comprehensive Markdown report for the Atalanta-aborted AI run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    pool = json.loads(args.pool.read_text())
    summary = json.loads(args.summary.read_text())
    with args.results.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    completed = len(rows)
    total = int(summary["total_faults"])
    solved_rows = [row for row in rows if _bool(row["ai_ok"])]
    failed_rows = [row for row in rows if not _bool(row["ai_ok"])]
    source_bt = sum(int(row["atalanta_backtracks"]) for row in solved_rows)
    ai_bt = sum(int(row["ai_backtracks"]) for row in solved_rows)
    reduction = source_bt - ai_bt
    reduction_pct = reduction / source_bt if source_bt else None
    lower_bt = sum(
        int(row["ai_backtracks"]) < int(row["atalanta_backtracks"])
        for row in solved_rows
    )
    zero_bt = sum(int(row["ai_backtracks"]) == 0 for row in solved_rows)
    precheck = sum(_bool(row["ai_precheck_success"]) for row in rows)
    with_pairs = sum(_bool(row["ai_has_reconv_pairs"]) for row in rows)
    errors = Counter(row["ai_error"] or "(none)" for row in failed_rows)

    circuits = defaultdict(
        lambda: {
            "attempted": 0,
            "solved": 0,
            "source_bt_success": 0,
            "ai_bt_success": 0,
        }
    )
    for row in rows:
        item = circuits[row["circuit"]]
        item["attempted"] += 1
        if _bool(row["ai_ok"]):
            item["solved"] += 1
            item["source_bt_success"] += int(row["atalanta_backtracks"])
            item["ai_bt_success"] += int(row["ai_backtracks"])

    status = "complete" if completed == total else "in progress"
    lines = [
        "# AI-PODEM on Atalanta-Aborted Faults",
        "",
        f"**Run status:** {status} — {completed:,}/{total:,} faults processed "
        f"({completed / max(1, total):.2%}).",
        "",
        "## Executive summary",
        "",
        f"The pool contains **{total:,} unique faults** that Atalanta marked `aborted` "
        f"in at least one source run. AI-guided PODEM detected **{len(solved_rows):,} of "
        f"{completed:,} attempted faults ({len(solved_rows) / max(1, completed):.2%})**.",
        "",
        f"On the **{len(solved_rows):,} AI successes**, Atalanta had accumulated "
        f"**{source_bt:,} representative abort backtracks**, while AI-guided PODEM used "
        f"**{ai_bt:,} search backtracks**: a reduction of **{reduction:,} "
        f"({_fmt(100 * reduction_pct if reduction_pct is not None else None)}%)**. "
        f"AI used fewer backtracks on **{lower_bt:,}/{len(solved_rows):,}** successes, "
        f"including **{zero_bt:,} zero-backtrack** detections.",
        "",
        "```mermaid",
        "flowchart TD",
        "    A[Atalanta CSV files] --> B{status = aborted?}",
        "    B -->|No| X[Exclude]",
        "    B -->|Yes| C[Map circuit to local BENCH]",
        "    C --> D[Deduplicate circuit + gate + stuck-at]",
        "    D --> E[AI structural solve and direct simulation]",
        "    E -->|Detected| F[Zero-backtrack AI success]",
        "    E -->|Not detected| G[AI-hinted PODEM search]",
        "    G --> H[Record outcome, time, and search backtracks]",
        "    F --> H",
        "```",
        "",
        "## Pool construction and provenance",
        "",
        f"- Included aborted source observations: "
        f"**{pool['source_aborted_observations_included']:,}**",
        f"- Duplicate observations collapsed: **{pool['duplicate_observations_removed']:,}**",
        f"- Unique compatible faults: **{pool['total_faults']:,}**",
        f"- Compatible circuits represented: **{len(pool['circuit_counts']):,}**",
        f"- Excluded aborted observations without a local BENCH mapping: "
        f"**{sum(pool['excluded_aborted_observations_no_local_bench'].values()):,}** "
        f"(`{pool['excluded_aborted_observations_no_local_bench']}`)",
        "",
        "Repeated Atalanta measurements were not counted as separate faults. The comparison "
        "uses the maximum recorded abort backtrack count as the representative source value; "
        "all original rows remain embedded in the JSON pool for auditability.",
        "",
        "## Aggregate results",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Faults completed | {completed:,}/{total:,} |",
        f"| AI detected | {len(solved_rows):,} ({len(solved_rows)/max(1,completed):.2%}) |",
        f"| AI failed/timed out | {len(failed_rows):,} |",
        f"| Faults with reconvergent pairs | {with_pairs:,} ({with_pairs/max(1,completed):.2%}) |",
        f"| Direct AI precheck successes | {precheck:,} ({precheck/max(1,completed):.2%}) |",
        f"| AI successes with fewer backtracks | {lower_bt:,} ({lower_bt/max(1,len(solved_rows)):.2%}) |",
        f"| Zero-backtrack AI successes | {zero_bt:,} ({zero_bt/max(1,len(solved_rows)):.2%}) |",
        f"| Atalanta representative backtracks on AI successes | {source_bt:,} |",
        f"| AI search backtracks on AI successes | {ai_bt:,} |",
        f"| Backtrack reduction on AI successes | {reduction:,} ({_fmt(100*reduction_pct if reduction_pct is not None else None)}%) |",
        "",
        "```mermaid",
        "pie showData",
        f'    title AI outcomes on {completed} processed Atalanta-aborted faults',
        f'    "Detected by AI" : {len(solved_rows)}',
        f'    "Not detected within budget" : {len(failed_rows)}',
        "```",
        "",
        "## Per-circuit comparison",
        "",
        "| Circuit | Attempted | AI solved | AI coverage | Atalanta BT on AI successes | AI BT on successes | BT reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for circuit, item in sorted(circuits.items()):
        src = item["source_bt_success"]
        ai = item["ai_bt_success"]
        pct = (src - ai) / src if src else None
        lines.append(
            f"| {circuit} | {item['attempted']:,} | {item['solved']:,} | "
            f"{item['solved']/max(1,item['attempted']):.2%} | {src:,} | {ai:,} | "
            f"{_fmt(100*pct if pct is not None else None)}% |"
        )

    lines.extend(
        [
            "",
            "## Backtrack interpretation",
            "",
            "The Atalanta number is an abort-bound observation: it establishes that Atalanta "
            "reached the configured ceiling without detecting the fault in that run. The AI "
            "number is the internal PODEM search backtrack counter for the executed AI-guided "
            "path. Therefore, the strongest apples-to-apples claim is conditional: **for faults "
            "that AI detected, how much search backtracking did AI require relative to the "
            "recorded Atalanta abort effort?** It is not a claim that the two implementations "
            "have identical per-backtrack computational cost.",
            "",
            "Source files used multiple abort ceilings (`101`, `10001`, `100001`, and `500001`). "
            "The per-fault CSV preserves minimum and maximum source observations so analyses can "
            "be restricted to the dominant `100001` cohort if desired.",
            "",
            "## Failure and error profile",
            "",
            "| AI error/status text | Count |",
            "|---|---:|",
        ]
    )
    for label, count in errors.most_common(20):
        lines.append(f"| {label.replace('|', '/')} | {count:,} |")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Fault pool: `{args.pool}`",
            f"- Per-fault results: `{args.results}`",
            f"- Machine-readable summary: `{args.summary}`",
            f"- Model: `{summary['model']}`",
            f"- AI timeout: **{summary['timeout_s']} seconds per fault**",
            f"- AI maximum backtracks: **{summary['max_backtracks']:,}**",
            f"- Device setting: `{summary['device']}`",
            "",
            "The benchmark checkpoints its CSV and summary throughout the run. Re-running the "
            "same command against the same output directory resumes from the first unrecorded "
            "fault.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {status} report to {args.output}")


if __name__ == "__main__":
    main()
