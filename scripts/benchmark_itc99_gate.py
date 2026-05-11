"""Benchmark AI-PODEM on the deterministic ITC99 gate subset.

This is the cheap held-out gate before running the full ITC99 benchmark.  It
never builds training data and never feeds results back into training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.atpg.ai_podem import (
    AiPodemConfig,
    HierarchicalReconvSolver,
    ModelPairPredictor,
    ai_podem,
)
from src.atpg.logic_sim_three import reset_gates
from src.atpg.podem import get_all_faults, initialize
from src.util.io import parse_bench_file
from src.util.struct import Fault, LogicValue


def _load_gate_faults(path: str) -> tuple[str, list[Fault], dict]:
    with open(path) as f:
        payload = json.load(f)
    faults = [
        Fault(int(item["gate_id"]), LogicValue(int(item["fault_val"])))
        for item in payload["faults"]
    ]
    return payload["bench"], faults, payload


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


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _write_csv(path: str, per_fault: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["fault_index", "gate_id", "fault_val", "ok", "time_s"]
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_fault)
    os.replace(tmp_path, path)


def _compact_gate_meta(gate_meta: dict) -> dict:
    compact = dict(gate_meta)
    faults = compact.pop("faults", None)
    if isinstance(faults, list):
        encoded = json.dumps(faults, sort_keys=True).encode()
        compact["faults_count"] = len(faults)
        compact["faults_sha256"] = hashlib.sha256(encoded).hexdigest()
        compact["first_faults"] = faults[:5]
    return compact


def _build_manifest(args: argparse.Namespace, outputs: list[str]) -> dict:
    return {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "scripts.benchmark_itc99_gate", *sys.argv[1:]],
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_status_short": _git_value(["status", "--short"]),
        "inputs": {
            "model": args.model,
            "fault_list": args.fault_list,
            "full": args.full,
            "limit_faults": args.limit_faults,
            "candidate_count": args.candidate_count,
            "candidate_seed_base": args.candidate_seed_base,
            "max_backtracks": args.max_backtracks,
            "coverage_target": args.coverage_target,
        },
        "outputs": outputs,
        "baseline": {
            "label": args.baseline_label,
            "coverage": args.baseline_coverage,
            "source": args.baseline_source,
        },
    }


def _write_notion_summary(path: str, report: dict, manifest_path: str | None) -> None:
    baseline = report["baseline_comparison"]
    comparison_text = (
        f"{baseline['delta']:+.4%} absolute coverage"
        if baseline["decision_comparable"]
        else f"not decision-comparable: {baseline['comparison_note']}"
    )
    lines = [
        f"## Experiment Log - {report['created_at'][:10]} ITC99 Gate Benchmark",
        "",
        f"- Command: `{shlex.join(report['command'])}`",
        f"- Inputs: model `{report['model']}`, fault list `{report['fault_list']}`",
        f"- Artifacts: `{report['artifact_paths']['json']}`",
    ]
    if report["artifact_paths"].get("csv"):
        lines.append(f"- Per-fault CSV: `{report['artifact_paths']['csv']}`")
    if manifest_path:
        lines.append(f"- Manifest: `{manifest_path}`")
    lines.extend(
        [
            f"- Metrics: {report['succeeded']}/{report['total']} faults detected "
            f"({report['coverage']:.4%} no-fallback coverage)",
            f"- Baseline: {baseline['label']} at {baseline['coverage']:.4%} "
            f"from `{baseline['source']}`",
            f"- Baseline comparison: {comparison_text}",
            f"- Coverage target: {report['coverage_target']:.4%}; "
            f"pass={report['passed_coverage_target']}",
            "- Result: measurement artifact created; no promotion decision without "
            "reviewing the full gate target.",
            "- Next step: validate the candidate checkpoint on the configured 10% ITC99 "
            "gate once this slice passes code review.",
            "",
        ]
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write("\n".join(lines))
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the ITC99 10% gate subset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fault-list", default="data/bench/ITC99/b17_gate_10pct_faults.json")
    parser.add_argument("--out", default="docs/itc99_gate_report.json")
    parser.add_argument("--max-backtracks", type=int, default=5000)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--candidate-seed-base", type=int, default=20260504)
    parser.add_argument("--full", action="store_true", help="Ignore fault-list and run all faults")
    parser.add_argument("--limit-faults", type=int, default=0)
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--notion-summary-out", default="")
    parser.add_argument("--baseline-coverage", type=float, default=0.1817)
    parser.add_argument("--baseline-label", default="unlinked_candidate 1% ITC99 gate")
    parser.add_argument("--baseline-source", default="docs/checkpoint_compatibility_summary.md")
    parser.add_argument("--coverage-target", type=float, default=1.0)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    if args.full:
        with open(args.fault_list) as f:
            payload = json.load(f)
        bench_path = payload["bench"]
        circuit, total_gates = parse_bench_file(bench_path)
        faults = get_all_faults(circuit, total_gates)
        gate_meta = {"bench": bench_path, "selected_faults": len(faults), "full": True}
    else:
        bench_path, faults, gate_meta = _load_gate_faults(args.fault_list)
        circuit, total_gates = parse_bench_file(bench_path)
    if args.limit_faults:
        if args.limit_faults < 1:
            raise ValueError("--limit-faults must be positive when provided")
        original_count = len(faults)
        faults = faults[: args.limit_faults]
        gate_meta = {
            **gate_meta,
            "limited_run": True,
            "limit_faults": args.limit_faults,
            "original_faults": original_count,
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AiPodemConfig(
        model_path=args.model,
        device=device,
        enable_ai_activation=True,
        enable_ai_propagation=False,
        verbose=False,
        no_fallback=True,
        candidate_count=args.candidate_count,
        candidate_seed_base=args.candidate_seed_base,
    )
    predictor = ModelPairPredictor(circuit, bench_path, config)
    solver = HierarchicalReconvSolver(circuit, predictor)

    succeeded = 0
    failed = 0
    total_time = 0.0
    per_fault = []

    initialize(circuit, total_gates)
    for idx, fault in enumerate(faults):
        reset_gates(circuit, total_gates)
        t0 = time.time()
        ok = ai_podem(
            circuit,
            fault,
            total_gates,
            predictor=predictor,
            solver=solver,
            enable_ai_activation=True,
            enable_ai_propagation=False,
            no_fallback=True,
            max_backtracks=args.max_backtracks,
            seed=args.candidate_seed_base + idx,
        )
        elapsed = time.time() - t0
        total_time += elapsed
        detected = bool(ok)
        succeeded += int(detected)
        failed += int(not detected)
        per_fault.append(
            {
                "fault_index": idx,
                "gate_id": int(fault.gate_id),
                "fault_val": int(fault.value),
                "ok": detected,
                "time_s": round(elapsed, 4),
            }
        )
        if (idx + 1) % 100 == 0:
            print(
                f"ITC99 gate progress {idx + 1}/{len(faults)} "
                f"coverage={succeeded / (idx + 1):.2%}",
                flush=True,
            )

    coverage = succeeded / max(1, len(faults))
    outputs = [args.out]
    if args.csv_out:
        outputs.append(args.csv_out)
    if args.notion_summary_out:
        outputs.append(args.notion_summary_out)
    if args.manifest_out:
        outputs.append(args.manifest_out)
    artifact_paths = {"json": args.out}
    if args.csv_out:
        artifact_paths["csv"] = args.csv_out
    if args.notion_summary_out:
        artifact_paths["notion_summary"] = args.notion_summary_out
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "scripts.benchmark_itc99_gate", *sys.argv[1:]],
        "run_id": args.run_id,
        "model": args.model,
        "bench": bench_path,
        "fault_list": args.fault_list,
        "gate_meta": _compact_gate_meta(gate_meta),
        "candidate_count": args.candidate_count,
        "candidate_seed_base": args.candidate_seed_base,
        "max_backtracks": args.max_backtracks,
        "total": len(faults),
        "succeeded": succeeded,
        "failed": failed,
        "coverage": coverage,
        "coverage_target": args.coverage_target,
        "passed_coverage_target": coverage >= args.coverage_target,
        "total_time_s": round(total_time, 2),
        "baseline_comparison": {
            "label": args.baseline_label,
            "source": args.baseline_source,
            "coverage": args.baseline_coverage,
            "observed": coverage,
            "delta": coverage - args.baseline_coverage,
            "decision_comparable": args.limit_faults == 0,
            "comparison_note": (
                "bounded --limit-faults smoke validates the benchmark path but is not "
                "a statistically valid baseline comparison"
                if args.limit_faults
                else "same configured benchmark scope"
            ),
        },
        "artifact_paths": artifact_paths,
        "per_fault": per_fault,
    }

    _write_json(args.out, report)
    if args.csv_out:
        _write_csv(args.csv_out, per_fault)
    if args.manifest_out:
        _write_json(args.manifest_out, _build_manifest(args, outputs))
    if args.notion_summary_out:
        _write_notion_summary(args.notion_summary_out, report, args.manifest_out or None)
    print(
        f"ITC99 gate coverage: {succeeded}/{len(faults)} "
        f"= {succeeded / max(1, len(faults)):.2%}; wrote {args.out}"
    )


if __name__ == "__main__":
    main()
