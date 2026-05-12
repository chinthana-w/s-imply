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
    AIBacktracer,
    AiPodemConfig,
    HierarchicalReconvSolver,
    ModelPairPredictor,
    StaticHintBacktracer,
    ai_podem,
)
from src.atpg.logic_sim_three import fault_is_at_po, logic_sim, reset_gates
from src.atpg.podem import (
    SUCCESS,
    get_all_faults,
    get_statistics,
    initialize,
    podem,
    reset_statistics,
    simple_backtrace,
)
from src.util.io import parse_bench_file
from src.util.struct import Fault, GateType, LogicValue


class ImprovedHintBacktracer:
    """Backtrace using AI hints, falling back to classic only when hints are missing.

    Unlike StaticHintBacktracer, this continues using hints for as many steps as
    possible. If a hint is missing for a particular gate, it falls back to
    simple_backtrace for the remainder of that specific backtrace path.
    """

    def __init__(self, hints: Dict[int, LogicValue], verbose: bool = False, no_fallback: bool = False):
        self.hints = {int(k): LogicValue(v) for k, v in hints.items()}
        self.verbose = verbose
        self.no_fallback = no_fallback

    def __call__(self, objective: Fault, circuit: list) -> Fault:
        curr_id = int(objective.gate_id)
        curr_target = LogicValue(objective.value)

        while circuit[curr_id].nfi != 0:
            gate = circuit[curr_id]
            x_fanins = [fin for fin in gate.fin if circuit[fin].val == LogicValue.XD]
            if not x_fanins:
                break

            required = self._required_fanin_value(gate.type, curr_target)
            if required is None:
                break

            # Try to find a fanin that matches our AI hints
            next_id = None
            for fin in x_fanins:
                if self.hints.get(fin) == required:
                    next_id = fin
                    break

            if next_id is None:
                if self.no_fallback:
                    raise RuntimeError(f"[AI-HINT] No hint for gate {curr_id} and fallback is disabled.")
                # No hint for this gate's fanins, fall back to simple_backtrace from here
                if self.verbose:
                    print(f"[AI-HINT] No hint for gate {curr_id}, falling back to classic.")
                return simple_backtrace(Fault(curr_id, curr_target), circuit)

            if self.verbose:
                print(
                    f"[AI-HINT] Gate {curr_id}={_logic_value_label(curr_target)} -> "
                    f"{next_id}={_logic_value_label(required)} (from hint)"
                )
            curr_id = next_id
            curr_target = required

        return Fault(curr_id, curr_target)

    @staticmethod
    def _required_fanin_value(gate_type: GateType, target: LogicValue) -> LogicValue | None:
        if gate_type == GateType.BUFF:
            return target
        if gate_type == GateType.NOT:
            return LogicValue.ONE if target == LogicValue.ZERO else LogicValue.ZERO
        if gate_type == GateType.AND:
            return LogicValue.ONE if target == LogicValue.ONE else LogicValue.ZERO
        if gate_type == GateType.NAND:
            return LogicValue.ONE if target == LogicValue.ZERO else LogicValue.ZERO
        if gate_type == GateType.OR:
            return LogicValue.ZERO if target == LogicValue.ZERO else LogicValue.ONE
        if gate_type == GateType.NOR:
            return LogicValue.ZERO if target == LogicValue.ONE else LogicValue.ONE
        return None


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
    fieldnames = [
        "fault_index",
        "gate_id",
        "fault_val",
        "ok",
        "attempts_used",
        "precheck_success",
        "precheck_attempts",
        "precheck_pi_assignments",
        "ai_backtracks",
        "classic_ok",
        "classic_backtracks",
        "ai_less_backtracks",
        "ai_error",
        "time_s",
        "classic_time_s",
    ]
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


def _select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return requested


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
            "device": args.device,
            "full": args.full,
            "limit_faults": args.limit_faults,
            "exclude_primary_input_faults": args.exclude_primary_input_faults,
            "candidate_count": args.candidate_count,
            "ai_attempts": args.ai_attempts,
            "activation_precheck": args.activation_precheck,
            "candidate_seed_base": args.candidate_seed_base,
            "enable_ai_propagation": args.enable_ai_propagation,
            "max_backtracks": args.max_backtracks,
            "compare_classic": args.compare_classic,
            "classic_timeout": args.classic_timeout,
            "coverage_target": args.coverage_target,
            "backtrack_target": args.backtrack_target,
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
            f"- Backtracks: AI `{report['ai_backtracks_total']}`, classic "
            f"`{report['classic_backtracks_total']}`; "
            f"AI less than classic={report['passed_backtrack_target']}",
            f"- Activation precheck: {report['activation_precheck_succeeded']} "
            f"zero-backtrack detections",
            f"- Baseline: {baseline['label']} at {baseline['coverage']:.4%} "
            f"from `{baseline['source']}`",
            f"- Baseline comparison: {comparison_text}",
            f"- Coverage target: {report['coverage_target']:.4%}; "
            f"pass={report['passed_coverage_target']}",
            f"- Backtrack target enabled: {report['backtrack_target']}; "
            f"pass={report['passed_backtrack_target']}",
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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fault-list", default="data/bench/ITC99/b17_gate_10pct_faults.json")
    parser.add_argument("--out", default="docs/itc99_gate_report.json")
    parser.add_argument("--max-backtracks", type=int, default=5000)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument(
        "--ai-attempts",
        type=int,
        default=1,
        help="Deterministic no-fallback AI activation attempts per fault",
    )
    parser.add_argument(
        "--enable-ai-propagation",
        action="store_true",
        help="Use the AI backtracer during propagation as well as activation",
    )
    parser.add_argument(
        "--no-activation-precheck",
        action="store_false",
        dest="activation_precheck",
        help="Disable zero-backtrack validation of the AI activation assignment",
    )
    parser.set_defaults(activation_precheck=True)
    parser.add_argument("--candidate-seed-base", type=int, default=20260504)
    parser.add_argument("--full", action="store_true", help="Ignore fault-list and run all faults")
    parser.add_argument("--limit-faults", type=int, default=0)
    parser.add_argument(
        "--exclude-primary-input-faults",
        action="store_true",
        help=(
            "Filter primary-input faults before --limit-faults. Useful for bounded code-gate "
            "smokes that need to exercise AI-guided internal justification."
        ),
    )
    parser.add_argument(
        "--compare-classic",
        action="store_true",
        help="Run classic PODEM on the same faults for backtrack comparison",
    )
    parser.add_argument("--classic-timeout", type=float, default=30.0)
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--notion-summary-out", default="")
    parser.add_argument("--baseline-coverage", type=float, default=0.1817)
    parser.add_argument("--baseline-label", default="unlinked_candidate 1% ITC99 gate")
    parser.add_argument("--baseline-source", default="docs/checkpoint_compatibility_summary.md")
    parser.add_argument("--coverage-target", type=float, default=1.0)
    parser.add_argument("--backtrack-target", action="store_true")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    if args.ai_attempts < 1:
        raise ValueError("--ai-attempts must be positive")

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
    if args.exclude_primary_input_faults:
        original_count = len(faults)
        faults = [fault for fault in faults if circuit[fault.gate_id].type != GateType.INPT]
        gate_meta = {
            **gate_meta,
            "primary_input_faults_excluded": True,
            "primary_input_faults_excluded_count": original_count - len(faults),
            "post_filter_faults": len(faults),
        }

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

    device = _select_device(args.device)
    config = AiPodemConfig(
        model_path=args.model,
        device=device,
        enable_ai_activation=True,
        enable_ai_propagation=args.enable_ai_propagation,
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
    ai_backtracks_total = 0
    ai_backtracks_on_success = 0
    activation_precheck_succeeded = 0
    classic_succeeded = 0
    classic_backtracks_total = 0
    classic_backtracks_on_ai_success = 0
    classic_time_total = 0.0
    ai_less_backtracks_count = 0
    per_fault = []

    initialize(circuit, total_gates)
    for idx, fault in enumerate(faults):
        detected = False
        attempts_used = 0
        precheck_success = False
        precheck_attempts = 0
        precheck_pi_assignments = 0
        fault_ai_backtracks = 0
        ai_error = None
        t0 = time.time()
        activation_val = (
            LogicValue.ONE if fault.value in (LogicValue.ZERO, LogicValue.D) else LogicValue.ZERO
        )
        if args.activation_precheck:
            for attempt in range(args.ai_attempts):
                precheck_attempts = attempt + 1
                attempts_used = attempt + 1
                current_seed = args.candidate_seed_base + idx + (attempt * len(faults))
                reset_gates(circuit, total_gates)
                ai_assignment = solver.solve(fault.gate_id, activation_val, seed=current_seed)
                if not ai_assignment:
                    continue
                precheck_pi_assignments = 0
                for gid, val in ai_assignment.items():
                    if circuit[gid].type == GateType.INPT:
                        circuit[gid].val = val
                        precheck_pi_assignments += 1
                if precheck_pi_assignments == 0:
                    continue
                logic_sim(circuit, total_gates, fault)
                if fault_is_at_po(circuit, total_gates):
                    detected = True
                    precheck_success = True
                    activation_precheck_succeeded += 1
                    break

        if not detected:
            for attempt in range(args.ai_attempts):
                attempts_used = attempt + 1
                reset_gates(circuit, total_gates)
                reset_statistics()
                try:
                    activation_val = (
                        LogicValue.ONE if fault.value in [LogicValue.ZERO, LogicValue.D] else LogicValue.ZERO
                    )
                    current_seed = args.candidate_seed_base + idx + (attempt * len(faults))
                    ai_assignment = solver.solve(fault.gate_id, activation_val, seed=current_seed)

                    backtracer = None
                    if ai_assignment:
                        for gid, val in ai_assignment.items():
                            if circuit[gid].type == GateType.INPT:
                                circuit[gid].val = val

                        if args.enable_ai_propagation:
                            backtracer = AIBacktracer(solver, no_fallback=True)
                        else:
                            backtracer = ImprovedHintBacktracer(ai_assignment, no_fallback=True)

                    result = podem(
                        circuit,
                        fault,
                        total_gates,
                        backtrace_func=backtracer,
                        max_backtracks=0,
                        timeout=5.0,
                    )
                    ok = (int(result) == SUCCESS)
                except Exception as exc:
                    ok = False
                    ai_error = str(exc)
                fault_ai_backtracks += int(get_statistics().get("backtrack_count", 0))
                if ok:
                    detected = True
                    ai_error = None
                    break
        elapsed = time.time() - t0
        total_time += elapsed
        ai_backtracks_total += fault_ai_backtracks
        succeeded += int(detected)
        failed += int(not detected)

        classic_ok = None
        classic_backtracks = None
        classic_elapsed = None
        ai_less_backtracks = None
        if args.compare_classic:
            reset_gates(circuit, total_gates)
            reset_statistics()
            classic_start = time.time()
            classic_result = podem(
                circuit,
                fault,
                total_gates,
                backtrace_func=simple_backtrace,
                timeout=args.classic_timeout,
                max_backtracks=args.max_backtracks,
            )
            classic_elapsed = time.time() - classic_start
            classic_ok = int(classic_result) == SUCCESS
            classic_backtracks = int(get_statistics().get("backtrack_count", 0))
            classic_succeeded += int(classic_ok)
            classic_backtracks_total += classic_backtracks
            classic_time_total += classic_elapsed
            if detected:
                ai_backtracks_on_success += fault_ai_backtracks
                classic_backtracks_on_ai_success += classic_backtracks
                ai_less_backtracks = fault_ai_backtracks < classic_backtracks
                ai_less_backtracks_count += int(ai_less_backtracks)
        per_fault.append(
            {
                "fault_index": idx,
                "gate_id": int(fault.gate_id),
                "fault_val": int(fault.value),
                "ok": detected,
                "attempts_used": attempts_used,
                "precheck_success": precheck_success,
                "precheck_attempts": precheck_attempts,
                "precheck_pi_assignments": precheck_pi_assignments,
                "ai_backtracks": fault_ai_backtracks,
                "classic_ok": classic_ok,
                "classic_backtracks": classic_backtracks,
                "ai_less_backtracks": ai_less_backtracks,
                "ai_error": ai_error,
                "time_s": round(elapsed, 4),
                "classic_time_s": (
                    round(classic_elapsed, 4) if classic_elapsed is not None else None
                ),
            }
        )
        if (idx + 1) % 100 == 0:
            print(
                f"ITC99 gate progress {idx + 1}/{len(faults)} "
                f"coverage={succeeded / (idx + 1):.2%}",
                flush=True,
            )

    coverage = succeeded / max(1, len(faults))
    backtrack_target_comparable = bool(args.compare_classic and succeeded)
    ai_backtrack_ratio = (
        ai_backtracks_on_success / classic_backtracks_on_ai_success
        if classic_backtracks_on_ai_success
        else None
    )
    passed_backtrack_target = (
        not args.backtrack_target
        or (
            backtrack_target_comparable
            and ai_backtracks_on_success < classic_backtracks_on_ai_success
        )
    )
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
        "device": device,
        "bench": bench_path,
        "fault_list": args.fault_list,
        "gate_meta": _compact_gate_meta(gate_meta),
        "candidate_count": args.candidate_count,
        "ai_attempts": args.ai_attempts,
        "activation_precheck": args.activation_precheck,
        "exclude_primary_input_faults": args.exclude_primary_input_faults,
        "activation_precheck_succeeded": activation_precheck_succeeded,
        "candidate_seed_base": args.candidate_seed_base,
        "enable_ai_propagation": args.enable_ai_propagation,
        "max_backtracks": args.max_backtracks,
        "total": len(faults),
        "succeeded": succeeded,
        "failed": failed,
        "coverage": coverage,
        "coverage_target": args.coverage_target,
        "passed_coverage_target": coverage >= args.coverage_target,
        "backtrack_target": args.backtrack_target,
        "passed_backtrack_target": passed_backtrack_target,
        "compare_classic": args.compare_classic,
        "classic_timeout": args.classic_timeout,
        "classic_succeeded": classic_succeeded,
        "ai_backtracks_total": ai_backtracks_total,
        "classic_backtracks_total": classic_backtracks_total,
        "ai_backtracks_on_success": ai_backtracks_on_success,
        "classic_backtracks_on_ai_success": classic_backtracks_on_ai_success,
        "ai_backtrack_ratio_on_success": ai_backtrack_ratio,
        "ai_less_backtracks_count": ai_less_backtracks_count,
        "total_time_s": round(total_time, 2),
        "classic_time_s": round(classic_time_total, 2),
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
