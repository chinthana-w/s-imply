"""Benchmark AI-PODEM on the deterministic ITC99 gate subset.

This is the cheap held-out gate before running the full ITC99 benchmark.  It
never builds training data and never feeds results back into training.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.atpg.ai_podem import (
    AIBacktracer,
    AiPodemConfig,
    HierarchicalReconvSolver,
    ModelPairPredictor,
)
from src.atpg.logic_sim_three import fault_is_at_po, logic_sim, reset_gates
from src.atpg.podem import (
    SUCCESS,
    UNTESTABLE,
    get_all_faults,
    get_objective,
    get_statistics,
    initialize,
    podem,
    reset_statistics,
    simple_backtrace,
)
from src.util.io import parse_bench_file
from src.util.struct import Fault, GateType, LogicValue


def _logic_value_label(value: LogicValue | int) -> str:
    mapping = {
        LogicValue.ZERO: "0",
        LogicValue.ONE: "1",
        LogicValue.XD: "X",
        LogicValue.D: "D",
        LogicValue.DB: "DB",
    }
    try:
        return mapping[LogicValue(value)]
    except Exception:
        return str(value)


class ImprovedHintBacktracer:
    """Backtrace using AI hints, falling back to classic only when hints are missing.

    Unlike StaticHintBacktracer, this continues using hints for as many steps as
    possible. If a hint is missing for a particular gate, it falls back to
    simple_backtrace for the remainder of that specific backtrace path.
    """

    def __init__(
        self,
        hints: Dict[int, LogicValue],
        verbose: bool = False,
        no_fallback: bool = False,
        strict_no_fallback: bool = False,
    ):
        self.hints = {int(k): LogicValue(v) for k, v in hints.items()}
        self.verbose = verbose
        self.no_fallback = no_fallback
        self.strict_no_fallback = strict_no_fallback

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
                if self.strict_no_fallback:
                    if self.verbose:
                        print(f"[AI-HINT] No hint for gate {curr_id}; strict no-fallback failed.")
                    return Fault(-1, -1)
                # No hint for this gate's fanins.  No-fallback means the
                # benchmark will not do a clean classic retry after AI fails;
                # ordinary PODEM still needs its base backtrace for objectives
                # outside the AI hint cone.
                if self.verbose:
                    print(f"[AI-HINT] No hint for gate {curr_id}; using base backtrace.")
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


def _fault_item(index: int, fault: Fault) -> dict:
    return {
        "index": int(index),
        "gate_id": int(fault.gate_id),
        "fault_val": int(fault.value),
    }


def _write_fault_list(
    path: str,
    *,
    bench_path: str,
    faults: list[Fault],
    source_meta: dict,
    filter_meta: dict,
) -> None:
    payload = {
        "bench": bench_path,
        "total_faults": len(faults),
        "faults": [_fault_item(idx, fault) for idx, fault in enumerate(faults)],
        "source": {
            "fault_list": source_meta.get("fault_list"),
            "full": source_meta.get("full", False),
            "pre_filter_faults": filter_meta.get("pre_reconv_filter_faults"),
            "non_reconv_faults_skipped": filter_meta.get("non_reconv_faults_skipped"),
        },
        "filter": {
            "type": "reconv_only",
            "reconv_faults": filter_meta.get("reconv_faults", len(faults)),
        },
    }
    _write_json(path, payload)


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
        "ai_precheck_solve_time_s",
        "ai_precheck_sim_time_s",
        "ai_hint_solve_time_s",
        "ai_podem_search_time_s",
        "ai_result_code",
        "search_backtracks_diagnostic",
        "classic_ok",
        "classic_result_code",
        "classic_backtracks",
        "classic_recursive_calls",
        "ai_less_backtracks",
        "ai_error",
        "time_s",
        "classic_time_s",
    ]
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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


def _read_meminfo() -> dict[str, float]:
    values = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, raw_value = line.split(":", 1)
                parts = raw_value.strip().split()
                if parts:
                    values[key] = float(parts[0]) / (1024 * 1024)
    except OSError:
        return {}
    return values


def _process_rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024 * 1024)
    except OSError:
        return 0.0
    return 0.0


def _resource_snapshot() -> dict[str, float]:
    meminfo = _read_meminfo()
    total_gb = meminfo.get("MemTotal", 0.0)
    available_gb = meminfo.get("MemAvailable", 0.0)
    used_percent = 0.0
    if total_gb > 0:
        used_percent = max(0.0, 100.0 * (1.0 - (available_gb / total_gb)))
    return {
        "mem_total_gb": total_gb,
        "mem_available_gb": available_gb,
        "mem_used_percent": used_percent,
        "process_rss_gb": _process_rss_gb(),
    }


def _resource_pressure_level(args: argparse.Namespace, snapshot: dict[str, float]) -> str:
    """Return 'ok', 'flush' (soft pressure), or 'abort' (hard limit).

    'flush'  → caches should be cleared; run can continue after.
    'abort'  → resources critically low; stop cleanly.
    """
    guard_mode = getattr(args, "memory_guard_mode", "both")
    check_system = guard_mode in {"system", "both"}
    check_process = guard_mode in {"process", "both"}

    flush_threshold_gb = getattr(args, "mem_flush_threshold_gb", 0.0)

    # --- Hard abort limits ---
    if check_system and args.min_available_memory_gb > 0:
        if snapshot["mem_available_gb"] < args.min_available_memory_gb:
            return "abort"
    if check_system and args.max_system_memory_percent > 0:
        if snapshot["mem_used_percent"] > args.max_system_memory_percent:
            return "abort"
    if check_process and args.max_rss_gb > 0:
        if snapshot["process_rss_gb"] > args.max_rss_gb:
            return "abort"

    # --- Soft flush threshold (above the hard abort floor) ---
    if check_system and flush_threshold_gb > 0:
        if snapshot["mem_available_gb"] < flush_threshold_gb:
            return "flush"

    return "ok"


def _resource_abort_reason(args: argparse.Namespace, snapshot: dict[str, float]) -> str | None:
    """Legacy wrapper — returns a description string if we should abort, else None."""
    level = _resource_pressure_level(args, snapshot)
    if level != "abort":
        return None
    guard_mode = getattr(args, "memory_guard_mode", "both")
    check_system = guard_mode in {"system", "both"}
    check_process = guard_mode in {"process", "both"}
    if check_system and args.min_available_memory_gb > 0:
        if snapshot["mem_available_gb"] < args.min_available_memory_gb:
            return (
                f"available memory {snapshot['mem_available_gb']:.2f} GB fell below "
                f"{args.min_available_memory_gb:.2f} GB"
            )
    if check_system and args.max_system_memory_percent > 0:
        if snapshot["mem_used_percent"] > args.max_system_memory_percent:
            return (
                f"system memory use {snapshot['mem_used_percent']:.1f}% exceeded "
                f"{args.max_system_memory_percent:.1f}%"
            )
    if check_process and args.max_rss_gb > 0:
        if snapshot["process_rss_gb"] > args.max_rss_gb:
            return (
                f"process RSS {snapshot['process_rss_gb']:.2f} GB exceeded {args.max_rss_gb:.2f} GB"
            )
    return "hard memory limit exceeded"


def _flush_runtime_caches(
    *,
    solver: HierarchicalReconvSolver | None = None,
    predictor: ModelPairPredictor | None = None,
    device: str | None = None,
) -> dict[str, int | bool]:
    """Drop benchmark-side caches that can grow across a long fault sweep."""
    stats: dict[str, int | bool] = {
        "solver_pair_cache": 0,
        "predictor_prediction_cache": 0,
        "cuda_cache_flushed": False,
    }

    if solver is not None and hasattr(solver, "_persist_pair_cache_if_needed"):
        try:
            solver._persist_pair_cache_if_needed()
        except Exception:
            pass
    if solver is not None and hasattr(solver, "pair_cache"):
        pair_cache = getattr(solver, "pair_cache")
        try:
            stats["solver_pair_cache"] = len(pair_cache)
            pair_cache.clear()
            if hasattr(solver, "_pair_cache_dirty"):
                solver._pair_cache_dirty = False
        except Exception:
            stats["solver_pair_cache"] = 0

    if predictor is not None and hasattr(predictor, "prediction_cache"):
        prediction_cache = getattr(predictor, "prediction_cache")
        try:
            stats["predictor_prediction_cache"] = len(prediction_cache)
            prediction_cache.clear()
        except Exception:
            stats["predictor_prediction_cache"] = 0

    gc.collect()
    if device and str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            stats["cuda_cache_flushed"] = True
        except Exception:
            stats["cuda_cache_flushed"] = False
    return stats


def _ai_podem_backtrack_budget(args: argparse.Namespace) -> int:
    """No-fallback disables classic fallback, not PODEM's search budget."""
    if getattr(args, "strict_ai_no_fallback", False):
        return 0
    if getattr(args, "no_backtrack_limit", False):
        return sys.maxsize
    return int(args.max_backtracks)


def _classic_podem_backtrack_budget(args: argparse.Namespace) -> int:
    if getattr(args, "no_backtrack_limit", False):
        return sys.maxsize
    return int(args.max_backtracks)


def _filter_reconv_faults(
    faults: list[Fault],
    solver: HierarchicalReconvSolver,
    *,
    progress_every: int = 0,
    flush_every: int = 0,
    device: str | None = None,
) -> tuple[list[Fault], dict]:
    """Keep only faults whose target gate has reconvergent path pairs."""
    reconv_faults: list[Fault] = []
    skipped = 0
    for idx, fault in enumerate(faults, start=1):
        if fault.gate_id in solver.pair_cache:
            has_pairs = bool(solver.pair_cache[fault.gate_id])
        else:
            pairs = solver._collect_and_sort_pairs(fault.gate_id)
            solver.pair_cache[fault.gate_id] = pairs
            has_pairs = bool(pairs)
        if has_pairs:
            reconv_faults.append(fault)
        else:
            skipped += 1
        if progress_every > 0 and idx % progress_every == 0:
            print(
                f"Reconv filter progress {idx}/{len(faults)} "
                f"kept={len(reconv_faults)} skipped={skipped}",
                flush=True,
            )
        if flush_every > 0 and idx % flush_every == 0:
            stats = _flush_runtime_caches(solver=solver, device=device)
            snapshot = _resource_snapshot()
            print(
                f"[MEMORY-FLUSH] reconv_filter={idx}/{len(faults)} "
                f"cleared_pair_cache={stats['solver_pair_cache']} "
                f"rss={snapshot['process_rss_gb']:.2f}GB "
                f"mem_avail={snapshot['mem_available_gb']:.1f}GB",
                flush=True,
            )
    return reconv_faults, {
        "reconv_only": True,
        "pre_reconv_filter_faults": len(faults),
        "reconv_faults": len(reconv_faults),
        "non_reconv_faults_skipped": skipped,
    }


def _direct_ai_assignment_detection(
    circuit: list,
    total_gates: int,
    fault: Fault,
    assignment: Dict[int, LogicValue] | None,
) -> tuple[bool, int, float]:
    """Apply AI-proposed PI values once and simulate once.

    Strict no-fallback benchmarking treats the AI solver as a direct assignment
    proposer for reconvergent faults.  It must not enter a recursive PODEM search
    that can consume the per-fault wall-clock timeout.
    """
    if not assignment:
        return False, 0, 0.0

    pi_assignments = 0
    for gid, val in assignment.items():
        if 0 <= int(gid) < len(circuit) and circuit[int(gid)].type == GateType.INPT:
            circuit[int(gid)].val = val
            pi_assignments += 1

    if pi_assignments == 0:
        return False, 0, 0.0

    sim_start = time.time()
    logic_sim(circuit, total_gates, fault)
    sim_time = time.time() - sim_start
    return fault_is_at_po(circuit, total_gates), pi_assignments, sim_time


def _single_pass_structural_assignment(
    solver: HierarchicalReconvSolver,
    target_node: int,
    target_val: LogicValue,
) -> Dict[int, LogicValue] | None:
    """Derive a direct activation pattern without recursive PODEM search."""
    solver.nodes_visited = 0
    solver.inferences = 0
    return solver._backward_justify(
        queue=[int(target_node)],
        assignment={int(target_node): LogicValue(target_val)},
        solved_pairs=set(),
        sorted_pairs=[],
    )


def _no_backtrack_podem_detection(
    circuit: list,
    total_gates: int,
    fault: Fault,
    hints: Dict[int, LogicValue] | None,
    timeout: float,
    max_decisions: int = 5000,
) -> tuple[bool, int, float]:
    """Run a deterministic PODEM pass without fallback backtracking."""
    start = time.time()
    pi_assignments = 0
    for gid, val in (hints or {}).items():
        if 0 <= int(gid) < len(circuit) and circuit[int(gid)].type == GateType.INPT:
            circuit[int(gid)].val = LogicValue(val)
            pi_assignments += 1

    backtracer = ImprovedHintBacktracer(
        hints or {},
        no_fallback=True,
        strict_no_fallback=False,
    )

    for _ in range(max_decisions):
        if time.time() - start > timeout:
            return False, pi_assignments, time.time() - start

        logic_sim(circuit, total_gates, fault)
        if fault_is_at_po(circuit, total_gates):
            return True, pi_assignments, time.time() - start

        objective = get_objective(circuit, fault)
        if objective.gate_id == -1:
            return False, pi_assignments, time.time() - start

        pi_assignment = backtracer(objective, circuit)
        if pi_assignment.gate_id == -1:
            return False, pi_assignments, time.time() - start

        pi_id = int(pi_assignment.gate_id)
        desired_val = LogicValue(pi_assignment.value)
        if circuit[pi_id].val != LogicValue.XD and circuit[pi_id].val != desired_val:
            return False, pi_assignments, time.time() - start
        if circuit[pi_id].val == LogicValue.XD:
            circuit[pi_id].val = desired_val
            pi_assignments += 1

    return False, pi_assignments, time.time() - start


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
            "reconv_only": args.reconv_only,
            "reconv_fault_list_in": args.reconv_fault_list_in,
            "reconv_fault_list_out": args.reconv_fault_list_out,
            "reconv_filter_only": args.reconv_filter_only,
            "candidate_count": args.candidate_count,
            "ai_attempts": args.ai_attempts,
            "activation_precheck": args.activation_precheck,
            "candidate_seed_base": args.candidate_seed_base,
            "enable_ai_propagation": args.enable_ai_propagation,
            "strict_ai_no_fallback": getattr(args, "strict_ai_no_fallback", False),
            "max_backtracks": args.max_backtracks,
            "no_backtrack_limit": getattr(args, "no_backtrack_limit", False),
            "max_confidence_retries": args.max_confidence_retries,
            "ai_timeout": args.ai_timeout,
            "compare_classic": args.compare_classic,
            "classic_timeout": args.classic_timeout,
            "coverage_target": args.coverage_target,
            "backtrack_target": args.backtrack_target,
            "min_available_memory_gb": args.min_available_memory_gb,
            "max_system_memory_percent": args.max_system_memory_percent,
            "max_rss_gb": args.max_rss_gb,
            "memory_guard_mode": args.memory_guard_mode,
            "flush_every": args.flush_every,
            "torch_num_threads": args.torch_num_threads,
            "cooldown_s": args.cooldown_s,
            "progress_every": args.progress_every,
            "log_fault_start": args.log_fault_start,
        },
        "outputs": outputs,
        "baseline": {
            "label": args.baseline_label,
            "coverage": args.baseline_coverage,
            "source": args.baseline_source,
        },
    }


def _coverage_target_metrics(
    *,
    succeeded: int,
    total: int,
    attempted: int | None = None,
    classic_succeeded: int,
    compare_classic: bool,
    coverage_target: float,
    complete: bool = True,
) -> dict:
    attempted = total if attempted is None else attempted
    if compare_classic:
        denominator = classic_succeeded
        denominator_name = "classic_succeeded"
        denominator_note = "target is measured against faults covered by classic PODEM"
    else:
        if complete:
            denominator = total
            denominator_name = "total_faults"
            denominator_note = (
                "classic comparison was not enabled; target is measured against all "
                "benchmark faults in the configured scope"
            )
        else:
            denominator = attempted
            denominator_name = "attempted_faults"
            denominator_note = (
                "run did not complete; target progress is measured against attempted faults only"
            )

    required = math.ceil(coverage_target * denominator) if denominator else 0
    observed = succeeded / denominator if denominator else 0.0
    return {
        "denominator": denominator,
        "denominator_name": denominator_name,
        "denominator_note": denominator_note,
        "observed": observed,
        "required": required,
        "passed": (succeeded >= required if denominator else False) if complete else False,
    }


def _write_notion_summary(path: str, report: dict, manifest_path: str | None) -> None:
    baseline = report["baseline_comparison"]
    comparison_text = (
        f"{baseline['delta']:+.4%} absolute coverage"
        if baseline["decision_comparable"]
        else f"not decision-comparable: {baseline['comparison_note']}"
    )
    if report["compare_classic"]:
        backtrack_line = (
            f"- Classic search effort: `{report['classic_backtracks_total']}` total "
            f"backtracks, `{report['classic_backtracks_on_ai_success']}` on AI-solved "
            "faults; AI/model backtrack comparison=N/A"
        )
    else:
        backtrack_line = (
            f"- AI-guided PODEM search diagnostic: `{report['ai_backtracks_total']}` "
            "internal PODEM backtracks; "
            "classic not measured (`--compare-classic` was not enabled); "
            "AI/model backtrack comparison=N/A"
        )

    if report["backtrack_target"]:
        backtrack_target_line = (
            "- Backtrack target enabled: True; pass=N/A because AI has no "
            "comparable backtrack metric"
        )
    else:
        backtrack_target_line = "- Backtrack target enabled: False; pass=N/A"

    target_denominator = report.get("coverage_target_denominator_count", report["total"])
    target_observed = report.get("coverage_target_observed", report["coverage"])
    target_required = report.get("coverage_target_required_faults", 0)
    target_note = report.get(
        "coverage_target_denominator_note",
        "target is measured against all benchmark faults",
    )
    coverage_scope = report.get("coverage_scope", "full_filtered_fault_set")
    attempted = report.get("attempted", report["total"])
    attempted_coverage = report.get("attempted_coverage", report["coverage"])
    full_scope_coverage = report.get("full_scope_coverage", report["coverage"])

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
            f"({report['coverage']:.4%} no-fallback coverage, scope `{coverage_scope}`)",
            f"- Attempted coverage: `{report['succeeded']}/{attempted}` = "
            f"{attempted_coverage:.4%}; full configured-scope progress: "
            f"`{report['succeeded']}/{report['total']}` = {full_scope_coverage:.4%}",
            backtrack_line,
            f"- Activation precheck: {report['activation_precheck_succeeded']} "
            f"zero-backtrack detections",
            f"- Baseline: {baseline['label']} at {baseline['coverage']:.4%} "
            f"from `{baseline['source']}`",
            f"- Baseline comparison: {comparison_text}",
            f"- Coverage target: {report['coverage_target']:.4%} of "
            f"`{target_denominator}` denominator faults ({target_note}); "
            f"observed `{report['succeeded']}/{target_denominator}` = "
            f"{target_observed:.4%}; required `{target_required}`; "
            f"pass={report['passed_coverage_target']}",
            backtrack_target_line,
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
    parser.add_argument(
        "--no-backtrack-limit",
        action="store_true",
        help="Disable the PODEM backtrack cap; wall-clock timeout still applies",
    )
    parser.add_argument("--ai-timeout", type=float, default=5.0)
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
    parser.add_argument(
        "--strict-ai-no-fallback",
        action="store_true",
        help=(
            "Require AI to provide all backtrace guidance: no clean classic retry, "
            "no base simple_backtrace for missing hints, and zero AI PODEM backtracks."
        ),
    )
    parser.add_argument("--candidate-seed-base", type=int, default=20260504)
    parser.add_argument(
        "--max-confidence-retries",
        type=int,
        default=3,
        help=(
            "Number of confidence-guided retries inside solve_with_retry(). "
            "On failure the committed pair prediction with the lowest min-confidence "
            "is bypassed and the solve is retried up to this many times. "
            "Set to 0 to disable retries (single-shot mode)."
        ),
    )
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
        "--reconv-only",
        action="store_true",
        help=(
            "Evaluate only faults whose target gate has reconvergent path pairs. "
            "The reported total and coverage denominator become this filtered fault set."
        ),
    )
    parser.add_argument(
        "--reconv-fault-list-in",
        default="",
        help="Load a previously saved reconv-only fault list and skip reconv filtering",
    )
    parser.add_argument(
        "--reconv-fault-list-out",
        default="",
        help="Write the reconv-only fault list after filtering so future runs can reuse it",
    )
    parser.add_argument(
        "--reconv-filter-only",
        action="store_true",
        help="Apply/load the reconv fault filter, write requested artifacts, and exit",
    )
    parser.add_argument(
        "--compare-classic",
        action="store_true",
        help="Run classic PODEM on the same faults for backtrack comparison",
    )
    parser.add_argument("--classic-timeout", type=float, default=30.0)
    parser.add_argument(
        "--classic-cache",
        default="",
        help="JSON file containing cached classic PODEM results to bypass execution",
    )
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--notion-summary-out", default="")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="When --csv-out is set, rewrite the per-fault CSV every N faults for long runs",
    )
    parser.add_argument("--baseline-coverage", type=float, default=0.1817)
    parser.add_argument("--baseline-label", default="unlinked_candidate 1% ITC99 gate")
    parser.add_argument("--baseline-source", default="docs/checkpoint_compatibility_summary.md")
    parser.add_argument("--coverage-target", type=float, default=1.0)
    parser.add_argument("--backtrack-target", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--min-available-memory-gb",
        type=float,
        default=4.0,
        help="Abort cleanly before a fault if host available RAM drops below this value",
    )
    parser.add_argument(
        "--mem-flush-threshold-gb",
        type=float,
        default=10.0,
        help=(
            "Flush solver/predictor caches when available RAM falls below this value "
            "(must be above --min-available-memory-gb). Set to 0 to disable adaptive flushing."
        ),
    )
    parser.add_argument(
        "--max-system-memory-percent",
        type=float,
        default=90.0,
        help="Abort cleanly before a fault if system RAM use exceeds this percentage",
    )
    parser.add_argument(
        "--max-rss-gb",
        type=float,
        default=24.0,
        help="Abort cleanly before a fault if this process exceeds the RSS limit",
    )
    parser.add_argument(
        "--memory-guard-mode",
        choices=("both", "system", "process"),
        default="both",
        help=(
            "Select which memory limits are enforced. 'process' ignores host-wide "
            "MemAvailable/used-percent checks and only guards this benchmark RSS."
        ),
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=2000,
        help="Clear benchmark-side solver/model caches every N attempted faults; use 0 to disable",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=1,
        help="Limit PyTorch CPU worker threads; use 0 to leave unchanged",
    )
    parser.add_argument(
        "--cooldown-s",
        type=float,
        default=0.0,
        help="Sleep this many seconds after each fault to keep the workstation responsive",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N attempted faults",
    )
    parser.add_argument(
        "--log-fault-start",
        action="store_true",
        help="Print one line before each fault starts so long timeouts look alive in logs",
    )
    parser.add_argument(
        "--start-from-fault",
        type=int,
        default=0,
        help=(
            "Skip the first N faults (0-indexed). Use to resume an aborted run. "
            "Pass the fault index printed in the last '[RESOURCE-GUARD] Aborting' line."
        ),
    )
    args = parser.parse_args()
    classic_cache = {}
    if args.classic_cache:
        with open(args.classic_cache) as f:
            cache_data = json.load(f)
        for row in cache_data.get("per_fault", []):
            f_idx = row["fault_index"]
            classic_cache[f_idx] = {
                "classic_ok": row.get("classic_ok"),
                "classic_result_code": row.get("classic_result_code"),
                "classic_backtracks": row.get("classic_backtracks", 0),
                "classic_recursive_calls": row.get("classic_recursive_calls", 0),
                "classic_time_s": row.get("classic_time_s", 0.0),
            }
    if args.ai_attempts < 1:
        raise ValueError("--ai-attempts must be positive")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if args.flush_every < 0:
        raise ValueError("--flush-every must be non-negative")
    flush_thresh = getattr(args, "mem_flush_threshold_gb", 0.0)
    abort_thresh = getattr(args, "min_available_memory_gb", 0.0)
    if flush_thresh > 0 and abort_thresh > 0 and flush_thresh <= abort_thresh:
        raise ValueError(
            "--mem-flush-threshold-gb must be above --min-available-memory-gb "
            f"(got flush={flush_thresh}, abort={abort_thresh})"
        )
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

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
        max_confidence_retries=args.max_confidence_retries,
    )
    predictor = ModelPairPredictor(circuit, bench_path, config)
    solver = HierarchicalReconvSolver(circuit, predictor, circuit_path=bench_path)

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

    if args.reconv_fault_list_in:
        loaded_bench_path, loaded_faults, reconv_list_meta = _load_gate_faults(
            args.reconv_fault_list_in
        )
        if loaded_bench_path != bench_path:
            raise ValueError(
                "--reconv-fault-list-in bench does not match selected benchmark: "
                f"{loaded_bench_path!r} != {bench_path!r}"
            )
        original_count = len(faults)
        faults = loaded_faults
        gate_meta = {
            **gate_meta,
            "reconv_only": True,
            "reconv_fault_list_in": args.reconv_fault_list_in,
            "loaded_reconv_faults": len(faults),
            "original_faults_before_reconv_load": original_count,
            "post_filter_faults": len(faults),
            "reconv_fault_list_meta": _compact_gate_meta(reconv_list_meta),
        }
        args.reconv_only = True
    elif args.reconv_only:
        original_count = len(faults)
        faults, reconv_meta = _filter_reconv_faults(
            faults,
            solver,
            progress_every=args.progress_every,
            flush_every=args.flush_every,
            device=device,
        )
        gate_meta = {
            **gate_meta,
            **reconv_meta,
            "original_faults_before_reconv_filter": original_count,
            "post_filter_faults": len(faults),
        }
        if not faults:
            raise RuntimeError("--reconv-only selected, but no reconvergent faults were found")
        if args.reconv_fault_list_out:
            source_meta = {
                **gate_meta,
                "fault_list": args.fault_list,
                "full": args.full,
            }
            _write_fault_list(
                args.reconv_fault_list_out,
                bench_path=bench_path,
                faults=faults,
                source_meta=source_meta,
                filter_meta=reconv_meta,
            )
            gate_meta = {
                **gate_meta,
                "reconv_fault_list_out": args.reconv_fault_list_out,
            }

    if args.reconv_filter_only:
        if not args.reconv_only:
            raise RuntimeError(
                "--reconv-filter-only requires --reconv-only or --reconv-fault-list-in"
            )
        outputs = [args.out]
        if args.manifest_out:
            outputs.append(args.manifest_out)
        artifact_paths = {"json": args.out}
        if args.reconv_fault_list_out:
            artifact_paths["reconv_fault_list"] = args.reconv_fault_list_out
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, "-m", "scripts.benchmark_itc99_gate", *sys.argv[1:]],
            "run_id": args.run_id,
            "model": args.model,
            "device": device,
            "bench": bench_path,
            "fault_list": args.fault_list,
            "gate_meta": _compact_gate_meta(gate_meta),
            "reconv_only": True,
            "reconv_filter_only": True,
            "total": len(faults),
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "coverage": 0.0,
            "coverage_scope": "not_applicable_filter_only",
            "artifact_paths": artifact_paths,
        }
        _write_json(args.out, report)
        if args.manifest_out:
            _write_json(args.manifest_out, _build_manifest(args, outputs))
        print(f"ITC99 reconv filter-only: {len(faults)} reconv faults; wrote {args.out}")
        return

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
    aborted_reason = None
    final_resource_snapshot = _resource_snapshot()

    initialize(circuit, total_gates)
    start_from = max(0, getattr(args, "start_from_fault", 0))
    if start_from > 0:
        print(
            f"[RESUME] Skipping first {start_from} faults (--start-from-fault={start_from})",
            flush=True,
        )
    _consecutive_flush_attempts = 0  # tracks back-to-back soft-pressure flushes
    for idx, fault in enumerate(faults):
        if idx < start_from:
            continue
        resource_snapshot = _resource_snapshot()
        final_resource_snapshot = resource_snapshot
        pressure = _resource_pressure_level(args, resource_snapshot)

        # --- Adaptive flush on soft pressure ---
        if pressure == "flush":
            _consecutive_flush_attempts += 1
            flush_stats = _flush_runtime_caches(solver=solver, predictor=predictor, device=device)
            time.sleep(min(2.0 * _consecutive_flush_attempts, 10.0))
            resource_snapshot = _resource_snapshot()
            final_resource_snapshot = resource_snapshot
            pressure = _resource_pressure_level(args, resource_snapshot)
            print(
                f"[ADAPTIVE-FLUSH] fault={idx} attempt={_consecutive_flush_attempts} "
                f"cleared_pair_cache={flush_stats['solver_pair_cache']} "
                f"cleared_pred_cache={flush_stats['predictor_prediction_cache']} "
                f"mem_avail={resource_snapshot['mem_available_gb']:.1f}GB "
                f"pressure_after={pressure}",
                flush=True,
            )
        else:
            _consecutive_flush_attempts = 0

        # --- Hard abort if still critical after flush ---
        if pressure == "abort":
            abort_reason = _resource_abort_reason(args, resource_snapshot)
            aborted_reason = f"before fault {idx}: {abort_reason}"
            print(
                f"[RESOURCE-GUARD] Aborting cleanly: {aborted_reason} "
                f"(hint: rerun with --start-from-fault {idx})",
                flush=True,
            )
            break
        if args.log_fault_start:
            print(
                f"ITC99 gate fault_start {idx + 1}/{len(faults)} "
                f"gate={int(fault.gate_id)} fault_val={int(fault.value)} "
                f"rss={resource_snapshot['process_rss_gb']:.2f}GB "
                f"mem_avail={resource_snapshot['mem_available_gb']:.1f}GB",
                flush=True,
            )

        detected = False
        attempts_used = 0
        precheck_success = False
        precheck_attempts = 0
        precheck_pi_assignments = 0
        ai_precheck_solve_time = 0.0
        ai_precheck_sim_time = 0.0
        ai_hint_solve_time = 0.0
        ai_podem_search_time = 0.0
        fault_ai_backtracks = 0
        ai_result_code = None
        ai_error = None
        t0 = time.time()
        activation_val = (
            LogicValue.ONE if fault.value in (LogicValue.ZERO, LogicValue.D) else LogicValue.ZERO
        )
        if fault.gate_id in solver.pair_cache:
            has_reconv_pairs = bool(solver.pair_cache[fault.gate_id])
        else:
            pairs = solver._collect_and_sort_pairs(fault.gate_id)
            solver.pair_cache[fault.gate_id] = pairs
            has_reconv_pairs = bool(pairs)
        classic_ok = None
        classic_result_code = None
        classic_backtracks = None
        classic_recursive_calls = None
        classic_elapsed = None
        ai_less_backtracks = None

        if args.compare_classic or not has_reconv_pairs:
            if args.classic_cache and idx in classic_cache:
                c_data = classic_cache[idx]
                classic_ok = c_data["classic_ok"]
                classic_result_code = c_data["classic_result_code"]
                classic_backtracks = c_data["classic_backtracks"]
                classic_recursive_calls = c_data["classic_recursive_calls"]
                classic_elapsed = c_data["classic_time_s"]
                classic_succeeded += int(classic_ok)
                classic_backtracks_total += classic_backtracks
                classic_time_total += classic_elapsed
            else:
                reset_gates(circuit, total_gates)
                reset_statistics()
                classic_start = time.time()
                classic_result = podem(
                    circuit,
                    fault,
                    total_gates,
                    backtrace_func=simple_backtrace,
                    timeout=args.classic_timeout,
                    max_backtracks=_classic_podem_backtrack_budget(args),
                )
                classic_elapsed = time.time() - classic_start
                classic_result_code = int(classic_result)
                classic_ok = classic_result_code == SUCCESS
                classic_stats = get_statistics()
                classic_backtracks = int(classic_stats.get("backtrack_count", 0))
                classic_recursive_calls = int(classic_stats.get("total_recursive_calls", 0))
                classic_succeeded += int(classic_ok)
                classic_backtracks_total += classic_backtracks
                classic_time_total += classic_elapsed

        if not has_reconv_pairs:
            detected = bool(classic_ok)
            ai_result_code = classic_result_code
            elapsed = classic_elapsed if classic_elapsed is not None else time.time() - t0
            fault_ai_backtracks = int(classic_backtracks or 0)
            total_time += elapsed
            ai_backtracks_total += fault_ai_backtracks
            succeeded += int(detected)
            failed += int(not detected)
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
                    "ai_precheck_solve_time_s": 0.0,
                    "ai_precheck_sim_time_s": 0.0,
                    "ai_hint_solve_time_s": 0.0,
                    "ai_podem_search_time_s": round(elapsed, 4),
                    "ai_result_code": ai_result_code,
                    "search_backtracks_diagnostic": fault_ai_backtracks,
                    "classic_ok": classic_ok,
                    "classic_result_code": classic_result_code,
                    "classic_backtracks": classic_backtracks,
                    "classic_recursive_calls": classic_recursive_calls,
                    "ai_less_backtracks": None,
                    "ai_error": None,
                    "time_s": round(elapsed, 4),
                    "classic_time_s": (
                        round(classic_elapsed, 4) if classic_elapsed is not None else None
                    ),
                }
            )
            if (idx + 1) % args.progress_every == 0:
                print(
                    f"ITC99 gate progress {idx + 1}/{len(faults)} "
                    f"attempted_coverage={succeeded / (idx + 1):.2%} "
                    f"rss={resource_snapshot['process_rss_gb']:.2f}GB "
                    f"mem_avail={resource_snapshot['mem_available_gb']:.1f}GB",
                    flush=True,
                )
            if args.csv_out and args.checkpoint_every and (idx + 1) % args.checkpoint_every == 0:
                _write_csv(args.csv_out, per_fault)
            if args.flush_every > 0 and (idx + 1) % args.flush_every == 0:
                flush_stats = _flush_runtime_caches(
                    solver=solver,
                    predictor=predictor,
                    device=device,
                )
                flush_snapshot = _resource_snapshot()
                print(
                    f"[MEMORY-FLUSH] attempted={idx + 1}/{len(faults)} "
                    f"cleared_pair_cache={flush_stats['solver_pair_cache']} "
                    f"cleared_prediction_cache={flush_stats['predictor_prediction_cache']} "
                    f"cuda={flush_stats['cuda_cache_flushed']} "
                    f"rss={flush_snapshot['process_rss_gb']:.2f}GB "
                    f"mem_avail={flush_snapshot['mem_available_gb']:.1f}GB",
                    flush=True,
                )
            if args.cooldown_s > 0:
                time.sleep(args.cooldown_s)
            gc.collect()
            continue

        if args.activation_precheck and has_reconv_pairs:
            for attempt in range(args.ai_attempts):
                precheck_attempts = attempt + 1
                attempts_used = attempt + 1
                current_seed = args.candidate_seed_base + idx + (attempt * len(faults))
                reset_gates(circuit, total_gates)
                solve_start = time.time()
                ai_assignment = solver.solve(fault.gate_id, activation_val, seed=current_seed)
                ai_precheck_solve_time += time.time() - solve_start
                detected, pi_count, sim_time = _direct_ai_assignment_detection(
                    circuit,
                    total_gates,
                    fault,
                    ai_assignment,
                )
                precheck_pi_assignments = pi_count
                ai_precheck_sim_time += sim_time
                if not detected:
                    reset_gates(circuit, total_gates)
                    solve_start = time.time()
                    structural_assignment = _single_pass_structural_assignment(
                        solver,
                        fault.gate_id,
                        activation_val,
                    )
                    ai_precheck_solve_time += time.time() - solve_start
                    detected, pi_count, sim_time = _direct_ai_assignment_detection(
                        circuit,
                        total_gates,
                        fault,
                        structural_assignment,
                    )
                    precheck_pi_assignments = max(precheck_pi_assignments, pi_count)
                    ai_precheck_sim_time += sim_time
                    if not detected:
                        reset_gates(circuit, total_gates)
                        detected, pi_count, search_time = _no_backtrack_podem_detection(
                            circuit,
                            total_gates,
                            fault,
                            structural_assignment or ai_assignment,
                            timeout=args.ai_timeout,
                        )
                        precheck_pi_assignments = max(precheck_pi_assignments, pi_count)
                        ai_podem_search_time += search_time
                if detected:
                    ai_result_code = SUCCESS
                    detected = True
                    precheck_success = True
                    activation_precheck_succeeded += 1
                    break

        if not detected:
            if args.strict_ai_no_fallback and has_reconv_pairs:
                if not args.activation_precheck:
                    for attempt in range(args.ai_attempts):
                        attempts_used = attempt + 1
                        current_seed = args.candidate_seed_base + idx + (attempt * len(faults))
                        reset_gates(circuit, total_gates)
                        solve_start = time.time()
                        ai_assignment = solver.solve(
                            fault.gate_id,
                            activation_val,
                            seed=current_seed,
                        )
                        ai_precheck_solve_time += time.time() - solve_start
                        detected, pi_count, sim_time = _direct_ai_assignment_detection(
                            circuit,
                            total_gates,
                            fault,
                            ai_assignment,
                        )
                        precheck_pi_assignments = pi_count
                        ai_precheck_sim_time += sim_time
                        if not detected:
                            reset_gates(circuit, total_gates)
                            solve_start = time.time()
                            structural_assignment = _single_pass_structural_assignment(
                                solver,
                                fault.gate_id,
                                activation_val,
                            )
                            ai_precheck_solve_time += time.time() - solve_start
                            detected, pi_count, sim_time = _direct_ai_assignment_detection(
                                circuit,
                                total_gates,
                                fault,
                                structural_assignment,
                            )
                            precheck_pi_assignments = max(precheck_pi_assignments, pi_count)
                            ai_precheck_sim_time += sim_time
                            if not detected:
                                reset_gates(circuit, total_gates)
                                detected, pi_count, search_time = _no_backtrack_podem_detection(
                                    circuit,
                                    total_gates,
                                    fault,
                                    structural_assignment or ai_assignment,
                                    timeout=args.ai_timeout,
                                )
                                precheck_pi_assignments = max(precheck_pi_assignments, pi_count)
                                ai_podem_search_time += search_time
                        if detected:
                            ai_result_code = SUCCESS
                            precheck_success = True
                            activation_precheck_succeeded += 1
                            break
                if not detected and ai_error is None:
                    ai_error = "strict no-fallback: direct AI assignment did not detect fault"
                if not detected and ai_result_code is None:
                    ai_result_code = UNTESTABLE

        if not detected and not (args.strict_ai_no_fallback and has_reconv_pairs):
            for attempt in range(args.ai_attempts):
                attempts_used = attempt + 1
                reset_gates(circuit, total_gates)
                reset_statistics()
                try:
                    activation_val = (
                        LogicValue.ONE
                        if fault.value in [LogicValue.ZERO, LogicValue.D]
                        else LogicValue.ZERO
                    )
                    current_seed = args.candidate_seed_base + idx + (attempt * len(faults))
                    solve_start = time.time()
                    ai_assignment = solver.solve(fault.gate_id, activation_val, seed=current_seed)
                    ai_hint_solve_time += time.time() - solve_start

                    backtracer = None
                    if ai_assignment:
                        # The precheck above is the only place where AI PI prefill
                        # is accepted as a complete zero-backtrack pattern.  If it
                        # did not detect the fault, do not lock those PI values into
                        # PODEM; use the assignment as guidance instead so ordinary
                        # PODEM backtracking can recover from bad model activations.
                        if args.enable_ai_propagation:
                            backtracer = AIBacktracer(solver, no_fallback=True)
                        else:
                            backtracer = ImprovedHintBacktracer(
                                ai_assignment,
                                no_fallback=True,
                                strict_no_fallback=args.strict_ai_no_fallback,
                            )
                    elif args.strict_ai_no_fallback and has_reconv_pairs:
                        ai_error = "strict no-fallback: AI solver returned no assignment"
                        break

                    search_start = time.time()
                    search_backtrack_budget = (
                        _ai_podem_backtrack_budget(args)
                        if has_reconv_pairs
                        else _classic_podem_backtrack_budget(args)
                    )
                    result = podem(
                        circuit,
                        fault,
                        total_gates,
                        backtrace_func=backtracer,
                        max_backtracks=search_backtrack_budget,
                        timeout=args.ai_timeout,
                    )
                    ai_podem_search_time += time.time() - search_start
                    ai_result_code = int(result)
                    ok = int(result) == SUCCESS
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

        if args.compare_classic and classic_result_code is None:
            if args.classic_cache and idx in classic_cache:
                c_data = classic_cache[idx]
                classic_ok = c_data["classic_ok"]
                classic_result_code = c_data["classic_result_code"]
                classic_backtracks = c_data["classic_backtracks"]
                classic_recursive_calls = c_data["classic_recursive_calls"]
                classic_elapsed = c_data["classic_time_s"]
                classic_succeeded += int(classic_ok)
                classic_backtracks_total += classic_backtracks
                classic_time_total += classic_elapsed
            else:
                reset_gates(circuit, total_gates)
                reset_statistics()
                classic_start = time.time()
                classic_result = podem(
                    circuit,
                    fault,
                    total_gates,
                    backtrace_func=simple_backtrace,
                    timeout=args.classic_timeout,
                    max_backtracks=_classic_podem_backtrack_budget(args),
                )
                classic_elapsed = time.time() - classic_start
                classic_result_code = int(classic_result)
                classic_ok = classic_result_code == SUCCESS
                classic_stats = get_statistics()
                classic_backtracks = int(classic_stats.get("backtrack_count", 0))
                classic_recursive_calls = int(classic_stats.get("total_recursive_calls", 0))
                classic_succeeded += int(classic_ok)
                classic_backtracks_total += classic_backtracks
                classic_time_total += classic_elapsed
        if args.compare_classic and classic_ok is not None and detected:
            ai_backtracks_on_success += fault_ai_backtracks
            classic_backtracks_on_ai_success += classic_backtracks
            ai_less_backtracks = None
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
                "ai_precheck_solve_time_s": round(ai_precheck_solve_time, 4),
                "ai_precheck_sim_time_s": round(ai_precheck_sim_time, 4),
                "ai_hint_solve_time_s": round(ai_hint_solve_time, 4),
                "ai_podem_search_time_s": round(ai_podem_search_time, 4),
                "ai_result_code": ai_result_code,
                "search_backtracks_diagnostic": fault_ai_backtracks,
                "classic_ok": classic_ok,
                "classic_result_code": classic_result_code,
                "classic_backtracks": classic_backtracks,
                "classic_recursive_calls": classic_recursive_calls,
                "ai_less_backtracks": ai_less_backtracks,
                "ai_error": ai_error,
                "time_s": round(elapsed, 4),
                "classic_time_s": (
                    round(classic_elapsed, 4) if classic_elapsed is not None else None
                ),
            }
        )
        if (idx + 1) % args.progress_every == 0:
            print(
                f"ITC99 gate progress {idx + 1}/{len(faults)} "
                f"attempted_coverage={succeeded / (idx + 1):.2%} "
                f"rss={resource_snapshot['process_rss_gb']:.2f}GB "
                f"mem_avail={resource_snapshot['mem_available_gb']:.1f}GB",
                flush=True,
            )
        if args.csv_out and args.checkpoint_every and (idx + 1) % args.checkpoint_every == 0:
            _write_csv(args.csv_out, per_fault)
        if args.flush_every > 0 and (idx + 1) % args.flush_every == 0:
            flush_stats = _flush_runtime_caches(
                solver=solver,
                predictor=predictor,
                device=device,
            )
            flush_snapshot = _resource_snapshot()
            print(
                f"[MEMORY-FLUSH] attempted={idx + 1}/{len(faults)} "
                f"cleared_pair_cache={flush_stats['solver_pair_cache']} "
                f"cleared_prediction_cache={flush_stats['predictor_prediction_cache']} "
                f"cuda={flush_stats['cuda_cache_flushed']} "
                f"rss={flush_snapshot['process_rss_gb']:.2f}GB "
                f"mem_avail={flush_snapshot['mem_available_gb']:.1f}GB",
                flush=True,
            )
        if args.cooldown_s > 0:
            time.sleep(args.cooldown_s)
        gc.collect()

    attempted = len(per_fault)
    attempted_coverage = succeeded / max(1, attempted)
    full_scope_coverage = succeeded / max(1, len(faults))
    run_complete = aborted_reason is None and attempted == len(faults)
    coverage = full_scope_coverage if run_complete else attempted_coverage
    coverage_target_metrics = _coverage_target_metrics(
        succeeded=succeeded,
        total=len(faults),
        attempted=attempted,
        classic_succeeded=classic_succeeded,
        compare_classic=args.compare_classic,
        coverage_target=args.coverage_target,
        complete=run_complete,
    )
    passed_backtrack_target = None
    backtrack_target_comparable = False
    ai_backtrack_ratio = None
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
        "reconv_only": args.reconv_only,
        "reconv_fault_list_in": args.reconv_fault_list_in,
        "reconv_fault_list_out": args.reconv_fault_list_out,
        "reconv_filter_only": args.reconv_filter_only,
        "activation_precheck_succeeded": activation_precheck_succeeded,
        "candidate_seed_base": args.candidate_seed_base,
        "enable_ai_propagation": args.enable_ai_propagation,
        "strict_ai_no_fallback": getattr(args, "strict_ai_no_fallback", False),
        "max_backtracks": args.max_backtracks,
        "no_backtrack_limit": getattr(args, "no_backtrack_limit", False),
        "ai_timeout": args.ai_timeout,
        "total": len(faults),
        "attempted": attempted,
        "aborted": aborted_reason is not None,
        "aborted_reason": aborted_reason,
        "complete": run_complete,
        "resource_snapshot": final_resource_snapshot,
        "resource_limits": {
            "min_available_memory_gb": args.min_available_memory_gb,
            "max_system_memory_percent": args.max_system_memory_percent,
            "max_rss_gb": args.max_rss_gb,
            "memory_guard_mode": args.memory_guard_mode,
            "flush_every": args.flush_every,
            "torch_num_threads": args.torch_num_threads,
            "cooldown_s": args.cooldown_s,
            "log_fault_start": args.log_fault_start,
        },
        "succeeded": succeeded,
        "failed": failed,
        "coverage": coverage,
        "coverage_scope": "full_filtered_fault_set" if run_complete else "attempted_faults",
        "attempted_coverage": attempted_coverage,
        "full_scope_coverage": full_scope_coverage,
        "coverage_target": args.coverage_target,
        "coverage_target_observed": coverage_target_metrics["observed"],
        "coverage_target_required_faults": coverage_target_metrics["required"],
        "coverage_target_denominator": coverage_target_metrics["denominator_name"],
        "coverage_target_denominator_count": coverage_target_metrics["denominator"],
        "coverage_target_denominator_note": coverage_target_metrics["denominator_note"],
        "classic_relative_coverage": (
            coverage_target_metrics["observed"] if args.compare_classic else None
        ),
        "passed_coverage_target": coverage_target_metrics["passed"],
        "backtrack_target": args.backtrack_target,
        "passed_backtrack_target": passed_backtrack_target,
        "backtrack_target_comparable": backtrack_target_comparable,
        "backtrack_comparison_note": (
            "classic backtracks were not measured; AI has no comparable backtrack metric"
            if not args.compare_classic
            else (
                "classic backtracks were measured for ranking only; AI has no "
                "comparable backtrack metric"
            )
        ),
        "ai_backtrack_metric_valid": False,
        "ai_backtrack_metric_note": (
            "search_backtracks_diagnostic is the internal PODEM search counter for the "
            "executed path; it is not an AI/model backtrack count"
        ),
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
            "decision_comparable": args.limit_faults == 0 and run_complete,
            "comparison_note": (
                "bounded --limit-faults smoke validates the benchmark path but is not "
                "a statistically valid baseline comparison"
                if args.limit_faults
                else (
                    "run did not complete; compare final full-scope coverage only after "
                    "all configured faults are attempted"
                    if not run_complete
                    else "same configured benchmark scope"
                )
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
    aborted_text = f"; aborted: {aborted_reason}" if aborted_reason else ""
    print(
        f"ITC99 gate coverage: {succeeded}/{len(faults)} "
        f"= {full_scope_coverage:.2%} full-scope; "
        f"attempted coverage {succeeded}/{attempted} = {attempted_coverage:.2%}; "
        f"attempted {attempted}/{len(faults)}"
        f"{aborted_text}; wrote {args.out}"
    )


if __name__ == "__main__":
    main()
