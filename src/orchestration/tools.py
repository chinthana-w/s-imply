"""Allowlisted local tools for the coordinator workflow."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALLOWED_ROOTS = (REPO_ROOT, Path("/tmp").resolve())


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ResultClaim:
    command: str
    artifact_path: str
    metric: str
    baseline: float
    observed: float
    conclusion: str


@dataclass(frozen=True)
class NotionDocumentationTarget:
    target: str
    content_format: str
    audience: str
    sync_style: str = "notion_canonical"
    experiment_log_format: str = (
        "append dated experiment log entries with commands, artifacts, results, and next steps"
    )
    owner: str = ""


def summarize_benchmark_artifact(path: str) -> dict[str, Any]:
    """Summarize a benchmark CSV or JSON artifact."""
    artifact = _resolve_repo_path(path)
    if artifact.suffix.lower() == ".json":
        with artifact.open() as f:
            payload = json.load(f)
        return _summarize_json_benchmark(payload)
    if artifact.suffix.lower() == ".csv":
        with artifact.open(newline="") as f:
            rows = list(csv.DictReader(f))
        return _summarize_csv_benchmark(rows)
    raise ValueError(f"Unsupported benchmark artifact type: {artifact.suffix}")


def inspect_checkpoint_config(path: str) -> dict[str, Any]:
    """Return checkpoint config metadata without constructing the model."""
    checkpoint = _resolve_repo_path(path)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local env
        return {"path": str(checkpoint), "ok": False, "error": f"torch unavailable: {exc}"}

    state: Any = torch.load(checkpoint, map_location="cpu")
    keys = sorted(state.keys()) if isinstance(state, dict) else []
    config = state.get("config") if isinstance(state, dict) else None
    return {
        "path": str(checkpoint),
        "ok": True,
        "keys": keys,
        "has_config": isinstance(config, dict),
        "config": config if isinstance(config, dict) else {},
    }


def run_focused_tests(test_targets: tuple[str, ...], dry_run: bool = True) -> CommandResult:
    """Run allowlisted focused pytest targets, or return the command in dry-run mode."""
    if not test_targets:
        raise ValueError("At least one focused test target is required")
    for target in test_targets:
        target_path = _resolve_repo_path(target)
        if not _is_relative_to(target_path, REPO_ROOT / "tests"):
            raise ValueError(f"Focused tests must live under tests/: {target}")

    command = ("python", "-m", "pytest", *test_targets)
    return _run_or_dry_run(command, dry_run=dry_run)


def run_test_coverage(
    test_targets: tuple[str, ...],
    coverage_json: str = "docs/test_coverage.json",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Run focused pytest targets with coverage JSON output."""
    if not test_targets:
        raise ValueError("At least one focused test target is required")
    for target in test_targets:
        target_path = _resolve_repo_path(target)
        if not _is_relative_to(target_path, REPO_ROOT / "tests"):
            raise ValueError(f"Focused tests must live under tests/: {target}")
    coverage_path = _resolve_repo_path(coverage_json, must_exist=False)
    command = (
        "python",
        "-m",
        "coverage",
        "run",
        "-m",
        "pytest",
        *test_targets,
    )
    report_command = ("python", "-m", "coverage", "json", "-o", str(coverage_path))
    if dry_run:
        return {
            "dry_run": True,
            "command": command,
            "report_command": report_command,
            "coverage_json": str(coverage_path),
        }

    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    report_result = subprocess.run(
        report_command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    summary: dict[str, Any] = {
        "dry_run": False,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "report_command": report_command,
        "report_returncode": report_result.returncode,
        "report_stdout": report_result.stdout,
        "report_stderr": report_result.stderr,
        "coverage_json": str(coverage_path),
    }
    if coverage_path.exists():
        with coverage_path.open() as f:
            coverage_payload = json.load(f)
        totals = coverage_payload.get("totals", {})
        summary["coverage_totals"] = totals
    return summary


def run_small_benchmark(
    model: str,
    fault_list: str = "data/bench/ITC99/b17_gate_10pct_faults.json",
    out: str = "docs/itc99_gate_report.json",
    dry_run: bool = True,
) -> CommandResult:
    """Run the allowlisted small ITC99 gate benchmark, or dry-run it."""
    _resolve_repo_path(model)
    _resolve_repo_path(fault_list)
    _resolve_repo_path(out, must_exist=False)
    command = (
        "python",
        "-m",
        "scripts.benchmark_itc99_gate",
        "--model",
        model,
        "--fault-list",
        fault_list,
        "--out",
        out,
    )
    return _run_or_dry_run(command, dry_run=dry_run)


def run_atpg(
    bench_path: str,
    limit_faults: int = 10,
    max_backtracks: int = 2000,
    timeout_s: float = 5.0,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Run vanilla PODEM over a bounded fault subset."""
    bench = _resolve_repo_path(bench_path)
    if limit_faults < 1:
        raise ValueError("limit_faults must be at least 1")
    if dry_run:
        return {
            "dry_run": True,
            "bench": str(bench),
            "limit_faults": limit_faults,
            "max_backtracks": max_backtracks,
            "timeout_s": timeout_s,
        }

    from src.atpg import podem as podem_module
    from src.atpg.podem import get_all_faults, initialize, podem
    from src.util.io import parse_bench_file

    circuit, total_gates = parse_bench_file(str(bench))
    faults = get_all_faults(circuit, total_gates)[:limit_faults]
    per_fault = []
    succeeded = 0
    for fault in faults:
        initialize(circuit, total_gates)
        detected = podem(
            circuit,
            fault,
            total_gates,
            timeout=timeout_s,
            max_backtracks=max_backtracks,
        )
        ok = int(detected) == podem_module.SUCCESS
        succeeded += int(ok)
        per_fault.append(
            {
                "gate_id": int(fault.gate_id),
                "fault_val": int(fault.value),
                "detected": ok,
                "result_code": int(detected),
                "backtracks": int(podem_module.backtrack_count),
            }
        )
    return {
        "dry_run": False,
        "bench": str(bench),
        "total_gates": total_gates,
        "faults_run": len(faults),
        "succeeded": succeeded,
        "failed": len(faults) - succeeded,
        "coverage": succeeded / max(1, len(faults)),
        "max_backtracks": max_backtracks,
        "timeout_s": timeout_s,
        "per_fault": per_fault,
    }


def simulate_circuit(
    bench_path: str,
    assignments: dict[str, int],
    fault_gate_id: int | None = None,
    fault_value: int | None = None,
) -> dict[str, Any]:
    """Forward-simulate a bench circuit with explicit gate/input assignments."""
    bench = _resolve_repo_path(bench_path)

    from src.atpg.logic_sim_three import logic_sim
    from src.atpg.util import get_topological_order
    from src.util.io import parse_bench_file
    from src.util.struct import Fault, GateType, LogicValue

    circuit, total_gates = parse_bench_file(str(bench))
    for raw_gate_id, raw_value in assignments.items():
        gate_id = int(raw_gate_id)
        if gate_id < 1 or gate_id > total_gates:
            raise ValueError(f"Assignment gate is outside circuit: {gate_id}")
        value = LogicValue(int(raw_value))
        circuit[gate_id].val = value

    fault = None
    if fault_gate_id is not None or fault_value is not None:
        if fault_gate_id is None or fault_value is None:
            raise ValueError("fault_gate_id and fault_value must be provided together")
        fault = Fault(int(fault_gate_id), LogicValue(int(fault_value)))

    topo_order = get_topological_order(circuit, total_gates)
    logic_sim(circuit, total_gates, fault=fault, topo_order=topo_order)
    outputs = {
        str(index): int(circuit[index].val)
        for index in range(1, total_gates + 1)
        if circuit[index].type != 0 and circuit[index].nfo == 0
    }
    primary_inputs = {
        str(index): int(circuit[index].val)
        for index in range(1, total_gates + 1)
        if circuit[index].type == GateType.INPT
    }
    return {
        "bench": str(bench),
        "total_gates": total_gates,
        "assignments": {str(key): int(value) for key, value in assignments.items()},
        "fault": None if fault is None else {"gate_id": fault.gate_id, "value": int(fault.value)},
        "primary_inputs": primary_inputs,
        "outputs": outputs,
    }


def validate_result_claim(claim: ResultClaim) -> dict[str, Any]:
    """Validate that a result claim has provenance and a parseable artifact."""
    artifact = _resolve_repo_path(claim.artifact_path)
    summary = summarize_benchmark_artifact(str(artifact))
    improvement = claim.observed - claim.baseline
    return {
        "ok": bool(claim.command and claim.metric and claim.conclusion and artifact.exists()),
        "claim": asdict(claim),
        "improvement": improvement,
        "artifact_summary": summary,
    }


def check_theory_doc_sync(changed_files: tuple[str, ...]) -> dict[str, Any]:
    """Require paper_draft.tex whenever a theoretical-framework file changed."""
    normalized = {path.replace(os.sep, "/") for path in changed_files}
    theory_touched = any(
        path.startswith("src/atpg/")
        or path in {"docs/project_summary.md"}
        or "theory" in path.lower()
        or "maamari" in path.lower()
        or "reconv" in path.lower()
        for path in normalized
    )
    paper_touched = "docs/paper_draft.tex" in normalized
    return {
        "ok": not theory_touched or paper_touched,
        "theory_touched": theory_touched,
        "paper_touched": paper_touched,
        "message": (
            "Theoretical-framework changes must update docs/paper_draft.tex."
            if theory_touched and not paper_touched
            else "Theory documentation is synchronized or not required."
        ),
    }


def validate_notion_documentation_target(target: NotionDocumentationTarget) -> dict[str, Any]:
    """Check whether Docs/Results has enough Notion setup detail to publish."""
    missing = []
    if not target.target:
        missing.append("target")
    if not target.content_format:
        missing.append("content_format")
    if not target.audience:
        missing.append("audience")
    if target.sync_style != "notion_canonical":
        missing.append("sync_style=notion_canonical")
    if "log" not in target.experiment_log_format.lower():
        missing.append("experiment_log_format")

    return {
        "ok": not missing,
        "missing": missing,
        "target": asdict(target),
        "message": (
            "Notion documentation target is configured."
            if not missing
            else "Configure the Notion target, content format, and audience before publishing."
        ),
    }


def _summarize_json_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "format": "json",
        "total": payload.get("total"),
        "succeeded": payload.get("succeeded"),
        "failed": payload.get("failed"),
        "coverage": payload.get("coverage"),
        "model": payload.get("model"),
        "bench": payload.get("bench"),
    }
    per_fault = payload.get("per_fault")
    if isinstance(per_fault, list) and per_fault:
        times = [float(item["time_s"]) for item in per_fault if "time_s" in item]
        if times:
            summary["mean_time_s"] = statistics.fmean(times)
            summary["max_time_s"] = max(times)
    return summary


def _summarize_csv_benchmark(rows: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"format": "csv", "rows": len(rows)}
    for field in ("ai_time_ms", "vanilla_time_ms", "vanilla_backtracks"):
        values = [_to_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            summary[f"{field}_mean"] = statistics.fmean(values)
            summary[f"{field}_max"] = max(values)
    if "ai_time_ms_mean" in summary and "vanilla_time_ms_mean" in summary:
        summary["ai_minus_vanilla_time_ms_mean"] = (
            summary["ai_time_ms_mean"] - summary["vanilla_time_ms_mean"]
        )
    return summary


def _run_or_dry_run(command: tuple[str, ...], dry_run: bool) -> CommandResult:
    if dry_run:
        return CommandResult(command=command, returncode=0, stdout="DRY RUN", stderr="")
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return CommandResult(
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _resolve_repo_path(path: str, must_exist: bool = True) -> Path:
    resolved = (REPO_ROOT / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if not any(_is_relative_to(resolved, root) for root in _allowed_roots()):
        raise ValueError(f"Path is outside allowed orchestration roots: {path}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(path)
    return resolved


def _allowed_roots() -> tuple[Path, ...]:
    extra = os.environ.get("S_IMPLY_ORCHESTRATION_ROOTS", "")
    roots = list(DEFAULT_ALLOWED_ROOTS)
    for raw_path in extra.split(os.pathsep):
        if raw_path:
            roots.append(Path(raw_path).resolve())
    return tuple(roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None
