"""Persistent runner state for the multi-agent orchestration workflow."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.orchestration.coordinator import TaskPacket, create_task_packet
from src.orchestration.tools import REPO_ROOT

DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "orchestration"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RunEvent:
    timestamp: str
    status: RunStatus
    message: str
    artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    goal: str
    status: RunStatus
    created_at: str
    updated_at: str
    run_dir: str
    task_packet: dict[str, Any]
    events: tuple[RunEvent, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LaunchResult:
    run_id: str
    command: tuple[str, ...]
    returncode: int
    stdout_path: str
    stderr_path: str
    final_status: RunStatus


@dataclass(frozen=True)
class RuntimeAgentSpec:
    role: str
    phase: str
    goal: str


@dataclass(frozen=True)
class MultiAgentRuntimeResult:
    run_id: str
    child_run_ids: tuple[str, ...]
    phase_results: tuple[LaunchResult, ...]
    final_status: RunStatus
    summary_path: str


def dispatch_task(
    goal: str,
    changed_files: tuple[str, ...] = (),
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    run_id: str | None = None,
) -> RunRecord:
    """Create a persistent run directory and specialist handoff packet."""
    packet = create_task_packet(goal, changed_files=changed_files)
    root = Path(runs_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    resolved_run_id = run_id or _make_run_id(goal)
    run_dir = root / resolved_run_id
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {resolved_run_id}")
    run_dir.mkdir(parents=True)

    now = _now()
    event = RunEvent(
        timestamp=now,
        status=RunStatus.QUEUED,
        message="Run dispatched. Assign the generated prompt to the owner agent.",
    )
    record = RunRecord(
        run_id=resolved_run_id,
        goal=goal,
        status=RunStatus.QUEUED,
        created_at=now,
        updated_at=now,
        run_dir=str(run_dir),
        task_packet=task_packet_to_dict(packet),
        events=(event,),
    )
    _write_run_files(record)
    _append_event(run_dir, event)
    return record


def record_checkpoint(
    run_id: str,
    message: str,
    artifacts: tuple[str, ...] = (),
    status: RunStatus = RunStatus.RUNNING,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
) -> RunRecord:
    """Append a progress event and update the run status."""
    if status not in {RunStatus.RUNNING, RunStatus.BLOCKED}:
        raise ValueError("Checkpoints may only set running or blocked status")
    return _append_progress(run_id, message, artifacts, status, runs_dir)


def complete_run(
    run_id: str,
    summary: str,
    artifacts: tuple[str, ...] = (),
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
) -> RunRecord:
    """Mark a run completed with a final summary."""
    return _append_progress(run_id, summary, artifacts, RunStatus.COMPLETED, runs_dir)


def fail_run(
    run_id: str,
    reason: str,
    artifacts: tuple[str, ...] = (),
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
) -> RunRecord:
    """Mark a run failed with a reason."""
    return _append_progress(run_id, reason, artifacts, RunStatus.FAILED, runs_dir)


def load_run(run_id: str, runs_dir: Path | str = DEFAULT_RUNS_DIR) -> RunRecord:
    """Load one run from its state file."""
    path = Path(runs_dir).resolve() / run_id / "state.json"
    with path.open() as f:
        return _record_from_dict(json.load(f))


def list_runs(runs_dir: Path | str = DEFAULT_RUNS_DIR) -> tuple[RunRecord, ...]:
    """Return all known runs, newest first."""
    root = Path(runs_dir).resolve()
    if not root.exists():
        return ()
    records = []
    for state_path in root.glob("*/state.json"):
        with state_path.open() as f:
            records.append(_record_from_dict(json.load(f)))
    return tuple(sorted(records, key=lambda record: record.created_at, reverse=True))


def latest_run(runs_dir: Path | str = DEFAULT_RUNS_DIR) -> RunRecord | None:
    """Return the newest run, if any."""
    runs = list_runs(runs_dir)
    return runs[0] if runs else None


def default_agent_command(agent: str | None = None) -> tuple[str, ...]:
    """Return the default command used to execute one agent prompt."""
    raw_command = os.environ.get("S_IMPLY_AGENT_CMD")
    if raw_command and agent is None:
        return tuple(shlex.split(raw_command))

    selected_agent = (agent or os.environ.get("S_IMPLY_AGENT", "codex")).lower()
    if selected_agent == "claude":
        return ("claude",)
    if selected_agent == "gemini":
        return ("gemini", "--skip-trust", "--approval-mode", "yolo", "-p", "")
    if selected_agent != "codex":
        raise ValueError(f"Unsupported agent profile: {selected_agent}")

    return (
        "codex",
        "exec",
        "--cd",
        str(REPO_ROOT),
        "--sandbox",
        "workspace-write",
        "-",
    )


def launch_run(
    run_id: str,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    agent_cmd: tuple[str, ...] | None = None,
    agent: str | None = None,
    timeout_s: int | None = None,
) -> LaunchResult:
    """Launch one queued run through an agent command and capture logs."""
    record = load_run(run_id, runs_dir)
    if record.status is not RunStatus.QUEUED:
        raise ValueError(f"Run must be queued before launch: {run_id} is {record.status.value}")

    command = agent_cmd or default_agent_command(agent)
    run_dir = Path(record.run_dir)
    stdout_path = run_dir / "agent_stdout.log"
    stderr_path = run_dir / "agent_stderr.log"
    prompt = (run_dir / "agent_prompt.md").read_text()
    record_checkpoint(
        run_id,
        f"Launching agent command: {shlex.join(command)}",
        artifacts=(str(stdout_path), str(stderr_path)),
        runs_dir=runs_dir,
    )

    try:
        with stdout_path.open("w") as f_out, stderr_path.open("w") as f_err:
            result = subprocess.run(
                command,
                input=prompt,
                cwd=REPO_ROOT,
                stdout=f_out,
                stderr=f_err,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        final = fail_run(
            run_id,
            f"Agent command timed out after {timeout_s} seconds.",
            artifacts=(str(stdout_path), str(stderr_path)),
            runs_dir=runs_dir,
        )
        return LaunchResult(
            run_id=run_id,
            command=command,
            returncode=124,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            final_status=final.status,
        )

    current = load_run(run_id, runs_dir)
    if current.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.BLOCKED}:
        final_status = current.status
    elif returncode == 0:
        final_status = complete_run(
            run_id,
            "Agent command exited with code 0.",
            artifacts=(str(stdout_path), str(stderr_path)),
            runs_dir=runs_dir,
        ).status
    else:
        final_status = fail_run(
            run_id,
            f"Agent command exited with code {returncode}.",
            artifacts=(str(stdout_path), str(stderr_path)),
            runs_dir=runs_dir,
        ).status

    return LaunchResult(
        run_id=run_id,
        command=command,
        returncode=returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        final_status=final_status,
    )


def launch_queued_runs(
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    agent_cmd: tuple[str, ...] | None = None,
    agent: str | None = None,
    max_runs: int | None = None,
    timeout_s: int | None = None,
) -> tuple[LaunchResult, ...]:
    """Launch queued runs oldest first."""
    queued = [record for record in list_runs(runs_dir) if record.status is RunStatus.QUEUED]
    queued.sort(key=lambda record: record.created_at)
    if max_runs is not None:
        queued = queued[:max_runs]
    return tuple(
        launch_run(
            record.run_id,
            runs_dir,
            agent_cmd=agent_cmd,
            agent=agent,
            timeout_s=timeout_s,
        )
        for record in queued
    )


def run_multi_agent_runtime(
    goal: str,
    changed_files: tuple[str, ...] = (),
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    agent_cmd: tuple[str, ...] | None = None,
    agent: str | None = None,
    code_agents: int = 1,
    include_docs_agent: bool = True,
    timeout_s: int | None = None,
) -> MultiAgentRuntimeResult:
    """Run a gated multi-agent workflow from one command.

    The runtime intentionally uses normal run directories for each child agent.
    Coding agents run first and may run in parallel. Test and quality agents are
    gates; docs runs only after both gates pass.
    """
    if code_agents < 1:
        raise ValueError("code_agents must be at least 1")

    parent = dispatch_task(goal, changed_files=changed_files, runs_dir=runs_dir)
    run_dir = Path(parent.run_dir)
    plan_path = run_dir / "multi_agent_plan.json"
    summary_path = run_dir / "multi_agent_summary.json"
    specs = _multi_agent_specs(goal, code_agents, include_docs_agent)
    plan_path.write_text(json.dumps(_runtime_plan_payload(parent, specs), indent=2) + "\n")
    record_checkpoint(
        parent.run_id,
        "Multi-agent runtime started.",
        artifacts=(str(plan_path),),
        runs_dir=runs_dir,
    )

    child_ids: list[str] = []
    phase_results: list[LaunchResult] = []
    code_specs = tuple(spec for spec in specs if spec.phase == "code")
    gate_specs = tuple(spec for spec in specs if spec.phase == "gate")
    docs_specs = tuple(spec for spec in specs if spec.phase == "docs")

    with ThreadPoolExecutor(max_workers=len(code_specs)) as executor:
        future_to_spec = {
            executor.submit(
                _dispatch_and_launch_child,
                parent,
                spec,
                runs_dir,
                agent_cmd,
                agent,
                timeout_s,
            ): spec
            for spec in code_specs
        }
        for future in as_completed(future_to_spec):
            child_id, result = future.result()
            child_ids.append(child_id)
            phase_results.append(result)

    if not _all_completed(phase_results):
        final = _finish_multi_agent_parent(
            parent.run_id,
            phase_results,
            child_ids,
            summary_path,
            "Stopped: at least one coding agent failed or blocked.",
            RunStatus.FAILED,
            runs_dir,
        )
        return MultiAgentRuntimeResult(
            run_id=parent.run_id,
            child_run_ids=tuple(child_ids),
            phase_results=tuple(phase_results),
            final_status=final.status,
            summary_path=str(summary_path),
        )

    for spec in gate_specs:
        child_id, result = _dispatch_and_launch_child(
            parent,
            spec,
            runs_dir,
            agent_cmd,
            agent,
            timeout_s,
        )
        child_ids.append(child_id)
        phase_results.append(result)
        if result.final_status is not RunStatus.COMPLETED:
            final = _finish_multi_agent_parent(
                parent.run_id,
                phase_results,
                child_ids,
                summary_path,
                f"Stopped: gate agent {spec.role} ended {result.final_status.value}.",
                RunStatus.FAILED,
                runs_dir,
            )
            return MultiAgentRuntimeResult(
                run_id=parent.run_id,
                child_run_ids=tuple(child_ids),
                phase_results=tuple(phase_results),
                final_status=final.status,
                summary_path=str(summary_path),
            )

    for spec in docs_specs:
        child_id, result = _dispatch_and_launch_child(
            parent,
            spec,
            runs_dir,
            agent_cmd,
            agent,
            timeout_s,
        )
        child_ids.append(child_id)
        phase_results.append(result)
        if result.final_status is not RunStatus.COMPLETED:
            final = _finish_multi_agent_parent(
                parent.run_id,
                phase_results,
                child_ids,
                summary_path,
                f"Stopped: docs agent ended {result.final_status.value}.",
                RunStatus.FAILED,
                runs_dir,
            )
            return MultiAgentRuntimeResult(
                run_id=parent.run_id,
                child_run_ids=tuple(child_ids),
                phase_results=tuple(phase_results),
                final_status=final.status,
                summary_path=str(summary_path),
            )

    final = _finish_multi_agent_parent(
        parent.run_id,
        phase_results,
        child_ids,
        summary_path,
        "Multi-agent runtime completed all phases.",
        RunStatus.COMPLETED,
        runs_dir,
    )
    return MultiAgentRuntimeResult(
        run_id=parent.run_id,
        child_run_ids=tuple(child_ids),
        phase_results=tuple(phase_results),
        final_status=final.status,
        summary_path=str(summary_path),
    )


def task_packet_to_dict(packet: TaskPacket) -> dict[str, Any]:
    """Serialize a TaskPacket using enum values instead of enum repr strings."""
    payload = asdict(packet)
    payload["owner_agent"] = packet.owner_agent.value
    payload["task_type"] = packet.task_type.value
    return payload


def format_run_status(record: RunRecord) -> str:
    """Render a compact human-readable run status."""
    packet = record.task_packet
    lines = [
        f"Run: {record.run_id}",
        f"Status: {record.status.value}",
        f"Owner: {packet['owner_agent']}",
        f"Task type: {packet['task_type']}",
        f"Goal: {record.goal}",
        f"Run dir: {record.run_dir}",
        "Validation plan:",
    ]
    lines.extend(f"- {step}" for step in packet["validation_plan"])
    if record.events:
        latest = record.events[-1]
        lines.extend(
            [
                "Latest event:",
                f"- {latest.timestamp} [{latest.status.value}] {latest.message}",
            ]
        )
    return "\n".join(lines)


def build_agent_prompt(record: RunRecord) -> str:
    """Build the handoff prompt that should be pasted into the assigned agent."""
    packet = record.task_packet
    packet_json = json.dumps(packet, indent=2)
    return f"""# S-Imply Orchestration Assignment

Run ID: {record.run_id}
Status: {record.status.value}
Goal: {record.goal}

You own this task as `{packet["owner_agent"]}`.

## Task Packet

```json
{packet_json}
```

## Operating Rules

- Work primarily within `files_owned`; explain any necessary expansion.
- Follow every listed constraint.
- Before substantial execution, create or update artifacts listed in `expected_artifacts`.
- Run the validation plan or record why a step could not be run.
- When available, use the repo MCP server
  `python -m src.orchestration.mcp_server` for ATPG, coverage, or circuit
  simulation checks instead of ad hoc scripts.
- Record progress with:
  `python -m src.orchestration.cli checkpoint {record.run_id} "message"`
- Finish with:
  `python -m src.orchestration.cli complete {record.run_id} "summary"`
- If blocked, record:
  `python -m src.orchestration.cli checkpoint {record.run_id} "blocker" --status blocked`
"""


def _multi_agent_specs(
    goal: str,
    code_agents: int,
    include_docs_agent: bool,
) -> tuple[RuntimeAgentSpec, ...]:
    specs = [
        RuntimeAgentSpec(
            role=f"coding_agent_{index}",
            phase="code",
            goal=(
                f"Implement coding slice {index} for: {goal}. "
                "Make repo-native code changes only within the needed scope, "
                "record artifacts, and stop when ready for validation."
            ),
        )
        for index in range(1, code_agents + 1)
    ]
    specs.extend(
        [
            RuntimeAgentSpec(
                role="test_coverage_gate",
                phase="gate",
                goal=(
                    f"Run tests and analyze coverage for: {goal}. "
                    "Provide concrete pass/fail feedback for the coding agents, "
                    "including commands, failures, coverage gaps, and artifacts."
                ),
            ),
            RuntimeAgentSpec(
                role="quality_review_gate",
                phase="gate",
                goal=(
                    f"Review code quality for: {goal}. "
                    "Check diffs, maintainability, repo rules, unsupported claims, "
                    "and whether the test artifacts justify passing the gate."
                ),
            ),
        ]
    )
    if include_docs_agent:
        specs.append(
            RuntimeAgentSpec(
                role="docs_results_agent",
                phase="docs",
                goal=(
                    f"Update documentation for validated changes from: {goal}. "
                    "Keep docs aligned with runtime behavior and record dated "
                    "experiment/result notes when results changed."
                ),
            )
        )
    return tuple(specs)


def _runtime_plan_payload(
    parent: RunRecord,
    specs: tuple[RuntimeAgentSpec, ...],
) -> dict[str, Any]:
    return {
        "parent_run_id": parent.run_id,
        "goal": parent.goal,
        "communication": "shared worktree plus run artifacts under runs/orchestration",
        "gate_policy": (
            "coding agents must complete; test_coverage_gate and quality_review_gate "
            "must complete before docs_results_agent runs"
        ),
        "agents": [asdict(spec) for spec in specs],
    }


def _dispatch_and_launch_child(
    parent: RunRecord,
    spec: RuntimeAgentSpec,
    runs_dir: Path | str,
    agent_cmd: tuple[str, ...] | None,
    agent: str | None,
    timeout_s: int | None,
) -> tuple[str, LaunchResult]:
    child_run_id = f"{parent.run_id}-{_safe_slug(spec.role)}"
    child = dispatch_task(spec.goal, runs_dir=runs_dir, run_id=child_run_id)
    _append_runtime_context(child, parent, spec)
    result = launch_run(
        child.run_id,
        runs_dir=runs_dir,
        agent_cmd=agent_cmd,
        agent=agent,
        timeout_s=timeout_s,
    )
    return child.run_id, result


def _append_runtime_context(
    child: RunRecord,
    parent: RunRecord,
    spec: RuntimeAgentSpec,
) -> None:
    prompt_path = Path(child.run_dir) / "agent_prompt.md"
    prompt_path.write_text(
        prompt_path.read_text()
        + f"""

## Multi-Agent Runtime Context

Parent run ID: {parent.run_id}
Parent goal: {parent.goal}
Runtime role: {spec.role}
Runtime phase: {spec.phase}

- Communicate through checkpoints and artifacts in the parent/child run directories.
- Inspect sibling artifacts under `{Path(parent.run_dir).parent}` when useful.
- Do not revert unrelated worktree changes or edits made by sibling agents.
- If this is a gate role, provide a clear pass/fail decision and actionable feedback.
- If this is the docs role, only document validated behavior and cite artifacts.
"""
    )


def _finish_multi_agent_parent(
    parent_run_id: str,
    phase_results: list[LaunchResult],
    child_ids: list[str],
    summary_path: Path,
    message: str,
    status: RunStatus,
    runs_dir: Path | str,
) -> RunRecord:
    summary = {
        "parent_run_id": parent_run_id,
        "status": status.value,
        "message": message,
        "child_run_ids": child_ids,
        "phase_results": [
            {
                "run_id": result.run_id,
                "status": result.final_status.value,
                "returncode": result.returncode,
                "stdout": result.stdout_path,
                "stderr": result.stderr_path,
            }
            for result in phase_results
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    artifacts = (str(summary_path),)
    if status is RunStatus.COMPLETED:
        return complete_run(parent_run_id, message, artifacts=artifacts, runs_dir=runs_dir)
    return fail_run(parent_run_id, message, artifacts=artifacts, runs_dir=runs_dir)


def _all_completed(results: list[LaunchResult]) -> bool:
    return all(result.final_status is RunStatus.COMPLETED for result in results)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "agent"


def _append_progress(
    run_id: str,
    message: str,
    artifacts: tuple[str, ...],
    status: RunStatus,
    runs_dir: Path | str,
) -> RunRecord:
    record = load_run(run_id, runs_dir)
    now = _now()
    event = RunEvent(timestamp=now, status=status, message=message, artifacts=artifacts)
    updated = RunRecord(
        run_id=record.run_id,
        goal=record.goal,
        status=status,
        created_at=record.created_at,
        updated_at=now,
        run_dir=record.run_dir,
        task_packet=record.task_packet,
        events=record.events + (event,),
    )
    _write_run_files(updated)
    _append_event(Path(updated.run_dir), event)
    return updated


def _write_run_files(record: RunRecord) -> None:
    run_dir = Path(record.run_dir)
    state_path = run_dir / "state.json"
    packet_path = run_dir / "task_packet.json"
    prompt_path = run_dir / "agent_prompt.md"
    manifest_path = run_dir / "run_manifest.json"

    state_path.write_text(json.dumps(_record_to_dict(record), indent=2) + "\n")
    packet_path.write_text(json.dumps(record.task_packet, indent=2) + "\n")
    prompt_text = build_agent_prompt(record)
    if prompt_path.exists():
        existing = prompt_path.read_text()
        marker = "\n## Multi-Agent Runtime Context\n"
        if marker in existing:
            prompt_text += existing[existing.index(marker) :]
    prompt_path.write_text(prompt_text)
    if record.task_packet["run_manifest_required"] and not manifest_path.exists():
        manifest_path.write_text(_default_manifest(record))


def _append_event(run_dir: Path, event: RunEvent) -> None:
    with (run_dir / "events.jsonl").open("a") as f:
        f.write(json.dumps(_event_to_dict(event)) + "\n")


def _default_manifest(record: RunRecord) -> str:
    payload = {
        "run_id": record.run_id,
        "goal": record.goal,
        "created_at": record.created_at,
        "command": "",
        "inputs": [],
        "outputs": [],
        "hardware": "",
        "notes": "Fill this before GPU, training, benchmark, or evaluation execution.",
    }
    return json.dumps(payload, indent=2) + "\n"


def _record_to_dict(record: RunRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "goal": record.goal,
        "status": record.status.value,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "run_dir": record.run_dir,
        "task_packet": record.task_packet,
        "events": [_event_to_dict(event) for event in record.events],
    }


def _record_from_dict(payload: dict[str, Any]) -> RunRecord:
    return RunRecord(
        run_id=payload["run_id"],
        goal=payload["goal"],
        status=RunStatus(payload["status"]),
        created_at=payload["created_at"],
        updated_at=payload["updated_at"],
        run_dir=payload["run_dir"],
        task_packet=payload["task_packet"],
        events=tuple(_event_from_dict(event) for event in payload.get("events", [])),
    )


def _event_to_dict(event: RunEvent) -> dict[str, Any]:
    return {
        "timestamp": event.timestamp,
        "status": event.status.value,
        "message": event.message,
        "artifacts": list(event.artifacts),
    }


def _event_from_dict(payload: dict[str, Any]) -> RunEvent:
    return RunEvent(
        timestamp=payload["timestamp"],
        status=RunStatus(payload["status"]),
        message=payload["message"],
        artifacts=tuple(payload.get("artifacts", [])),
    )


def _make_run_id(goal: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")[:48]
    return f"{stamp}-{slug or 'task'}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
