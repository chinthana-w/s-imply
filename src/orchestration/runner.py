"""Persistent runner state for the multi-agent orchestration workflow."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
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


def default_agent_command() -> tuple[str, ...]:
    """Return the default command used to execute one agent prompt."""
    raw_command = os.environ.get("S_IMPLY_AGENT_CMD")
    if raw_command:
        return tuple(shlex.split(raw_command))

    agent = os.environ.get("S_IMPLY_AGENT", "codex").lower()
    if agent == "claude":
        return ("claude",)
    elif agent == "gemini":
        return ("gemini", "--skip-trust", "-y", "-p", "")

    return (
        "codex",
        "exec",
        "--cd",
        str(REPO_ROOT),
        "--sandbox",
        "workspace-write",
        "--ask-for-approval",
        "never",
        "-",
    )


def launch_run(
    run_id: str,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    agent_cmd: tuple[str, ...] | None = None,
    timeout_s: int | None = None,
) -> LaunchResult:
    """Launch one queued run through an agent command and capture logs."""
    record = load_run(run_id, runs_dir)
    if record.status is not RunStatus.QUEUED:
        raise ValueError(f"Run must be queued before launch: {run_id} is {record.status.value}")

    command = agent_cmd or default_agent_command()
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
        result = subprocess.run(
            command,
            input=prompt,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        stdout_path.write_text(result.stdout)
        stderr_path.write_text(result.stderr)
        returncode = result.returncode
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stdout_path.write_text(out)
        stderr_path.write_text(err)
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
    max_runs: int | None = None,
    timeout_s: int | None = None,
) -> tuple[LaunchResult, ...]:
    """Launch queued runs oldest first."""
    queued = [record for record in list_runs(runs_dir) if record.status is RunStatus.QUEUED]
    queued.sort(key=lambda record: record.created_at)
    if max_runs is not None:
        queued = queued[:max_runs]
    return tuple(
        launch_run(record.run_id, runs_dir, agent_cmd=agent_cmd, timeout_s=timeout_s)
        for record in queued
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
- Record progress with:
  `python -m src.orchestration.cli checkpoint {record.run_id} "message"`
- Finish with:
  `python -m src.orchestration.cli complete {record.run_id} "summary"`
- If blocked, record:
  `python -m src.orchestration.cli checkpoint {record.run_id} "blocker" --status blocked`
"""


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
    prompt_path.write_text(build_agent_prompt(record))
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
