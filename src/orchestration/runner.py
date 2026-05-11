"""Persistent runner state for the multi-agent orchestration workflow."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
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
    SLEEPING = "sleeping"
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


@dataclass(frozen=True)
class TmuxMonitorResult:
    session_name: str
    run_ids: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]
    attach_command: tuple[str, ...]
    dry_run: bool = False


TOKEN_LIMIT_PATTERNS = (
    "token limit",
    "tokens limit",
    "usage limit",
    "rate limit",
    "limit reached",
    "quota exceeded",
    "context length",
    "context window",
    "maximum context",
    "too many tokens",
)


def dispatch_task(
    goal: str,
    changed_files: tuple[str, ...] = (),
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    run_id: str | None = None,
    phase_override: str | None = None,
) -> RunRecord:
    """Create a persistent run directory and specialist handoff packet."""
    packet = create_task_packet(
        goal,
        changed_files=changed_files,
        phase_override=phase_override,
    )
    root = Path(runs_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    resolved_run_id = run_id or _make_run_id(goal)
    run_dir = root / resolved_run_id
    if run_dir.exists():
        raise FileExistsError(
            f"Run already exists: {resolved_run_id}"
        )
    run_dir.mkdir(parents=True)

    now = _now()
    event = RunEvent(
        timestamp=now,
        status=RunStatus.QUEUED,
        message="Run dispatched.",
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
    if status not in {RunStatus.RUNNING, RunStatus.SLEEPING, RunStatus.BLOCKED}:
        raise ValueError("Checkpoints may only set running, sleeping, or blocked status")
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
    codex_auto_resume: bool = False,
    codex_resume_delay_s: int = 18_000,
    codex_max_resumes: int = 1,
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

    returncode = _run_agent_command(command, prompt, stdout_path, stderr_path, timeout_s, "w")
    resume_attempts = 0
    while (
        codex_auto_resume
        and _is_codex_command(command)
        and returncode != 0
        and resume_attempts < codex_max_resumes
        and _logs_indicate_token_limit(stdout_path, stderr_path)
    ):
        resume_attempts += 1
        record_checkpoint(
            run_id,
            (
                "Codex token or usage limit reached; sleeping for "
                f"{codex_resume_delay_s} seconds before automatic resume "
                f"({resume_attempts}/{codex_max_resumes})."
            ),
            artifacts=(str(stdout_path), str(stderr_path)),
            status=RunStatus.SLEEPING,
            runs_dir=runs_dir,
        )
        time.sleep(codex_resume_delay_s)
        record_checkpoint(
            run_id,
            f"Resuming Codex after token-limit sleep ({resume_attempts}/{codex_max_resumes}).",
            artifacts=(str(stdout_path), str(stderr_path)),
            status=RunStatus.RUNNING,
            runs_dir=runs_dir,
        )
        returncode = _run_agent_command(command, prompt, stdout_path, stderr_path, timeout_s, "a")

    if returncode == 124:
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
    codex_auto_resume: bool = False,
    codex_resume_delay_s: int = 18_000,
    codex_max_resumes: int = 1,
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
            codex_auto_resume=codex_auto_resume,
            codex_resume_delay_s=codex_resume_delay_s,
            codex_max_resumes=codex_max_resumes,
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
    codex_auto_resume: bool = False,
    codex_resume_delay_s: int = 18_000,
    codex_max_resumes: int = 1,
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
                codex_auto_resume,
                codex_resume_delay_s,
                codex_max_resumes,
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

    # -- Collect summaries from coding phase for downstream --
    code_summaries = _collect_sibling_summaries(
        child_ids, phase_results, runs_dir
    )

    # -- Gate phase: run test + quality gates in PARALLEL --
    if gate_specs:
        workers = max(1, len(gate_specs))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            gate_futures = {
                ex.submit(
                    _dispatch_and_launch_child,
                    parent,
                    spec,
                    runs_dir,
                    agent_cmd,
                    agent,
                    timeout_s,
                    codex_auto_resume,
                    codex_resume_delay_s,
                    codex_max_resumes,
                    code_summaries,
                ): spec
                for spec in gate_specs
            }
            for future in as_completed(gate_futures):
                child_id, result = future.result()
                child_ids.append(child_id)
                phase_results.append(result)

    gate_results = phase_results[len(code_specs):]
    if gate_results and not _all_completed(gate_results):
        failed_gates = [
            r for r in gate_results
            if r.final_status is not RunStatus.COMPLETED
        ]
        gate_msg = ", ".join(r.run_id for r in failed_gates)
        final = _finish_multi_agent_parent(
            parent.run_id,
            phase_results,
            child_ids,
            summary_path,
            f"Stopped: gate(s) failed: {gate_msg}.",
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

    # -- Docs phase: forward all prior summaries --
    all_summaries = code_summaries + _collect_sibling_summaries(
        child_ids[len(code_specs):],
        gate_results,
        runs_dir,
    )
    for spec in docs_specs:
        child_id, result = _dispatch_and_launch_child(
            parent,
            spec,
            runs_dir,
            agent_cmd,
            agent,
            timeout_s,
            codex_auto_resume,
            codex_resume_delay_s,
            codex_max_resumes,
            all_summaries,
        )
        child_ids.append(child_id)
        phase_results.append(result)
        if result.final_status is not RunStatus.COMPLETED:
            final = _finish_multi_agent_parent(
                parent.run_id,
                phase_results,
                child_ids,
                summary_path,
                f"Stopped: docs agent ended "
                f"{result.final_status.value}.",
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


def launch_tmux_monitor(
    run_id: str,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    session_name: str | None = None,
    attach: bool = False,
    dry_run: bool = False,
) -> TmuxMonitorResult:
    """Create a tmux session tailing events for a parent run and its child runs."""
    resolved_run_id = _resolve_monitor_run_id(run_id, runs_dir)
    run_ids = _monitor_run_ids(resolved_run_id, runs_dir)
    if not run_ids:
        raise ValueError(f"No run event files found for monitor target: {run_id}")

    resolved_session = session_name or f"s-imply-{_safe_slug(resolved_run_id)[:64]}"
    commands = _tmux_monitor_commands(resolved_session, run_ids, runs_dir)
    attach_command = ("tmux", "attach-session", "-t", resolved_session)

    if not dry_run:
        if shutil.which("tmux") is None:
            raise RuntimeError("tmux is not installed or not available on PATH")
        for command in commands:
            if command[:3] == ("tmux", "kill-session", "-t"):
                subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                continue
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        if attach:
            subprocess.run(attach_command, cwd=REPO_ROOT, check=True)

    return TmuxMonitorResult(
        session_name=resolved_session,
        run_ids=run_ids,
        commands=commands,
        attach_command=attach_command,
        dry_run=dry_run,
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
    """Build a compact handoff prompt for the assigned agent."""
    p = record.task_packet
    rid = record.run_id
    files = ", ".join(p["files_owned"])
    constraints = "\n".join(f"- {c}" for c in p["constraints"])
    validation = "\n".join(f"- {s}" for s in p["validation_plan"])
    artifacts = ", ".join(p["expected_artifacts"])
    cli = "python -m src.orchestration.cli"
    return f"""# Assignment: {p["owner_agent"]}

Run: {rid}
Goal: {record.goal}
Files: {files}

## Constraints
{constraints}

## Validation
{validation}

## Artifacts
{artifacts}

## Commands
- Progress: `{cli} checkpoint {rid} "msg"`
- Done: `{cli} complete {rid} "summary"`
- Blocked: `{cli} checkpoint {rid} "blocker" --status blocked`
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
                f"Do code slice {index}: {goal}. "
                "Change repo code in scope. Make artifact. Stop when ready to validate."
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
                    f"Test and check coverage: {goal}. "
                    "Give pass/fail to code agent. Include command, failure, gap, artifact."
                ),
            ),
            RuntimeAgentSpec(
                role="quality_review_gate",
                phase="gate",
                goal=(
                    f"Review code: {goal}. "
                    "Check diff, repo rule, unsupported claim. Do tests justify passing?"
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
                    f"Update doc for validated change: {goal}. "
                    "Keep doc in sync with runtime. Note dated experiment result if changed."
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
        "communication": "share worktree, run artifact in runs/orchestration",
        "gate_policy": (
            "code agent must finish; test/quality gate "
            "must pass before doc agent run"
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
    codex_auto_resume: bool = False,
    codex_resume_delay_s: int = 18_000,
    codex_max_resumes: int = 1,
    sibling_summaries: tuple[tuple[str, str], ...] = (),
) -> tuple[str, LaunchResult]:
    child_run_id = (
        f"{parent.run_id}-{_safe_slug(spec.role)}"
    )
    child = dispatch_task(
        spec.goal,
        runs_dir=runs_dir,
        run_id=child_run_id,
        phase_override=spec.phase,
    )
    _append_runtime_context(
        child, parent, spec, sibling_summaries
    )
    result = launch_run(
        child.run_id,
        runs_dir=runs_dir,
        agent_cmd=agent_cmd,
        agent=agent,
        timeout_s=timeout_s,
        codex_auto_resume=codex_auto_resume,
        codex_resume_delay_s=codex_resume_delay_s,
        codex_max_resumes=codex_max_resumes,
    )
    return child.run_id, result


def _run_agent_command(
    command: tuple[str, ...],
    prompt: str,
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: int | None,
    mode: str,
) -> int:
    try:
        with stdout_path.open(mode) as f_out, stderr_path.open(mode) as f_err:
            if mode == "a":
                f_out.write("\n\n=== automatic resume attempt ===\n")
                f_err.write("\n\n=== automatic resume attempt ===\n")
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
        return result.returncode
    except subprocess.TimeoutExpired:
        return 124


def _is_codex_command(command: tuple[str, ...]) -> bool:
    return bool(command) and Path(command[0]).name == "codex"


def _logs_indicate_token_limit(stdout_path: Path, stderr_path: Path) -> bool:
    text = "\n".join(_read_tail(path).lower() for path in (stdout_path, stderr_path))
    return any(pattern in text for pattern in TOKEN_LIMIT_PATTERNS)


def _read_tail(path: Path, limit: int = 16_384) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def _append_runtime_context(
    child: RunRecord,
    parent: RunRecord,
    spec: RuntimeAgentSpec,
    sibling_summaries: tuple[tuple[str, str], ...] = (),
) -> None:
    prompt_path = Path(child.run_dir) / "agent_prompt.md"
    sibling_block = ""
    if sibling_summaries:
        lines = ["\n## Sibling Results\n"]
        for sib_id, sib_detail in sibling_summaries:
            lines.append(f"### {sib_id}")
            lines.append(sib_detail)
            lines.append("")
        sibling_block = "\n".join(lines)
    prompt_path.write_text(
        prompt_path.read_text()
        + f"""
## Runtime Context

Parent: {parent.run_id}
Role: {spec.role} | Phase: {spec.phase}
Goal: {parent.goal}

- Talk via checkpoint/artifact in run dir.
- No undo sibling work.
- Gate role: give clear pass/fail + feedback.
- Docs role: only validated facts. Cite artifacts.
{sibling_block}"""
    )


def _collect_sibling_summaries(
    child_ids: list[str],
    results: list[LaunchResult],
    runs_dir: Path | str,
) -> tuple[tuple[str, str], ...]:
    """Build rich summaries from finished siblings.

    Each summary includes status, all checkpoint messages,
    artifact paths, and stdout tail so downstream agents
    have full context.
    """
    summaries: list[tuple[str, str]] = []
    for cid, res in zip(child_ids, results):
        parts: list[str] = []
        status = res.final_status.value
        parts.append(f"**{status}**")
        try:
            rec = load_run(cid, runs_dir)
            # Include all checkpoint messages (skip queued/launch)
            for ev in rec.events:
                if ev.status in (
                    RunStatus.QUEUED,
                ):
                    continue
                parts.append(
                    f"  - [{ev.status.value}] {ev.message}"
                )
                if ev.artifacts:
                    for art in ev.artifacts:
                        parts.append(f"    artifact: {art}")
        except Exception:
            pass
        # Append stdout tail for concrete output
        stdout = Path(res.stdout_path)
        tail = _read_tail(stdout, limit=2048).strip()
        if tail:
            # Keep only last 20 lines max
            tail_lines = tail.splitlines()[-20:]
            parts.append("  stdout (tail):")
            for line in tail_lines:
                parts.append(f"    {line}")
        summaries.append((cid, "\n".join(parts)))
    return tuple(summaries)


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


def _resolve_monitor_run_id(run_id: str, runs_dir: Path | str) -> str:
    root = Path(runs_dir).resolve()
    if (root / run_id / "state.json").exists():
        return run_id

    matches = sorted(
        record.run_id for record in list_runs(runs_dir) if record.run_id.startswith(run_id)
    )
    if not matches:
        raise ValueError(f"No orchestration run matches: {run_id}")
    if len(matches) == 1:
        return matches[0]

    shortest = min(matches, key=len)
    if all(match == shortest or match.startswith(f"{shortest}-") for match in matches):
        return shortest

    preview = ", ".join(matches[:8])
    if len(matches) > 8:
        preview += f", ... +{len(matches) - 8} more"
    raise ValueError(f"Run id prefix is ambiguous: {run_id}. Matches: {preview}")


def _monitor_run_ids(run_id: str, runs_dir: Path | str) -> tuple[str, ...]:
    parent = load_run(run_id, runs_dir)
    run_ids = [parent.run_id]
    summary_path = Path(parent.run_dir) / "multi_agent_summary.json"
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)
        run_ids.extend(str(child_id) for child_id in summary.get("child_run_ids", []))
    else:
        prefix = f"{parent.run_id}-"
        child_ids = [
            record.run_id
            for record in list_runs(runs_dir)
            if record.run_id.startswith(prefix)
        ]
        child_ids.sort()
        run_ids.extend(child_ids)

    seen: set[str] = set()
    existing = []
    for candidate in run_ids:
        if candidate in seen:
            continue
        seen.add(candidate)
        event_path = Path(runs_dir).resolve() / candidate / "events.jsonl"
        if event_path.exists():
            existing.append(candidate)
    return tuple(existing)


def _tmux_monitor_commands(
    session_name: str,
    run_ids: tuple[str, ...],
    runs_dir: Path | str,
) -> tuple[tuple[str, ...], ...]:
    root = Path(runs_dir).resolve()
    first_run_id, *rest = run_ids
    commands: list[tuple[str, ...]] = [
        ("tmux", "kill-session", "-t", session_name),
        (
            "tmux",
            "new-session",
            "-d",
            "-s",
            session_name,
            "-n",
            "events",
            _tail_events_command(first_run_id, root / first_run_id / "events.jsonl"),
        ),
        ("tmux", "select-pane", "-t", f"{session_name}:events.0", "-T", first_run_id),
    ]
    for index, child_run_id in enumerate(rest, start=1):
        commands.extend(
            [
                (
                    "tmux",
                    "split-window",
                    "-t",
                    f"{session_name}:events",
                    _tail_events_command(child_run_id, root / child_run_id / "events.jsonl"),
                ),
                (
                    "tmux",
                    "select-pane",
                    "-t",
                    f"{session_name}:events.{index}",
                    "-T",
                    child_run_id,
                ),
                ("tmux", "select-layout", "-t", f"{session_name}:events", "tiled"),
            ]
        )
    return tuple(commands)


def _tail_events_command(run_id: str, path: Path) -> str:
    return (
        "bash -lc "
        + shlex.quote(
            f"printf '== {run_id} ==\\n'; "
            f"touch {shlex.quote(str(path))}; "
            f"tail -n 80 -F {shlex.quote(str(path))}"
        )
    )


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
        marker = "\n## Runtime Context\n"
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
