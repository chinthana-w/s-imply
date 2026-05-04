"""Command-line entrypoint for the coordinator scaffold."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from src.orchestration.coordinator import create_task_packet
from src.orchestration.runner import (
    DEFAULT_RUNS_DIR,
    RunStatus,
    complete_run,
    dispatch_task,
    fail_run,
    format_run_status,
    latest_run,
    launch_queued_runs,
    launch_run,
    list_runs,
    load_run,
    record_checkpoint,
    task_packet_to_dict,
)
from src.orchestration.tools import (
    NotionDocumentationTarget,
    check_theory_doc_sync,
    summarize_benchmark_artifact,
    validate_notion_documentation_target,
)

DEFAULT_EXPERIMENT_LOG_FORMAT = (
    "append dated experiment log entries with commands, artifacts, results, and next steps"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="s-imply multi-agent orchestration helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    route = sub.add_parser("route", help="Create a specialist task packet")
    route.add_argument("goal")
    route.add_argument("--changed-file", action="append", default=[])

    dispatch = sub.add_parser("dispatch", help="Create a persistent orchestration run")
    dispatch.add_argument("goal")
    dispatch.add_argument("--changed-file", action="append", default=[])
    dispatch.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    status = sub.add_parser("status", help="Show run status, or the latest run when omitted")
    status.add_argument("run_id", nargs="?")
    status.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    list_cmd = sub.add_parser("list", help="List orchestration runs")
    list_cmd.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    prompt = sub.add_parser("prompt", help="Print the agent handoff prompt for a run")
    prompt.add_argument("run_id", nargs="?")
    prompt.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    checkpoint = sub.add_parser("checkpoint", help="Record progress on a run")
    checkpoint.add_argument("run_id")
    checkpoint.add_argument("message")
    checkpoint.add_argument("--artifact", action="append", default=[])
    checkpoint.add_argument(
        "--status",
        choices=[RunStatus.RUNNING.value, RunStatus.BLOCKED.value],
        default=RunStatus.RUNNING.value,
    )
    checkpoint.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    complete = sub.add_parser("complete", help="Mark a run completed")
    complete.add_argument("run_id")
    complete.add_argument("summary")
    complete.add_argument("--artifact", action="append", default=[])
    complete.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    fail = sub.add_parser("fail", help="Mark a run failed")
    fail.add_argument("run_id")
    fail.add_argument("reason")
    fail.add_argument("--artifact", action="append", default=[])
    fail.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    launch = sub.add_parser("launch", help="Launch one queued run through an agent command")
    launch.add_argument("run_id")
    launch.add_argument("--agent-cmd", default="")
    launch.add_argument("--timeout-s", type=int)
    launch.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    worker = sub.add_parser("worker", help="Launch queued runs oldest first")
    worker.add_argument("--agent-cmd", default="")
    worker.add_argument("--max-runs", type=int)
    worker.add_argument("--timeout-s", type=int)
    worker.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    bench = sub.add_parser("summarize-benchmark", help="Summarize a benchmark CSV/JSON artifact")
    bench.add_argument("path")

    theory = sub.add_parser("check-theory-doc-sync", help="Check paper sync for theory changes")
    theory.add_argument("changed_files", nargs="+")

    notion = sub.add_parser("check-notion-target", help="Check Notion docs target configuration")
    notion.add_argument("--target", default="")
    notion.add_argument("--format", default="", dest="content_format")
    notion.add_argument("--audience", default="")
    notion.add_argument("--sync-style", default="notion_canonical")
    notion.add_argument(
        "--experiment-log-format",
        default=DEFAULT_EXPERIMENT_LOG_FORMAT,
    )
    notion.add_argument("--owner", default="")

    args = parser.parse_args()
    if args.command == "route":
        packet = create_task_packet(args.goal, changed_files=tuple(args.changed_file))
        print(json.dumps(task_packet_to_dict(packet), indent=2))
    elif args.command == "dispatch":
        record = dispatch_task(
            args.goal,
            changed_files=tuple(args.changed_file),
            runs_dir=Path(args.runs_dir),
        )
        print(format_run_status(record))
        print(f"\nAgent prompt: {Path(record.run_dir) / 'agent_prompt.md'}")
    elif args.command == "status":
        record = _load_selected_run(args.run_id, args.runs_dir)
        print(format_run_status(record))
    elif args.command == "list":
        records = list_runs(Path(args.runs_dir))
        if not records:
            print("No orchestration runs found.")
        for record in records:
            print(
                f"{record.run_id}\t{record.status.value}\t"
                f"{record.task_packet['owner_agent']}\t{record.goal}"
            )
    elif args.command == "prompt":
        record = _load_selected_run(args.run_id, args.runs_dir)
        print((Path(record.run_dir) / "agent_prompt.md").read_text(), end="")
    elif args.command == "checkpoint":
        record = record_checkpoint(
            args.run_id,
            args.message,
            artifacts=tuple(args.artifact),
            status=RunStatus(args.status),
            runs_dir=Path(args.runs_dir),
        )
        print(format_run_status(record))
    elif args.command == "complete":
        record = complete_run(
            args.run_id,
            args.summary,
            artifacts=tuple(args.artifact),
            runs_dir=Path(args.runs_dir),
        )
        print(format_run_status(record))
    elif args.command == "fail":
        record = fail_run(
            args.run_id,
            args.reason,
            artifacts=tuple(args.artifact),
            runs_dir=Path(args.runs_dir),
        )
        print(format_run_status(record))
    elif args.command == "launch":
        result = launch_run(
            args.run_id,
            runs_dir=Path(args.runs_dir),
            agent_cmd=_parse_agent_cmd(args.agent_cmd),
            timeout_s=args.timeout_s,
        )
        print(_format_launch_result(result))
    elif args.command == "worker":
        results = launch_queued_runs(
            runs_dir=Path(args.runs_dir),
            agent_cmd=_parse_agent_cmd(args.agent_cmd),
            max_runs=args.max_runs,
            timeout_s=args.timeout_s,
        )
        if not results:
            print("No queued orchestration runs found.")
        for result in results:
            print(_format_launch_result(result))
    elif args.command == "summarize-benchmark":
        print(json.dumps(summarize_benchmark_artifact(args.path), indent=2, default=str))
    elif args.command == "check-theory-doc-sync":
        print(json.dumps(check_theory_doc_sync(tuple(args.changed_files)), indent=2))
    elif args.command == "check-notion-target":
        target = NotionDocumentationTarget(
            target=args.target,
            content_format=args.content_format,
            audience=args.audience,
            sync_style=args.sync_style,
            experiment_log_format=args.experiment_log_format,
            owner=args.owner,
        )
        print(json.dumps(validate_notion_documentation_target(target), indent=2))


def _load_selected_run(run_id: str | None, runs_dir: str):
    if run_id:
        return load_run(run_id, Path(runs_dir))
    record = latest_run(Path(runs_dir))
    if record is None:
        raise SystemExit("No orchestration runs found.")
    return record


def _parse_agent_cmd(raw_command: str):
    return tuple(shlex.split(raw_command)) if raw_command else None


def _format_launch_result(result) -> str:
    return (
        f"Run: {result.run_id}\n"
        f"Status: {result.final_status.value}\n"
        f"Return code: {result.returncode}\n"
        f"Command: {shlex.join(result.command)}\n"
        f"Stdout: {result.stdout_path}\n"
        f"Stderr: {result.stderr_path}"
    )


if __name__ == "__main__":
    main()
