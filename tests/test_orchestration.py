import csv
import json
import subprocess
import sys

from src.orchestration.coordinator import AgentRole, TaskType, classify_task, create_task_packet
from src.orchestration.mcp_server import handle_request
from src.orchestration.runner import (
    _SIBLING_PER_CAP,
    RunStatus,
    _collect_sibling_summaries,
    _inject_gemini_prompt,
    _is_gemini_command,
    _multi_agent_specs,
    _recovery_specs,
    _supports_auto_resume,
    complete_run,
    default_agent_command,
    dispatch_task,
    launch_queued_runs,
    launch_run,
    launch_tmux_monitor,
    list_runs,
    load_run,
    record_checkpoint,
    run_multi_agent_runtime,
)
from src.orchestration.tools import (
    NotionDocumentationTarget,
    check_theory_doc_sync,
    run_atpg,
    run_test_coverage,
    simulate_circuit,
    summarize_benchmark_artifact,
    validate_notion_documentation_target,
)


def test_classify_task_prefers_theory_for_framework_changes():
    assert classify_task("Update the theoretical framework for LRR constraints") == TaskType.THEORY


def test_create_ml_task_packet_requires_manifest():
    packet = create_task_packet("Fix GradScaler checkpoint compatibility in train.py")

    assert packet.owner_agent == AgentRole.ML_TRAINING
    assert packet.task_type == TaskType.ML
    assert packet.run_manifest_required
    assert any("checkpoint compatibility" in step for step in packet.validation_plan)


def test_create_docs_packet_includes_pending_notion_publication():
    packet = create_task_packet("Publish documentation update to Notion wiki")

    assert packet.owner_agent == AgentRole.DOCS_RESULTS
    assert packet.task_type == TaskType.DOCS
    assert packet.notion_publication.enabled
    assert packet.notion_publication.status == "configured"
    assert packet.notion_publication.sync_style == "notion_canonical"


def test_summarize_csv_benchmark(tmp_path):
    artifact = tmp_path / "bench.csv"
    with artifact.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ai_time_ms", "vanilla_time_ms", "vanilla_backtracks"],
        )
        writer.writeheader()
        writer.writerow({"ai_time_ms": "10", "vanilla_time_ms": "8", "vanilla_backtracks": "1"})
        writer.writerow({"ai_time_ms": "14", "vanilla_time_ms": "9", "vanilla_backtracks": "3"})

    summary = summarize_benchmark_artifact(str(artifact))

    assert summary["rows"] == 2
    assert summary["ai_time_ms_mean"] == 12
    assert summary["vanilla_backtracks_max"] == 3


def test_summarize_json_benchmark(tmp_path):
    artifact = tmp_path / "bench.json"
    artifact.write_text(
        json.dumps(
            {
                "total": 2,
                "succeeded": 1,
                "failed": 1,
                "coverage": 0.5,
                "per_fault": [{"time_s": 0.1}, {"time_s": 0.3}],
            }
        )
    )

    summary = summarize_benchmark_artifact(str(artifact))

    assert summary["coverage"] == 0.5
    assert summary["mean_time_s"] == 0.2


def test_theory_doc_sync_requires_paper_update():
    missing = check_theory_doc_sync(("src/atpg/reconv_podem.py",))
    present = check_theory_doc_sync(("src/atpg/reconv_podem.py", "docs/paper_draft.tex"))

    assert not missing["ok"]
    assert present["ok"]


def test_validate_notion_documentation_target_requires_core_fields():
    missing = validate_notion_documentation_target(
        NotionDocumentationTarget(target="", content_format="", audience="")
    )
    configured = validate_notion_documentation_target(
        NotionDocumentationTarget(
            target="Back Implication Prediction using Attention",
            content_format="canonical wiki page with method and results",
            audience="research engineers",
        )
    )

    assert not missing["ok"]
    assert missing["missing"] == ["target", "content_format", "audience"]
    assert configured["ok"]


def test_repo_tool_wrappers_support_atpg_coverage_and_simulation(tmp_path):
    atpg = run_atpg("data/bench/ISCAS85/c17.bench", limit_faults=2, dry_run=True)
    coverage = run_test_coverage(
        ("tests/test_orchestration.py",),
        coverage_json=str(tmp_path / "coverage.json"),
        dry_run=True,
    )
    simulation = simulate_circuit(
        "data/bench/ISCAS85/c17.bench",
        assignments={"1": 1, "2": 0, "3": 1, "6": 1, "7": 0},
    )

    assert atpg["bench"].endswith("data/bench/ISCAS85/c17.bench")
    assert atpg["limit_faults"] == 2
    assert coverage["command"][:4] == ("python", "-m", "coverage", "run")
    assert simulation["outputs"]


def test_mcp_server_lists_and_calls_repo_tools():
    listed = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    called = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "run_atpg",
                "arguments": {
                    "bench_path": "data/bench/ISCAS85/c17.bench",
                    "limit_faults": 1,
                    "dry_run": True,
                },
            },
        }
    )

    assert listed is not None
    assert {tool["name"] for tool in listed["result"]["tools"]} == {
        "run_atpg",
        "run_test_coverage",
        "simulate_circuit",
    }
    assert called is not None
    assert "data/bench/ISCAS85/c17.bench" in called["result"]["content"][0]["text"]


def test_dispatch_task_creates_persistent_run(tmp_path):
    record = dispatch_task(
        "Fix GradScaler checkpoint compatibility",
        runs_dir=tmp_path,
        run_id="run-001",
    )

    run_dir = tmp_path / "run-001"
    loaded = load_run("run-001", runs_dir=tmp_path)

    assert record.status == RunStatus.QUEUED
    assert loaded.task_packet["owner_agent"] == AgentRole.ML_TRAINING.value
    assert (run_dir / "state.json").exists()
    assert (run_dir / "task_packet.json").exists()
    assert (run_dir / "agent_prompt.md").exists()
    assert (run_dir / "run_manifest.json").exists()


def test_runner_records_checkpoints_and_completion(tmp_path):
    dispatch_task("Review ATPG regression risk", runs_dir=tmp_path, run_id="run-002")

    running = record_checkpoint(
        "run-002",
        "Inspected solver files",
        artifacts=("notes.md",),
        runs_dir=tmp_path,
    )
    completed = complete_run("run-002", "Validated focused checks", runs_dir=tmp_path)
    runs = list_runs(tmp_path)

    assert running.status == RunStatus.RUNNING
    assert running.events[-1].artifacts == ("notes.md",)
    assert completed.status == RunStatus.COMPLETED
    assert completed.events[-1].message == "Validated focused checks"
    assert runs[0].run_id == "run-002"


def test_launch_run_executes_agent_command_and_captures_logs(tmp_path):
    dispatch_task("Review ATPG regression risk", runs_dir=tmp_path, run_id="run-003")
    command = (
        sys.executable,
        "-c",
        "import sys; data=sys.stdin.read(); print('received', 'Run:' in data)",
    )

    result = launch_run("run-003", runs_dir=tmp_path, agent_cmd=command)
    loaded = load_run("run-003", runs_dir=tmp_path)

    assert result.returncode == 0
    assert result.final_status == RunStatus.COMPLETED
    assert loaded.status == RunStatus.COMPLETED
    assert "received True" in (tmp_path / "run-003" / "agent_stdout.log").read_text()


def test_launch_run_auto_resumes_codex_after_token_limit(tmp_path):
    dispatch_task("Review ATPG regression risk", runs_dir=tmp_path, run_id="run-token")
    marker = tmp_path / "attempts.txt"
    fake_codex = tmp_path / "codex"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "count = int(marker.read_text()) if marker.exists() else 0\n"
        "marker.write_text(str(count + 1))\n"
        "sys.stdin.read()\n"
        "if count == 0:\n"
        "    print('token limit reached; try later', file=sys.stderr)\n"
        "    sys.exit(2)\n"
        "print('resumed ok')\n"
    )
    fake_codex.chmod(0o755)

    result = launch_run(
        "run-token",
        runs_dir=tmp_path,
        agent_cmd=(str(fake_codex), "exec", "-"),
        codex_auto_resume=True,
        codex_resume_delay_s=0,
        codex_max_resumes=1,
    )
    loaded = load_run("run-token", runs_dir=tmp_path)
    stderr = (tmp_path / "run-token" / "agent_stderr.log").read_text()
    stdout = (tmp_path / "run-token" / "agent_stdout.log").read_text()

    assert result.final_status == RunStatus.COMPLETED
    assert loaded.status == RunStatus.COMPLETED
    assert marker.read_text() == "2"
    assert "token limit reached" in stderr
    assert "automatic resume attempt" in stdout
    assert any(event.status == RunStatus.SLEEPING for event in loaded.events)


def test_launch_queued_runs_skips_non_queued_runs(tmp_path):
    dispatch_task("Review ATPG regression risk", runs_dir=tmp_path, run_id="run-004")
    dispatch_task("Publish documentation update", runs_dir=tmp_path, run_id="run-005")
    complete_run("run-005", "Already handled", runs_dir=tmp_path)
    command = (sys.executable, "-c", "import sys; sys.stdin.read(); print('done')")

    results = launch_queued_runs(tmp_path, agent_cmd=command)

    assert [result.run_id for result in results] == ["run-004"]
    assert load_run("run-004", runs_dir=tmp_path).status == RunStatus.COMPLETED
    assert load_run("run-005", runs_dir=tmp_path).status == RunStatus.COMPLETED


def test_launch_cli_explains_non_queued_run_without_traceback(tmp_path):
    dispatch_task("Review ATPG regression risk", runs_dir=tmp_path, run_id="run-006")
    record_checkpoint("run-006", "initial", runs_dir=tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.orchestration.cli",
            "launch",
            "run-006",
            "--runs-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "launch only starts queued runs" in result.stderr
    assert "python -m src.orchestration.cli status run-006" in result.stderr
    assert "Traceback" not in result.stderr


def test_run_cli_dispatches_and_launches_with_one_command(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.orchestration.cli",
            "run",
            "Review ATPG regression risk",
            "--runs-dir",
            str(tmp_path),
            "--agent-cmd",
            f'{sys.executable} -c "import sys; data=sys.stdin.read(); '
            "print('received', 'Run:' in data)\"",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    records = list_runs(tmp_path)

    assert result.returncode == 0
    assert "Created run:" in result.stdout
    assert "Status: completed" in result.stdout
    assert len(records) == 1
    assert records[0].status == RunStatus.COMPLETED
    assert "received True" in (tmp_path / records[0].run_id / "agent_stdout.log").read_text()


def test_multi_agent_runtime_runs_code_gates_and_docs(tmp_path):
    command = (
        sys.executable,
        "-c",
        "import sys; data=sys.stdin.read(); print('role_context', 'Runtime Context' in data)",
    )

    result = run_multi_agent_runtime(
        "Improve training coverage",
        runs_dir=tmp_path,
        agent_cmd=command,
        code_agents=2,
    )
    parent = load_run(result.run_id, runs_dir=tmp_path)
    summary = json.loads((tmp_path / result.run_id / "multi_agent_summary.json").read_text())

    assert result.final_status == RunStatus.COMPLETED
    assert parent.status == RunStatus.COMPLETED
    # 2 coding + 2 lean gates + 1 docs = 5 children
    assert len(result.child_run_ids) == 5
    assert summary["status"] == "completed"
    assert any("coding-agent-1" in run_id for run_id in result.child_run_ids)
    for child_run_id in result.child_run_ids:
        prompt = (tmp_path / child_run_id / "agent_prompt.md").read_text()
        stdout = (tmp_path / child_run_id / "agent_stdout.log").read_text()
        assert "Runtime Context" in prompt
        assert "role_context True" in stdout


def test_multi_agent_runtime_recovers_and_retries_failed_gate(tmp_path):
    marker = tmp_path / "workaround.txt"
    fake_agent = tmp_path / "fake_agent.py"
    fake_agent.write_text(
        "import pathlib, sys\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "data = sys.stdin.read()\n"
        "if 'workaround_agent' in data:\n"
        "    marker.write_text('done')\n"
        "    print('workaround applied')\n"
        "    sys.exit(0)\n"
        "print('agent ran')\n"
        "if 'Role: test_coverage_gate' in data and not marker.exists():\n"
        "    sys.exit(2)\n"
    )
    command = (
        sys.executable,
        str(fake_agent),
    )

    result = run_multi_agent_runtime(
        "Improve training coverage",
        runs_dir=tmp_path,
        agent_cmd=command,
        code_agents=1,
    )

    parent = load_run(result.run_id, runs_dir=tmp_path)
    summary = json.loads((tmp_path / result.run_id / "multi_agent_summary.json").read_text())

    assert result.final_status == RunStatus.COMPLETED
    assert parent.status == RunStatus.COMPLETED
    assert summary["status"] == "completed"
    assert any("test-coverage-gate" in run_id for run_id in result.child_run_ids)
    assert any("gate-workaround-agent-attempt-1" in run_id for run_id in result.child_run_ids)
    assert any("test-coverage-gate-retry-1" in run_id for run_id in result.child_run_ids)
    assert any("docs-results-agent" in run_id for run_id in result.child_run_ids)


def test_multi_agent_runtime_fails_after_recovery_limit(tmp_path):
    command = (
        sys.executable,
        "-c",
        "import sys; sys.stdin.read(); sys.exit(2)",
    )

    result = run_multi_agent_runtime(
        "Improve training coverage",
        runs_dir=tmp_path,
        agent_cmd=command,
        code_agents=1,
        max_recovery_attempts=1,
    )

    parent = load_run(result.run_id, runs_dir=tmp_path)
    summary = json.loads((tmp_path / result.run_id / "multi_agent_summary.json").read_text())

    assert result.final_status == RunStatus.FAILED
    assert parent.status == RunStatus.FAILED
    assert "coordinated recovery" in summary["message"]
    assert any("code-workaround-agent-attempt-1" in run_id for run_id in result.child_run_ids)
    assert not any("docs-results-agent" in run_id for run_id in result.child_run_ids)


def test_tmux_monitor_dry_run_includes_parent_and_child_events(tmp_path):
    command = (
        sys.executable,
        "-c",
        "import sys; sys.stdin.read(); print('done')",
    )
    runtime = run_multi_agent_runtime(
        "Improve training coverage",
        runs_dir=tmp_path,
        agent_cmd=command,
        code_agents=1,
        include_docs_agent=False,
    )

    monitor = launch_tmux_monitor(
        runtime.run_id,
        runs_dir=tmp_path,
        session_name="s-imply-test",
        dry_run=True,
    )

    assert monitor.session_name == "s-imply-test"
    assert monitor.run_ids[0] == runtime.run_id
    assert set(monitor.run_ids[1:]) == set(runtime.child_run_ids)
    assert monitor.attach_command == ("tmux", "attach-session", "-t", "s-imply-test")
    assert monitor.commands[0] == ("tmux", "kill-session", "-t", "s-imply-test")
    assert any("events.jsonl" in " ".join(command) for command in monitor.commands)


def test_tmux_monitor_accepts_parent_run_prefix(tmp_path):
    command = (
        sys.executable,
        "-c",
        "import sys; sys.stdin.read(); print('done')",
    )
    runtime = run_multi_agent_runtime(
        "Improve training coverage",
        runs_dir=tmp_path,
        agent_cmd=command,
        code_agents=1,
        include_docs_agent=False,
    )

    # Timestamp-only prefix matches the parent and all child runs.
    prefix = runtime.run_id.split("-", 1)[0]
    monitor = launch_tmux_monitor(prefix, runs_dir=tmp_path, dry_run=True)

    assert monitor.run_ids[0] == runtime.run_id
    assert set(monitor.run_ids[1:]) == set(runtime.child_run_ids)


def test_monitor_cli_dry_run_prints_attach_command(tmp_path):
    dispatch_task("Review ATPG regression risk", runs_dir=tmp_path, run_id="run-007")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.orchestration.cli",
            "monitor",
            "run-007",
            "--runs-dir",
            str(tmp_path),
            "--session-name",
            "s-imply-test",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Attach: tmux attach-session -t s-imply-test" in result.stdout
    assert "Dry run commands:" in result.stdout


def test_monitor_cli_reports_missing_prefix_without_traceback(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.orchestration.cli",
            "monitor",
            "missing-run",
            "--runs-dir",
            str(tmp_path),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "No orchestration run matches: missing-run" in result.stderr
    assert "Traceback" not in result.stderr


def test_run_cli_accepts_codex_resume_flags(tmp_path):
    fake_codex = tmp_path / "codex"
    fake_codex.write_text("#!/usr/bin/env python3\nimport sys\nsys.stdin.read()\nprint('ok')\n")
    fake_codex.chmod(0o755)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.orchestration.cli",
            "run",
            "Review ATPG regression risk",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--agent-cmd",
            f"{fake_codex} exec -",
            "--codex-auto-resume",
            "--codex-resume-delay-s",
            "0",
            "--codex-max-resumes",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Status: completed" in result.stdout


def test_default_agent_command_supports_gemini_profile():
    command = default_agent_command("gemini")

    assert command[:2] == ("gemini", "--skip-trust")
    assert "-p" in command


# --- Anti-thrash invariant tests ---


def test_sibling_summary_per_entry_capped(tmp_path):
    """Each sibling summary must not exceed _SIBLING_PER_CAP chars (+truncation marker)."""
    command = (sys.executable, "-c", "import sys; sys.stdin.read(); print('x' * 5000)")
    dispatch_task("Review ATPG regression risk", runs_dir=tmp_path, run_id="cap-001")
    result = launch_run("cap-001", runs_dir=tmp_path, agent_cmd=command)

    summaries = _collect_sibling_summaries(["cap-001"], [result], tmp_path)
    _, detail = summaries[0]

    assert len(detail) <= _SIBLING_PER_CAP + len(" [truncated]")


def test_gate_prompt_contains_pytest_command(tmp_path):
    command = (sys.executable, "-c", "import sys; sys.stdin.read(); print('done')")
    run_multi_agent_runtime(
        "Improve training coverage",
        runs_dir=tmp_path,
        agent_cmd=command,
        code_agents=1,
        include_docs_agent=False,
    )
    gate_runs = [r for r in list_runs(tmp_path) if "test-coverage-gate" in r.run_id]
    assert gate_runs
    prompt = (tmp_path / gate_runs[0].run_id / "agent_prompt.md").read_text()
    assert "pytest" in prompt


def test_recovery_spec_goal_contains_failed_run_id(tmp_path):
    dispatch_task("Review ATPG regression risk", runs_dir=tmp_path, run_id="fail-001")
    failed_result = launch_run(
        "fail-001",
        runs_dir=tmp_path,
        agent_cmd=(sys.executable, "-c", "import sys; sys.stdin.read(); sys.exit(1)"),
    )
    specs = _recovery_specs("Fix training bug", "code", 1, (failed_result,))
    assert "fail-001" in specs[0].goal


def test_coding_spec_goal_contains_file_scope():
    specs = _multi_agent_specs("Fix ATPG backtrace bug", code_agents=1, include_docs_agent=False)
    code_spec = next(s for s in specs if s.phase == "code")
    assert any(
        kw in code_spec.goal for kw in ("src/atpg/", "src/ml/", "scripts/", "docs/", "<repo>")
    )


def test_build_agent_prompt_contains_budget_section(tmp_path):
    dispatch_task("Fix GradScaler checkpoint compatibility", runs_dir=tmp_path, run_id="budget-001")
    prompt = (tmp_path / "budget-001" / "agent_prompt.md").read_text()
    assert "## Budget" in prompt
    assert "Max file edits" in prompt


def test_two_lean_gates_replace_single_merged_gate():
    specs = _multi_agent_specs("Improve training coverage", code_agents=1, include_docs_agent=False)
    gate_specs = [s for s in specs if s.phase == "gate"]
    assert len(gate_specs) == 2
    roles = {s.role for s in gate_specs}
    assert roles == {"test_coverage_gate", "quality_review_gate"}
    cov = next(s for s in gate_specs if s.role == "test_coverage_gate")
    qual = next(s for s in gate_specs if s.role == "quality_review_gate")
    # Coverage gate must invoke pytest; quality gate must check ruff, not rerun tests.
    assert "pytest" in cov.goal
    assert "ruff" in qual.goal
    assert "pytest" not in qual.goal


# --- Gemini CLI compatibility tests ---


def test_gemini_prompt_injected_into_p_arg():
    """_inject_gemini_prompt replaces the -p placeholder with the actual prompt."""
    base_cmd = ("gemini", "--skip-trust", "--approval-mode", "yolo", "-p", "")
    result = _inject_gemini_prompt(base_cmd, "hello agent")
    idx = list(result).index("-p")
    assert result[idx + 1] == "hello agent"
    # Original command must be unchanged.
    assert base_cmd[-1] == ""


def test_gemini_prompt_appended_when_p_absent():
    """_inject_gemini_prompt falls back to appending -p when not present."""
    base_cmd = ("gemini", "--skip-trust")
    result = _inject_gemini_prompt(base_cmd, "hello agent")
    assert result[-2] == "-p"
    assert result[-1] == "hello agent"


def test_supports_auto_resume_covers_gemini_and_codex():
    assert _supports_auto_resume(("gemini", "-p", ""))
    assert _supports_auto_resume(("codex", "exec", "-"))
    assert not _supports_auto_resume(("claude",))
    assert not _supports_auto_resume(("python", "-m", "something"))


def test_is_gemini_command_matches_by_basename():
    assert _is_gemini_command(("/usr/local/bin/gemini", "--skip-trust"))
    assert not _is_gemini_command(("codex",))
    assert not _is_gemini_command(())


def test_default_agent_command_gemini_has_p_placeholder():
    cmd = default_agent_command("gemini")
    assert "-p" in cmd
    # Placeholder must be the empty string so _inject_gemini_prompt can replace it.
    idx = list(cmd).index("-p")
    assert cmd[idx + 1] == ""


def test_launch_run_with_fake_gemini_delivers_prompt(tmp_path):
    """A fake 'gemini' binary that reads argv prints the -p content."""
    fake_gemini = tmp_path / "gemini"
    fake_gemini.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "args = sys.argv[1:]\n"
        "if '-p' in args:\n"
        "    idx = args.index('-p')\n"
        "    print('got_prompt', len(args[idx + 1]) > 0)\n"
        "else:\n"
        "    print('no_prompt_arg')\n"
    )
    fake_gemini.chmod(0o755)

    dispatch_task("Review ATPG regression risk", runs_dir=tmp_path, run_id="gem-001")
    result = launch_run(
        "gem-001",
        runs_dir=tmp_path,
        agent_cmd=(str(fake_gemini), "--skip-trust", "--approval-mode", "yolo", "-p", ""),
    )
    stdout = (tmp_path / "gem-001" / "agent_stdout.log").read_text()

    assert result.final_status == RunStatus.COMPLETED
    assert "got_prompt True" in stdout
