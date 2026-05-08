import csv
import json
import subprocess
import sys

from src.orchestration.coordinator import AgentRole, TaskType, classify_task, create_task_packet
from src.orchestration.runner import (
    RunStatus,
    complete_run,
    default_agent_command,
    dispatch_task,
    launch_queued_runs,
    launch_run,
    list_runs,
    load_run,
    record_checkpoint,
    run_multi_agent_runtime,
)
from src.orchestration.mcp_server import handle_request
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
        "import sys; data=sys.stdin.read(); print('received', 'Run ID:' in data)",
    )

    result = launch_run("run-003", runs_dir=tmp_path, agent_cmd=command)
    loaded = load_run("run-003", runs_dir=tmp_path)

    assert result.returncode == 0
    assert result.final_status == RunStatus.COMPLETED
    assert loaded.status == RunStatus.COMPLETED
    assert "received True" in (tmp_path / "run-003" / "agent_stdout.log").read_text()


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
            f"{sys.executable} -c \"import sys; data=sys.stdin.read(); "
            "print('received', 'Run ID:' in data)\"",
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
    assert "received True" in (
        tmp_path / records[0].run_id / "agent_stdout.log"
    ).read_text()


def test_multi_agent_runtime_runs_code_gates_and_docs(tmp_path):
    command = (
        sys.executable,
        "-c",
        "import sys; data=sys.stdin.read(); print('role_context', "
        "'Multi-Agent Runtime Context' in data)",
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
    assert len(result.child_run_ids) == 5
    assert summary["status"] == "completed"
    assert any("coding-agent-1" in run_id for run_id in result.child_run_ids)
    for child_run_id in result.child_run_ids:
        prompt = (tmp_path / child_run_id / "agent_prompt.md").read_text()
        stdout = (tmp_path / child_run_id / "agent_stdout.log").read_text()
        assert "Multi-Agent Runtime Context" in prompt
        assert "role_context True" in stdout


def test_multi_agent_runtime_stops_when_gate_fails(tmp_path):
    command = (
        sys.executable,
        "-c",
        "import sys; data=sys.stdin.read(); "
        "sys.exit(2 if 'test_coverage_gate' in data else 0)",
    )

    result = run_multi_agent_runtime(
        "Improve training coverage",
        runs_dir=tmp_path,
        agent_cmd=command,
        code_agents=1,
    )

    assert result.final_status == RunStatus.FAILED
    assert len(result.child_run_ids) == 2
    assert any("test-coverage-gate" in run_id for run_id in result.child_run_ids)
    assert not any("docs-results-agent" in run_id for run_id in result.child_run_ids)


def test_default_agent_command_supports_gemini_profile():
    command = default_agent_command("gemini")

    assert command[:2] == ("gemini", "--skip-trust")
    assert "-p" in command
