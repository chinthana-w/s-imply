"""Coordinator-only multi-agent routing for s-imply.

This module is intentionally framework-light.  LangGraph can wrap these pure
functions as nodes, while tests and local scripts can call them directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskType(str, Enum):
    ATPG = "atpg"
    ML = "ml"
    BENCHMARK = "benchmark"
    DOCS = "docs"
    THEORY = "theory"
    REVIEW = "review"


class AgentRole(str, Enum):
    PLANNER = "planner_router"
    ATPG_SOLVER = "atpg_solver"
    ML_TRAINING = "ml_training"
    BENCHMARK_EXPERIMENT = "benchmark_experiment"
    QUALITY_REVIEWER = "quality_reviewer"
    DOCS_RESULTS = "docs_results"


class AutonomyLevel(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(frozen=True)
class NotionPublicationTarget:
    """Configuration required before publishing docs into Notion."""

    enabled: bool = True
    destination: str = "https://www.notion.so/2f8182e9694b80358497e9e102837c4a"
    content_format: str = "canonical wiki page with method, results, and experiment log"
    sync_style: str = "notion_canonical"
    experiment_log_format: str = (
        "append dated experiment log entries with commands, artifacts, results, and next steps"
    )
    status: str = "configured"


@dataclass(frozen=True)
class OrchestrationState:
    goal: str
    task_type: TaskType
    owner_agent: AgentRole
    files_owned: tuple[str, ...]
    risk_level: str
    validation_plan: tuple[str, ...]
    expected_artifacts: tuple[str, ...]
    autonomy: AutonomyLevel = AutonomyLevel.MEDIUM
    run_manifest_required: bool = False
    notion_publication: NotionPublicationTarget = field(default_factory=NotionPublicationTarget)
    reviewer_notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TaskPacket:
    owner_agent: AgentRole
    task_type: TaskType
    files_owned: tuple[str, ...]
    constraints: tuple[str, ...]
    validation_plan: tuple[str, ...]
    expected_artifacts: tuple[str, ...]
    run_manifest_required: bool
    notion_publication: NotionPublicationTarget = field(default_factory=NotionPublicationTarget)


_KEYWORDS: tuple[tuple[TaskType, tuple[str, ...]], ...] = (
    (
        TaskType.THEORY,
        (
            "theory",
            "theoretical",
            "framework",
            "maamari",
            "lrr",
            "reconvergent region",
        ),
    ),
    (
        TaskType.ATPG,
        (
            "podem",
            "atpg",
            "backtrace",
            "fault",
            "reconv",
            "reconvergence",
            "solver",
        ),
    ),
    (
        TaskType.ML,
        (
            "train",
            "gradscaler",
            "checkpoint",
            "dataset",
            "loss",
            "model",
            "amp",
            "deepgate",
        ),
    ),
    (
        TaskType.BENCHMARK,
        (
            "benchmark",
            "coverage",
            "itc99",
            "iscas",
            "experiment",
            "csv",
            "metric",
            "result",
        ),
    ),
    (TaskType.REVIEW, ("review", "regression", "lint", "quality", "audit")),
    (
        TaskType.DOCS,
        ("docs", "readme", "paper", "summary", "documentation", "notion", "wiki"),
    ),
)


_ROLE_BY_TASK: dict[TaskType, AgentRole] = {
    TaskType.ATPG: AgentRole.ATPG_SOLVER,
    TaskType.ML: AgentRole.ML_TRAINING,
    TaskType.BENCHMARK: AgentRole.BENCHMARK_EXPERIMENT,
    TaskType.DOCS: AgentRole.DOCS_RESULTS,
    TaskType.THEORY: AgentRole.DOCS_RESULTS,
    TaskType.REVIEW: AgentRole.QUALITY_REVIEWER,
}


_FILES_BY_TASK: dict[TaskType, tuple[str, ...]] = {
    TaskType.ATPG: ("src/atpg/", "tests/test_*podem*.py", "tests/test_reconv_solver.py"),
    TaskType.ML: ("src/ml/", "tests/test_model_pe.py"),
    TaskType.BENCHMARK: ("scripts/", "docs/*report*", "data/bench/"),
    TaskType.DOCS: ("docs/project_summary.md", "README.md"),
    TaskType.THEORY: ("docs/paper_draft.tex", "docs/project_summary.md", "src/atpg/"),
    TaskType.REVIEW: ("<diff>", "<validation artifacts>"),
}


def classify_task(goal: str, changed_files: tuple[str, ...] = ()) -> TaskType:
    """Classify work into the first specialist lane that should own it."""
    text = " ".join((goal, *changed_files)).lower()
    for task_type, keywords in _KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return task_type
    return TaskType.REVIEW


def create_task_packet(
    goal: str,
    changed_files: tuple[str, ...] = (),
    autonomy: AutonomyLevel = AutonomyLevel.MEDIUM,
) -> TaskPacket:
    """Create a decision-complete handoff packet for one specialist agent."""
    task_type = classify_task(goal, changed_files)
    owner = _ROLE_BY_TASK[task_type]
    run_manifest_required = task_type in {TaskType.BENCHMARK, TaskType.ML}
    if autonomy is AutonomyLevel.LARGE:
        run_manifest_required = True

    constraints = [
        "Use the deepgate conda environment.",
        "Prefer python -m entrypoints for repo modules.",
        "Keep edits compatible with Ruff line length 100 and rules E/F/I.",
        "Do not make unsupported result claims without artifact provenance.",
    ]
    notion_publication = NotionPublicationTarget(enabled=False, status="not_applicable")
    if task_type in {TaskType.DOCS, TaskType.THEORY, TaskType.BENCHMARK}:
        constraints.append("Treat Notion as canonical for method/results documentation.")
        constraints.append("Update experiment steps in Notion using dated log entries.")
        notion_publication = NotionPublicationTarget()
    if task_type is TaskType.THEORY:
        constraints.append("Synchronize theoretical-framework changes with docs/paper_draft.tex.")

    return TaskPacket(
        owner_agent=owner,
        task_type=task_type,
        files_owned=_FILES_BY_TASK[task_type],
        constraints=tuple(constraints),
        validation_plan=_validation_plan(task_type),
        expected_artifacts=_expected_artifacts(task_type),
        run_manifest_required=run_manifest_required,
        notion_publication=notion_publication,
    )


def _validation_plan(task_type: TaskType) -> tuple[str, ...]:
    common = (
        "Run focused tests for touched behavior.",
        "Run scoped Ruff checks or note baseline debt.",
    )
    if task_type is TaskType.ATPG:
        return common + ("Run a small AI-vs-vanilla ATPG benchmark subset if behavior changed.",)
    if task_type is TaskType.ML:
        return common + ("Run import/config smoke checks and checkpoint compatibility checks.",)
    if task_type is TaskType.BENCHMARK:
        return (
            "Create a run manifest before execution.",
            "Compare metrics against a documented baseline.",
            "Store CSV/JSON artifacts with command provenance.",
            "Prepare a Notion-ready result summary after target configuration.",
        )
    if task_type is TaskType.THEORY:
        return common + (
            "Confirm docs/paper_draft.tex reflects the theory change.",
            "Prepare a Notion-ready theory-change summary after target configuration.",
        )
    if task_type is TaskType.DOCS:
        return (
            "Verify docs match current runtime behavior.",
            "Prepare a Notion-ready documentation summary after target configuration.",
        )
    return ("Review diff, tests, artifacts, and result claims.",)


def _expected_artifacts(task_type: TaskType) -> tuple[str, ...]:
    if task_type is TaskType.BENCHMARK:
        return (
            "run_manifest.json",
            "benchmark_report.json or benchmark_report.csv",
            "Notion-ready result summary when configured",
        )
    if task_type is TaskType.ML:
        return ("run_manifest.json for GPU/eval jobs", "checkpoint compatibility summary")
    if task_type is TaskType.THEORY:
        return (
            "docs/paper_draft.tex update",
            "docs/project_summary.md update if architecture changed",
            "Notion-ready theory-change summary when configured",
        )
    if task_type is TaskType.DOCS:
        return ("documentation diff", "Notion-ready documentation summary when configured")
    return ("test output summary", "review notes")
