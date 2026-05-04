"""LangGraph wrapper around the coordinator."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.orchestration.coordinator import create_task_packet


class CoordinatorGraphState(TypedDict, total=False):
    goal: str
    changed_files: list[str]
    task_packet: dict[str, Any]


def route_task(state: CoordinatorGraphState) -> CoordinatorGraphState:
    """LangGraph node: create a specialist task packet from the user goal."""
    packet = create_task_packet(
        state.get("goal", ""),
        changed_files=tuple(state.get("changed_files", [])),
    )
    return {
        **state,
        "task_packet": {
            "owner_agent": packet.owner_agent.value,
            "task_type": packet.task_type.value,
            "files_owned": list(packet.files_owned),
            "constraints": list(packet.constraints),
            "validation_plan": list(packet.validation_plan),
            "expected_artifacts": list(packet.expected_artifacts),
            "run_manifest_required": packet.run_manifest_required,
            "notion_publication": {
                "enabled": packet.notion_publication.enabled,
                "destination": packet.notion_publication.destination,
                "content_format": packet.notion_publication.content_format,
                "sync_style": packet.notion_publication.sync_style,
                "experiment_log_format": packet.notion_publication.experiment_log_format,
                "status": packet.notion_publication.status,
            },
        },
    }


def build_graph() -> Any:
    """Build the minimal coordinator graph."""
    graph = StateGraph(CoordinatorGraphState)
    graph.add_node("route_task", route_task)
    graph.add_edge(START, "route_task")
    graph.add_edge("route_task", END)
    return graph.compile()
