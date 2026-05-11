"""Dependency-free stdio MCP server for s-imply repo tools.

The server exposes a small allowlisted tool surface for local agents:

- ``run_atpg``: bounded vanilla PODEM over a fault subset.
- ``run_test_coverage``: focused pytest targets with coverage JSON output.
- ``simulate_circuit``: forward simulation for a bench file and assignments.

It implements the MCP stdio JSON-RPC framing directly so the repo does not need
an additional Python MCP package just to run local tools.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

from src.orchestration.tools import run_atpg, run_test_coverage, simulate_circuit

SERVER_NAME = "s-imply-local-tools"
SERVER_VERSION = "0.1.0"


TOOLS: dict[str, dict[str, Any]] = {
    "run_atpg": {
        "description": "Run bounded vanilla PODEM ATPG on a bench circuit.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bench_path": {"type": "string"},
                "limit_faults": {"type": "integer", "default": 10, "minimum": 1},
                "max_backtracks": {"type": "integer", "default": 2000, "minimum": 1},
                "timeout_s": {"type": "number", "default": 5.0},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["bench_path"],
        },
    },
    "run_test_coverage": {
        "description": "Run focused pytest targets and emit coverage JSON.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "test_targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "coverage_json": {"type": "string", "default": "docs/test_coverage.json"},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["test_targets"],
        },
    },
    "simulate_circuit": {
        "description": "Forward-simulate a bench circuit with explicit assignments.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bench_path": {"type": "string"},
                "assignments": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                },
                "fault_gate_id": {"type": ["integer", "null"], "default": None},
                "fault_value": {"type": ["integer", "null"], "default": None},
            },
            "required": ["bench_path", "assignments"],
        },
    },
}


CALLABLES: dict[str, Callable[..., dict[str, Any]]] = {
    "run_atpg": run_atpg,
    "run_test_coverage": lambda test_targets, **kwargs: run_test_coverage(
        tuple(test_targets),
        **kwargs,
    ),
    "simulate_circuit": simulate_circuit,
}


def main() -> None:
    while True:
        request = _read_message()
        if request is None:
            break
        response = handle_request(request)
        if response is not None:
            _write_message(response)


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    if request_id is None:
        return None
    try:
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
        elif method == "tools/list":
            result = {"tools": [_tool_descriptor(name, spec) for name, spec in TOOLS.items()]}
        elif method == "tools/call":
            result = _call_tool(request.get("params") or {})
        else:
            return _error_response(request_id, -32601, f"Unsupported method: {method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except Exception as exc:  # pragma: no cover - exercised through client behavior
        return _error_response(request_id, -32000, str(exc))


def _call_tool(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if name not in CALLABLES:
        raise ValueError(f"Unknown tool: {name}")
    payload = CALLABLES[name](**arguments)
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, indent=2, default=str),
            }
        ],
        "isError": False,
    }


def _tool_descriptor(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": spec["description"],
        "inputSchema": spec["inputSchema"],
    }


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _read_message() -> dict[str, Any] | None:
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        name, _, value = line.decode("ascii").partition(":")
        headers[name.lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def _write_message(message: dict[str, Any]) -> None:
    body = json.dumps(message, separators=(",", ":"), default=str).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
