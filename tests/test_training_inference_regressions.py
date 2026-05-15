from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

from scripts import benchmark_itc99_gate, build_fault_dataset
from src.atpg.podem import SUCCESS
from src.util.struct import Fault, GateType, LogicValue


def test_collect_sorted_pairs_accepts_solver_list_return():
    circuit = [None] * 6
    circuit[1] = SimpleNamespace(fin=[])
    circuit[2] = SimpleNamespace(fin=[1])
    circuit[3] = SimpleNamespace(fin=[1])
    circuit[4] = SimpleNamespace(fin=[2, 3])
    circuit[5] = SimpleNamespace(fin=[4])

    long_near_pair = {
        "start": 1,
        "reconv": 5,
        "paths": [[1, 2, 4, 5], [1, 3, 4, 5]],
    }
    short_far_pair = {
        "start": 1,
        "reconv": 4,
        "paths": [[1, 2, 4], [1, 3, 4]],
    }

    solver = SimpleNamespace(
        circuit=circuit,
        _collect_and_sort_pairs=lambda _target: [long_near_pair, short_far_pair],
        _get_transitive_fanin=lambda _target: {1, 2, 3, 4, 5},
    )

    assert build_fault_dataset._collect_sorted_pairs(solver, 5) == [
        short_far_pair,
        long_near_pair,
    ]


def test_itc99_no_fallback_keeps_configured_backtrack_budget():
    args = Namespace(max_backtracks=123)

    assert benchmark_itc99_gate._ai_podem_backtrack_budget(args) == 123


def test_improved_hint_backtracer_uses_base_backtrace_for_unhinted_objective():
    circuit = [None] * 4
    circuit[1] = SimpleNamespace(type=GateType.INPT, val=LogicValue.XD, nfi=0, fin=[])
    circuit[2] = SimpleNamespace(type=GateType.INPT, val=LogicValue.XD, nfi=0, fin=[])
    circuit[3] = SimpleNamespace(
        type=GateType.AND,
        val=LogicValue.XD,
        nfi=2,
        fin=[1, 2],
    )

    backtracer = benchmark_itc99_gate.ImprovedHintBacktracer({}, no_fallback=True)
    result = backtracer(Fault(3, LogicValue.ONE), circuit)

    assert result.gate_id in {1, 2}
    assert result.value == LogicValue.ONE


def test_detecting_pattern_teacher_returns_good_circuit_assignment():
    circuit = [None] * 3
    circuit[1] = SimpleNamespace(type=GateType.INPT, val=LogicValue.XD)
    circuit[2] = SimpleNamespace(type=GateType.BUFF, val=LogicValue.XD, fin=[1])
    fault = SimpleNamespace(gate_id=2, value=LogicValue.D)

    def fake_podem(circuit, *_args, **_kwargs):
        circuit[1].val = LogicValue.ONE
        return SUCCESS

    def fake_logic_sim(circuit, *_args, **_kwargs):
        circuit[2].val = circuit[1].val

    with (
        patch.object(build_fault_dataset, "podem", side_effect=fake_podem),
        patch.object(build_fault_dataset, "initialize"),
        patch.object(build_fault_dataset, "logic_sim", side_effect=fake_logic_sim),
    ):
        assignment = build_fault_dataset._detecting_pattern_assignment(
            circuit,
            total_gates=2,
            topo_order=[1, 2],
            fault=fault,
            timeout=1.0,
            max_backtracks=10,
        )

    assert assignment == {1: 1, 2: 1}


def test_notion_summary_marks_classic_backtracks_unmeasured(tmp_path):
    report = {
        "created_at": "2026-05-13T00:00:00+00:00",
        "command": ["python", "-m", "scripts.benchmark_itc99_gate"],
        "model": "checkpoint.pth",
        "fault_list": "gate.json",
        "artifact_paths": {"json": "report.json", "csv": "per_fault.csv"},
        "succeeded": 8,
        "total": 10,
        "coverage": 0.8,
        "ai_backtracks_total": 42,
        "classic_backtracks_total": 0,
        "compare_classic": False,
        "activation_precheck_succeeded": 1,
        "coverage_target": 0.8,
        "passed_coverage_target": True,
        "backtrack_target": False,
        "passed_backtrack_target": None,
        "baseline_comparison": {
            "label": "baseline",
            "coverage": 0.1,
            "source": "source.md",
            "delta": 0.7,
            "decision_comparable": True,
        },
    }
    out = tmp_path / "summary.md"

    benchmark_itc99_gate._write_notion_summary(str(out), report, None)

    text = out.read_text()
    assert "internal PODEM backtracks" in text
    assert "classic not measured" in text
    assert "AI/model backtrack comparison=N/A" in text
    assert "Backtrack target enabled: False; pass=N/A" in text
    assert "classic `0`" not in text


def test_notion_summary_treats_compare_backtrack_target_as_not_comparable(tmp_path):
    report = {
        "created_at": "2026-05-13T00:00:00+00:00",
        "command": ["python", "-m", "scripts.benchmark_itc99_gate", "--compare-classic"],
        "model": "checkpoint.pth",
        "fault_list": "gate.json",
        "artifact_paths": {"json": "report.json", "csv": "per_fault.csv"},
        "succeeded": 8,
        "total": 10,
        "coverage": 0.8,
        "ai_backtracks_total": 42,
        "classic_backtracks_total": 123,
        "classic_backtracks_on_ai_success": 100,
        "compare_classic": True,
        "activation_precheck_succeeded": 1,
        "coverage_target": 0.8,
        "passed_coverage_target": True,
        "backtrack_target": True,
        "passed_backtrack_target": None,
        "baseline_comparison": {
            "label": "baseline",
            "coverage": 0.1,
            "source": "source.md",
            "delta": 0.7,
            "decision_comparable": True,
        },
    }
    out = tmp_path / "summary.md"

    benchmark_itc99_gate._write_notion_summary(str(out), report, None)

    text = out.read_text()
    assert "Classic search effort: `123` total backtracks" in text
    assert "AI/model backtrack comparison=N/A" in text
    assert "pass=N/A because AI has no comparable backtrack metric" in text
    assert "AI less than classic" not in text
