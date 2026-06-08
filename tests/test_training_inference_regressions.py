from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import src.atpg.podem as podem_mod
from scripts import benchmark_itc99_gate, build_fault_dataset
from src.atpg.podem import SUCCESS
from src.atpg.recursive_reconv_solver import HierarchicalReconvSolver
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


def test_reconv_only_filter_keeps_only_faults_with_pairs():
    faults = [
        Fault(1, LogicValue.ZERO),
        Fault(2, LogicValue.ONE),
        Fault(3, LogicValue.ZERO),
    ]
    solver = SimpleNamespace(
        _collect_and_sort_pairs=lambda gate_id: [{"reconv": gate_id}] if gate_id in {1, 3} else []
    )

    filtered, meta = benchmark_itc99_gate._filter_reconv_faults(faults, solver)

    assert [fault.gate_id for fault in filtered] == [1, 3]
    assert meta["pre_reconv_filter_faults"] == 3
    assert meta["reconv_faults"] == 2
    assert meta["non_reconv_faults_skipped"] == 1


def test_write_fault_list_round_trips_reusable_reconv_faults(tmp_path):
    out = tmp_path / "reconv_faults.json"
    faults = [
        Fault(10, LogicValue.ZERO),
        Fault(20, LogicValue.ONE),
    ]

    benchmark_itc99_gate._write_fault_list(
        str(out),
        bench_path="b17.bench",
        faults=faults,
        source_meta={"fault_list": "input.json", "full": True},
        filter_meta={
            "pre_reconv_filter_faults": 3,
            "reconv_faults": 2,
            "non_reconv_faults_skipped": 1,
        },
    )

    bench, loaded_faults, meta = benchmark_itc99_gate._load_gate_faults(str(out))

    assert bench == "b17.bench"
    assert [(fault.gate_id, fault.value) for fault in loaded_faults] == [
        (10, LogicValue.ZERO),
        (20, LogicValue.ONE),
    ]
    assert meta["filter"]["type"] == "reconv_only"
    assert meta["source"]["pre_filter_faults"] == 3


def test_podem_reset_statistics_clears_recursion_depth():
    podem_mod.depth = 5001
    podem_mod.backtrack_count = 17

    podem_mod.reset_statistics()

    assert podem_mod.depth == 0
    assert podem_mod.get_statistics()["backtrack_count"] == 0


def test_reconv_solver_skips_unqueued_pairs_until_relevant():
    circuit = [None] * 6
    for gid in range(1, 6):
        circuit[gid] = SimpleNamespace(type=GateType.INPT, fin=[], fot=[])
    circuit[4] = SimpleNamespace(type=GateType.BUFF, fin=[1], fot=[5])
    circuit[5] = SimpleNamespace(type=GateType.BUFF, fin=[4], fot=[])

    class FailingPredictor:
        def predict(self, *_args, **_kwargs):
            raise AssertionError("unrelated reconv pair should not be queried")

    solver = HierarchicalReconvSolver(circuit, FailingPredictor())
    unrelated_pair = {
        "start": 2,
        "reconv": 3,
        "paths": [[2, 3], [2, 3]],
    }

    assignment = solver._backward_justify(
        queue=[5],
        assignment={5: LogicValue.ONE},
        solved_pairs=set(),
        sorted_pairs=[unrelated_pair],
    )

    assert assignment[1] == LogicValue.ONE
    assert assignment[4] == LogicValue.ONE
    assert assignment[5] == LogicValue.ONE


def test_single_pass_structural_assignment_reaches_primary_inputs():
    circuit = [None] * 4
    circuit[1] = SimpleNamespace(type=GateType.INPT, fin=[], fot=[3])
    circuit[2] = SimpleNamespace(type=GateType.INPT, fin=[], fot=[3])
    circuit[3] = SimpleNamespace(type=GateType.AND, fin=[1, 2], fot=[])

    class UnusedPredictor:
        def predict(self, *_args, **_kwargs):
            raise AssertionError("single-pass structural repair must not query the model")

    solver = HierarchicalReconvSolver(circuit, UnusedPredictor())

    assignment = benchmark_itc99_gate._single_pass_structural_assignment(
        solver,
        target_node=3,
        target_val=LogicValue.ONE,
    )

    assert assignment == {
        1: LogicValue.ONE,
        2: LogicValue.ONE,
        3: LogicValue.ONE,
    }


def test_no_backtrack_podem_detection_never_flips_or_resets_decisions():
    circuit = [None] * 3
    circuit[1] = SimpleNamespace(type=GateType.INPT, val=LogicValue.XD, nfi=0, fin=[])
    circuit[2] = SimpleNamespace(type=GateType.BUFF, val=LogicValue.XD, nfi=1, fin=[1])
    fault = Fault(2, LogicValue.D)
    calls = {"at_po": 0}

    def fake_logic_sim(circuit, *_args, **_kwargs):
        circuit[2].val = LogicValue.D if circuit[1].val == LogicValue.ONE else LogicValue.XD

    def fake_fault_is_at_po(*_args, **_kwargs):
        calls["at_po"] += 1
        return calls["at_po"] > 1

    with (
        patch.object(benchmark_itc99_gate, "logic_sim", side_effect=fake_logic_sim),
        patch.object(benchmark_itc99_gate, "fault_is_at_po", side_effect=fake_fault_is_at_po),
        patch.object(
            benchmark_itc99_gate,
            "get_objective",
            return_value=Fault(2, LogicValue.ONE),
        ),
    ):
        detected, pi_count, elapsed = benchmark_itc99_gate._no_backtrack_podem_detection(
            circuit,
            total_gates=2,
            fault=fault,
            hints={},
            timeout=1.0,
        )

    assert detected is True
    assert pi_count == 1
    assert circuit[1].val == LogicValue.ONE
    assert elapsed >= 0.0


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


def test_direct_ai_assignment_detection_simulates_only_pi_assignments():
    circuit = [None] * 3
    circuit[1] = SimpleNamespace(type=GateType.INPT, val=LogicValue.XD)
    circuit[2] = SimpleNamespace(type=GateType.BUFF, val=LogicValue.XD)
    fault = Fault(2, LogicValue.D)

    def fake_logic_sim(circuit, *_args, **_kwargs):
        circuit[2].val = LogicValue.D if circuit[1].val == LogicValue.ONE else LogicValue.XD

    with (
        patch.object(benchmark_itc99_gate, "logic_sim", side_effect=fake_logic_sim) as sim,
        patch.object(benchmark_itc99_gate, "fault_is_at_po", return_value=True) as at_po,
    ):
        detected, pi_count, sim_time = benchmark_itc99_gate._direct_ai_assignment_detection(
            circuit,
            total_gates=2,
            fault=fault,
            assignment={1: LogicValue.ONE, 2: LogicValue.ONE},
        )

    assert detected is True
    assert pi_count == 1
    assert sim_time >= 0.0
    assert circuit[1].val == LogicValue.ONE
    sim.assert_called_once()
    at_po.assert_called_once()


def test_direct_ai_assignment_detection_skips_simulation_without_pi_assignment():
    circuit = [None] * 3
    circuit[1] = SimpleNamespace(type=GateType.INPT, val=LogicValue.XD)
    circuit[2] = SimpleNamespace(type=GateType.BUFF, val=LogicValue.XD)

    with (
        patch.object(benchmark_itc99_gate, "logic_sim") as sim,
        patch.object(benchmark_itc99_gate, "fault_is_at_po") as at_po,
    ):
        detected, pi_count, sim_time = benchmark_itc99_gate._direct_ai_assignment_detection(
            circuit,
            total_gates=2,
            fault=Fault(2, LogicValue.D),
            assignment={2: LogicValue.ONE},
        )

    assert detected is False
    assert pi_count == 0
    assert sim_time == 0.0
    sim.assert_not_called()
    at_po.assert_not_called()


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
        "coverage_target_observed": 0.8,
        "coverage_target_required_faults": 8,
        "coverage_target_denominator_count": 10,
        "coverage_target_denominator_note": "target is measured against all benchmark faults",
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
    assert "Coverage target: 80.0000% of `10` denominator faults" in text
    assert "classic `0`" not in text


def test_coverage_target_uses_classic_covered_denominator_when_comparing():
    metrics = benchmark_itc99_gate._coverage_target_metrics(
        succeeded=56,
        total=100,
        classic_succeeded=70,
        compare_classic=True,
        coverage_target=0.8,
    )

    assert metrics["denominator_name"] == "classic_succeeded"
    assert metrics["denominator"] == 70
    assert metrics["required"] == 56
    assert metrics["observed"] == 0.8
    assert metrics["passed"] is True


def test_coverage_target_fails_below_classic_relative_threshold():
    metrics = benchmark_itc99_gate._coverage_target_metrics(
        succeeded=55,
        total=100,
        classic_succeeded=70,
        compare_classic=True,
        coverage_target=0.8,
    )

    assert metrics["required"] == 56
    assert metrics["passed"] is False


def test_incomplete_coverage_target_uses_attempted_denominator_without_passing():
    metrics = benchmark_itc99_gate._coverage_target_metrics(
        succeeded=80,
        total=1000,
        attempted=100,
        classic_succeeded=0,
        compare_classic=False,
        coverage_target=0.8,
        complete=False,
    )

    assert metrics["denominator_name"] == "attempted_faults"
    assert metrics["denominator"] == 100
    assert metrics["observed"] == 0.8
    assert metrics["required"] == 80
    assert metrics["passed"] is False


def test_resource_abort_reason_trips_on_low_available_memory():
    args = Namespace(
        min_available_memory_gb=16.0,
        max_system_memory_percent=0.0,
        max_rss_gb=0.0,
        memory_guard_mode="both",
    )
    snapshot = {
        "mem_available_gb": 8.0,
        "mem_used_percent": 20.0,
        "process_rss_gb": 1.0,
    }

    reason = benchmark_itc99_gate._resource_abort_reason(args, snapshot)

    assert "available memory" in reason


def test_resource_abort_reason_accepts_disabled_limits():
    args = Namespace(
        min_available_memory_gb=0.0,
        max_system_memory_percent=0.0,
        max_rss_gb=0.0,
        memory_guard_mode="both",
    )

    assert benchmark_itc99_gate._resource_abort_reason(args, {}) is None


def test_resource_abort_reason_process_mode_ignores_system_memory():
    args = Namespace(
        min_available_memory_gb=16.0,
        max_system_memory_percent=50.0,
        max_rss_gb=2.0,
        memory_guard_mode="process",
    )
    snapshot = {
        "mem_available_gb": 1.0,
        "mem_used_percent": 99.0,
        "process_rss_gb": 1.0,
    }

    assert benchmark_itc99_gate._resource_abort_reason(args, snapshot) is None


def test_resource_abort_reason_process_mode_trips_on_rss():
    args = Namespace(
        min_available_memory_gb=0.0,
        max_system_memory_percent=0.0,
        max_rss_gb=2.0,
        memory_guard_mode="process",
    )
    snapshot = {
        "mem_available_gb": 100.0,
        "mem_used_percent": 10.0,
        "process_rss_gb": 3.0,
    }

    reason = benchmark_itc99_gate._resource_abort_reason(args, snapshot)

    assert "process RSS" in reason


def test_flush_runtime_caches_clears_solver_and_predictor_caches():
    solver = SimpleNamespace(
        pair_cache={1: "a", 2: "b"},
        _pair_cache_dirty=True,
        _persist_pair_cache_if_needed=lambda: None,
    )
    predictor = SimpleNamespace(prediction_cache={("x",): 1})

    stats = benchmark_itc99_gate._flush_runtime_caches(
        solver=solver,
        predictor=predictor,
        device="cpu",
    )

    assert stats["solver_pair_cache"] == 2
    assert stats["predictor_prediction_cache"] == 1
    assert solver.pair_cache == {}
    assert solver._pair_cache_dirty is False
    assert predictor.prediction_cache == {}


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
        "coverage_target_observed": 0.8,
        "coverage_target_required_faults": 8,
        "coverage_target_denominator_count": 10,
        "coverage_target_denominator_note": (
            "target is measured against faults covered by classic PODEM"
        ),
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
