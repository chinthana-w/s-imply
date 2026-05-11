import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.getcwd())

from src.atpg import benchmark_ai_podem
from src.atpg.ai_podem import (
    AIBacktracer,
    ai_podem,
)
from src.atpg.podem import BACKTRACK_LIMIT, SUCCESS, TIMEOUT
from src.util.struct import Fault, Gate, GateType, LogicValue


# Mock circuit
def create_mock_circuit():
    circuit = [None] * 5
    circuit[0] = Gate("dummy", GateType.INPT, 0, 0)
    circuit[1] = Gate("1", GateType.INPT, 0, 1)
    circuit[2] = Gate("2", GateType.INPT, 0, 1)
    circuit[3] = Gate("3", GateType.AND, 2, 0)
    circuit[3].fin = [1, 2]
    circuit[4] = Gate("4", GateType.NOT, 1, 0)

    circuit[1].fot = [3]
    circuit[2].fot = [3]
    circuit[1].val = LogicValue.XD
    circuit[2].val = LogicValue.XD

    return circuit, 3


class TestAIPodem(unittest.TestCase):
    @patch("src.atpg.ai_podem.ModelPairPredictor")
    @patch("src.atpg.ai_podem.HierarchicalReconvSolver")
    @patch("src.atpg.ai_podem.mogu_podem_wrapper")
    def test_ai_podem_fallback(self, mock_podem, mock_solver_cls, mock_predictor_cls):
        """Test that ai_podem falls back if AI fails."""
        circuit, total_gates = create_mock_circuit()
        fault = Fault(3, LogicValue.D)

        mock_solver_instance = mock_solver_cls.return_value
        mock_solver_instance.solve.return_value = None  # AI Fails

        mock_podem.return_value = True

        result = ai_podem(
            circuit,
            fault,
            total_gates,
            circuit_path="dummy.bench",
            enable_ai_activation=True,
        )

        self.assertTrue(result)
        self.assertEqual(mock_solver_instance.solve.call_count, 5)
        # Should call mogu_podem_wrapper for the clean retry
        mock_podem.assert_called_once()

    @patch("src.atpg.ai_podem.ModelPairPredictor")
    @patch("src.atpg.ai_podem.HierarchicalReconvSolver")
    @patch("src.atpg.ai_podem.mogu_podem_wrapper")
    def test_ai_propagation_only(self, mock_podem, mock_solver_cls, mock_predictor_cls):
        """Test enablement of AI Propagation without Activation."""
        circuit, total_gates = create_mock_circuit()
        fault = Fault(3, LogicValue.D)

        mock_podem.return_value = True

        # Disable AI Activation, Enable AI Propagation
        ai_podem(
            circuit,
            fault,
            total_gates,
            circuit_path="dummy.bench",
            enable_ai_activation=False,
            enable_ai_propagation=True,
        )

        # Solver should NOT be called for activation constraint (step 1)
        mock_solver_instance = mock_solver_cls.return_value
        mock_solver_instance.solve.assert_not_called()

        # But 'mogu_podem_wrapper' should be called with a backtrace_func
        args, kwargs = mock_podem.call_args
        self.assertIsNotNone(kwargs.get("backtrace_func"))
        self.assertIsInstance(kwargs.get("backtrace_func"), AIBacktracer)
        print("\nTest AI Prop Only: OK")

    @patch("src.atpg.ai_podem.ModelPairPredictor")
    @patch("src.atpg.ai_podem.HierarchicalReconvSolver")
    @patch("src.atpg.ai_podem.mogu_podem_wrapper")
    def test_ai_propagation_only_respects_max_backtracks(
        self, mock_podem, mock_solver_cls, mock_predictor_cls
    ):
        """Propagation-only PODEM must honor the caller's backtrack budget."""
        circuit, total_gates = create_mock_circuit()
        fault = Fault(3, LogicValue.D)

        mock_podem.return_value = True

        ai_podem(
            circuit,
            fault,
            total_gates,
            circuit_path="dummy.bench",
            enable_ai_activation=False,
            enable_ai_propagation=True,
            max_backtracks=12345,
        )

        _, kwargs = mock_podem.call_args
        self.assertEqual(kwargs.get("max_backtracks"), 12345)

    @patch("src.atpg.ai_podem.ModelPairPredictor")
    @patch("src.atpg.ai_podem.HierarchicalReconvSolver")
    @patch("src.atpg.ai_podem.mogu_podem_wrapper")
    def test_ai_activation_clean_retry_respects_max_backtracks(
        self, mock_podem, mock_solver_cls, mock_predictor_cls
    ):
        """Clean fallback retry must honor the caller's backtrack budget."""
        circuit, total_gates = create_mock_circuit()
        fault = Fault(3, LogicValue.D)
        mock_solver_cls.return_value.solve.return_value = {1: LogicValue.ONE}
        mock_podem.side_effect = [False, True]

        result = ai_podem(
            circuit,
            fault,
            total_gates,
            circuit_path="dummy.bench",
            enable_ai_activation=True,
            enable_ai_propagation=False,
            max_backtracks=12345,
        )

        self.assertTrue(result)
        self.assertEqual(mock_podem.call_count, 2)
        for call in mock_podem.call_args_list:
            self.assertEqual(call.kwargs.get("max_backtracks"), 12345)

    @patch("src.atpg.ai_podem.ModelPairPredictor")
    @patch("src.atpg.ai_podem.HierarchicalReconvSolver")
    @patch("src.atpg.ai_podem.mogu_podem_wrapper")
    def test_ai_propagation_only_treats_backtrack_limit_as_failure(
        self, mock_podem, mock_solver_cls, mock_predictor_cls
    ):
        """Truthiness of PODEM status codes must not create false detections."""
        circuit, total_gates = create_mock_circuit()
        fault = Fault(3, LogicValue.D)

        mock_podem.return_value = BACKTRACK_LIMIT

        result = ai_podem(
            circuit,
            fault,
            total_gates,
            circuit_path="dummy.bench",
            enable_ai_activation=False,
            enable_ai_propagation=True,
        )

        self.assertFalse(result)

    @patch("src.atpg.ai_podem.ModelPairPredictor")
    @patch("src.atpg.ai_podem.HierarchicalReconvSolver")
    @patch("src.atpg.ai_podem.mogu_podem_wrapper")
    def test_ai_activation_backtrack_limit_uses_clean_retry(
        self, mock_podem, mock_solver_cls, mock_predictor_cls
    ):
        """AI activation failures with non-success PODEM status should retry clean PODEM."""
        circuit, total_gates = create_mock_circuit()
        fault = Fault(3, LogicValue.D)
        mock_solver_cls.return_value.solve.return_value = {1: LogicValue.ONE}
        mock_podem.side_effect = [BACKTRACK_LIMIT, True]

        result = ai_podem(
            circuit,
            fault,
            total_gates,
            circuit_path="dummy.bench",
            enable_ai_activation=True,
            enable_ai_propagation=False,
        )

        self.assertTrue(result)
        self.assertEqual(mock_podem.call_count, 2)

    @patch("src.atpg.ai_podem.ModelPairPredictor")
    @patch("src.atpg.ai_podem.HierarchicalReconvSolver")
    @patch("src.atpg.ai_podem.mogu_podem_wrapper")
    def test_ai_backtracer_receives_no_fallback(
        self, mock_podem, mock_solver_cls, mock_predictor_cls
    ):
        """No-fallback mode must propagate into AI backtrace during PODEM."""
        circuit, total_gates = create_mock_circuit()
        fault = Fault(3, LogicValue.D)

        mock_podem.return_value = False

        ai_podem(
            circuit,
            fault,
            total_gates,
            circuit_path="dummy.bench",
            enable_ai_activation=False,
            enable_ai_propagation=True,
            no_fallback=True,
        )

        _, kwargs = mock_podem.call_args
        self.assertTrue(kwargs["backtrace_func"].no_fallback)

    def test_ai_backtracer_logic(self):
        """Test the AIBacktracer __call__ logic."""
        circuit, total_gates = create_mock_circuit()
        solver = MagicMock()
        solver.circuit = circuit

        backtracer = AIBacktracer(solver)

        # Case 1: AI Solver finds assignment for Gate 1=1 to satisfy objective
        solver.solve.return_value = {1: LogicValue.ONE}

        objective = Fault(3, LogicValue.ONE)
        res = backtracer(objective, circuit)

        self.assertEqual(res.gate_id, 1)
        self.assertEqual(res.value, LogicValue.ONE)

        # Case 2: AI Solver fails -> Fallback to simple
        solver.solve.return_value = None
        # Should fallback to simple_backtrace.
        # simple_backtrace(obj=Gate3, val=1) -> Gate 3 is AND. Needs 1,1.
        # It picks an X input (Gate 1 or 2).
        res = backtracer(objective, circuit)
        self.assertIn(res.gate_id, [1, 2])
        print("\nTest AI Backtracer Logic: OK")


class TestAIPodemBenchmark(unittest.TestCase):
    def test_vanilla_benchmark_counts_only_success_and_accumulates_stats(self):
        """Benchmark coverage and counters must reflect per-fault PODEM result codes."""
        faults = [Fault(idx, LogicValue.D) for idx in range(1, 5)]
        per_fault_stats = [
            {"backtrack_count": 1, "backtrace_count": 2, "total_recursive_calls": 3},
            {"backtrack_count": 3, "backtrace_count": 4, "total_recursive_calls": 5},
            {"backtrack_count": 5, "backtrace_count": 6, "total_recursive_calls": 7},
            {"backtrack_count": 7, "backtrace_count": 8, "total_recursive_calls": 9},
        ]

        with (
            patch.object(benchmark_ai_podem, "parse_bench_file", return_value=([None], 0)),
            patch.object(benchmark_ai_podem, "get_all_faults", return_value=faults),
            patch.object(benchmark_ai_podem, "reset_gates"),
            patch.object(benchmark_ai_podem, "initialize"),
            patch.object(benchmark_ai_podem, "reset_statistics"),
            patch.object(
                benchmark_ai_podem,
                "podem",
                side_effect=[SUCCESS, TIMEOUT, BACKTRACK_LIMIT, False],
            ),
            patch.object(benchmark_ai_podem, "get_statistics", side_effect=per_fault_stats),
        ):
            result = benchmark_ai_podem.run_benchmark(
                "data/bench/ISCAS85/c17.bench",
                "checkpoints/missing.pt",
                mode="vanilla",
            )

        self.assertEqual(result["faults"], 4)
        self.assertEqual(result["detected"], 1)
        self.assertEqual(result["coverage"], 25.0)
        self.assertEqual(result["backtracks"], 16)
        self.assertEqual(result["backtraces"], 20)


if __name__ == "__main__":
    unittest.main()
