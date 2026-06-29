import os
import sys

# Add src to path
sys.path.append(os.getcwd())

import unittest

from src.atpg.reconv_podem import PathConsistencySolver
from src.atpg.recursive_reconv_solver import (
    HierarchicalReconvSolver,
    ReconvPairPredictor,
)
from src.util.struct import Gate, GateType, LogicValue


class MockPredictor(ReconvPairPredictor):
    def __init__(self, circuit, verbose=False):
        self.circuit = circuit
        self.solver = PathConsistencySolver(circuit)
        self.verbose = verbose
        self.call_log = []

    def predict(self, pair_info, constraints, seed=None):
        if self.verbose:
            print(
                f"Predict called for {pair_info['start']}->{pair_info['reconv']} "
                f"with constraints {constraints}"
            )

        self.call_log.append(pair_info)

        # Use the actual solver to find ALL valid assignments for the reconvergent node
        # But wait, PathConsistencySolver takes a TARGET value for R.
        # HierarchicalReconvSolver doesn't pass a target value for R in Pair!
        # It relies on constraints to implicitly set R or S.

        # If R is constrained, we solve for that value.
        # If R is NOT constrained, we might need to try both 0 and 1?
        # Or does the Recursion enforce R later?
        # "Actual podem flow will start from a single gate... verified backwards"
        # The target_node given to solve() is the root R. So R is constrained initially.
        # For intermediate pairs, they must be part of the path to R.

        reconv_node = pair_info["reconv"]
        target_val = constraints.get(reconv_node, None)

        candidates = []

        possible_targets = (
            [target_val] if target_val is not None else [LogicValue.ZERO, LogicValue.ONE]
        )

        # Try to find MULTIPLE candidates by forcing the start node to different values
        # This simulates a model returning a distribution of options.
        start_node = pair_info["start"]
        possible_start_vals = [LogicValue.ZERO, LogicValue.ONE]

        # If start is already constrained, respect it
        if start_node in constraints:
            possible_start_vals = [constraints[start_node]]

        seen_solutions = []

        for s_val in possible_start_vals:
            # Force start value in constraints passed to solver
            # But we must not mutate the original constraints dict permanently for
            # the loop
            temp_constraints = constraints.copy()
            temp_constraints[start_node] = s_val

            for t_val in possible_targets:
                res = self.solver.solve(pair_info, t_val, temp_constraints)
                if res:
                    # Deduplicate based on content
                    if res not in seen_solutions:
                        candidates.append(res)
                        seen_solutions.append(res)

        return candidates


class TestRecursiveSolver(unittest.TestCase):
    def setUp(self):
        # Construct a simple diamond circuit
        # S(1) -> A(2), B(3) -> R(4) (AND)
        self.circuit = [None] * 5
        self.circuit[0] = Gate("dummy", GateType.INPT, 0, 0)  # 0
        self.circuit[1] = Gate("S", GateType.INPT, 0, 2)  # 1
        self.circuit[2] = Gate("A", GateType.BUFF, 1, 1)  # 2
        self.circuit[3] = Gate("B", GateType.NOT, 1, 1)  # 3
        self.circuit[4] = Gate("R", GateType.AND, 2, 0)  # 4

        # Connections
        self.circuit[1].fot = [2, 3]

        self.circuit[2].fin = [1]
        self.circuit[2].fot = [4]

        self.circuit[3].fin = [1]
        self.circuit[3].fot = [4]

        self.circuit[4].fin = [2, 3]

    def test_solve_simple_diamond(self):
        # S=0 -> A=0, B=1 -> R=0 (AND) -> Valid
        # S=1 -> A=1, B=0 -> R=0 (AND) -> Valid
        # Target: R=1 -> IMPOSSIBLE (A=1 and B=0 -> R=0)

        predictor = MockPredictor(self.circuit, verbose=True)
        solver = HierarchicalReconvSolver(self.circuit, predictor)

        # Case 1: Target R=0
        print("\n--- Test R=0 ---")
        assignment = solver.solve(4, LogicValue.ZERO)
        print("Assignment:", assignment)
        self.assertIsNotNone(assignment)
        self.assertEqual(assignment[4], LogicValue.ZERO)
        # Check inputs: if S is assigned, check logic
        if 1 in assignment:
            assignment[1]
            # S=0 or S=1 are both valid for R=0
            pass

        # Case 2: Target R=1 (Impossible)
        print("\n--- Test R=1 ---")
        assignment = solver.solve(4, LogicValue.ONE)
        print("Assignment:", assignment)
        self.assertIsNone(assignment)

    def test_nested_diamond(self):
        # S1(1) -> [S2(2), A(3)]
        # S2(2) -> [B(4), C(5)] -> R1(6) (OR)
        # R1(6) -> D(7)
        # A(3) -> E(8)
        # D(7), E(8) -> R2(9) (AND)

        # Pairs:
        # Inner: S2 -> R1 (Short)
        # Outer: S1 -> R2 (Long)

        circuit = [None] * 10
        circuit[0] = Gate("dummy", GateType.INPT)
        circuit[1] = Gate("S1", GateType.INPT, 0, 2)
        circuit[2] = Gate("S2", GateType.BUFF, 1, 2)  # S2 fed by S1
        circuit[3] = Gate("A", GateType.BUFF, 1, 1)  # A fed by S1

        circuit[4] = Gate("B", GateType.BUFF, 1, 1)  # B fed by S2
        circuit[5] = Gate("C", GateType.NOT, 1, 1)  # C fed by S2
        circuit[6] = Gate("R1", GateType.OR, 2, 1)  # R1 fed by B, C

        circuit[7] = Gate("D", GateType.BUFF, 1, 1)  # D fed by R1
        circuit[8] = Gate("E", GateType.BUFF, 1, 1)  # E fed by A
        circuit[9] = Gate("R2", GateType.AND, 2, 0)  # R2 fed by D, E

        # Connections
        circuit[1].fot = [2, 3]

        circuit[2].fin = [1]
        circuit[2].fot = [4, 5]
        circuit[3].fin = [1]
        circuit[3].fot = [8]

        circuit[4].fin = [2]
        circuit[4].fot = [6]
        circuit[5].fin = [2]
        circuit[5].fot = [6]

        circuit[6].fin = [4, 5]
        circuit[6].fot = [7]

        circuit[7].fin = [6]
        circuit[7].fot = [9]
        circuit[8].fin = [3]
        circuit[8].fot = [9]

        circuit[9].fin = [7, 8]

        # Logic:
        # S2 = S1
        # R1 = B | C = S2 | ~S2 = 1 (Always 1)
        # D = R1 = 1
        # E = A = S1
        # R2 = D & E = 1 & S1 = S1

        # So R2=1 requires S1=1.
        # R2=0 requires S1=0.

        # Also, R1 should always be 1.

        predictor = MockPredictor(circuit, verbose=True)
        solver = HierarchicalReconvSolver(circuit, predictor)

        print("\n--- Test Nested R2=1 ---")
        assignment = solver.solve(9, LogicValue.ONE)

        # The solver starts from the root (R2=9) and resolves the outer pair
        # (S1->R2) first since R2 is the initial queue entry.  The inner pair
        # (S2->R1) is solved during the subsequent recursion once S1 is committed.
        # Both pairs must appear somewhere in the call log.

        print("Call Log:", [(p["start"], p["reconv"]) for p in predictor.call_log])

        call_starts = [p["start"] for p in predictor.call_log]
        call_reconvs = [p["reconv"] for p in predictor.call_log]

        # Outer pair S1->R2 must appear
        self.assertIn(1, call_starts)
        self.assertIn(9, call_reconvs)

        # Inner pair S2->R1 must also appear (solved during recursion)
        self.assertIn(2, call_starts)
        self.assertIn(6, call_reconvs)

        self.assertIsNotNone(assignment)
        self.assertEqual(assignment[9], LogicValue.ONE)
        self.assertEqual(assignment[1], LogicValue.ONE)


if __name__ == "__main__":
    unittest.main()
