"""Reconvergent PODEM utilities.

This module currently focuses on finding reconvergent fanout structures using a
beam search over the circuit graph. A reconvergent structure is characterized by
two (or more) distinct branches leaving a start node S that eventually feed into
the same downstream node R. Identifying such structures is useful for ATPG
heuristics (e.g., selecting candidate objectives that increase observability or
expose difficult to control lines.)

Circuit representation:
    Each element in `circuit` is a `Gate` (see src.util.struct). Relevant fields:
        - name: string identifier (often its numeric id as string)
        - fin: list[int]  fanin node indices
        - fot: list[int]  fanout node indices
        - nfo: int        number of fanouts (len(fot))

Beam search strategy:
    For a start node S with >=2 fanouts, we spawn an initial path per direct
    fanout (S, fo). We then iteratively expand a frontier of paths while keeping
    only the top K (beam_width) according to a heuristic score designed to bias
    towards nodes that themselves branch and/or overlap with original fanouts.
    We record for each reached node the set of distinct first-branch fanouts of
    S that led to it. When a node R is reached via >=2 distinct first branches
    we report a reconvergent structure.

Heuristic score (higher is better):
    score = (branching_factor * 2) + overlap
    where branching_factor = nfo of current path end
          overlap = |fot(end) ∩ initial_fanouts(S)|

Returned structure:
    {
        'start': S,
        'reconv': R,
        'branches': [b1, b2, ...],  # first-level fanouts of S that reconverged
        'paths': {b1: [...], b2: [...], ...} # complete node id paths from S to R
    }
"""

from __future__ import annotations

import collections
import os
import pickle
import random
import re
import shutil
import signal
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ml.data.embedding import EmbeddingExtractor
from src.util.io import parse_bench_file
from src.util.struct import Gate, GateType, LogicValue

MAX_RECONV_PATH_COUNT = 1000
# NOTE: Retained for compatibility with other modules (e.g., RL trainers).
# Dataset generation below no longer uses logic values, but other codepaths may
# still rely on this table.
FANIN_LUT = {
    GateType.AND: {
        LogicValue.ZERO: [LogicValue.ZERO, LogicValue.ONE],
        LogicValue.ONE: [LogicValue.ONE],
    },
    GateType.NAND: {
        LogicValue.ZERO: [LogicValue.ONE],
        LogicValue.ONE: [LogicValue.ZERO, LogicValue.ONE],
    },
    GateType.OR: {
        LogicValue.ZERO: [LogicValue.ZERO],
        LogicValue.ONE: [LogicValue.ZERO, LogicValue.ONE],
    },
    GateType.NOR: {
        LogicValue.ZERO: [LogicValue.ONE],
        LogicValue.ONE: [LogicValue.ZERO, LogicValue.ONE],
    },
    GateType.XOR: {
        LogicValue.ZERO: [LogicValue.ZERO, LogicValue.ONE],
        LogicValue.ONE: [LogicValue.ZERO, LogicValue.ONE],
    },
    GateType.XNOR: {
        LogicValue.ZERO: [LogicValue.ZERO, LogicValue.ONE],
        LogicValue.ONE: [LogicValue.ZERO, LogicValue.ONE],
    },
    GateType.BUFF: {
        LogicValue.ZERO: [LogicValue.ZERO],
        LogicValue.ONE: [LogicValue.ONE],
    },
    GateType.NOT: {
        LogicValue.ZERO: [LogicValue.ONE],
        LogicValue.ONE: [LogicValue.ZERO],
    },
}


class PathConsistencySolver:
    """Determines if a specific logic value at the Reconvergent Node (R) is possible.

    This solver checks if there exists an assignment to the gates on the reconvergent
    paths (and their side-inputs) such that the Start Node (S) has a consistent
    value and the Reconvergent Node (R) evaluates to a target value (0 or 1).
    """

    def __init__(self, circuit: List[Gate]):
        self.circuit = circuit
        self.total_gates = len(circuit) - 1

    def solve(
        self,
        pair_info: Dict[str, Any],
        target_val: LogicValue,
        constraints: Dict[int, LogicValue] = None,
        timeout: float = 200.0,
        max_steps: int = 500000,
    ) -> Optional[Dict[int, LogicValue]]:
        """
        Determine if target_val at reconv_node is possible.
        Returns assignment if possible, else None.
        """
        self.start_time = time.time()
        self.timeout = timeout
        self.max_steps = max_steps
        self.step_count = 0

        start_node = pair_info["start"]
        reconv_node = pair_info["reconv"]
        paths = pair_info["paths"]

        # Maamari: Calculate LRR and Exit Lines
        lrr_nodes = get_lrr(self.circuit, start_node, reconv_node)
        exit_lines = identify_exit_lines(self.circuit, lrr_nodes)

        # Identify all gates on the paths (excluding S and R for now, or including?)
        # We need to know which gates are "on path" to distinguish side inputs.
        path_gates = set()
        for p in paths:
            path_gates.update(p)

        # Iterate over possible values for Start Node (0 and 1)
        # Unless Start Node is constrained!
        possible_s_vals = [LogicValue.ZERO, LogicValue.ONE]
        if constraints and start_node in constraints:
            possible_s_vals = [constraints[start_node]]

        for s_val in possible_s_vals:
            # We now pass LRR info to the solver helper
            assignment = self._try_solve_for_s(
                start_node,
                s_val,
                reconv_node,
                target_val,
                paths,
                path_gates,
                constraints=constraints,
                lrr_nodes=lrr_nodes,
                exit_lines=exit_lines,
            )
            if assignment:
                return assignment

        if self.step_count > self.max_steps or (time.time() - self.start_time > self.timeout):
            print(
                f"[RECONV-SOLVER] Warning: Solve timed out or hit step limit "
                f"({self.step_count} steps) for start={start_node} "
                f"reconv={reconv_node}"
            )

        return None

    def _try_solve_for_s(
        self,
        start_node: int,
        s_val: LogicValue,
        reconv_node: int,
        target_val: LogicValue,
        paths: List[List[int]],
        path_gates: Set[int],
        constraints: Dict[int, LogicValue] = None,
        lrr_nodes: Set[int] = None,
        exit_lines: List[Tuple[int, int]] = None,
    ) -> Optional[Dict[int, LogicValue]]:
        """Try to find a consistent assignment given S=s_val."""

        possible_values: Dict[int, Set[LogicValue]] = {}
        possible_values[start_node] = {s_val}

        # Initialize constraints in possible_values
        if constraints:
            for node, val in constraints.items():
                if node in path_gates:
                    possible_values[node] = {val}

        sorted_gates = sorted(list(path_gates))

        for gate_idx in sorted_gates:
            if gate_idx == start_node:
                continue

            gate = self.circuit[gate_idx]

            # If gate is constrained, we must respect it
            is_constrained = constraints and gate_idx in constraints
            constrained_val = constraints[gate_idx] if is_constrained else None

            input_possibilities = []
            for fin in gate.fin:
                if fin in path_gates:
                    if fin not in possible_values:
                        return None
                    input_possibilities.append(possible_values[fin])
                else:
                    input_possibilities.append({LogicValue.ZERO, LogicValue.ONE})

            if not input_possibilities and not gate.fin:
                # PI or Constant? But in path_gates?
                # If PI is in path, it should be start_node or side input?
                # If start_node, skipped.
                # If PI is side-input, it's not in sorted_gates loop (path_gates
                # usually doesn't include side PIs unless they are start)
                pass

            # print(f"DEBUG: Gate {gate_idx} input_possibilities: "
            # f"{input_possibilities}")
            can_produce_0 = self._can_gate_produce(gate.type, input_possibilities, LogicValue.ZERO)
            can_produce_1 = self._can_gate_produce(gate.type, input_possibilities, LogicValue.ONE)

            res_set = set()
            if can_produce_0:
                res_set.add(LogicValue.ZERO)
            if can_produce_1:
                res_set.add(LogicValue.ONE)

            if is_constrained:
                if constrained_val not in res_set:
                    # Conflict with constraint
                    return None
                res_set = {constrained_val}

            if not res_set:
                return None

            possible_values[gate_idx] = res_set

        if target_val not in possible_values.get(reconv_node, set()):
            return None

        final_assignment = {}
        # Pre-fill constraints into assignment?
        # No, backtrace will handle it, but we should ensure consistency check
        # respects them.
        # Actually, _backtrace_assignment fills `assignment`.
        # We can pass constraints to it or just rely on `possible_values` which are
        # already restricted.
        # However, side inputs might be constrained too!
        # If a side input is constrained, we must respect it.
        # `possible_values` only tracks path gates.
        # So we need to handle side input constraints in `_backtrace_assignment`.

        if constraints:
            final_assignment.update(constraints)

        # Optimize exit lines lookup
        exit_map = None
        if exit_lines:
            exit_map = collections.defaultdict(list)
            for src, dst in exit_lines:
                exit_map[src].append(dst)

        if self._backtrace_assignment(
            reconv_node,
            target_val,
            possible_values,
            path_gates,
            final_assignment,
            constraints,
            lrr_nodes,
            exit_map,
        ):
            return final_assignment

        return None

    def _can_gate_produce(
        self, gtype: int, input_possibilities: List[Set[LogicValue]], target: LogicValue
    ) -> bool:
        """Check if a gate can produce `target` given input possibilities."""
        # Optimization for common gates
        if gtype == GateType.BUFF:
            return target in input_possibilities[0]
        if gtype == GateType.NOT:
            prev = LogicValue.ZERO if target == LogicValue.ONE else LogicValue.ONE
            return prev in input_possibilities[0]

        if gtype == GateType.AND:
            if target == LogicValue.ONE:
                return all(LogicValue.ONE in s for s in input_possibilities)
            else:  # target 0
                return any(LogicValue.ZERO in s for s in input_possibilities)

        if gtype == GateType.NAND:
            if target == LogicValue.ZERO:  # Output 0 -> AND is 1 -> all inputs 1
                return all(LogicValue.ONE in s for s in input_possibilities)
            else:  # target 1 -> AND is 0 -> any input 0
                return any(LogicValue.ZERO in s for s in input_possibilities)

        if gtype == GateType.OR:
            if target == LogicValue.ZERO:  # All inputs 0
                return all(LogicValue.ZERO in s for s in input_possibilities)
            else:  # Any input 1
                return any(LogicValue.ONE in s for s in input_possibilities)

        if gtype == GateType.NOR:
            if target == LogicValue.ONE:  # Output 1 -> OR is 0 -> all inputs 0
                return all(LogicValue.ZERO in s for s in input_possibilities)
            else:  # Output 0 -> OR is 1 -> any input 1
                return any(LogicValue.ONE in s for s in input_possibilities)

        flexible_count = sum(1 for s in input_possibilities if len(s) > 1)
        if flexible_count > 0:
            return True

        fixed_inputs = [list(s)[0] for s in input_possibilities]
        val = self._compute_gate(gtype, fixed_inputs)
        return val == target

    def _compute_gate(self, gtype: int, inputs: List[LogicValue]) -> LogicValue:
        # Helper to compute gate logic for fixed inputs
        # Using the LUTs or logic
        if gtype == GateType.AND:
            return LogicValue.ONE if all(i == LogicValue.ONE for i in inputs) else LogicValue.ZERO
        elif gtype == GateType.NAND:
            return LogicValue.ZERO if all(i == LogicValue.ONE for i in inputs) else LogicValue.ONE
        elif gtype == GateType.OR:
            return LogicValue.ONE if any(i == LogicValue.ONE for i in inputs) else LogicValue.ZERO
        elif gtype == GateType.NOR:
            return LogicValue.ZERO if any(i == LogicValue.ONE for i in inputs) else LogicValue.ONE
        elif gtype == GateType.XOR:
            # Count 1s
            ones = sum(1 for i in inputs if i == LogicValue.ONE)
            return LogicValue.ONE if ones % 2 == 1 else LogicValue.ZERO
        elif gtype == GateType.XNOR:
            ones = sum(1 for i in inputs if i == LogicValue.ONE)
            return LogicValue.ONE if ones % 2 == 0 else LogicValue.ZERO
        elif gtype == GateType.BUFF:
            return inputs[0]
        elif gtype == GateType.NOT:
            return LogicValue.ZERO if inputs[0] == LogicValue.ONE else LogicValue.ONE
        return LogicValue.XD

    def _backtrace_assignment(
        self,
        gate_idx: int,
        target_val: LogicValue,
        possible_values: Dict[int, Set[LogicValue]],
        path_gates: Set[int],
        assignment: Dict[int, LogicValue],
        constraints: Dict[int, LogicValue] = None,
        lrr_nodes: Set[int] = None,
        exit_map: Dict[int, List[int]] = None,
    ) -> bool:
        """Recursively fix values to achieve target_val."""
        self.step_count += 1
        if self.step_count > self.max_steps:
            return False
        if (self.step_count % 100 == 0) and (time.time() - self.start_time > self.timeout):
            return False

        # If already assigned, check consistency
        if gate_idx in assignment:
            return assignment[gate_idx] == target_val

        # Check immediate consistency with neighbors before assigning
        if not self._check_consistency(gate_idx, target_val, assignment):
            return False

        # Maamari: Regional Consistency Check
        # If gate is in LRR, assigning it might imply values on Exit Lines.
        # If those Exit Lines are constrained, we must check for conflict.
        if lrr_nodes and gate_idx in lrr_nodes and constraints and exit_map:
            if gate_idx in exit_map:
                for dst in exit_map[gate_idx]:
                    # This gate drives an exit line to dst.
                    # If dst is constrained, does src=target_val conflict?
                    if dst in constraints:
                        dst_req = constraints[dst]
                        dst_gate = self.circuit[dst]
                        # Check if dst(..., src=target_val, ...) CAN produce dst_req
                        # simplified check for single known input

                        inp_poss = []
                        found_conn = False
                        for fin in dst_gate.fin:
                            if fin == gate_idx:
                                inp_poss.append({target_val})
                                found_conn = True
                            else:
                                inp_poss.append({LogicValue.ZERO, LogicValue.ONE})

                        if found_conn:
                            if not self._can_gate_produce(dst_gate.type, inp_poss, dst_req):
                                # Conflict detected at Exit Line!
                                return False

        assignment[gate_idx] = target_val

        gate = self.circuit[gate_idx]

        # Leaf node (e.g. INPT) - if we assigned it, we are done
        if not gate.fin:
            return True

        # We need to pick input values that produce target_val
        # and are consistent with possible_values[fin] (if fin is on path)
        # or are just {0,1} (if fin is side input).

        # Get candidates for each input
        input_candidates = []
        for fin in gate.fin:
            if fin in path_gates:
                input_candidates.append(list(possible_values[fin]))
            else:
                # Side input
                # If constrained, respect it
                if constraints and fin in constraints:
                    input_candidates.append([constraints[fin]])
                else:
                    input_candidates.append([LogicValue.ZERO, LogicValue.ONE])

        # Find a combination of inputs that works
        import itertools

        # Guard against exponential blowup for gates with many inputs
        total_combos = 1
        for cand in input_candidates:
            total_combos *= len(cand)
            if total_combos > 1024:  # Hard limit for a single gate's breadth
                return False

        for input_combo in itertools.product(*input_candidates):
            # Check if this combo produces target_val
            if self._compute_gate(gate.type, input_combo) == target_val:
                # Try to recursively satisfy this combo

                # Let's implement the recursion with undo
                snapshot = assignment.copy()
                valid_combo = True

                for i, fin in enumerate(gate.fin):
                    val = input_combo[i]
                    if fin in path_gates:
                        if not self._backtrace_assignment(
                            fin,
                            val,
                            possible_values,
                            path_gates,
                            assignment,
                            constraints,
                            lrr_nodes,
                            exit_map,
                        ):
                            valid_combo = False
                            break
                    else:
                        # Side input
                        # Check consistency before assigning
                        if fin in assignment:
                            if assignment[fin] != val:
                                valid_combo = False
                                break
                        else:
                            if not self._check_consistency(fin, val, assignment):
                                valid_combo = False
                                break
                            assignment[fin] = val

                if valid_combo:
                    return True
                else:
                    # Restore
                    assignment.clear()
                    assignment.update(snapshot)

        return False

    def _check_consistency(
        self, gate_idx: int, val: LogicValue, assignment: Dict[int, LogicValue]
    ) -> bool:
        """Check if assigning gate_idx=val contradicts existing assignment."""
        gate = self.circuit[gate_idx]

        # 1. Check consistency with inputs (if inputs are assigned)
        # If enough inputs are assigned to force a value, check if it matches val.
        # We can reuse _compute_gate but we need to handle unassigned inputs (XD).
        # Since _compute_gate expects LogicValue, we can pass XD.
        # But _compute_gate might not handle XD correctly for all gates?
        # Let's assume _compute_gate handles XD or we assume unassigned are XD.
        # Actually, _compute_gate in this class assumes 0/1 inputs for the backtrace.
        # We need a robust compute that handles XD.
        # Let's just implement a simple check.

        input_vals = []
        any_input_assigned = False
        for fin in gate.fin:
            if fin in assignment:
                input_vals.append(assignment[fin])
                any_input_assigned = True
            else:
                input_vals.append(LogicValue.XD)

        if any_input_assigned:
            # We need a compute function that handles XD.
            # Since we don't have one exposed, let's implement a simple one
            # or rely on logic_sim_three?
            # logic_sim_three is not available here easily.
            # Let's implement a simple check.
            computed = self._compute_gate_robust(gate.type, input_vals)
            if computed != LogicValue.XD and computed != val:
                return False

        # 2. Check consistency with outputs (fanout)
        # If a fanout is assigned, check if its value is consistent with this
        # gate being val.
        for fout in gate.fot:
            if fout in assignment:
                fout_gate = self.circuit[fout]
                fout_val = assignment[fout]
                # Get inputs of fout
                fout_inputs = []
                for fin in fout_gate.fin:
                    if fin == gate_idx:
                        fout_inputs.append(val)
                    elif fin in assignment:
                        fout_inputs.append(assignment[fin])
                    else:
                        fout_inputs.append(LogicValue.XD)

                computed = self._compute_gate_robust(fout_gate.type, fout_inputs)
                if computed != LogicValue.XD and computed != fout_val:
                    return False

        return True

    def _compute_gate_robust(self, gtype: int, inputs: List[LogicValue]) -> LogicValue:
        """Compute gate output handling XD."""
        if gtype == GateType.AND:
            if any(i == LogicValue.ZERO for i in inputs):
                return LogicValue.ZERO
            if all(i == LogicValue.ONE for i in inputs):
                return LogicValue.ONE
            return LogicValue.XD
        elif gtype == GateType.NAND:
            if any(i == LogicValue.ZERO for i in inputs):
                return LogicValue.ONE
            if all(i == LogicValue.ONE for i in inputs):
                return LogicValue.ZERO
            return LogicValue.XD
        elif gtype == GateType.OR:
            if any(i == LogicValue.ONE for i in inputs):
                return LogicValue.ONE
            if all(i == LogicValue.ZERO for i in inputs):
                return LogicValue.ZERO
            return LogicValue.XD
        elif gtype == GateType.NOR:
            if any(i == LogicValue.ONE for i in inputs):
                return LogicValue.ZERO
            if all(i == LogicValue.ZERO for i in inputs):
                return LogicValue.ONE
            return LogicValue.XD
        elif gtype == GateType.NOT:
            if inputs[0] == LogicValue.ONE:
                return LogicValue.ZERO
            if inputs[0] == LogicValue.ZERO:
                return LogicValue.ONE
            return LogicValue.XD
        elif gtype == GateType.BUFF:
            return inputs[0]
        elif gtype == GateType.XOR:
            if any(i == LogicValue.XD for i in inputs):
                return LogicValue.XD
            ones = sum(1 for i in inputs if i == LogicValue.ONE)
            return LogicValue.ONE if ones % 2 == 1 else LogicValue.ZERO
        elif gtype == GateType.XNOR:
            if any(i == LogicValue.XD for i in inputs):
                return LogicValue.XD
            ones = sum(1 for i in inputs if i == LogicValue.ONE)
            return LogicValue.ONE if ones % 2 == 0 else LogicValue.ZERO

        return LogicValue.XD


def check_path_pair_consistency(
    circuit: List[Gate],
    pair_info: Dict[str, Any],
    constraints: Dict[int, LogicValue] = None,
) -> Dict[int, Optional[Dict[int, LogicValue]]]:
    """
    Check consistency for both target 0 and 1 at reconvergent node.
    Returns dict {0: assignment_or_None, 1: assignment_or_None}
    """
    solver = PathConsistencySolver(circuit)
    results = {}

    for target in [LogicValue.ZERO, LogicValue.ONE]:
        results[target] = solver.solve(pair_info, target, constraints)

    return results


def get_lrr(circuit: List[Gate], start_node: int, reconv_node: int) -> Set[int]:
    """Identify the Local Reconvergent Region (LRR) Logic from Maamari & Rajski (1990).
    The LRR consists of all gates G such that:
      1. There is a path from S to G (S -> ... -> G)
      2. There is a path from G to R (G -> ... -> R)
    """
    # 1. Forward BFS from S
    reachable_from_s = set()
    queue = collections.deque([start_node])
    reachable_from_s.add(start_node)

    while queue:
        curr = queue.popleft()
        gate = circuit[curr]
        for fo in getattr(gate, "fot", []) or []:
            if fo not in reachable_from_s:
                reachable_from_s.add(fo)
                queue.append(fo)

    # 2. Backward BFS from R
    reaches_r = set()
    queue = collections.deque([reconv_node])
    reaches_r.add(reconv_node)

    while queue:
        curr = queue.popleft()
        gate = circuit[curr]
        for fin in gate.fin:
            if fin not in reaches_r:
                reaches_r.add(fin)
                queue.append(fin)

    # Intersection is the LRR
    return reachable_from_s & reaches_r


def identify_exit_lines(circuit: List[Gate], lrr_nodes: Set[int]) -> List[Tuple[int, int]]:
    """Identify Exit Lines for the LRR.
    An Exit Line is a connection from a gate INSIDE the LRR to a gate OUTSIDE the LRR.
    These represent leakage points where signal logic escapes the reconvergent loop.
    """
    exit_lines = []
    sorted_lrr = sorted(list(lrr_nodes))  # Deterministic order

    for gid in sorted_lrr:
        gate = circuit[gid]
        for fo in getattr(gate, "fot", []) or []:
            if fo not in lrr_nodes:
                exit_lines.append((gid, fo))

    return exit_lines


def reconv_podem(circuit_path: str, output_idx: int, desired_output: int):
    """Entry point invoking reconvergent fanout finder and consistency checker.

    Parameters
    ----------
    circuit_path : str
        Path to .bench file.
    output_idx : int
        Target output gate index (unused placeholder).
    desired_output : int
        Desired logic value at the output (unused placeholder).
    """
    circuit, _ = parse_bench_file(circuit_path)

    # Find one pair
    info_list = find_all_reconv_pairs(circuit, beam_width=16, max_depth=25, max_pairs=1)
    if not info_list:
        print("[ERROR] No reconvergent paths found in the circuit.")
        return None

    info = info_list[0]
    print(f"Found reconvergent pair: Start={info['start']}, Reconv={info['reconv']}")

    # Check consistency
    results = check_path_pair_consistency(circuit, info)

    print("Consistency Results:")
    print(f"  Target 0: {'Possible' if results[0] else 'Impossible'}")
    print(f"  Target 1: {'Possible' if results[1] else 'Impossible'}")

    return {**info, "consistency": results}


def find_shortest_reconv_pair_ending_at(
    circuit: List[Gate], reconv_node: int
) -> Optional[Dict[str, Any]]:
    """
    Find the shortest reconvergent path pair ending at `reconv_node` using backward BFS.
    Returns pair_info or None.
    """
    gate = circuit[reconv_node]
    if len(gate.fin) < 2:
        return None

    # Initialize BFS
    # We want to find a common ancestor S reachable from at least two distinct
    # inputs of R.
    # We propagate "Branch IDs" (indices of R's inputs) backwards.

    # Map: node_idx -> Set[branch_id]
    reachable_branches = {}
    queue = []

    for i, fin in enumerate(gate.fin):
        if fin not in reachable_branches:
            reachable_branches[fin] = set()
        reachable_branches[fin].add(i)
        queue.append(fin)

    visited = set(gate.fin)

    # To reconstruct paths, we can store predecessors: node -> branch_id -> pred_node
    # But simple BFS path reconstruction is easier if we just find S first.

    # Limit depth to avoid explosion
    MAX_DEPTH = 50
    depths = {n: 0 for n in gate.fin}

    import collections

    queue = collections.deque(gate.fin)

    while queue:
        curr = queue.popleft()

        if depths[curr] > MAX_DEPTH:
            continue

        # Check if this node is a stem (reachable from >1 branch)
        if len(reachable_branches[curr]) >= 2:
            break

        # Backtrace
        curr_gate = circuit[curr]
        for fin in curr_gate.fin:
            if fin not in reachable_branches:
                reachable_branches[fin] = set()

            # Add branches from current node
            original_len = len(reachable_branches[fin])
            reachable_branches[fin].update(reachable_branches[curr])

            if len(reachable_branches[fin]) > original_len:
                # If we added new info, process this node
                if fin not in visited:  # Or if we updated it? BFS usually visits once.
                    # But here we merge sets.
                    # Standard BFS: visit once.
                    # But we need to propagate set union.
                    # If we visit again, we might add more branches.
                    # But if we do strict BFS, the first time we see a node with
                    # >1 branch, it's the closest stem.
                    # Wait, if paths have different lengths, we might reach S
                    # from branch 0 at depth 5
                    # and from branch 1 at depth 10.
                    # We need to wait until we see >1 branch.
                    pass

            if fin not in visited:
                visited.add(fin)
                depths[fin] = depths[curr] + 1
                queue.append(fin)
            else:
                # If already visited, we still update its branches!
                # And if it now has >1 branch, it's a candidate.
                # But we processed it already?
                # If we use a queue, we might need to re-queue if set changes?
                # Or just check condition immediately.
                pass

        # Check again after update (if we allow re-visiting logic, which is complex)
        # Simplified approach:
        # Just standard BFS.
        # We track `sources` for each node.
        # If a node is reached from different initial branches, it's a stem.

    # Let's restart with a cleaner BFS
    return _find_shortest_reconv_helper(circuit, reconv_node)


def _find_shortest_reconv_helper(circuit: List[Gate], reconv_node: int) -> Optional[Dict[str, Any]]:
    gate = circuit[reconv_node]
    if len(gate.fin) < 2:
        return None

    # node -> set of starting branches (indices in gate.fin)
    node_branches = {fin: {i} for i, fin in enumerate(gate.fin)}
    queue = collections.deque(gate.fin)
    {fin: 1 for fin in gate.fin}  # How many times visited/updated?

    # To reconstruct paths: branch_idx -> node -> prev_node
    # This is hard.
    # Let's just find S, then find paths S->R.

    stem = -1

    # BFS for Stem
    seen = set(gate.fin)

    while queue:
        curr = queue.popleft()

        if len(node_branches[curr]) >= 2:
            stem = curr
            break

        curr_gate = circuit[curr]
        for fin in curr_gate.fin:
            if fin not in node_branches:
                node_branches[fin] = set()

            before = len(node_branches[fin])
            node_branches[fin].update(node_branches[curr])
            after = len(node_branches[fin])

            if after >= 2:
                stem = fin
                break

            if fin not in seen:
                seen.add(fin)
                queue.append(fin)
            elif after > before:
                # If we updated the set, we might need to propagate further?
                # For "shortest", maybe not strictly necessary if we just want ANY stem.
                # But to find the *closest* stem, strict BFS is good.
                # If we reach fin via branch 0, then later via branch 1, we merge.
                # If we don't re-queue, we stop.
                # But we check `after >= 2` immediately. So we catch it.
                pass

        if stem != -1:
            break

    if stem == -1:
        return None

    # Found Stem. Now find paths S -> R.
    # We need 2 disjoint paths (first branches disjoint).
    # We know which branches of R reach S (from node_branches[stem]).
    # Let's pick two branches, say b1 and b2.
    branches = list(node_branches[stem])
    if len(branches) < 2:
        return None

    b1_idx = branches[0]
    b2_idx = branches[1]

    # Find path from S to R via gate.fin[b1_idx]
    path1 = _find_path(circuit, stem, gate.fin[b1_idx])
    path1.append(reconv_node)  # Add R

    # Find path from S to R via gate.fin[b2_idx]
    path2 = _find_path(circuit, stem, gate.fin[b2_idx])
    path2.append(reconv_node)

    # Construct pair info
    # paths should exclude S and R?
    # The existing solver expects paths as list of gates.
    # Let's include intermediate gates.

    # path1: [S, ..., fin1, R]
    # We want [..., fin1] (excluding S and R?)
    # Solver logic:
    # path_gates.update(p)
    # _try_solve_for_s iterates sorted(path_gates).
    # If S is in path_gates, it's fine (it skips S in loop).
    # If R is in path_gates, it's fine.
    # Let's return full paths.

    return {"start": stem, "reconv": reconv_node, "paths": [path1, path2]}


def _find_path(circuit: List[Gate], start: int, end: int) -> List[int]:
    # BFS from start to end
    q = collections.deque([[start]])
    visited = {start}
    while q:
        path = q.popleft()
        curr = path[-1]
        if curr == end:
            return path

        for fout in circuit[curr].fot:
            if fout not in visited:
                visited.add(fout)
                new_path = list(path)
                new_path.append(fout)
                q.append(new_path)
    return []


def find_all_reconv_pairs(
    circuit: List[Any],
    beam_width: int = 16,
    max_depth: int = 25,
    max_pairs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Enumerate reconvergent fanout structures in the circuit.

    This function explores all start nodes with at least two fanouts and uses a
    beam search to find all nodes that can be reached from two distinct first
    fanout branches. For each reconvergent node found, it emits every unique
    pair of first-branch paths that reach it.

    Notes
    -----
    - This is exhaustive with respect to the chosen beam search parameters. It
      doesn't guarantee global exhaustiveness if pruning eliminates some paths.

    Parameters
    ----------
    circuit : list[Gate]
        Circuit gate list.
    beam_width : int
        Maximum frontier width per expansion.
    max_depth : int
        Maximum path expansion depth from a start node.
    max_pairs : int, optional
        Optional cap on the number of reconvergent pairs to return.

    Returns
    -------
    list[dict]
        A list of reconvergent structures. Each entry has keys:
        - 'start': start node id
        - 'reconv': reconvergent node id
        - 'branches': [b1, b2] the two first fanouts from start
        - 'paths': [path1, path2] node id sequences from start to reconv
    """
    results: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, int, Tuple[int, int]]] = set()  # (start, reconv, (b1,b2))

    node_ids = list(range(1, len(circuit)))  # skip index 0

    for s in node_ids:
        start_gate = circuit[s]
        fanouts: List[int] = getattr(start_gate, "fot", []) or []
        if len(fanouts) < 2:
            continue

        # Initialize one path per direct fanout.
        frontier: List[List[int]] = [[s, fo] for fo in fanouts]
        initial_fanouts = set(fanouts)

        # reached[node][first_branch] = path
        reached: Dict[int, Dict[int, List[int]]] = {}
        depth = 0

        while frontier and depth < max_depth:
            # Score current frontier and prune to beam width
            scored = []
            for path in frontier:
                last = path[-1]
                gate = circuit[last]
                branching = getattr(gate, "nfo", 0)
                last_fot = getattr(gate, "fot", []) or []
                overlap = len(set(last_fot) & initial_fanouts)
                score = branching * 2 + overlap
                scored.append((score, path))
            scored.sort(key=lambda x: x[0], reverse=True)
            frontier = [p for _, p in scored[:beam_width]]

            next_frontier: List[List[int]] = []
            for path in frontier:
                last = path[-1]
                if len(path) < 2:
                    continue
                first_branch = path[1]

                # Record arrival at 'last' via this first branch
                reached.setdefault(last, {})
                if first_branch not in reached[last]:
                    reached[last][first_branch] = path.copy()

                # If reconvergence at 'last', emit all unique branch pairs
                if len(reached[last]) >= 2 and last != s:
                    branches = list(reached[last].keys())
                    # generate all unique pairs
                    for i in range(len(branches)):
                        for j in range(i + 1, len(branches)):
                            b1, b2 = branches[i], branches[j]
                            ordered_pair: Tuple[int, int] = (b1, b2) if b1 < b2 else (b2, b1)
                            key = (s, last, ordered_pair)
                            if key in seen:
                                continue
                            seen.add(key)
                            paths = [reached[last][b1], reached[last][b2]]
                            results.append(
                                {
                                    "start": s,
                                    "reconv": last,
                                    "branches": [b1, b2],
                                    "paths": paths,
                                }
                            )
                            if max_pairs is not None and len(results) >= max_pairs:
                                return results

                # Expand path
                for fo in getattr(circuit[last], "fot", []) or []:
                    if fo in path:  # prevent simple cycles
                        continue
                    next_frontier.append(path + [fo])

            frontier = next_frontier
            depth += 1

    return results


def pick_reconv_pair(
    circuit: List[Any],
    beam_width: int = 8,
    max_depth: int = 20,
) -> Optional[Dict[str, Any]]:
    """Beam-search for a reconvergent fanout structure.

    A reconvergent structure exists if there is a start node S with at least
    two fanouts such that two (or more) distinct first fanout branches from S
    eventually reach a common node R (R != S).

    Parameters
    ----------
    circuit : list[Gate]
        Gate list (indexable by integer id) produced by parse_bench_file.
    beam_width : int
        Max number of frontier paths kept per expansion step.
    max_depth : int
        Maximum path length expansions (number of edge traversals) before
        abandoning a start node.

    Returns
    -------
    dict | None
        Reconvergent structure info or None if none found.
    """
    node_ids = list(range(1, len(circuit)))  # skip index 0 (often dummy)
    random.shuffle(node_ids)

    for s in node_ids:
        start_gate = circuit[s]
        fanouts: List[int] = getattr(start_gate, "fot", []) or []
        if len(fanouts) < 2:
            continue

        # Initialize one path per direct fanout.
        frontier: List[List[int]] = [[s, fo] for fo in fanouts]
        initial_fanouts = set(fanouts)

        # reached[node][first_branch] = path
        reached: Dict[int, Dict[int, List[int]]] = {}
        depth = 0

        while frontier and depth < max_depth:
            # Score current frontier and prune to beam width
            scored = []
            for path in frontier:
                last = path[-1]
                gate = circuit[last]
                branching = getattr(gate, "nfo", 0)
                last_fot = getattr(gate, "fot", []) or []
                overlap = len(set(last_fot) & initial_fanouts)
                score = branching * 2 + overlap
                scored.append((score, path))
            scored.sort(key=lambda x: x[0], reverse=True)
            frontier = [p for _, p in scored[:beam_width]]

            next_frontier: List[List[int]] = []
            for path in frontier:
                last = path[-1]
                if len(path) < 2:
                    continue  # should not happen
                first_branch = path[1]

                # Record arrival at 'last' via this first branch
                reached.setdefault(last, {})
                if first_branch not in reached[last]:
                    reached[last][first_branch] = path.copy()

                # Check reconvergence condition
                if len(reached[last]) >= 2 and last != s:
                    branches = list(reached[last].keys())
                    paths = [reached[last][b] for b in branches]
                    return {
                        "start": s,
                        "reconv": last,
                        "branches": branches,
                        "paths": paths,
                    }

                # Expand path
                for fo in getattr(circuit[last], "fot", []) or []:
                    if fo in path:  # prevent simple cycles
                        continue
                    next_frontier.append(path + [fo])

            frontier = next_frontier
            depth += 1

    return None


class RecursiveStructureSolver:
    def __init__(self, circuit: List[Gate]):
        self.circuit = circuit
        self.path_solver = PathConsistencySolver(circuit)

    def solve(self, target_node: int, target_val: LogicValue) -> Optional[Dict[int, LogicValue]]:
        """
        Recursively justify target_val at target_node.
        Returns assignment dict if successful, else None.
        """
        assignment = {}
        queue = [(target_node, target_val)]

        # Track processed nodes to avoid infinite loops or redundant work?
        # Just using assignment as visited set for values.

        while queue:
            node, val = queue.pop(0)

            # 1. Check consistency
            if node in assignment:
                if assignment[node] != val:
                    return None  # Conflict
                continue

            # 2. Check immediate consistency with existing assignment
            # (Simple check: if inputs are assigned, do they produce val?)
            # This is handled implicitly if we process in order, but we are
            # jumping around.
            # Let's just assign and verify later or trust the process.

            # Actually, we should check if this assignment contradicts any ALREADY
            # assigned neighbors.
            # Similar to _check_consistency in PathConsistencySolver.
            if not self._check_global_consistency(node, val, assignment):
                return None

            assignment[node] = val

            gate = self.circuit[node]
            if not gate.fin:  # PI
                continue

            # 3. Try to find a Reconvergent Pair ending at 'node'
            pair_info = find_shortest_reconv_pair_ending_at(self.circuit, node)

            if pair_info:
                # Solve the pair
                # We pass current assignment as constraints
                pair_res = self.path_solver.solve(pair_info, val, constraints=assignment)

                if pair_res:
                    # Verify pair_res is consistent with existing assignment
                    # before merging
                    conflict = False
                    for gid, gval in pair_res.items():
                        if gid in assignment and assignment[gid] != gval:
                            conflict = True
                            break
                        # Also check if this new assignment would conflict
                        # with neighbors
                        if gid not in assignment:
                            if not self._check_global_consistency(gid, gval, assignment):
                                conflict = True
                                break

                    if conflict:
                        # Path pair solution conflicts with existing
                        # assignments
                        return None

                    # Merge result
                    assignment.update(pair_res)

                    # Maamari Pruning: Logic Constraint
                    # Utilizing LRR boundary to prune justification queue.
                    start_node = pair_info["start"]
                    if start_node in pair_res:
                        queue.append((start_node, pair_res[start_node]))

                    path_gates = set()
                    for p in pair_info["paths"]:
                        path_gates.update(p)

                    for gid in pair_res:
                        if gid in path_gates:
                            # It's on path. Its inputs might be side inputs.
                            g = self.circuit[gid]
                            for fin in g.fin:
                                if fin not in path_gates:
                                    # Side input
                                    if fin in pair_res:
                                        # Strict pruning: only add if it's a
                                        # PI or contributes to LRR.
                                        # Since solve() is generic, we
                                        # shouldn't kill useful work.
                                        # But for LRR-focused solving, we
                                        # assume external constraints are
                                        # fixed or PIs.
                                        queue.append((fin, pair_res[fin]))

                    continue
                else:
                    return None

            # 4. No pair found, or pair logic not applicable (e.g. simple gate)
            # Standard Backtrace / Justification
            input_reqs = self._justify_gate(node, val, assignment)
            if input_reqs is None:
                return None

            for fin, fval in input_reqs.items():
                queue.append((fin, fval))

        return assignment

    def _justify_gate(
        self, node: int, val: LogicValue, assignment: Dict[int, LogicValue]
    ) -> Optional[Dict[int, LogicValue]]:
        """
        Determine required input values for 'node' to produce 'val'.
        Returns dict {input_idx: value} or None if impossible.
        """
        gate = self.circuit[node]

        # Identify unassigned inputs
        unassigned = [fin for fin in gate.fin if fin not in assignment]

        # First, verify existing assignments are consistent with target val
        if not unassigned:
            # All assigned. Check if output matches.
            input_vals = [assignment[fin] for fin in gate.fin]
            res = self._compute_simple(gate.type, input_vals)
            return {} if res == val else None

        # Check if already-assigned inputs force a particular output
        input_vals_partial = [assignment.get(fin, LogicValue.XD) for fin in gate.fin]
        computed = self._compute_gate_robust(gate.type, input_vals_partial)

        # If already forced to wrong value, fail
        if computed != LogicValue.XD and computed != val:
            return None

        # If already satisfied, no need to assign more
        if computed == val:
            return {}

        # Now pick values for unassigned inputs to produce val
        # Use smart heuristics for common gates
        reqs = {}

        if gate.type == GateType.AND:
            if val == LogicValue.ONE:
                # All must be 1
                for fin in unassigned:
                    reqs[fin] = LogicValue.ONE
            else:
                # One must be 0 (if not already 0)
                if any(assignment.get(fin) == LogicValue.ZERO for fin in gate.fin):
                    return {}  # Already satisfied
                # Pick one unassigned to be 0, verify it doesn't conflict

                for fin in unassigned:
                    if self._check_global_consistency(fin, LogicValue.ZERO, assignment):
                        reqs[fin] = LogicValue.ZERO
                        break
                else:
                    # No valid choice found
                    return None

        elif gate.type == GateType.OR:
            if val == LogicValue.ZERO:
                # All must be 0
                for fin in unassigned:
                    reqs[fin] = LogicValue.ZERO
            else:
                # One must be 1
                if any(assignment.get(fin) == LogicValue.ONE for fin in gate.fin):
                    return {}

                for fin in unassigned:
                    if self._check_global_consistency(fin, LogicValue.ONE, assignment):
                        reqs[fin] = LogicValue.ONE
                        break
                else:
                    return None

        elif gate.type == GateType.NAND:
            if val == LogicValue.ZERO:
                # All must be 1
                for fin in unassigned:
                    reqs[fin] = LogicValue.ONE
            else:
                # One must be 0
                if any(assignment.get(fin) == LogicValue.ZERO for fin in gate.fin):
                    return {}

                for fin in unassigned:
                    if self._check_global_consistency(fin, LogicValue.ZERO, assignment):
                        reqs[fin] = LogicValue.ZERO
                        break
                else:
                    return None

        elif gate.type == GateType.NOR:
            if val == LogicValue.ONE:
                # All must be 0
                for fin in unassigned:
                    reqs[fin] = LogicValue.ZERO
            else:
                # One must be 1
                if any(assignment.get(fin) == LogicValue.ONE for fin in gate.fin):
                    return {}

                for fin in unassigned:
                    if self._check_global_consistency(fin, LogicValue.ONE, assignment):
                        reqs[fin] = LogicValue.ONE
                        break
                else:
                    return None

        elif gate.type == GateType.NOT:
            req_val = LogicValue.ZERO if val == LogicValue.ONE else LogicValue.ONE
            if self._check_global_consistency(gate.fin[0], req_val, assignment):
                reqs[gate.fin[0]] = req_val
            else:
                return None

        elif gate.type == GateType.BUFF:
            if self._check_global_consistency(gate.fin[0], val, assignment):
                reqs[gate.fin[0]] = val
            else:
                return None

        else:
            # XOR/XNOR - harder, need to solve parity
            return None

        # Verify all requirements are consistent
        for req_node, req_val in reqs.items():
            if not self._check_global_consistency(req_node, req_val, assignment):
                return None

        return reqs

    def _compute_simple(self, gtype: int, inputs: List[LogicValue]) -> LogicValue:
        # Copy of simple compute
        if gtype == GateType.AND:
            return LogicValue.ONE if all(i == LogicValue.ONE for i in inputs) else LogicValue.ZERO
        if gtype == GateType.NAND:
            return LogicValue.ZERO if all(i == LogicValue.ONE for i in inputs) else LogicValue.ONE
        if gtype == GateType.OR:
            return LogicValue.ONE if any(i == LogicValue.ONE for i in inputs) else LogicValue.ZERO
        if gtype == GateType.NOR:
            return LogicValue.ZERO if any(i == LogicValue.ONE for i in inputs) else LogicValue.ONE
        if gtype == GateType.NOT:
            return LogicValue.ZERO if inputs[0] == LogicValue.ONE else LogicValue.ONE
        if gtype == GateType.BUFF:
            return inputs[0]
        return LogicValue.XD

    def _compute_gate_robust(self, gtype: int, inputs: List[LogicValue]) -> LogicValue:
        """Compute gate output handling XD (unknown) values."""
        if gtype == GateType.AND:
            if any(i == LogicValue.ZERO for i in inputs):
                return LogicValue.ZERO
            if all(i == LogicValue.ONE for i in inputs):
                return LogicValue.ONE
            return LogicValue.XD
        elif gtype == GateType.NAND:
            if any(i == LogicValue.ZERO for i in inputs):
                return LogicValue.ONE
            if all(i == LogicValue.ONE for i in inputs):
                return LogicValue.ZERO
            return LogicValue.XD
        elif gtype == GateType.OR:
            if any(i == LogicValue.ONE for i in inputs):
                return LogicValue.ONE
            if all(i == LogicValue.ZERO for i in inputs):
                return LogicValue.ZERO
            return LogicValue.XD
        elif gtype == GateType.NOR:
            if any(i == LogicValue.ONE for i in inputs):
                return LogicValue.ZERO
            if all(i == LogicValue.ZERO for i in inputs):
                return LogicValue.ONE
            return LogicValue.XD
        elif gtype == GateType.NOT:
            if inputs[0] == LogicValue.ONE:
                return LogicValue.ZERO
            if inputs[0] == LogicValue.ZERO:
                return LogicValue.ONE
            return LogicValue.XD
        elif gtype == GateType.BUFF:
            return inputs[0]
        elif gtype == GateType.XOR:
            if any(i == LogicValue.XD for i in inputs):
                return LogicValue.XD
            ones = sum(1 for i in inputs if i == LogicValue.ONE)
            return LogicValue.ONE if ones % 2 == 1 else LogicValue.ZERO
        elif gtype == GateType.XNOR:
            if any(i == LogicValue.XD for i in inputs):
                return LogicValue.XD
            ones = sum(1 for i in inputs if i == LogicValue.ONE)
            return LogicValue.ONE if ones % 2 == 0 else LogicValue.ZERO
        return LogicValue.XD

    def _check_global_consistency(
        self, node: int, val: LogicValue, assignment: Dict[int, LogicValue]
    ) -> bool:
        """Check if assigning node=val conflicts with any existing assignments."""
        gate = self.circuit[node]

        # Check fanin consistency: if inputs are assigned, do they produce val?
        if gate.fin:
            input_vals = [assignment.get(fin, LogicValue.XD) for fin in gate.fin]
            # If any inputs are assigned, check if they're consistent with val
            if any(v != LogicValue.XD for v in input_vals):
                computed = self._compute_gate_robust(gate.type, input_vals)
                if computed != LogicValue.XD and computed != val:
                    return False

        # Check fanout consistency: if fanouts are assigned, is node=val consistent?
        for fout in gate.fot:
            if fout in assignment:
                fout_gate = self.circuit[fout]
                fout_val = assignment[fout]
                input_vals = []
                for fin in fout_gate.fin:
                    if fin == node:
                        input_vals.append(val)
                    else:
                        input_vals.append(assignment.get(fin, LogicValue.XD))

                computed = self._compute_gate_robust(fout_gate.type, input_vals)
                if computed != LogicValue.XD and computed != fout_val:
                    return False

        return True


## Legacy logic-justification utilities removed (paths-only datasets).


def _is_sequential(bench_path: str) -> bool:
    """Return True if the bench file contains sequential elements.

    Three detection strategies are applied, from cheapest to most thorough:

    1. **Name heuristic**: ISCAS89 (``s####.bench``) and ITC99 (``b##.bench``)
       circuits are always sequential by convention.
    2. **Text scan**: any gate definition that uses ``DFF`` or ``LATCH`` as a
       gate type (e.g. ``q = DFF(d)``), covering formats with explicit markers.
    3. **Parser fallback**: ``parse_bench_file`` maps unknown gate types to
       ``GateType.INPT``. A gate that is ``INPT`` *and* has fanins was parsed
       from an unknown type (e.g. ``DFF``). This catches formats where DFF is
       written in the body without an ``INPUT(...)`` declaration.
    """
    basename = os.path.splitext(os.path.basename(bench_path))[0]
    # ISCAS89: s\d+  |  ITC99: b\d+
    if re.fullmatch(r"s\d+", basename, re.IGNORECASE) or re.fullmatch(
        r"b\d+", basename, re.IGNORECASE
    ):
        return True

    _SEQ_PATTERN = re.compile(r"=\s*(DFF|LATCH)\s*\(", re.IGNORECASE)
    with open(bench_path) as f:
        for line in f:
            if _SEQ_PATTERN.search(line):
                return True

    # Parser-level check: INPT-typed gate with fanins → parsed from unknown type
    try:
        circuit, _ = parse_bench_file(bench_path)
        for gate in circuit:
            if gate is None:
                continue
            if gate.type == GateType.INPT and gate.fin and len(gate.fin) > 0:
                return True
    except Exception:
        pass  # If parsing fails we'll catch it in the main loop anyway

    return False


def build_dataset(
    base_path: str | List[str],
) -> List[Dict[str, Any]]:
    """Build a dataset of reconvergent path pairs with structural embeddings.

    The resulting entries include the circuit file path, the reconvergent
    structure info (start, reconv, branches, paths), and structural embeddings
    extracted from the AIG representation of the circuit.

    Parameters
    ----------
    base_path : str | list[str]
        One directory (or a list of directories) containing .bench files.

    Returns
    -------
    list[dict]
        Dataset entries with keys:
        - 'file': str - path to circuit file
        - 'info': dict - reconvergent structure (start, reconv, branches, paths)
        - 'struct_emb': torch.Tensor - structural embeddings
        - 'gate_mapping': dict - mapping from original to AIG gate IDs
    """
    dirs = [base_path] if isinstance(base_path, str) else list(base_path)
    bench_files = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"[WARNING] Benchmark directory not found, skipping: {d}")
            continue
        # Walk recursively so nested layouts (e.g. RCCG/<circuit>/<circuit>.bench)
        # are discovered automatically.
        for root, _, files in os.walk(d):
            bench_files.extend(
                os.path.join(root, f) for f in files if f.endswith(".bench")
            )

    _CIRCUIT_TIMEOUT_SECS = 30 * 60  # 30 minutes per circuit

    def _timeout_handler(signum, frame):  # noqa: ARG001
        raise TimeoutError("Circuit processing exceeded 30-minute limit")

    # Install the SIGALRM handler when called from the main thread; fall back
    # to no-timeout silently when called from a worker thread.
    _timeout_supported = False
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        _timeout_supported = True
    except (OSError, ValueError):
        print("[WARNING] SIGALRM not available — per-circuit timeout disabled")

    dataset: List[Dict[str, Any]] = []
    extractor = EmbeddingExtractor()

    try:
        for bench_file in bench_files:
            if _is_sequential(bench_file):
                print(f"  [SKIP] Sequential circuit detected: {bench_file}")
                continue
            print(f"Processing {bench_file}...")
            if _timeout_supported:
                signal.alarm(_CIRCUIT_TIMEOUT_SECS)
            try:
                circuit, _ = parse_bench_file(bench_file)

                # Extract embeddings for this circuit
                struct_emb, func_emb, gate_mapping, _ = extractor.extract_embeddings(
                    bench_file
                )
                print(f"  Extracted embeddings: struct_emb shape {struct_emb.shape}")

                # Enumerate all reconvergent pairs (subject to beam/depth constraints)
                infos: List[Dict[str, Any]] = find_all_reconv_pairs(
                    circuit, beam_width=16, max_depth=25
                )
                print(f"  Found {len(infos)} reconvergent path pairs")

                # Add each reconvergent pair as a separate dataset entry
                for info in infos:
                    dataset.append(
                        {
                            "file": bench_file,
                            "info": info,
                            "struct_emb": struct_emb,
                            "gate_mapping": gate_mapping,
                        }
                    )
            except TimeoutError:
                print(f"  [WARNING] Timeout after 30 minutes — skipping {bench_file}")
            except Exception as e:
                print(f"  [WARNING] Failed to process {bench_file}: {e}")
                print(f"  Skipping {bench_file}")
            finally:
                if _timeout_supported:
                    signal.alarm(0)  # cancel any pending alarm
    finally:
        extractor.cleanup()

    return dataset


def save_dataset(dataset, output_path):
    """Save dataset to pickle file.

    Parameters
    ----------
    dataset : list[dict]
        Dataset to save.
    output_path : str
        Path to save the pickle file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {output_path} ({len(dataset)} entries)")


def load_dataset(dataset_path):
    """Load dataset from pickle file.

    Parameters
    ----------
    dataset_path : str
        Path to the pickle file.

    Returns
    -------
    list[dict]
        Loaded dataset.
    """
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"Dataset loaded from {dataset_path} ({len(dataset)} entries)")
    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build reconvergent dataset from one or more benchmark directories"
    )
    parser.add_argument(
        "--bench-dirs",
        nargs="+",
        default=[
            "data/bench/ISCAS85",
            "data/bench/iscas89",
            "data/bench/ITC99",
            "data/bench/RCCG",
        ],
        help=(
            "Directories to search recursively for .bench files "
            "(default: ISCAS85 + iscas89 + ITC99 + RCCG). "
            "Sequential circuits (s####, b## naming or explicit DFF/LATCH gates) "
            "are detected and skipped automatically."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/datasets/reconv_dataset_combinational.pkl",
        help="Output pickle path",
    )
    _args = parser.parse_args()

    # Clean up stale staging directories left by previous interrupted runs
    _data_dir = "data"
    if os.path.isdir(_data_dir):
        _cur_pid = os.getpid()
        for _entry in os.listdir(_data_dir):
            if _entry.startswith("staging_") and _entry != f"staging_{_cur_pid}":
                _stale = os.path.join(_data_dir, _entry)
                try:
                    shutil.rmtree(_stale)
                    print(f"[INFO] Removed stale staging dir: {_stale}")
                except Exception as _e:
                    print(f"[WARNING] Could not remove stale staging dir {_stale}: {_e}")

    _bench_dirs = [d for d in _args.bench_dirs if os.path.isdir(d)]
    if not _bench_dirs:
        print("No valid benchmark directories found. Check --bench-dirs.")
    else:
        print(f"Building dataset from: {_bench_dirs}")
        _dataset = build_dataset(_bench_dirs)
        save_dataset(_dataset, _args.output)
