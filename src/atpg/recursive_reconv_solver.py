"""
Recursive Reconvergent Path Pair Solver.

This module implements a hierarchical justification flow that solves reconvergent
path pairs in a specific order (shortest to longest) to justify a target value.
It is designed to work with a predictive model (or oracle) that provides
candidate assignments for these pairs.
"""

from __future__ import annotations

import abc
import collections
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from src.atpg.reconv_podem import PathConsistencySolver
from src.util.struct import Gate, GateType, LogicValue


class ReconvPairPredictor(abc.ABC):
    """Abstract base class for predicting solutions to reconvergent pairs."""

    @abc.abstractmethod
    def predict(
        self,
        pair_info: Dict[str, Any],
        constraints: Dict[int, LogicValue],
        seed: Optional[int] = None,
    ) -> List[Dict[int, LogicValue]] | Tuple[List[Dict[int, LogicValue]], Any]:
        """
        Predict a list of valid assignments for the given pair, respecting constraints.

        Args:
            pair_info: Dictionary containing 'start', 'reconv', 'paths', etc.
            constraints: Dictionary of current node assignments {node_id: value}.
            seed: Optional random seed for deterministic sampling.

        Returns:
            A list of assignment dictionaries (partial solutions) OR
            (list of assignment dictionaries, snapshot information).
            The list is ordered by likelihood/preference.
        """
        pass


class HierarchicalReconvSolver:
    """
    Solves for a target value by recursively justifying reconvergent path pairs,
    ordered from shortest to longest.
    """

    def __init__(
        self,
        circuit: List[Gate],
        predictor: ReconvPairPredictor,
        recorder=None,
        verbose: bool = False,
        circuit_path: str = None,
    ):
        self.circuit = circuit
        self.predictor = predictor
        self.recorder = recorder
        self.verbose = verbose
        self.circuit_path = circuit_path
        self.nodes_visited_limit = 1000
        self.nodes_visited = 0
        self.inference_limit = 1000
        self.inferences = 0
        # Helper for consistency checks, reused from existing codebase
        self.consistency_checker = PathConsistencySolver(circuit)

        # Populate fanout lists from fanin (if not already present)
        self._populate_fanouts()

        # Try loading pair cache from disk (avoids expensive BFS on second run)
        self._pair_cache_dirty = False
        if circuit_path:
            from src.atpg.reconv_cache import load_pair_cache

            cached = load_pair_cache(circuit_path)
            if cached is not None:
                self.pair_cache = cached
                print(f"[INFO] Loaded reconvergent pair cache for {os.path.basename(circuit_path)}")
            else:
                self.pair_cache = {}
        else:
            self.pair_cache = {}  # Cache reconv pairs per root node

    def _persist_pair_cache_if_needed(self):
        """Persist pair cache to disk if it was updated during this run."""
        if self._pair_cache_dirty and self.circuit_path:
            from src.atpg.reconv_cache import persist_pair_cache

            persist_pair_cache(self.circuit_path, self.pair_cache)
            self._pair_cache_dirty = False

    def solve(
        self,
        target_node: int,
        target_val: LogicValue,
        constraints: Dict[int, LogicValue] = None,
        seed: Optional[int] = None,
    ) -> Optional[Dict[int, LogicValue]]:
        """
        Main entry point. Tries to justify target_node = target_val.
        """
        if self.verbose:
            print(
                f"[Solver] Solving for Gate {target_node} = {target_val} with "
                f"{len(constraints) if constraints else 0} constraints"
            )

        # 1. & 2. Find relevant pairs (use cache to avoid redundant BFS)
        if len(self.pair_cache) > 200:
            self.pair_cache.clear()

        if target_node not in self.pair_cache:
            self.pair_cache[target_node] = self._collect_and_sort_pairs(target_node)
            self._pair_cache_dirty = True  # Mark for disk persistence
        pairs_by_reconv = self.pair_cache[target_node]

        if self.verbose:
            print("[Solver] Collected reconvergent pairs")

        # Initial constraints: target + provided constraints
        initial_constraints = {}
        if constraints:
            initial_constraints.update(
                {k: LogicValue(v) if isinstance(v, int) else v for k, v in constraints.items()}
            )

        # Set/Overwrite target requirement
        initial_constraints[target_node] = (
            LogicValue(target_val) if isinstance(target_val, int) else target_val
        )

        # 3. Recursive Solve
        queue = [target_node]
        solved_pairs = set()
        self.nodes_visited = 0
        self.inferences = 0
        final_assignment = self._backward_justify(
            queue, initial_constraints, solved_pairs, pairs_by_reconv, seed
        )

        return final_assignment

    def _collect_and_sort_pairs(self, root_node: int) -> List[Dict[str, Any]]:
        """Identify and sort reconvergent pairs in the transitive fanin of root_node."""
        cone_nodes = self._get_transitive_fanin(root_node)
        pairs = self._find_pairs_in_set(cone_nodes)

        from collections import deque

        distances = {root_node: 0}
        queue = deque([root_node])

        while queue:
            curr = queue.popleft()
            curr_dist = distances[curr]
            gate = self.circuit[curr]
            if gate is None:
                continue
            for fin in gate.fin:
                if fin in cone_nodes and fin not in distances:
                    distances[fin] = curr_dist + 1
                    queue.append(fin)

        def pair_cost(p):
            reconv_node = p["reconv"]
            dist_to_target = distances.get(reconv_node, 9999)
            total_path_len = len(p["paths"][0]) + len(p["paths"][1])
            return (total_path_len + dist_to_target, total_path_len)

        pairs.sort(key=pair_cost)

        pairs_by_reconv = {}
        for p in pairs:
            r = p["reconv"]
            if r not in pairs_by_reconv:
                pairs_by_reconv[r] = []
            pairs_by_reconv[r].append(p)

        return pairs_by_reconv

    def _get_transitive_fanin(self, root: int, max_depth: int = 20) -> Set[int]:
        """BFS backwards to find all nodes feeding root up to max_depth."""
        seen = set()
        queue = collections.deque([(root, 0)])
        seen.add(root)
        while queue:
            curr, depth = queue.popleft()
            if depth >= max_depth:
                continue
            gate = self.circuit[curr]
            if gate is None:
                continue
            for fin in gate.fin:
                if fin not in seen:
                    seen.add(fin)
                    queue.append((fin, depth + 1))
        return seen

    def _populate_fanouts(self):
        """Build fanout lists from fanin relationships if not present."""
        for gate in self.circuit:
            if gate is None:
                continue
            if not hasattr(gate, "fot") or gate.fot is None:
                gate.fot = []

        for gate_id, gate in enumerate(self.circuit):
            if gate is None:
                continue
            for fin_id in gate.fin:
                if fin_id < len(self.circuit):
                    target_gate = self.circuit[fin_id]
                    if target_gate is None:
                        continue
                    if not hasattr(target_gate, "fot") or target_gate.fot is None:
                        target_gate.fot = []
                    if gate_id not in target_gate.fot:
                        target_gate.fot.append(gate_id)

    def _find_pairs_in_set(self, allowed_nodes: Set[int]) -> List[Dict[str, Any]]:
        """Find reconvergent pairs within a set of allowed nodes."""
        stems = []
        for nid in allowed_nodes:
            gate = self.circuit[nid]
            if gate is None:
                continue
            valid_fot = [fo for fo in (getattr(gate, "fot", []) or []) if fo in allowed_nodes]
            if len(valid_fot) >= 2:
                stems.append(nid)

        results = []
        for s in stems:
            start_gate = self.circuit[s]
            valid_fot = [fo for fo in (getattr(start_gate, "fot", []) or []) if fo in allowed_nodes]
            reported_reconvs = set()
            reached = {}
            queue = collections.deque()
            for i, fo in enumerate(valid_fot):
                reached[fo] = {i: [s, fo]}
                queue.append(fo)

            while queue:
                curr = queue.popleft()
                if len(reached[curr]) >= 2 and curr != s:
                    if curr not in reported_reconvs:
                        reported_reconvs.add(curr)
                        bs = list(reached[curr].keys())
                        results.append(
                            {
                                "start": s,
                                "reconv": curr,
                                "branches": [valid_fot[bs[0]], valid_fot[bs[1]]],
                                "paths": [reached[curr][bs[0]], reached[curr][bs[1]]],
                            }
                        )

                gate = self.circuit[curr]
                curr_branches = reached[curr].keys()
                valid_fot_curr = [
                    fo for fo in (getattr(gate, "fot", []) or []) if fo in allowed_nodes
                ]

                for fo in valid_fot_curr:
                    if fo == s:
                        continue
                    if fo not in reached:
                        reached[fo] = {}
                    changed = False
                    for b_idx in curr_branches:
                        if b_idx not in reached[fo]:
                            reached[fo][b_idx] = reached[curr][b_idx] + [fo]
                            changed = True
                    if changed:
                        queue.append(fo)
        return results

    def _backward_justify(
        self,
        queue: List[int],
        assignment: Dict[int, LogicValue],
        solved_pairs: Set[int],
        pairs_by_reconv: Dict[int, List[Dict[str, Any]]],
        seed: Optional[int] = None,
    ) -> Optional[Dict[int, LogicValue]]:
        """Queue-based backward justification with just-in-time AI predictor."""
        self.nodes_visited += 1

        # Periodic RAM check (every 50 nodes visited)
        if self.nodes_visited % 50 == 0:
            import psutil

            if psutil.virtual_memory().percent > 95:
                if self.verbose:
                    print("[Solver] Critical RAM usage. Bailing out.")
                return None

        if self.nodes_visited >= self.nodes_visited_limit:
            return None

        if not queue:
            return assignment

        # Sort queue: highest ID (furthest back) first
        queue.sort(reverse=True)

        gate = queue.pop(0)
        val = assignment[gate]

        # Check if already justified
        gate_obj = self.circuit[gate]
        if gate_obj.type != GateType.INPT:
            input_vals = [assignment.get(fin, LogicValue.XD) for fin in gate_obj.fin]
            if self._compute_gate_robust(gate_obj.type, input_vals) == val:
                return self._backward_justify(
                    list(queue), assignment, solved_pairs, pairs_by_reconv, seed
                )

        # Check if gate is a terminus for any UNsolved path pairs
        unsolved_pairs = []
        if gate in pairs_by_reconv:
            unsolved_pairs = [p for p in pairs_by_reconv[gate] if id(p) not in solved_pairs]

        if unsolved_pairs:
            # Suspend normal backtrace, try AI model on all available unsolved pairs
            for pair in unsolved_pairs:
                if (
                    self.nodes_visited >= self.nodes_visited_limit
                    or self.inferences >= self.inference_limit
                ):
                    return None
                self.inferences += 1
                prediction_result = self.predictor.predict(pair, assignment, seed=seed)
                inputs_snapshot = None
                if isinstance(prediction_result, tuple):
                    candidates, inputs_snapshot = prediction_result
                else:
                    candidates = prediction_result

                if not candidates:
                    continue

                for i, assignment_part in enumerate(candidates):
                    if self.nodes_visited >= self.nodes_visited_limit:
                        return None
                    step_record = None
                    if self.recorder and inputs_snapshot:
                        step_record = self.recorder.log_step(
                            node_ids=inputs_snapshot["node_ids"],
                            mask_valid=inputs_snapshot["mask_valid"],
                            gate_types=inputs_snapshot["gate_types"],
                            files=inputs_snapshot["files"],
                            pair_info=pair,
                            selected_assignment=assignment_part,
                        )

                    new_assignment = assignment.copy()
                    conflict = False
                    for k, v in assignment_part.items():
                        if not self._check_global_consistency(k, v, new_assignment):
                            conflict = True
                            break
                        new_assignment[k] = v

                    if conflict:
                        if step_record and self.recorder:
                            self.recorder.mark_backtrack(penalty=-0.5)
                        continue

                    # Detailed Logic Consistency Check
                    for k in assignment_part.keys():
                        gate_obj = self.circuit[k]
                        if gate_obj.type != GateType.INPT:
                            input_vals = [
                                new_assignment.get(fin, LogicValue.XD) for fin in gate_obj.fin
                            ]
                            comp_val = self._compute_gate_robust(gate_obj.type, input_vals)
                            if comp_val != LogicValue.XD and comp_val != new_assignment[k]:
                                conflict = True
                                break
                        for fout in gate_obj.fot:
                            if fout in new_assignment:
                                fout_obj = self.circuit[fout]
                                input_vals = [
                                    new_assignment.get(fin, LogicValue.XD) for fin in fout_obj.fin
                                ]
                                comp_val = self._compute_gate_robust(fout_obj.type, input_vals)
                                if comp_val != LogicValue.XD and comp_val != new_assignment[fout]:
                                    conflict = True
                                    break
                        if conflict:
                            break

                    if conflict:
                        if step_record and self.recorder:
                            self.recorder.mark_backtrack(penalty=-0.5)
                        continue

                    new_queue = list(queue)
                    # Enqueue new requirements that are not PIs
                    for k in assignment_part.keys():
                        if k not in new_queue and self.circuit[k].type != GateType.INPT:
                            new_queue.append(k)

                    new_solved = set(solved_pairs)
                    new_solved.add(id(pair))

                    result = self._backward_justify(
                        new_queue, new_assignment, new_solved, pairs_by_reconv, seed
                    )
                    if result is not None:
                        return result

                    if step_record and self.recorder:
                        self.recorder.mark_backtrack(penalty=-0.5)

            # If all pairs failed/contradicted, fail fast instead of hanging in standard DFS
            return None

        # Standard Gate Justification
        gate_obj = self.circuit[gate]
        if gate_obj.type == GateType.INPT:
            return self._backward_justify(
                list(queue), assignment, solved_pairs, pairs_by_reconv, seed
            )

        options = self._justify_gate(gate, val, assignment)
        if not options:
            return None

        for reqs in options:
            if self.nodes_visited >= self.nodes_visited_limit:
                return None
            new_assignment = assignment.copy()
            new_queue = list(queue)
            for r_node, r_val in reqs.items():
                new_assignment[r_node] = r_val
                if self.circuit[r_node].type != GateType.INPT and r_node not in new_queue:
                    new_queue.append(r_node)

            result = self._backward_justify(
                new_queue, new_assignment, solved_pairs, pairs_by_reconv, seed
            )
            if result is not None:
                return result

        return None

    def _check_global_consistency(
        self, node: int, val: LogicValue, assignment: Dict[int, LogicValue]
    ) -> bool:
        if node in assignment:
            return assignment[node] == val
        gate = self.circuit[node]
        if gate.fin:
            input_vals = [assignment.get(fin, LogicValue.XD) for fin in gate.fin]
            if any(v != LogicValue.XD for v in input_vals):
                computed = self._compute_gate_robust(gate.type, input_vals)
                if computed != LogicValue.XD and computed != val:
                    return False
        for fout in getattr(gate, "fot", []) or []:
            if fout in assignment:
                fout_gate = self.circuit[fout]
                fout_val = assignment[fout]
                input_vals = [
                    val if fin == node else assignment.get(fin, LogicValue.XD)
                    for fin in fout_gate.fin
                ]
                computed = self._compute_gate_robust(fout_gate.type, input_vals)
                if computed != LogicValue.XD and computed != fout_val:
                    return False
        return True

    def _justify_gate(
        self, node: int, val: LogicValue, assignment: Dict[int, LogicValue]
    ) -> List[Dict[int, LogicValue]]:
        """Return all valid justification options for a gate."""
        gate = self.circuit[node]
        unassigned = [fin for fin in gate.fin if fin not in assignment]
        if not unassigned:
            input_vals = [assignment[fin] for fin in gate.fin]
            res = self._compute_simple(gate.type, input_vals)
            return [{}] if res == val else []

        input_vals_partial = [assignment.get(fin, LogicValue.XD) for fin in gate.fin]
        computed = self._compute_gate_robust(gate.type, input_vals_partial)
        if computed != LogicValue.XD and computed != val:
            return []
        if computed == val:
            return [{}]

        options = []
        if gate.type == GateType.AND:
            if val == LogicValue.ONE:
                reqs = {}
                for fin in unassigned:
                    reqs[fin] = LogicValue.ONE
                options.append(reqs)
            else:
                for fin in unassigned:
                    if self._check_global_consistency(fin, LogicValue.ZERO, assignment):
                        options.append({fin: LogicValue.ZERO})
        elif gate.type == GateType.NAND:
            if val == LogicValue.ZERO:
                reqs = {}
                for fin in unassigned:
                    reqs[fin] = LogicValue.ONE
                options.append(reqs)
            else:
                for fin in unassigned:
                    if self._check_global_consistency(fin, LogicValue.ZERO, assignment):
                        options.append({fin: LogicValue.ZERO})
        elif gate.type == GateType.OR:
            if val == LogicValue.ZERO:
                reqs = {}
                for fin in unassigned:
                    reqs[fin] = LogicValue.ZERO
                options.append(reqs)
            else:
                for fin in unassigned:
                    if self._check_global_consistency(fin, LogicValue.ONE, assignment):
                        options.append({fin: LogicValue.ONE})
        elif gate.type == GateType.NOR:
            if val == LogicValue.ONE:
                reqs = {}
                for fin in unassigned:
                    reqs[fin] = LogicValue.ZERO
                options.append(reqs)
            else:
                for fin in unassigned:
                    if self._check_global_consistency(fin, LogicValue.ONE, assignment):
                        options.append({fin: LogicValue.ONE})
        elif gate.type == GateType.NOT:
            options.append(
                {gate.fin[0]: LogicValue.ZERO if val == LogicValue.ONE else LogicValue.ONE}
            )
        elif gate.type == GateType.BUFF:
            options.append({gate.fin[0]: val})
        elif gate.type in (GateType.XOR, GateType.XNOR):
            # Only support 2-input XOR/XNOR for simplicity
            if len(gate.fin) == 2:
                target_val = (
                    val
                    if gate.type == GateType.XOR
                    else (LogicValue.ONE if val == LogicValue.ZERO else LogicValue.ZERO)
                )
                if len(unassigned) == 1:
                    assigned_fin = [fin for fin in gate.fin if fin not in unassigned][0]
                    assigned_val = assignment[assigned_fin]
                    req_val = LogicValue.ONE if assigned_val != target_val else LogicValue.ZERO
                    options.append({unassigned[0]: req_val})
                elif len(unassigned) == 2:
                    not_target = LogicValue.ZERO if target_val == LogicValue.ONE else LogicValue.ONE
                    options.append({unassigned[0]: LogicValue.ZERO, unassigned[1]: target_val})
                    options.append({unassigned[0]: LogicValue.ONE, unassigned[1]: not_target})

        # Filter out globally inconsistent options
        valid_options = []
        for reqs in options:
            valid = True
            for r_node, r_val in reqs.items():
                if not self._check_global_consistency(r_node, r_val, assignment):
                    valid = False
                    break
            if valid:
                valid_options.append(reqs)

        # Sort options: prioritize PIs or lower topological IDs
        # To avoid deep backtracking without AI guidance
        def _cost(opt: Dict[int, LogicValue]) -> int:
            if not opt:
                return 0
            # We want to pick options that rely on smaller node IDs first (closer to PIs)
            return max(opt.keys())

        valid_options.sort(key=_cost)
        return valid_options

    def _compute_simple(self, gtype: int, inputs: List[LogicValue]) -> LogicValue:
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
        if gtype == GateType.XOR:
            ones = sum(1 for i in inputs if i == LogicValue.ONE)
            return LogicValue.ONE if ones % 2 == 1 else LogicValue.ZERO
        if gtype == GateType.XNOR:
            ones = sum(1 for i in inputs if i == LogicValue.ONE)
            return LogicValue.ZERO if ones % 2 == 1 else LogicValue.ONE
        return LogicValue.XD

    def _compute_gate_robust(self, gtype: int, inputs: List[LogicValue]) -> LogicValue:
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
            if inputs[0] == LogicValue.XD:
                return LogicValue.XD
            return LogicValue.ZERO if inputs[0] == LogicValue.ONE else LogicValue.ONE
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
            return LogicValue.ZERO if ones % 2 == 1 else LogicValue.ONE
        return LogicValue.XD
