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
from typing import List, Dict, Any, Optional, Set, Tuple
import heapq

from src.util.struct import LogicValue, Gate
from src.atpg.reconv_podem import PathConsistencySolver

class ReconvPairPredictor(abc.ABC):
    """Abstract base class for predicting solutions to reconvergent pairs."""
    
    @abc.abstractmethod
    def predict(
        self, 
        pair_info: Dict[str, Any], 
        constraints: Dict[int, LogicValue]
    ) -> List[Dict[int, LogicValue]]:
        """
        Predict a list of valid assignments for the given pair, respecting constraints.
        
        Args:
            pair_info: Dictionary containing 'start', 'reconv', 'paths', etc.
            constraints: Dictionary of current node assignments {node_id: value}.
            
        Returns:
            A list of assignment dictionaries (partial solutions).
            The list is ordered by likelihood/preference.
        """
        pass


class HierarchicalReconvSolver:
    """
    Solves for a target value by recursively justifying reconvergent path pairs,
    ordered from shortest to longest.
    """

    def __init__(self, circuit: List[Gate], predictor: ReconvPairPredictor, recorder = None):
        self.circuit = circuit
        self.predictor = predictor
        self.recorder = recorder
        # Helper for consistency checks, reused from existing codebase
        self.consistency_checker = PathConsistencySolver(circuit)
        
        # Populate fanout lists from fanin (if not already present)
        self._populate_fanouts()

    def solve(self, target_node: int, target_val: LogicValue, constraints: Dict[int, LogicValue] = None) -> Optional[Dict[int, LogicValue]]:
        """
        Main entry point. Tries to justify target_node = target_val.
        
        1. Identify logic cone of target_node.
        2. Find and sort reconvergent pairs within the cone.
        3. Recursively solve pairs (backtracking).
        4. Return final assignment or None.
        """
        # 1. & 2. Find relevant pairs
        pairs = self._collect_and_sort_pairs(target_node)
        
        # Initial constraints: target + provided constraints
        initial_constraints = {}
        if constraints:
            initial_constraints.update(constraints)
        
        # Set/Overwrite target requirement (critical!)
        initial_constraints[target_node] = target_val
        
        # 3. Recursive Solve
        final_assignment = self._solve_recursive(0, pairs, initial_constraints)
        
        return final_assignment

    def _collect_and_sort_pairs(self, root_node: int) -> List[Dict[str, Any]]:
        """
        Identify reconvergent pairs in the transitive fanin of root_node,
        sorted by their 'span' (shortest path length or loop size).
        """
        # A. Transitive Fanin Cone
        cone_nodes = self._get_transitive_fanin(root_node)
        
        # B. Find Pairs within this cone
        # This is computationally expensive if we scan the whole circuit.
        # We can try to use a localized search or assume we have a precomputed list.
        # For this implementation, let's implement a cone-restricted search.
        # We look for stems (nodes with >1 fanout in the cone) and see if they converge.
        
        pairs = self._find_pairs_in_set(cone_nodes)
        
        # C. Sort by "Shortest"
        # Metric: Sum of lengths of the two branches (loop perimeter)
        def pair_cost(p):
            # paths[0] + paths[1] length
            # Note: paths include intermediate nodes.
            return len(p['paths'][0]) + len(p['paths'][1])
            
        pairs.sort(key=pair_cost)
        
        return pairs

    def _get_transitive_fanin(self, root: int) -> Set[int]:
        """BFS backwards to find all nodes feeding root."""
        seen = set()
        queue = collections.deque([root])
        seen.add(root)
        while queue:
            curr = queue.popleft()
            gate = self.circuit[curr]
            for fin in gate.fin:
                if fin not in seen:
                    seen.add(fin)
                    queue.append(fin)
        return seen

    def _populate_fanouts(self):
        """Build fanout lists from fanin relationships if not present."""
        # Initialize empty fanout lists
        for gate in self.circuit:
            if not hasattr(gate, 'fot') or gate.fot is None:
                gate.fot = []
        
        # Build fanouts from fanins
        for gate_id, gate in enumerate(self.circuit):
            for fin_id in gate.fin:
                if fin_id < len(self.circuit):
                    if gate_id not in self.circuit[fin_id].fot:
                        self.circuit[fin_id].fot.append(gate_id)

    def _find_pairs_in_set(self, allowed_nodes: Set[int]) -> List[Dict[str, Any]]:
        """
        Find reconvergent pairs where all path nodes are in allowed_nodes.
        Simplified BFS-based detection.
        """
        # Identify potential stems: nodes in allowed_nodes with multiple fanouts also in allowed_nodes
        stems = []
        for nid in allowed_nodes:
            gate = self.circuit[nid]
            valid_fot = [fo for fo in (getattr(gate, 'fot', []) or []) if fo in allowed_nodes]
            if len(valid_fot) >= 2:
                stems.append(nid)
                
        results = []
        
        # For each stem, launch BFS to find reconvergence points within allowed_nodes
        for s in stems:
            # We track which branch (index in valid_fot) reached a node
            start_gate = self.circuit[s]
            valid_fot = [fo for fo in (getattr(start_gate, 'fot', []) or []) if fo in allowed_nodes]
            
            # Track reported reconvergence nodes for this stem to avoid duplicates
            reported_reconvs = set()
            
            # reached[node] = {branch_idx: path_list}
            reached = {} 
            
            # Initial frontier
            queue = collections.deque()
            for i, fo in enumerate(valid_fot):
                path = [s, fo]
                reached.setdefault(fo, {})
                reached[fo][i] = path
                queue.append(fo)
                
            processed_nodes = {s} # Avoid cycles back to start
            
            # Limit search depth roughly
            while queue:
                curr = queue.popleft()
                
                # Check if this node is a reconvergence point for S
                # i.e., reached by >= 2 distinct branches
                if len(reached[curr]) >= 2 and curr != s:
                    # Found a pair!
                    # Extract pairs. If >2 branches, we can take all combinations or just first 2.
                    # Taking first 2 distinct branches for simplicity.
                    bs = list(reached[curr].keys())
                    b1, b2 = bs[0], bs[1]
                    p1 = reached[curr][b1]
                    p2 = reached[curr][b2]
                    
                    # Avoid duplicates? (s, curr)
                    # We add to results if not already reported for this stem.
                    if curr not in reported_reconvs:
                        reported_reconvs.add(curr)
                        results.append({
                            'start': s,
                            'reconv': curr,
                            'branches': [valid_fot[b1], valid_fot[b2]], # Branch specific nodes
                            'paths': [p1, p2]
                        })
                    
                    # Do we continue from here? Yes, might reach further reconvergence.
                    
                # Expand
                gate = self.circuit[curr]
                curr_branches = reached[curr].keys()
                
                valid_fot_curr = [fo for fo in (getattr(gate, 'fot', []) or []) if fo in allowed_nodes]
                
                for fo in valid_fot_curr:
                    if fo == s: continue 
                    
                    # Need to merge branch info
                    if fo not in reached:
                        reached[fo] = {}
                        new_visit = True
                    else:
                        new_visit = False
                        
                    changed = False
                    for b_idx in curr_branches:
                        if b_idx not in reached[fo]:
                            # Extend path
                            old_path = reached[curr][b_idx]
                            new_path = old_path + [fo]
                            reached[fo][b_idx] = new_path
                            changed = True
                            
                    if changed:
                        # If we added info, we must propagate, even if visited before (DAG)
                        # To avoid infinite loops in cyclic circuits we might need checks, but Bench circuits are usually DAGs.
                        # Simple optimization: only append if not already in queue? Set based queue?
                        # For now, just append.
                        queue.append(fo)

        # Post-processing: remove partial overlaps or duplicates if needed
        # For now, return all found.
        return results

    def _solve_recursive(
        self, 
        pair_idx: int, 
        pairs: List[Dict[str, Any]], 
        current_constraints: Dict[int, LogicValue]
    ) -> Optional[Dict[int, LogicValue]]:
        """
        Backtracking solver.
        
        Args:
            pair_idx: Index of the pair we are currently solving.
            pairs: Sorted list of all pairs.
            current_constraints: Assignments made so far.
            
        Returns:
            Full assignment dictionary if solvable, None otherwise.
        """
        # Base Case: All pairs processed
        if pair_idx >= len(pairs):
            return current_constraints
        
        pair = pairs[pair_idx]
        
        # Get candidate solutions from Oracle
        # The predictor should return solutions that respect `current_constraints`.
        
        # UPDATE: Predictor now returns (candidates, inputs_snapshot)
        prediction_result = self.predictor.predict(pair, current_constraints)
        
        # Handle backward compatibility if someone hasn't updated their predictor class
        inputs_snapshot = None
        if isinstance(prediction_result, tuple):
             candidates, inputs_snapshot = prediction_result
        else:
             candidates = prediction_result
        
        if not candidates:
            # If the model cannot find any solution for this pair given constraints,
            # this path is dead. Backtrack.
            return None
            
        for assignment_part in candidates:
            # Log this decision attempt if recording
            step_record = None
            if self.recorder and inputs_snapshot:
                 # Log the attempt. 
                 # We record the snapshot and the chosen assignment.
                 # Note: "candidates" is a ranked list. We are trying "assignment_part" now.
                 step_record = self.recorder.log_step(
                     node_ids=inputs_snapshot['node_ids'],
                     mask_valid=inputs_snapshot['mask_valid'],
                     gate_types=inputs_snapshot['gate_types'],
                     files=inputs_snapshot['files'],
                     pair_info=pair,
                     selected_assignment=assignment_part
                 )

            # 1. Merge assignment
            # (Logic consistency is assumed enforced by predictor, but we can double check)
            new_constraints = current_constraints.copy()
            conflict = False
            for k, v in assignment_part.items():
                if k in new_constraints:
                    if new_constraints[k] != v:
                        conflict = True
                        break
                else:
                    new_constraints[k] = v
            
            if conflict:
                if step_record and self.recorder:
                     # Immediate conflict -> local failure
                     self.recorder.mark_backtrack(penalty=-0.5) 
                continue
                
            # 2. Recurse
            result = self._solve_recursive(pair_idx + 1, pairs, new_constraints)
            
            if result is not None:
                # Found a valid complete assignment!
                # We do NOT mark success here per pair, 
                # but implicit success is that we don't penalize.
                # Global success will reward everyone later.
                return result
            
            # If we returned None, it means a conflict happened deeper in the recursion.
            # This choice (assignment_part) led to a failure.
            if step_record and self.recorder:
                 self.recorder.mark_backtrack(penalty=-0.5)
                 
        # If no candidates lead to a solution, backtrack.
        return None

