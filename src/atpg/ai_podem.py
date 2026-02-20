import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.atpg.logic_sim_three import print_pi, reset_gates
from src.atpg.podem import (
    get_all_faults,
    initialize,
    podem,
    simple_backtrace,
)
from src.atpg.reconv_podem import PathConsistencySolver
from src.atpg.recursive_reconv_solver import (
    HierarchicalReconvSolver,
    ReconvPairPredictor,
)
from src.ml.core.model import MultiPathTransformer
from src.ml.data.embedding import EmbeddingExtractor
from src.util.struct import Fault, Gate, GateType, LogicValue


@dataclass
class AiPodemConfig:
    model_path: str
    device: str = "cpu"
    enable_ai_activation: bool = True
    enable_ai_propagation: bool = True
    verbose: bool = False


class AIBacktracer:
    """
    Backtrace function that uses HierarchicalReconvSolver to satisfy objectives.
    Falls back to simple_backtrace if AI fails.
    """

    def __init__(self, solver: HierarchicalReconvSolver, verbose: bool = False):
        self.solver = solver
        self.circuit = solver.circuit
        self.verbose = verbose
        # Precompute PI indices to avoid O(N) iteration in every __call__
        self.pi_indices = [i for i, g in enumerate(self.circuit) if g.type == GateType.INPT]

    def __call__(self, objective: Fault, circuit: List[Gate]) -> Fault:
        # Objective: gate_id, value. Try AI Solve.
        if self.verbose:
            print(f"[AI-BT] Objective: Gate {objective.gate_id} = {objective.value}")
        try:
            # Fast path: skip AI if no reconvergent structure exists
            if hasattr(self.solver, "pair_cache"):
                if objective.gate_id not in self.solver.pair_cache:
                    pairs = self.solver._collect_and_sort_pairs(objective.gate_id)
                    self.solver.pair_cache[objective.gate_id] = pairs
                else:
                    pairs = self.solver.pair_cache[objective.gate_id]

                if not pairs:
                    if self.verbose:
                        print(
                            f"  [AI-BT] No reconv pairs for gate {objective.gate_id}, skipping AI."
                        )
                    return simple_backtrace(objective, circuit)

            # Build constraints from currently assigned PIs
            current_constraints = {}
            for i in self.pi_indices:
                g = self.circuit[i]
                if g.val in (LogicValue.ZERO, LogicValue.ONE):
                    current_constraints[i] = g.val

            # Generate random seed based on timestamp
            import time

            current_seed = int(time.time() * 1000) % 10000000
            if self.verbose:
                print(f"  [AI-BT] Constraints: {current_constraints}, Seed: {current_seed}")

            solution = self.solver.solve(
                objective.gate_id,
                objective.value,
                current_constraints,
                seed=current_seed,
            )

            if solution:
                if self.verbose:
                    print(f"  [AI-BT] Solution: {solution}")
                # 1. Try to find a direct PI assignment
                for gid, val in solution.items():
                    if (
                        self.circuit[gid].type == GateType.INPT
                        and self.circuit[gid].val == LogicValue.XD
                    ):
                        if self.verbose:
                            print(f"  [AI-BT] Returning assignment: Gate {gid}={val}")
                        return Fault(gid, val)

                # 2. If no PI, finding an internal node in solution that needs
                # justification and use simple_backtrace to reach a PI from there.
                if self.verbose:
                    print("  [AI-BT] No direct PI found. Looking for intermediate objectives...")
                for gid, val in solution.items():
                    if self.circuit[gid].val == LogicValue.XD:
                        if self.verbose:
                            print(
                                "  [AI-BT] Delegating to simple_backtrace for "
                                f"internal objective: Gate {gid}={val}"
                            )
                        return simple_backtrace(Fault(gid, val), circuit)

                if self.verbose:
                    print("  [AI-BT] Solution found but all nodes already assigned/consistent?")
            else:
                if self.verbose:
                    print("  [AI-BT] No solution from solver.")
        except Exception as e:
            if self.verbose:
                print(f"  [AI-BT] Error: {e}")
                import traceback

                traceback.print_exc()
            pass

        # Fallback to simple
        if self.verbose:
            print("  [AI-BT] Fallback to simple_backtrace")
        return simple_backtrace(objective, circuit)


def post_process_logic_gates(
    vals: torch.Tensor,  # [P, L] predicted values (0 or 1)
    gate_types: torch.Tensor,  # [P, L] gate type for each position
    mask: torch.Tensor,  # [P, L] valid mask (True for real nodes)
    constraints: Optional[Dict[int, "LogicValue"]] = None,
    node_ids: Optional[torch.Tensor] = None,  # [P, L] for constraint lookup
) -> torch.Tensor:
    """Forward-propagate deterministic gate rules (NOT/BUFF) through paths.

    For each path, iterates from position 0 forward. At each position:
    - NOT gate: force cur = 1 - prev
    - BUFF gate: force cur = prev
    - Others: keep model prediction (AND/OR/NAND/NOR satisfy inequality constraints)

    Also respects any externally provided constraints.

    Returns: corrected vals tensor [P, L]
    """
    corrected = vals.clone()
    P, L = vals.shape

    for p in range(P):
        path_len = mask[p].sum().item()
        if path_len <= 1:
            continue

        # If constraints exist, apply them to the first position
        if constraints is not None and node_ids is not None:
            nid = int(node_ids[p, 0].item())
            if nid in constraints:
                corrected[p, 0] = 0 if constraints[nid] == LogicValue.ZERO else 1

        # Forward propagate
        for pos in range(1, int(path_len)):
            gt = int(gate_types[p, pos].item())
            prev_val = int(corrected[p, pos - 1].item())

            # Apply constraints first if available
            if constraints is not None and node_ids is not None:
                nid = int(node_ids[p, pos].item())
                if nid in constraints:
                    corrected[p, pos] = 0 if constraints[nid] == LogicValue.ZERO else 1
                    continue

            # Deterministic gate rules
            if gt == GateType.NOT:
                corrected[p, pos] = 1 - prev_val
            elif gt == GateType.BUFF:
                corrected[p, pos] = prev_val
            # AND/NAND/OR/NOR: keep model prediction (inequality-based)

    return corrected


class ModelPairPredictor(ReconvPairPredictor):
    def __init__(self, circuit: List[Gate], circuit_path: str, config: AiPodemConfig):
        self.circuit_path = circuit_path
        self.circuit = circuit
        self.config = config
        self.device = torch.device("cpu")
        if config.verbose:
            print("[AI-BT] Forcing CPU for inference due to environment CUDA mismatch (sm_120).")

        # Load embeddings (SLOW step: ideally cached)
        self.extractor = EmbeddingExtractor()
        # DeepGate is strictly required — no dummy fallback.
        self.struct_emb, _, self.gate_mapping, _ = self.extractor.extract_embeddings(circuit_path)
        self.struct_emb = self.struct_emb.to(self.device)
        # Map str(id) -> int(aig_id)
        self.gate_mapping = {int(k): int(v) for k, v in self.gate_mapping.items()}

        # Load Model
        self.model = self._load_model(config.model_path)
        self.solver = PathConsistencySolver(circuit)

    def _load_model(self, model_path: str):
        # Infer dimensions from embeddings if possible, or use defaults matching
        # train_reconv.py
        input_dim = 132  # 128 struct + 4 logic
        model = MultiPathTransformer(
            input_dim=input_dim,
            model_dim=512,
            nhead=4,
            num_encoder_layers=3,
            num_interaction_layers=3,
            dim_feedforward=512,  # Match training default
        ).to(self.device)

        if os.path.exists(model_path):
            try:
                # Assuming model checkpoint is full state dict
                checkpoint = torch.load(model_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
            except Exception as e:
                print(f"[AI-PODEM] Failed to load model weights: {e}")
        else:
            print(f"[AI-PODEM] Model path not found: {model_path}. Using random weights.")

        return model

    def predict(
        self,
        pair_info: Dict[str, Any],
        constraints: Dict[int, LogicValue],
        seed: Optional[int] = None,
    ) -> Tuple[List[Dict[int, LogicValue]], Optional[Dict[str, Any]]]:
        if self.struct_emb is None or self.model is None:
            # Fallback to pure solver if model failed
            return self._fallback_solve(pair_info, constraints)[0], None

        # 1. Prepare Batch for Model

        paths = pair_info["paths"]

        # Optimization: If all nodes in paths already have values, use them directly
        all_constrained = True
        precomputed_assignment = {}
        for p in paths:
            for nid in p:
                if nid in constraints:
                    precomputed_assignment[nid] = constraints[nid]
                else:
                    all_constrained = False
                    break
            if not all_constrained:
                break

        if all_constrained and precomputed_assignment:
            # All gates already have values - skip model, return existing values
            # Need strict type match for return tuple
            return [precomputed_assignment], None

        # Convert path node IDs to AIG IDs to get embeddings
        path_embs_list = []
        gate_types_list = []
        node_ids_list = []  # For saving state

        max_len = max(len(p) for p in paths)

        for p in paths:
            p_emb = []
            p_types = []
            p_ids = []
            for nid in p:
                p_ids.append(nid)
                if nid in self.gate_mapping:
                    aig_id = self.gate_mapping[nid]
                    if aig_id < self.struct_emb.size(0):
                        p_emb.append(self.struct_emb[aig_id])
                    else:
                        p_emb.append(torch.zeros(128, device=self.device))
                else:
                    p_emb.append(torch.zeros(128, device=self.device))
                # Pad to 132 (4 extra logic dims)
                if len(p_emb[-1]) < 132:
                    pad = torch.zeros(132 - len(p_emb[-1]), device=self.device)
                    p_emb[-1] = torch.cat([p_emb[-1], pad])

                # Gate Type
                if nid < len(self.circuit):
                    p_types.append(self.circuit[nid].type)
                else:
                    p_types.append(0)  # Unknown

            # Pad sequence
            while len(p_emb) < max_len:
                p_emb.append(torch.zeros(132, device=self.device))
                p_types.append(0)
                p_ids.append(0)

            path_embs_list.append(torch.stack(p_emb))
            gate_types_list.append(torch.tensor(p_types, device=self.device))
            node_ids_list.append(torch.tensor(p_ids, device=self.device))

        # Stack to [1, P, L, D]
        batch_embs = torch.stack(path_embs_list).unsqueeze(0)
        batch_types = torch.stack(gate_types_list).unsqueeze(0)
        batch_ids = torch.stack(node_ids_list).unsqueeze(0)

        batch_mask = torch.ones((1, len(paths), max_len), dtype=torch.bool, device=self.device)
        for i, p in enumerate(paths):
            batch_mask[0, i, len(p) :] = False

        # Snapshot for RL (Clone to CPU)
        inputs_snapshot = {
            "node_ids": batch_ids.cpu(),
            "mask_valid": batch_mask.cpu(),
            "gate_types": batch_types.cpu(),
            "files": [self.circuit_path],
        }

        # 2. Run Inference
        with torch.no_grad():
            # Inject noise if seed is provided. Scale could be configurable.
            perturb_scale = 0.5 if seed is not None else 0.0
            logits, solv_logits = self.model(
                batch_embs,
                batch_mask,
                batch_types,
                node_ids=batch_ids,  # Pass Node IDs for embedding
                seed=seed,
                perturb_scale=perturb_scale,
            )

        # 3. Decode Logits
        # Output is strictly binary (0 or 1) logits [B, P, L, 2]
        probs = torch.softmax(logits, dim=-1)  # [1, P, L, 2]
        vals = torch.argmax(probs, dim=-1).squeeze(0)  # [P, L] -> 0 or 1

        # 4. Post-process: enforce NOT/BUFF deterministic gate rules
        # (This remains valid for 0/1 predictions)
        vals = post_process_logic_gates(
            vals,
            batch_types.squeeze(0),  # [P, L]
            batch_mask.squeeze(0),  # [P, L]
            constraints=constraints,
            node_ids=batch_ids.squeeze(0),  # [P, L]
        )

        predicted_assignment = {}
        # Track per-node confidence for conflict resolution
        node_confidence = {}  # nid -> float (prob of chosen class)

        for i, p in enumerate(paths):
            for j, nid in enumerate(p):
                if j >= len(p):
                    continue
                val = int(vals[i, j].item())
                lv = LogicValue.ZERO if val == 0 else LogicValue.ONE
                conf = float(probs[0, i, j, val].item())

                # Constraints always override model
                if nid in constraints:
                    lv = constraints[nid]
                    conf = 1.0  # Constraints are absolute

                # Resolve cross-path conflicts: keep highest confidence
                if nid in predicted_assignment:
                    if predicted_assignment[nid] != lv:
                        prev_conf = node_confidence.get(nid, 0.0)
                        if conf <= prev_conf:
                            continue  # Keep previous (higher confidence)

                predicted_assignment[nid] = lv
                node_confidence[nid] = conf

        return self._rank_solutions_with_model(
            pair_info,
            constraints,
            probs,
            paths,
            predicted_assignment,
            inputs_snapshot,
        )

    def _rank_solutions_with_model(
        self,
        pair_info,
        constraints,
        probs,
        paths,
        predicted_assignment,
        inputs_snapshot,
    ):
        violations = self._verify_assignment_logic(predicted_assignment)

        # Always return model prediction as first candidate.
        # The HierarchicalReconvSolver._solve_recursive does its own
        # consistency checks, so let it decide whether to accept.
        candidates = [predicted_assignment]

        # If model prediction has violations, also add fallback as backup
        if violations > 0:
            fallback, _ = self._fallback_solve(pair_info, constraints)
            candidates.extend(fallback)

        return candidates, inputs_snapshot

    def _verify_assignment_logic(self, assignment: Dict[int, LogicValue]) -> int:
        """Verify logical consistency and return the count of violations.

        Returns 0 if all gates are consistent, otherwise the number of
        gates whose predicted value contradicts their Boolean truth table.
        """
        from src.atpg.logic_sim_three import compute_gate_value

        violations = 0
        for nid, val in assignment.items():
            if nid >= len(self.circuit):
                continue
            gate = self.circuit[nid]
            if not gate.fin:
                continue

            # Check if all inputs are present in the assignment
            if all(fin in assignment for fin in gate.fin):
                original_vals = {fin: self.circuit[fin].val for fin in gate.fin}
                original_gate_val = gate.val

                for fin in gate.fin:
                    self.circuit[fin].val = assignment[fin]

                expected_val = compute_gate_value(self.circuit, gate)

                for fin, v in original_vals.items():
                    self.circuit[fin].val = v
                gate.val = original_gate_val

                if expected_val != val:
                    violations += 1
                    if self.config.verbose:
                        print(
                            f"  [AI-BT] Logic Mismatch at Gate "
                            f"{nid} ({gate.type}): "
                            f"Expected {expected_val}, "
                            f"Predicted {val}"
                        )

        return violations

    def _fallback_solve(
        self, pair_info, constraints
    ) -> Tuple[List[Dict[int, LogicValue]], Optional[Dict[str, Any]]]:
        # Try both 0 and 1 for Reconvergence Node
        reconv_node = pair_info["reconv"]
        targets = []
        if reconv_node in constraints:
            targets.append(constraints[reconv_node])
        else:
            targets = [LogicValue.ZERO, LogicValue.ONE]

        # Create minimal snapshot for RL tracking even in fallback
        paths = pair_info.get("paths", [])
        if paths:
            max_len = max(len(p) for p in paths)
            node_ids = torch.zeros(1, len(paths), max_len, dtype=torch.long)
            mask_valid = torch.zeros(1, len(paths), max_len, dtype=torch.bool)
            gate_types = torch.zeros(1, len(paths), max_len, dtype=torch.long)

            for i, p in enumerate(paths):
                for j, nid in enumerate(p):
                    node_ids[0, i, j] = nid
                    mask_valid[0, i, j] = True
                    if nid < len(self.circuit):
                        gate_types[0, i, j] = self.circuit[nid].type

            snapshot = {
                "node_ids": node_ids,
                "mask_valid": mask_valid,
                "gate_types": gate_types,
                "files": [self.circuit_path],
            }
        else:
            snapshot = None

        candidates = []
        for t in targets:
            res = self.solver.solve(pair_info, t, constraints)
            if res:
                candidates.append(res)
        return candidates, snapshot


def ai_podem(
    circuit: List[Gate],
    fault: Fault,
    total_gates: int,
    model_path: str = "checkpoints/reconv_minimal_model.pt",
    circuit_path: str = "",
    enable_ai_activation: bool = True,
    enable_ai_propagation: bool = False,
    predictor: Optional[ModelPairPredictor] = None,
    solver: Optional[HierarchicalReconvSolver] = None,
    verbose: bool = False,
) -> bool:
    """
    AI-Assisted PODEM with configurable modes.

    Args:
        enable_ai_activation: Use AI Solver to justify fault activation (pre-fill).
        enable_ai_propagation: Use AI Solver for backtracing during propagation.
    """

    # Initialize shared PODEM structures
    initialize(circuit, total_gates)
    reset_gates(circuit, total_gates)

    # Predictor & Solver Setup

    if enable_ai_activation or enable_ai_propagation:
        if solver is None:
            if not circuit_path:
                print("[AI-PODEM] Warning: circuit_path missing, AI might fail.")
            if predictor is None:
                # Create config from args
                config = AiPodemConfig(
                    model_path=model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    enable_ai_activation=enable_ai_activation,
                    enable_ai_propagation=enable_ai_propagation,
                )
                predictor = ModelPairPredictor(circuit, circuit_path, config)
            solver = HierarchicalReconvSolver(circuit, predictor)

    ai_assignment = None

    # --- Step 1: AI Justification (Activation) ---
    if enable_ai_activation and solver:
        # Target: Fault Activation
        activation_val = LogicValue.ONE if fault.value == LogicValue.D else LogicValue.ZERO
        if verbose:
            print(
                "[AI-PODEM] Attempting AI Justification for Fault "
                f"{fault.gate_id} @ {activation_val}"
            )

        ai_assignment = solver.solve(fault.gate_id, activation_val)

        if ai_assignment:
            if verbose:
                print(f"[AI-PODEM] AI found activation assignment ({len(ai_assignment)} gates).")
            # Apply to PIs
            pi_cnt = 0
            for gid, val in ai_assignment.items():
                if circuit[gid].type == GateType.INPT:
                    circuit[gid].val = val
                    pi_cnt += 1
            if verbose:
                print(f"[AI-PODEM] Applied {pi_cnt} PI assignments.")
        else:
            if verbose:
                print("[AI-PODEM] AI Activation failed. Continuing clean.")
            reset_gates(circuit, total_gates)

    # --- Step 2: Standard/Hybrid PODEM ---
    if verbose:
        print(f"[AI-PODEM] Starting PODEM (AI Prop={enable_ai_propagation})...")

    backtracer = None
    if enable_ai_propagation and solver:
        backtracer = AIBacktracer(solver, verbose=verbose)

    result = mogu_podem_wrapper(circuit, fault, total_gates, backtrace_func=backtracer)

    if result:
        if verbose:
            print("[AI-PODEM] Success!")
            print("Test Pattern:", print_pi(circuit, total_gates))
        return True

    # --- Step 3: Fallback ---
    # If we used AI Activation and failed, retry Clean
    if enable_ai_activation and ai_assignment and not result:
        if verbose:
            print("[AI-PODEM] AI-Activated run failed. Retrying CLEAN (Standard PODEM)...")
        reset_gates(circuit, total_gates)

        result_retry = mogu_podem_wrapper(circuit, fault, total_gates, backtrace_func=None)
        if result_retry:
            if verbose:
                print("[AI-PODEM] Clean retry Success!")
                print("Test Pattern:", print_pi(circuit, total_gates))
            return True

    print("[AI-PODEM] Failure.")
    return False


def mogu_podem_wrapper(circuit, fault, total_gates, backtrace_func=None):
    # Wrapper to call the global `podem` function from src.atpg.podem
    return podem(circuit, fault, total_gates, backtrace_func=backtrace_func)


if __name__ == "__main__":
    # Test runner
    import sys

    bench = sys.argv[1] if len(sys.argv) > 1 else "data/bench/c17.bench"
    print(f"Testing AI-PODEM on {bench}")

    # Parse
    from src.util.io import parse_bench_file

    circuit, total_gates = parse_bench_file(bench)
    faults = get_all_faults(circuit, total_gates)

    # Pick a fault
    fault = faults[0]
    result = ai_podem(circuit, fault, total_gates, circuit_path=bench)
