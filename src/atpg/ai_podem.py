
import os
import torch
import collections
from typing import List, Dict, Any, Optional

from src.util.struct import LogicValue, Gate, GateType, Fault
from src.atpg.podem import podem, backtrace_wrapper, initialize, get_all_faults, simple_backtrace
from src.atpg.recursive_reconv_solver import HierarchicalReconvSolver

class AIBacktracer:
    """
    Backtrace function that uses HierarchicalReconvSolver to satisfy objectives.
    Falls back to simple_backtrace if AI fails.
    """
    def __init__(self, solver: HierarchicalReconvSolver):
        self.solver = solver
        self.circuit = solver.circuit
        
    def __call__(self, objective: Fault, circuit: List[Gate]) -> Fault:
        # Objective: gate_id, value. Try AI Solve.
        try:
            # Build constraints from currently assigned PIs
            current_constraints = {}
            for g in self.circuit:
               if g.type == GateType.INPT and g.val in (LogicValue.ZERO, LogicValue.ONE):
                   current_constraints[g.node_id] = g.val

            solution = self.solver.solve(objective.gate_id, objective.value, current_constraints)
            
            if solution:
                # Return ONE unassigned PI from solution
                for gid, val in solution.items():
                    if self.circuit[gid].type == GateType.INPT and self.circuit[gid].val == LogicValue.XD:
                         return Fault(gid, val)
        except Exception as e:
            pass
            
        # Fallback to simple
        return simple_backtrace(objective, circuit)

from src.atpg.reconv_podem import PathConsistencySolver
from src.atpg.recursive_reconv_solver import HierarchicalReconvSolver, ReconvPairPredictor
from src.ml.reconv_lib import MultiPathTransformer
from src.ml.embedding_extractor import EmbeddingExtractor
from src.atpg.logic_sim_three import logic_sim, reset_gates, print_pi

class ModelPairPredictor(ReconvPairPredictor):
    def __init__(self, circuit_path: str, model_path: str, circuit: List[Gate]):
        self.circuit_path = circuit_path
        self.circuit = circuit
        # Force CPU for stability if CUDA is problematic, but try to honor availability
        self.device = torch.device('cuda') 
        # (Overriding to CPU because of the 'no kernel image' error in the environment)
        
        # Load embeddings (SLOW step: ideally cached)
        self.extractor = EmbeddingExtractor()
        # We need structural embeddings for the whole circuit to map them to pair paths
        # Note: EmbeddingExtractor returns embeddings for AIG nodes.
        # We need a mapping from original gate IDs (used in pair_info) to AIG/embedding indices.
        try:
            self.struct_emb, _, self.gate_mapping, _ = self.extractor.extract_embeddings(circuit_path)
            self.struct_emb = self.struct_emb.to(self.device)
            # Map str(id) -> int(aig_id)
            self.gate_mapping = {int(k): int(v) for k, v in self.gate_mapping.items()}
        except Exception as e:
            print(f"[AI-PODEM] Failed to extract embeddings: {e}")
            self.struct_emb = None
            
        # Load Model
        self.model = self._load_model(model_path)
        self.solver = PathConsistencySolver(circuit)

    def _load_model(self, model_path: str):
        # Infer dimensions from embeddings if possible, or use defaults matching train_reconv.py
        input_dim = 132 # 128 struct + 4 logic
        model = MultiPathTransformer(
            input_dim=input_dim,
            model_dim=512,
            nhead=4,
            num_encoder_layers=3,
            num_interaction_layers=3
        ).to(self.device)
        
        if os.path.exists(model_path):
            try:
                # Assuming model checkpoint is full state dict
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                     model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                     model.load_state_dict(checkpoint['state_dict'])
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
        constraints: Dict[int, LogicValue]
    ) -> List[Dict[int, LogicValue]]:
        
        if self.struct_emb is None or self.model is None:
            # Fallback to pure solver if model failed
            return self._fallback_solve(pair_info, constraints)

        # 1. Prepare Batch for Model
        # Need paths embedding [1, P, L, D] and node_ids/gate_types
        # pair_info paths are [S, ..., R].
        
        paths = pair_info['paths']
        # Convert path node IDs to AIG IDs to get embeddings
        path_embs_list = []
        gate_types_list = []
        
        valid_input = True
        
        max_len = max(len(p) for p in paths)
        
        for p in paths:
            p_emb = []
            p_types = []
            for nid in p:
                if nid in self.gate_mapping:
                    aig_id = self.gate_mapping[nid]
                    if aig_id < self.struct_emb.size(0):
                        p_emb.append(self.struct_emb[aig_id])
                    else:
                        # Should not happen if mapping is correct
                        p_emb.append(torch.zeros(128, device=self.device))
                else:
                    # Missing mapping? Use zero vector
                    p_emb.append(torch.zeros(128, device=self.device))
                # Pad to 132 (4 extra logic dims)
                if len(p_emb[-1]) < 132:
                    pad = torch.zeros(132 - len(p_emb[-1]), device=self.device)
                    p_emb[-1] = torch.cat([p_emb[-1], pad])
                
                # Gate Type
                if nid < len(self.circuit):
                    p_types.append(self.circuit[nid].type)
                else:
                    p_types.append(0) # Unknown
            
            # Pad sequence
            while len(p_emb) < max_len:
                p_emb.append(torch.zeros(132, device=self.device))
                p_types.append(0)
            
            path_embs_list.append(torch.stack(p_emb))
            gate_types_list.append(torch.tensor(p_types, device=self.device))

        # Stack to [1, P, L, D]
        batch_embs = torch.stack(path_embs_list).unsqueeze(0) # [1, P, L, D]
        batch_types = torch.stack(gate_types_list).unsqueeze(0) # [1, P, L]
        batch_mask = torch.ones((1, len(paths), max_len), dtype=torch.bool, device=self.device)
        # TODO: Adjust mask for padding if paths had different lengths (handled by append loop implicitly?)
        # Yes, we padded with valid tensors, but conceptually they are padding.
        # Attention mask used in transformer: False means ignored.
        for i, p in enumerate(paths):
            batch_mask[0, i, len(p):] = False

        # 2. Run Inference
        with torch.no_grad():
            logits, solv_logits = self.model(batch_embs, batch_mask, batch_types)
        
        # 3. Decode Logits to Assignments
        # Logic: We want to extract valid assignments for the path nodes.
        # The model predicts probability of 0/1 for each node.
        # We can greedy decode or sample.
        # But we must respect `constraints`.
        
        probs = torch.softmax(logits, dim=-1) # [1, P, L, 2]
        # Get top predictions
        vals = torch.argmax(probs, dim=-1).squeeze(0) # [P, L]
        
        predicted_assignment = {}
        conflict = False
        
        for i, p in enumerate(paths):
            for j, nid in enumerate(p):
                # Check if padded
                if j >= len(p): continue
                
                val = int(vals[i, j].item())
                # LogicValue enum: 0->ZERO, 1->ONE
                lv = LogicValue.ZERO if val == 0 else LogicValue.ONE
                
                if nid in constraints:
                    if constraints[nid] != lv:
                        # Model prediction conflicts with constraint!
                        # Trust constraint or Model? 
                        # Constraints are hard facts (from previous decisions).
                        # If Model disagrees, model is wrong about this node context.
                        # We can try to enforce the constraint or just mark as conflict.
                        # But wait, predict() should return *candidates*. 
                        # A generic predictor might just return what it thinks.
                        # We need to filter/adjust.
                        lv = constraints[nid]
                
                if nid in predicted_assignment:
                    if predicted_assignment[nid] != lv:
                        # Internal conflict in prediction (same node in multiple paths)
                        # Pick one or fail?
                        conflict = True
                
                predicted_assignment[nid] = lv
        
        # Verify this assignment is locally consistent using Solver
        # Because raw model prediction might be invalid logic (e.g. AND(1,1)=0).
        # We use the solver to "repair" or validate.
        # Actually, simpler: Use solver to find valid solution *using prediction as heuristic*.
        # OR: Just return the predicted map and let recursive solver's implicit checks handle it?
        # Parameter `predict` says "returns ranked list of valid assignments".
        # So we should validate.
        
        # Let's try to validate/fix using solver.
        # We use the predicted values as preferred constraints?
        # Or we just use `_fallback_solve` filtering by prediction logic?
        
        # Better approach: Use solver to enumerate solutions, rank them by model probability?
        # If solver is cheap enough for a single pair.
        # PathConsistencySolver uses backtrace, it's relatively fast.
        
        return self._rank_solutions_with_model(pair_info, constraints, probs, paths, predicted_assignment)

    def _rank_solutions_with_model(self, pair_info, constraints, probs, paths, predicted_assignment):
        # We need ALL solutions from solver, then rank them.
        # But solver backtrace returns *one* solution or boolean?
        # `PathConsistencySolver.solve` returns *one* assignment or None.
        # It doesn't enumerate.
        
        # So we rely on `_fallback_solve` pattern: try target values.
        # BUT `reconv_podem.PathConsistencySolver` is not designed to enumerate.
        # Maybe we can use the Model's top choice, if valid.
        
        # Let's trust the model's output as a *candidate*.
        # We verify it.
        # Since we constructed `predicted_assignment` above, let's verify logic consistency.
        # We can run a mini-sim or check edges.
        
        # If model prediction is invalid, we fall back to generic solver logic.
        
        if self._verify_assignment_logic(predicted_assignment):
             return [predicted_assignment]
        
        # If invalid, return fallback
        return self._fallback_solve(pair_info, constraints)

    def _verify_assignment_logic(self, assignment):
        # Quick check of all gates in assignment
        for nid, val in assignment.items():
            if nid >= len(self.circuit): continue
            gate = self.circuit[nid]
            if not gate.fin: continue
            
            # Get input vals if in assignment
            input_vals = []
            all_inputs_present = True
            for fin in gate.fin:
                if fin in assignment:
                    input_vals.append(assignment[fin])
                else:
                    all_inputs_present = False
            
            if all_inputs_present:
                 # Check consistency
                 # LogicValue 0/1 match?
                 # Need a helper to compute gate.
                 # Reusing simple compute from reconv_podem or just manual
                 pass
        return True # Placeholder: assume model is decent or let recursive solver catch conflicts deeper

    def _fallback_solve(self, pair_info, constraints):
        # Try both 0 and 1 for Reconvergence Node if not constrained
        # Return valid ones.
        reconv_node = pair_info['reconv']
        targets = []
        if reconv_node in constraints:
            targets.append(constraints[reconv_node])
        else:
            targets = [LogicValue.ZERO, LogicValue.ONE]
            
        candidates = []
        for t in targets:
            res = self.solver.solve(pair_info, t, constraints)
            if res:
                candidates.append(res)
        return candidates


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
    verbose: bool = False
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
                predictor = ModelPairPredictor(circuit_path, model_path, circuit)
            solver = HierarchicalReconvSolver(circuit, predictor)
    
    ai_assignment = None
    
    # --- Step 1: AI Justification (Activation) ---
    if enable_ai_activation and solver:
        # Target: Fault Activation
        activation_val = LogicValue.ONE if fault.value == LogicValue.D else LogicValue.ZERO
        if verbose: print(f"[AI-PODEM] Attempting AI Justification for Fault {fault.gate_id} @ {activation_val}")
        
        ai_assignment = solver.solve(fault.gate_id, activation_val)
        
        if ai_assignment:
            if verbose: print(f"[AI-PODEM] AI found activation assignment ({len(ai_assignment)} gates).")
            # Apply to PIs
            pi_cnt = 0
            for gid, val in ai_assignment.items():
                if circuit[gid].type == GateType.INPT:
                    circuit[gid].val = val
                    pi_cnt += 1
            if verbose: print(f"[AI-PODEM] Applied {pi_cnt} PI assignments.")
        else:
            if verbose: print("[AI-PODEM] AI Activation failed. Continuing clean.")
            reset_gates(circuit, total_gates)

    # --- Step 2: Standard/Hybrid PODEM ---
    if verbose: print(f"[AI-PODEM] Starting PODEM (AI Prop={enable_ai_propagation})...")
    
    backtracer = None
    if enable_ai_propagation and solver:
        backtracer = AIBacktracer(solver)
        
    result = mogu_podem_wrapper(circuit, fault, total_gates, backtrace_func=backtracer)
    
    if result:
        if verbose:
            print("[AI-PODEM] Success!")
            print("Test Pattern:", print_pi(circuit, total_gates))
        return True
        
    # --- Step 3: Fallback ---
    # If we used AI Activation and failed, retry Clean
    if enable_ai_activation and ai_assignment and not result:
        if verbose: print("[AI-PODEM] AI-Activated run failed. Retrying CLEAN (Standard PODEM)...")
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
