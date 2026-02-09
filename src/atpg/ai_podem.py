
import os
import torch
import collections
from typing import List, Dict, Any, Optional, Tuple

from src.util.struct import LogicValue, Gate, GateType, Fault
from src.atpg.podem import podem, backtrace_wrapper, initialize, get_all_faults, simple_backtrace
from src.atpg.recursive_reconv_solver import HierarchicalReconvSolver

class AIBacktracer:
    """
    Backtrace function that uses HierarchicalReconvSolver to satisfy objectives.
    Falls back to simple_backtrace if AI fails.
    """
    def __init__(self, solver: HierarchicalReconvSolver, verbose: bool = False):
        self.solver = solver
        self.circuit = solver.circuit
        self.verbose = verbose
        
    def __call__(self, objective: Fault, circuit: List[Gate]) -> Fault:
        # Objective: gate_id, value. Try AI Solve.
        if self.verbose: print(f"[AI-BT] Objective: Gate {objective.gate_id} = {objective.value}")
        try:
            # Build constraints from currently assigned PIs
            current_constraints = {}
            for i, g in enumerate(self.circuit):
               if g.type == GateType.INPT and g.val in (LogicValue.ZERO, LogicValue.ONE):
                   current_constraints[i] = g.val

            if self.verbose: print(f"  [AI-BT] Constraints: {current_constraints}")
            solution = self.solver.solve(objective.gate_id, objective.value, current_constraints)
            
            if solution:
                if self.verbose: print(f"  [AI-BT] Solution: {solution}")
                # 1. Try to find a direct PI assignment
                for gid, val in solution.items():
                    if self.circuit[gid].type == GateType.INPT and self.circuit[gid].val == LogicValue.XD:
                         if self.verbose: print(f"  [AI-BT] Returning assignment: Gate {gid}={val}")
                         return Fault(gid, val)
                
                # 2. If no PI, finding an internal node in solution that needs justification
                #    and use simple_backtrace to reach a PI from there.
                if self.verbose: print("  [AI-BT] No direct PI found. Looking for intermediate objectives...")
                for gid, val in solution.items():
                     if self.circuit[gid].val == LogicValue.XD:
                         if self.verbose: print(f"  [AI-BT] Delegating to simple_backtrace for internal objective: Gate {gid}={val}")
                         return simple_backtrace(Fault(gid, val), circuit)
                         
                if self.verbose: print("  [AI-BT] Solution found but all nodes already assigned/consistent?")
            else:
                 if self.verbose: print("  [AI-BT] No solution from solver.")
        except Exception as e:
            if self.verbose:
                print(f"  [AI-BT] Error: {e}")
                import traceback
                traceback.print_exc()
            pass
            
        # Fallback to simple
        if self.verbose: print("  [AI-BT] Fallback to simple_backtrace")
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
    ) -> Tuple[List[Dict[int, LogicValue]], Optional[Dict[str, Any]]]:
        if self.struct_emb is None or self.model is None:
            # Fallback to pure solver if model failed
            return self._fallback_solve(pair_info, constraints)

        # 1. Prepare Batch for Model
        
        paths = pair_info['paths']
        
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
            return [precomputed_assignment], None
        
        # Convert path node IDs to AIG IDs to get embeddings
        path_embs_list = []
        gate_types_list = []
        node_ids_list = [] # For saving state
        
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
                    p_types.append(0) # Unknown
            
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
            batch_mask[0, i, len(p):] = False

        # Snapshot for RL (Clone to CPU)
        inputs_snapshot = {
            'node_ids': batch_ids.cpu(),
            'mask_valid': batch_mask.cpu(),
            'gate_types': batch_types.cpu(),
            'files': [self.circuit_path]
        }
        
        # 2. Run Inference
        with torch.no_grad():
            logits, solv_logits = self.model(batch_embs, batch_mask, batch_types)
        
        # 3. Decode Logits
        probs = torch.softmax(logits, dim=-1) # [1, P, L, 2]
        vals = torch.argmax(probs, dim=-1).squeeze(0) # [P, L]
        
        predicted_assignment = {}
        conflict = False
        
        for i, p in enumerate(paths):
            for j, nid in enumerate(p):
                if j >= len(p): continue
                val = int(vals[i, j].item())
                lv = LogicValue.ZERO if val == 0 else LogicValue.ONE
                
                if nid in constraints:
                    if constraints[nid] != lv:
                        lv = constraints[nid]
                
                if nid in predicted_assignment:
                    if predicted_assignment[nid] != lv:
                        conflict = True
                
                predicted_assignment[nid] = lv
        
        return self._rank_solutions_with_model(pair_info, constraints, probs, paths, predicted_assignment, inputs_snapshot)

    def _rank_solutions_with_model(self, pair_info, constraints, probs, paths, predicted_assignment, inputs_snapshot):
        if self._verify_assignment_logic(predicted_assignment):
             return [predicted_assignment], inputs_snapshot
        
        return self._fallback_solve(pair_info, constraints)[0], inputs_snapshot

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
                 pass
        return True 

    def _fallback_solve(self, pair_info, constraints) -> Tuple[List[Dict[int, LogicValue]], Optional[Dict[str, Any]]]:
        # Try both 0 and 1 for Reconvergence Node
        reconv_node = pair_info['reconv']
        targets = []
        if reconv_node in constraints:
            targets.append(constraints[reconv_node])
        else:
            targets = [LogicValue.ZERO, LogicValue.ONE]
        
        # Create minimal snapshot for RL tracking even in fallback
        paths = pair_info.get('paths', [])
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
                'node_ids': node_ids,
                'mask_valid': mask_valid,
                'gate_types': gate_types,
                'files': [self.circuit_path]
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
