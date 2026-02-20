import argparse
import os
import sys

import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.atpg.ai_podem import AiPodemConfig, ModelPairPredictor, ai_podem
from src.util.io import parse_bench_file
from src.util.struct import Fault, GateType, LogicValue


def main():
    parser = argparse.ArgumentParser(description="Debug AI PODEM Execution logic")
    parser.add_argument("circuit", type=str, help="Path to circuit .bench file")
    parser.add_argument("fault", type=str, help="Fault string (e.g. '1-0' for gate 1 stuck-at-0)")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    print(f"Loading circuit: {args.circuit}")
    circuit, total_gates = parse_bench_file(args.circuit)
    circuit_path = args.circuit

    # Parse fault
    try:
        gate_id, val = map(int, args.fault.split("-"))
        fault = Fault(gate_id, LogicValue(val))
        print(f"Target Fault: Gate {gate_id} s-a-{val}")
    except Exception:
        print(f"Invalid fault format: {args.fault}. Use gate-val (e.g. 1-0)")
        return

    # Initialize Predictor
    print("Loading Model...")
    config = AiPodemConfig(
        model_path=args.model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_ai_activation=True,
        enable_ai_propagation=True,
    )

    predictor = ModelPairPredictor(circuit, circuit_path, config)

    # Monkey patch the solver to trace execution
    import src.atpg.recursive_reconv_solver as solver_module

    # 0. Force Verbose Solver
    original_solver_init = solver_module.HierarchicalReconvSolver.__init__

    def verbose_solver_init(self, *args, **kwargs):
        original_solver_init(self, *args, **kwargs)
        self.verbose = True

    solver_module.HierarchicalReconvSolver.__init__ = verbose_solver_init

    # Helper to get fanin cone size
    def get_fanin_cone(circuit, root_id):
        cone = set()
        queue = [root_id]
        while queue:
            curr = queue.pop(0)
            if curr not in cone:
                cone.add(curr)
                queue.extend(circuit[curr].fin)
        return cone

    # 1. Trace Pair Collection with Cone Stats
    original_collect_pairs = solver_module.HierarchicalReconvSolver._collect_and_sort_pairs

    def traced_collect_pairs(self, root_node):
        cone = get_fanin_cone(self.circuit, root_node)
        pi_in_cone = [n for n in cone if self.circuit[n].type == GateType.INPT]
        print(f"\n[Solver] === Analyzing Fanin Cone of Gate {root_node} ===")
        print(f"[Solver] Cone size: {len(cone)} gates ({len(pi_in_cone)} PIs).")

        pairs = original_collect_pairs(self, root_node)
        print(f"[Solver] Identified {len(pairs)} reconvergent pairs in cone.")
        if pairs:
            print("[Solver] Top 10 pairs by search priority:")
            for i, p in enumerate(pairs[:10]):
                stem = p.get("start", p.get("stem"))
                reconv = p["reconv"]
                p_len = len(p["paths"][0]) + len(p["paths"][1])
                print(f"  {i + 1}. Stem {stem} -> Reconv {reconv} (Total Path Len: {p_len})")
            if len(pairs) > 10:
                print(f"  ... +{len(pairs) - 10} more")
        else:
            print("[Solver] ℹ No reconvergence found. Solver will only enforce target requirement.")
        return pairs

    solver_module.HierarchicalReconvSolver._collect_and_sort_pairs = traced_collect_pairs

    # 2. Trace Recursive Solver Step
    original_solve_recursive = solver_module.HierarchicalReconvSolver._solve_recursive
    recursion_depth = [0]

    def traced_solve_recursive(self, pair_idx, pairs, current_constraints, seed=None):
        depth = recursion_depth[0]
        indent = "  " * depth

        if pair_idx < len(pairs):
            pair = pairs[pair_idx]
            stem = pair.get("start", pair.get("stem"))
            reconv = pair.get("reconv")

            seed_info = f" [Seed: {seed}]" if seed is not None else ""
            print(
                f"\n{indent}┌────── [Step {pair_idx + 1}/{len(pairs)}] "
                f"Solving Pair: Stem {stem} -> Reconv {reconv}{seed_info}"
            )
            print(f"{indent}│   [State] Active Constraints: {len(current_constraints)} nodes")
            recursion_depth[0] += 1
        else:
            print(
                f"\n{indent}┌────── [Base Case] All pairs processed. Checking final consistency..."
            )

        # Call original
        result = original_solve_recursive(self, pair_idx, pairs, current_constraints, seed)

        if pair_idx < len(pairs):
            recursion_depth[0] -= 1
            indent = "  " * recursion_depth[0]
            if result:
                print(f"{indent}└────── ✓ SOLVED Step {pair_idx + 1}")
            else:
                print(f"{indent}└────── ✗ FAILED Step {pair_idx + 1} (Backtracking)")
        else:
            if result:
                print(f"{indent}└────── ✓ Solution Validated")
            else:
                print(f"{indent}└────── ✗ Solution Invalid")

        return result

    solver_module.HierarchicalReconvSolver._solve_recursive = traced_solve_recursive

    # 3. Trace Predictor Decisions
    original_predict = predictor.predict

    def traced_predict(pair_info, constraints, seed=None):
        stem = pair_info.get("start", pair_info.get("stem"))
        reconv = pair_info.get("reconv")
        depth = recursion_depth[0]
        indent = "  " * depth

        print(f"{indent}│   [Inference] Querying AI Model for Stem {stem} -> Reconv {reconv}")

        # Call original predict
        predict_res = original_predict(pair_info, constraints, seed)

        # Support both old and new return types (candidates vs (candidates, snapshot))
        if isinstance(predict_res, tuple):
            candidates, snapshot = predict_res
        else:
            candidates = predict_res

        if candidates:
            n_cands = len(candidates)
            tag = "" if n_cands == 1 else f" (+{n_cands - 1} fallback)"
            print(f"{indent}│   [Inference] Model returned {n_cands} candidate(s){tag}.")

            # Show the model prediction (first candidate), path by path
            sample = candidates[0]
            paths = pair_info.get("paths", [])
            for p_idx, path in enumerate(paths):
                path_vals = []
                for nid in path:
                    val = sample.get(nid, None)
                    if val is None:
                        path_vals.append(f"{nid}:?")
                    else:
                        path_vals.append(f"{nid}:{int(val)}")
                print(f"{indent}│   [Inference] Path {p_idx}: [{' -> '.join(path_vals)}]")
        else:
            print(f"{indent}│   [Inference] ⚠ Model returned NO candidates (logical conflict).")

        return predict_res

    predictor.predict = traced_predict

    # 4. Trace AIBacktracer
    import src.atpg.ai_podem as ai_podem_module

    original_backtracer_call = ai_podem_module.AIBacktracer.__call__
    backtracer_call_count = [0]

    def traced_backtracer_call(self, objective, circuit):
        backtracer_call_count[0] += 1
        bt_id = backtracer_call_count[0]

        print("\n  ╔═══════════════════════════════════════════════════════════")
        print(f"  ║ [AI-Backtrace {bt_id}] PROPAGATION OBJECTIVE")
        print(f"  ║ Target: Gate {objective.gate_id} = {objective.value}")
        print("  ╚═══════════════════════════════════════════════════════════")

        result = original_backtracer_call(self, objective, circuit)

        if result and result.gate_id != -1:
            print(f"    ✓ Assigned PI: Gate {result.gate_id} = {result.value}\n")
        else:
            print("    ✗ Failed to find PI assignment via AI (Fallback used)\n")

        return result

    ai_podem_module.AIBacktracer.__call__ = traced_backtracer_call

    # 1.5 Trace PODEM Internals (Granular Decisions)
    import src.atpg.podem as podem_module

    original_simple_backtrace = podem_module.simple_backtrace

    def traced_simple_backtrace(objective, circuit):
        res = original_simple_backtrace(objective, circuit)
        if res.gate_id != -1:
            print(
                f"    [PODEM] Decision: Obj(Gate {objective.gate_id}="
                f"{objective.value}) -> Assigned PI {res.gate_id}={res.value}"
            )
        else:
            print(
                f"    [PODEM] Decision: Obj(Gate {objective.gate_id}={objective.value}) -> ✗ FAILED"
            )
        return res

    podem_module.simple_backtrace = traced_simple_backtrace

    # Non-invasive PODEM monitoring: do NOT replace podem_recursion
    # or set_trace_decisions — both interfere with recursion/output.
    # Instead, we poll get_statistics() after the run.

    # Trace Logic Sim to see if it's the crash site
    import src.atpg.logic_sim_three as logic_sim_module

    original_logic_sim = logic_sim_module.logic_sim

    def traced_logic_sim(*args, **kwargs):
        # print(".", end="", flush=True)
        return original_logic_sim(*args, **kwargs)

    logic_sim_module.logic_sim = traced_logic_sim

    # Add verification after solve
    original_solve_main = solver_module.HierarchicalReconvSolver.solve

    def traced_solve_with_verification(self, target_node, target_val, constraints=None, seed=None):
        print(f"\n[Solver] === AI Justification Started: Gate {target_node} = {target_val} ===")
        if constraints:
            print(f"[Solver] Current Constraints: {list(constraints.keys())}")

        result = original_solve_main(self, target_node, target_val, constraints, seed)

        if result:
            print(f"\n    {'─' * 40}")
            print("    LOGIC CONSISTENCY VERIFICATION")
            print(f"    {'─' * 40}")

            inconsistencies = []
            for gid, expected_val in result.items():
                if gid > total_gates:
                    continue
                gate = circuit[gid]
                if gate.type == GateType.INPT:
                    continue

                # Check if all inputs are assigned
                if not all(fin in result for fin in gate.fin):
                    continue

                # Read-only verification: use result dict values directly
                # DO NOT mutate circuit[fin].val — that corrupts PODEM state.
                from src.util.struct import GateType as GT

                fin_vals = [int(result[fin]) for fin in gate.fin]
                gt = gate.type
                if gt == GT.AND:
                    computed_val = int(all(v == 1 for v in fin_vals))
                elif gt == GT.NAND:
                    computed_val = int(not all(v == 1 for v in fin_vals))
                elif gt == GT.OR:
                    computed_val = int(any(v == 1 for v in fin_vals))
                elif gt == GT.NOR:
                    computed_val = int(not any(v == 1 for v in fin_vals))
                elif gt == GT.NOT:
                    computed_val = 1 - fin_vals[0]
                elif gt == GT.BUFF:
                    computed_val = fin_vals[0]
                elif gt == GT.XOR:
                    computed_val = fin_vals[0] ^ fin_vals[1] if len(fin_vals) == 2 else 0
                elif gt == GT.XNOR:
                    computed_val = 1 - (fin_vals[0] ^ fin_vals[1]) if len(fin_vals) == 2 else 0
                else:
                    continue  # Unknown gate type

                if computed_val != int(expected_val):
                    inconsistencies.append({"gate": gid, "exp": expected_val, "got": computed_val})

            if inconsistencies:
                print(f"    ✗ FAILED: {len(inconsistencies)} inconsistencies found.")
                for inc in inconsistencies[:3]:
                    print(f"      Gate {inc['gate']}: Model={inc['exp']}, Logic={inc['got']}")
            else:
                print(f"    ✓ SUCCESS: All {len(result)} assignments are logically consistent.")

            print(f"    {'─' * 40}\n")
        else:
            print("\n[Solver] === AI Justification FAILED ===\n")

        return result

    solver_module.HierarchicalReconvSolver.solve = traced_solve_with_verification

    # --- Trace internal Justification (the "regular" part) ---
    original_justify_all = solver_module.HierarchicalReconvSolver._justify_all

    def traced_justify_all(self, assignment):
        print("  ┌────── [Justification] Solving remaining logic paths to PIs...")
        res = original_justify_all(self, assignment)
        if res:
            print("  └────── ✓ Justification Complete.")
        else:
            print("  └────── ✗ Justification Failed.")
        return res

    solver_module.HierarchicalReconvSolver._justify_all = traced_justify_all

    original_justify_gate = solver_module.HierarchicalReconvSolver._justify_gate

    def traced_justify_gate(self, node, val, assignment):
        res = original_justify_gate(self, node, val, assignment)
        if res:
            for fin, fval in res.items():
                # Filter entries that are already in assignment to avoid spam,
                # or just show the new ones
                if assignment.get(fin) != fval:
                    print(
                        f"  │   [Trace] Gate {node}={val} justification -> setting Input {fin}={fval}"
                    )
        return res

    solver_module.HierarchicalReconvSolver._justify_gate = traced_justify_gate

    print("=" * 80)
    print("RUNNING AI PODEM (Activation + Propagation - Detailed Trace)")
    print("=" * 80 + "\n")

    # patterns = [] # Unused

    # Reset circuit
    from src.atpg.podem import initialize

    initialize(circuit, total_gates)

    try:
        # Use AI Activation only
        print("[AI-PODEM] Starting AI Justification for Activation...")
        activation_val = LogicValue.ONE if fault.value == LogicValue.D else LogicValue.ZERO

        # Actually, let's use the solver directly to illustrate the process we saw in logs
        solver = solver_module.HierarchicalReconvSolver(circuit, predictor, verbose=True)
        ai_assignment = solver.solve(fault.gate_id, activation_val)

        if ai_assignment:
            print("\n" + "=" * 80)
            print("AI ACTIVATION SUCCESS")
            print(f"Total assignments in AI dictionary: {len(ai_assignment)}")

            # Print PI assignments
            pi_assignments = {
                gid: val for gid, val in ai_assignment.items() if circuit[gid].type == GateType.INPT
            }
            print(f"PI assignments ({len(pi_assignments)}): {pi_assignments}")

            # Verify internal logic consistency of the AI assignment
            violations = predictor._verify_assignment_logic(ai_assignment)
            if violations == 0:
                print("✓ AI Logic Verification: SUCCESS")
            else:
                print(f"✗ AI Logic Verification: FAILED ({violations} violations)")

            # --- ILLUSTRATE PODEM-STYLE ACTIVATION COMPLETION ---
            print("\n[AI-PODEM] Applying PI assignments to circuit...")
            for gid, val in pi_assignments.items():
                circuit[gid].val = val

            print("[AI-PODEM] Running logic simulation (Forward Prop)...")
            from src.atpg.logic_sim_three import logic_sim
            from src.atpg.util import get_topological_order

            topo_order = get_topological_order(circuit, total_gates)

            # Simulation propagates PIs through the cone.
            # If the AI solver was correct, Gate 259 should reach the target value.
            logic_sim(circuit, total_gates, fault, topo_order=topo_order)

            actual_val = circuit[fault.gate_id].val
            print(f"[AI-PODEM] Gate {fault.gate_id} value after simulation: {actual_val}")

            # Check for activation (Fault site should be D or DB depending on objective)
            # activation_val was the GOOD value we wanted.
            if actual_val in [LogicValue.D, LogicValue.DB]:
                print("✓ ACTIVATION VERIFIED: Fault site contains D/D-bar.")
            elif actual_val == activation_val:
                # In some cases if D/DB isn't handled by sim yet for activation phase
                print("✓ ACTIVATION VERIFIED: Fault site reached objective value.")
            else:
                print(f"✗ ACTIVATION FAILED: Expected {activation_val}, reached {actual_val}")

            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("AI ACTIVATION FAILURE")
            print("=" * 80)

        # EXIT before propagation
        print("\nExiting after activation analysis as requested.")
        return

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


if __name__ == "__main__":
    f = open("debug_ai_podem_execution.log", "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    try:
        main()
    finally:
        sys.stdout = original_stdout
        f.close()
