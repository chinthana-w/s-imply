import argparse
import os
import sys

import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.atpg.ai_podem import AiPodemConfig, ModelPairPredictor
from src.util.io import parse_bench_file
from src.util.struct import Fault, GateType, LogicValue


def main():
    parser = argparse.ArgumentParser(description="Debug AI PODEM Execution logic")
    parser.add_argument("circuit", type=str, help="Path to circuit .bench file")
    parser.add_argument("fault", type=str, help="Fault string (e.g. '1-0' for gate 1 stuck-at-0)")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (cuda/cpu)",
    )
    args = parser.parse_args()

    print(f"Loading circuit: {args.circuit}")
    circuit, total_gates = parse_bench_file(args.circuit)
    circuit_path = args.circuit

    # Parse fault
    try:
        gate_id, val = map(int, args.fault.split("-"))
        fault_val = LogicValue.D if val == 0 else LogicValue.DB
        fault = Fault(gate_id, fault_val)
        print(f"Target Fault: Gate {gate_id} s-a-{val} (Assigned state: {fault_val})")
    except Exception:
        print(f"Invalid fault format: {args.fault}. Use gate-val (e.g. 1-0)")
        return

    # Initialize Predictor
    print("Loading Model...")
    config = AiPodemConfig(
        model_path=args.model,
        device=args.device,
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

        pairs_by_reconv = original_collect_pairs(self, root_node)

        all_pairs = []
        for r, plist in pairs_by_reconv.items():
            all_pairs.extend(plist)

        print(f"[Solver] Identified {len(all_pairs)} reconvergent pairs in cone.")
        if all_pairs:
            print("[Solver] Top 10 pairs by search priority:")
            for i, p in enumerate(all_pairs[:10]):
                stem = p.get("start", p.get("stem"))
                reconv = p["reconv"]
                p_len = len(p["paths"][0]) + len(p["paths"][1])
                print(f"  {i + 1}. Stem {stem} -> Reconv {reconv} (Total Path Len: {p_len})")
            if len(all_pairs) > 10:
                print(f"  ... +{len(all_pairs) - 10} more")
        else:
            print("[Solver] ℹ No reconvergence found. Solver will only enforce target requirement.")
        return pairs_by_reconv

    solver_module.HierarchicalReconvSolver._collect_and_sort_pairs = traced_collect_pairs

    # State for tracking detailed failure reasons
    last_failure_info = {"reason": "Unknown", "node": -1}

    # State for target tracking
    solving_target = {"node": -1, "val": -1}

    # 2. Trace Backtracking solver
    original_backward_justify = solver_module.HierarchicalReconvSolver._backward_justify
    recursion_depth = [0]

    def traced_backward_justify(self, queue, assignment, solved_pairs, pairs_by_reconv, seed=None):
        depth = recursion_depth[0]
        indent = "  " * depth

        if queue:
            gate = queue[0]
            val = assignment[gate]
            seed_info = f" [Seed: {seed}]" if seed is not None else ""
            target_str = f"(Target: {solving_target['node']}={solving_target['val']})"

            unsolved_pairs = []
            if gate in pairs_by_reconv:
                unsolved_pairs = [p for p in pairs_by_reconv[gate] if id(p) not in solved_pairs]

            if unsolved_pairs:
                stem = unsolved_pairs[0].get("start", unsolved_pairs[0].get("stem"))
                print(
                    f"\n{indent}┌────── [AI-Solve] "
                    f"Intersecting Path Pair: Stem {stem} -> Terminus {gate} "
                    f"{target_str}{seed_info}"
                )

                # Only show assignments relevant to the pair to avoid noise
                pair_nodes = {stem, gate}
                for path in unsolved_pairs[0]["paths"]:
                    pair_nodes.update(path)
                pair_assignments = {k: v for k, v in assignment.items() if k in pair_nodes}
                c_list = sorted([f"{k}:{v}" for k, v in pair_assignments.items()])
                constraints_summary = (
                    f"{len(pair_assignments)} target constraints in cone: " + ", ".join(c_list)
                )
            else:
                print(f"\n{indent}┌────── [Std-Backtrace] Handling gate {gate}={val}")
                constraints_summary = f"{len(assignment)} total assignments across trace"

            print(f"{indent}│   [State] Active Path: {constraints_summary}")
            recursion_depth[0] += 1
        else:
            print(f"\n{indent}┌────── [Base Case] Queue empty. Solution converged.")

        # Call original
        result = original_backward_justify(
            self, queue, assignment, solved_pairs, pairs_by_reconv, seed
        )

        if queue:
            recursion_depth[0] -= 1
            indent = "  " * recursion_depth[0]
            if result:
                print(f"{indent}└────── ✓ SOLVED node {queue[0]}")
            else:
                print(f"{indent}└────── ✗ FAILED node {queue[0]} (Backtracking)")
                reason = last_failure_info["reason"]
                node_info = (
                    f" at Node {last_failure_info['node']}"
                    if last_failure_info["node"] != -1
                    else ""
                )
                print(f"{indent}│   [Reason] {reason}{node_info}")
        else:
            if result:
                print(f"{indent}└────── ✓ Convergence Path Validated")
            else:
                print(f"{indent}└────── ✗ Convergence Path Invalid")

        return result

    solver_module.HierarchicalReconvSolver._backward_justify = traced_backward_justify

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
            print(f"{indent}│   [Inference] Model returned {len(candidates)} candidate(s).")

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
            print(f"{indent}│   [Inference] ✗ Model returned ZERO candidates (Total conflict).")

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
        solving_target["node"] = target_node
        solving_target["val"] = target_val
        print(f"\n[Solver] === AI Justification Started: Gate {target_node} = {target_val} ===")
        if constraints:
            c_list = sorted([f"{k}:{v}" for k, v in constraints.items()])
            if len(c_list) > 15:
                # Show first 10 and last 5
                constraints_str = ", ".join(c_list[:10]) + " ... " + ", ".join(c_list[-5:])
            else:
                constraints_str = ", ".join(c_list)
            print(f"[Solver] Current Constraints: {constraints_str}")

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
    original_justify_gate = solver_module.HierarchicalReconvSolver._justify_gate

    def traced_justify_gate(self, node, val, assignment):
        res = original_justify_gate(self, node, val, assignment)
        if res:
            pass  # we can skip printing options to reduce noise
        else:
            if recursion_depth[0] > 0:
                indent = "  " * recursion_depth[0]
                print(f"{indent}│   [Conflict] Gate {node}={val} logic cannot be justified.")
                last_failure_info["reason"] = f"Logic Conflict at Gate {node}={val}"
                last_failure_info["node"] = node
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
        # Target: Fault Activation (If s-a-0 or D, we want 1. If s-a-1 or DB, we want 0)
        activation_val = (
            LogicValue.ONE if fault.value in [LogicValue.ZERO, LogicValue.D] else LogicValue.ZERO
        )

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

            print(f"[DEBUG SIM] fault is {fault}, value: {type(fault.value)} {fault.value}")
            # Simulation propagates PIs through the cone.
            # DO NOT use topo_order, otherwise the logic_sim explicitly skips D-frontier updates
            logic_sim(circuit, total_gates, fault)

            actual_val = circuit[fault.gate_id].val
            print(f"[DEBUG SIM] Post logic_sim actual_val: {type(actual_val)} {actual_val}")
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
            print("\n" + "=" * 80)
            print("--- STARTING AI PROPAGATION LOOP ---")
            print("=" * 80)

            from src.atpg.logic_sim_three import d_frontier, fault_is_at_po, set_d_frontier_sort
            from src.atpg.podem import get_objective
            from src.atpg.util import (
                calculate_distance_to_primary_inputs,
                calculate_distance_to_primary_outputs,
            )

            # Need distance metrics for get_objective
            gate_distances_back = calculate_distance_to_primary_inputs(circuit, total_gates)
            gate_distances_fwd = calculate_distance_to_primary_outputs(circuit, total_gates)
            set_d_frontier_sort(gate_distances_fwd)
            # Make the module scoped distances match so get_objective works
            import src.atpg.podem as podem_module

            podem_module.gate_distances_back = gate_distances_back

            # State tracker for ALL PI assignments across Activation + Propagation
            all_pi_assignments = pi_assignments.copy()

            prop_step = 1
            while not fault_is_at_po(circuit, total_gates):
                print(f"\n[AI-Prop Step {prop_step}] Checking D-Frontier...")

                if d_frontier.is_empty():
                    print(
                        f"[AI-Prop Step {prop_step}] ✗ FAILURE: D-Frontier is empty, "
                        "fault cannot be propagated."
                    )
                    break

                d_front_gate = d_frontier.get_first()
                print(f"[AI-Prop Step {prop_step}] Selected D-Frontier Gate: {d_front_gate}")

                # We use standard get_objective which picks the X-fanin closest to PIs
                prop_objective = get_objective(circuit, fault)

                if prop_objective.gate_id == -1:
                    print(
                        f"[AI-Prop Step {prop_step}] ✗ FAILURE: D-Frontier gate "
                        f"{d_front_gate} has no eligible X-fanins."
                    )
                    break

                print(
                    f"[AI-Prop Step {prop_step}] Target Objective via SCOAP: "
                    f"Gate {prop_objective.gate_id} = {prop_objective.value}"
                )

                # --- Unleash AI Solver on the single objective ---
                print(f"[AI-Prop Step {prop_step}] Delegating objective to AI Solver...")
                # Natively find PIs, while strictly enforcing previously found PIs
                prop_assignment = solver.solve(
                    prop_objective.gate_id, prop_objective.value, constraints=all_pi_assignments
                )

                if not prop_assignment:
                    print(
                        f"[AI-Prop Step {prop_step}] ✗ AI Solver could not justify "
                        f"Gate {prop_objective.gate_id}={prop_objective.value}. Falling back to "
                        "standard backtrace."
                    )
                    res_fault = original_simple_backtrace(prop_objective, circuit)
                    pi, val = res_fault.gate_id, res_fault.value
                    if pi != -1 and pi not in all_pi_assignments:
                        new_pi_assignments = {pi: val}
                    else:
                        print(
                            f"[AI-Prop Step {prop_step}] ✗ Standard fallback also failed or "
                            "yielded no new PI."
                        )
                        break
                else:
                    # Extract the newly assigned PIs (excluding ones we already had)
                    new_pi_assignments = {
                        gid: val
                        for gid, val in prop_assignment.items()
                        if circuit[gid].type == GateType.INPT and gid not in all_pi_assignments
                    }

                all_pi_assignments.update(new_pi_assignments)

                print(
                    f"[AI-Prop Step {prop_step}] AI Solver produced PI assignments: "
                    f"{new_pi_assignments}"
                )

                # Apply and Simulate
                for gid, val in all_pi_assignments.items():
                    circuit[gid].val = val

                print(f"[AI-Prop Step {prop_step}] Simulating circuit for Forward-Prop...")
                logic_sim(circuit, total_gates, fault)

                prop_step += 1

            if fault_is_at_po(circuit, total_gates):
                print("\n" + "=" * 80)
                print(
                    "FINAL RESULT: SUCCESS - Fault propagated to Primary Output "
                    f"in {prop_step - 1} steps."
                )
                print("=" * 80)
            else:
                print("\n" + "=" * 80)
                print(f"FINAL RESULT: FAILURE - Propagation stalled after {prop_step - 1} steps.")
                print("=" * 80)

        else:
            print("\n" + "=" * 80)
            print("AI ACTIVATION FAILURE")
            print("=" * 80)

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
