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
                print(f"  {i+1}. Stem {stem} -> Reconv {reconv} " f"(Total Path Len: {p_len})")
            if len(pairs) > 10:
                print(f"  ... +{len(pairs)-10} more")
        else:
            print(
                "[Solver] ℹ No reconvergence found. " "Solver will only enforce target requirement."
            )
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
                f"\n{indent}┌────── [Step {pair_idx+1}/{len(pairs)}] "
                f"Solving Pair: Stem {stem} -> Reconv {reconv}{seed_info}"
            )
            print(f"{indent}│   [State] Active Constraints: " f"{len(current_constraints)} nodes")
            recursion_depth[0] += 1
        else:
            print(
                f"\n{indent}┌────── [Base Case] All pairs processed. "
                f"Checking final consistency..."
            )

        # Call original
        result = original_solve_recursive(self, pair_idx, pairs, current_constraints, seed)

        if pair_idx < len(pairs):
            recursion_depth[0] -= 1
            indent = "  " * recursion_depth[0]
            if result:
                print(f"{indent}└────── ✓ SOLVED Step {pair_idx+1}")
            else:
                print(f"{indent}└────── ✗ FAILED Step {pair_idx+1} (Backtracking)")
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

        print(f"{indent}│   [Inference] Querying AI Model for " f"Stem {stem} -> Reconv {reconv}")

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
            print(f"{indent}│   [Inference] Model returned " f"{n_cands} candidate(s){tag}.")

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
                print(f"{indent}│   [Inference] Path {p_idx}: " f"[{' -> '.join(path_vals)}]")
        else:
            print(f"{indent}│   [Inference] ⚠ Model returned " f"NO candidates (logical conflict).")

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
                f"    [PODEM] Decision: Obj(Gate {objective.gate_id}="
                f"{objective.value}) -> ✗ FAILED"
            )
        return res

    podem_module.simple_backtrace = traced_simple_backtrace

    original_podem_recursion = podem_module.podem_recursion
    podem_stats = {"calls": 0, "backtracks": 0}

    def traced_podem_recursion(circuit, total_gates, fault):
        podem_stats["calls"] += 1
        res = original_podem_recursion(circuit, total_gates, fault)
        if res == 0:  # UNTESTABLE / Backtrack
            podem_stats["backtracks"] += 1
            # Only print every 10th backtrack if there are many
            if podem_stats["backtracks"] % 10 == 0:
                print(f"    [PODEM] Info: Backtrack count = {podem_stats['backtracks']}")
        return res

    podem_module.podem_recursion = traced_podem_recursion

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
        print(f"\n[Solver] === AI Justification Started: " f"Gate {target_node} = {target_val} ===")
        if constraints:
            print(f"[Solver] Current Constraints: {list(constraints.keys())}")

        result = original_solve_main(self, target_node, target_val, constraints, seed)

        if result:
            print(f"\n    {'─'*40}")
            print("    LOGIC CONSISTENCY VERIFICATION")
            print(f"    {'─'*40}")

            from src.atpg.logic_sim_three import compute_gate_value

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

                # Temporarily set for computation
                saved_vals = {fin: circuit[fin].val for fin in gate.fin}
                for fin in gate.fin:
                    circuit[fin].val = result[fin]
                computed_val = compute_gate_value(circuit, gate)
                for fin, val in saved_vals.items():
                    circuit[fin].val = val

                if computed_val != expected_val:
                    inconsistencies.append({"gate": gid, "exp": expected_val, "got": computed_val})

            if inconsistencies:
                print(f"    ✗ FAILED: {len(inconsistencies)} inconsistencies found.")
                for inc in inconsistencies[:3]:
                    print(f"      Gate {inc['gate']}: Model={inc['exp']}, " f"Logic={inc['got']}")
            else:
                print(f"    ✓ SUCCESS: All {len(result)} assignments are " f"logically consistent.")

            print(f"    {'─'*40}\n")
        else:
            print("\n[Solver] === AI Justification FAILED ===\n")

        return result

    solver_module.HierarchicalReconvSolver.solve = traced_solve_with_verification

    print("=" * 80)
    print("RUNNING 100 AI PODEM CYCLES (AI Activation ONLY - Detailed Trace)")
    print("=" * 80 + "\n")

    # patterns = [] # Unused

    # Reset circuit
    from src.atpg.podem import initialize

    initialize(circuit, total_gates)

    try:
        result = ai_podem(
            circuit,
            fault,
            total_gates,
            circuit_path=circuit_path,
            predictor=predictor,
            enable_ai_activation=True,
            enable_ai_propagation=True,  # AI activation and propagation
            verbose=True,
        )

        if result:
            print("\n" + "=" * 80)
            print("FINAL RESULT: SUCCESS - Found pattern")
            print(f"Pattern: {result}")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("FINAL RESULT: FAILURE - No pattern found")
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
