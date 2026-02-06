
import os
import sys
import time
import torch
import pandas as pd
from typing import List, Dict, Any

# Add root to sys.path
sys.path.append(os.getcwd())

from src.util.struct import Gate, Fault, LogicValue
from src.util.io import parse_bench_file
from src.atpg.podem import get_all_faults, reset_statistics, get_statistics, initialize, podem, simple_backtrace
from src.atpg.ai_podem import ai_podem, ModelPairPredictor, HierarchicalReconvSolver
from src.atpg.logic_sim_three import reset_gates

def run_benchmark(bench_path: str, model_path: str, mode: str = "vanilla", limit_faults: int = 0):
    """
    Run PODEM on a single bench file in specified mode.
    Modes: vanilla, ai_activation, ai_all
    """
    circuit, total_gates = parse_bench_file(bench_path)
    faults = get_all_faults(circuit, total_gates)
    
    if limit_faults > 0:
        faults = faults[:limit_faults]
    
    total = len(faults)
    
    # Initialize Predictor/Solver if in AI mode
    predictor = None
    solver = None
    if mode in ("ai_activation", "ai_all"):
        print(f"  [{mode}] Initializing AI Model/Predictor/Solver...")
        predictor = ModelPairPredictor(bench_path, model_path, circuit)
        solver = HierarchicalReconvSolver(circuit, predictor)
    
    reset_statistics()
    start_time = time.time()
    
    succ = 0
    
    for i, fault in enumerate(faults):
        # ALWAYS reset gates and PODEM structures before each fault
        reset_gates(circuit, total_gates)
        initialize(circuit, total_gates)
        
        if mode == "vanilla":
            detected = podem(circuit, fault, total_gates, backtrace_func=simple_backtrace)
        elif mode == "ai_activation":
            detected = ai_podem(circuit, fault, total_gates, 
                                model_path=model_path, 
                                circuit_path=bench_path,
                                enable_ai_activation=True,
                                enable_ai_propagation=False,
                                predictor=predictor,
                                solver=solver)
        elif mode == "ai_all":
            detected = ai_podem(circuit, fault, total_gates, 
                                model_path=model_path, 
                                circuit_path=bench_path,
                                enable_ai_activation=True,
                                enable_ai_propagation=True,
                                predictor=predictor,
                                solver=solver)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        if detected:
            succ += 1
            
        # More frequent status reporting
        if (i+1) % 10 == 0 or total < 10:
            print(f"  [{mode}] {i+1}/{total} faults processed... FC={(succ/(i+1)*100):.1f}%")

    elapsed = time.time() - start_time
    stats = get_statistics()
    
    return {
        "bench": os.path.basename(bench_path),
        "mode": mode,
        "faults": total,
        "detected": succ,
        "coverage": (succ / total * 100) if total > 0 else 0,
        "time": elapsed,
        "backtracks": stats["backtrack_count"],
        "backtraces": stats["backtrace_count"],
        "avg_time_ms": (elapsed * 1000) / total if total > 0 else 0
    }

def main():
    # Use best model
    model_path = "checkpoints/reconv_optimized/best_model.pth"
    if not os.path.exists(model_path):
        model_path = "checkpoints/reconv_minimal_model.pt"
        
    bench_dir = "data/bench/ISCAS85"
    # c17 for quick test, c432 for real test
    bench_files = ["c17.bench", "c432.bench", "c2670.bench"]
    
    # Process all faults
    LIMIT = 0 
    
    print(f"Using Model: {model_path}")
    print(f"Fault Limit per circuit: {LIMIT}")
    
    all_results = []
    
    for bfile in bench_files:
        path = os.path.join(bench_dir, bfile)
        if not os.path.exists(path):
            print(f"Skipping {bfile}, not found.")
            continue
            
        print(f"\n>>> Benchmarking {bfile} <<<")
        
        # 1. Vanilla PODEM
        print(f"Running Vanilla...")
        res_v = run_benchmark(path, model_path, mode="vanilla", limit_faults=LIMIT)
        all_results.append(res_v)
        
        # 2. AI Activation
        print(f"Running AI-Activation...")
        res_ai_act = run_benchmark(path, model_path, mode="ai_activation", limit_faults=LIMIT)
        all_results.append(res_ai_act)
        # 3. AI All (Activation + Propagation)
        print(f"Running AI-All...")
        res_ai_all = run_benchmark(path, model_path, mode="ai_all", limit_faults=LIMIT)
        all_results.append(res_ai_all)
        
    # Save Results
    df = pd.DataFrame(all_results)
    print("\n=== Benchmark Summary ===")
    summary = df[["bench", "mode", "coverage", "time", "backtracks", "avg_time_ms"]]
    print(summary.to_string(index=False))
    
    df.to_csv("ai_podem_benchmark_results.csv", index=False)
    print("\nFull results saved to ai_podem_benchmark_results.csv")

if __name__ == "__main__":
    main()
