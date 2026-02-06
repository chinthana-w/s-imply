
import os
import sys
import time
import pandas as pd
import torch

# Add root to sys.path
sys.path.append(os.getcwd())

from src.util.io import parse_bench_file
from src.atpg.podem import get_all_faults, reset_statistics, get_statistics, initialize, podem, simple_backtrace
from src.atpg.ai_podem import ai_podem, ModelPairPredictor, HierarchicalReconvSolver
from src.atpg.logic_sim_three import reset_gates

def run_mode(bench_path, model_path, mode, faults, circuit, total_gates):
    print(f"\n--- Running Mode: {mode} ---")
    reset_statistics()
    start_time = time.time()
    
    succ = 0
    total = len(faults)
    
    # Initialize AI components if needed
    predictor = None
    solver = None
    if mode in ("ai_activation", "ai_all"):
        print(f"  [{mode}] Initializing AI Model...")
        try:
            predictor = ModelPairPredictor(bench_path, model_path, circuit)
            solver = HierarchicalReconvSolver(circuit, predictor)
        except Exception as e:
            print(f"  [ERROR] Failed to initialize AI: {e}")
            return None

    for i, fault in enumerate(faults):
        # Reset per fault
        reset_gates(circuit, total_gates)
        reset_statistics()
        # Initialize calculates distance maps. Costly? 
        # For vanilla, we did it once per circuit. 
        # But `podem` might modify structures? 
        # Let's trust `initialize` is fast enough or needed.
        # Actually in parallel script we did it once. 
        # But `ai_podem` might rely on specific initialization?
        # Let's keep it safe: initialize once per circuit, but reset gates per fault.
        # wait, `initialize` resets distance maps. We should do it once OUTSIDE loop if possible.
        # But `podem` code uses global vars.
        # The safest for consistent comparison is to match `benchmark_ai_podem.py`:
        # It calls `initialize` inside the loop. Let's optimize that if we can, 
        # but for fairness with previous numbers, let's stick to the loop for now 
        # UNLESS we are sure. 
        # Actually `benchmark_ai_podem.py` DOES call initialize inside the loop.
        initialize(circuit, total_gates)
        
        detected = False
        try:
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
        except Exception as e:
            print(f"  [Error] Fault {i} ({fault}): {e}")
            
        if detected:
            succ += 1
            
        if (i+1) % 50 == 0:
            print(f"  Progress: {i+1}/{total} - FC={(succ/(i+1)*100):.2f}%")

    elapsed = time.time() - start_time
    stats = get_statistics()
    
    return {
        "Mode": mode,
        "Coverage (%)": (succ / total * 100) if total > 0 else 0,
        "Time (s)": elapsed,
        "Avg Time (ms)": (elapsed * 1000) / total if total > 0 else 0,
        "Backtracks": stats.get("backtrack_count", 0)
    }

def main():
    bench_path = "data/bench/ISCAS85/c432.bench"
    model_path = "checkpoints/reconv_optimized/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, trying minimal model...")
        model_path = "checkpoints/reconv_minimal_model.pt"
        
    print(f"Benchmarking c432.bench")
    print(f"Model: {model_path}")
    
    circuit, total_gates = parse_bench_file(bench_path)
    faults = get_all_faults(circuit, total_gates)
    print(f"Total Faults: {len(faults)}")
    
    results = []
    
    # 1. Vanilla
    res = run_mode(bench_path, model_path, "vanilla", faults, circuit, total_gates)
    if res: results.append(res)
    
    # 2. AI Activation
    res = run_mode(bench_path, model_path, "ai_activation", faults, circuit, total_gates)
    if res: results.append(res)
    
    # 3. AI All
    res = run_mode(bench_path, model_path, "ai_all", faults, circuit, total_gates)
    if res: results.append(res)
    
    # Report
    print("\n=== c432 Comparison Results ===")
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
