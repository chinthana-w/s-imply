import os
import time

import pandas as pd
import torch

from src.atpg.ai_podem import (
    AiPodemConfig,
    HierarchicalReconvSolver,
    ModelPairPredictor,
    ai_podem,
)
from src.atpg.logic_sim_three import reset_gates
from src.atpg.podem import (
    get_all_faults,
    get_statistics,
    initialize,
    podem,
    reset_statistics,
    simple_backtrace,
)
from src.util.io import parse_bench_file

def main():
    bench_path = "data/bench/ISCAS85/c1908.bench"
    model_path = "checkpoints/supervised_v5/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Analyzing AI Success Subset for: {os.path.basename(bench_path)}")
    print(f"Model: {model_path}")

    circuit, total_gates = parse_bench_file(bench_path)
    all_faults = get_all_faults(circuit, total_gates)
    total_all = len(all_faults)
    print(f"Total Faults in Circuit: {total_all}")

    # Initialize AI
    config = AiPodemConfig(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_ai_activation=True,
        enable_ai_propagation=False,
        verbose=False
    )
    predictor = ModelPairPredictor(circuit, bench_path, config)
    solver = HierarchicalReconvSolver(circuit, predictor)

    ai_success_data = []

    print("\nStep 1: Running AI (No Fallback) to find successful subset...")
    
    for i, fault in enumerate(all_faults):
        reset_gates(circuit, total_gates)
        initialize(circuit, total_gates)
        
        start_ai = time.time()
        detected = ai_podem(
            circuit,
            fault,
            total_gates,
            predictor=predictor,
            solver=solver,
            enable_ai_activation=True,
            enable_ai_propagation=False,
            no_fallback=True
        )
        elapsed_ai = time.time() - start_ai
        
        if detected:
            ai_success_data.append({
                "fault": fault,
                "ai_time_ms": elapsed_ai * 1000
            })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{total_all} faults... Found {len(ai_success_data)} AI successes.")

    num_success = len(ai_success_data)
    print(f"\nAI Success Subset Size: {num_success} ({ (num_success/total_all*100):.2f}%)")

    if num_success == 0:
        print("No AI successes found. Exiting.")
        return

    print("\nStep 2: Rerunning Vanilla PODEM on AI Success Subset...")
    
    final_results = []
    
    for i, data in enumerate(ai_success_data):
        fault = data["fault"]
        
        reset_gates(circuit, total_gates)
        initialize(circuit, total_gates)
        reset_statistics()
        
        start_vanilla = time.time()
        detected = podem(circuit, fault, total_gates, backtrace_func=simple_backtrace)
        elapsed_vanilla = time.time() - start_vanilla
        stats = get_statistics()
        
        final_results.append({
            "fault_id": fault.gate_id,
            "fault_val": "D" if fault.value == 3 else "DB",
            "ai_time_ms": data["ai_time_ms"],
            "vanilla_time_ms": elapsed_vanilla * 1000,
            "vanilla_backtracks": stats.get("backtrack_count", 0)
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_success} vanilla reruns...")

    # Report
    df = pd.DataFrame(final_results)
    
    print("\n=== AI Success Subset Analysis Report (c1908) ===")
    print(f"Total Faults Studied: {num_success}")
    
    summary = {
        "Avg AI Time (ms)": df["ai_time_ms"].mean(),
        "Avg Vanilla Time (ms)": df["vanilla_time_ms"].mean(),
        "Avg Vanilla Backtracks": df["vanilla_backtracks"].mean(),
        "Max Vanilla Backtracks": df["vanilla_backtracks"].max(),
        "Min Vanilla Backtracks": df["vanilla_backtracks"].min(),
        "Total Vanilla Backtracks": df["vanilla_backtracks"].sum()
    }
    
    for k, v in summary.items():
        print(f"{k:25}: {v:.2f}")

    # Output detailed CSV
    output_file = "ai_success_subset_c1908.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed report saved to {output_file}")

if __name__ == "__main__":
    main()
