import os
import sys
import time
import pandas as pd
import torch

# Add root to sys.path
sys.path.append(os.getcwd())

from src.atpg.ai_podem import AiPodemConfig, HierarchicalReconvSolver, ModelPairPredictor, ai_podem
from src.atpg.logic_sim_three import reset_gates
from src.atpg.podem import (
    Fault,
    LogicValue,
    get_statistics,
    initialize,
    podem,
    reset_statistics,
    simple_backtrace,
)
from src.util.io import parse_bench_file

def main():
    bench_path = "data/bench/ISCAS85/c1908.bench"
    
    # Faults identified from previous analysis
    hard_faults = [
        {"id": 1163, "val": LogicValue.DB},
        {"id": 1167, "val": LogicValue.DB}
    ]

    circuit, total_gates = parse_bench_file(bench_path)
    
    results = []

    print(f"Rerunning {len(hard_faults)} hard faults with 100,000 backtrack limit...")

    for hf in hard_faults:
        fault = Fault(hf["id"], hf["val"])
        
        # 1. Run Vanilla with high limit
        reset_gates(circuit, total_gates)
        initialize(circuit, total_gates)
        reset_statistics()
        
        start_v = time.time()
        detected_v = podem(circuit, fault, total_gates, backtrace_func=simple_backtrace, max_backtracks=100000)
        elapsed_v = time.time() - start_v
        stats_v = get_statistics()
        
        print(f"Fault {hf['id']} Vanilla: Detected={detected_v}, Backtracks={stats_v.get('backtrack_count', 0)}, Time={elapsed_v*1000:.2f}ms")
        
        # 2. Re-verify AI (for reference/timing consistency)
        # Note: We already know AI solves these, but let's get a fresh timing in this environment
        model_path = "checkpoints/supervised_v5/best_model.pth"
        config = AiPodemConfig(
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            enable_ai_activation=True,
            enable_ai_propagation=False
        )
        predictor = ModelPairPredictor(circuit, bench_path, config)
        solver = HierarchicalReconvSolver(circuit, predictor)
        
        reset_gates(circuit, total_gates)
        initialize(circuit, total_gates)
        
        start_ai = time.time()
        detected_ai = ai_podem(
            circuit, fault, total_gates, 
            predictor=predictor, solver=solver, 
            enable_ai_activation=True, no_fallback=True
        )
        elapsed_ai = time.time() - start_ai
        
        results.append({
            "Fault ID": hf["id"],
            "Type": "DB",
            "Vanilla Detected": "Yes" if detected_v else "No",
            "Vanilla Backtracks": stats_v.get("backtrack_count", 0),
            "Vanilla Time (ms)": elapsed_v * 1000,
            "AI Detected": "Yes" if detected_ai else "No",
            "AI Time (ms)": elapsed_ai * 1000
        })

    df = pd.DataFrame(results)
    print("\n" + df.to_markdown(index=False))

if __name__ == "__main__":
    main()
