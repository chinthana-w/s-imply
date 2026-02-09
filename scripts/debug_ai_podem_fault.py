
import os
import sys
import torch

# Add root to sys.path
sys.path.append(os.getcwd())

from src.util.io import parse_bench_file
from src.atpg.podem import get_all_faults, set_trace_decisions, initialize
from src.atpg.ai_podem import ai_podem

def debug_fault():
    bench_path = "data/bench/ISCAS85/c432.bench"
    model_path = "checkpoints/reconv_optimized/best_model.pth"
    
    if not os.path.exists(model_path):
        model_path = "checkpoints/reconv_minimal_model.pt"

    print(f"Debug run for {bench_path} Fault #4")
    
    circuit, total_gates = parse_bench_file(bench_path)
    
    # Initialize stuff
    initialize(circuit, total_gates)
    
    faults = get_all_faults(circuit, total_gates)
    target_fault = faults[4]
    
    print(f"Target Fault: {target_fault}")
    
    # Enable verbose tracing
    set_trace_decisions(True)
    
    print("\n--- Starting AI-PODEM (Activation + Propagation) ---")
    detected = ai_podem(
        circuit, 
        target_fault, 
        total_gates, 
        model_path=model_path,
        circuit_path=bench_path,
        enable_ai_activation=True,
        enable_ai_propagation=True,
        verbose=True
    )
    
    print(f"\nFinal Result: {'DETECTED' if detected else 'UNDETECTED'}")

if __name__ == "__main__":
    debug_fault()
