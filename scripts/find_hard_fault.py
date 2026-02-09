import os
import sys
import time

# Add root to sys.path
sys.path.append(os.getcwd())

from src.util.io import parse_bench_file
from src.atpg.podem import get_all_faults, reset_statistics, initialize, podem, simple_backtrace

def find_hard_fault():
    bench_path = "data/bench/ISCAS85/c432.bench"
    print(f"Scanning {bench_path} for hard faults (Vanilla PODEM)...")
    
    circuit, total_gates = parse_bench_file(bench_path)
    faults = get_all_faults(circuit, total_gates)
    
    # Sort faults to hit hard ones? Or just linear scan?
    # Linear scan is fine, but maybe reverse if hard ones are late? 
    # c432 hard faults are usually related to reconvergence.
    
    initialize(circuit, total_gates)
    
    hard_faults = []
    
    for i, fault in enumerate(faults):
        # We want to find a fault that takes > 0.1s or fails.
        # Strict timeout to find "interesting" ones quickly.
        
        # Reset
        for g in circuit:
            g.val = 2 # XD
            
        start = time.time()
        # Use a short timeout to catch "slow" faults which likely means hard
        result = podem(circuit, fault, total_gates, backtrace_func=simple_backtrace, timeout=0.1)
        elapsed = time.time() - start
        
        if not result:
            print(f"Found HARD Fault [{i}]: {fault} (Result: {result}, Time: {elapsed:.4f}s)")
            return fault
            
        if elapsed > 0.05:
             print(f"Found SLOW Fault [{i}]: {fault} (Result: {result}, Time: {elapsed:.4f}s)")
             # We prefer a failure, but a slow one is also good for trace.
             
    print("No hard faults found with current parameters.")
    return None

if __name__ == "__main__":
    find_hard_fault()
