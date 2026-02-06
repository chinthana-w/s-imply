
import os
import sys
import time
import pandas as pd
from typing import List, Dict
import multiprocessing

# Add root to sys.path
sys.path.append(os.getcwd())

from src.util.io import parse_bench_file
from src.atpg.podem import podem, get_all_faults, initialize, reset_statistics, get_statistics, simple_backtrace
from src.atpg.logic_sim_three import reset_gates

def run_benchmark_for_circuit(bench_path: str, timeout_per_fault: float = 10.0):
    circuit, total_gates = parse_bench_file(bench_path)
    faults = get_all_faults(circuit, total_gates)
    total_faults = len(faults)
    
    # Initialize circuit once (calculates distance maps and topological order)
    initialize(circuit, total_gates)
    
    detected_count = 0
    total_time = 0
    total_backtracks = 0
    timeouts = 0
    
    start_circuit_time = time.time()
    
    for i, fault in enumerate(faults):
        reset_gates(circuit, total_gates)
        reset_statistics()
        
        start_fault_time = time.time()
        
        # Pass timeout to podem
        detected = podem(circuit, fault, total_gates, backtrace_func=simple_backtrace, timeout=timeout_per_fault)
        
        elapsed = time.time() - start_fault_time
        total_time += elapsed
        
        if detected:
            detected_count += 1
        else:
            if elapsed >= timeout_per_fault:
                timeouts += 1
        
        stats = get_statistics()
        total_backtracks += stats.get("backtrack_count", 0)

    avg_time_ms = (total_time / total_faults * 1000) if total_faults > 0 else 0
    print(f"Finished {os.path.basename(bench_path)}: FC={(detected_count/total_faults*100):.2f}%, Time={total_time:.2f}s, Avg={avg_time_ms:.2f}ms")

    return {
        "Circuit": os.path.basename(bench_path),
        "Total Faults": total_faults,
        "Detected": detected_count,
        "Coverage (%)": (detected_count / total_faults * 100) if total_faults > 0 else 0,
        "Time (s)": total_time,
        "Avg Time/Fault (ms)": avg_time_ms,
        "Backtracks": total_backtracks,
        "Timeouts": timeouts
    }

def main():
    bench_dir = "data/bench/ISCAS85"
    bench_files = [
        "c17.bench", "c432.bench", "c499.bench", "c880.bench", 
        "c1355.bench", "c1908.bench", "c2670.bench", "c3540.bench", 
        "c5315.bench", "c6288.bench", "c7552.bench"
    ]
    
    print(f"Starting Parallel Vanilla ISCAS85 Benchmark with 10s timeout per fault...")
    start_all = time.time()
    
    all_paths = [os.path.join(bench_dir, bf) for bf in bench_files if os.path.exists(os.path.join(bench_dir, bf))]
    
    # Run circuits in parallel
    pool = multiprocessing.Pool(processes=min(len(all_paths), multiprocessing.cpu_count()))
    results = pool.map(run_benchmark_for_circuit, all_paths)
    pool.close()
    pool.join()
    
    # Create Markdown report
    df = pd.DataFrame(results)
    
    # Format floats
    df_disp = df.copy()
    df_disp['Coverage (%)'] = df_disp['Coverage (%)'].map('{:,.2f}'.format)
    df_disp['Time (s)'] = df_disp['Time (s)'].map('{:,.2f}'.format)
    df_disp['Avg Time/Fault (ms)'] = df_disp['Avg Time/Fault (ms)'].map('{:,.2f}'.format)
    
    md_table = df_disp.to_markdown(index=False)
    
    total_f = df['Total Faults'].sum()
    total_d = df['Detected'].sum()
    
    report = f"""# ISCAS85 Vanilla PODEM Benchmark Results
**Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}**
**Per-Fault Timeout: 10s**
**Optimization: Single-pass Logic Simulation with Topological Order**

{md_table}

## Summary
- **Total Circuits**: {len(results)}
- **Total Faults**: {total_f}
- **Total Detected**: {total_d}
- **Overall Coverage**: {(total_d / total_f * 100):.2f}%
- **Total Duration**: {(time.time() - start_all):.2f}s
"""
    
    with open("vanilla_iscas85_results.md", "w") as f:
        f.write(report)
    
    print("\nBenchmark Complete! Results saved to vanilla_iscas85_results.md")

if __name__ == "__main__":
    main()
