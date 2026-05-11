import os
import sys
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.atpg.recursive_reconv_solver import HierarchicalReconvSolver
from src.util.io import parse_bench_file

def main():
    bench_path = "data/bench/ITC99/b17.bench"
    print(f"Precomputing reconvergent pairs for {bench_path}")
    
    circuit, total_gates = parse_bench_file(bench_path)
    
    # We don't need a real predictor for topology collection
    solver = HierarchicalReconvSolver(circuit, None, circuit_path=bench_path)
    
    # Only pick some nodes to avoid massive file size
    # Or just run it for all and see how it goes
    for i in tqdm(range(total_gates)):
        if i not in solver.pair_cache:
            try:
                # This will populate the cache and mark it dirty
                solver.pair_cache[i] = solver._collect_and_sort_pairs(i)
                solver._pair_cache_dirty = True
            except Exception:
                pass
        
        # Periodically save
        if i % 100 == 0:
            solver._persist_pair_cache_if_needed()

    solver._persist_pair_cache_if_needed()
    print("Done.")

if __name__ == "__main__":
    main()
