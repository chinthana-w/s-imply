import argparse
import os
import random
import resource
import sys
from typing import List

import torch
from tqdm import tqdm

from src.atpg.ai_podem import ModelPairPredictor, ai_podem
from src.atpg.logic_sim_three import reset_gates
from src.atpg.podem import get_all_faults, get_statistics, reset_statistics
from src.atpg.recursive_reconv_solver import HierarchicalReconvSolver
from src.ml.rl.rl_recorder import ExperienceRecorder
from src.util.io import parse_bench_file

# Increase recursion limit and stack size for deep circuit solving
sys.setrecursionlimit(100000)
try:
    resource.setrlimit(resource.RLIMIT_STACK, (256 * 1024 * 1024, resource.RLIM_INFINITY))
except Exception:
    pass


def collect_experience(
    bench_dirs: List[str] = ["data/bench/ISCAS85"],
    model_path: str = "checkpoints/unlinked_candidate/best_model.pth",
    output_dir: str = "data/rl_experience",
    max_faults_per_circuit: int = 50,
    exploration_attempts: int = 10,
    gpu_id: int = 0,
):
    """
    Run AI-PODEM on benchmarks to collect RL experience, emphasizing
    exploration using random seeds for assignments.
    """
    os.makedirs(output_dir, exist_ok=True)

    benchmarks = []
    for bench_dir in bench_dirs:
        print(f"Checking bench_dir: {bench_dir}")
        if os.path.exists(bench_dir) and os.path.isdir(bench_dir):
            files = [(bench_dir, f) for f in os.listdir(bench_dir) if f.endswith(".bench")]
            benchmarks.extend(files)
        else:
            print(f"Benchmark directory {bench_dir} not found.")

    if not benchmarks:
        print("No benchmarks found.")
        return

    benchmarks.sort(key=lambda x: x[1])

    recorder = ExperienceRecorder(save_dir=output_dir)

    # Global stats
    total_episodes = 0
    total_success = 0

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    for bench_dir, bench_file in benchmarks:
        print(f"Processing {bench_file} (from {bench_dir})...")
        circuit_path = os.path.join(bench_dir, bench_file)

        try:
            circuit, total_gates = parse_bench_file(circuit_path)
        except Exception as e:
            print(f"Failed to parse {bench_file}: {e}")
            continue

        faults = get_all_faults(circuit, total_gates)
        if not faults:
            continue

        # Shuffle and select faults uniformly
        random.shuffle(faults)
        selected_faults = faults[:max_faults_per_circuit]

        try:
            from src.atpg.ai_podem import AiPodemConfig

            cfg = AiPodemConfig(
                model_path=model_path,
                device=device,
                enable_ai_activation=True,
                enable_ai_propagation=True,
                verbose=False,
            )
            predictor = ModelPairPredictor(circuit, circuit_path, cfg)
        except Exception as e:
            print(f"Failed to load predictor for {bench_file}: {e}")
            continue

        pbar = tqdm(selected_faults, desc=f"Faults ({bench_file})")
        for fault in pbar:
            # For each fault, attempt aggressive exploration across multiple seeds
            # Each seed is treated as an RL episode
            for attempt in range(exploration_attempts):
                seed = random.randint(0, 10000)

                try:
                    # Fresh start for the attempt
                    reset_gates(circuit, total_gates)
                    reset_statistics()

                    # Note: We create a fresh solver per episode so it doesn't hold old caches
                    solver = HierarchicalReconvSolver(circuit, predictor, recorder=recorder)

                    # Start Episode
                    episode_id = f"{bench_file}_{fault.gate_id}_{fault.value}_s{seed}"
                    recorder.start_episode(episode_id)

                    # Note: Need ai_podem to accept seed and pass it to solver.
                    # As a temporary workaround if it doesn't, solver takes seed directly
                    # in its API.
                    # We will rely on ai_podem's internal try loop but cap it to 1 attempt to force
                    # the exploration loop here to record episodes independently.

                    # For data gathering, we only use AI paths (no clean fallback if we want
                    # purely AI data)
                    print(f"    -> Running ai_podem...", flush=True)
                    success = ai_podem(
                        circuit=circuit,
                        fault=fault,
                        total_gates=total_gates,
                        circuit_path=circuit_path,
                        predictor=predictor,
                        solver=solver,
                        enable_ai_activation=True,
                        enable_ai_propagation=True,
                        verbose=False,
                        seed=seed,
                    )
                    print(f"    -> Finished ai_podem (success={success}).", flush=True)

                    stats = get_statistics()
                    bt_count = stats.get("backtrack_count", 0)

                    # Reward shaping: Base reward + backtracking penalty
                    # Reward success heavily, penalize timeouts/huge backtracks
                    if success:
                        final_reward = 10.0 - (bt_count * 0.001)  # Penalize inefficient success
                    else:
                        final_reward = -5.0 - (bt_count * 0.001)

                    recorder.finish_episode(final_reward=final_reward)

                    total_episodes += 1
                    if success:
                        total_success += 1

                except Exception as e:
                    print(f"  Error processing fault {fault.gate_id} attempt {attempt}: {e}")
                    recorder.finish_episode(final_reward=-10.0)
                    total_episodes += 1

                # Periodic save
                if total_episodes % 20 == 0:
                    recorder.save_buffer()

                pbar.set_postfix({"succ": total_success, "eps": total_episodes})

    # Final save
    recorder.save_buffer()
    print(f"Collection complete. {total_episodes} episodes, {total_success} successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench_dirs",
        nargs="+",
        default=["data/bench/ISCAS85"],
        help="Directories with .bench files",
    )
    parser.add_argument(
        "--model",
        default="checkpoints/unlinked_candidate/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--max_faults", type=int, default=50, help="Max faults per circuit")
    parser.add_argument(
        "--exploration", type=int, default=10, help="Random exploration attempts per fault"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()

    collect_experience(
        bench_dirs=args.bench_dirs,
        model_path=args.model,
        max_faults_per_circuit=args.max_faults,
        exploration_attempts=args.exploration,
        gpu_id=args.gpu,
    )
