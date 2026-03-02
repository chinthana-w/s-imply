#!/usr/bin/env python
"""
Unified RL Training Pipeline
Allows running individual stages or the full pipeline with CLI arguments
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import psutil

# import torch (Moved inside functions to save memory)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_experience_collection(args):
    """Stage 1: Collect experience data from AI-PODEM runs"""
    print_section("STAGE 1: Experience Collection")

    import torch

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    available_ram_gb = psutil.virtual_memory().available / (1024**3)

    if num_gpus > 1:
        # Avoid OOM by capping processes if RAM is low
        # Assume each collection process needs ~5GB (DeepGate + Model + Circuit)
        ram_capped_gpus = int(available_ram_gb // 5)
        if ram_capped_gpus < num_gpus:
            print(
                f"  [Warning] Low RAM ({available_ram_gb:.1f} GB). "
                f"Capping collection to {ram_capped_gpus} parallel processes."
            )
            num_gpus = max(1, ram_capped_gpus)

        # Gather all benchmarks first
        all_benchmarks = []
        for d in args.bench_dirs:
            if os.path.exists(d):
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        if f.endswith(".bench"):
                            all_benchmarks.append(os.path.join(d, f))
                elif os.path.isfile(d) and d.endswith(".bench"):
                    all_benchmarks.append(d)

        all_benchmarks.sort()
        if not all_benchmarks:
            print("❌ No benchmark files found in specified directories.")
            return False

        print(
            f"Discovered {len(all_benchmarks)} benchmarks. Distributing across {num_gpus} GPUs..."
        )

        processes = []
        for i in range(num_gpus):
            # Chunking logic
            chunk = all_benchmarks[i::num_gpus]
            if not chunk:
                continue

            cmd = [
                sys.executable,
                "-m",
                "scripts.collect_experience",
                "--max_faults",
                str(args.max_faults // num_gpus),
            ]

            if args.model:
                cmd.extend(["--model", args.model])

            cmd.extend(["--gpu", str(i)])
            cmd.extend(["--exploration", str(args.exploration)])
            cmd.extend(["--bench_dirs"] + chunk)

            print(f"  [GPU {i}] Running: {' '.join(cmd)}")
            p = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
            processes.append(p)

        failed = False
        for i, p in enumerate(processes):
            p.wait()
            if p.returncode != 0:
                print(f"❌ Experience collection on GPU {i} failed with code {p.returncode}")
                failed = True

        if failed:
            return False
    else:
        cmd = [
            "python",
            "-m",
            "scripts.collect_experience",
            "--max_faults",
            str(args.max_faults),
        ]
        if args.model:
            cmd.extend(["--model", args.model])

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
        if result.returncode != 0:
            print(f"❌ Experience collection failed with code {result.returncode}")
            return False

    print("✓ Experience collection completed")
    return True


def run_training(args):
    """Stage 2: Train the RL model"""
    print_section("STAGE 2: Model Training")

    cmd = [
        "python",
        "-m",
        "scripts.train_rl",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--max_paths",
        str(args.max_paths),
        "--amp",
    ]

    if args.model:
        cmd.extend(["--model", args.model])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))

    if result.returncode != 0:
        print(f"❌ Training failed with code {result.returncode}")
        return False

    print("✓ Training completed")
    return True


def run_benchmark(args):
    """Stage 3: Benchmark the trained model"""
    print_section("STAGE 3: Benchmarking")

    model_path = args.output_model or "checkpoints/reconv_rl_model.pt"

    cmd = [
        "python",
        "-m",
        "scripts.benchmark_podem",
        "--model",
        model_path,
        "--gate",
        "259",
        "--sa",
        "1",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))

    if result.returncode != 0:
        print(f"❌ Benchmarking failed with code {result.returncode}")
        return False

    print("✓ Benchmarking completed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Unified RL Training Pipeline for AI-PODEM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/run_rl_pipeline.py --all
  
  # Run only experience collection
  python scripts/run_rl_pipeline.py --collect
  
  # Run training and benchmarking
  python scripts/run_rl_pipeline.py --train --benchmark
  
  # Custom parameters
  python scripts/run_rl_pipeline.py --all --max_faults 200 --epochs 30
        """,
    )

    # Stage selection
    stage_group = parser.add_argument_group("Stage Selection")
    stage_group.add_argument(
        "--all", action="store_true", help="Run all stages (collect, train, benchmark)"
    )
    stage_group.add_argument("--collect", action="store_true", help="Run experience collection")
    stage_group.add_argument("--train", action="store_true", help="Run model training")
    stage_group.add_argument("--benchmark", action="store_true", help="Run benchmarking")

    # Experience collection parameters
    collect_group = parser.add_argument_group("Experience Collection Parameters")
    collect_group.add_argument(
        "--max_faults",
        type=int,
        default=100,
        help="Maximum faults per circuit (default: 100)",
    )
    collect_group.add_argument(
        "--bench_dirs",
        nargs="+",
        default=["data/bench/ISCAS85", "data/bench/iscas89"],
        help="Directories with .bench files (default: ISCAS85 and iscas89)",
    )
    collect_group.add_argument(
        "--exploration",
        type=int,
        default=5,
        help="Random exploration attempts per fault (default: 5)",
    )

    # Training parameters
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs (default: 20)"
    )
    train_group.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size (default: 256)"
    )
    train_group.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    train_group.add_argument(
        "--model",
        type=str,
        default="checkpoints/unlinked_candidate/best_model.pth",
        help="Pretrained model path to start/continue training",
    )
    train_group.add_argument(
        "--max_paths",
        type=int,
        default=250,
        help="Maximum paths per sample during training (default: 250)",
    )
    train_group.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision (default: True)",
    )

    # Output parameters
    output_group = parser.add_argument_group("Output Parameters")
    output_group.add_argument(
        "--output_model",
        type=str,
        help="Output model path (default: checkpoints/reconv_rl_model.pt)",
    )

    args = parser.parse_args()

    # If no stage specified, show help
    if not (args.all or args.collect or args.train or args.benchmark):
        parser.print_help()
        return 1

    # Determine which stages to run
    run_collect = args.all or args.collect
    run_train = args.all or args.train
    run_bench = args.all or args.benchmark

    # Print pipeline configuration
    print_section("RL Training Pipeline Configuration")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nStages to run:")
    print(f"  - Experience Collection: {'✓' if run_collect else '✗'}")
    print(f"  - Model Training:        {'✓' if run_train else '✗'}")
    print(f"  - Benchmarking:          {'✓' if run_bench else '✗'}")
    print("\nParameters:")
    print(f"  - Max faults per circuit: {args.max_faults}")
    print(f"  - Bench dirs:             {', '.join(args.bench_dirs)}")
    print(f"  - Training epochs:        {args.epochs}")
    print(f"  - Batch size:             {args.batch_size}")
    print(f"  - Learning rate:          {args.lr}")

    # Run stages
    success = True

    if run_collect:
        if not run_experience_collection(args):
            success = False
            if args.all:
                print("\n❌ Pipeline aborted due to collection failure")
                return 1

    if run_train and success:
        if not run_training(args):
            success = False
            if args.all:
                print("\n❌ Pipeline aborted due to training failure")
                return 1

    if run_bench and success:
        if not run_benchmark(args):
            success = False

    # Final summary
    print_section("Pipeline Summary")
    if success:
        print("✓ All requested stages completed successfully!")
        return 0
    else:
        print("❌ Some stages failed. Check logs above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
