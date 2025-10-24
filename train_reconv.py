"""Script to build dataset and train the reconvergent path justification model."""

import argparse
import os
import torch

from src.atpg.reconv_podem import build_dataset, save_dataset, load_dataset
from src.ml.reconv_rl_trainer import ReconvRLTrainer


def build_and_save_dataset(bench_dir: str, output_path: str, max_samples: int = 1000):
    """Build dataset from bench files and save."""
    print(f"Building dataset from {bench_dir}...")
    dataset = build_dataset(bench_dir, max_samples_per_file=max_samples)
    
    print("\nDataset statistics:")
    print(f"  Total samples: {len(dataset)}")
    
    # Count samples per file
    file_counts = {}
    for entry in dataset:
        file_name = os.path.basename(entry['file'])
        file_counts[file_name] = file_counts.get(file_name, 0) + 1
    
    print("  Samples per file:")
    for file_name, count in sorted(file_counts.items()):
        print(f"    {file_name}: {count}")
    
    # Save dataset
    save_dataset(dataset, output_path)
    
    return dataset


def find_largest_batch_size(trainer, dataset, start=4, max_batch=8192, step=32, verbose=False):
    """Find the largest batch size that fits in GPU memory."""
    batch_size = start
    last_good = batch_size
    while batch_size <= max_batch:
        try:
            batch = dataset[:batch_size]
            # Try a forward and backward pass
            path_emb, attn_mask, targets = trainer.prepare_batch(batch, target_value=1)
            trainer.model.train()
            out = trainer.model(path_emb, attn_mask)
            loss = out.sum()
            loss.backward()
            if verbose:
                print(f"Batch size {batch_size} OK")
            last_good = batch_size
            batch_size += step
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if verbose:
                    print(f"Batch size {batch_size} OOM")
                torch.cuda.empty_cache()
                break
            else:
                raise
    if last_good < max_batch:
        print(f"[INFO] Max batch size found: {last_good}. If GPU utilization is still low, try increasing model size (embedding_dim, num_encoder_layers, num_interaction_layers).")
    else:
        print(f"[INFO] Batch size {last_good} fits, but GPU may still be underutilized if the model is small.")
    return last_good

def train_model(
    dataset_path: str,
    checkpoint_dir: str,
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    supervised_weight: float = 0.7,
    rl_weight: float = 0.3,
    embedding_dim: int = 128,
    nhead: int = 8,
    num_encoder_layers: int = 4,
    num_interaction_layers: int = 2,
    verbose: bool = False,
    amp: bool = False,
    auto_batch: bool = False
):
    """Train the model."""
    print(f"\nLoading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    
    print("\nInitializing trainer...")
    trainer = ReconvRLTrainer(
        embedding_dim=embedding_dim,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_interaction_layers=num_interaction_layers,
        learning_rate=learning_rate,
        verbose=verbose,
        amp=amp
    )

    print("\nAuto-scaling batch size for maximum GPU usage...")
    batch_size = find_largest_batch_size(trainer, dataset, start=batch_size, verbose=verbose)
    print(f"  Using batch size: {batch_size}")
    
    print("\nTraining configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  AMP (mixed precision): {'ON' if amp else 'OFF'}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Supervised weight: {supervised_weight}")
    print(f"  RL weight: {rl_weight}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    
    # Training loop
    best_reward = float('-inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train epoch
        metrics = trainer.train_epoch(
            dataset,
            batch_size=batch_size,
            supervised_weight=supervised_weight,
            rl_weight=rl_weight,
            amp=amp
        )
        
        # Print metrics
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{epoch + 1}.pth"
            )
            trainer.save_checkpoint(checkpoint_path, epoch + 1, metrics)
        
        # Save best model based on reward
        if 'avg_reward' in metrics and metrics['avg_reward'] > best_reward:
            best_reward = metrics['avg_reward']
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            trainer.save_checkpoint(best_path, epoch + 1, metrics)
            print(f"\n  New best model saved! Reward: {best_reward:.4f}")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best reward: {best_reward:.4f}")
    print(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build dataset and train reconvergent path justification model"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build dataset command
    build_parser = subparsers.add_parser('build', help='Build dataset from bench files')
    build_parser.add_argument(
        '--bench-dir',
        type=str,
        default='data/bench/ISCAS85',
        help='Directory containing .bench files'
    )
    build_parser.add_argument(
        '--output',
        type=str,
        default='data/datasets/reconv_dataset.pkl',
        help='Output path for dataset'
    )
    build_parser.add_argument(
        '--max-samples',
        type=int,
        default=1000,
        help='Maximum samples per file'
    )
    # Exhaustive enumeration is now default behavior.
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=8192,
        help='Batch size (aggressive for GPU)'
    )
    train_parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/reconv_rl',
        help='Directory to save checkpoints'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    # Removed duplicate --batch-size argument to fix argparse conflict
    train_parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    train_parser.add_argument(
        '--supervised-weight',
        type=float,
        default=0.7,
        help='Weight for supervised loss'
    )
    train_parser.add_argument(
        '--rl-weight',
        type=float,
        default=0.3,
        help='Weight for RL loss'
    )
    train_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose training logs'
    )
    train_parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable mixed precision (AMP) training'
    )
    # Removed --auto-batch argument; auto-batch is now always enabled
    
    # Both command
    both_parser = subparsers.add_parser('both', help='Build dataset and train')
    both_parser.add_argument(
        '--bench-dir',
        type=str,
        default='data/bench/ISCAS85',
        help='Directory containing .bench files'
    )
    both_parser.add_argument(
        '--dataset',
        type=str,
        default='data/datasets/reconv_dataset.pkl',
        help='Path to save/load dataset'
    )
    both_parser.add_argument(
        '--batch-size',
        type=int,
        default=8192,
        help='Batch size (aggressive for GPU)'
    )
    both_parser.add_argument(
        '--max-samples',
        type=int,
        default=1000,
        help='Maximum samples per file'
    )
    # Exhaustive enumeration is now default behavior.
    both_parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    both_parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/reconv_rl',
        help='Directory to save checkpoints'
    )
    both_parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    both_parser.add_argument(
        '--supervised-weight',
        type=float,
        default=0.7,
        help='Weight for supervised loss'
    )
    both_parser.add_argument(
        '--rl-weight',
        type=float,
        default=0.3,
        help='Weight for RL loss'
    )
    both_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose training logs'
    )
    both_parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable mixed precision (AMP) training'
    )
    # Removed --auto-batch argument; auto-batch is now always enabled
    
    args = parser.parse_args()
    
    if args.command == 'build':
        build_and_save_dataset(args.bench_dir, args.output, args.max_samples)
    
    elif args.command == 'train':
        train_model(
            dataset_path=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            supervised_weight=args.supervised_weight,
            rl_weight=args.rl_weight,
            verbose=getattr(args, 'verbose', False),
            amp=getattr(args, 'amp', False)
        )
    
    elif args.command == 'both':
        # Build dataset
        build_and_save_dataset(args.bench_dir, args.dataset, args.max_samples)
        
        # Train model
        train_model(
            dataset_path=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            supervised_weight=args.supervised_weight,
            rl_weight=args.rl_weight,
            verbose=getattr(args, 'verbose', False),
            amp=getattr(args, 'amp', False)
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
