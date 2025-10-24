"""Utility to analyze and visualize training results."""

import os
import pickle
import torch
import argparse
from collections import defaultdict


def analyze_dataset(dataset_path: str):
    """Analyze dataset statistics."""
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"\nDataset Statistics:")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")
    
    # File statistics
    file_counts = defaultdict(int)
    path_lengths = []
    num_paths_list = []
    
    for entry in dataset:
        file_name = os.path.basename(entry['file'])
        file_counts[file_name] += 1
        
        info = entry['info']
        paths = info['paths']
        num_paths_list.append(len(paths))
        
        for path in paths:
            path_lengths.append(len(path))
    
    print(f"\nSamples per file:")
    for file_name in sorted(file_counts.keys()):
        print(f"  {file_name}: {file_counts[file_name]}")
    
    print(f"\nPath statistics:")
    print(f"  Total paths: {len(path_lengths)}")
    print(f"  Average path length: {sum(path_lengths)/len(path_lengths):.2f}")
    print(f"  Min path length: {min(path_lengths)}")
    print(f"  Max path length: {max(path_lengths)}")
    print(f"  Average paths per sample: {sum(num_paths_list)/len(num_paths_list):.2f}")
    
    # Justification statistics
    just1_sizes = []
    just0_sizes = []
    xd_count_1 = 0
    xd_count_0 = 0
    
    for entry in dataset:
        j1 = entry['justification_1']
        j0 = entry['justification_0']
        
        just1_sizes.append(len(j1))
        just0_sizes.append(len(j0))
        
        # Count XD (unknown) values
        xd_count_1 += sum(1 for v in j1.values() if v == 2)
        xd_count_0 += sum(1 for v in j0.values() if v == 2)
    
    print(f"\nJustification statistics:")
    print(f"  Average justification_1 size: {sum(just1_sizes)/len(just1_sizes):.2f}")
    print(f"  Average justification_0 size: {sum(just0_sizes)/len(just0_sizes):.2f}")
    print(f"  Total XD values in justification_1: {xd_count_1}")
    print(f"  Total XD values in justification_0: {xd_count_0}")
    
    if xd_count_1 > 0 or xd_count_0 > 0:
        print(f"\n  ⚠ Warning: Dataset contains {xd_count_1 + xd_count_0} unassigned (XD) values")
        print(f"    This may indicate issues with the justification algorithm.")


def analyze_checkpoint(checkpoint_path: str):
    """Analyze training checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nCheckpoint Information:")
    print(f"{'='*60}")
    print(f"Epoch: {checkpoint['epoch']}")
    
    if 'metrics' in checkpoint:
        print(f"\nMetrics:")
        for key, value in checkpoint['metrics'].items():
            print(f"  {key}: {value:.4f}")
    
    if 'reward_weights' in checkpoint:
        print(f"\nReward weights:")
        for key, value in checkpoint['reward_weights'].items():
            print(f"  {key}: {value}")
    
    # Model statistics
    model_state = checkpoint['model_state_dict']
    total_params = sum(p.numel() for p in model_state.values())
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Number of layers: {len(model_state)}")
    
    # Layer sizes
    print(f"\nLayer shapes:")
    for name, param in list(model_state.items())[:10]:  # Show first 10 layers
        print(f"  {name}: {tuple(param.shape)}")
    if len(model_state) > 10:
        print(f"  ... and {len(model_state) - 10} more layers")


def compare_checkpoints(checkpoint_dir: str):
    """Compare all checkpoints in a directory."""
    print(f"Analyzing checkpoints in {checkpoint_dir}...")
    
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pth') and f.startswith('checkpoint_epoch_')
    ]
    
    if not checkpoint_files:
        print("No checkpoint files found.")
        return
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"\nFound {len(checkpoint_files)} checkpoints")
    print(f"\nTraining Progress:")
    print(f"{'='*60}")
    print(f"{'Epoch':<10} {'Sup Loss':<15} {'Policy Loss':<15} {'Avg Reward':<15}")
    print(f"{'-'*60}")
    
    for ckpt_file in checkpoint_files:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        epoch = checkpoint['epoch']
        metrics = checkpoint.get('metrics', {})
        
        sup_loss = metrics.get('supervised_loss', 0.0)
        policy_loss = metrics.get('policy_loss', 0.0)
        avg_reward = metrics.get('avg_reward', 0.0)
        
        print(f"{epoch:<10} {sup_loss:<15.4f} {policy_loss:<15.4f} {avg_reward:<15.4f}")
    
    # Check for best model
    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location='cpu')
        print(f"\n{'='*60}")
        print(f"Best Model (Epoch {checkpoint['epoch']}):")
        metrics = checkpoint.get('metrics', {})
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze training data and checkpoints")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Dataset analysis
    dataset_parser = subparsers.add_parser('dataset', help='Analyze dataset')
    dataset_parser.add_argument(
        '--path',
        type=str,
        default='data/datasets/reconv_dataset.pkl',
        help='Path to dataset file'
    )
    
    # Checkpoint analysis
    checkpoint_parser = subparsers.add_parser('checkpoint', help='Analyze single checkpoint')
    checkpoint_parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )
    
    # Compare checkpoints
    compare_parser = subparsers.add_parser('compare', help='Compare all checkpoints')
    compare_parser.add_argument(
        '--dir',
        type=str,
        default='checkpoints/reconv_rl',
        help='Directory containing checkpoints'
    )
    
    args = parser.parse_args()
    
    if args.command == 'dataset':
        analyze_dataset(args.path)
    elif args.command == 'checkpoint':
        analyze_checkpoint(args.path)
    elif args.command == 'compare':
        compare_checkpoints(args.dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
