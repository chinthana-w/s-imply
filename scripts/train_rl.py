#!/usr/bin/env python
"""
RL Training Script - Policy Gradient for Multi-Path Transformer

This script:
1. Loads experience batches collected by collect_experience.py
2. Reconstructs model inputs from saved snapshots
3. Computes policy gradient loss using reward signals
4. Updates the MultiPathTransformer model
"""

import argparse
import gc
import glob
import os
import pickle
from dataclasses import dataclass

import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Legacy mapping for pickle loads
from src.ml.core.model import MultiPathTransformer


@dataclass
class TrainConfig:
    experience_dir: str = "data/rl_experience"
    model_path: str = "checkpoints/unlinked_candidate/best_model.pth"
    output_path: str = "checkpoints/reconv_rl_model.pt"

    # Training params
    epochs: int = 10
    batch_size: int = 256  # Increased for dual GPUs
    lr: float = 1e-4
    amp: bool = True

    # Model architecture (must match existing model)
    input_dim: int = 132  # 128 struct + 4 logic
    model_dim: int = 512
    nhead: int = 4
    num_encoder_layers: int = 3
    num_interaction_layers: int = 3

    # RL params
    gamma: float = 0.99  # Discount factor
    entropy_beta: float = 0.01  # Entropy regularization
    max_paths: int = 200  # Memory optimization: limit paths per sample


class EmbeddingRegistry:
    """Dynamically loads and caches DeepGate embeddings to avoid memory bloat."""

    def __init__(self):
        self.cache = {}

    def get_embedding(self, bench_file: str) -> torch.Tensor:
        if not bench_file:
            return None

        if bench_file in self.cache:
            return self.cache[bench_file]

        cache_dir = os.path.join(os.path.dirname(bench_file), ".deepgate_cache")
        cache_name = os.path.basename(bench_file) + ".embed.pt"
        cache_path = os.path.join(cache_dir, cache_name)

        if os.path.exists(cache_path):
            try:
                checkpoint = torch.load(cache_path, map_location="cpu")
                if "hs" in checkpoint:
                    self.cache[bench_file] = checkpoint["hs"]
                    return checkpoint["hs"]
            except Exception as e:
                print(f"[Warning] Failed to load embedding from {cache_path}: {e}")
        return None


class ExperienceDataset(Dataset):
    """Dataset that loads experience batches from disk lazily."""

    def __init__(self, experience_dir: str, cache_size: int = 5):
        self.experience_dir = experience_dir
        self.cache_size = cache_size
        self.index = []  # List of (file_path, ep_idx, step_idx)
        self.file_cache = {}  # file_path -> episodes

        # Load all batch files and index them
        batch_files = sorted(glob.glob(os.path.join(experience_dir, "batch_*.pkl")))
        print(f"Indexing {len(batch_files)} experience batch files...")

        for bf in batch_files:
            try:
                # We only need to know how many episodes/steps are in each file
                with open(bf, "rb") as f:
                    episodes = pickle.load(f)
                    for ep_idx, ep in enumerate(episodes):
                        for st_idx in range(len(ep)):
                            self.index.append((bf, ep_idx, st_idx))

                    # Force cleanup after each file
                    del episodes
                    if psutil.virtual_memory().percent > 85:
                        gc.collect()
            except Exception as e:
                print(f"Failed to index {bf}: {e}")

        print(f"Indexed {len(self.index)} experience steps")

    def __len__(self):
        return len(self.index)

    def _get_episodes(self, file_path):
        if file_path not in self.file_cache:
            if len(self.file_cache) >= self.cache_size:
                # Simple cache eviction: clear all (LRU would be better but this is simple)
                self.file_cache.clear()
            with open(file_path, "rb") as f:
                self.file_cache[file_path] = pickle.load(f)
        return self.file_cache[file_path]

    def __getitem__(self, idx):
        file_path, ep_idx, st_idx = self.index[idx]
        episodes = self._get_episodes(file_path)
        step = episodes[ep_idx][st_idx]

        # Extract tensors
        # Note: These were saved as CPU tensors
        node_ids = (
            step.node_ids if step.node_ids is not None else torch.zeros(1, 1, 1, dtype=torch.long)
        )
        mask_valid = (
            step.mask_valid
            if step.mask_valid is not None
            else torch.ones(1, 1, 1, dtype=torch.bool)
        )
        gate_types = (
            step.gate_types
            if step.gate_types is not None
            else torch.zeros(1, 1, 1, dtype=torch.long)
        )

        reward = torch.tensor(step.reward, dtype=torch.float32)

        # Target: the assignment that was selected
        # For now, we just use reward as the signal
        return {
            "node_ids": node_ids.squeeze(0) if node_ids.ndim > 3 else node_ids,
            "mask_valid": mask_valid.squeeze(0) if mask_valid.ndim > 3 else mask_valid,
            "gate_types": gate_types.squeeze(0) if gate_types.ndim > 3 else gate_types,
            "reward": reward,
            "bench_file": getattr(step, "bench_file", ""),
        }


def collate_experience(batch, max_paths_limit=0, embedding_registry=None):
    """Custom collate function for variable-sized experience."""

    # Find max dimensions
    actual_max_paths = max(b["node_ids"].shape[0] for b in batch)
    if max_paths_limit > 0:
        max_paths = min(actual_max_paths, max_paths_limit)
    else:
        max_paths = actual_max_paths

    max_len = max(b["node_ids"].shape[1] if b["node_ids"].ndim >= 2 else 1 for b in batch)

    batch_size = len(batch)

    # Prepare output tensors
    node_ids = torch.zeros(batch_size, max_paths, max_len, dtype=torch.long)
    mask_valid = torch.zeros(batch_size, max_paths, max_len, dtype=torch.bool)
    gate_types = torch.zeros(batch_size, max_paths, max_len, dtype=torch.long)
    rewards = torch.stack([b["reward"] for b in batch])

    # Optional embeddings tensor [B, P, L, 132]
    # We pad to 132 for backwards compatibility with the current model which expects logic constraint dims
    paths_emb = torch.zeros(batch_size, max_paths, max_len, 132, dtype=torch.float32)

    for i, b in enumerate(batch):
        nids = b["node_ids"]
        if nids.ndim == 2:
            p_i, length = nids.shape
            p_curr = min(p_i, max_paths)
            nids_clamped = nids[:p_curr]
            node_ids[i, :p_curr, :length] = nids_clamped
            mask_valid[i, :p_curr, :length] = b["mask_valid"][:p_curr]
            gate_types[i, :p_curr, :length] = b["gate_types"][:p_curr]

            # Dynamically reconstruct DeepGate embeddings if registry exists
            if embedding_registry and b["bench_file"]:
                circuit_emb = embedding_registry.get_embedding(b["bench_file"])
                if circuit_emb is not None:
                    # Map nodes to embeddings. Default to 0 vector if OOB
                    vocab_size = circuit_emb.size(0)
                    for p_idx in range(p_curr):
                        for l_idx in range(length):
                            nid = nids_clamped[p_idx, l_idx].item()
                            # We assume the AI Podem gate mapping roughly tracks with AIG up to a point,
                            # but realistically we just use the node ID directly since embedding extractor mapped it
                            if 0 < nid < vocab_size:
                                paths_emb[i, p_idx, l_idx, :128] = circuit_emb[nid]

        elif nids.ndim == 1:
            length = nids.shape[0]
            node_ids[i, 0, :length] = nids
            mask_valid[i, 0, :length] = b["mask_valid"]
            gate_types[i, 0, :length] = b["gate_types"]

            if embedding_registry and b["bench_file"]:
                circuit_emb = embedding_registry.get_embedding(b["bench_file"])
                if circuit_emb is not None:
                    vocab_size = circuit_emb.size(0)
                    for l_idx in range(length):
                        nid = nids[l_idx].item()
                        if 0 < nid < vocab_size:
                            paths_emb[i, 0, l_idx, :128] = circuit_emb[nid]

    return {
        "node_ids": node_ids,
        "mask_valid": mask_valid,
        "gate_types": gate_types,
        "rewards": rewards,
        "paths_emb": paths_emb,
    }


def train_epoch(model, dataloader, optimizer, config, device, epoch):
    """Train for one epoch using policy gradient."""

    model.train()
    total_loss = 0.0
    total_steps = 0
    total_entropy = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)

    for batch in pbar:
        node_ids = batch["node_ids"].to(device)
        mask_valid = batch["mask_valid"].to(device)
        gate_types = batch["gate_types"].to(device)
        rewards = batch["rewards"].to(device)

        B, P, L = node_ids.shape

        # Use the dynamically reconstructed embeddings from the collate function
        paths_emb = batch["paths_emb"].to(device)

        # We must align input dims if they mismatch, but the model expects input_dim=132 (128 struct + 4 aux).
        # We padded `paths_emb` to 132 in collate, leaving the last 4 dims as zeros which is safe since they represent constraints.

        # Forward pass
        optimizer.zero_grad()

        try:
            with torch.amp.autocast("cuda", enabled=config.amp):
                logits, solv_logits = model(paths_emb, mask_valid, gate_types)
        except Exception as e:
            print(f"Forward error: {e}")
            import traceback

            traceback.print_exc()
            continue

        # logits: [B, P, L, 2] - predictions for 0/1 at each node

        # Policy gradient loss
        # Using REINFORCE: loss = -log_prob(action) * advantage

        # Since we saved the selected_assignment but not explicit action indices,
        # we use softmax to get probabilities and encourage high-prob predictions
        probs = F.softmax(logits, dim=-1)  # [B, P, L, 2]
        log_probs = F.log_softmax(logits, dim=-1)

        # Entropy for exploration
        entropy = -(probs * log_probs).sum(dim=-1)  # [B, P, L]
        masked_entropy = (entropy * mask_valid.float()).sum() / mask_valid.float().sum().clamp(
            min=1
        )

        # Policy loss: Take the max prediction and weight by reward
        # Positive reward -> reinforce current predictions
        # Negative reward -> push away from current predictions

        # Normalize rewards (advantage estimation)
        if rewards.std() > 1e-6:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = rewards - rewards.mean()

        # Get log prob of predicted action (argmax)
        predicted_actions = logits.argmax(dim=-1)  # [B, P, L]

        # Gather log probs of selected actions
        action_log_probs = log_probs.gather(-1, predicted_actions.unsqueeze(-1)).squeeze(
            -1
        )  # [B, P, L]

        # Mask and aggregate
        masked_log_probs = (action_log_probs * mask_valid.float()).sum(dim=[1, 2])  # [B]
        num_valid = mask_valid.float().sum(dim=[1, 2]).clamp(min=1)  # [B]
        per_sample_log_prob = masked_log_probs / num_valid  # [B]

        # Policy gradient loss: -log_prob * advantage
        policy_loss = -(per_sample_log_prob * advantages).mean()

        # Total loss with entropy bonus
        loss = policy_loss - config.entropy_beta * masked_entropy

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_entropy += masked_entropy.item()
        total_steps += 1

        # RAM awareness in pbar
        ram_percent = psutil.virtual_memory().percent
        pbar.set_postfix(
            {
                "loss": f"{total_loss / total_steps:.4f}",
                "entropy": f"{total_entropy / total_steps:.4f}",
                "reward": f"{rewards.mean().item():.2f}",
                "ram": f"{ram_percent}%",
            }
        )

        # Emergency GC if RAM is too high
        if ram_percent > 90:
            gc.collect()

    return total_loss / max(total_steps, 1)


def main():
    parser = argparse.ArgumentParser(description="RL Training for Multi-Path Transformer")
    parser.add_argument("--experience_dir", default="data/rl_experience")
    parser.add_argument("--model", default="checkpoints/unlinked_candidate/best_model.pth")
    parser.add_argument("--output", default="checkpoints/reconv_rl_model.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--max_paths", type=int, default=200, help="Limit paths per sample for memory"
    )
    args = parser.parse_args()

    config = TrainConfig(
        experience_dir=args.experience_dir,
        model_path=args.model,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        amp=args.amp,
        max_paths=args.max_paths,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            # Test if CUDA actually works (e.g. check for sm_120 compatibility)
            # A simple tensor creation might succeed, but an operation might fail.
            a = torch.randn(2, 2).to(device)
            b = torch.randn(2, 2).to(device)
            _ = torch.matmul(a, b)
            print(f"Using device: {device}")
        except Exception as e:
            print(f"CUDA detected but incompatible: {e}. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        print(f"Using device: {device}")

    # Load dataset
    dataset = ExperienceDataset(config.experience_dir)

    if len(dataset) == 0:
        print("No experience data found. Run collect_experience.py first.")
        return

    # RAM Aware worker adjustment
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    if available_ram_gb < 4:
        print(f"Low RAM detected ({available_ram_gb:.1f} GB). Reducing workers and batch size.")
        args.num_workers = min(args.num_workers, 2)
        config.batch_size = min(config.batch_size, 128)
    elif available_ram_gb < 8:
        args.num_workers = min(args.num_workers, 4)

    import functools

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=functools.partial(collate_experience, max_paths_limit=config.max_paths),
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Create model
    model = MultiPathTransformer(
        input_dim=config.input_dim,
        model_dim=config.model_dim,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_interaction_layers=config.num_interaction_layers,
    ).to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # Load pretrained weights if available
    if os.path.exists(config.model_path):
        try:
            ckpt = torch.load(config.model_path, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            elif "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
            print(f"Loaded pretrained model from {config.model_path}")
        except Exception as e:
            print(f"Could not load pretrained model: {e}")
    else:
        print("No pretrained model found, training from scratch")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Training loop
    best_loss = float("inf")

    for epoch in range(config.epochs):
        loss = train_epoch(model, dataloader, optimizer, config, device, epoch)
        print(f"Epoch {epoch+1}/{config.epochs} - Loss: {loss:.4f}")

        # Save checkpoint
        if loss < best_loss:
            best_loss = loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                config.output_path,
            )
            print(f"Saved best model to {config.output_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()
