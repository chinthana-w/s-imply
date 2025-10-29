"""
Minimal trainer for the Multi-Path reconvergent transformer.

This script focuses on a supervised-only training loop with cross-entropy loss
over per-node labels derived from available justifications in the dataset.

Usage (example):
    conda activate torch
    python -m src.ml.train_reconv train \
            --dataset data/datasets/reconv_dataset.pkl \
            --output checkpoints/reconv_minimal \
            --epochs 5

Notes:
- Embedding dimension defaults to 128 to match the dummy embeddings path.
- Mixed precision, RL, auto batch scaling, etc., are out-of-scope for this
  minimal baseline.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.ml.reconv_lib import MultiPathTransformer
from src.ml.reconv_ds import ReconvergentPathsDataset, reconv_collate
from src.util.io import parse_bench_file
from src.util.struct import GateType


@dataclass
class TrainConfig:
    dataset: str
    output: str
    epochs: int = 10
    # Internal defaults; not exposed via CLI for simplicity
    batch_size: int = 8
    lr: float = 1e-4
    embedding_dim: int = 128
    nhead: int = 4
    num_encoder_layers: int = 1
    num_interaction_layers: int = 1
    prefer_value: int = 1
    verbose: bool = False


def make_dataloaders(cfg: TrainConfig, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    # Auto-detect processed shards: look for processed/ subdirectory next to dataset
    dataset_dir = os.path.dirname(cfg.dataset)
    processed_dir = os.path.join(dataset_dir, 'reconv_processed')
    load_processed = os.path.isdir(processed_dir)
    
    dataset = ReconvergentPathsDataset(
        cfg.dataset,
        device=device,
        prefer_value=cfg.prefer_value,
        processed_dir=processed_dir if load_processed else None,
        load_processed=load_processed,
    )
    # Minimal split: 90/10 train/val
    n = len(dataset)
    n_train = max(1, int(0.9 * n))
    n_val = max(1, n - n_train)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=reconv_collate)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=reconv_collate)
    return train_loader, val_loader


@lru_cache(maxsize=64)
def _load_circuit(bench_file: str):
    circuit, _ = parse_bench_file(bench_file)
    return circuit


def _pair_constraint_ok(gate_type: int, prev_val: int, cur_val: int) -> bool:
    """Local constraint along a path edge assuming side-input freedom.

    - NOT: output is inversion of previous
    - BUFF: output equals previous
    - AND: output 1 requires prev 1; output 0 has no constraint
    - NAND: output 0 requires prev 1; output 1 no constraint
    - OR: output 0 requires prev 0; output 1 no constraint
    - NOR: output 1 requires prev 0; output 0 no constraint
    - XOR/XNOR/INPT/FROM: considered satisfiable with some side-input
    """
    if gate_type == GateType.NOT:
        return (1 - prev_val) == cur_val
    if gate_type == GateType.BUFF:
        return prev_val == cur_val
    if gate_type == GateType.AND:
        return True if cur_val == 0 else (prev_val == 1)
    if gate_type == GateType.NAND:
        return True if cur_val == 1 else (prev_val == 1)
    if gate_type == GateType.OR:
        return True if cur_val == 1 else (prev_val == 0)
    if gate_type == GateType.NOR:
        return True if cur_val == 0 else (prev_val == 0)
    return True


def policy_loss_and_metrics(
    logits: torch.Tensor,
    node_ids: torch.Tensor,
    mask_valid: torch.Tensor,
    files: list[str],
) -> tuple[torch.Tensor, float, float]:
    """Compute REINFORCE loss with LUT-inspired constraints and reconv consistency.

    Returns: (loss, avg_reward, valid_rate)
    """
    B, P, L, C = logits.shape
    # Sample actions; detach to avoid retaining graph history through the sample op.
    actions = torch.distributions.Categorical(logits=logits).sample().detach()  # [B, P, L]
    # Compute log-probabilities directly from logits to avoid distribution caching quirks.
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, P, L, C]
    logp = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [B, P, L]

    wrong_count_list: list[float] = []
    checked_list: list[float] = []
    for b in range(B):
        circuit = _load_circuit(files[b])
        wrong = 0
        checked = 0
        # Local edge constraints along each path
        for p in range(P):
            valid_positions = mask_valid[b, p]  # [L]
            for li in range(1, L):
                if not (bool(valid_positions[li]) and bool(valid_positions[li-1])):
                    continue
                prev_id = int(node_ids[b, p, li-1].item())
                cur_id = int(node_ids[b, p, li].item())
                if prev_id <= 0 or cur_id <= 0:
                    continue
                gate_type = int(circuit[cur_id].type)
                prev_val = int(actions[b, p, li-1].item())
                cur_val = int(actions[b, p, li].item())
                ok = _pair_constraint_ok(gate_type, prev_val, cur_val)
                checked += 1
                if not ok:
                    wrong += 1
        # Reconvergence consistency: last values of all paths should match if present
        last_vals: list[int] = []
        for p in range(P):
            valid_positions = mask_valid[b, p]
            if bool(valid_positions.any()):
                last_idx = int(valid_positions.sum().item()) - 1
                last_vals.append(int(actions[b, p, last_idx].item()))
        if len(last_vals) >= 2:
            ref = last_vals[0]
            for v in last_vals[1:]:
                checked += 1
                if v != ref:
                    wrong += 1

        wrong_count_list.append(float(wrong))
        checked_list.append(float(checked))

    checked_t = torch.tensor(checked_list, dtype=torch.float32, device=logits.device)
    wrong_t = torch.tensor(wrong_count_list, dtype=torch.float32, device=logits.device)
    valid = (wrong_t == 0) & (checked_t > 0)
    denom = torch.clamp(checked_t, min=1.0)
    reward = torch.where(checked_t > 0, torch.where(valid, torch.ones_like(denom), -wrong_t / denom), torch.zeros_like(denom))
    reward = reward.detach()

    # Policy gradient loss: -mean(reward * mean logp over valid positions)
    logp_sum = (logp * mask_valid).sum(dim=(1, 2))  # [B]
    count = torch.clamp(mask_valid.sum(dim=(1, 2)).float(), min=1.0)
    per_sample_loss = -(reward * (logp_sum / count))  # [B]
    loss = per_sample_loss.mean()

    avg_reward = float(reward.mean().item())
    valid_rate = float(valid.float().mean().item())
    return loss, avg_reward, valid_rate


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    total_reward = 0.0
    total_valid = 0.0

    for batch in loader:
        paths = batch['paths_emb']  # [B, P, L, D]
        masks = batch['attn_mask']  # [B, P, L]
        node_ids = batch['node_ids']  # [B, P, L]
        files = batch['files']        # list[str]

        optim.zero_grad(set_to_none=True)
        logits = model(paths, masks)  # [B, P, L, 2]
        loss, avg_reward, valid_rate = policy_loss_and_metrics(logits, node_ids, masks, files)
        # Backprop. Some GPU/Transformer combos can spuriously detect graph reuse when
        # sampling-based objectives are mixed with encoder reuse; retain_graph mitigates it.
        loss.backward(retain_graph=True)
        optim.step()

        total_loss += float(loss.item())
        total_reward += float(avg_reward)
        total_valid += float(valid_rate)
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    avg_reward = total_reward / max(1, total_batches)
    valid_rate = total_valid / max(1, total_batches)
    # Return avg loss and avg reward for logging
    return avg_loss, avg_reward


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_reward = 0.0
    total_valid = 0.0
    total_batches = 0

    for batch in loader:
        paths = batch['paths_emb']
        masks = batch['attn_mask']
        node_ids = batch['node_ids']
        files = batch['files']

        logits = model(paths, masks)
        loss, avg_reward, valid_rate = policy_loss_and_metrics(logits, node_ids, masks, files)
        total_loss += float(loss.item())
        total_reward += float(avg_reward)
        total_valid += float(valid_rate)
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    avg_reward = total_reward / max(1, total_batches)
    valid_rate = total_valid / max(1, total_batches)
    return avg_loss, avg_reward


def save_checkpoint(path: str, model: nn.Module, cfg: TrainConfig, best: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'config': asdict(cfg),
        'best': best,
    }, path)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        dataset=args.dataset,
        output=args.output,
        epochs=args.epochs,
        batch_size=getattr(args, 'batch_size', 8),
        verbose=getattr(args, 'verbose', False),
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_loader, val_loader = make_dataloaders(cfg, device)

    model = MultiPathTransformer(
        embedding_dim=cfg.embedding_dim,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_interaction_layers=cfg.num_interaction_layers,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val = float('inf')
    if cfg.verbose:
        nb_train = len(train_loader) if hasattr(train_loader, '__len__') else 0
        nb_val = len(val_loader) if hasattr(val_loader, '__len__') else 0
        print(f"Starting training: train_batches={nb_train}, val_batches={nb_val}, batch_size={cfg.batch_size}")
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_reward = train_one_epoch(model, train_loader, optim, device)
        va_loss, va_reward = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} avg_reward={tr_reward:.4f} | val_loss={va_loss:.4f} avg_reward={va_reward:.4f}")

        # Save periodic checkpoint
        if epoch % 10 == 0 or epoch == cfg.epochs:
            save_checkpoint(os.path.join(cfg.output, f"checkpoint_epoch_{epoch}.pth"), model, cfg, best=False)

        # Save best by validation reward (maximize)
        if (-va_reward) < best_val:
            best_val = -va_reward
            save_checkpoint(os.path.join(cfg.output, "best_model.pth"), model, cfg, best=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal reconv transformer trainer")
    sub = p.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train', help='Run supervised training')
    t.add_argument('--dataset', type=str, default='data/datasets/reconv_dataset.pkl', help='Path to dataset .pkl')
    t.add_argument('--output', type=str, default='checkpoints/reconv_minimal', help='Output checkpoint directory')
    t.add_argument('--epochs', type=int, default=10)
    t.add_argument('--batch-size', type=int, default=8)
    t.add_argument('--verbose', action='store_true')

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.cmd == 'train':
        cmd_train(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == '__main__':
    main()