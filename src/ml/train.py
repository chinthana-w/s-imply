"""
Minimal trainer for the Multi-Path reconvergent transformer.

This script focuses on a supervised-only training loop with cross-entropy loss
over per-node labels derived from available justifications in the dataset.

Usage (example):
    conda activate torch
    python -m src.ml.train train \
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
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ml.core.dataset import (
    ReconvergentPathsDataset,
    _generate_anchor,
    _inject_anchor_into_embeddings,
    generate_constraints,
    resolve_gate_types,
)
from src.ml.core.loss import (
    _debug_metrics_from_logits,  # Added
    reinforce_loss,
)
from src.ml.core.model import MultiPathTransformer

# Suppress annoying prototype warnings from PyTorch transformer
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")


@dataclass
class TrainConfig:
    dataset: str
    output: str
    epochs: int = 10
    # Internal defaults; not exposed via CLI for simplicity
    batch_size: int = 128
    lr: float = 1e-4
    embedding_dim: int = 128  # Base structural embedding dimension
    nhead: int = 4
    num_encoder_layers: int = 1
    num_interaction_layers: int = 1
    dim_feedforward: int = 512
    model_dim: int = 512
    prefer_value: int = 1
    verbose: bool = False
    add_logic_value: bool = True  # Whether to add logic value feature (+3 dims)
    # Anchor hint controls (training-time only; not from dataset)
    anchor_hint: bool = True
    anchor_reward_alpha: float = 0.1
    # Runtime controls
    max_train_batches: int = 0  # 0 = no limit
    max_val_batches: int = 0  # 0 = no limit
    log_interval: int = 500  # batches between progress prints when verbose
    # Dataset-level anchor integration (generate anchor in dataset loader)
    dataset_anchor_hint: bool = True
    # DataLoader performance
    num_workers: int = 8
    pin_memory: bool = True
    # RL stabilization
    normalize_reward: bool = True
    entropy_beta: float = 0.01

    # New arguments
    bench_dir: str = ""
    soft_edge_lambda: float = 1.0
    amp: bool = False
    include_hard_negatives: bool = False
    max_len: int = 0
    use_gate_type_embedding: bool = True

    # Phase 6: Constrained Training
    constrained_curriculum: bool = False
    inject_constraints: bool = False
    max_constraint_prob: float = 0.5
    enforce_constraints: bool = True
    processed_dir: Optional[str] = None

    # Phase 7: Logic Consistency
    lambda_logic: float = 0.0  # Weight for reconvergence logic consistency loss
    lambda_full_logic: float = 0.0  # Weight for full-path gate consistency loss

    # Gumbel Softmax
    gumbel_temp: float = 1.0
    gumbel_anneal_rate: float = 0.99

    # Memory optimization
    grad_accum: int = 1
    checkpointing: bool = False
    shard_cache_size: int = 1
    max_paths: int = 200
    pretrained: Optional[str] = None


def make_dataloaders(cfg: TrainConfig, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    # Use config processed_dir if provided, else auto-detect
    if cfg.processed_dir:
        processed_dir = cfg.processed_dir
        load_processed = os.path.isdir(processed_dir)
        if not load_processed:
            print(f"[WARNING] Processed dir {processed_dir} not found. Falling back to raw pickle.")
    else:
        # Auto-detect processed shards: look for processed/ subdirectory next to dataset
        dataset_dir = os.path.dirname(cfg.dataset)
        processed_dir = os.path.join(dataset_dir, "reconv_processed")
        load_processed = os.path.isdir(processed_dir)

    # For best throughput, keep dataset tensors on CPU and move whole batches to GPU
    dataset_device = torch.device("cpu") if device.type == "cuda" else device

    # Import collate function and partial it if needed
    import functools

    from src.ml.core.dataset import reconv_collate

    final_collate_fn = reconv_collate
    if cfg.max_paths > 0:
        final_collate_fn = functools.partial(reconv_collate, max_paths=cfg.max_paths)

    print(f"Creating dataset (device={dataset_device})...", flush=True)
    dataset = ReconvergentPathsDataset(
        cfg.dataset,
        device=dataset_device,
        prefer_value=cfg.prefer_value,
        processed_dir=processed_dir if load_processed else None,
        load_processed=load_processed,
        add_logic_value=cfg.add_logic_value,
        anchor_in_dataset=cfg.dataset_anchor_hint,
        max_len_filter=cfg.max_len,
        cache_size=cfg.shard_cache_size,
        inject_constraints=cfg.inject_constraints,
        constraint_prob=cfg.max_constraint_prob if cfg.inject_constraints else 0.0,
    )
    print(f"Dataset ready. Splitting {len(dataset)} samples...", flush=True)
    # Minimal split: 90/10 train/val
    n = len(dataset)
    n_train = max(1, int(0.9 * n))
    n_val = max(1, n - n_train)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    print(f"Split done: {n_train} train, {n_val} val.", flush=True)

    # TUNING: For large batch sizes, we MUST reduce prefetch_factor to save RAM
    # With batch=1024 and factor=2, 8 workers buffer 16k samples (~20GB RAM!)
    prefetch = 2
    if cfg.batch_size >= 512:
        prefetch = 1
        print(
            f"[RECONV-MEM] Large batch size ({cfg.batch_size}). "
            f"Reducing prefetch_factor to {prefetch} to save RAM."
        )

    # CAPPING: Shard cache size per worker
    if cfg.num_workers >= 4 and cfg.shard_cache_size > 2:
        print(
            f"[RECONV-MEM] High worker count ({cfg.num_workers}). "
            "Capping shard_cache_size to 2 per worker."
        )
        cfg.shard_cache_size = 2

    # Use workers and pinned memory for faster host->device transfer when on CUDA
    if device.type == "cuda":
        print(
            f"Initializing DataLoaders (workers={cfg.num_workers}, batch_size={cfg.batch_size})...",
            flush=True,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=final_collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=prefetch if cfg.num_workers > 0 else None,
            persistent_workers=cfg.num_workers > 0,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=final_collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=prefetch if cfg.num_workers > 0 else None,
            persistent_workers=cfg.num_workers > 0,
        )
        print("DataLoaders initialized.", flush=True)
    else:
        # For CPU
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=final_collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.num_workers > 0,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=final_collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.num_workers > 0,
        )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int = 1,
) -> Tuple[float, float, float, float, float]:
    model.train()
    total_loss, total_reward, total_valid, total_batches, total_edge_acc = (
        0.0,
        0.0,
        0.0,
        0,
        0.0,
    )
    total_trivial = 0.0  # Stores constraint violation rate

    time.time()
    bdone = 0

    # Target batch count for ETA
    try:
        loader_len = len(loader)
    except Exception:
        loader_len = -1
    target_batches = loader_len if loader_len > 0 else None
    if cfg.max_train_batches > 0:
        target_batches = (
            cfg.max_train_batches
            if target_batches is None
            else min(cfg.max_train_batches, target_batches)
        )

    # Curriculum: Phase 1 (0-25%) = Pure Learning (No Constraints)
    #             Phase 2 (25%-100%) = Linear Ramp (0 -> Max Constraints)
    constraint_prob = 0.0
    if cfg.constrained_curriculum and cfg.epochs > 0:
        free_epochs = cfg.epochs // 4
        if epoch <= free_epochs:
            constraint_prob = 0.0
        else:
            # Progress within the ramp-up phase (0.0 to 1.0)
            ramp_epochs = cfg.epochs - free_epochs
            epoch_in_ramp = epoch - free_epochs
            ramp_progress = epoch_in_ramp / max(1, ramp_epochs)
            constraint_prob = cfg.max_constraint_prob * ramp_progress

    if cfg.verbose:
        print(
            f"[curriculum] epoch={epoch} constraint_prob={constraint_prob:.3f} "
            f"(Phase {'1' if epoch <= cfg.epochs // 2 else '2'})"
        )

    if cfg.verbose:
        print(f"Starting epoch {epoch} loop (waiting for DataLoader)...", flush=True)
    pbar = tqdm(loader, desc=f"Epoch {epoch}", total=target_batches, unit="batch")

    for batch_idx, batch in enumerate(pbar):
        pbar.set_description(f"Epoch {epoch}")
        paths = batch["paths_emb"].to(device)
        masks = batch["attn_mask"].to(device)
        node_ids = batch["node_ids"].to(device)
        files = batch["files"]
        c_mask = batch["constraint_mask"].to(device) if "constraint_mask" in batch else None
        c_vals = batch["constraint_vals"].to(device) if "constraint_vals" in batch else None

        # 1. Initialize Logic Value Vector to "Unknown" [0, 0, 1] ONLY IF dataset didn't do it
        # (inject_constraints=True in dataset already handles this)
        if cfg.add_logic_value and not cfg.inject_constraints:
            B_cur, P_cur, L_cur, D_cur = paths.shape
            if D_cur >= 3:
                # We initialize to [0, 0, 1] (Unknown)
                paths[..., D_cur - 3] = 0.0
                paths[..., D_cur - 2] = 0.0
                paths[..., D_cur - 1] = 1.0

        # 2. Support manual constraint generation if dataset didn't provide them
        if c_mask is None and not cfg.inject_constraints:
            c_prob = cfg.max_constraint_prob if cfg.constrained_curriculum else 0.0
            if c_prob > 0:
                c_mask, c_vals = generate_constraints(node_ids, files, c_prob)
                c_mask = c_mask.to(device)
                c_vals = c_vals.to(device)

        # 3. Inject constraints (either from batch or manually generated) into embeddings
        if cfg.add_logic_value and c_mask is not None:
            if c_mask.any():
                D = paths.shape[-1]
                if D >= 3:
                    targets = c_vals[c_mask]
                    one_hot = torch.zeros((targets.shape[0], 3), device=device, dtype=paths.dtype)
                    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
                    paths_flat = paths.view(-1, D)
                    mask_flat = c_mask.view(-1)
                    paths_flat[mask_flat, D - 3 : D] = one_hot

        # Anchor hint generation if needed
        if cfg.anchor_hint and "anchor_p" not in batch:
            ap_cpu, al_cpu, av_cpu, s_cpu = _generate_anchor(
                node_ids.detach().cpu(), masks.detach().cpu(), files, cfg.prefer_value
            )
            anchor_p = ap_cpu.to(device)
            anchor_l = al_cpu.to(device)
            anchor_v = av_cpu.to(device)
            solv_labels = s_cpu.to(device)
            paths = _inject_anchor_into_embeddings(
                paths, anchor_p, anchor_l, anchor_v, enable=cfg.add_logic_value
            )
        else:
            anchor_p = batch.get("anchor_p", None)
            if anchor_p is not None:
                anchor_p = anchor_p.to(device)
                anchor_l = batch["anchor_l"].to(device)
                anchor_v = batch["anchor_v"].to(device)
                solv_labels = batch["solvability"].to(device)
            else:
                anchor_p = anchor_l = anchor_v = solv_labels = None  # type: ignore

        # Resolve gate types
        if "gate_types" in batch:
            gtypes = batch["gate_types"].to(device)
        else:
            gtypes = resolve_gate_types(node_ids, files, device)

        # Gumbel Annealing
        gumbel_t = max(0.1, cfg.gumbel_temp * (cfg.gumbel_anneal_rate ** (epoch - 1)))

        with torch.amp.autocast("cuda", enabled=cfg.amp):
            logits, solv_logits = model(
                paths,
                masks,
                gate_types=gtypes if cfg.use_gate_type_embedding else None,
                checkpointing=cfg.checkpointing,
            )

            loss, avg_reward, valid_rate, batch_edge_acc, batch_c_viol = reinforce_loss(
                logits=logits,
                gate_types=gtypes,
                mask_valid=masks,
                solvability_logits=solv_logits,
                solvability_labels=solv_labels,
                anchor_p=anchor_p,
                anchor_l=anchor_l,
                anchor_v=anchor_v,
                entropy_beta=cfg.entropy_beta,
                constraint_mask=c_mask,
                constraint_vals=c_vals,
                node_ids=node_ids,
                lambda_logic=cfg.lambda_logic,
                lambda_full_logic=cfg.lambda_full_logic,
                soft_edge_lambda=cfg.soft_edge_lambda,
                normalize_reward=cfg.normalize_reward,
                anchor_alpha=cfg.anchor_reward_alpha,
                gumbel_temp=gumbel_t,
            )

        # NaN/Inf guard: skip corrupt batches
        loss_val = loss.mean()
        if not torch.isfinite(loss_val):
            print(f"[WARNING] Non-finite loss at batch {batch_idx}, skipping.")
            optim.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss_val).backward()
        # Gradient clipping to prevent AMP-induced explosions
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

        total_loss += float(loss_val.item())
        total_reward += float(avg_reward.mean().item())
        total_valid += float(valid_rate.mean().item())
        total_edge_acc += float(batch_edge_acc.mean().item())
        total_trivial += float(batch_c_viol.mean().item())
        total_batches += 1
        bdone += paths.size(0)

        # 4. RAM AWARENESS in main loop
        if (batch_idx + 1) % 50 == 0:
            import gc

            import psutil

            mem = psutil.virtual_memory()
            if mem.percent > 90.0:
                print(f"[RECONV-MEM] Main loop detected high RAM ({mem.percent}%). Cleaning up.")
                gc.collect()
                torch.cuda.empty_cache()

        if cfg.verbose and cfg.log_interval > 0 and (batch_idx + 1) % cfg.log_interval == 0:
            dbg = _debug_metrics_from_logits(
                logits,
                node_ids,
                masks,
                files,
                anchor_p=anchor_p,
                anchor_l=anchor_l,
                anchor_v=anchor_v,
                solvability_logits=solv_logits,
                solvability_labels=solv_labels,
            )
            pbar.set_postfix(
                {
                    "loss": f"{total_loss / max(1, total_batches):.4f}",
                    "path_acc": f"{total_valid / max(1, total_batches):.4f}",
                    "solv": f"{dbg['solvability_acc']:.3f}",
                    "edge": f"{dbg['edge_acc']:.3f}",
                    "reconv": f"{dbg['reconv_match_rate']:.3f}",
                }
            )

        if cfg.max_train_batches > 0 and (batch_idx + 1) >= cfg.max_train_batches:
            break

        if cfg.max_train_batches > 0 and (batch_idx + 1) >= cfg.max_train_batches:
            break

    return (
        total_loss / max(1, total_batches),
        total_reward / max(1, total_batches),
        total_valid / max(1, total_batches),
        total_edge_acc / max(1, total_batches),
        total_trivial / max(1, total_batches),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, cfg: TrainConfig
) -> Tuple[float, float, float, float, float]:
    model.eval()
    (
        total_loss,
        total_reward,
        total_valid,
        total_edge_acc,
        total_trivial,
        total_batches,
    ) = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    start_time = time.time()

    try:
        loader_len = len(loader)
    except Exception:
        loader_len = -1
    target_batches = loader_len if loader_len > 0 else None
    if cfg.max_val_batches > 0:
        target_batches = (
            cfg.max_val_batches
            if target_batches is None
            else min(cfg.max_val_batches, target_batches)
        )

    pbar = tqdm(loader, desc="Eval", total=target_batches, unit="batch")
    for batch_idx, batch in enumerate(pbar):
        paths = batch["paths_emb"]
        masks = batch["attn_mask"]
        node_ids = batch["node_ids"]
        files = batch["files"]

        if device.type == "cuda":
            paths = paths.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            node_ids = node_ids.to(device, non_blocking=True)

        c_mask = batch["constraint_mask"].to(device) if "constraint_mask" in batch else None
        c_vals = batch["constraint_vals"].to(device) if "constraint_vals" in batch else None

        # 1. Initialize Logic Value Vector to "Unknown" [0, 0, 1] ONLY IF dataset didn't do it
        if cfg.add_logic_value and not cfg.inject_constraints:
            D_cur = paths.shape[-1]
            if D_cur >= 3:
                paths[..., D_cur - 3] = 0.0
                paths[..., D_cur - 2] = 0.0
                paths[..., D_cur - 1] = 1.0

        # 2. Support manual constraint generation if dataset didn't provide them
        if c_mask is None and not cfg.inject_constraints:
            c_prob = cfg.max_constraint_prob if cfg.constrained_curriculum else 0.0
            if c_prob > 0:
                c_mask, c_vals = generate_constraints(node_ids, files, c_prob)
                c_mask = c_mask.to(device)
                c_vals = c_vals.to(device)

        # 3. Inject constraints into embeddings
        if cfg.add_logic_value and c_mask is not None:
            if c_mask.any():
                D = paths.shape[-1]
                if D >= 3:
                    targets = c_vals[c_mask]
                    one_hot = torch.zeros((targets.shape[0], 3), device=device, dtype=paths.dtype)
                    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
                    paths_flat = paths.view(-1, D)
                    mask_flat = c_mask.view(-1)
                    paths_flat[mask_flat, D - 3 : D] = one_hot

        # Use pre-computed anchors from shards (avoid on-the-fly solver)
        if "anchor_p" in batch:
            anchor_p = batch["anchor_p"].to(device)
            anchor_l = batch["anchor_l"].to(device)
            anchor_v = batch["anchor_v"].to(device)
            solv_labels = batch["solvability"].to(device)
        else:
            anchor_p = anchor_l = anchor_v = solv_labels = None  # type: ignore

        if "gate_types" in batch:
            gtypes = batch["gate_types"].to(device)
        else:
            gtypes = resolve_gate_types(node_ids, files, device)

        with torch.amp.autocast("cuda", enabled=cfg.amp):
            logits, solv_logits = model(
                paths, masks, gate_types=gtypes if cfg.use_gate_type_embedding else None
            )
            loss, avg_reward, valid_rate, batch_edge_acc, batch_c_viol = reinforce_loss(
                logits=logits,
                gate_types=gtypes,
                mask_valid=masks,
                solvability_logits=solv_logits,
                solvability_labels=solv_labels,
                anchor_p=anchor_p,
                anchor_l=anchor_l,
                anchor_v=anchor_v,
                entropy_beta=cfg.entropy_beta,
                constraint_mask=c_mask,
                constraint_vals=c_vals,
                node_ids=node_ids,
                lambda_logic=cfg.lambda_logic,
                lambda_full_logic=cfg.lambda_full_logic,
                soft_edge_lambda=cfg.soft_edge_lambda,
                normalize_reward=cfg.normalize_reward,
                anchor_alpha=cfg.anchor_reward_alpha,
                gumbel_temp=0.1,
            )

        loss_val = loss.mean()
        total_loss += float(loss_val.item())
        total_reward += float(avg_reward.mean().item())
        total_valid += float(valid_rate.mean().item())
        total_edge_acc += float(batch_edge_acc.mean().item())
        total_trivial += float(batch_c_viol.mean().item())
        total_batches += 1

        if cfg.verbose and cfg.log_interval > 0 and (batch_idx + 1) % cfg.log_interval == 0:
            elapsed = time.time() - start_time
            it_per_s = (batch_idx + 1) / max(1e-6, elapsed)
            dbg = _debug_metrics_from_logits(
                logits,
                node_ids,
                masks,
                files,
                anchor_p=anchor_p,
                anchor_l=anchor_l,
                anchor_v=anchor_v,
                solvability_logits=solv_logits,
                solvability_labels=solv_labels,
            )
            print(
                f"[val] batch {batch_idx + 1} "
                f"avg_loss={total_loss / max(1, total_batches):.4f} "
                f"path_acc={total_valid / max(1, total_batches):.4f} "
                f"edge_acc={dbg['edge_acc']:.3f} "
                f"reconv={dbg['reconv_match_rate']:.3f} "
                f"solv_acc={dbg['solvability_acc']:.3f} speed={it_per_s:.2f} it/s"
            )

        if cfg.max_val_batches > 0 and (batch_idx + 1) >= cfg.max_val_batches:
            break

    return (
        total_loss / max(1, total_batches),
        total_reward / max(1, total_batches),
        total_valid / max(1, total_batches),
        total_edge_acc / max(1, total_batches),
        total_trivial / max(1, total_batches),
    )


def save_checkpoint(path: str, model: nn.Module, cfg: TrainConfig, best: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": (model.module if hasattr(model, "module") else model).state_dict(),
            "config": asdict(cfg),
            "best": best,
        },
        path,
    )


def cmd_train(args: argparse.Namespace) -> None:
    print(f"[DEBUG] Entering cmd_train with args: {args.cmd}", flush=True)
    cfg = TrainConfig(
        dataset=args.dataset,
        output=args.output,
        epochs=args.epochs,
        batch_size=getattr(args, "batch_size", 8),
        verbose=getattr(args, "verbose", False),
        add_logic_value=getattr(args, "add_logic_value", True),
        max_train_batches=getattr(args, "max_train_batches", 0),
        max_val_batches=getattr(args, "max_val_batches", 0),
        log_interval=getattr(args, "log_interval", 500),
        dataset_anchor_hint=getattr(args, "dataset_anchor_hint", True),
        nhead=getattr(args, "nhead", 4),
        num_encoder_layers=getattr(args, "enc_layers", 1),
        num_interaction_layers=getattr(args, "int_layers", 1),
        dim_feedforward=getattr(args, "ffn_dim", 512),
        model_dim=getattr(args, "model_dim", 512),
        num_workers=getattr(args, "num_workers", 8),
        pin_memory=getattr(args, "pin_memory", True),
        bench_dir=getattr(args, "bench_dir", ""),
        amp=getattr(args, "amp", False),
        include_hard_negatives=getattr(args, "include_hard_negatives", False),
        soft_edge_lambda=getattr(args, "soft_edge_lambda", 1.0),
        max_len=getattr(args, "max_len", 0),
        entropy_beta=getattr(args, "entropy_beta", 0.0),
        constrained_curriculum=getattr(args, "constrained_curriculum", False),
        max_constraint_prob=getattr(args, "max_constraint_prob", 0.5),
        processed_dir=getattr(args, "processed_dir", None),
        lambda_logic=getattr(args, "lambda_logic", 0.0),
        lambda_full_logic=getattr(args, "lambda_full_logic", 0.0),
        gumbel_temp=getattr(args, "gumbel_temp", 1.0),
        gumbel_anneal_rate=getattr(args, "gumbel_anneal_rate", 0.99),
        grad_accum=getattr(args, "grad_accum", 1),
        checkpointing=getattr(args, "checkpointing", False),
        shard_cache_size=getattr(args, "shard_cache_size", 10),
        max_paths=getattr(args, "max_paths", 200),
        inject_constraints=getattr(args, "inject_constraints", False),
        pretrained=getattr(args, "pretrained", None),
    )

    # Diagnostic logs for memory
    eff_batch = cfg.batch_size * cfg.grad_accum
    print(f"[RECONV-MEM] Batch Size (Physical): {cfg.batch_size}")
    print(f"[RECONV-MEM] Grad Accumulation: {cfg.grad_accum}")
    print(f"[RECONV-MEM] Effective Batch Size: {eff_batch}")

    # Handle checkpoint-dir alias
    if getattr(args, "checkpoint_dir", None):
        cfg.output = args.checkpoint_dir

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if cfg.verbose:
        if device.type == "cuda":
            try:
                dev_name = torch.cuda.get_device_name(device)
            except Exception:
                dev_name = "CUDA device"
            print(f"Using device: {device} ({dev_name})", flush=True)
        else:
            print("Using device: cpu (CUDA not available)")

    train_loader, val_loader = make_dataloaders(cfg, device)

    # Infer actual embedding dimension from a single sample to avoid spawning
    # the entire heavy DataLoader queue just for a shape check.
    print(
        "Probing dataset dimensions...",
        flush=True,
    )
    import time

    start_time = time.time()

    # Get a single sample directly from the underlying dataset
    probe_sample = train_loader.dataset[0]
    raw_dim = int(probe_sample["paths_emb"].shape[-1])

    # Account for the padding that reconv_collate performs (divisibility by nhead)
    nhead = cfg.nhead
    if raw_dim % nhead == 0:
        observed_dim = raw_dim
    else:
        observed_dim = ((raw_dim // nhead) + 1) * nhead

    elapsed = time.time() - start_time
    print(f"Dimensions probed in {elapsed:.4f}s. Observed dim: {observed_dim}")
    if cfg.verbose:
        print(f"Observed embedding dimension from batch: {observed_dim}")
        print(f"Number of attention heads: {nhead}")
    # Choose internal model dimension and ensure divisibility by nhead
    model_dim = int(cfg.model_dim)
    if model_dim % nhead != 0:
        new_dim = ((model_dim // nhead) + 1) * nhead
        if cfg.verbose:
            print(
                f"Adjusting model_dim from {model_dim} to {new_dim} "
                f"to be divisible by nhead={nhead}"
            )
        model_dim = new_dim

    print(
        f"Initializing MultiPathTransformer (input={observed_dim}, "
        f"model={model_dim}, heads={nhead})...",
        flush=True,
    )
    model = MultiPathTransformer(
        input_dim=observed_dim,
        model_dim=model_dim,
        nhead=nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_interaction_layers=cfg.num_interaction_layers,
        dim_feedforward=cfg.dim_feedforward,
    ).to(device)
    print("Model initialized and moved to device.")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # 4. LOAD WEIGHTS (Pretrained or Resume)
    # Priority: 1. --pretrained flag  2. --output/best_model.pth (auto-resume)
    weight_path = None
    if cfg.pretrained:
        if os.path.isfile(cfg.pretrained):
            weight_path = cfg.pretrained
        else:
            weight_path = os.path.join(cfg.pretrained, "best_model.pth")
    else:
        auto_resume = os.path.join(cfg.output, "best_model.pth")
        if os.path.exists(auto_resume):
            weight_path = auto_resume

    if weight_path and os.path.exists(weight_path):
        print(f"Loading weights from {weight_path}...")
        try:
            state = torch.load(weight_path, map_location=device)
            # Check all possible key names for model weights
            if "state_dict" in state:
                weights = state["state_dict"]
            elif "model_state_dict" in state:
                weights = state["model_state_dict"]
            else:
                weights = state

            has_module = any(k.startswith("module.") for k in weights.keys())
            is_dp = isinstance(model, nn.DataParallel)
            if is_dp and not has_module:
                weights = {"module." + k: v for k, v in weights.items()}
            elif not is_dp and has_module:
                weights = {k.replace("module.", ""): v for k, v in weights.items()}

            model.load_state_dict(weights, strict=False)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load weights: {e}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    # Cosine LR scheduler with linear warmup
    warmup_epochs = min(3, cfg.epochs // 4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, cfg.epochs - warmup_epochs), eta_min=1e-6
    )

    best_val = float("inf")
    if cfg.verbose:
        nb_train = len(train_loader) if hasattr(train_loader, "__len__") else 0
        nb_val = len(val_loader) if hasattr(val_loader, "__len__") else 0
        print(
            f"Starting training: train_batches={nb_train}, "
            f"val_batches={nb_val}, batch_size={cfg.batch_size}"
        )
    for epoch in range(1, cfg.epochs + 1):
        # LR Warmup (first few epochs)
        if epoch <= warmup_epochs:
            warmup_lr = cfg.lr * (epoch / max(1, warmup_epochs))
            for pg in optim.param_groups:
                pg["lr"] = warmup_lr

        tr_loss, tr_reward, tr_acc, tr_edge, tr_c_viol = train_one_epoch(
            model, train_loader, optim, scaler, device, cfg, epoch=epoch
        )
        va_loss, va_reward, va_acc, va_edge, va_c_viol = evaluate(model, val_loader, device, cfg)

        current_lr = optim.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f} avg_reward={tr_reward:.4f} path_acc={tr_acc:.4f} "
            f"edge_acc={tr_edge:.4f} c_viol={tr_c_viol:.4f} | "
            f"val_loss={va_loss:.4f} avg_reward={va_reward:.4f} path_acc={va_acc:.4f} "
            f"edge_acc={va_edge:.4f} c_viol={va_c_viol:.4f} "
            f"lr={current_lr:.2e}"
        )

        # Step LR scheduler after warmup
        if epoch > warmup_epochs:
            scheduler.step()

        # Save periodic checkpoint
        if epoch % 10 == 0 or epoch == cfg.epochs:
            save_checkpoint(
                os.path.join(cfg.output, f"checkpoint_epoch_{epoch}.pth"),
                model,
                cfg,
                best=False,
            )

        # Save best by validation reward (maximize)
        if (-va_reward) < best_val:
            best_val = -va_reward
            save_checkpoint(os.path.join(cfg.output, "best_model.pth"), model, cfg, best=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal reconv transformer trainer")
    sub = p.add_subparsers(dest="cmd", required=True)

    # SHARD command
    s = sub.add_parser("shard", help="Convert pickle dataset to lazy-loadable shards")
    s.add_argument("--dataset", type=str, required=True, help="Input .pkl dataset")
    s.add_argument("--output-dir", type=str, required=True, help="Output directory for shards")
    s.add_argument("--shard-size", type=int, default=5000, help="Samples per shard")
    s.add_argument("--max-path-length", type=int, default=50, help="Max length to pad/truncate")

    # TRAIN command
    t = sub.add_parser("train", help="Run supervised training")
    t.add_argument(
        "--dataset",
        type=str,
        default="data/datasets/reconv_dataset.pkl",
        help="Path to dataset .pkl",
    )
    t.add_argument(
        "--processed-dir",
        type=str,
        help="Directory containing pre-processed shard_*.pt files",
    )
    t.add_argument(
        "--output",
        type=str,
        default="checkpoints/reconv_minimal",
        help="Output checkpoint directory",
    )
    t.add_argument(
        "--pretrained",
        type=str,
        help="Path to pretrained model file or directory (loads best_model.pth)",
    )
    t.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Output checkpoint directory (alias for --output)",
    )
    t.add_argument("--bench-dir", type=str, default="", help="Base directory for benchmark files")
    t.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP)")
    t.add_argument(
        "--include-hard-negatives",
        action="store_true",
        help="Include hard negatives (currently ignored)",
    )
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument(
        "--constrained-curriculum",
        action="store_true",
        help="Enable constrained path training curriculum (random constraints in loop)",
    )
    t.add_argument(
        "--inject-constraints",
        action="store_true",
        help="Enable solver-consistent constraint injection at dataset level",
    )
    t.add_argument(
        "--max-constraint-prob",
        type=float,
        default=0.5,
        help="Maximum probability of masking a constrained node",
    )

    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--verbose", action="store_true")
    # Model capacity
    t.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    t.add_argument("--enc-layers", type=int, default=3, help="Number of shared path encoder layers")
    t.add_argument("--int-layers", type=int, default=3, help="Number of path interaction layers")
    t.add_argument("--ffn-dim", type=int, default=512, help="Transformer feedforward dimension")
    t.add_argument(
        "--model-dim",
        type=int,
        default=512,
        help="Internal Transformer model dimension (must be divisible by nhead)",
    )
    t.add_argument(
        "--add-logic-value",
        action="store_true",
        default=True,
        help="Add logic value (0/1/X) as one-hot feature to embeddings",
    )
    t.add_argument(
        "--no-logic-value",
        dest="add_logic_value",
        action="store_false",
        help="Disable logic value feature",
    )
    t.add_argument(
        "--dataset-anchor-hint",
        action="store_true",
        default=True,
        help="Generate anchor hint inside the dataset loader and include in batch",
    )
    t.add_argument(
        "--no-dataset-anchor-hint",
        dest="dataset_anchor_hint",
        action="store_false",
        help="Disable dataset-level anchor generation",
    )
    # DataLoader performance
    t.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers when CUDA is available",
    )
    t.add_argument(
        "--pin-memory",
        action="store_true",
        default=True,
        help="Enable DataLoader pin_memory for CUDA",
    )
    t.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable DataLoader pin_memory",
    )
    t.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Limit number of training batches per epoch (0 = no limit)",
    )
    t.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="Limit number of validation batches per eval (0 = no limit)",
    )
    t.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="Batches between progress logs when --verbose is set",
    )
    t.add_argument(
        "--soft-edge-lambda",
        type=float,
        default=1.0,
        help="Weight for soft edge consistency loss",
    )
    t.add_argument(
        "--max-len",
        type=int,
        default=0,
        help="Filter dataset for max path length (Curriculum Learning)",
    )
    t.add_argument(
        "--entropy-beta",
        type=float,
        default=0.0,
        help="Entropy regularization weight (negative to minimize)",
    )
    t.add_argument(
        "--lambda-logic",
        type=float,
        default=0.0,
        help="Weight for reconvergence logic consistency loss (Phase 7)",
    )
    t.add_argument(
        "--lambda-full-logic",
        type=float,
        default=0.0,
        help="Weight for full-path gate consistency loss",
    )
    t.add_argument(
        "--gumbel-temp",
        type=float,
        default=1.0,
        help="Initial Gumbel Softmax temperature",
    )
    t.add_argument(
        "--gumbel-anneal-rate",
        type=float,
        default=0.99,
        help="Annealing rate per epoch",
    )
    t.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    t.add_argument(
        "--checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save VRAM.",
    )

    t.add_argument(
        "--shard-cache-size",
        type=int,
        default=25,
        help="Number of shards to cache in memory per worker. Maximize this to fill RAM.",
    )
    t.add_argument(
        "--max-paths",
        type=int,
        default=200,
        help="Maximum number of paths per sample (truncation) to prevent OOM.",
    )

    return p


def main() -> None:
    print("[DEBUG] Program started", flush=True)
    parser = build_argparser()
    args = parser.parse_args()
    print(f"[DEBUG] Args parsed: {args.cmd}", flush=True)

    if args.cmd == "train":
        print("[DEBUG] Orchestrating training command...", flush=True)
        cmd_train(args)
    elif args.cmd == "shard":
        print("[DEBUG] Orchestrating sharding command...", flush=True)
        ReconvergentPathsDataset.preprocess_to_shards(
            input_dataset_path=args.dataset,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            max_path_length=args.max_path_length,
        )
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
