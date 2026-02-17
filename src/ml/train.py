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
    reconv_collate,
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
    max_constraint_prob: float = 0.5
    enforce_constraints: bool = True
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
    micro_batch_size: int = 256  # Safe default to avoid prefetch memory blowup


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
    )
    print(f"Dataset ready. Splitting {len(dataset)} samples...", flush=True)

    using_processed_shards = bool(getattr(dataset, "_use_processed", False))
    # Minimal split: 90/10 train/val
    n = len(dataset)
    n_train = max(1, int(0.9 * n))
    n_val = max(1, n - n_train)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    print(f"Split done: {n_train} train, {n_val} val.", flush=True)

    # Use workers and pinned memory for faster host->device transfer when on CUDA
    if device.type == "cuda":
        dl_workers = cfg.num_workers
        prefetch_factor = 2 if dl_workers > 0 else None
        persistent_workers = dl_workers > 0

        # Processed shard loading is I/O heavy and each worker keeps one shard
        # resident. Tune worker behavior to reduce host RAM pressure and avoid
        # loading too many shards concurrently.
        if using_processed_shards and dl_workers > 0:
            dl_workers = min(dl_workers, 2)
            prefetch_factor = 1
            persistent_workers = False

        print(
            f"Initializing DataLoaders (workers={dl_workers}, batch_size={cfg.micro_batch_size}, "
            f"processed_shards={using_processed_shards})...",
            flush=True,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.micro_batch_size,
            shuffle=True,
            collate_fn=reconv_collate,
            num_workers=dl_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.micro_batch_size,
            shuffle=False,
            collate_fn=reconv_collate,
            num_workers=dl_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        print("DataLoaders initialized.", flush=True)
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=reconv_collate,
        )
        val_loader = DataLoader(
            val_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=reconv_collate
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
    pbar = tqdm(loader, desc=f"Epoch {epoch}", total=target_batches, disable=not cfg.verbose)

    # Calculate actual grad accumulation needed to hit target batch size
    # logical_batch_size = cfg.batch_size (user's intended total batch)
    accum_steps = max(1, cfg.batch_size // cfg.micro_batch_size)
    if cfg.grad_accum > 1:
        # If user explicitly set grad_accum, honor it as well (multiplier)
        accum_steps *= cfg.grad_accum

    print(
        f"[Memory] Total Batch={cfg.batch_size}, \
            Micro-Batch={cfg.micro_batch_size}, \
            Accumulation Steps={accum_steps}"
    )
    if cfg.micro_batch_size > 1024:
        print("[WARNING] micro-batch-size > 1024 may crash host RAM due to DataLoader prefetching.")

    for batch_idx, batch in enumerate(pbar):
        paths = batch["paths_emb"].to(device)
        masks = batch["attn_mask"].to(device)
        node_ids = batch["node_ids"].to(device)
        files = batch["files"]
        c_mask = batch["constraint_mask"].to(device) if "constraint_mask" in batch else None
        c_vals = batch["constraint_vals"].to(device) if "constraint_vals" in batch else None

        # Initialize Logic Value Vector to "Unknown" [0, 0, 1] if enabled
        if cfg.add_logic_value:
            B_cur, P_cur, L_cur, D_cur = paths.shape
            if D_cur >= 3:
                # IMPORTANT: Perform this on CPU to avoid
                # "CUDA error: no kernel image is available"
                # which seems to happen on some GPU architectures with specific indexing
                # patterns in DataParallel.
                # This block is now applied after moving to device, assuming paths are on device.
                # If this causes issues, it might need to be done in collate_fn or dataset.
                paths[..., D_cur - 3] = 0.0
                paths[..., D_cur - 2] = 0.0
                paths[..., D_cur - 1] = 1.0

        # Inject constraints into embeddings
        if cfg.add_logic_value and c_mask is not None:
            # We always check for constraints but only inject if mask is present
            if c_mask.any():
                D = paths.shape[-1]
                if D >= 3:
                    valid_mask = c_mask
                    if valid_mask.any():
                        targets = c_vals[valid_mask]
                        one_hot = torch.zeros(
                            (targets.shape[0], 3), device=device, dtype=paths.dtype
                        )
                        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
                        paths_flat = paths.view(-1, D)
                        mask_flat = valid_mask.view(-1)
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
            # Scale loss for accumulation
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        total_loss += float(loss.item()) * accum_steps
        total_reward += float(avg_reward)
        total_valid += float(valid_rate)
        total_edge_acc += float(batch_edge_acc)
        total_trivial += float(batch_c_viol)
        total_batches += 1
        bdone += paths.size(0)

        if (
            cfg.verbose
            and cfg.log_interval > 0
            and (batch_idx + 1) % (cfg.log_interval // accum_steps + 1) == 0
        ):
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
                    "acc": f"{total_valid / max(1, total_batches):.4f}",
                    "solv": f"{dbg['solvability_acc']:.3f}",
                    "edge": f"{dbg['edge_acc']:.3f}",
                }
            )

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

    pbar = tqdm(loader, desc="Eval", total=target_batches, disable=not cfg.verbose)
    for batch_idx, batch in enumerate(pbar):
        paths = batch["paths_emb"]
        masks = batch["attn_mask"]
        node_ids = batch["node_ids"]
        files = batch["files"]

        if device.type == "cuda":
            paths = paths.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            node_ids = node_ids.to(device, non_blocking=True)

        c_prob = cfg.max_constraint_prob if cfg.constrained_curriculum else 0.0
        c_mask, c_vals = generate_constraints(node_ids, files, c_prob)
        c_mask = c_mask.to(device)
        c_vals = c_vals.to(device)

        if cfg.add_logic_value and c_mask.any():
            D = paths.shape[-1]
            if D >= 3:
                valid_mask = c_mask
                if valid_mask.any():
                    targets = c_vals[valid_mask]
                    one_hot = torch.zeros((targets.shape[0], 3), device=device, dtype=paths.dtype)
                    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
                    paths_flat = paths.view(-1, D)
                    mask_flat = valid_mask.view(-1)
                    paths_flat[mask_flat, D - 3 : D] = one_hot

        if cfg.anchor_hint:
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

        total_loss += float(loss.item())
        total_reward += float(avg_reward)
        total_valid += float(valid_rate)
        total_edge_acc += float(batch_edge_acc)
        total_trivial += float(batch_c_viol)
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
                f"acc={total_valid / max(1, total_batches):.4f} "
                f"edge_acc={dbg['edge_acc']:.3f} "
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
    print(f"[DEBUG] Entering cmd_train with args: {args}", flush=True)
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
        micro_batch_size=getattr(args, "micro_batch_size", 256),
    )

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

    # Infer actual embedding dimension from a real batch to avoid mismatches
    # with processed shards and logic-value features.
    print(
        "Retrieving first batch to probe dimensions (spawning workers, might take a moment)...",
        flush=True,
    )
    import time

    start_time = time.time()
    probe_batch = next(iter(train_loader))
    elapsed = time.time() - start_time
    observed_dim = int(probe_batch["paths_emb"].shape[-1])
    nhead = cfg.nhead
    print(f"First batch retrieved in {elapsed:.2f}s")
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

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    best_val = float("inf")
    if cfg.verbose:
        nb_train = len(train_loader) if hasattr(train_loader, "__len__") else 0
        nb_val = len(val_loader) if hasattr(val_loader, "__len__") else 0
        print(
            f"Starting training: train_batches={nb_train}, "
            f"val_batches={nb_val}, batch_size={cfg.batch_size}"
        )
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_reward, tr_acc, tr_edge, tr_c_viol = train_one_epoch(
            model, train_loader, optim, scaler, device, cfg, epoch=epoch
        )
        va_loss, va_reward, va_acc, va_edge, va_c_viol = evaluate(model, val_loader, device, cfg)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f} avg_reward={tr_reward:.4f} acc={tr_acc:.4f} "
            f"edge_acc={tr_edge:.4f} c_viol={tr_c_viol:.4f} | "
            f"val_loss={va_loss:.4f} avg_reward={va_reward:.4f} acc={va_acc:.4f} "
            f"edge_acc={va_edge:.4f} c_viol={va_c_viol:.4f}"
        )

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
        help="Enable constrained path training curriculum",
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
        default=False,
        help="Use gradient checkpointing to save memory",
    )
    t.add_argument(
        "--micro-batch-size",
        type=int,
        default=256,
        help="Amount of data processed at once. Keep small for memory safety.",
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
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
