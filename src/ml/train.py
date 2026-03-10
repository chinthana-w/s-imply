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
import gc
import os
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ml.core.dataset import (
    ReconvergentPathsDataset,
    ShardBatchSampler,
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


def _curriculum_constraint_prob(epoch: int, total_epochs: int) -> float:
    """Return constraint probability for curriculum training.

    Phase 1 (first 25% epochs): 0%
    Phase 2 (remaining 75%): fixed 0.8
    """
    if total_epochs <= 0:
        return 0.0

    free_epochs = max(1, total_epochs // 4)
    if epoch <= free_epochs:
        return 0.0

    return 0.8


def _build_training_constraints(
    node_ids: torch.Tensor,
    masks: torch.Tensor,
    epoch: int,
    total_epochs: int,
    terminus_vals: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate curriculum constraints with always-constrained path termini.

    Args:
        terminus_vals: [B] optional pre-computed terminus targets (e.g. from anchor).
            If None, random 0/1 is used per sample. Providing circuit-consistent values
            (from the anchor) avoids conflicts with gate-logic loss and reduces c_viol.

    Returns:
        constraint_mask: [B, P, L] bool
        constraint_vals: [B, P, L] long (0 or 1)
    """
    device = node_ids.device
    B, P, _ = node_ids.shape
    valid_nodes = masks & (node_ids > 0)

    c_mask = torch.zeros_like(node_ids, dtype=torch.bool)
    c_vals = torch.zeros_like(node_ids, dtype=torch.long)

    # Pick one target value per sample and apply it to all valid path termini.
    # Prefer circuit-consistent values derived from anchor over pure random.
    if terminus_vals is None:
        terminus_vals = torch.randint(0, 2, (B,), device=device, dtype=torch.long)

    # Vectorised terminus detection — replaces the O(B*P) Python loop.
    # valid_len[b, p] = number of valid positions in path p of sample b.
    valid_len = masks.long().sum(dim=2)  # [B, P]
    term_idx = (valid_len - 1).clamp(min=0)  # [B, P] index of last valid position
    has_valid = valid_len > 0  # [B, P]

    # Gather node_ids at the terminus position for each (b, p).
    term_node = node_ids.gather(2, term_idx.unsqueeze(2)).squeeze(2)  # [B, P]
    terminus_ok = has_valid & (term_node > 0)  # [B, P]

    # Write terminus constraint: scatter True/value only where terminus_ok.
    c_mask.scatter_(2, term_idx.unsqueeze(2), terminus_ok.unsqueeze(2))
    # Broadcast terminus_vals [B] -> [B, P] -> [B, P, 1] for scatter.
    tv_bp = terminus_vals.unsqueeze(1).expand(B, P)
    c_vals.scatter_(2, term_idx.unsqueeze(2), (tv_bp * terminus_ok.long()).unsqueeze(2))

    # Additional constraints follow stepped curriculum only on non-terminus nodes.
    extra_prob = _curriculum_constraint_prob(epoch, total_epochs)
    if extra_prob > 0.0:
        extra_candidates = valid_nodes & (~c_mask)
        sampled = (torch.rand_like(node_ids, dtype=torch.float) < extra_prob) & extra_candidates
        c_mask = c_mask | sampled
        random_vals = torch.randint_like(node_ids, 0, 2)
        c_vals[sampled] = random_vals[sampled]

    return c_mask, c_vals


def _shuffle_paths_training_batch(
    paths: torch.Tensor,
    masks: torch.Tensor,
    node_ids: torch.Tensor,
    gate_types: Optional[torch.Tensor] = None,
    anchor_p: Optional[torch.Tensor] = None,
    constraint_mask: Optional[torch.Tensor] = None,
    constraint_vals: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Shuffle path dimension during training while keeping related tensors aligned."""
    B, P, _, _ = paths.shape
    perm = torch.argsort(torch.rand(B, P, device=paths.device), dim=1)

    gather_pl = perm.unsqueeze(-1).expand(-1, -1, paths.size(2))
    gather_pld = gather_pl.unsqueeze(-1).expand(-1, -1, -1, paths.size(3))

    paths = torch.gather(paths, dim=1, index=gather_pld)
    masks = torch.gather(masks, dim=1, index=gather_pl)
    node_ids = torch.gather(node_ids, dim=1, index=gather_pl)

    if gate_types is not None:
        gate_types = torch.gather(gate_types, dim=1, index=gather_pl)

    if constraint_mask is not None:
        constraint_mask = torch.gather(constraint_mask, dim=1, index=gather_pl)

    if constraint_vals is not None:
        constraint_vals = torch.gather(constraint_vals, dim=1, index=gather_pl)

    if anchor_p is not None:
        inverse_perm = torch.argsort(perm, dim=1)
        valid_anchor = anchor_p >= 0
        mapped_anchor = inverse_perm.gather(1, anchor_p.clamp(min=0).unsqueeze(1)).squeeze(1)
        anchor_p = torch.where(valid_anchor, mapped_anchor, anchor_p)

    return paths, masks, node_ids, gate_types, anchor_p, constraint_mask, constraint_vals


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

    # Constraint loss weight — increase (e.g. 5.0) to force the model to honour
    # constraints even when they conflict with the soft-edge gate-logic loss.
    lambda_constraint: float = 1.0

    # CUDA OOM recovery: exponential backoff retry settings.
    oom_max_retries: int = 3   # number of retry attempts before skipping a batch
    oom_base_wait: float = 2.0  # base wait in seconds; actual wait = base * 2^attempt
    resume: bool = False  # if True, load full training state from output/resume.pth


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

    # Use ShardBatchSampler when shards are available: groups all indices from the
    # same shard into each batch so get_shard_batch() can slice the whole batch in
    # one tensor op instead of 4096 individual .clone() calls.  num_workers=0 is
    # intentional — the shard is already in RAM and a single slice in the main
    # process is ~10x faster than forking workers that each re-clone per sample.
    use_shard_sampler = getattr(dataset, "_use_processed", False)

    if use_shard_sampler:
        # Keep all shards resident — set cache_size to total shards before warmup,
        # otherwise the default cache_size=1 would evict shards during warmup and
        # get_shard_batch() would raise KeyError when it tries to access an evicted shard.
        dataset.cache_size = len(dataset._shard_lens)
        dataset.warmup_shards(0)  # 0 = all shards
        print(f"Dataset ready: {len(dataset)} samples across {len(dataset._shard_lens)} shards.")

        # Split at the shard level: last 10% of shards → val, rest → train.
        n_shards = len(dataset._shard_lens)
        n_val_shards = max(1, n_shards // 10)
        train_shard_lens = dataset._shard_lens[: n_shards - n_val_shards]
        val_shard_lens = dataset._shard_lens[n_shards - n_val_shards :]
        # Adjust val offsets: ShardBatchSampler needs lens + correct global offsets,
        # so build separate samplers with an offset shift for the val portion.
        val_offset = sum(train_shard_lens)

        class _OffsetShardBatchSampler(ShardBatchSampler):
            """ShardBatchSampler with a fixed global index offset for val shards."""

            def __init__(self, shard_lens, batch_size, offset, shuffle):
                super().__init__(shard_lens, batch_size, shuffle=shuffle)
                self._global_offset = offset
                # Recompute offsets relative to the full dataset
                off = offset
                self._offsets = []
                for n in shard_lens:
                    self._offsets.append(off)
                    off += n

        train_sampler = ShardBatchSampler(
            train_shard_lens, cfg.batch_size, shuffle=True
        )
        val_sampler = _OffsetShardBatchSampler(
            val_shard_lens, cfg.batch_size, offset=val_offset, shuffle=False
        )

        # PyTorch's DataLoader always calls dataset[i] for each index yielded by
        # batch_sampler, then passes the list of results to collate_fn.  We cannot
        # intercept the raw index list that way.  Instead, wrap the samplers in a
        # simple IterableDataset that calls get_shard_batch() directly and yields
        # ready-made batch dicts — no __getitem__ or collate involved.
        from torch.utils.data import IterableDataset as _IterableDataset

        class _ShardIterDataset(_IterableDataset):
            def __init__(self, ds, sampler):
                self._ds = ds
                self._sampler = sampler

            def __iter__(self):
                for idx_list in self._sampler:
                    yield self._ds.get_shard_batch(idx_list)

            def __len__(self):
                return len(self._sampler)

        def _identity_collate(x):
            # With batch_size=None on an IterableDataset, PyTorch passes the
            # yielded item directly (not wrapped in a list) — just return it.
            return x

        n_train = sum(train_shard_lens)
        n_val = sum(val_shard_lens)
        print(f"Shard split: {n_train} train / {n_val} val (num_workers=0, shard-slice mode)")
        train_loader = DataLoader(
            _ShardIterDataset(dataset, train_sampler),
            batch_size=None,
            collate_fn=_identity_collate,
            num_workers=0,
        )
        val_loader = DataLoader(
            _ShardIterDataset(dataset, val_sampler),
            batch_size=None,
            collate_fn=_identity_collate,
            num_workers=0,
        )
        print("DataLoaders initialized (shard-slice mode).", flush=True)
        return train_loader, val_loader

    # Fallback: raw pickle path — use worker-based loading as before
    dataset.warmup_shards(cfg.shard_cache_size)

    print(f"Dataset ready. Splitting {len(dataset)} samples...", flush=True)
    n = len(dataset)
    n_train = max(1, int(0.9 * n))
    n_val = max(1, n - n_train)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    print(f"Split done: {n_train} train, {n_val} val.", flush=True)

    prefetch = 2
    if cfg.batch_size >= 512:
        prefetch = 1
        print(
            f"[RECONV-MEM] Large batch size ({cfg.batch_size}). "
            f"Reducing prefetch_factor to {prefetch} to save RAM."
        )

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
    scaler: object,
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

    constraint_prob = (
        _curriculum_constraint_prob(epoch, cfg.epochs) if cfg.constrained_curriculum else 0.0
    )

    if cfg.verbose:
        free_epochs_disp = max(1, cfg.epochs // 4)
        print(
            f"[curriculum] epoch={epoch} constraint_prob={constraint_prob:.3f} "
            f"(Phase {'1' if epoch <= free_epochs_disp else '2'})"
        )

    if cfg.verbose:
        print(f"Starting epoch {epoch} loop (waiting for DataLoader)...", flush=True)
    # Compute once per epoch — constant within an epoch.
    gumbel_t = max(0.1, cfg.gumbel_temp * (cfg.gumbel_anneal_rate ** (epoch - 1)))

    pbar = tqdm(loader, desc=f"Epoch {epoch}", total=target_batches, unit="batch")

    for batch_idx, batch in enumerate(pbar):
        paths = batch["paths_emb"].to(device, non_blocking=True)
        masks = batch["attn_mask"].to(device, non_blocking=True)
        node_ids = batch["node_ids"].to(device, non_blocking=True)
        files = batch["files"]
        c_mask = (
            batch["constraint_mask"].to(device, non_blocking=True)
            if "constraint_mask" in batch
            else None
        )
        c_vals = (
            batch["constraint_vals"].to(device, non_blocking=True)
            if "constraint_vals" in batch
            else None
        )

        # Resolve gate types before shuffling so all per-path tensors can be shuffled together.
        if "gate_types" in batch:
            gtypes = batch["gate_types"].to(device)
        else:
            gtypes = resolve_gate_types(node_ids, files, device)

        # Keep anchor path indices aligned if dataset supplied anchors.
        batch_anchor_p = batch.get("anchor_p", None)
        if batch_anchor_p is not None:
            batch_anchor_p = batch_anchor_p.to(device)

        # Shuffle path order during training.
        (
            paths,
            masks,
            node_ids,
            gtypes,
            batch_anchor_p,
            c_mask,
            c_vals,
        ) = _shuffle_paths_training_batch(
            paths=paths,
            masks=masks,
            node_ids=node_ids,
            gate_types=gtypes,
            anchor_p=batch_anchor_p,
            constraint_mask=c_mask,
            constraint_vals=c_vals,
        )
        if gtypes is None:
            raise RuntimeError("Gate types must be available during training.")

        # 1. Initialize Logic Value Vector to "Unknown" [0, 0, 1] ONLY IF dataset didn't do it
        # (inject_constraints=True in dataset already handles this)
        if cfg.add_logic_value and not cfg.inject_constraints:
            D_cur = paths.shape[-1]
            if D_cur >= 3:
                # We initialize to [0, 0, 1] (Unknown)
                paths[..., D_cur - 3] = 0.0
                paths[..., D_cur - 2] = 0.0
                paths[..., D_cur - 1] = 1.0

        # 2. Curriculum constraints: phase-1 terminus-only, then stepped ramp constraints.
        if cfg.constrained_curriculum:
            # Derive circuit-consistent terminus targets from the batch anchor when
            # the anchor node sits at the path terminus.  This replaces pure-random
            # values that conflict with gate-logic ~22% of the time and cause a
            # training plateau that no amount of extra epochs can resolve.
            _terminus_vals: Optional[torch.Tensor] = None
            if batch_anchor_p is not None:
                al_batch = batch.get("anchor_l")
                av_batch = batch.get("anchor_v")
                if al_batch is not None and av_batch is not None:
                    al_dev = al_batch.to(device)
                    av_dev = av_batch.to(device)
                    B_cur = node_ids.shape[0]
                    valid_anchor = batch_anchor_p >= 0
                    if valid_anchor.any():
                        anc_p_safe = batch_anchor_p.clamp(0, masks.shape[1] - 1)
                        anc_lens = (
                            masks[torch.arange(B_cur, device=device), anc_p_safe].long().sum(dim=1)
                        )
                        anc_terminus = (anc_lens - 1).clamp(min=0)
                        at_term = valid_anchor & (al_dev == anc_terminus) & (av_dev >= 0)
                        if at_term.any():
                            _terminus_vals = torch.randint(
                                0, 2, (B_cur,), device=device, dtype=torch.long
                            )
                            _terminus_vals[at_term] = av_dev[at_term].long().clamp(0, 1)
            c_mask, c_vals = _build_training_constraints(
                node_ids=node_ids,
                masks=masks,
                epoch=epoch,
                total_epochs=cfg.epochs,
                terminus_vals=_terminus_vals,
            )
        elif c_mask is None and not cfg.inject_constraints:
            c_prob = cfg.max_constraint_prob
            if c_prob > 0:
                c_mask, c_vals = generate_constraints(node_ids, files, c_prob)
                c_mask = c_mask.to(device)
                c_vals = c_vals.to(device)

        # 3. Inject constraints (either from batch or manually generated) into embeddings
        if cfg.add_logic_value and c_mask is not None and c_vals is not None:
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
        if cfg.anchor_hint and batch_anchor_p is None:
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
            anchor_p = batch_anchor_p
            if anchor_p is not None:
                anchor_l = batch["anchor_l"].to(device)
                anchor_v = batch["anchor_v"].to(device)
                solv_labels = batch["solvability"].to(device)
            else:
                anchor_p = anchor_l = anchor_v = solv_labels = None  # type: ignore

        # Forward + loss + backward with OOM exponential backoff.
        # On CUDA OOM: free cache, wait 2^attempt seconds, then retry.
        # After max_oom_retries failures the batch is skipped to preserve progress.
        # Initialize to satisfy type-checkers; these are always set on a successful iteration.
        loss_val: torch.Tensor = torch.tensor(0.0)
        avg_reward = valid_rate = batch_edge_acc = batch_c_viol = 0.0
        logits = solv_logits = None
        _oom_batch_skipped = False
        for _oom_attempt in range(cfg.oom_max_retries + 1):
            try:
                with torch.amp.autocast("cuda", enabled=cfg.amp):  # type: ignore[attr-defined]
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
                        lambda_constraint=cfg.lambda_constraint,
                    )

                # NaN/Inf guard: skip corrupt batches
                loss_val = loss.mean()
                if not torch.isfinite(loss_val):
                    print(f"[WARNING] Non-finite loss at batch {batch_idx}, skipping.")
                    optim.zero_grad(set_to_none=True)
                    _oom_batch_skipped = True
                    break

                scaler.scale(loss_val / cfg.grad_accum).backward()

                if (batch_idx + 1) % cfg.grad_accum == 0:
                    # Gradient clipping to prevent AMP-induced explosions
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)

                break  # success — exit OOM retry loop

            except torch.cuda.OutOfMemoryError:
                import time as _time

                optim.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()

                if _oom_attempt < cfg.oom_max_retries:
                    wait = cfg.oom_base_wait * (2**_oom_attempt)
                    print(
                        f"[OOM] batch {batch_idx} attempt {_oom_attempt + 1}"
                        f"/{cfg.oom_max_retries}: retrying in {wait}s ...",
                        flush=True,
                    )
                    _time.sleep(wait)
                else:
                    print(
                        f"[OOM] batch {batch_idx}: exhausted {cfg.oom_max_retries} retries,"
                        " skipping batch.",
                        flush=True,
                    )
                    _oom_batch_skipped = True

        if _oom_batch_skipped:
            continue

        total_loss += float(loss_val.item())
        total_reward += float(avg_reward)
        total_valid += float(valid_rate)
        total_edge_acc += float(batch_edge_acc)
        total_trivial += float(batch_c_viol)
        total_batches += 1

        # 4. RAM AWARENESS in main loop
        if (batch_idx + 1) % 50 == 0:
            mem = psutil.virtual_memory()
            if mem.percent > 90.0:
                print(f"[RECONV-MEM] Main loop detected high RAM ({mem.percent}%). Cleaning up.")
                gc.collect()
                torch.cuda.empty_cache()

        if (
            cfg.verbose
            and cfg.log_interval > 0
            and (batch_idx + 1) % cfg.log_interval == 0
            and logits is not None
        ):
            dbg = _debug_metrics_from_logits(
                logits,
                node_ids,
                masks,
                files,
                gate_types=gtypes,
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

    return (
        total_loss / max(1, total_batches),
        total_reward / max(1, total_batches),
        total_valid / max(1, total_batches),
        total_edge_acc / max(1, total_batches),
        total_trivial / max(1, total_batches),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int = 1,
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

        # Extract anchor early so constraint building can use circuit-consistent values.
        if "anchor_p" in batch:
            anchor_p = batch["anchor_p"].to(device)
            anchor_l = batch["anchor_l"].to(device)
            anchor_v = batch["anchor_v"].to(device)
            solv_labels = batch["solvability"].to(device)
        else:
            anchor_p = anchor_l = anchor_v = solv_labels = None  # type: ignore

        # 1. Initialize Logic Value Vector to "Unknown" [0, 0, 1] ONLY IF dataset didn't do it
        if cfg.add_logic_value and not cfg.inject_constraints:
            D_cur = paths.shape[-1]
            if D_cur >= 3:
                paths[..., D_cur - 3] = 0.0
                paths[..., D_cur - 2] = 0.0
                paths[..., D_cur - 1] = 1.0

        # 2. Keep evaluation curriculum-consistent with training schedule.
        if cfg.constrained_curriculum:
            # Mirror the anchor-guided terminus derivation used in train_one_epoch
            # so validation c_viol is measured on achievable constraint targets.
            _eval_terminus_vals: Optional[torch.Tensor] = None
            if anchor_p is not None and anchor_l is not None and anchor_v is not None:
                B_cur = node_ids.shape[0]
                valid_anchor = anchor_p >= 0
                if valid_anchor.any():
                    anc_p_safe = anchor_p.clamp(0, masks.shape[1] - 1)
                    anc_lens = (
                        masks[torch.arange(B_cur, device=device), anc_p_safe].long().sum(dim=1)
                    )
                    anc_terminus = (anc_lens - 1).clamp(min=0)
                    at_term = valid_anchor & (anchor_l == anc_terminus) & (anchor_v >= 0)
                    if at_term.any():
                        _eval_terminus_vals = torch.randint(
                            0, 2, (B_cur,), device=device, dtype=torch.long
                        )
                        _eval_terminus_vals[at_term] = anchor_v[at_term].long().clamp(0, 1)
            c_mask, c_vals = _build_training_constraints(
                node_ids=node_ids,
                masks=masks,
                epoch=epoch,
                total_epochs=cfg.epochs,
                terminus_vals=_eval_terminus_vals,
            )
        elif c_mask is None and not cfg.inject_constraints:
            c_prob = cfg.max_constraint_prob
            if c_prob > 0:
                c_mask, c_vals = generate_constraints(node_ids, files, c_prob)
                c_mask = c_mask.to(device)
                c_vals = c_vals.to(device)

        # 3. Inject constraints into embeddings
        if cfg.add_logic_value and c_mask is not None and c_vals is not None:
            if c_mask.any():
                D = paths.shape[-1]
                if D >= 3:
                    targets = c_vals[c_mask]
                    one_hot = torch.zeros((targets.shape[0], 3), device=device, dtype=paths.dtype)
                    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
                    paths_flat = paths.view(-1, D)
                    mask_flat = c_mask.view(-1)
                    paths_flat[mask_flat, D - 3 : D] = one_hot

        if "gate_types" in batch:
            gtypes = batch["gate_types"].to(device)
        else:
            gtypes = resolve_gate_types(node_ids, files, device)

        with torch.amp.autocast("cuda", enabled=cfg.amp):  # type: ignore[attr-defined]
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
                lambda_constraint=cfg.lambda_constraint,
            )

        loss_val = loss.mean()
        total_loss += float(loss_val.item())
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
                gate_types=gtypes,
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


def save_checkpoint(
    path: str,
    model: nn.Module,
    cfg: TrainConfig,
    best: bool = False,
    epoch: int = 0,
    optim: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    scaler: Optional[object] = None,
    best_val: float = float("inf"),
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    inner: nn.Module = getattr(model, "module", model)
    payload: dict = {
        "state_dict": inner.state_dict(),
        "config": asdict(cfg),
        "best": best,
        "epoch": epoch,
        "best_val": best_val,
    }
    if optim is not None:
        payload["optim_state_dict"] = optim.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    # Atomic write: save to a temp file then rename so a crash mid-save never
    # corrupts the checkpoint that was already there.
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


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
        lambda_constraint=getattr(args, "lambda_constraint", 1.0),
        oom_max_retries=getattr(args, "oom_max_retries", 3),
        oom_base_wait=getattr(args, "oom_base_wait", 2.0),
        resume=getattr(args, "resume", False),
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
    start_time = time.time()

    # Get a single sample directly from the underlying dataset.
    # train_loader.dataset may be a _ShardIterDataset wrapper; fall back to the
    # inner ReconvergentPathsDataset which always supports __getitem__.
    probe_ds = getattr(train_loader.dataset, "_ds", train_loader.dataset)
    probe_sample = probe_ds[0]
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
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)  # type: ignore[attr-defined]

    # Cosine LR scheduler with linear warmup
    warmup_epochs = min(3, cfg.epochs // 4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, cfg.epochs - warmup_epochs), eta_min=1e-6
    )

    best_val = float("inf")
    start_epoch = 1

    # Resume training state (optimizer, scheduler, scaler, epoch) if the
    # resume checkpoint carries them.  Weight-only checkpoints (e.g. --pretrained)
    # already loaded the model weights above; we only restore training state
    # from a full resume checkpoint stored in cfg.output.
    resume_ckpt_path = os.path.join(cfg.output, "resume.pth")
    if cfg.resume and os.path.isfile(resume_ckpt_path):
        print(f"Resuming full training state from {resume_ckpt_path}...")
        resume_state = torch.load(resume_ckpt_path, map_location=device)
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        best_val = float(resume_state.get("best_val", float("inf")))
        if "optim_state_dict" in resume_state:
            optim.load_state_dict(resume_state["optim_state_dict"])
        if "scheduler_state_dict" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        if "scaler_state_dict" in resume_state:
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        print(f"Resumed from epoch {start_epoch - 1}. best_val={best_val:.4f}")
    elif cfg.resume:
        print(
            f"[WARNING] --resume requested but {resume_ckpt_path} not found. "
            "Starting from scratch."
        )

    if cfg.verbose:
        nb_train = len(train_loader) if hasattr(train_loader, "__len__") else 0
        nb_val = len(val_loader) if hasattr(val_loader, "__len__") else 0
        print(
            f"Starting training: epoch={start_epoch}/{cfg.epochs} "
            f"train_batches={nb_train}, val_batches={nb_val}, batch_size={cfg.batch_size}"
        )

    _MAX_EPOCH_RETRIES = 5

    for epoch in range(start_epoch, cfg.epochs + 1):
        # LR Warmup (first few epochs)
        if epoch <= warmup_epochs:
            warmup_lr = cfg.lr * (epoch / max(1, warmup_epochs))
            for pg in optim.param_groups:
                pg["lr"] = warmup_lr

        # Retry loop: on any crash (DataLoader worker killed, CUDA error, etc.)
        # rebuild the DataLoader and retry the epoch up to _MAX_EPOCH_RETRIES times.
        tr_loss = tr_reward = tr_acc = tr_edge = tr_c_viol = 0.0
        va_loss = va_reward = va_acc = va_edge = va_c_viol = 0.0
        for _attempt in range(1, _MAX_EPOCH_RETRIES + 1):
            try:
                tr_loss, tr_reward, tr_acc, tr_edge, tr_c_viol = train_one_epoch(
                    model, train_loader, optim, scaler, device, cfg, epoch=epoch
                )
                va_loss, va_reward, va_acc, va_edge, va_c_viol = evaluate(
                    model, val_loader, device, cfg, epoch=epoch
                )
                break  # success — exit retry loop
            except Exception as exc:
                print(
                    f"\n[EPOCH RETRY] Epoch {epoch} attempt {_attempt}/{_MAX_EPOCH_RETRIES} "
                    f"failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if _attempt == _MAX_EPOCH_RETRIES:
                    # Save an emergency checkpoint before propagating so we don't
                    # lose the progress from epochs before this one.
                    emergency_path = os.path.join(cfg.output, "emergency.pth")
                    print(
                        f"[EPOCH RETRY] All {_MAX_EPOCH_RETRIES} attempts exhausted. "
                        f"Saving emergency checkpoint to {emergency_path} and re-raising.",
                        flush=True,
                    )
                    save_checkpoint(
                        emergency_path,
                        model,
                        cfg,
                        best=False,
                        epoch=epoch - 1,
                        optim=optim,
                        scheduler=scheduler,
                        scaler=scaler,
                        best_val=best_val,
                    )
                    raise
                # Rebuild DataLoaders to get fresh worker processes.
                print(
                    f"[EPOCH RETRY] Rebuilding DataLoaders for retry {_attempt + 1}...",
                    flush=True,
                )
                try:
                    del train_loader, val_loader
                except Exception:
                    pass
                gc.collect()
                train_loader, val_loader = make_dataloaders(cfg, device)

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

        # Always save the resumable training state so any crash can be recovered.
        save_checkpoint(
            resume_ckpt_path,
            model,
            cfg,
            best=False,
            epoch=epoch,
            optim=optim,
            scheduler=scheduler,
            scaler=scaler,
            best_val=best_val,
        )

        # Save periodic weights-only snapshot every 10 epochs
        if epoch % 10 == 0 or epoch == cfg.epochs:
            save_checkpoint(
                os.path.join(cfg.output, f"checkpoint_epoch_{epoch}.pth"),
                model,
                cfg,
                best=False,
                epoch=epoch,
                best_val=best_val,
            )

        # Save best by validation loss (minimize)
        if va_loss < best_val:
            best_val = va_loss
            save_checkpoint(
                os.path.join(cfg.output, "best_model.pth"),
                model,
                cfg,
                best=True,
                epoch=epoch,
                best_val=best_val,
            )


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
    t.add_argument(
        "--lambda-constraint",
        type=float,
        default=1.0,
        help=(
            "Weight for constraint CE loss term. Increase to 3–5 when c_viol "
            "is stuck: the constraint signal must outweigh the soft-edge gate-logic loss."
        ),
    )
    t.add_argument(
        "--oom-max-retries",
        type=int,
        default=3,
        help="Maximum CUDA OOM retry attempts per batch before skipping it (exponential backoff).",
    )
    t.add_argument(
        "--oom-base-wait",
        type=float,
        default=2.0,
        help="Base wait (s) for CUDA OOM exponential backoff; actual wait = base * 2^attempt.",
    )
    t.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume full training state (epoch, optimizer, scheduler, scaler) "
            "from <output>/resume.pth. Use this after a crash to continue without "
            "losing progress."
        ),
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
