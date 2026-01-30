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
from typing import List, Optional, Dict, Tuple
import os
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Tuple
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
    max_train_batches: int = 0   # 0 = no limit
    max_val_batches: int = 0     # 0 = no limit
    log_interval: int = 500      # batches between progress prints when verbose
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


def make_dataloaders(cfg: TrainConfig, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    # Auto-detect processed shards: look for processed/ subdirectory next to dataset
    dataset_dir = os.path.dirname(cfg.dataset)
    processed_dir = os.path.join(dataset_dir, 'reconv_processed')
    load_processed = os.path.isdir(processed_dir)
    
    # For best throughput, keep dataset tensors on CPU and move whole batches to GPU
    dataset_device = torch.device('cpu') if device.type == 'cuda' else device
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
    # Minimal split: 90/10 train/val
    n = len(dataset)
    n_train = max(1, int(0.9 * n))
    n_val = max(1, n - n_train)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    # Use workers and pinned memory for faster host->device transfer when on CUDA
    if device.type == 'cuda':
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=reconv_collate,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.num_workers > 0,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=reconv_collate,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.num_workers > 0,
        )
    else:
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, collate_fn=reconv_collate)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=reconv_collate)
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


def _compatible_anchor_value(gate_type: int, prefer_value: int) -> int:
    """Pick a compatible output value for the gate near the circuit output.

    Heuristic mapping (non-controlling bias):
    - AND/NAND -> 1
    - OR/NOR   -> 0
    - BUFF/NOT/XOR/XNOR/INPT/FROM -> prefer_value (either is satisfiable)
    """
    if gate_type == GateType.AND or gate_type == GateType.NAND:
        return 1
    if gate_type == GateType.OR or gate_type == GateType.NOR:
        return 0
    # Default: either value can be made compatible; bias to prefer_value
    return int(prefer_value)


def _generate_anchor(
    node_ids: torch.Tensor,
    mask_valid: torch.Tensor,
    files: list[str],
    prefer_value: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each batch item, randomly pick one valid path and its last gate,
    then assign a compatible anchor value based on gate type.

    Returns:
      - anchor_p: LongTensor [B] with path index, or -1 if none
      - anchor_l: LongTensor [B] with last index along that path, or -1 if none
      - anchor_v: LongTensor [B] with value in {0,1} (0 if none)
    """
    device = node_ids.device
    B, P, L = node_ids.shape[:3]
    anchor_p = torch.full((B,), -1, dtype=torch.long, device=device)
    anchor_l = torch.full((B,), -1, dtype=torch.long, device=device)
    anchor_v = torch.zeros((B,), dtype=torch.long, device=device)

    for b in range(B):
        # Collect candidate (path, last_idx) pairs
        candidates: list[tuple[int, int, int]] = []  # (p, last_idx, gate_type)
        circuit = _load_circuit(files[b])
        for p in range(P):
            valid_positions = mask_valid[b, p]
            if bool(valid_positions.any()):
                last_idx = int(valid_positions.sum().item()) - 1
                cur_id = int(node_ids[b, p, last_idx].item())
                if cur_id > 0:
                    gate_type = int(circuit[cur_id].type)
                    candidates.append((p, last_idx, gate_type))
        if not candidates:
            continue
        # Randomly pick one candidate to avoid bias
        pick_i = torch.randint(low=0, high=len(candidates), size=(1,), device=device).item()
        pick_idx = int(pick_i)
        p_sel, l_sel, g_sel = candidates[pick_idx]
        v_sel = _compatible_anchor_value(g_sel, prefer_value)
        anchor_p[b] = int(p_sel)
        anchor_l[b] = int(l_sel)
        anchor_v[b] = int(v_sel)

    return anchor_p, anchor_l, anchor_v


def _inject_anchor_into_embeddings(
    paths_emb: torch.Tensor,
    anchor_p: torch.Tensor,
    anchor_l: torch.Tensor,
    anchor_v: torch.Tensor,
    enable: bool,
) -> torch.Tensor:
    """Write the anchor logic value into the last 3 dims (one-hot 0/1/X) of the
    embedding tensor at the anchor positions. No-op if disabled or dims < 3.
    """
    if not enable:
        return paths_emb
    B, P, L, D = paths_emb.shape
    if D < 3:
        return paths_emb
    present = anchor_p.ge(0) & anchor_l.ge(0)
    if not bool(present.any()):
        return paths_emb
    bs = torch.arange(B, device=paths_emb.device)[present]
    ps = anchor_p[present]
    ls = anchor_l[present]
    vs = anchor_v[present].clamp(0, 1)  # only {0,1}
    # One-hot into last 3 dims: map 0->[1,0,0], 1->[0,1,0]
    onehots = F.one_hot(vs, num_classes=3).to(paths_emb.dtype)
    paths_emb[bs, ps, ls, D-3:D] = onehots
    return paths_emb


def _format_seconds(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


@torch.no_grad()
def _debug_metrics_from_logits(
    logits: torch.Tensor,
    node_ids: torch.Tensor,
    mask_valid: torch.Tensor,
    files: "List[str]",
    anchor_p: "Optional[torch.Tensor]" = None,
    anchor_l: "Optional[torch.Tensor]" = None,
    anchor_v: "Optional[torch.Tensor]" = None,
) -> "Dict[str, float]":
    """Compute diagnostic metrics using greedy actions (argmax).

    Returns a dict with keys:
      - edge_acc: fraction of valid edges satisfying local constraints (0..1)
      - reconv_match_rate: fraction of samples with all-last-values-equal among those with >=2 paths
      - anchor_match_rate: fraction of present anchors where predicted value matches (if anchors provided)
      - edges_per_sample: average number of valid edges per sample
      - samples_with_edges_frac: fraction of samples with any valid edges
    """
    device = logits.device
    B, P, L, _ = logits.shape
    actions = torch.argmax(logits, dim=-1)  # [B,P,L]

    # Edge checks (vectorized per batch)
    valid_edges = mask_valid[:, :, 1:] & mask_valid[:, :, :-1]  # [B,P,L-1]
    prev_vals_all = actions[:, :, :-1]
    cur_vals_all = actions[:, :, 1:]

    total_edges = valid_edges.sum(dtype=torch.float32).item()
    samples_with_edges = (valid_edges.view(B, -1).any(dim=1)).float().mean().item()

    wrong_edges_total = 0.0
    for b in range(B):
        nid_b = node_ids[b]
        ids_b = nid_b[nid_b > 0].unique().tolist()
        circuit = _load_circuit(files[b])
        if ids_b:
            max_id = int(max(ids_b))
            gt_lookup = torch.full((max_id + 1,), -1, dtype=torch.long, device=device)
            for nid in ids_b:
                try:
                    gt_lookup[int(nid)] = int(circuit[int(nid)].type)
                except Exception:
                    pass
            gtypes_b = gt_lookup[nid_b.clamp(min=0, max=max_id).to(device)]
        else:
            gtypes_b = torch.full_like(nid_b, -1, dtype=torch.long, device=device)

        gt_cur = gtypes_b[:, 1:]
        prev_vals = prev_vals_all[b]
        cur_vals = cur_vals_all[b]
        ve_mask = valid_edges[b]

        ok = torch.ones_like(prev_vals, dtype=torch.bool, device=device)
        # NOT
        m = gt_cur == GateType.NOT
        ok[m] &= (1 - prev_vals[m]) == cur_vals[m]
        # BUFF
        m = gt_cur == GateType.BUFF
        ok[m] &= (prev_vals[m] == cur_vals[m])
        # AND
        m = gt_cur == GateType.AND
        if bool(m.any()):
            cur_m = cur_vals[m]
            prev_m = prev_vals[m]
            ok_m = (cur_m == 0) | ((cur_m == 1) & (prev_m == 1))
            ok[m] &= ok_m
        # NAND
        m = gt_cur == GateType.NAND
        if bool(m.any()):
            cur_m = cur_vals[m]
            prev_m = prev_vals[m]
            ok_m = (cur_m == 1) | ((cur_m == 0) & (prev_m == 1))
            ok[m] &= ok_m
        # OR
        m = gt_cur == GateType.OR
        if bool(m.any()):
            cur_m = cur_vals[m]
            prev_m = prev_vals[m]
            ok_m = (cur_m == 1) | ((cur_m == 0) & (prev_m == 0))
            ok[m] &= ok_m
        # NOR
        m = gt_cur == GateType.NOR
        if bool(m.any()):
            cur_m = cur_vals[m]
            prev_m = prev_vals[m]
            ok_m = (cur_m == 0) | ((cur_m == 1) & (prev_m == 0))
            ok[m] &= ok_m

        wrong_edges_total += ((~ok) & ve_mask).sum(dtype=torch.float32).item()

    edge_acc = 0.0 if total_edges == 0 else float((total_edges - wrong_edges_total) / max(1.0, total_edges))

    # Reconvergence match rate
    reconv_present = 0
    reconv_ok = 0
    for b in range(B):
        mask_b = mask_valid[b]
        last_idx = mask_b.sum(dim=-1) - 1
        present = last_idx >= 0
        if not bool(present.any()):
            continue
        last_idx_clamped = last_idx.clamp(min=0)
        arange_p = torch.arange(mask_b.size(0), device=device)
        last_vals = actions[b, arange_p, last_idx_clamped][present]
        if last_vals.numel() >= 2:
            reconv_present += 1
            if bool(torch.all(last_vals == last_vals[0])):
                reconv_ok += 1

    reconv_match_rate = float(reconv_ok / max(1, reconv_present))

    # Anchor match rate (if provided)
    anchor_match_rate = 0.0
    if anchor_p is not None and anchor_l is not None and anchor_v is not None:
        present = (anchor_p >= 0) & (anchor_l >= 0)
        if bool(present.any()):
            idx = torch.arange(B, device=device)[present]
            pred_vals = actions[idx, anchor_p[present], anchor_l[present]]
            matches = (pred_vals == anchor_v[present]).float().mean().item()
            anchor_match_rate = float(matches)

    return {
        'edge_acc': edge_acc,
        'reconv_match_rate': reconv_match_rate,
        'anchor_match_rate': anchor_match_rate,
        'edges_per_sample': float(total_edges / max(1, B)),
        'samples_with_edges_frac': samples_with_edges,
    }

def resolve_gate_types(node_ids: torch.Tensor, files: list[str], device: torch.device) -> torch.Tensor:
    B, P, L = node_ids.shape
    gtypes_batch = torch.full((B, P, L), -1, dtype=torch.long, device=device)
    
    for b in range(B):
        nid_b = node_ids[b]
        # Filter positive IDs for lookup
        valid_mask = nid_b > 0
        ids_b = nid_b[valid_mask].unique().tolist()
        
        circuit = _load_circuit(files[b])
        if ids_b:
            max_id = int(max(ids_b))
            gt_lookup = torch.full((max_id + 1,), -1, dtype=torch.long, device=device)
            # Safe caching trick
            for nid in ids_b:
                try:
                    if int(nid) < len(circuit):
                        gt_lookup[int(nid)] = int(circuit[int(nid)].type)
                except Exception:
                    pass
            
            # Apply lookup
            gtypes_batch[b] = gt_lookup[nid_b.clamp(min=0, max=max_id).to(device)]
            
    return gtypes_batch


def policy_loss_and_metrics(
    logits: torch.Tensor,
    node_ids: torch.Tensor,
    mask_valid: torch.Tensor,
    files: list[str],
    gate_types: torch.Tensor,
    anchor_p: torch.Tensor | None = None,
    anchor_l: torch.Tensor | None = None,
    anchor_v: torch.Tensor | None = None,
    anchor_alpha: float = 0.1,
    normalize_reward: bool = True,
    entropy_beta: float = 0.0,
    soft_edge_lambda: float = 1.0,
) -> tuple[torch.Tensor, float, float, float, float]:
    """Compute REINFORCE loss with LUT-inspired constraints and reconv consistency.

    Returns: (loss, avg_reward, valid_rate, edge_acc, trivial_rate)
    """

    B, P, L, C = logits.shape
    # Sample actions; detach to avoid retaining graph history through the sample op.
    actions = torch.distributions.Categorical(logits=logits).sample().detach()  # [B, P, L]
    # Compute log-probabilities directly from logits to avoid distribution caching quirks.
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, P, L, C]
    logp = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [B, P, L]
    
    # Get probability of 1 (Logic-1) for soft constraints
    probs_1 = torch.softmax(logits, dim=-1)[..., 1]  # [B, P, L]

    wrong_count_list: list[float] = []
    checked_list: list[float] = []
    local_wrong_list: list[float] = []
    reconv_wrong_list: list[float] = []
    edge_wrong_sum = 0.0
    edge_total_sum = 0.0
    soft_edge_loss_list: list[torch.Tensor] = []

    # Precompute masks for valid adjacent edges once
    valid_edges = mask_valid[:, :, 1:] & mask_valid[:, :, :-1]  # [B, P, L-1]
    prev_vals_all = actions[:, :, :-1]
    cur_vals_all = actions[:, :, 1:]
    
    prev_probs_all = probs_1[:, :, :-1]
    cur_probs_all = probs_1[:, :, 1:]
    for b in range(B):
        # Use pre-resolved gate types
        gtypes_b = gate_types[b]

        prev_vals = prev_vals_all[b].to(logits.device)
        cur_vals = cur_vals_all[b].to(logits.device)
        ve_mask = valid_edges[b].to(logits.device)

        # Edge-local constraints (vectorized over P, L-1)
        gt_cur = gtypes_b[:, 1:]  # gate type at current node for each edge
        
        # Define weights for gate types
        # Boost NOT (4) by 12x (Final Attempt)
        edge_weights = torch.ones_like(gt_cur, dtype=torch.float32)
        edge_weights[gt_cur == GateType.NOT] = 12.0
        



        ok = torch.ones_like(prev_vals, dtype=torch.bool, device=logits.device)
        # NOT
        m = gt_cur == GateType.NOT
        ok[m] &= (1 - prev_vals[m]) == cur_vals[m]
        # BUFF
        m = gt_cur == GateType.BUFF
        ok[m] &= (prev_vals[m] == cur_vals[m])
        # AND: cur==0 ok; cur==1 requires prev==1
        m = gt_cur == GateType.AND
        if m.any():
            cur_m = cur_vals[m]
            prev_m = prev_vals[m]
            ok_m = (cur_m == 0) | ((cur_m == 1) & (prev_m == 1))
            ok[m] &= ok_m
        # NAND: cur==1 ok; cur==0 requires prev==1
        m = gt_cur == GateType.NAND
        if m.any():
            cur_m = cur_vals[m]
            prev_m = prev_vals[m]
            ok_m = (cur_m == 1) | ((cur_m == 0) & (prev_m == 1))
            ok[m] &= ok_m
        # OR: cur==1 ok; cur==0 requires prev==0
        m = gt_cur == GateType.OR
        if m.any():
            cur_m = cur_vals[m]
            prev_m = prev_vals[m]
            ok_m = (cur_m == 1) | ((cur_m == 0) & (prev_m == 0))
            ok[m] &= ok_m
        # NOR: cur==0 ok; cur==1 requires prev==0
        m = gt_cur == GateType.NOR
        if m.any():
            cur_m = cur_vals[m]
            prev_m = prev_vals[m]
            ok_m = (cur_m == 0) | ((cur_m == 1) & (prev_m == 0))
            ok[m] &= ok_m

        # Soft Edge Constraints (Differentiable)
        # Using probs_1: prev_p, cur_p
        prev_p = prev_probs_all[b].to(logits.device)
        cur_p = cur_probs_all[b].to(logits.device)
        
        # Calculate violation per edge
        viol = torch.zeros_like(prev_p, dtype=torch.float32)

        # NOT: |cur - (1-prev)|^2
        m = gt_cur == GateType.NOT
        if m.any():
            viol[m] = (cur_p[m] - (1.0 - prev_p[m])) ** 2
            
        # BUFF: |cur - prev|^2
        m = gt_cur == GateType.BUFF
        if m.any():
            viol[m] = (cur_p[m] - prev_p[m]) ** 2
            
        # AND: cur <= prev  => ReLU(cur - prev)^2
        m = gt_cur == GateType.AND
        if m.any():
            viol[m] = F.relu(cur_p[m] - prev_p[m]) ** 2
            
        # NAND: cur >= 1 - prev => ReLU((1-prev) - cur)^2
        m = gt_cur == GateType.NAND
        if m.any():
            viol[m] = F.relu((1.0 - prev_p[m]) - cur_p[m]) ** 2
            
        # OR: cur >= prev => ReLU(prev - cur)^2
        m = gt_cur == GateType.OR
        if m.any():
            viol[m] = F.relu(prev_p[m] - cur_p[m]) ** 2

        # NOR: cur <= 1 - prev => ReLU(cur - (1-prev))^2
        m = gt_cur == GateType.NOR
        if m.any():
            viol[m] = F.relu(cur_p[m] - (1.0 - prev_p[m])) ** 2
            
        # Mask out invalid edges from loss
        viol = viol * ve_mask.float()
        
        # Apply weights 
        viol = viol * edge_weights
        
        if ve_mask.any():
             soft_edge_loss_list.append(viol.sum() / ve_mask.sum().float().clamp(min=1.0))

        wrong_edges = (~ok) & ve_mask
        wrong = wrong_edges.sum(dtype=torch.float32).item()
        checked = ve_mask.sum(dtype=torch.float32).item()

        # Track local vs reconv errors separately
        local_wrong = float(wrong) # 'wrong' currently has only edge errors
        rec_wrong = 0.0
        
        edge_wrong_sum += local_wrong
        edge_total_sum += checked
        
        # Accumulate edge accuracy lists for batch reporting
        # (Assuming edge_wrong_list/edge_total_list were placeholders, actually removed them in previous edits but let's just stick to vars)

        # Reconvergence consistency: compare last values across paths
        mask_b = mask_valid[b]
        last_idx = mask_b.sum(dim=-1) - 1  # [P]
        present = last_idx >= 0
        if bool(present.any()):
            # clamp for gather safety
            last_idx_clamped = last_idx.clamp(min=0)
            arange_p = torch.arange(mask_b.size(0), device=logits.device)
            last_vals = actions[b, arange_p, last_idx_clamped]
            last_vals = last_vals[present]
            n_present = int(present.sum().item())
            if n_present >= 2:
                ref = last_vals[0]
                mism = (last_vals[1:] != ref).sum(dtype=torch.float32).item()
                rec_wrong = float(mism)
                # Don't add to checked for edge_accuracy calculation, but for reward denom yes.
                checked += float(n_present - 1)

        local_wrong_list.append(local_wrong)
        reconv_wrong_list.append(rec_wrong)
        # Total wrong for "valid" status check (still useful for valid_rate)
        wrong_count_list.append(local_wrong + rec_wrong)
        checked_list.append(float(checked))
        
    checked_t = torch.tensor(checked_list, dtype=torch.float32, device=logits.device)
    wrong_t = torch.tensor(wrong_count_list, dtype=torch.float32, device=logits.device)
    local_wrong_t = torch.tensor(local_wrong_list, dtype=torch.float32, device=logits.device)
    reconv_wrong_t = torch.tensor(reconv_wrong_list, dtype=torch.float32, device=logits.device)
    
    # NEW DEFINITION:
    # A sample is VALID if it has NO errors (wrong_t == 0).
    # If checked_t == 0 (trivial case), it has no errors, so it is valid.
    # We track 'trivial' separately.
    valid = (wrong_t == 0)
    trivial = (checked_t == 0)
    
    # Reward:
    # If trivial: 0.0 (neutral) or 1.0 (good)? 
    # Let's say 1.0 because it satisfies constraints (vacuously).
    # If non-trivial and valid: 1.0.
    # If invalid: -wrong/checked.
    
    denom = torch.clamp(checked_t, min=1.0)
    # If valid (wrong==0), reward is 1.0. Even if trivial.
    # If invalid (wrong>0), reward is -wrong/denom.
    # GATED REWARD LOGIC:
    # 1. If Local Logic Fails (local_wrong > 0): Reward = -1.0 (Hard Penalty)
    # 2. Else (Local Logic OK):
    #    If Reconv Fails (reconv_wrong > 0): Reward = -reconv_wrong / checked (Proportional Penalty)
    #    Else (Perfect): Reward = 1.0
    
    # Base reward tensor filled with -1.0 (Worst case default)
    base_reward = torch.full_like(checked_t, -1.0)
    
    # Mask where local logic is OK
    local_ok = (local_wrong_t == 0)
    
    # For local_ok samples, calculate reward based on reconv
    # If reconv_wrong_t == 0, reward is 1.0
    # If reconv_wrong_t > 0, reward is -reconv_wrong / checked
    
    denom = torch.clamp(checked_t, min=1.0)
    reconv_penalty = -reconv_wrong_t / denom
    
    # If local OK:
    #   if reconv OK (wrong_t == 0) -> 1.0
    #   else -> reconv_penalty
    
    reconv_ok = (reconv_wrong_t == 0)
    
    # Where local_ok is true:
    # reward = 1.0 if reconv_ok else reconv_penalty
    reward_if_local_ok = torch.where(reconv_ok, torch.ones_like(base_reward), reconv_penalty)
    
    # Apply to base_reward
    base_reward = torch.where(local_ok, reward_if_local_ok, base_reward)
    
    # Special case: Trivial samples (checked == 0) -> 1.0
    base_reward = torch.where(trivial, torch.ones_like(base_reward), base_reward)
    
    reward = base_reward
    
    # Anchor reward shaping: encourage model to place the anchor value.
    if anchor_p is not None and anchor_l is not None and anchor_v is not None:
        idx = torch.arange(B, device=logits.device)
        present = (anchor_p >= 0) & (anchor_l >= 0)
        if bool(present.any()):
            pred_vals = torch.zeros(B, dtype=torch.long, device=logits.device)
            pred_vals[present] = actions[idx[present], anchor_p[present], anchor_l[present]]
            matches = (pred_vals == anchor_v) & present
            anchor_signal = torch.zeros(B, dtype=torch.float32, device=logits.device)
            anchor_signal[matches] = 1.0
            anchor_signal[present & (~matches)] = -1.0
            reward = reward + float(anchor_alpha) * anchor_signal
            
    # Keep reward in a reasonable range
    reward = torch.clamp(reward, min=-1.0, max=1.0)

    # Optional per-batch normalization to stabilize gradients
    if normalize_reward:
        mean_r = reward.mean()
        std_r = reward.std().clamp(min=1e-6)
        reward = (reward - mean_r) / std_r

    # Detach reward for REINFORCE
    reward = reward.detach()

    # Policy gradient loss: -mean(reward * mean logp over valid positions)
    logp_sum = (logp * mask_valid).sum(dim=(1, 2))  # [B]
    count = torch.clamp(mask_valid.sum(dim=(1, 2)).float(), min=1.0)
    per_sample_loss = -(reward * (logp_sum / count))  # [B]
    loss = per_sample_loss.mean()

    # Entropy regularization to encourage exploration
    if entropy_beta > 0.0:
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)  # [B,P,L]
        ent_mean = (ent * mask_valid.float()).sum() / torch.clamp(mask_valid.float().sum(), min=1.0)
        loss = loss - float(entropy_beta) * ent_mean

    # Auxiliary Supervised Loss for Anchors
    if anchor_p is not None and anchor_l is not None and anchor_v is not None:
        valid_anchors = (anchor_p >= 0) & (anchor_l >= 0)
        if valid_anchors.any():
            b_idx = torch.arange(B, device=logits.device)[valid_anchors]
            p_idx = anchor_p[valid_anchors]
            l_idx = anchor_l[valid_anchors]
            targets = anchor_v[valid_anchors].long().clamp(0, 1) # Ensure 0 or 1
            
            pred_logits = logits[b_idx, p_idx, l_idx] # [N, 2]
            sup_loss = F.cross_entropy(pred_logits, targets)
            loss = loss + 1.0 * sup_loss

    # Auxiliary Consistency Loss for Reconvergent Paths
    # Penalize variance of logits at the reconvergence point across paths
    cons_loss_list = []
    for b in range(B):
        mask_b = mask_valid[b]
        last_idx = mask_b.sum(dim=-1) - 1
        present = last_idx >= 0
        if present.sum() >= 2:
            last_idx_clamped = last_idx.clamp(min=0)
            arange_p = torch.arange(mask_b.size(0), device=logits.device)
            # Gather logits at the end of each valid path
            val_logits = logits[b, arange_p, last_idx_clamped][present] # [K, 2]
            # specific logic: variance from mean
            mean_logits = val_logits.mean(dim=0, keepdim=True)
            dist = ((val_logits - mean_logits)**2).mean()
            cons_loss_list.append(dist)
    
    if cons_loss_list:
        loss = loss + 0.5 * torch.stack(cons_loss_list).mean()
        
    # Soft Edge Consistency Loss
    if soft_edge_loss_list and soft_edge_lambda > 0:
        edge_loss = torch.stack(soft_edge_loss_list).mean()
        loss = loss + float(soft_edge_lambda) * edge_loss

    # Report the unnormalized average reward-like signal for monitoring:
    # recompute a view without normalization (no-grad) for logging
    with torch.no_grad():
        avg_reward = float(base_reward.mean().item())
        
    valid_rate = float(valid.float().mean().item())
    trivial_rate = float(trivial.float().mean().item())
    edge_acc = float((edge_total_sum - edge_wrong_sum) / max(1.0, edge_total_sum))
    
    return loss, avg_reward, valid_rate, edge_acc, trivial_rate


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, device: torch.device, cfg: TrainConfig) -> Tuple[float, float, float, float, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    total_reward = 0.0
    total_valid = 0.0
    total_edge_acc = 0.0
    total_trivial = 0.0
    start_time = time.time()
    # Target batch count for ETA (respect caps if provided)
    try:
        loader_len = len(loader)
    except Exception:
        loader_len = -1
    target_batches = loader_len if loader_len > 0 else None
    if cfg.max_train_batches > 0:
        target_batches = cfg.max_train_batches if target_batches is None else min(cfg.max_train_batches, target_batches)

    for batch_idx, batch in enumerate(loader):
        paths = batch['paths_emb']  # [B, P, L, D]
        masks = batch['attn_mask']  # [B, P, L]
        node_ids = batch['node_ids']  # [B, P, L]
        files = batch['files']        # list[str]

        # Resolve paths if bench_dir is provided
        if cfg.bench_dir:
            resolved_files = []
            for f in files:
                if os.path.exists(f):
                    resolved_files.append(f)
                else:
                    joined = os.path.join(cfg.bench_dir, f)
                    if os.path.exists(joined):
                        resolved_files.append(joined)
                    else:
                        resolved_files.append(os.path.join(cfg.bench_dir, os.path.basename(f)))
            files = resolved_files
        # Move to device once per batch for efficiency
        if device.type == 'cuda':
            paths = paths.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            node_ids = node_ids.to(device, non_blocking=True)
        if cfg.verbose and batch_idx == 0:
            print(f"[device] batch0 paths_emb device={paths.device}, masks device={masks.device}")

        # Prefer dataset-provided anchors if available; else generate procedurally
        if cfg.anchor_hint and 'anchor_p' in batch and 'anchor_l' in batch and 'anchor_v' in batch:
            anchor_p = batch['anchor_p']
            anchor_l = batch['anchor_l']
            anchor_v = batch['anchor_v']
            if device.type == 'cuda':
                anchor_p = anchor_p.to(device, non_blocking=True)
                anchor_l = anchor_l.to(device, non_blocking=True)
                anchor_v = anchor_v.to(device, non_blocking=True)
            # Embeddings already contain the anchor one-hot from dataset (if enabled)
        elif cfg.anchor_hint:
            # Generate on CPU to avoid syncs with Python-side circuit parsing
            anchor_p_cpu, anchor_l_cpu, anchor_v_cpu = _generate_anchor(node_ids.detach().cpu(), masks.detach().cpu(), files, cfg.prefer_value)
            anchor_p = anchor_p_cpu.to(device)
            anchor_l = anchor_l_cpu.to(device)
            anchor_v = anchor_v_cpu.to(device)
            paths = _inject_anchor_into_embeddings(paths, anchor_p, anchor_l, anchor_v, enable=cfg.add_logic_value)
        else:
            anchor_p = anchor_l = anchor_v = None  # type: ignore
        
        # Resolve gate types if needed (for embedding or loss)
        # Note: We need them for loss anyway now.
        gtypes = resolve_gate_types(node_ids, files, device)



        optim.zero_grad(set_to_none=True)
        
        # AMP context
        with torch.amp.autocast('cuda', enabled=cfg.amp):
            # Pass gate types to model if configured
            logits = model(paths, masks, gate_types=gtypes if cfg.use_gate_type_embedding else None)  # [B, P, L, 2]

            loss, avg_reward, valid_rate, batch_edge_acc, batch_trivial = policy_loss_and_metrics(
                logits, node_ids, masks, files, gtypes,
                anchor_p=anchor_p, anchor_l=anchor_l, anchor_v=anchor_v,
                anchor_alpha=cfg.anchor_reward_alpha,
                normalize_reward=cfg.normalize_reward,
                entropy_beta=cfg.entropy_beta,
                soft_edge_lambda=cfg.soft_edge_lambda,
            )
        
        # Backprop (with scaler if AMP)
        scaler.scale(loss).backward(retain_graph=True)
        scaler.step(optim)
        scaler.update()

        total_loss += float(loss.item())
        total_reward += float(avg_reward)
        total_valid += float(valid_rate)
        total_edge_acc += float(batch_edge_acc)
        total_trivial += float(batch_trivial)
        total_batches += 1
        # Periodic progress log
        if cfg.verbose and cfg.log_interval > 0 and (batch_idx + 1) % cfg.log_interval == 0:
            try:
                total_len = len(loader)
            except Exception:
                total_len = -1
            elapsed = time.time() - start_time
            bdone = batch_idx + 1
            it_per_s = bdone / max(1e-6, elapsed)
            # Prefer capped total for ETA if available
            total_for_eta = target_batches if target_batches is not None else (total_len if total_len > 0 else None)
            if total_for_eta is not None:
                remaining = max(0, total_for_eta - bdone)
                eta = _format_seconds(remaining / max(1e-6, it_per_s))
                denom_str = f"/{total_for_eta}"
                eta_str = f" eta={eta}"
            else:
                denom_str = ""
                eta_str = ""
            dbg = _debug_metrics_from_logits(
                logits, node_ids, masks, files,
                anchor_p=anchor_p if isinstance(anchor_p, torch.Tensor) else None,
                anchor_l=anchor_l if isinstance(anchor_l, torch.Tensor) else None,
                anchor_v=anchor_v if isinstance(anchor_v, torch.Tensor) else None,
            )
            print(
                f"[train] batch {bdone}{denom_str} avg_loss={total_loss / max(1, total_batches):.4f} "
                f"acc={total_valid / max(1, total_batches):.4f} edge_acc={dbg['edge_acc']:.3f} "
                f"reconv={dbg['reconv_match_rate']:.3f} anchor={dbg['anchor_match_rate']:.3f} "
                f"edges/sample={dbg['edges_per_sample']:.1f} speed={it_per_s:.2f} it/s{eta_str}"
            )

        # Optional limit on number of batches per epoch
        if cfg.max_train_batches > 0 and (batch_idx + 1) >= cfg.max_train_batches:
            break

    avg_loss = total_loss / max(1, total_batches)
    avg_reward = total_reward / max(1, total_batches)
    valid_rate = total_valid / max(1, total_batches)
    avg_edge_acc = total_edge_acc / max(1, total_batches)
    avg_trivial = total_trivial / max(1, total_batches)
    # Return avg loss, avg reward, accuracy (valid rate), edge accuracy, trivial rate
    return avg_loss, avg_reward, valid_rate, avg_edge_acc, avg_trivial


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, cfg: TrainConfig) -> Tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_reward = 0.0
    total_valid = 0.0
    total_edge_acc = 0.0
    total_trivial = 0.0
    total_batches = 0
    start_time = time.time()
    # Target batch count for ETA (respect caps if provided)
    try:
        loader_len = len(loader)
    except Exception:
        loader_len = -1
    target_batches = loader_len if loader_len > 0 else None
    if cfg.max_val_batches > 0:
        target_batches = cfg.max_val_batches if target_batches is None else min(cfg.max_val_batches, target_batches)

    for batch_idx, batch in enumerate(loader):
        paths = batch['paths_emb']
        masks = batch['attn_mask']
        node_ids = batch['node_ids']
        files = batch['files']
        
        # Resolve paths if bench_dir is provided
        if cfg.bench_dir:
            resolved_files = []
            for f in files:
                if os.path.exists(f):
                    resolved_files.append(f)
                else:
                    joined = os.path.join(cfg.bench_dir, f)
                    if os.path.exists(joined):
                        resolved_files.append(joined)
                    else:
                        resolved_files.append(os.path.join(cfg.bench_dir, os.path.basename(f)))
            files = resolved_files
        if device.type == 'cuda':
            paths = paths.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            node_ids = node_ids.to(device, non_blocking=True)

        # Prefer dataset-provided anchors if available; else generate procedurally
        if cfg.anchor_hint and 'anchor_p' in batch and 'anchor_l' in batch and 'anchor_v' in batch:
            anchor_p = batch['anchor_p']
            anchor_l = batch['anchor_l']
            anchor_v = batch['anchor_v']
            if device.type == 'cuda':
                anchor_p = anchor_p.to(device, non_blocking=True)
                anchor_l = anchor_l.to(device, non_blocking=True)
                anchor_v = anchor_v.to(device, non_blocking=True)
        elif cfg.anchor_hint:
            anchor_p_cpu, anchor_l_cpu, anchor_v_cpu = _generate_anchor(node_ids.detach().cpu(), masks.detach().cpu(), files, cfg.prefer_value)
            anchor_p = anchor_p_cpu.to(device)
            anchor_l = anchor_l_cpu.to(device)
            anchor_v = anchor_v_cpu.to(device)
            paths = _inject_anchor_into_embeddings(paths, anchor_p, anchor_l, anchor_v, enable=cfg.add_logic_value)
        else:
            anchor_p = anchor_l = anchor_v = None  # type: ignore

        gtypes = resolve_gate_types(node_ids, files, device)

        with torch.amp.autocast('cuda', enabled=cfg.amp):
            logits = model(paths, masks, gate_types=gtypes if cfg.use_gate_type_embedding else None)
            loss, avg_reward, valid_rate, batch_edge_acc, batch_trivial = policy_loss_and_metrics(
                logits, node_ids, masks, files, gtypes,
                anchor_p=anchor_p, anchor_l=anchor_l, anchor_v=anchor_v,
                anchor_alpha=cfg.anchor_reward_alpha,
                normalize_reward=cfg.normalize_reward,
                entropy_beta=cfg.entropy_beta,
                soft_edge_lambda=cfg.soft_edge_lambda,
            )
        total_loss += float(loss.item())
        total_reward += float(avg_reward)
        total_valid += float(valid_rate)
        total_edge_acc += float(batch_edge_acc)
        total_trivial += float(batch_trivial)
        total_batches += 1
        # Periodic progress log
        if cfg.verbose and cfg.log_interval > 0 and (batch_idx + 1) % cfg.log_interval == 0:
            try:
                total_len = len(loader)
            except Exception:
                total_len = -1
            elapsed = time.time() - start_time
            bdone = batch_idx + 1
            it_per_s = bdone / max(1e-6, elapsed)
            total_for_eta = target_batches if target_batches is not None else (total_len if total_len > 0 else None)
            if total_for_eta is not None:
                remaining = max(0, total_for_eta - bdone)
                eta = _format_seconds(remaining / max(1e-6, it_per_s))
                denom_str = f"/{total_for_eta}"
                eta_str = f" eta={eta}"
            else:
                denom_str = ""
                eta_str = ""
            dbg = _debug_metrics_from_logits(
                logits, node_ids, masks, files,
                anchor_p=anchor_p if isinstance(anchor_p, torch.Tensor) else None,
                anchor_l=anchor_l if isinstance(anchor_l, torch.Tensor) else None,
                anchor_v=anchor_v if isinstance(anchor_v, torch.Tensor) else None,
            )
            print(
                f"[val] batch {bdone}{denom_str} avg_loss={total_loss / max(1, total_batches):.4f} "
                f"acc={total_valid / max(1, total_batches):.4f} edge_acc={dbg['edge_acc']:.3f} "
                f"reconv={dbg['reconv_match_rate']:.3f} anchor={dbg['anchor_match_rate']:.3f} "
                f"edges/sample={dbg['edges_per_sample']:.1f} speed={it_per_s:.2f} it/s{eta_str}"
            )

        # Optional limit on number of batches per evaluation
        if cfg.max_val_batches > 0 and (batch_idx + 1) >= cfg.max_val_batches:
            break

    avg_loss = total_loss / max(1, total_batches)
    avg_reward = total_reward / max(1, total_batches)
    valid_rate = total_valid / max(1, total_batches)
    avg_edge_acc = total_edge_acc / max(1, total_batches)
    avg_trivial = total_trivial / max(1, total_batches)
    return avg_loss, avg_reward, valid_rate, avg_edge_acc, avg_trivial


def save_checkpoint(path: str, model: nn.Module, cfg: TrainConfig, best: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'state_dict': (model.module if hasattr(model, 'module') else model).state_dict(),
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
        add_logic_value=getattr(args, 'add_logic_value', True),
        max_train_batches=getattr(args, 'max_train_batches', 0),
        max_val_batches=getattr(args, 'max_val_batches', 0),
        log_interval=getattr(args, 'log_interval', 500),
        dataset_anchor_hint=getattr(args, 'dataset_anchor_hint', True),
        nhead=getattr(args, 'nhead', 4),
        num_encoder_layers=getattr(args, 'enc_layers', 1),
        num_interaction_layers=getattr(args, 'int_layers', 1),
        dim_feedforward=getattr(args, 'ffn_dim', 512),
        model_dim=getattr(args, 'model_dim', 512),
        num_workers=getattr(args, 'num_workers', 8),
        pin_memory=getattr(args, 'pin_memory', True),
        bench_dir=getattr(args, 'bench_dir', ""),
        amp=getattr(args, 'amp', False),
        include_hard_negatives=getattr(args, 'include_hard_negatives', False),
        soft_edge_lambda=getattr(args, 'soft_edge_lambda', 1.0),
        max_len=getattr(args, 'max_len', 0),
        entropy_beta=getattr(args, 'entropy_beta', 0.0),
    )
    
    # Handle checkpoint-dir alias
    if getattr(args, 'checkpoint_dir', None):
        cfg.output = args.checkpoint_dir

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if cfg.verbose:
        if device.type == 'cuda':
            try:
                dev_name = torch.cuda.get_device_name(device)
            except Exception:
                dev_name = 'CUDA device'
            print(f"Using device: {device} ({dev_name})")
        else:
            print("Using device: cpu (CUDA not available)")

    train_loader, val_loader = make_dataloaders(cfg, device)

    # Infer actual embedding dimension from a real batch to avoid mismatches
    # with processed shards and logic-value features.
    probe_batch = next(iter(train_loader))
    observed_dim = int(probe_batch['paths_emb'].shape[-1])
    nhead = cfg.nhead
    if cfg.verbose:
        print(f"Observed embedding dimension from batch: {observed_dim}")
        print(f"Number of attention heads: {nhead}")
    # Choose internal model dimension and ensure divisibility by nhead
    model_dim = int(cfg.model_dim)
    if model_dim % nhead != 0:
        new_dim = ((model_dim // nhead) + 1) * nhead
        if cfg.verbose:
            print(f"Adjusting model_dim from {model_dim} to {new_dim} to be divisible by nhead={nhead}")
        model_dim = new_dim
    
    model = MultiPathTransformer(
        input_dim=observed_dim,
        model_dim=model_dim,
        nhead=nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_interaction_layers=cfg.num_interaction_layers,
        dim_feedforward=cfg.dim_feedforward,
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)

    best_val = float('inf')
    if cfg.verbose:
        nb_train = len(train_loader) if hasattr(train_loader, '__len__') else 0
        nb_val = len(val_loader) if hasattr(val_loader, '__len__') else 0
        print(f"Starting training: train_batches={nb_train}, val_batches={nb_val}, batch_size={cfg.batch_size}")
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_reward, tr_acc, tr_edge, tr_triv = train_one_epoch(model, train_loader, optim, scaler, device, cfg)
        va_loss, va_reward, va_acc, va_edge, va_triv = evaluate(model, val_loader, device, cfg)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f} avg_reward={tr_reward:.4f} acc={tr_acc:.4f} edge_acc={tr_edge:.4f} triv={tr_triv:.4f} | "
            f"val_loss={va_loss:.4f} avg_reward={va_reward:.4f} acc={va_acc:.4f} edge_acc={va_edge:.4f} triv={va_triv:.4f}"
        )

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
    t.add_argument('--checkpoint-dir', type=str, help='Output checkpoint directory (alias for --output)')
    t.add_argument('--bench-dir', type=str, default='', help='Base directory for benchmark files')
    t.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision (AMP)')
    t.add_argument('--include-hard-negatives', action='store_true', help='Include hard negatives (currently ignored)')
    t.add_argument('--epochs', type=int, default=10)
    t.add_argument('--batch-size', type=int, default=128)
    t.add_argument('--verbose', action='store_true')
    # Model capacity
    t.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    t.add_argument('--enc-layers', type=int, default=1, help='Number of shared path encoder layers')
    t.add_argument('--int-layers', type=int, default=1, help='Number of path interaction layers')
    t.add_argument('--ffn-dim', type=int, default=512, help='Transformer feedforward dimension')
    t.add_argument('--model-dim', type=int, default=512, help='Internal Transformer model dimension (must be divisible by nhead)')
    t.add_argument('--add-logic-value', action='store_true', default=True, 
                   help='Add logic value (0/1/X) as one-hot feature to embeddings')
    t.add_argument('--no-logic-value', dest='add_logic_value', action='store_false',
                   help='Disable logic value feature')
    t.add_argument('--dataset-anchor-hint', action='store_true', default=True,
                   help='Generate anchor hint inside the dataset loader and include in batch')
    t.add_argument('--no-dataset-anchor-hint', dest='dataset_anchor_hint', action='store_false',
                   help='Disable dataset-level anchor generation')
    # DataLoader performance
    t.add_argument('--num-workers', type=int, default=4, help='DataLoader workers when CUDA is available')
    t.add_argument('--pin-memory', action='store_true', default=True, help='Enable DataLoader pin_memory for CUDA')
    t.add_argument('--no-pin-memory', dest='pin_memory', action='store_false', help='Disable DataLoader pin_memory')
    t.add_argument('--max-train-batches', type=int, default=0,
                   help='Limit number of training batches per epoch (0 = no limit)')
    t.add_argument('--max-val-batches', type=int, default=0,
                   help='Limit number of validation batches per eval (0 = no limit)')
    t.add_argument('--log-interval', type=int, default=500,
                   help='Batches between progress logs when --verbose is set')
    t.add_argument('--soft-edge-lambda', type=float, default=1.0, 
                   help='Weight for soft edge consistency loss')
    t.add_argument('--max-len', type=int, default=0, help='Filter dataset for max path length (Curriculum Learning)')
    t.add_argument('--entropy-beta', type=float, default=0.0, help='Entropy regularization weight (negative to minimize)')

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