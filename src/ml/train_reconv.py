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
from src.util.struct import GateType, LogicValue
from src.atpg.logic_sim_three import logic_sim, reset_gates


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
    
    # Phase 6: Constrained Training
    constrained_curriculum: bool = False
    max_constraint_prob: float = 0.5
    enforce_constraints: bool = True


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each batch item, randomly pick one valid path and its last gate,
    then assign a compatible anchor value based on gate type.
    
    Verifies if the pick is SOLVABLE or UNSAT using PathConsistencySolver.

    Returns:
      - anchor_p: LongTensor [B] with path index, or -1 if none
      - anchor_l: LongTensor [B] with last index along that path, or -1 if none
      - anchor_v: LongTensor [B] with value in {0,1} (0 if none)
      - solvability: LongTensor [B] with 0 (SAT) or 1 (UNSAT)
    """
    device = node_ids.device
    B, P, L = node_ids.shape[:3]
    anchor_p = torch.full((B,), -1, dtype=torch.long, device=device)
    anchor_l = torch.full((B,), -1, dtype=torch.long, device=device)
    anchor_v = torch.zeros((B,), dtype=torch.long, device=device)
    solvability = torch.zeros((B,), dtype=torch.long, device=device)

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
        # Randomly pick one candidate
        pick_i = torch.randint(low=0, high=len(candidates), size=(1,), device=device).item()
        pick_idx = int(pick_i)
        p_sel, l_sel, g_sel = candidates[pick_idx]
        v_sel = _compatible_anchor_value(g_sel, prefer_value)
        
        # Verify Solvability
        from src.atpg.reconv_podem import PathConsistencySolver
        solver = PathConsistencySolver(circuit)
        # We need to construct pair_info for the solver
        # Solver expects 'paths' which is list of list of node IDs.
        # We can extract them from node_ids for the current sample.
        current_paths = []
        for p_idx in range(P):
            path_b = node_ids[b, p_idx].tolist()
            # filter zeros
            path_b = [int(nid) for nid in path_b if nid > 0]
            if path_b:
                 current_paths.append(path_b)
        
        # Start node is usually the first node of all paths (assuming they all start at same stem)
        # But let's be safe and check if they do.
        stems = set(p[0] for p in current_paths)
        if not stems:
             continue
        start_node = list(stems)[0]
        reconv_node = int(node_ids[b, p_sel, l_sel].item())
        
        pair_info = {
            'start': start_node,
            'reconv': reconv_node,
            'paths': current_paths
        }
        
        # Check if target v_sel is possible
        from src.util.struct import LogicValue
        target_lv = LogicValue.ZERO if v_sel == 0 else LogicValue.ONE
        assignment = solver.solve(pair_info, target_lv)
        
        anchor_p[b] = int(p_sel)
        anchor_l[b] = int(l_sel)
        anchor_v[b] = int(v_sel)
        solvability[b] = 0 if assignment is not None else 1

    return anchor_p, anchor_l, anchor_v, solvability


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
    solvability_logits: "Optional[torch.Tensor]" = None,
    solvability_labels: "Optional[torch.Tensor]" = None,
) -> "Dict[str, float]":
    """Compute diagnostic metrics using greedy actions (argmax).

    Returns a dict with keys:
      - edge_acc: fraction of valid edges satisfying local constraints (0..1)
      - reconv_match_rate: fraction of samples with all-last-values-equal among those with >=2 paths
      - anchor_match_rate: fraction of present anchors where predicted value matches (if anchors provided)
      - solvability_acc: overall accuracy of SAT/UNSAT prediction
      - false_unsat_rate: rate of predicting UNSAT when SAT exists
      - true_unsat_rate: rate of correctly identifying UNSAT cases
    """
    device = logits.device
    B, P, L, _ = logits.shape
    actions = torch.argmax(logits, dim=-1)  # [B,P,L]

    # Edge checks (vectorized per batch)
    valid_edges = mask_valid[:, :, 1:] & mask_valid[:, :, :-1]  # [B,P,L-1]
    prev_vals_all = actions[:, :, :-1]
    cur_vals_all = actions[:, :, 1:]

    total_edges = valid_edges.sum(dtype=torch.float32).item()

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

    # Anchor match rate (only relevant for SAT cases)
    anchor_match_rate = 0.0
    if anchor_p is not None and anchor_l is not None and anchor_v is not None:
        present = (anchor_p >= 0) & (anchor_l >= 0)
        # Filter for SAT cases if labels provided
        if solvability_labels is not None:
             present = present & (solvability_labels == 0)
             
        if bool(present.any()):
            idx = torch.arange(B, device=device)[present]
            pred_vals = actions[idx, anchor_p[present], anchor_l[present]]
            matches = (pred_vals == anchor_v[present]).float().mean().item()
            anchor_match_rate = float(matches)

    # Solvability Metrics
    solvability_acc = 0.0
    false_unsat_rate = 0.0
    true_unsat_rate = 0.0
    if solvability_logits is not None and solvability_labels is not None:
        pred_solv = torch.argmax(solvability_logits, dim=-1)
        correct = (pred_solv == solvability_labels).float()
        solvability_acc = correct.mean().item()
        
        sat_mask = (solvability_labels == 0)
        unsat_mask = (solvability_labels == 1)
        
        if bool(sat_mask.any()):
            # False UNSAT = Predicted 1 (UNSAT) when actually 0 (SAT)
            false_unsat_rate = (pred_solv[sat_mask] == 1).float().mean().item()
            
        if bool(unsat_mask.any()):
            # True UNSAT = Predicted 1 when actually 1
            true_unsat_rate = (pred_solv[unsat_mask] == 1).float().mean().item()

    return {
        'edge_acc': edge_acc,
        'reconv_match_rate': reconv_match_rate,
        'anchor_match_rate': anchor_match_rate,
        'solvability_acc': solvability_acc,
        'false_unsat_rate': false_unsat_rate,
        'true_unsat_rate': true_unsat_rate,
        'edges_per_sample': float(total_edges / max(1, B)),
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


def generate_constraints(
    node_ids: torch.Tensor, 
    files: List[str], 
    prob: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate ground-truth constraints for a batch using valid logic simulation.
    
    Args:
        node_ids: [B, P, L] Tensor of node IDs
        files: List of B file paths
        prob: Probability of masking a node as a constraint
        
    Returns:
        constraint_mask: [B, P, L] Bool tensor (True = constrained)
        constraint_vals: [B, P, L] Long tensor (0 or 1) - valid values
    """
    B, P, L = node_ids.shape
    device = node_ids.device
    
    constraint_mask = torch.zeros((B, P, L), dtype=torch.bool, device=device)
    constraint_vals = torch.zeros((B, P, L), dtype=torch.long, device=device)
    
    if prob <= 0.0:
        return constraint_mask, constraint_vals

    for b in range(B):
        circuit = _load_circuit(files[b])
        # Reset and Simulate with random inputs
        reset_gates(circuit, len(circuit)-1)
        
        # Assign random values to PIs
        for gate in circuit:
            if gate.type == GateType.INPT:
                gate.val = int(torch.randint(0, 2, (1,)).item())
        
        # Propagate
        logic_sim(circuit, len(circuit)-1)
        
        # Read values for path nodes
        # We perform a coin flip for each node to decide if it's constrained
        # Faster to do it computationally on CPU/numpy then moving to Tensor?
        # Let's iterate.
        
        # Optimize: get all unique node IDs in this sample that are valid (>0)
        # But we need per-position values.
        
        # Batch-level logic for efficiency not fully possible here due to circuit obj structure
        nid_b = node_ids[b] # [P, L]
        
        # Read valid values from circuit
        # We need to handle the case where node_ids includes padding (0)
        # valid nodes > 0.
        
        # Flatten P, L
        flat_nid = nid_b.view(-1).cpu().numpy()
        flat_vals = []
        for nid in flat_nid:
            if nid > 0 and nid < len(circuit):
                val = circuit[nid].val
                # logic_sim uses {0, 1, 2(X)}. We should only constrain if 0 or 1.
                if val == 2: # LogicValue.XD
                    flat_vals.append(-1) 
                elif val > 2: # D/D_bar etc? logic_sim_three usually produces clean 0/1/X for logic_sim
                     # logic_sim uses arrays like AND_GATE which output 0,1,2.
                     # Assuming standard simulation.
                     flat_vals.append(-1)
                else:
                     flat_vals.append(val)
            else:
                flat_vals.append(-1)
        
        vals_t = torch.tensor(flat_vals, dtype=torch.long, device=device).view(P, L)
        
        # Mask generation: Only where we have a valid value (0 or 1)
        valid_val_mask = vals_t >= 0
        
        # Random mask based on prob
        rand_probs = torch.rand((P, L), device=device)
        mask_b = (rand_probs < prob) & valid_val_mask
        
        constraint_mask[b] = mask_b
        constraint_vals[b] = vals_t.clamp(0, 1) # -1 becomes 0 but masked out

    return constraint_mask, constraint_vals

def policy_loss_and_metrics(
    logits: torch.Tensor,
    node_ids: torch.Tensor,
    mask_valid: torch.Tensor,
    files: list[str],
    gate_types: torch.Tensor,
    # Constraints
    constraint_mask: torch.Tensor | None = None,
    constraint_vals: torch.Tensor | None = None,
    # Anchors
    anchor_p: torch.Tensor | None = None,
    anchor_l: torch.Tensor | None = None,
    anchor_v: torch.Tensor | None = None,
    solvability_logits: torch.Tensor | None = None,
    solvability_labels: torch.Tensor | None = None,
    anchor_alpha: float = 0.1,
    normalize_reward: bool = True,
    entropy_beta: float = 0.0,
    soft_edge_lambda: float = 1.0,
) -> tuple[torch.Tensor, float, float, float, float]:
    """Compute REINFORCE loss with LUT-inspired constraints and reconv consistency.

    Returns: (loss, avg_reward, valid_rate, edge_acc, constraint_violation_rate)
    """

    B, P, L, C = logits.shape
    # Sample actions
    actions = torch.distributions.Categorical(logits=logits).sample().detach()  # [B, P, L]
    
    # constraint_metrics
    constraint_loss = torch.tensor(0.0, device=logits.device)
    constraint_violation_rate = 0.0
    
    # Enforce constraints if provided
    if constraint_mask is not None and constraint_vals is not None: 
        if constraint_mask.any():
            valid_constraints = constraint_mask.view(-1)
            flat_logits = logits.view(-1, 2)
            flat_targets = constraint_vals.view(-1)
            
            # CE Loss on constrained nodes
            c_loss = F.cross_entropy(flat_logits[valid_constraints], flat_targets[valid_constraints])
            constraint_loss = c_loss * 500.0 
            
            # Metric: Violation rate
            preds = actions[constraint_mask]
            targets = constraint_vals[constraint_mask]
            violations = (preds != targets).float().sum()
            total_c = targets.numel()
            constraint_violation_rate = (violations / max(1, total_c)).item()
            
            actions[constraint_mask] = constraint_vals[constraint_mask]

    # Solvability Loss
    solvability_loss = torch.tensor(0.0, device=logits.device)
    if solvability_logits is not None and solvability_labels is not None:
        # Penalize False UNSAT heavily (User: "predicting all instances as unsolvable ... easy to approach")
        # False UNSAT = Label 0 (SAT), Pred 1 (UNSAT).
        # We can use weighted cross entropy.
        weights = torch.tensor([10.0, 1.0], device=logits.device) # Heavy weight on Label 0 to avoid predicting 1
        solvability_loss = F.cross_entropy(solvability_logits, solvability_labels, weight=weights) * 100.0

    # RL Reward Logic
    # If UNSAT: reward = 1.0 if correct solvability pred, else -1.0
    # If SAT: reward based on path consistency as before
    
    sat_reward_mask = torch.ones(B, dtype=torch.bool, device=logits.device)
    unsat_reward = torch.zeros(B, dtype=torch.float32, device=logits.device)
    
    if solvability_labels is not None:
        unsat_mask = (solvability_labels == 1)
        sat_mask = (solvability_labels == 0)
        sat_reward_mask = sat_mask
        
        if unsat_mask.any():
            pred_solv = torch.argmax(solvability_logits, dim=-1)
            correct = (pred_solv[unsat_mask] == 1).float()
            unsat_reward[unsat_mask] = torch.where(correct == 1, torch.ones_like(correct), torch.full_like(correct, -1.0))

    # Compute log-probabilities directly from logits to avoid distribution caching quirks.
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, P, L, C]
    logp = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [B, P, L]
    
    # Get probability of 1 (Logic-1) for soft constraints
    probs_1 = torch.softmax(logits, dim=-1)[..., 1]  # [B, P, L]

    # Initialize loss accumulator
    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Vectorized Logic Consistency
    # ----------------------------
    valid_edges = mask_valid[:, :, 1:] & mask_valid[:, :, :-1]  # [B, P, L-1]
    
    prev_vals = actions[:, :, :-1]
    cur_vals = actions[:, :, 1:]
    gt_cur = gate_types[:, :, 1:] # [B, P, L-1]
    
    edge_ok = torch.ones_like(prev_vals, dtype=torch.bool)
    
    # NOT: cur == 1 - prev
    m = gt_cur == GateType.NOT
    edge_ok[m] &= (cur_vals[m] == (1 - prev_vals[m]))
    
    # BUFF: cur == prev
    m = gt_cur == GateType.BUFF
    edge_ok[m] &= (cur_vals[m] == prev_vals[m])
    
    # AND: cur <= prev
    m = gt_cur == GateType.AND
    edge_ok[m] &= (cur_vals[m] <= prev_vals[m])
    
    # NAND: cur >= 1 - prev
    m = gt_cur == GateType.NAND
    edge_ok[m] &= (cur_vals[m] >= (1 - prev_vals[m]))
    
    # OR: cur >= prev
    m = gt_cur == GateType.OR
    edge_ok[m] &= (cur_vals[m] >= prev_vals[m])
    
    # NOR: cur <= 1 - prev
    m = gt_cur == GateType.NOR
    edge_ok[m] &= (cur_vals[m] <= (1 - prev_vals[m]))

    # Gather Edge Errors
    wrong_edges = (~edge_ok) & valid_edges # [B, P, L-1]
    
    # Sample-level stats
    # local_wrong [B]: number of wrong edges per sample
    local_wrong = wrong_edges.sum(dim=(1, 2)) # [B]
    checked = valid_edges.sum(dim=(1, 2)) # [B]
    
    edge_wrong_sum = local_wrong.sum().item()
    edge_total_sum = checked.sum().item()

    # Vectorized Reconvergence Failures
    # -------------------------------
    path_len = mask_valid.long().sum(dim=-1) # [B, P]
    last_idx = (path_len - 1).clamp(min=0)
    
    # Gather last values
    last_idx_exp = last_idx.unsqueeze(-1) # [B, P, 1]
    last_vals = actions.gather(2, last_idx_exp).squeeze(-1) # [B, P]
    
    path_valid_mask = (path_len > 0) # [B, P]
    
    # Min/Max across valid paths
    neg_inf = -999.0
    pos_inf = 999.0
    
    lv_float = last_vals.float()
    vm_float = path_valid_mask.float()
    
    # Max of valid: invalid -> -inf
    max_v = (lv_float * vm_float + neg_inf * (1 - vm_float)).max(dim=-1).values
    # Min of valid: invalid -> +inf
    min_v = (lv_float * vm_float + pos_inf * (1 - vm_float)).min(dim=-1).values
    
    # Failure if min < max
    has_valid_paths = (path_valid_mask.sum(dim=-1) > 0)
    reconv_fail_mask = (min_v < max_v) & has_valid_paths
    
    reconv_wrong = reconv_fail_mask.float() # [B]

    # Reconvergence Consistency Loss (MSE on Logits)
    # Gather logits at last_idx: [B, P, C]
    last_idx_logits = last_idx_exp.unsqueeze(-1).expand(actions.size(0), actions.size(1), 1, logits.shape[-1])
    last_logits = logits.gather(2, last_idx_logits).squeeze(2) # [B, P, C]
    
    # Mean logit per sample
    mask_exp = path_valid_mask.unsqueeze(-1) # [B, P, 1]
    count_paths = mask_exp.sum(dim=1).clamp(min=1.0) # [B, 1]
    sum_logits = (last_logits * mask_exp).sum(dim=1) # [B, C]
    mean_logits = sum_logits / count_paths # [B, C]
    
    # MSE
    diff = (last_logits - mean_logits.unsqueeze(1))
    mse_per_sample = ((diff ** 2) * mask_exp).sum(dim=(1,2)) / count_paths.squeeze(-1) # [B]
    
    loss = loss + 0.5 * mse_per_sample.mean()

    # Vectorized Soft Edge Loss
    # -------------------------
    prev_p = probs_1[:, :, :-1]
    cur_p = probs_1[:, :, 1:]
    
    viol = torch.zeros_like(prev_p)
    
    # Apply soft constraints
    # NOT: |cur - (1-prev)|^2
    m = gt_cur == GateType.NOT
    viol[m] = (cur_p[m] - (1.0 - prev_p[m])) ** 2
    
    # BUFF: |cur - prev|^2
    m = gt_cur == GateType.BUFF
    viol[m] = (cur_p[m] - prev_p[m]) ** 2
    
    # AND: cur <= prev => ReLU(cur - prev)^2
    m = gt_cur == GateType.AND
    viol[m] = F.relu(cur_p[m] - prev_p[m]) ** 2
    
    # NAND: cur >= 1-prev => ReLU((1-prev) - cur)^2
    m = gt_cur == GateType.NAND
    viol[m] = F.relu((1.0 - prev_p[m]) - cur_p[m]) ** 2
    
    # OR: cur >= prev => ReLU(prev - cur)^2
    m = gt_cur == GateType.OR
    viol[m] = F.relu(prev_p[m] - cur_p[m]) ** 2
    
    # NOR: cur <= 1-prev => ReLU(cur - (1-prev))^2
    m = gt_cur == GateType.NOR
    viol[m] = F.relu(cur_p[m] - (1.0 - prev_p[m])) ** 2
    
    # Weighting and masking
    edge_weights = torch.ones_like(gt_cur, dtype=torch.float32)
    edge_weights[gt_cur == GateType.NOT] = 12.0
    viol = viol * edge_weights * valid_edges.float()
    
    # Loss per sample
    valid_edge_counts = valid_edges.sum(dim=(1,2)).float().clamp(min=1.0)
    edge_loss_per_sample = viol.sum(dim=(1,2)) / valid_edge_counts
    
    if soft_edge_lambda > 0:
        loss = loss + float(soft_edge_lambda) * edge_loss_per_sample.mean()

    # Reward Logic
    # ----------------------------
    # Granular reward based on edge satisfaction to provide a smoother gradient.
    local_err = local_wrong.float()
    reconv_err = reconv_wrong.float()
    denom_edges = checked.float().clamp(min=1.0)
    
    # Calculate local consistency reward as a fraction in [0, 1] mapped to [-1, 1]
    local_reward_shaping = (1.0 - (local_err / denom_edges)) * 2.0 - 1.0
    
    # Base reward starts with local shaping
    base_reward = local_reward_shaping
    
    # Reconvergence Logic for SAT cases (Agreement = Bonus, Disagreement = Penalty)
    reconv_bonus = 0.5
    reconv_penalty = -1.0
    
    # If agree (err=0): reward += bonus
    # If disagree (err=1): reward = min(reward, penalty)
    
    # We construct 'sat_base_reward' which applies this logic
    sat_base_reward = torch.where(reconv_err == 0, 
                                  base_reward + reconv_bonus, 
                                  torch.min(base_reward, torch.tensor(reconv_penalty, device=logits.device)))

    # Neutral/positive for trivial samples (no edges)
    trivial = (checked == 0)
    base_reward = torch.where(trivial, torch.ones_like(base_reward), base_reward)
    sat_base_reward = torch.where(trivial, torch.ones_like(sat_base_reward), sat_base_reward)
    
    # Combined Reward for SAT/UNSAT
    reward = sat_base_reward.clone() # Default to SAT logic
    
    if solvability_labels is not None:
        # UNSAT samples (label=1): 
        # Use only local_reward_shaping (Edges Only) + Solvability Bonus.
        # Ignore reconvergence status (do not bonus or penalize).
        
        unsat_mask = (solvability_labels == 1)
        if unsat_mask.any():
             reward[unsat_mask] = local_reward_shaping[unsat_mask]
             
        # Add Solvability Prediction Bonus/Penalty
        pred_solv = torch.argmax(solvability_logits, dim=-1)
        correct_solv = (pred_solv == solvability_labels).float()
        
        # Reward +1 for correct solvability, -1 for incorrect
        solv_signal = torch.where(correct_solv == 1.0, torch.ones_like(correct_solv), torch.full_like(correct_solv, -1.0))
        
        # Weighted addition
        reward = reward + 0.5 * solv_signal

    # Anchor reward shaping (Alpha increased to 1.0 by user request)
    # We override the function argument default or just multiply here.
    # The argument `anchor_alpha` defaults to 0.1 in signature, but we can multiply.
    # Actually, better to change the call site, but for now I will boost it here.
    effective_anchor_alpha = 1.0 
    
    if anchor_p is not None and anchor_l is not None and anchor_v is not None:
        idx = torch.arange(logits.size(0), device=logits.device)
        present = (anchor_p >= 0) & (anchor_l >= 0)
        if solvability_labels is not None:
            present = present & (solvability_labels == 0)
            
        if bool(present.any()):
            pred_vals = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            p_idx = anchor_p[present]
            l_idx = anchor_l[present]
            pred_vals[present] = actions[idx[present], p_idx, l_idx]
            matches = (pred_vals == anchor_v) & present
            anchor_signal = torch.zeros(logits.size(0), dtype=torch.float32, device=logits.device)
            anchor_signal[matches] = 1.0
            anchor_signal[present & (~matches)] = -1.0
            reward = reward + float(effective_anchor_alpha) * anchor_signal
            
    reward = torch.clamp(reward, min=-1.0, max=1.0)

    # Optional per-batch normalization
    if normalize_reward:
        mean_r = reward.mean()
        std_r = reward.std().clamp(min=1e-6)
        reward = (reward - mean_r) / std_r

    # Detach reward for REINFORCE
    reward = reward.detach()

    # Policy gradient loss
    logp_sum = (logp * mask_valid).sum(dim=(1, 2))  # [B]
    count = torch.clamp(mask_valid.sum(dim=(1, 2)).float(), min=1.0)
    per_sample_loss = -(reward * (logp_sum / count))  # [B]
    loss = loss + per_sample_loss.mean()

    # Entropy regularization
    if entropy_beta > 0.0:
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)  # [B,P,L]
        ent_mean = (ent * mask_valid.float()).sum() / torch.clamp(mask_valid.float().sum(), min=1.0)
        loss = loss - float(entropy_beta) * ent_mean

    # Auxiliary Supervised Loss for Anchors (ONLY if SAT)
    if anchor_p is not None and anchor_l is not None and anchor_v is not None:
        valid_anchors = (anchor_p >= 0) & (anchor_l >= 0)
        if solvability_labels is not None:
            valid_anchors = valid_anchors & (solvability_labels == 0)
            
        if valid_anchors.any():
            b_idx = torch.arange(logits.size(0), device=logits.device)[valid_anchors]
            p_idx = anchor_p[valid_anchors]
            l_idx = anchor_l[valid_anchors]
            targets = anchor_v[valid_anchors].long().clamp(0, 1)
            pred_logits = logits[b_idx, p_idx, l_idx] # [N, 2]
            sup_loss = F.cross_entropy(pred_logits, targets)
            loss = loss + 1.0 * sup_loss
    
    # Add constraint loss and solvability loss
    loss = loss + constraint_loss + solvability_loss

    # Recompute metrics for logging
    with torch.no_grad():
        avg_reward = float(reward.mean().item()) # Using adjusted reward
        
        # Valid = local OK & reconv OK & non-trivial
        trivial = (checked == 0)
        valid = (local_err == 0) & (reconv_err == 0) & (~trivial)
        
        # For valid_rate, we only consider SAT cases as potentially "valid" in terms of consistency.
        # This keeps the metric stable.
        if solvability_labels is not None:
             valid = valid & (solvability_labels == 0)
             denom_count = (solvability_labels == 0).float().sum().item()
        else:
             denom_count = B
             
        valid_rate = float(valid.float().sum().item() / max(1.0, denom_count))
        
        edge_acc = float((edge_total_sum - edge_wrong_sum) / max(1.0, edge_total_sum))

    return loss, avg_reward, valid_rate, edge_acc, constraint_violation_rate


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, device: torch.device, cfg: TrainConfig, epoch: int = 1) -> Tuple[float, float, float, float, float]:
    model.train()
    total_loss, total_reward, total_valid, total_batches, total_edge_acc = 0.0, 0.0, 0.0, 0, 0.0
    total_trivial = 0.0 # Stores constraint violation rate
    
    start_time = time.time()
    bdone = 0
    
    # Target batch count for ETA
    try:
        loader_len = len(loader)
    except Exception:
        loader_len = -1
    target_batches = loader_len if loader_len > 0 else None
    if cfg.max_train_batches > 0:
        target_batches = cfg.max_train_batches if target_batches is None else min(cfg.max_train_batches, target_batches)

    # Curriculum for constraints
    constraint_prob = 0.0
    if cfg.constrained_curriculum and cfg.epochs > 0:
        progress = epoch / cfg.epochs
        constraint_prob = cfg.max_constraint_prob * progress
        
    if cfg.verbose:
        print(f"[curriculum] epoch={epoch} constraint_prob={constraint_prob:.3f}")

    for batch_idx, batch in enumerate(loader):
        paths = batch['paths_emb']
        masks = batch['attn_mask']
        node_ids = batch['node_ids']
        files = batch['files']
        
        if device.type == 'cuda':
            paths = paths.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            node_ids = node_ids.to(device, non_blocking=True)

        # Generate Constraints
        c_prob = constraint_prob
        c_mask, c_vals = generate_constraints(node_ids, files, prob=c_prob)

        # Inject constraints into embeddings
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
                    paths_flat[mask_flat, D-3:D] = one_hot

        # Anchors and Solvability
        if cfg.anchor_hint:
             ap_cpu, al_cpu, av_cpu, s_cpu = _generate_anchor(node_ids.detach().cpu(), masks.detach().cpu(), files, cfg.prefer_value)
             anchor_p = ap_cpu.to(device)
             anchor_l = al_cpu.to(device)
             anchor_v = av_cpu.to(device)
             solv_labels = s_cpu.to(device)
             paths = _inject_anchor_into_embeddings(paths, anchor_p, anchor_l, anchor_v, enable=cfg.add_logic_value)
        else:
             anchor_p = anchor_l = anchor_v = solv_labels = None  # type: ignore

        # Resolve gate types 
        gtypes = resolve_gate_types(node_ids, files, device)

        optim.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', enabled=cfg.amp):
            logits, solv_logits = model(paths, masks, gate_types=gtypes if cfg.use_gate_type_embedding else None)

            loss, avg_reward, valid_rate, batch_edge_acc, batch_c_viol = policy_loss_and_metrics(
                logits, node_ids, masks, files, gtypes,
                constraint_mask=c_mask, constraint_vals=c_vals,
                anchor_p=anchor_p, anchor_l=anchor_l, anchor_v=anchor_v,
                solvability_logits=solv_logits, solvability_labels=solv_labels,
                anchor_alpha=cfg.anchor_reward_alpha,
                normalize_reward=cfg.normalize_reward,
                entropy_beta=cfg.entropy_beta,
                soft_edge_lambda=cfg.soft_edge_lambda,
            )
        
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        total_loss += float(loss.item())
        total_reward += float(avg_reward)
        total_valid += float(valid_rate)
        total_edge_acc += float(batch_edge_acc)
        total_trivial += float(batch_c_viol) 
        total_batches += 1
        bdone += paths.size(0)
        
        if cfg.verbose and cfg.log_interval > 0 and (batch_idx + 1) % cfg.log_interval == 0:
            elapsed = time.time() - start_time
            it_per_s = (batch_idx + 1) / max(1e-6, elapsed)
            eta_str = ""
            if target_batches is not None:
                remaining = max(0, target_batches - (batch_idx + 1))
                eta = _format_seconds(remaining / max(1e-6, it_per_s))
                eta_str = f" eta={eta}"
            
            dbg = _debug_metrics_from_logits(
                logits, node_ids, masks, files,
                anchor_p=anchor_p, anchor_l=anchor_l, anchor_v=anchor_v,
                solvability_logits=solv_logits, solvability_labels=solv_labels,
            )
            print(
                f"[train] batch {batch_idx+1} avg_loss={total_loss / max(1, total_batches):.4f} "
                f"acc={total_valid / max(1, total_batches):.4f} edge_acc={dbg['edge_acc']:.3f} "
                f"solv_acc={dbg['solvability_acc']:.3f} false_unsat={dbg['false_unsat_rate']:.3f} "
                f"anchor={dbg['anchor_match_rate']:.3f} speed={it_per_s:.2f} it/s{eta_str}"
            )

        if cfg.max_train_batches > 0 and (batch_idx + 1) >= cfg.max_train_batches:
            break

    return (total_loss / max(1, total_batches), 
            total_reward / max(1, total_batches), 
            total_valid / max(1, total_batches), 
            total_edge_acc / max(1, total_batches), 
            total_trivial / max(1, total_batches))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, cfg: TrainConfig) -> Tuple[float, float, float, float, float]:
    model.eval()
    total_loss, total_reward, total_valid, total_edge_acc, total_trivial, total_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    start_time = time.time()
    
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
        
        if device.type == 'cuda':
            paths = paths.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            node_ids = node_ids.to(device, non_blocking=True)

        c_prob = cfg.max_constraint_prob if cfg.constrained_curriculum else 0.0
        c_mask, c_vals = generate_constraints(node_ids, files, c_prob)
        
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
                    paths_flat[mask_flat, D-3:D] = one_hot

        if cfg.anchor_hint:
            ap_cpu, al_cpu, av_cpu, s_cpu = _generate_anchor(node_ids.detach().cpu(), masks.detach().cpu(), files, cfg.prefer_value)
            anchor_p = ap_cpu.to(device)
            anchor_l = al_cpu.to(device)
            anchor_v = av_cpu.to(device)
            solv_labels = s_cpu.to(device)
            paths = _inject_anchor_into_embeddings(paths, anchor_p, anchor_l, anchor_v, enable=cfg.add_logic_value)
        else:
            anchor_p = anchor_l = anchor_v = solv_labels = None  # type: ignore

        gtypes = resolve_gate_types(node_ids, files, device)

        with torch.amp.autocast('cuda', enabled=cfg.amp):
            logits, solv_logits = model(paths, masks, gate_types=gtypes if cfg.use_gate_type_embedding else None)
            loss, avg_reward, valid_rate, batch_edge_acc, batch_c_viol = policy_loss_and_metrics(
                logits, node_ids, masks, files, gtypes,
                constraint_mask=c_mask, constraint_vals=c_vals,
                anchor_p=anchor_p, anchor_l=anchor_l, anchor_v=anchor_v,
                solvability_logits=solv_logits, solvability_labels=solv_labels,
                anchor_alpha=cfg.anchor_reward_alpha,
                normalize_reward=cfg.normalize_reward,
                entropy_beta=cfg.entropy_beta,
                soft_edge_lambda=cfg.soft_edge_lambda,
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
                logits, node_ids, masks, files,
                anchor_p=anchor_p, anchor_l=anchor_l, anchor_v=anchor_v,
                solvability_logits=solv_logits, solvability_labels=solv_labels,
            )
            print(
                f"[val] batch {batch_idx+1} avg_loss={total_loss / max(1, total_batches):.4f} "
                f"acc={total_valid / max(1, total_batches):.4f} edge_acc={dbg['edge_acc']:.3f} "
                f"solv_acc={dbg['solvability_acc']:.3f} speed={it_per_s:.2f} it/s"
            )

        if cfg.max_val_batches > 0 and (batch_idx + 1) >= cfg.max_val_batches:
            break

    return (total_loss / max(1, total_batches), 
            total_reward / max(1, total_batches), 
            total_valid / max(1, total_batches), 
            total_edge_acc / max(1, total_batches), 
            total_trivial / max(1, total_batches))


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
        constrained_curriculum=getattr(args, 'constrained_curriculum', False),
        max_constraint_prob=getattr(args, 'max_constraint_prob', 0.5),
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
        tr_loss, tr_reward, tr_acc, tr_edge, tr_c_viol = train_one_epoch(model, train_loader, optim, scaler, device, cfg, epoch=epoch)
        va_loss, va_reward, va_acc, va_edge, va_c_viol = evaluate(model, val_loader, device, cfg)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f} avg_reward={tr_reward:.4f} acc={tr_acc:.4f} edge_acc={tr_edge:.4f} c_viol={tr_c_viol:.4f} | "
            f"val_loss={va_loss:.4f} avg_reward={va_reward:.4f} acc={va_acc:.4f} edge_acc={va_edge:.4f} c_viol={va_c_viol:.4f}"
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
    t.add_argument('--constrained-curriculum', action='store_true', help='Enable constrained path training curriculum')
    t.add_argument('--max-constraint-prob', type=float, default=0.5, help='Maximum probability of masking a constrained node')
    

    t.add_argument('--batch-size', type=int, default=128)
    t.add_argument('--verbose', action='store_true')
    # Model capacity
    t.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    t.add_argument('--enc-layers', type=int, default=3, help='Number of shared path encoder layers')
    t.add_argument('--int-layers', type=int, default=3, help='Number of path interaction layers')
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