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
import warnings
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Tuple
import time
from tqdm import tqdm

# Suppress annoying prototype warnings from PyTorch transformer
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

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
    enforce_constraints: bool = True
    processed_dir: Optional[str] = None
    
    # Phase 7: Logic Consistency
    lambda_logic: float = 0.0  # Weight for reconvergence logic consistency loss
    lambda_full_logic: float = 0.0  # Weight for full-path gate consistency loss

    # Gumbel Softmax
    gumbel_temp: float = 1.0
    gumbel_anneal_rate: float = 0.99


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


@lru_cache(maxsize=32)
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
    # IMPORTANT: Use CPU for initial creation of these buffers to avoid generic CUDA kernel errors 
    # during initialization which were observed on certain GPU architectures.
    constraint_mask = torch.zeros((B, P, L), dtype=torch.bool, device='cpu')
    constraint_vals = torch.zeros((B, P, L), dtype=torch.long, device='cpu')
    
    if prob <= 0.0:
        return constraint_mask, constraint_vals

    # Subsampling optimization: only simulate a subset of the batch
    # This prevents the main thread from blocking for too long.
    max_samples = 16
    indices = torch.randperm(B)[:max_samples].tolist()
    
    # for b in range(B): # OLD: simulate all
    for b in indices:
        # if b % 10 == 0:
        #      print(f"[Constraint Gen] Simulating {files[b]} ({b+1}/{B})")
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
        
        vals_t = torch.tensor(flat_vals, dtype=torch.long, device='cpu').view(P, L)
        
        # Mask generation: Only where we have a valid value (0 or 1)
        valid_val_mask = vals_t >= 0
        
        # Random mask based on prob
        rand_probs = torch.rand((P, L), device='cpu')
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


def calculate_logic_loss(
    node_ids: torch.Tensor,     # [B, P, L]
    gate_types: torch.Tensor,   # [B, P, L]
    probs: torch.Tensor,        # [B, P, L, 2] - Gumbel Softmax Outputs
    mask_valid: torch.Tensor,   # [B, P, L]
    device: torch.device
) -> torch.Tensor:
    """
    Computes a Logic Consistency Loss by checking local gate logic.
    Vectorized implementation to avoid Python loops.
    """
    B, P, L_dim = node_ids.shape
    
    # 1. Identify Reconvergence Gate Indices per Batch Element
    # The reconvergence gate is the last valid node in each path.
    # Paths in a sample all reconverge to the same gate (by dataset construction).
    # We can pick the first valid path to find the location.
    
    path_lens = mask_valid.long().sum(dim=2) # [B, P]
    # We need at least one path with length >= 2 (input -> reconv)
    valid_paths_mask = (path_lens >= 2) # [B, P]
    
    # Filter batches that have no valid paths
    batch_has_valid = valid_paths_mask.any(dim=1) # [B]
    if not batch_has_valid.any():
        return torch.tensor(0.0, device=device)

    # 2. Gather Inputs and Outputs
    # Inputs are at index (len-2), Output (Reconv) is at index (len-1)
    
    # We create a gather index for the last node (reconv) and 2nd last (input)
    # Indices: [B, P]
    last_idx = (path_lens - 1).clamp(min=0)
    second_last = (last_idx - 1).clamp(min=0)
    
    # Get Probabilities of logic-1 (Using Gumbel Softmax output passed in)
    probs_1 = probs[..., 1] # [B, P, L]
    
    # Gather output probs: [B, P]
    # We gather from [B, P, L] using indices [B, P]
    idx_reconv = last_idx.unsqueeze(-1) # [B, P, 1]
    p_out_per_path = probs_1.gather(2, idx_reconv).squeeze(-1) # [B, P]
    
    # Gather input probs: [B, P]
    idx_input = second_last.unsqueeze(-1) # [B, P, 1]
    p_in_per_path = probs_1.gather(2, idx_input).squeeze(-1) # [B, P]
    
    # Gate Types for Reconv Gate: [B, P]
    # Should be identical across P for the same B (all paths reconverge to same gate)
    # We'll just take the mean/mode or assume consistency.
    gt_reconv = gate_types.gather(2, idx_reconv).squeeze(-1) # [B, P]
    
    # 3. Compute Implied Probabilities (Vectorized)
    # We need to aggregate inputs for each batch sample.
    # Inputs are p_in_per_path[b, :] where valid_paths_mask[b, :] is True.
    
    # Since P is small (fixed max paths), we can mask and compute.
    # Mask invalid paths with identity values for the operation.
    # AND/NAND: identity = 1.0
    # OR/NOR: identity = 0.0
    
    # Prepare floating point mask
    f_mask = valid_paths_mask.float()
    
    # AND Logic: Product of inputs
    # Mask invalid entries with 1.0 (so they don't affect product)
    in_and = p_in_per_path * f_mask + (1.0 - f_mask) 
    # Product across P dim
    implied_and = in_and.prod(dim=1) # [B]
    
    # OR Logic: 1 - Product(1-inputs)
    # Mask invalid entries with 0.0 (so 1-0=1, product neutral)
    # Wait, for OR: we want prod(1-x). If specific x is invalid, we want it to be 0? 
    # No, we want 1-x to be 1. So x must be 0.
    in_or = p_in_per_path * f_mask # invalid -> 0
    implied_or = 1.0 - (1.0 - in_or).prod(dim=1) # [B]
    
    # Reconv Gate Type (per batch)
    # We can take the max/first valid gate type per batch
    # (Assuming all paths in sample agree)
    # Gate types are integers. We need to index into operation results.
    
    # Let's get a representative gate type for each batch
    # We can pick the type from the first valid path.
    # Argmax of mask gives index of first True.
    first_valid_idx = valid_paths_mask.long().argmax(dim=1) # [B]
    # Gather gate type
    batch_gt = gt_reconv.gather(1, first_valid_idx.unsqueeze(1)).squeeze(1) # [B]
    
    # Calculate implied p1 for all types
    # Initialize with 0.5
    target_p1 = torch.full((B,), 0.5, device=device)
    
    is_and = (batch_gt == GateType.AND)
    is_nand = (batch_gt == GateType.NAND)
    is_or = (batch_gt == GateType.OR)
    is_nor = (batch_gt == GateType.NOR)
    is_not = (batch_gt == GateType.NOT)
    is_buff = (batch_gt == GateType.BUFF)
    
    # Assign Implied Targets
    # AND
    target_p1[is_and] = implied_and[is_and]
    
    # NAND (1 - AND)
    target_p1[is_nand] = 1.0 - implied_and[is_nand]
    
    # OR
    target_p1[is_or] = implied_or[is_or]
    
    # NOR (1 - OR)
    target_p1[is_nor] = 1.0 - implied_or[is_nor]
    
    # BUFF/NOT
    # For single input gates, we usually have 1 path, or duplicate paths.
    # Mean of inputs is a safe bet for duplicate paths.
    # But strictly, BUFF/NOT should have 1 input.
    # Sum / Count is better if we only have valid paths.
    sum_in = (p_in_per_path * f_mask).sum(dim=1)
    count_in = f_mask.sum(dim=1).clamp(min=1.0)
    mean_in = sum_in / count_in
    
    target_p1[is_buff] = mean_in[is_buff]
    target_p1[is_not] = 1.0 - mean_in[is_not]
    
    # 4. Compute Loss
    # Compare against Predicted Output
    # We assume 'mean' prediction across duplicate paths for the output as well
    sum_out = (p_out_per_path * f_mask).sum(dim=1)
    mean_out = sum_out / count_in
    
    # Only compute loss for supported gate types and valid batches
    supported = (is_and | is_nand | is_or | is_nor | is_not | is_buff)
    mask_calc = (batch_has_valid & supported)
    
    if not mask_calc.any():
        return torch.tensor(0.0, device=device)
        
    loss = F.mse_loss(mean_out[mask_calc], target_p1[mask_calc])
    
    return loss


def calculate_full_logic_loss(
    gate_types: torch.Tensor,   # [B, P, L]
    probs: torch.Tensor,       # [B, P, L, 2]
    mask_valid: torch.Tensor,   # [B, P, L]
    device: torch.device
) -> torch.Tensor:
    """
    Full-path gate consistency loss.
    Penalizes ALL edge-level gate logic violations along paths as a direct
    """
    probs_1 = probs[..., 1]  # [B, P, L]
    
    valid_edges = mask_valid[:, :, 1:] & mask_valid[:, :, :-1]  # [B, P, L-1]
    if not valid_edges.any():
        return torch.tensor(0.0, device=device)
    
    prev_p = probs_1[:, :, :-1]
    cur_p = probs_1[:, :, 1:]
    gt_cur = gate_types[:, :, 1:]  # [B, P, L-1]
    
    viol = torch.zeros_like(prev_p)
    
    # NOT: |cur - (1-prev)|^2
    m = gt_cur == GateType.NOT
    if m.any():
        viol[m] = (cur_p[m] - (1.0 - prev_p[m])) ** 2
    
    # BUFF: |cur - prev|^2
    m = gt_cur == GateType.BUFF
    if m.any():
        viol[m] = (cur_p[m] - prev_p[m]) ** 2
    
    # AND: cur <= prev => ReLU(cur - prev)^2
    m = gt_cur == GateType.AND
    if m.any():
        viol[m] = F.relu(cur_p[m] - prev_p[m]) ** 2
    
    # NAND: cur >= 1-prev => ReLU((1-prev) - cur)^2
    m = gt_cur == GateType.NAND
    if m.any():
        viol[m] = F.relu((1.0 - prev_p[m]) - cur_p[m]) ** 2
    
    # OR: cur >= prev => ReLU(prev - cur)^2
    m = gt_cur == GateType.OR
    if m.any():
        viol[m] = F.relu(prev_p[m] - cur_p[m]) ** 2
    
    # NOR: cur <= 1-prev => ReLU(cur - (1-prev))^2
    m = gt_cur == GateType.NOR
    if m.any():
        viol[m] = F.relu(cur_p[m] - (1.0 - prev_p[m])) ** 2
    
    # Heavy weights for deterministic gates
    edge_weights = torch.ones_like(gt_cur, dtype=torch.float32)
    edge_weights[gt_cur == GateType.NOT] = 20.0
    edge_weights[gt_cur == GateType.BUFF] = 20.0
    
    viol = viol * edge_weights * valid_edges.float()
    
    valid_edge_counts = valid_edges.sum(dim=(1,2)).float().clamp(min=1.0)
    loss_per_sample = viol.sum(dim=(1,2)) / valid_edge_counts
    
    return loss_per_sample.mean()


def reinforce_loss(
    logits: torch.Tensor,
    gate_types: torch.Tensor,
    mask_valid: torch.Tensor,
    solvability_logits: Optional[torch.Tensor] = None,
    solvability_labels: Optional[torch.Tensor] = None,
    anchor_p: Optional[torch.Tensor] = None,
    anchor_l: Optional[torch.Tensor] = None,
    anchor_v: Optional[torch.Tensor] = None,
    entropy_beta: float = 0.01,
    constraint_mask: Optional[torch.Tensor] = None,
    constraint_vals: Optional[torch.Tensor] = None,
    node_ids: Optional[torch.Tensor] = None,
    lambda_logic: float = 0.0,
    lambda_full_logic: float = 0.0,
    soft_edge_lambda: float = 1.0,
    normalize_reward: bool = True,
    anchor_alpha: float = 0.1,
    gumbel_temp: float = 1.0, 
) -> tuple[torch.Tensor, float, float, float, float]:
    """Compute Loss using Gumbel-Softmax for differentiable logic consistency.

    The 'reward' concepts from RL are kept for metric logging but removed from the
    gradient path in favor of direct logic differentiation.

    Returns: (loss, avg_reward, valid_rate, edge_acc, constraint_violation_rate)
    """

    B, P, L, C = logits.shape

    # Gumbel Softmax Action Sampling (Straight-Through Estimator)
    # hard=True:
    #   Forward pass: one-hot discrete values (0 or 1)
    #   Backward pass: uses gradients of the Gumbel-Softmax distribution
    # We use this ONE-HOT output for all logic calculations to propagate gradients.
    actions_one_hot = F.gumbel_softmax(logits, tau=gumbel_temp, hard=True, dim=-1) # [B, P, L, 2]
    
    # Extract indices for legacy code (constraint masking, metrics)
    actions = actions_one_hot.argmax(dim=-1) # [B, P, L] - No grad flow through argmax usually, but OK here as we use one_hot for loss
    
    # constraint_metrics
    constraint_loss = torch.tensor(0.0, device=logits.device)
    constraint_violation_rate = 0.0
    
    # Enforce constraints if provided
    if constraint_mask is not None and constraint_vals is not None: 
        if constraint_mask.any():
            valid_constraints = constraint_mask.view(-1)
            flat_logits = logits.view(-1, 2)
            flat_targets = constraint_vals.view(-1)
            
            # CE Loss on constrained nodes (Supervised)
            c_loss = F.cross_entropy(flat_logits[valid_constraints], flat_targets[valid_constraints])
            constraint_loss = c_loss * 1.0 
            
            # Metric: Violation rate
            preds = actions[constraint_mask]
            targets = constraint_vals[constraint_mask]
            violations = (preds != targets).float().sum()
            total_c = targets.numel()
            constraint_violation_rate = (violations / max(1, total_c)).item()
            
            # Forcing: Update actions to match constraints to ensure downstream path consistency?
            # In RL we could overwrite actions. In Gumbel matching loss is usually enough.
            # But to help convergence we can overwrite the 'hard' actions for the forward pass context
            # if we wanted. But Gumbel hard=True makes it discrete.
            # Let's overwrite `actions` indices for metrics, but we can't easily overwrite `actions_one_hot`
            # without breaking gradient flow unless we are careful.
            # Since constraint_loss is strong, we'll assume it converges.
            # However, existing code overwrote actions.
            actions[constraint_mask] = constraint_vals[constraint_mask]

    # Solvability Loss
    solvability_loss = torch.tensor(0.0, device=logits.device)
    if solvability_logits is not None and solvability_labels is not None:
        weights = torch.tensor([10.0, 1.0], device=logits.device) 
        solvability_loss = F.cross_entropy(solvability_logits, solvability_labels, weight=weights) * 1.0

    # RL Reward Logic (For Logging Only)
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

    # Initialize loss
    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Vectorized Logic Consistency (Metrics Only)
    # We use `actions` (indices) for metrics to see "Hard" violations
    valid_edges = mask_valid[:, :, 1:] & mask_valid[:, :, :-1]  # [B, P, L-1]
    prev_vals = actions[:, :, :-1]
    cur_vals = actions[:, :, 1:]
    gt_cur = gate_types[:, :, 1:] 
    
    edge_ok = torch.ones_like(prev_vals, dtype=torch.bool)
    m = gt_cur == GateType.NOT
    edge_ok[m] &= (cur_vals[m] == (1 - prev_vals[m]))
    m = gt_cur == GateType.BUFF
    edge_ok[m] &= (cur_vals[m] == prev_vals[m])
    m = gt_cur == GateType.AND
    edge_ok[m] &= (cur_vals[m] <= prev_vals[m])
    m = gt_cur == GateType.NAND
    edge_ok[m] &= (cur_vals[m] >= (1 - prev_vals[m]))
    m = gt_cur == GateType.OR
    edge_ok[m] &= (cur_vals[m] >= prev_vals[m])
    m = gt_cur == GateType.NOR
    edge_ok[m] &= (cur_vals[m] <= (1 - prev_vals[m]))

    wrong_edges = (~edge_ok) & valid_edges
    local_wrong = wrong_edges.sum(dim=(1, 2))
    checked = valid_edges.sum(dim=(1, 2))
    edge_wrong_sum = local_wrong.sum().item()
    edge_total_sum = checked.sum().item()

    # Vectorized Reconvergence Failures (Metrics Only)
    path_len = mask_valid.long().sum(dim=-1)
    last_idx = (path_len - 1).clamp(min=0)
    last_idx_exp = last_idx.unsqueeze(-1)
    last_vals = actions.gather(2, last_idx_exp).squeeze(-1)
    path_valid_mask = (path_len > 0)
    
    neg_inf = -999.0
    pos_inf = 999.0
    lv_float = last_vals.float()
    vm_float = path_valid_mask.float()
    max_v = (lv_float * vm_float + neg_inf * (1 - vm_float)).max(dim=-1).values
    min_v = (lv_float * vm_float + pos_inf * (1 - vm_float)).min(dim=-1).values
    
    has_valid_paths = (path_valid_mask.sum(dim=-1) > 0)
    reconv_fail_mask = (min_v < max_v) & has_valid_paths
    reconv_wrong = reconv_fail_mask.float()
    
    # Metrics
    with torch.no_grad():
        trivial = (checked == 0)
        valid = (local_err := local_wrong.float()) == 0
        valid = valid & (reconv_wrong == 0) & (~trivial)
        if solvability_labels is not None:
             valid = valid & (solvability_labels == 0)
             denom_count = (solvability_labels == 0).float().sum().item()
        else:
             denom_count = B
        valid_rate = float(valid.float().sum().item() / max(1.0, denom_count))
        edge_acc = float((edge_total_sum - edge_wrong_sum) / max(1.0, edge_total_sum))
        
        # Calculate a pseudo-reward for logging consistency
        local_reward_shaping = (1.0 - (local_err / checked.clamp(min=1.0))) * 2.0 - 1.0
        reconv_bonus = 0.5
        reconv_penalty = -2.0
        sat_base_reward = torch.where(reconv_wrong == 0, 
                                      local_reward_shaping + reconv_bonus, 
                                      torch.min(local_reward_shaping, torch.tensor(reconv_penalty, device=logits.device)))
        reward = sat_base_reward.clone()
        avg_reward = float(reward.mean().item())


    # -----------------------------------------------------------------------
    # Differentiable Losses
    # -----------------------------------------------------------------------
    
    # 1. Reconvergence Consistency Loss (MSE on Logits / Gumbel Probs)
    # We use logits or gumbel outputs. Gumbel outputs are safer for consistency.
    # Gather output probs [B, P, 2]
    # last_idx_logits = last_idx_exp.unsqueeze(-1).expand(actions.size(0), actions.size(1), 1, 2)
    # last_probs = actions_one_hot.gather(2, last_idx_logits).squeeze(2) # [B, P, 2]
    # Actually, let's use the code from before but on LOGITS to permit gradients 
    # even without Gumbel hard path if needed, BUT mixing them is tricky.
    # Let's use logits for the MSE consistency as it provides a smooth signal.
    
    # Gather logits at last_idx: [B, P, C]
    last_idx_logits = last_idx_exp.unsqueeze(-1).expand(actions.size(0), actions.size(1), 1, logits.shape[-1])
    last_logits = logits.gather(2, last_idx_logits).squeeze(2) # [B, P, C]
    
    mask_exp = path_valid_mask.unsqueeze(-1) # [B, P, 1]
    count_paths = mask_exp.sum(dim=1).clamp(min=1.0) # [B, 1]
    sum_logits = (last_logits * mask_exp).sum(dim=1) # [B, C]
    mean_logits = sum_logits / count_paths # [B, C]
    
    diff = (last_logits - mean_logits.unsqueeze(1))
    mse_per_sample = ((diff ** 2) * mask_exp).sum(dim=(1,2)) / count_paths.squeeze(-1) # [B]
    loss = loss + 0.5 * mse_per_sample.mean()

    # 2. Vectorized Soft Edge Loss (Using Gumbel Outputs one_hot)
    # This acts as a differentiable consistency check.
    probs_1 = actions_one_hot[..., 1] # [B, P, L]
    prev_p = probs_1[:, :, :-1]
    cur_p = probs_1[:, :, 1:]
    
    viol = torch.zeros_like(prev_p)
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
    
    edge_weights = torch.ones_like(gt_cur, dtype=torch.float32)
    edge_weights[gt_cur == GateType.NOT] = 20.0 # Heavy penalty for inverters
    viol = viol * edge_weights * valid_edges.float()
    
    valid_edge_counts = valid_edges.sum(dim=(1,2)).float().clamp(min=1.0)
    edge_loss_per_sample = viol.sum(dim=(1,2)) / valid_edge_counts
    
    if soft_edge_lambda > 0:
        loss = loss + float(soft_edge_lambda) * edge_loss_per_sample.mean()

    # 3. Entropy regularization (Using logits)
    if entropy_beta > 0.0:
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        ent_mean = (ent * mask_valid.float()).sum() / torch.clamp(mask_valid.float().sum(), min=1.0)
        loss = loss - float(entropy_beta) * ent_mean

    # 4. Anchor Supervision (Standard CE)
    if anchor_p is not None and anchor_l is not None and anchor_v is not None:
        valid_anchors = (anchor_p >= 0) & (anchor_l >= 0)
        if solvability_labels is not None:
            valid_anchors = valid_anchors & (solvability_labels == 0)
        if valid_anchors.any():
            b_idx = torch.arange(logits.size(0), device=logits.device)[valid_anchors]
            p_idx = anchor_p[valid_anchors]
            l_idx = anchor_l[valid_anchors]
            targets = anchor_v[valid_anchors].long().clamp(0, 1)
            pred_logits = logits[b_idx, p_idx, l_idx] 
            sup_loss = F.cross_entropy(pred_logits, targets)
            loss = loss + 1.0 * sup_loss
    
    # 5. Add accumulated auxiliary losses
    loss = loss + constraint_loss + solvability_loss

    # 6. Reconvergence Logic Loss (Differentiable via Gumbel)
    if lambda_logic > 0.0 and node_ids is not None:
        logic_loss = calculate_logic_loss(
            node_ids=node_ids,
            gate_types=gate_types,
            probs=actions_one_hot, # Pass Gumbel outputs
            mask_valid=mask_valid,
            device=logits.device
        )
        loss = loss + (lambda_logic * logic_loss)

    # 7. Full-Path Gate Consistency Loss (Differentiable via Gumbel)
    if lambda_full_logic > 0.0:
        full_logic_loss = calculate_full_logic_loss(
            gate_types=gate_types,
            probs=actions_one_hot, # Pass Gumbel outputs
            mask_valid=mask_valid,
            device=logits.device
        )
        loss = loss + (lambda_full_logic * full_logic_loss)

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

    # Curriculum: Phase 1 (0-50%) = Pure Learning (No Constraints)
    #             Phase 2 (50%-100%) = Linear Ramp (0 -> Max Constraints)
    constraint_prob = 0.0
    if cfg.constrained_curriculum and cfg.epochs > 0:
        half_epochs = cfg.epochs // 2
        if epoch <= half_epochs:
            constraint_prob = 0.0
        else:
            # Progress within Phase 2 (0.0 to 1.0)
            phase2_progress = (epoch - half_epochs) / (cfg.epochs - half_epochs)
            constraint_prob = cfg.max_constraint_prob * phase2_progress
        
    if cfg.verbose:
        print(f"[curriculum] epoch={epoch} constraint_prob={constraint_prob:.3f} (Phase {'1' if epoch <= cfg.epochs//2 else '2'})")

    pbar = tqdm(loader, desc=f"Epoch {epoch}", total=target_batches, disable=not cfg.verbose)
    for batch_idx, batch in enumerate(pbar):
        paths = batch['paths_emb']
        masks = batch['attn_mask']
        node_ids = batch['node_ids']
        files = batch['files']
        
        # Initialize Logic Value Vector to "Unknown" [0, 0, 1] if enabled
        if cfg.add_logic_value:
            B_cur, P_cur, L_cur, D_cur = paths.shape
            if D_cur >= 3:
                # IMPORTANT: Perform this on CPU to avoid "CUDA error: no kernel image is available"
                # which seems to happen on some GPU architectures with specific indexing patterns in DataParallel.
                paths[..., D_cur-3] = 0.0
                paths[..., D_cur-2] = 0.0
                paths[..., D_cur-1] = 1.0

        if device.type == 'cuda':
            paths = paths.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            node_ids = node_ids.to(device, non_blocking=True)

        # Generate Constraints
        c_prob = constraint_prob
        c_mask, c_vals = generate_constraints(node_ids, files, prob=c_prob)
        c_mask = c_mask.to(device)
        c_vals = c_vals.to(device)

        # Inject constraints into embeddings
        if cfg.add_logic_value:
            # We always check for constraints but only inject if mask is present
            if c_mask.any():
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

        # Try to use anchors provided by the dataset (parallelized in workers)
        if 'anchor_p' in batch and batch['anchor_p'] is not None and 'solvability' in batch:
             anchor_p = batch['anchor_p'].to(device)
             anchor_l = batch['anchor_l'].to(device)
             anchor_v = batch['anchor_v'].to(device)
             solv_labels = batch['solvability'].to(device)
             # Inject into embeddings if enabled
             if cfg.add_logic_value:
                  paths = _inject_anchor_into_embeddings(paths, anchor_p, anchor_l, anchor_v, enable=True)
                  
        elif cfg.anchor_hint:
             # Fallback to main-thread generation (slow)
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
        
        # Gumbel Annealing
        gumbel_t = max(0.1, cfg.gumbel_temp * (cfg.gumbel_anneal_rate ** (epoch - 1)))
        
        with torch.amp.autocast('cuda', enabled=cfg.amp):
            logits, solv_logits = model(paths, masks, gate_types=gtypes if cfg.use_gate_type_embedding else None)

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
            dbg = _debug_metrics_from_logits(
                logits, node_ids, masks, files,
                anchor_p=anchor_p, anchor_l=anchor_l, anchor_v=anchor_v,
                solvability_logits=solv_logits, solvability_labels=solv_labels,
            )
            pbar.set_postfix({
                'loss': f"{total_loss / max(1, total_batches):.4f}",
                'acc': f"{total_valid / max(1, total_batches):.4f}",
                'solv': f"{dbg['solvability_acc']:.3f}",
                'edge': f"{dbg['edge_acc']:.3f}"
            })

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

    pbar = tqdm(loader, desc="Eval", total=target_batches, disable=not cfg.verbose)
    for batch_idx, batch in enumerate(pbar):
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
        processed_dir=getattr(args, 'processed_dir', None),
        lambda_logic=getattr(args, 'lambda_logic', 0.0),
        lambda_full_logic=getattr(args, 'lambda_full_logic', 0.0),
        gumbel_temp=getattr(args, 'gumbel_temp', 1.0),
        gumbel_anneal_rate=getattr(args, 'gumbel_anneal_rate', 0.99),
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
    t.add_argument('--processed-dir', type=str, help='Directory containing pre-processed shard_*.pt files')
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
    t.add_argument('--lambda-logic', type=float, default=0.0, help='Weight for reconvergence logic consistency loss (Phase 7)')
    t.add_argument('--lambda-full-logic', type=float, default=0.0, help='Weight for full-path gate consistency loss')
    t.add_argument('--gumbel-temp', type=float, default=1.0, help='Initial Gumbel Softmax temperature')
    t.add_argument('--gumbel-anneal-rate', type=float, default=0.99, help='Annealing rate per epoch')

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