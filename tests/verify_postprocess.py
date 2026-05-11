#!/usr/bin/env python3
"""
Quick verification of post-processing fix for NOT/BUFF gates.
Compares accuracy before and after applying deterministic gate rule propagation.
"""

import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ml.core.model import MultiPathTransformer
from src.ml.train import TrainConfig, make_dataloaders, resolve_gate_types
from src.util.struct import GateType


def post_process_batch(actions, gate_types, mask):
    """Batch post-processing: forward-propagate NOT/BUFF rules.

    Args:
        actions: [B, P, L] predicted values (0 or 1)
        gate_types: [B, P, L] gate type for each position
        mask: [B, P, L] valid mask
    Returns:
        corrected: [B, P, L] corrected values
    """
    corrected = actions.clone()
    B, P, L = actions.shape

    for b in range(B):
        for p in range(P):
            path_len = int(mask[b, p].sum().item())
            if path_len <= 1:
                continue
            for pos in range(1, path_len):
                gt = int(gate_types[b, p, pos].item())
                prev_val = int(corrected[b, p, pos - 1].item())
                if gt == GateType.NOT:
                    corrected[b, p, pos] = 1 - prev_val
                elif gt == GateType.BUFF:
                    corrected[b, p, pos] = prev_val
    return corrected


def check_edges(actions, gate_types, mask):
    """Count edge errors."""
    valid_edges = mask[:, :, 1:] & mask[:, :, :-1]
    prev_vals = actions[:, :, :-1]
    cur_vals = actions[:, :, 1:]
    gt_cur = gate_types[:, :, 1:]

    edge_ok = torch.ones_like(prev_vals, dtype=torch.bool)

    m = gt_cur == GateType.NOT
    edge_ok[m] &= cur_vals[m] == (1 - prev_vals[m])
    m = gt_cur == GateType.BUFF
    edge_ok[m] &= cur_vals[m] == prev_vals[m]
    m = gt_cur == GateType.AND
    edge_ok[m] &= cur_vals[m] <= prev_vals[m]
    m = gt_cur == GateType.NAND
    edge_ok[m] &= cur_vals[m] >= (1 - prev_vals[m])
    m = gt_cur == GateType.OR
    edge_ok[m] &= cur_vals[m] >= prev_vals[m]
    m = gt_cur == GateType.NOR
    edge_ok[m] &= cur_vals[m] <= (1 - prev_vals[m])

    wrong_edges = (~edge_ok) & valid_edges
    local_wrong = wrong_edges.sum(dim=(1, 2))  # [B]
    checked = valid_edges.sum(dim=(1, 2))  # [B]

    return local_wrong, checked, wrong_edges


def check_reconv(actions, mask):
    """Check reconvergence failures."""
    path_len = mask.long().sum(dim=-1)
    last_idx = (path_len - 1).clamp(min=0)
    last_vals = actions.gather(2, last_idx.unsqueeze(-1)).squeeze(-1)
    path_valid = path_len > 0

    lv_f = last_vals.float()
    vm_f = path_valid.float()
    max_v = (lv_f * vm_f + (-999.0) * (1 - vm_f)).max(dim=-1).values
    min_v = (lv_f * vm_f + (999.0) * (1 - vm_f)).min(dim=-1).values
    has_valid = path_valid.sum(dim=-1) > 0
    reconv_fail = (min_v < max_v) & has_valid
    return reconv_fail


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data/datasets/shards_anchored")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=20)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})

    model_dim = cfg_dict.get("model_dim", 512)
    nhead = cfg_dict.get("nhead", 4)
    enc_layers = cfg_dict.get("num_encoder_layers", cfg_dict.get("enc_layers", 3))
    int_layers = cfg_dict.get("num_interaction_layers", cfg_dict.get("int_layers", 3))
    ffn_dim = cfg_dict.get("dim_feedforward", cfg_dict.get("ffn_dim", 512))

    cfg = TrainConfig(
        dataset="",
        output="",
        processed_dir=args.data_dir,
        batch_size=args.batch_size,
        model_dim=model_dim,
        nhead=nhead,
        num_encoder_layers=enc_layers,
        num_interaction_layers=int_layers,
        dim_feedforward=ffn_dim,
        use_gate_type_embedding=cfg_dict.get("use_gate_type_embedding", True),
        verbose=False,
    )

    train_loader, val_loader = make_dataloaders(cfg, device)
    loader = val_loader if args.split == "val" else train_loader

    emb_dim = next(iter(loader))["paths_emb"].shape[-1]
    if model_dim % nhead != 0:
        model_dim = ((model_dim // nhead) + 1) * nhead

    model = MultiPathTransformer(
        input_dim=emb_dim,
        model_dim=model_dim,
        nhead=nhead,
        num_encoder_layers=enc_layers,
        num_interaction_layers=int_layers,
        dim_feedforward=ffn_dim,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Metrics
    before = {
        "samples": 0,
        "zero_err": 0,
        "total_wrong": 0,
        "total_edges": 0,
        "reconv_fail": 0,
    }
    after = {
        "samples": 0,
        "zero_err": 0,
        "total_wrong": 0,
        "total_edges": 0,
        "reconv_fail": 0,
    }

    print(f"Analyzing {args.max_batches} batches from {args.split}...\n")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break

            paths = batch["paths_emb"].to(device)
            masks = batch["attn_mask"].to(device)
            node_ids = batch["node_ids"].to(device)
            files = batch["files"]
            gtypes = resolve_gate_types(node_ids, files, device)

            logits, _ = model(
                paths, masks, gate_types=gtypes if cfg.use_gate_type_embedding else None
            )
            actions_raw = logits.argmax(dim=-1)  # [B, P, L]
            actions_pp = post_process_batch(actions_raw, gtypes, masks)

            B = actions_raw.shape[0]

            # Before post-processing
            lw, ch, _ = check_edges(actions_raw, gtypes, masks)
            rf = check_reconv(actions_raw, masks)
            before["samples"] += B
            before["zero_err"] += (lw == 0).sum().item()
            before["total_wrong"] += lw.sum().item()
            before["total_edges"] += ch.sum().item()
            before["reconv_fail"] += rf.sum().item()

            # After post-processing
            lw, ch, _ = check_edges(actions_pp, gtypes, masks)
            rf = check_reconv(actions_pp, masks)
            after["samples"] += B
            after["zero_err"] += (lw == 0).sum().item()
            after["total_wrong"] += lw.sum().item()
            after["total_edges"] += ch.sum().item()
            after["reconv_fail"] += rf.sum().item()

    # Report
    print("=" * 70)
    print(f"POST-PROCESSING VERIFICATION — {before['samples']} samples from {args.split}")
    print("=" * 70)

    print(f"\n{'Metric':<30s}  {'Before':>12s}  {'After':>12s}  {'Δ':>10s}")
    print("-" * 70)

    def pct(n, d):
        return f"{n/max(1,d):.4f}" if d > 0 else "N/A"

    b_acc = before["zero_err"] / max(1, before["samples"])
    a_acc = after["zero_err"] / max(1, after["samples"])
    print(
        f"{'Zero-error rate (acc)':<30s}  "
        f"{pct(before['zero_err'], before['samples']):>12s}  "
        f"{pct(after['zero_err'], after['samples']):>12s}  {a_acc - b_acc:>+10.4f}"
    )

    b_ea = 1 - before["total_wrong"] / max(1, before["total_edges"])
    a_ea = 1 - after["total_wrong"] / max(1, after["total_edges"])
    print(f"{'Edge accuracy':<30s}  {b_ea:>12.4f}  {a_ea:>12.4f}  {a_ea - b_ea:>+10.4f}")

    print(
        f"{'Total edge errors':<30s}  {before['total_wrong']:>12d}  "
        f"{after['total_wrong']:>12d}  "
        f"{after['total_wrong'] - before['total_wrong']:>+10d}"
    )
    print(
        f"{'Reconv failures':<30s}  {before['reconv_fail']:>12d}  "
        f"{after['reconv_fail']:>12d}  "
        f"{after['reconv_fail'] - before['reconv_fail']:>+10d}"
    )


if __name__ == "__main__":
    main()
