"""
Minimal RL-aware evaluator for Multi-Path reconvergent transformer checkpoints.

Loads a saved checkpoint produced by the minimal trainer and reports the
REINFORCE-style loss proxy and average reward on the provided dataset.
"""

from __future__ import annotations

import argparse

import torch

from src.ml.core.dataset import ReconvergentPathsDataset, reconv_collate, resolve_gate_types
from src.ml.core.loss import reinforce_loss
from src.ml.core.model import MultiPathTransformer


@torch.no_grad()
def evaluate(model: MultiPathTransformer, loader) -> tuple[float, float, float]:
    """Evaluate average policy loss, reward, and accuracy over the dataset.

    Accuracy is the percentage of path pairs that successfully justify to the
    required output (zero constraint violations).
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_reward = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = 0

    for batch in loader:
        paths = batch["paths_emb"].to(device)
        masks = batch["attn_mask"].to(device)
        node_ids = batch["node_ids"].to(device)
        files = batch["files"]
        gtypes = (
            batch["gate_types"].to(device)
            if "gate_types" in batch
            else resolve_gate_types(node_ids, files, device)
        )

        logits, _solv_logits = model(paths, masks, gate_types=gtypes)
        loss, avg_reward, valid_rate, _edge_acc, _c_viol = reinforce_loss(
            logits=logits,
            gate_types=gtypes,
            mask_valid=masks,
        )
        total_loss += float(loss.item())
        total_reward += float(avg_reward)

        # valid_rate is the fraction of samples in this batch with zero violations
        batch_size = paths.size(0)
        total_correct += int(valid_rate * batch_size)
        total_samples += batch_size
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    avg_reward = total_reward / max(1, total_batches)
    accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0
    return avg_loss, avg_reward, accuracy


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal reconv RL evaluator")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-encoder-layers", type=int, default=1)
    ap.add_argument("--num-interaction-layers", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = (
        torch.device("cpu") if args.cpu or (not torch.cuda.is_available()) else torch.device("cuda")
    )

    ds = ReconvergentPathsDataset(args.dataset, device=device)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, collate_fn=reconv_collate
    )

    model = MultiPathTransformer(
        input_dim=args.embedding_dim,
        model_dim=512,  # Default model_dim used in training
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_interaction_layers=args.num_interaction_layers,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    loss, avg_reward, accuracy = evaluate(model, dl)
    print(f"EVAL | loss={loss:.4f} avg_reward={avg_reward:.4f} accuracy={accuracy:.2%}")


if __name__ == "__main__":
    main()
