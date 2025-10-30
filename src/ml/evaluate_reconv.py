"""
Minimal RL-aware evaluator for Multi-Path reconvergent transformer checkpoints.

Loads a saved checkpoint produced by the minimal trainer and reports the
REINFORCE-style loss proxy and average reward on the provided dataset.
"""

from __future__ import annotations

import argparse
import torch

from src.ml.reconv_lib import MultiPathTransformer
from src.ml.reconv_ds import ReconvergentPathsDataset, reconv_collate
from src.ml.train_reconv import policy_loss_and_metrics


@torch.no_grad()
def evaluate(model: MultiPathTransformer, loader) -> tuple[float, float, float]:
    """Evaluate average policy loss, reward, and accuracy over the dataset.
    
    Accuracy is the percentage of path pairs that successfully justify to the
    required output (zero constraint violations).
    """
    model.eval()
    total_loss = 0.0
    total_reward = 0.0
    total_correct = 0
    total_samples = 0
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
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--dataset', type=str, required=True)
    ap.add_argument('--embedding-dim', type=int, default=128)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--num-encoder-layers', type=int, default=1)
    ap.add_argument('--num-interaction-layers', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    device = torch.device('cpu') if args.cpu or (not torch.cuda.is_available()) else torch.device('cuda')

    ds = ReconvergentPathsDataset(args.dataset, device=device)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=reconv_collate)

    # Model uses embedding_dim + 1 to account for logic value feature
    model = MultiPathTransformer(
        embedding_dim=args.embedding_dim + 1,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_interaction_layers=args.num_interaction_layers,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    if 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)

    loss, avg_reward, accuracy = evaluate(model, dl)
    print(f"EVAL | loss={loss:.4f} avg_reward={avg_reward:.4f} accuracy={accuracy:.2%}")


if __name__ == '__main__':
    main()
