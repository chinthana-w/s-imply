
import torch
import os
import argparse
from torch.utils.data import DataLoader
from src.ml.reconv_ds import ReconvergentPathsDataset, reconv_collate
from src.ml.train_reconv import TrainConfig, MultiPathTransformer, _debug_metrics_from_logits, _load_circuit
from src.ml.reconv_lib import MultiPathTransformer

def inspect_failures(dataset_path, checkpoint_path, batch_size=1, num_workers=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = ReconvergentPathsDataset(
        dataset_path,
        device=torch.device('cpu'),
        prefer_value=1,
        anchor_in_dataset=True
    )
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=reconv_collate,
        num_workers=num_workers
    )
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    # Infer dims from dataset if possible, or use defaults
    # For now, use defaults from TrainConfig but we need to match what was trained.
    # The training log said "Observed embedding dimension from batch: 132"
    cfg = TrainConfig(dataset=dataset_path, output="checkpoints/debug_inspect")
    cfg.embedding_dim = 132 # Set to observed value
    
    model = MultiPathTransformer(
        input_dim=cfg.embedding_dim,
        model_dim=cfg.model_dim,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_interaction_layers=cfg.num_interaction_layers,
        dim_feedforward=cfg.dim_feedforward
    ).to(device)
    
    # Load weights if available
    try:
        model_path = os.path.join(checkpoint_path, "best_model.pth")
        if os.path.exists(model_path):
            # The saved checkpoint might be a dict with state_dict
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if saved with DataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            print("Model loaded successfully.")
        else:
            print(f"Warning: Checkpoint {model_path} not found. Using random weights.")
    except Exception as e:
        print(f"Error loading model: {e}")

    model.eval()
    
    failures = []
    successes = []
    
    print("Inspecting samples...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            paths = batch['paths_emb'].to(device)
            masks = batch['attn_mask'].to(device)
            node_ids = batch['node_ids'].to(device)
            files = batch['files']
            
            # Forward pass
            logits = model(paths, masks)
            
            # Get metrics
            dbg = _debug_metrics_from_logits(
                logits, node_ids, masks, files,
                anchor_p=batch.get('anchor_p').to(device) if 'anchor_p' in batch else None,
                anchor_l=batch.get('anchor_l').to(device) if 'anchor_l' in batch else None,
                anchor_v=batch.get('anchor_v').to(device) if 'anchor_v' in batch else None,
            )
            
            # Check edge_acc
            edge_acc = dbg['edge_acc']
            reconv_acc = dbg['reconv_match_rate']
            
            # Note: For strict correctness, we should include reconv check, 
            # but inspect_failures uses edge_acc only for classification of 'failure'.
            # We will keep it consistent for now but log the stats.
            valid = (edge_acc == 1.0)
            
            sample_info = {
                'index': i,
                'file': files[0],
                'edge_acc': edge_acc,
                'reconv_match_rate': dbg['reconv_match_rate'],
                'edges_per_sample': dbg['edges_per_sample'],
                'gate_types': [] # Will populate below
            }
            
            # Analyze gate types involved
            # We need to look at the node_ids and the circuit file
            try:
                circuit = _load_circuit(files[0])
                nid_b = node_ids[0]
                ids_b = nid_b[nid_b > 0].unique().tolist()
                gtypes = []
                for nid in ids_b:
                    if int(nid) < len(circuit):
                         gtypes.append(circuit[int(nid)].type)
                sample_info['gate_types'] = gtypes
            except Exception as e:
                sample_info['error'] = str(e)

            if valid:
                successes.append(sample_info)
            else:
                failures.append(sample_info)
                
            if i % 10 == 0:
                curr_total = len(successes) + len(failures)
                if curr_total > 0:
                    curr_acc = len(successes) / curr_total
                    curr_edge = sum(x['edge_acc'] for x in successes+failures) / curr_total
                    curr_reconv = sum(x['reconv_match_rate'] for x in successes+failures) / curr_total
                    print(f"Processed {i} batches ({curr_total} samples)... Acc: {curr_acc:.4f} Edge: {curr_edge:.4f} Reconv: {curr_reconv:.4f}")
                    
                    if len(failures) > 0:
                         gcounts = {}
                         for f in failures:
                             for gt in f['gate_types']:
                                 gcounts[gt] = gcounts.get(gt, 0) + 1
                         top_g = sorted(gcounts.items(), key=lambda x: x[1], reverse=True)[:3]
                         print(f"  Top Fail Gates: {top_g}")

    print(f"\nTotal Samples: {len(dataset)}")
    print(f"Failures: {len(failures)}")
    print(f"Successes: {len(successes)}")

    if len(dataset) > 0:
        avg_edge_acc = sum(x['edge_acc'] for x in (successes+failures)) / len(dataset)
        avg_reconv = sum(x['reconv_match_rate'] for x in (successes+failures)) / len(dataset)
        print(f"Avg Edge Acc: {avg_edge_acc:.4f}")
        print(f"Avg Reconv Match: {avg_reconv:.4f}")
    
    # Analyze failures
    print("\n--- Failure Analysis ---")
    gate_type_counts = {}
    for f in failures:
        for gt in f['gate_types']:
            gate_type_counts[gt] = gate_type_counts.get(gt, 0) + 1
            
    print("Common Gate Types in Failures:")
    for gt, count in sorted(gate_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"Type {gt}: {count}")

    # Print first few failures details
    print("\nExample Failures:")
    for f in failures[:5]:
        print(f"Index: {f['index']}, Edge Acc: {f['edge_acc']:.2f}, Gates: {f['gate_types']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Path to dataset pickle')
    parser.add_argument('checkpoint', help='Path to checkpoint directory')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for inspection')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    args = parser.parse_args()
    
    inspect_failures(args.dataset, args.checkpoint, batch_size=args.batch_size, num_workers=args.num_workers)
