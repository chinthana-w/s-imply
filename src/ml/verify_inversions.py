
import torch
import os
import argparse
from torch.utils.data import DataLoader
from src.ml.reconv_ds import ReconvergentPathsDataset, reconv_collate
from src.ml.train_reconv import TrainConfig, MultiPathTransformer, _load_circuit, resolve_gate_types
from src.util.struct import GateType

def verify_inversions(dataset_path, checkpoint_path, num_samples=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = ReconvergentPathsDataset(
        dataset_path,
        device=torch.device('cpu'), # Load to CPU initially
        prefer_value=1,
        anchor_in_dataset=True,
    )
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=reconv_collate,
        num_workers=0
    )
    
    # Auto-detect input dimension
    probe_batch = next(iter(loader))
    input_dim = probe_batch['paths_emb'].shape[-1]
    print(f"Detected input dimension: {input_dim}")

    # Load model configuration
    cfg = TrainConfig(dataset=dataset_path, output="checkpoints/debug")
    
    # Check if checkpoint exists
    model_path = os.path.join(checkpoint_path, "best_model.pth")
    if not os.path.exists(model_path):
            model_path = os.path.join(checkpoint_path, "checkpoint_epoch_5.pth")
    
    model = MultiPathTransformer(
        input_dim=input_dim,
        model_dim=cfg.model_dim,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_interaction_layers=cfg.num_interaction_layers,
        dim_feedforward=cfg.dim_feedforward
    ).to(device)
    
    # Load weights
    try:

        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Using random weights.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    inversion_checks = 0
    inversion_failures = 0
    buffer_checks = 0
    buffer_failures = 0
    
    # Per-gate stats
    stats = {
        GateType.NOT: {'checks': 0, 'fails': 0},
        GateType.NAND: {'checks': 0, 'fails': 0},
        GateType.NOR: {'checks': 0, 'fails': 0},
        GateType.BUFF: {'checks': 0, 'fails': 0},
        GateType.AND: {'checks': 0, 'fails': 0},
        GateType.OR: {'checks': 0, 'fails': 0},
    }
    
    print("\n--- Verifying Logic Gates ---")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if inversion_checks + buffer_checks >= num_samples:
                break
                
            paths_emb = batch['paths_emb'].to(device)
            masks = batch['attn_mask'].to(device)
            node_ids = batch['node_ids'].to(device)
            files = batch['files']
            
            files = batch['files']
            
            gtypes = resolve_gate_types(node_ids, files, device)
            logits = model(paths_emb, masks, gate_types=gtypes)
            actions = logits.argmax(dim=-1) # Deterministic check
            
            b = 0
            file_path = files[b]
            circuit = _load_circuit(file_path)
            
            P, L = node_ids.shape[1], node_ids.shape[2]
            
            for p in range(P):
                valid_len = int(masks[b, p].sum().item())
                if valid_len == 0: continue
                
                prev_val = -1
                
                for l in range(valid_len):
                    nid = int(node_ids[b, p, l].item())
                    val = int(actions[b, p, l].item())
                    
                    if l > 0:
                        # Check strict gate logic
                        gate_type = circuit[nid].type
                        
                        if gate_type in stats:
                            stats[gate_type]['checks'] += 1
                            
                        # Inverters (NOT, NOR, NAND)
                        if gate_type in [GateType.NOT, GateType.NOR, GateType.NAND]:
                            if gate_type == GateType.NOT:
                                inversion_checks += 1
                                if val == prev_val: 
                                    inversion_failures += 1
                                    stats[GateType.NOT]['fails'] += 1
                                    
                            elif gate_type == GateType.NOR:
                                if prev_val == 1:
                                    inversion_checks += 1
                                    if val == 1: 
                                        inversion_failures += 1
                                        stats[GateType.NOR]['fails'] += 1
                                        
                            elif gate_type == GateType.NAND:
                                if prev_val == 0:
                                    inversion_checks += 1
                                    if val == 0: 
                                        inversion_failures += 1
                                        stats[GateType.NAND]['fails'] += 1
                        
                        # Buffers (BUFF, OR, AND)
                        elif gate_type in [GateType.BUFF, GateType.OR, GateType.AND]:
                            if gate_type == GateType.BUFF:
                                buffer_checks += 1
                                if val != prev_val:
                                    buffer_failures += 1
                                    stats[GateType.BUFF]['fails'] += 1
                            elif gate_type == GateType.OR:
                                if prev_val == 1:
                                    buffer_checks += 1
                                    if val == 0:
                                        buffer_failures += 1
                                        stats[GateType.OR]['fails'] += 1
                            elif gate_type == GateType.AND:
                                if prev_val == 0:
                                    buffer_checks += 1
                                    if val == 1:
                                        buffer_failures += 1
                                        stats[GateType.AND]['fails'] += 1

                    prev_val = val

    print(f"\n--- Results ---")
    print(f"Inversion Logic Checks (NOT/NOR/NAND): {inversion_checks}")
    print(f"Inversion Failures: {inversion_failures} ({inversion_failures/max(1, inversion_checks)*100:.2f}%)")
    print(f"Buffer Logic Checks (BUFF/OR/AND): {buffer_checks}")
    print(f"Buffer Failures: {buffer_failures} ({buffer_failures/max(1, buffer_checks)*100:.2f}%)")
    
    print("\n--- Breakdown ---")
    for gt, s in stats.items():
        if s['checks'] > 0:
            print(f"{gt.name}: {s['fails']}/{s['checks']} ({s['fails']/s['checks']*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Path to dataset pickle')
    parser.add_argument('checkpoint', help='Path to checkpoint directory')
    parser.add_argument('--num-samples', type=int, default=1000, help='Total gate checks to perform')
    args = parser.parse_args()
    
    verify_inversions(args.dataset, args.checkpoint, num_samples=args.num_samples)
