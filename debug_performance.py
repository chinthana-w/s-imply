
import torch
import os
import sys
from src.ml.reconv_lib import MultiPathTransformer
from src.ml.train_reconv import _debug_metrics_from_logits, TrainConfig, resolve_gate_types, _pair_constraint_ok
from src.ml.reconv_ds import reconv_collate
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def debug_failures():
    # Configuration
    dataset_path = "data/datasets/shards_anchored/shard_00010.pt"
    if not os.path.exists(dataset_path):
        print(f"Dataset shard not found: {dataset_path}")
        return

    # Load a single shard explicitly
    print(f"Loading shard: {dataset_path}")
    shard = torch.load(dataset_path, map_location='cpu')
    
    # We'll treat the shard as a dataset of size N
    # Construct a batch manually
    # Let's take more examples to find interesting ones
    batch_size = 200
    batch_items = []
    
    paths_emb = shard['paths_emb']
    attn_mask = shard['attn_mask']
    node_ids = shard['node_ids']
    files = shard['files']
    
    # Anchors
    anchor_p = shard.get('anchor_p')
    anchor_l = shard.get('anchor_l')
    anchor_v = shard.get('anchor_v')
    solvability = shard.get('solvability')

    print(f"Shard contains {paths_emb.shape[0]} samples.")

    for i in range(min(batch_size, paths_emb.shape[0])):
        item = {
            'paths_emb': paths_emb[i],
            'attn_mask': attn_mask[i],
            'node_ids': node_ids[i],
            'file': files[i]
        }
        if anchor_p is not None:
            item['anchor_p'] = anchor_p[i]
            item['anchor_l'] = anchor_l[i]
            item['anchor_v'] = anchor_v[i]
            item['solvability'] = solvability[i]
        batch_items.append(item)
        
    # Collate
    batch = reconv_collate(batch_items)
    
    # Prepare inputs
    # Force CPU to avoid CUDA errors on this environment
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    b_paths = batch['paths_emb'].to(device)
    b_masks = batch['attn_mask'].to(device)
    b_node_ids = batch['node_ids'].to(device)
    # files is a list
    b_files = batch['files']
    
    b_anchor_p = batch.get('anchor_p', None)
    b_anchor_l = batch.get('anchor_l', None)
    b_anchor_v = batch.get('anchor_v', None)
    b_solvability = batch.get('solvability', None)

    if b_anchor_p is not None: b_anchor_p = b_anchor_p.to(device)
    if b_anchor_l is not None: b_anchor_l = b_anchor_l.to(device)
    if b_anchor_v is not None: b_anchor_v = b_anchor_v.to(device)
    if b_solvability is not None: b_solvability = b_solvability.to(device)
    
    # Initialize Model (Random Weights)
    # We want to see how a random model performs vs trained, or just behavior.
    # But ideally we load a checkpoint.
    # Let's check for checkpoints.
    checkpoint_dir = "checkpoints/debug_run"
    model_path = os.path.join(checkpoint_dir, "checkpoint_epoch_1.pth")
    
    input_dim = b_paths.shape[-1]
    
    model_args = {
        'input_dim': input_dim,
        'model_dim': 512,
        'nhead': 4,
        'num_encoder_layers': 1,
        'num_interaction_layers': 1,
        'dim_feedforward': 512
    }
    
    state_dict = None
    
    if os.path.exists(model_path):
        print(f"Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
             if 'config' in checkpoint:
                 cfg = checkpoint['config']
                 print(f"  Loaded config from checkpoint: {cfg}")
                 # Map TrainConfig/dict to model args
                 # The checkpoint config might be a TrainConfig object or dict
                 # We need to extract what MultiPathTransformer needs.
                 # We assume keys match or we interpret them.
                 
                 # Helper to get value
                 def get_cfg(k, default):
                     if isinstance(cfg, dict): return cfg.get(k, default)
                     return getattr(cfg, k, default)
                 
                 model_args['model_dim'] = get_cfg('model_dim', 512)
                 model_args['nhead'] = get_cfg('nhead', 4)
                 model_args['num_encoder_layers'] = get_cfg('num_encoder_layers', 1)
                 model_args['num_interaction_layers'] = get_cfg('num_interaction_layers', 1)
                 model_args['dim_feedforward'] = get_cfg('dim_feedforward', 512)

             if 'state_dict' in checkpoint:
                 state_dict = checkpoint['state_dict']
             else:
                 # valid if checkpoint is just state dict, but we handled that check above
                 state_dict = checkpoint
        else:
            state_dict = checkpoint
            
    print(f"Initializing model with args: {model_args}")
    model = MultiPathTransformer(**model_args).to(device)
    
    if state_dict is not None:
        # Handle prefix 'model.' if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        try:
            model.load_state_dict(new_state_dict)
        except RuntimeError as e:
            print(f"Warning: partial load or mismatch: {e}")
            model.load_state_dict(new_state_dict, strict=False)
            
    elif not os.path.exists(model_path):
        print("No checkpoint found. Using initialized weights.")

    model.eval()
    
    # Get Gate Types for logic checking
    b_gate_types = resolve_gate_types(b_node_ids, b_files, device)

    # Forward
    with torch.no_grad():
        logits, solv_logits = model(b_paths, b_masks, gate_types=b_gate_types)
        
    # Metrics
    metrics = _debug_metrics_from_logits(
        logits, b_node_ids, b_masks, b_files,
        anchor_p=b_anchor_p, anchor_l=b_anchor_l, anchor_v=b_anchor_v,
        solvability_logits=solv_logits, solvability_labels=b_solvability
    )
    
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Run on all batch items in the full shard (via manual batching if needed, or just iterate)
    # We loaded 200 samples into 'batch', but we want to check ALL 5000 in shard if needed.
    # To do this efficiently, we should iterate over the tensor in chunks.
    
    print("\n--- Scanning FULL SHARD for Failures & Non-Trivial Cases ---")
    
    # We will iterate in chunks of 100
    chunk_size = 100
    total_samples = paths_emb.shape[0]
    
    found_failure = False
    found_inverter = False
    found_one_anchor = False
    found_unsat = False
    
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        
        # Prepare chunk
        c_paths = paths_emb[start_idx:end_idx].to(device)
        c_masks = attn_mask[start_idx:end_idx].to(device)
        c_nodes = node_ids[start_idx:end_idx].to(device)
        c_files = files[start_idx:end_idx]
        
        c_ap = anchor_p[start_idx:end_idx].to(device) if anchor_p is not None else None
        c_al = anchor_l[start_idx:end_idx].to(device) if anchor_l is not None else None
        c_av = anchor_v[start_idx:end_idx].to(device) if anchor_v is not None else None
        c_sv = solvability[start_idx:end_idx].to(device) if solvability is not None else None
        
        # Collate is handled by just slicing tensors (already padded in shard)
        # But we need to add gate types
        c_gtypes = resolve_gate_types(c_nodes, c_files, device)
        
        # PAD embedding dimension if needed to match model
        # Model input_dim (from config) vs Current dim
        curr_dim = c_paths.shape[-1]
        target_dim = model.input_dim
        if curr_dim < target_dim:
            pad_amt = target_dim - curr_dim
            # Pad last dim
            c_paths = torch.nn.functional.pad(c_paths, (0, pad_amt))
        
        with torch.no_grad():
             logits, solv_logits = model(c_paths, c_masks, gate_types=c_gtypes)
             
        actions_chunk = torch.argmax(logits, dim=-1)
        pred_solv_chunk = torch.argmax(solv_logits, dim=-1)
        
        for i in range(end_idx - start_idx):
            global_idx = start_idx + i
            actions = actions_chunk[i]
            p_masks = c_masks[i]
            g_types = c_gtypes[i]
            n_ids = c_nodes[i]
            
            # Check Inverters
            has_inv = (g_types == 2).any() or (g_types == 3).any() or (g_types == 4).any()
            if has_inv and not found_inverter:
                found_inverter = True
                print(f"[INFO] Found Inverter at index {global_idx}")
            
            # Check Anchor
            anchor_ok = True
            is_one_anchor = False
            if c_ap is not None:
                ap = c_ap[i].item()
                if ap >= 0:
                    al = c_al[i].item()
                    av = c_av[i].item()
                    if av == 1:
                        if not found_one_anchor:
                             found_one_anchor = True
                             print(f"[INFO] Found Anchor=1 at index {global_idx}")
                        is_one_anchor = True
                        
                    pred_v = actions[ap, al].item()
                    if pred_v != av:
                        # Only report mismatch if meaningful (not just trivial 0 vs 0 mismatch?) 
                        # No, any anchor mismatch is bad.
                        anchor_ok = False
            
            # Check Solvability
            if c_sv is not None:
                tru = c_sv[i].item()
                pred = pred_solv_chunk[i].item()
                if tru == 1:
                    if not found_unsat:
                        found_unsat = True
                        print(f"[INFO] Found UNSAT at index {global_idx}")
                    if pred != tru:
                         # Missed UNSAT
                         pass 
                         
            # FAILURE CONDITION
            if not anchor_ok:
                print(f"\n[FAILURE] Sample {global_idx}: Anchor Mismatch")
                print(f"  File: {c_files[i]}")
                print(f"  Anchor: Path {ap} at {al} = {av}, Pred = {pred_v}")
                found_failure = True
                # Print Path if short
                valid_len = p_masks[ap].sum().item()
                if valid_len < 10:
                    for l in range(valid_len):
                        print(f"    {l}: Node {n_ids[ap,l].item()} Type {g_types[ap,l].item()} Val {actions[ap,l].item()}")
                break # Found one failure details, iterate to locate others/stop?
                
            if is_one_anchor:
                # Always interesting to see if it gets logic 1 right
                if anchor_ok:
                     pass # It worked!
                else: 
                     pass # Handled above
                     
        if found_failure and found_inverter and found_one_anchor and found_unsat:
             break # Found examples of everything
             
    if not found_failure:
        print("\nNo anchor failures found.")
    if not found_one_anchor:
        print("No Anchor=1 cases found.")
    if not found_inverter:
        print("No Inverter cases found.")


if __name__ == "__main__":
    debug_failures()
