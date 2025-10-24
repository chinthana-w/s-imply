"""Quick test of dataset building and training setup."""

import os
import sys

# Test 1: Import modules
print("Test 1: Importing modules...")
try:
    from src.atpg.reconv_podem import build_dataset, save_dataset, load_dataset
    from src.ml.reconv_rl_trainer import ReconvRLTrainer
    print("  ✓ All imports successful")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Build small dataset
print("\nTest 2: Building small dataset...")
try:
    # Use c17.bench for quick test
    test_bench_dir = "data/bench/ISCAS85"
    test_output = "data/datasets/test_dataset.pkl"
    
    # Build dataset with only 5 samples per file
    dataset = build_dataset(test_bench_dir, max_samples_per_file=5)
    
    if len(dataset) > 0:
        print(f"  ✓ Built dataset with {len(dataset)} entries")
        
        # Show first entry structure
        print("\n  Sample entry structure:")
        entry = dataset[0]
        print(f"    - file: {os.path.basename(entry['file'])}")
        print(f"    - start: {entry['info']['start']}")
        print(f"    - reconv: {entry['info']['reconv']}")
        print(f"    - num_paths: {len(entry['info']['paths'])}")
        print(f"    - justification_1 gates: {len(entry['justification_1'])}")
        print(f"    - justification_0 gates: {len(entry['justification_0'])}")
    else:
        print("  ✗ Dataset is empty")
        sys.exit(1)
        
except Exception as e:
    print(f"  ✗ Dataset building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Save and load dataset
print("\nTest 3: Saving and loading dataset...")
try:
    save_dataset(dataset, test_output)
    loaded_dataset = load_dataset(test_output)
    
    if len(loaded_dataset) == len(dataset):
        print(f"  ✓ Dataset saved and loaded successfully ({len(loaded_dataset)} entries)")
    else:
        print(f"  ✗ Dataset size mismatch: {len(loaded_dataset)} vs {len(dataset)}")
        sys.exit(1)
        
except Exception as e:
    print(f"  ✗ Save/load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Initialize trainer
print("\nTest 4: Initializing trainer...")
try:
    trainer = ReconvRLTrainer(
        embedding_dim=64,  # Smaller for testing
        nhead=4,
        num_encoder_layers=2,
        num_interaction_layers=1,
        learning_rate=1e-4
    )
    print("  ✓ Trainer initialized successfully")
    print(f"    - Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"    - Device: {trainer.device}")
    
except Exception as e:
    print(f"  ✗ Trainer initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Prepare a small batch
print("\nTest 5: Preparing batch...")
try:
    # Take just 2 entries for testing
    test_entries = dataset[:2]
    
    path_emb, attn_mask, targets = trainer.prepare_batch(test_entries, target_value=1)
    
    print(f"  ✓ Batch prepared successfully")
    print(f"    - Path embeddings shape: {path_emb.shape}")
    print(f"    - Attention mask shape: {attn_mask.shape}")
    print(f"    - Number of targets: {len(targets)}")
    
except Exception as e:
    print(f"  ✗ Batch preparation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Forward pass
print("\nTest 6: Testing forward pass...")
try:
    trainer.model.eval()
    import torch
    with torch.no_grad():
        predictions = trainer.model(path_emb, attn_mask)
    
    print(f"  ✓ Forward pass successful")
    print(f"    - Predictions shape: {predictions.shape}")
    
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Clean up test file
print("\nCleaning up...")
if os.path.exists(test_output):
    os.remove(test_output)
    print("  ✓ Test dataset removed")

print("\n" + "="*60)
print("ALL TESTS PASSED! ✓")
print("="*60)
print("\nYou can now:")
print("  1. Build full dataset: python train_reconv.py build")
print("  2. Train model: python train_reconv.py train")
print("  3. Or do both: python train_reconv.py both")
