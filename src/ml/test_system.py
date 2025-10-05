"""
Test Script for Reverse Circuit Simulator System

This script tests the complete system to ensure all components work together correctly.
"""

import os
import sys
import torch
import numpy as np
import glob

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml.reverse_simulator import create_model
from src.ml.training_env import create_training_env
from src.ml.trainer import ReverseSimulatorTrainer
from src.ml.gcn import bench_to_embed
from src.util.io import parse_bench_file


def test_model_creation():
    """Test that the transformer model can be created and run."""
    print("Testing model creation...")
    
    try:
        # Force CPU to avoid CUDA compatibility issues
        device = torch.device('cpu')
        model = create_model(device=device)
        
        # Test forward pass with dummy data
        batch_size = 2
        num_inputs = 4
        embedding_dim = 128
        max_inputs = 100
        
        # Create input embeddings and pad to max_inputs
        input_embeddings = torch.randn(batch_size, num_inputs, embedding_dim).to(device)
        if num_inputs < max_inputs:
            padding = torch.zeros(batch_size, max_inputs - num_inputs, embedding_dim).to(device)
            input_embeddings = torch.cat([input_embeddings, padding], dim=1)
        
        output_embedding = torch.randn(batch_size, embedding_dim).to(device)
        desired_output = torch.randint(0, 2, (batch_size, 1)).float().to(device)
        
        with torch.no_grad():
            prediction = model(input_embeddings, output_embedding, desired_output)
        
        assert prediction.shape == (batch_size, 100), f"Expected shape ({batch_size}, 100), got {prediction.shape}"
        assert torch.all((prediction >= 0) & (prediction <= 1)), "Predictions should be in [0, 1]"
        
        print("✓ Model creation and forward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_embedding_extraction():
    """Test that embeddings can be extracted from circuits."""
    print("Testing embedding extraction...")
    
    try:
        # Find a test circuit
        test_circuits = glob.glob("/home/local1/chinthana/s-imply/data/bench/arbitrary/*.bench")
        
        if not test_circuits:
            print("✗ No test circuits found")
            return False
        
        circuit_path = test_circuits[0]
        print(f"Testing with circuit: {circuit_path}")
        
        # Extract embeddings
        struct_emb, func_emb = bench_to_embed(circuit_path)
        
        # Parse circuit
        circuit_gates, max_node_id = parse_bench_file(circuit_path)
        input_gates = [g for g in circuit_gates if g.type == 1 and g.nfi == 0]
        output_gates = [g for g in circuit_gates if g.type != 0 and g.nfo == 0]
        
        print(f"  Circuit has {len(input_gates)} inputs and {len(output_gates)} outputs")
        print(f"  Structural embedding shape: {struct_emb.shape}")
        print(f"  Functional embedding shape: {func_emb.shape}")
        
        assert func_emb.shape[1] == 128, f"Expected embedding dim 128, got {func_emb.shape[1]}"
        assert len(input_gates) > 0, "Circuit should have at least one input"
        assert len(output_gates) > 0, "Circuit should have at least one output"
        
        print("✓ Embedding extraction successful")
        return True
        
    except Exception as e:
        print(f"✗ Embedding extraction failed: {e}")
        return False


def test_training_environment():
    """Test that the training environment works."""
    print("Testing training environment...")
    
    try:
        # Get circuit pool
        circuit_pool = glob.glob("/home/local1/chinthana/s-imply/data/bench/arbitrary/*.bench")
        
        if not circuit_pool:
            print("✗ No circuits found for environment")
            return False
        
        # Create environment
        env = create_training_env(circuit_pool)
        
        # Test reset
        obs, info = env.reset()
        
        assert 'input_embeddings' in obs, "Observation should contain input_embeddings"
        assert 'output_embedding' in obs, "Observation should contain output_embedding"
        assert 'desired_output' in obs, "Observation should contain desired_output"
        assert 'num_inputs' in obs, "Observation should contain num_inputs"
        
        print(f"  Circuit: {info['circuit_path']}")
        print(f"  Number of inputs: {obs['num_inputs']}")
        print(f"  Desired output: {info['desired_output']}")
        
        # Test step
        action = np.random.rand(env.max_inputs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert terminated, "Episode should terminate after one step"
        assert 'simulation_success' in info, "Info should contain simulation_success"
        
        print(f"  Reward: {reward}")
        print(f"  Simulation success: {info['simulation_success']}")
        
        print("✓ Training environment successful")
        return True
        
    except Exception as e:
        print(f"✗ Training environment failed: {e}")
        return False


def test_end_to_end_prediction():
    """Test complete end-to-end prediction pipeline."""
    print("Testing end-to-end prediction...")
    
    try:
        # Get a test circuit
        test_circuits = glob.glob("/home/local1/chinthana/s-imply/data/bench/arbitrary/*.bench")
        
        if not test_circuits:
            print("✗ No test circuits found")
            return False
        
        circuit_path = test_circuits[0]
        device = torch.device('cpu')  # Force CPU to avoid CUDA issues
        
        # Extract embeddings
        struct_emb, func_emb = bench_to_embed(circuit_path)
        
        # Parse circuit
        circuit_gates, max_node_id = parse_bench_file(circuit_path)
        input_gates = [g for g in circuit_gates if g.type == 1 and g.nfi == 0]
        output_gates = [g for g in circuit_gates if g.type != 0 and g.nfo == 0]
        
        num_inputs = len(input_gates)
        max_inputs = 100
        
        # Prepare model inputs
        input_embeddings = func_emb[:num_inputs]  # [num_inputs, 128]
        
        # Pad to max_inputs
        if num_inputs < max_inputs:
            padding = torch.zeros(max_inputs - num_inputs, 128, device=device)
            input_embeddings = torch.cat([input_embeddings, padding], dim=0)
        
        input_embeddings = input_embeddings.unsqueeze(0)  # [1, max_inputs, 128]
        
        # Output embedding
        output_embedding = func_emb[num_inputs:num_inputs+len(output_gates)][-1].unsqueeze(0)  # [1, 128]
        
        # Create model and predict
        model = create_model(device=device)
        
        for desired_output_val in [0, 1]:
            desired_output = torch.tensor([[desired_output_val]], dtype=torch.float32, device=device)
            
            with torch.no_grad():
                prediction = model(input_embeddings, output_embedding, desired_output)
                binary_prediction = (prediction > 0.5).float()
            
            print(f"  Desired output: {desired_output_val}")
            print(f"  Predicted pattern: {binary_prediction[0, :num_inputs].cpu().numpy()}")
            print(f"  Confidence: {prediction[0, :num_inputs].cpu().numpy()}")
        
        print("✓ End-to-end prediction successful")
        return True
        
    except Exception as e:
        print(f"✗ End-to-end prediction failed: {e}")
        return False


def test_trainer_creation():
    """Test that the trainer can be created."""
    print("Testing trainer creation...")
    
    try:
        # Get circuit pool
        circuit_pool = glob.glob("/home/local1/chinthana/s-imply/data/bench/arbitrary/*.bench")
        
        if not circuit_pool:
            print("✗ No circuits found for trainer")
            return False
        
        # Create model and trainer
        device = torch.device('cpu')  # Force CPU to avoid CUDA issues
        model = create_model(device=device)
        
        trainer = ReverseSimulatorTrainer(
            model=model,
            circuit_pool=circuit_pool,
            learning_rate=1e-4,
            batch_size=4,
            save_dir='test_checkpoints'
        )
        
        assert trainer.model is not None, "Trainer should have a model"
        assert trainer.env is not None, "Trainer should have an environment"
        assert trainer.optimizer is not None, "Trainer should have an optimizer"
        
        print("✓ Trainer creation successful")
        return True
        
    except Exception as e:
        print(f"✗ Trainer creation failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("REVERSE CIRCUIT SIMULATOR - SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        test_model_creation,
        test_embedding_extraction,
        test_training_environment,
        test_end_to_end_prediction,
        test_trainer_creation
    ]
    
    results = []
    for test in tests:
        print()
        result = test()
        results.append(result)
        print("-" * 40)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
