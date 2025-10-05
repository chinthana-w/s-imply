"""
Circuit filtering utilities to identify and avoid problematic circuits.
"""

import os
import glob
from typing import List, Tuple


def test_circuit_with_deepgate(circuit_path: str) -> bool:
    """Test if a circuit can be parsed by DeepGate without errors."""
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        from src.ml.gcn import bench_to_embed
        from src.util.aig import bench_to_aig_file
        import tempfile
        import shutil
        
        # Create temporary AIG file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bench', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Convert to AIG
            bench_to_aig_file(circuit_path, tmp_path)
            
            # Test DeepGate parsing
            struct_emb, func_emb = bench_to_embed(tmp_path)
            
            # Check if embeddings are valid
            if struct_emb.shape[0] == 0 or func_emb.shape[0] == 0:
                return False
                
            return True
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        print(f"[WARNING] Circuit {circuit_path} failed DeepGate test: {e}")
        return False


def filter_working_circuits(circuit_paths: List[str], max_test: int = 50) -> Tuple[List[str], List[str]]:
    """
    Filter circuits to only include those that work with DeepGate.
    
    Args:
        circuit_paths: List of circuit file paths
        max_test: Maximum number of circuits to test (for efficiency)
        
    Returns:
        Tuple of (working_circuits, failed_circuits)
    """
    working_circuits = []
    failed_circuits = []
    
    # Test a subset first to avoid long delays
    test_paths = circuit_paths[:max_test] if len(circuit_paths) > max_test else circuit_paths
    
    print(f"Testing {len(test_paths)} circuits for DeepGate compatibility...")
    
    for i, circuit_path in enumerate(test_paths):
        print(f"Testing circuit {i+1}/{len(test_paths)}: {os.path.basename(circuit_path)}")
        
        if test_circuit_with_deepgate(circuit_path):
            working_circuits.append(circuit_path)
            print(f"  ✓ Working")
        else:
            failed_circuits.append(circuit_path)
            print(f"  ✗ Failed")
    
    # If we tested a subset and found working circuits, assume the rest work too
    if len(test_paths) < len(circuit_paths) and working_circuits:
        remaining_circuits = circuit_paths[max_test:]
        working_circuits.extend(remaining_circuits)
        print(f"Added {len(remaining_circuits)} untested circuits (assuming they work)")
    
    print(f"Filtering complete: {len(working_circuits)} working, {len(failed_circuits)} failed")
    return working_circuits, failed_circuits


def get_filtered_circuits(train_dir: str, test_dir: str, max_test: int = 50) -> Tuple[List[str], List[str]]:
    """
    Get filtered circuits for training and testing.
    
    Args:
        train_dir: Directory containing training circuits
        test_dir: Directory containing test circuits
        max_test: Maximum number of circuits to test
        
    Returns:
        Tuple of (train_circuits, test_circuits)
    """
    # Get all circuit files
    train_circuits = glob.glob(os.path.join(train_dir, "*.bench"))
    test_circuits = glob.glob(os.path.join(test_dir, "*.bench"))
    
    print(f"Found {len(train_circuits)} training circuits and {len(test_circuits)} test circuits")
    
    # Filter training circuits
    working_train, failed_train = filter_working_circuits(train_circuits, max_test)
    
    # Filter test circuits (test fewer since they're smaller)
    working_test, failed_test = filter_working_circuits(test_circuits, min(max_test, 20))
    
    return working_train, working_test


if __name__ == "__main__":
    # Test the filtering
    train_circuits, test_circuits = get_filtered_circuits(
        "data/bench/RCCG", 
        "data/bench/arbitrary",
        max_test=10
    )
    
    print(f"\nFinal results:")
    print(f"Working training circuits: {len(train_circuits)}")
    print(f"Working test circuits: {len(test_circuits)}")
