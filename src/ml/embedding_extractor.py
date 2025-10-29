"""
Embedding extraction module that properly handles AIG conversion and gate mappings.
"""

import os
import shutil
import torch
from typing import Tuple, Dict, List

from src.util.aig import bench_to_aig_file
from src.ml.gcn import bench_to_embed
from src.util.io import parse_bench_file
from src.util.struct import Gate

class EmbeddingExtractor:
    """Extracts embeddings with proper AIG conversion and gate mapping handling."""
    
    def __init__(self, staging_dir: str = "data/staging"):
        self.staging_dir = staging_dir
        self.aig_path = os.path.join(staging_dir, "circuit.bench")
        self._ensure_staging_dir()
    
    def _ensure_staging_dir(self):
        """Ensure staging directory exists."""
        os.makedirs(self.staging_dir, exist_ok=True)
    
    def _clean_staging_dir(self):
        """Clean staging directory."""
        if os.path.exists(self.staging_dir):
            for filename in os.listdir(self.staging_dir):
                file_path = os.path.join(self.staging_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
    
    def extract_embeddings(self, circuit_path: str) -> Tuple[torch.Tensor, torch.Tensor, Dict, List[Gate]]:
        """
        Extract embeddings with AIG conversion and return gate mappings.
        
        Args:
            circuit_path: Path to the original circuit file
            
        Returns:
            struct_emb: Structural embeddings
            func_emb: Functional embeddings  
            gate_mapping: Mapping from original to AIG gate IDs
            original_circuit: Original circuit gates
        """
        # Ensure staging directory exists
        self._ensure_staging_dir()
        
        # Clean staging directory
        self._clean_staging_dir()
        
        # Copy circuit to staging
        shutil.copy(circuit_path, self.staging_dir)
        
        # Convert to AIG and get mapping
        aig_circuit, gate_mapping = bench_to_aig_file(circuit_path, self.aig_path)
        
        # Verify AIG file exists before trying to read it
        if not os.path.exists(self.aig_path):
            raise FileNotFoundError(f"AIG file not created: {self.aig_path}")
        
        # Extract embeddings from AIG circuit
        try:
            struct_emb, func_emb = bench_to_embed(self.aig_path)
        except Exception as e:
            print(f"[ERROR] Failed to extract embeddings from {self.aig_path}: {e}")
            # Return dummy embeddings with correct shape
            struct_emb = torch.zeros(1, 128, device='cuda' if torch.cuda.is_available() else 'cpu')
            func_emb = torch.zeros(1, 128, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parse original circuit for reference
        original_circuit, _ = parse_bench_file(circuit_path)
        
        return struct_emb, func_emb, gate_mapping, original_circuit
    
    def get_input_output_info(self, original_circuit: List[Gate]) -> Tuple[List[Gate], List[Gate]]:
        """Get input and output gates from original circuit."""
        input_gates = [g for g in original_circuit if g.type == 1 and g.nfi == 0]
        output_gates = [g for g in original_circuit if g.type != 0 and g.nfo == 0]
        return input_gates, output_gates
    
    def map_predictions_to_original(self, struct_emb: torch.Tensor, func_emb: torch.Tensor, gate_mapping: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map predictions back to original circuit using gate mapping.
        
        Args:
            struct_emb: Structural embeddings from AIG
            func_emb: Functional embeddings from AIG
            gate_mapping: Mapping from original to AIG gate IDs
            
        Returns:
            Mapped structural and functional embeddings
        """
        mapped_struct_emb = []
        mapped_func_emb = []

        for _, aig_id in gate_mapping.items():
            aig_id = int(aig_id)
            mapped_struct_emb.append(struct_emb[aig_id])
            mapped_func_emb.append(func_emb[aig_id])

        return torch.stack(mapped_struct_emb), torch.stack(mapped_func_emb)


if __name__ == "__main__":
    extractor = EmbeddingExtractor()
    struct_emb, func_emb, gate_mapping, original_circuit = extractor.extract_embeddings("data/bench/c17.bench")

    print(f"Structural Embeddings: {struct_emb.shape} -> {struct_emb}")
    print(f"Functional Embeddings: {func_emb.shape} -> {func_emb}")
    print(f"Gate Mapping: {gate_mapping}")


    mapped_struct_emb, mapped_func_emb = extractor.map_predictions_to_original(struct_emb, func_emb, gate_mapping)
    input_gates, output_gates = extractor.get_input_output_info(original_circuit)
    print(f"Mapped Structural Embeddings: {mapped_struct_emb.shape} -> {mapped_struct_emb}")
    print(f"Mapped Functional Embeddings: {mapped_func_emb.shape} -> {mapped_func_emb}")
    print(f"Input Gates: {[g.name for g in input_gates]}")
    print(f"Output Gates: {[g.name for g in output_gates]}")