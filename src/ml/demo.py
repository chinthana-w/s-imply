"""
Demo script for Reverse Circuit Simulator

This script demonstrates the trained model's ability to predict input patterns
that produce desired outputs for given circuits.
"""

import os
import glob
import torch
import random
import numpy as np
from typing import List, Dict

from src.ml.reverse_simulator import ReverseCircuitTransformer
from src.ml.embedding_extractor import EmbeddingExtractor
from src.util.struct import GateType


class CircuitDemo:
    """Demo class for showcasing reverse circuit simulation predictions."""
    
    def __init__(self, model_path: str = "data/weights/best_model.pth"):
        """Initialize the demo with a trained model."""
        self.device = 'cpu'  # Use CPU for demo
        self.embedding_dim = 128
        self.max_inputs = 100
        
        # Load the trained model
        self.model = self._load_model(model_path)
        self.embedding_extractor = EmbeddingExtractor()
        
        print(f"✅ Loaded model from {model_path}")
    
    def _load_model(self, model_path: str) -> ReverseCircuitTransformer:
        """Load the trained model from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Create model architecture
        model = ReverseCircuitTransformer(
            embedding_dim=self.embedding_dim,
            d_model=256,
            nhead=8,
            num_layers=6
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model.to(self.device)
    
    def _format_gate_type(self, gate_type: int) -> str:
        """Convert gate type enum to readable string."""
        type_map = {
            GateType.INPT: "INPUT",
            GateType.FROM: "FROM", 
            GateType.BUFF: "BUFFER",
            GateType.NOT: "NOT",
            GateType.AND: "AND",
            GateType.NAND: "NAND",
            GateType.OR: "OR",
            GateType.NOR: "NOR",
            GateType.XOR: "XOR",
            GateType.XNOR: "XNOR"
        }
        return type_map.get(gate_type, f"UNKNOWN({gate_type})")
    
    def _print_circuit_structure(self, circuit_gates: List, circuit_path: str):
        """Print a formatted view of the circuit structure."""
        print(f"\n{'='*80}")
        print(f"📋 CIRCUIT STRUCTURE: {os.path.basename(circuit_path)}")
        print(f"{'='*80}")
        
        # Group gates by type
        input_gates = [g for g in circuit_gates if g.type == GateType.INPT and g.nfi == 0]
        output_gates = [g for g in circuit_gates if g.type != GateType.INPT and g.nfo == 0]
        logic_gates = [g for g in circuit_gates if g.type not in [GateType.INPT, GateType.FROM] and g.nfo > 0]
        
        print(f"\n🔌 INPUTS ({len(input_gates)}):")
        for gate in input_gates:
            print(f"  • {gate.name}: {self._format_gate_type(gate.type)}")
        
        print(f"\n🔌 OUTPUTS ({len(output_gates)}):")
        for gate in output_gates:
            print(f"  • {gate.name}: {self._format_gate_type(gate.type)}")
        
        print(f"\n⚡ LOGIC GATES ({len(logic_gates)}):")
        for gate in logic_gates[:10]:  # Show first 10 logic gates
            fanin_str = ", ".join(map(str, gate.fin)) if gate.fin else "None"
            fanout_str = ", ".join(map(str, gate.fot)) if gate.fot else "None"
            print(f"  • {gate.name}: {self._format_gate_type(gate.type)} (fanin: [{fanin_str}], fanout: [{fanout_str}])")
        
        if len(logic_gates) > 10:
            print(f"  ... and {len(logic_gates) - 10} more logic gates")
        
        # Show circuit logic for simple circuits
        self._print_circuit_logic(circuit_gates, circuit_path)
    
    def _print_circuit_logic(self, circuit_gates: List, circuit_path: str):
        """Print the logical expression of the circuit."""
        print("\n🧮 CIRCUIT LOGIC:")
        
        # Read the original bench file to show the logic
        try:
            with open(circuit_path, 'r') as f:
                lines = f.readlines()
            
            # Find the logic lines (skip comments and INPUT/OUTPUT declarations)
            logic_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('INPUT') and not line.startswith('OUTPUT'):
                    logic_lines.append(line)
            
            if logic_lines:
                print("  Logic expressions:")
                for line in logic_lines:
                    print(f"    {line}")
            else:
                print("  No logic expressions found")
                
        except Exception as e:
            print(f"  Could not read circuit file: {e}")
    
    def _print_prediction_results(self, desired_output: int, predicted_pattern: np.ndarray, 
                                input_gates: List, success: bool):
        """Print the prediction results in a formatted way."""
        print("\n🎯 PREDICTION RESULTS:")
        print(f"{'='*50}")
        print(f"Desired Output: {desired_output}")
        print(f"Success: {'✅ YES' if success else '❌ NO'}")
        
        print("\n📊 PREDICTED INPUT PATTERN:")
        print(f"{'Input':<10} {'Predicted':<12} {'Binary':<8}")
        print(f"{'-'*30}")
        
        for i, gate in enumerate(input_gates):
            pred_val = predicted_pattern[i]
            binary_val = 1 if pred_val > 0.5 else 0
            print(f"{gate.name:<10} {pred_val:<12.4f} {binary_val:<8}")
    
    def predict_for_circuit(self, circuit_path: str, desired_output: int = None) -> Dict:
        """Make a prediction for a given circuit and desired output."""
        print(f"\n🔍 ANALYZING CIRCUIT: {os.path.basename(circuit_path)}")
        
        try:
            # Extract embeddings with AIG conversion
            struct_emb, func_emb, gate_mapping, circuit_gates = self.embedding_extractor.extract_embeddings(circuit_path)
            
            # Find inputs and outputs
            input_gates = [g for g in circuit_gates if g.type == GateType.INPT and g.nfi == 0]
            output_gates = [g for g in circuit_gates if g.type != GateType.INPT and g.nfo == 0]
            
            if len(input_gates) == 0 or len(output_gates) == 0:
                print("❌ Circuit has no inputs or outputs")
                return None
            
            num_inputs = len(input_gates)
            
            # Generate random desired output if not provided
            if desired_output is None:
                desired_output = random.randint(0, 1)
            
            # Prepare input embeddings
            input_embeddings = func_emb[:num_inputs]
            if num_inputs < self.max_inputs:
                padding = torch.zeros(self.max_inputs - num_inputs, self.embedding_dim, device=self.device)
                input_embeddings = torch.cat([input_embeddings, padding], dim=0)
            elif num_inputs > self.max_inputs:
                input_embeddings = input_embeddings[:self.max_inputs]
            
            # Prepare output embedding (use last output)
            output_embedding = func_emb[num_inputs:num_inputs+len(output_gates)][-1]
            
            # Prepare desired output tensor
            desired_output_tensor = torch.tensor([[desired_output]], dtype=torch.float32, device=self.device)
            
            # Make prediction
            with torch.no_grad():
                input_embeddings = input_embeddings.unsqueeze(0)  # Add batch dimension
                output_embedding = output_embedding.unsqueeze(0)
                
                prediction = self.model(input_embeddings, output_embedding, desired_output_tensor)
                predicted_pattern = prediction[0, :num_inputs].cpu().numpy()
            
            # Convert to binary pattern
            binary_pattern = (predicted_pattern > 0.5).astype(int)
            
            # Simulate the circuit to check if prediction is correct
            success = self._simulate_circuit(circuit_gates, binary_pattern, desired_output)
            
            # Print results
            self._print_circuit_structure(circuit_gates, circuit_path)
            self._print_prediction_results(desired_output, predicted_pattern, input_gates, success)
            
            return {
                'circuit_path': circuit_path,
                'desired_output': desired_output,
                'predicted_pattern': predicted_pattern,
                'binary_pattern': binary_pattern,
                'success': success,
                'num_inputs': num_inputs,
                'num_outputs': len(output_gates)
            }
            
        except Exception as e:
            print(f"❌ Error processing circuit: {e}")
            return None
    
    def _simulate_circuit(self, circuit_gates: List, input_pattern: np.ndarray, desired_output: int) -> bool:
        """Simple circuit simulation to verify the prediction."""
        try:
            # Create a mapping of gate values
            gate_values = {}
            
            # Set input values
            input_gates = [g for g in circuit_gates if g.type == GateType.INPT and g.nfi == 0]
            for i, gate in enumerate(input_gates):
                gate_values[gate.name] = input_pattern[i]
            
            # Simulate logic gates (simplified)
            for gate in circuit_gates:
                if gate.type == GateType.INPT:
                    continue
                
                if gate.type == GateType.NOT:
                    if gate.fin and len(gate.fin) > 0:
                        input_val = gate_values.get(str(gate.fin[0]), 0)
                        gate_values[gate.name] = 1 - input_val
                
                elif gate.type == GateType.AND:
                    if gate.fin and len(gate.fin) >= 2:
                        val1 = gate_values.get(str(gate.fin[0]), 0)
                        val2 = gate_values.get(str(gate.fin[1]), 0)
                        gate_values[gate.name] = val1 and val2
                
                elif gate.type == GateType.OR:
                    if gate.fin and len(gate.fin) >= 2:
                        val1 = gate_values.get(str(gate.fin[0]), 0)
                        val2 = gate_values.get(str(gate.fin[1]), 0)
                        gate_values[gate.name] = val1 or val2
                
                elif gate.type == GateType.XOR:
                    if gate.fin and len(gate.fin) >= 2:
                        val1 = gate_values.get(str(gate.fin[0]), 0)
                        val2 = gate_values.get(str(gate.fin[1]), 0)
                        gate_values[gate.name] = val1 ^ val2
            
            # Check output
            output_gates = [g for g in circuit_gates if g.type != GateType.INPT and g.nfo == 0]
            if output_gates:
                output_value = gate_values.get(output_gates[-1].name, 0)
                return output_value == desired_output
            
            return False
            
        except Exception as e:
            print(f"Simulation error: {e}")
            return False
    
    def run_demo(self, use_simple_circuits: bool = True):
        """Run demo predictions on simple circuits."""
        print("\n🚀 REVERSE CIRCUIT SIMULATOR DEMO")
        print(f"{'='*80}")
        
        if use_simple_circuits:
            # Use specific simple circuits for clear demonstration
            demo_circuits = [
                "data/bench/arbitrary/single_and.bench",
                "data/bench/arbitrary/single_not_and1.bench", 
                "data/bench/arbitrary/single_nand.bench"
            ]
            print("Using simple circuits for clear demonstration...")
        else:
            # Get available circuits
            circuit_pool = glob.glob("data/bench/arbitrary/*.bench")
            circuit_pool.extend(glob.glob("data/bench/ISCAS85/*.bench"))
            
            if not circuit_pool:
                print("❌ No circuits found!")
                return
            
            # Select random circuits for demo
            demo_circuits = random.sample(circuit_pool, min(3, len(circuit_pool)))
            print(f"Running {len(demo_circuits)} demonstration predictions...")
        
        results = []
        for i, circuit_path in enumerate(demo_circuits, 1):
            if not os.path.exists(circuit_path):
                print(f"⚠️  Circuit not found: {circuit_path}")
                continue
                
            print(f"\n{'='*80}")
            print(f"DEMO {i}/{len(demo_circuits)}")
            print(f"{'='*80}")
            
            result = self.predict_for_circuit(circuit_path)
            if result:
                results.append(result)
        
        # Summary
        print(f"\n{'='*80}")
        print("📈 DEMO SUMMARY")
        print(f"{'='*80}")
        print(f"Total circuits tested: {len(results)}")
        if results:
            successful = sum(1 for r in results if r['success'])
            print(f"Successful predictions: {successful}")
            print(f"Success rate: {successful / len(results) * 100:.1f}%")
        else:
            print("No circuits were successfully processed.")


def main():
    """Main demo function."""
    try:
        # Create demo instance
        demo = CircuitDemo()
        
        # Run demo with simple circuits
        demo.run_demo(use_simple_circuits=True)
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please train the model first by running: python -m src.ml.trainer")
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    main()
