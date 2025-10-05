"""
Training Environment for Reverse Circuit Simulator

This module provides a reinforcement learning environment that integrates
the transformer model with circuit simulation for training.
"""

import random
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
import gymnasium as gym
from gymnasium import spaces

from src.ml.reverse_simulator import ReverseCircuitTransformer
from src.atpg.logic_sim_three import logic_sim, reset_gates
from src.util.struct import LogicValue
from src.util.io import parse_bench_file


class ReverseSimulatorEnv(gym.Env):
    """
    Gym environment for training the reverse circuit simulator.
    
    The environment:
    1. Loads a random circuit from a pool
    2. Extracts functional embeddings using DeepGate
    3. Provides the transformer with input/output embeddings
    4. Simulates predicted patterns and provides rewards
    """
    
    def __init__(
        self,
        circuit_pool: List[str],
        embedding_dim: int = 128,
        max_inputs: int = 100,
        reward_correct: float = 1.0,
        reward_incorrect: float = -1.0,
        device: str = 'cpu'  # Force CPU to avoid CUDA compatibility issues
    ):
        super().__init__()
        
        self.circuit_pool = circuit_pool
        self.embedding_dim = embedding_dim
        self.max_inputs = max_inputs
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.device = device
        
        # Initialize the transformer model
        self.model = ReverseCircuitTransformer(
            embedding_dim=embedding_dim,
            max_inputs=max_inputs
        ).to(device)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(max_inputs,), 
            dtype=np.float32
        )
        
        # Observation space: input embeddings + output embedding + desired output
        self.observation_space = spaces.Dict({
            'input_embeddings': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_inputs, embedding_dim),
                dtype=np.float32
            ),
            'output_embedding': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(embedding_dim,),
                dtype=np.float32
            ),
            'desired_output': spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            'num_inputs': spaces.Discrete(max_inputs + 1)
        })
        
        # Current episode state
        self.current_circuit = None
        self.current_circuit_gates = None
        self.current_input_embeddings = None
        self.current_output_embedding = None
        self.current_desired_output = None
        self.current_num_inputs = 0
        self.episode_count = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment with a new random circuit."""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Select random circuit
        circuit_path = random.choice(self.circuit_pool)
        self._load_circuit(circuit_path)
        
        # Generate random desired output
        self.current_desired_output = torch.tensor(
            [[random.randint(0, 1)]], 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Create observation
        observation = {
            'input_embeddings': self.current_input_embeddings.detach().cpu().numpy(),
            'output_embedding': self.current_output_embedding.detach().cpu().numpy(),
            'desired_output': self.current_desired_output.detach().cpu().numpy(),
            'num_inputs': self.current_num_inputs
        }
        
        info = {
            'circuit_path': circuit_path,
            'desired_output': self.current_desired_output.item()
        }
        
        self.episode_count += 1
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Predicted input pattern [max_inputs]
        
        Returns:
            observation: Next observation
            reward: Reward for the action
            terminated: Whether episode is terminated
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Convert action to binary pattern
        binary_pattern = (action > 0.5).astype(int)
        
        # Simulate the circuit with predicted pattern
        reward, simulation_success = self._simulate_circuit(binary_pattern)
        
        # Create next observation (same circuit, new desired output)
        self.current_desired_output = torch.tensor(
            [[random.randint(0, 1)]], 
            dtype=torch.float32, 
            device=self.device
        )
        
        observation = {
            'input_embeddings': self.current_input_embeddings.detach().cpu().numpy(),
            'output_embedding': self.current_output_embedding.detach().cpu().numpy(),
            'desired_output': self.current_desired_output.detach().cpu().numpy(),
            'num_inputs': self.current_num_inputs
        }
        
        info = {
            'simulation_success': simulation_success,
            'predicted_pattern': binary_pattern[:self.current_num_inputs].tolist(),
            'desired_output': self.current_desired_output.item()
        }
        
        # Episode ends after one step (single prediction task)
        terminated = True
        truncated = False
        
        return observation, reward, terminated, truncated, info
    
    def _load_circuit(self, circuit_path: str) -> None:
        """Load a circuit and extract embeddings."""
        try:
            # Extract embeddings using AIG conversion
            if hasattr(self, 'embedding_extractor'):
                struct_emb, func_emb, gate_mapping, circuit_gates = self.embedding_extractor.extract_embeddings(circuit_path)
            else:
                # Fallback to direct embedding extraction
                from src.ml.gcn import bench_to_embed
                struct_emb, func_emb = bench_to_embed(circuit_path)
                circuit_gates, max_node_id = parse_bench_file(circuit_path)
            
            # Find input and output gates
            input_gates = [g for g in circuit_gates if g.type == 1 and g.nfi == 0]  # Primary inputs
            output_gates = [g for g in circuit_gates if g.type != 0 and g.nfo == 0]  # Primary outputs
            
            self.current_num_inputs = len(input_gates)
            self.current_circuit_gates = circuit_gates
            
            # Extract input embeddings (first num_inputs from func_emb)
            if self.current_num_inputs > 0:
                input_embeddings = func_emb[:self.current_num_inputs]  # [num_inputs, embedding_dim]
                
                # Pad to max_inputs if necessary
                if self.current_num_inputs < self.max_inputs:
                    padding = torch.zeros(
                        self.max_inputs - self.current_num_inputs, 
                        self.embedding_dim,
                        device=self.device
                    )
                    input_embeddings = torch.cat([input_embeddings, padding], dim=0)
                
                self.current_input_embeddings = input_embeddings.unsqueeze(0)  # [1, max_inputs, embedding_dim]
            else:
                self.current_input_embeddings = torch.zeros(
                    1, self.max_inputs, self.embedding_dim, 
                    device=self.device
                )
            
            # Extract output embedding (last output gate)
            if len(output_gates) > 0:
                # Use the last output gate's embedding
                output_idx = len(func_emb) - 1
                self.current_output_embedding = func_emb[output_idx:output_idx+1]  # [1, embedding_dim]
            else:
                self.current_output_embedding = torch.zeros(
                    1, self.embedding_dim, 
                    device=self.device
                )
            
        except Exception as e:
            print(f"Error loading circuit {circuit_path}: {e}")
            # Fallback to dummy data
            self.current_num_inputs = 4
            self.current_input_embeddings = torch.randn(
                1, self.max_inputs, self.embedding_dim, 
                device=self.device
            )
            self.current_output_embedding = torch.randn(
                1, self.embedding_dim, 
                device=self.device
            )
            self.current_circuit_gates = []
    
    def _simulate_circuit(self, input_pattern: np.ndarray) -> Tuple[float, bool]:
        """
        Simulate the circuit with the given input pattern.
        
        Args:
            input_pattern: Binary input pattern [num_inputs]
        
        Returns:
            reward: Reward value
            success: Whether simulation produced correct output
        """
        if self.current_circuit_gates is None or len(self.current_circuit_gates) == 0:
            return self.reward_incorrect, False
        
        try:
            # Reset circuit gates
            reset_gates(self.current_circuit_gates, len(self.current_circuit_gates) - 1)
            
            # Set input values
            input_gates = [g for g in self.current_circuit_gates if g.type == 1 and g.nfi == 0]
            for i, gate in enumerate(input_gates):
                if i < len(input_pattern) and i < self.current_num_inputs:
                    gate.val = int(input_pattern[i])
                else:
                    gate.val = LogicValue.XD
            
            # Run logic simulation
            logic_sim(self.current_circuit_gates, len(self.current_circuit_gates) - 1)
            
            # Get output value
            output_gates = [g for g in self.current_circuit_gates if g.type != 0 and g.nfo == 0]
            if len(output_gates) > 0:
                actual_output = output_gates[-1].val  # Use last output
                desired_output = int(self.current_desired_output.item())
                
                # Check if simulation was successful
                success = (actual_output == desired_output)
                reward = self.reward_correct if success else self.reward_incorrect
                
                return reward, success
            else:
                return self.reward_incorrect, False
                
        except Exception as e:
            print(f"Simulation error: {e}")
            return self.reward_incorrect, False
    
    def get_model(self) -> ReverseCircuitTransformer:
        """Get the transformer model for training."""
        return self.model
    
    def predict_action(self, observation: Dict) -> np.ndarray:
        """
        Use the model to predict an action given an observation.
        
        Args:
            observation: Environment observation
        
        Returns:
            action: Predicted input pattern
        """
        with torch.no_grad():
            # Convert observation to tensors
            input_embeddings = torch.tensor(
                observation['input_embeddings'], 
                dtype=torch.float32, 
                device=self.device
            )
            
            # Ensure input_embeddings has the correct shape [batch_size, max_inputs, embedding_dim]
            if input_embeddings.dim() == 2:
                input_embeddings = input_embeddings.unsqueeze(0)  # Add batch dimension
            elif input_embeddings.dim() == 1:
                # If it's 1D, reshape to [1, max_inputs, embedding_dim]
                input_embeddings = input_embeddings.view(1, -1, self.embedding_dim)
            
            output_embedding = torch.tensor(
                observation['output_embedding'], 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(0)
            
            desired_output = torch.tensor(
                observation['desired_output'], 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(0)
            
            # Get model prediction
            prediction = self.model(input_embeddings, output_embedding, desired_output)
            
            return prediction.detach().cpu().numpy().squeeze(0)


def create_training_env(
    circuit_pool: List[str],
    embedding_dim: int = 128,
    max_inputs: int = 100,
    device: str = 'cpu'  # Force CPU to avoid CUDA compatibility issues
) -> ReverseSimulatorEnv:
    """Factory function to create a training environment."""
    
    return ReverseSimulatorEnv(
        circuit_pool=circuit_pool,
        embedding_dim=embedding_dim,
        max_inputs=max_inputs,
        device=device
    )


if __name__ == "__main__":
    # Test the environment
    import glob
    
    # Get circuit pool
    circuit_pool = glob.glob("/home/local1/chinthana/s-imply/data/bench/arbitrary/*.bench")
    
    if circuit_pool:
        env = create_training_env(circuit_pool)
        
        # Test reset
        obs, info = env.reset()
        print(f"Observation keys: {obs.keys()}")
        print(f"Number of inputs: {obs['num_inputs']}")
        print(f"Circuit path: {info['circuit_path']}")
        
        # Test step
        action = np.random.rand(env.max_inputs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}")
        print(f"Simulation success: {info['simulation_success']}")
        print(f"Predicted pattern: {info['predicted_pattern']}")
    else:
        print("No circuits found in the pool!")
