"""
Training Module for Reverse Circuit Simulator

This module provides training utilities for the transformer-based reverse circuit simulator,
including data loading, training loops, and evaluation metrics.
"""

import os
import glob
import random
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from src.ml.reverse_simulator import ReverseCircuitTransformer, ReverseSimulatorLoss
from src.ml.training_env import ReverseSimulatorEnv
from src.ml.embedding_extractor import EmbeddingExtractor


class CircuitDataset(Dataset):
    """Dataset for loading circuit embeddings and generating training pairs."""
    
    def __init__(
        self, 
        circuit_paths: List[str], 
        embedding_dim: int = 128,
        max_inputs: int = 100,
        num_samples_per_circuit: int = 10,
        device: str = 'cpu'  # Force CPU due to CUDA compatibility issues
    ):
        self.circuit_paths = circuit_paths
        self.embedding_dim = embedding_dim
        self.max_inputs = max_inputs
        self.num_samples_per_circuit = num_samples_per_circuit
        self.device = device
        
        # Pre-compute embeddings for all circuits
        self.circuit_data = []
        self.embedding_extractor = EmbeddingExtractor()
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Pre-compute embeddings for all circuits."""
        print("Pre-computing circuit embeddings...")
        
        for circuit_path in tqdm(self.circuit_paths):
            try:
                # Extract embeddings with AIG conversion
                struct_emb, func_emb, gate_mapping, circuit_gates = self.embedding_extractor.extract_embeddings(circuit_path)
                
                # Find inputs and outputs
                input_gates = [g for g in circuit_gates if g.type == 1 and g.nfi == 0]
                output_gates = [g for g in circuit_gates if g.type != 0 and g.nfo == 0]
                
                num_inputs = len(input_gates)
                num_outputs = len(output_gates)
                
                if num_inputs > 0 and num_outputs > 0:
                    # Extract input embeddings
                    input_embeddings = func_emb[:num_inputs]  # [num_inputs, embedding_dim]
                    
                    # Ensure input_embeddings is exactly max_inputs size
                    if num_inputs < self.max_inputs:
                        padding = torch.zeros(
                            self.max_inputs - num_inputs, 
                            self.embedding_dim,
                            device=self.device
                        )
                        input_embeddings = torch.cat([input_embeddings, padding], dim=0)
                    elif num_inputs > self.max_inputs:
                        # Truncate if too many inputs
                        input_embeddings = input_embeddings[:self.max_inputs]
                    
                    # Ensure input_embeddings is exactly [max_inputs, embedding_dim]
                    if input_embeddings.shape[0] != self.max_inputs:
                        if input_embeddings.shape[0] < self.max_inputs:
                            padding = torch.zeros(
                                self.max_inputs - input_embeddings.shape[0], 
                                self.embedding_dim,
                                device=self.device
                            )
                            input_embeddings = torch.cat([input_embeddings, padding], dim=0)
                        else:
                            input_embeddings = input_embeddings[:self.max_inputs]
                    
                    # Extract output embedding (use last output)
                    output_embedding = func_emb[num_inputs:num_inputs+num_outputs][-1]  # [embedding_dim]
                    
                    self.circuit_data.append({
                        'path': circuit_path,
                        'input_embeddings': input_embeddings,
                        'output_embedding': output_embedding,
                        'num_inputs': num_inputs,
                        'circuit_gates': circuit_gates
                    })
                    
            except Exception as e:
                print(f"Error processing {circuit_path}: {e}")
                continue
        
        print(f"Successfully processed {len(self.circuit_data)} circuits")
    
    def __len__(self):
        return len(self.circuit_data) * self.num_samples_per_circuit
    
    def __getitem__(self, idx):
        circuit_idx = idx // self.num_samples_per_circuit
        
        circuit_data = self.circuit_data[circuit_idx]
        
        # Generate random desired output
        desired_output = torch.tensor([random.randint(0, 1)], dtype=torch.float32, device=self.device)
        
        # For supervised learning, we need ground truth input patterns
        # For now, generate random patterns (in practice, you'd want to solve this)
        num_inputs = circuit_data['num_inputs']
        if num_inputs > 0:
            # Generate random input pattern
            input_pattern = torch.randint(0, 2, (num_inputs,), device=self.device).float()
            
            # Ensure input_pattern is exactly max_inputs size
            if num_inputs < self.max_inputs:
                padding = torch.zeros(self.max_inputs - num_inputs, device=self.device)
                input_pattern = torch.cat([input_pattern, padding], dim=0)
            elif num_inputs > self.max_inputs:
                # Truncate if too many inputs
                input_pattern = input_pattern[:self.max_inputs]
        else:
            input_pattern = torch.zeros(self.max_inputs, device=self.device)
        
        # Ensure input_pattern is exactly max_inputs size
        if input_pattern.shape[0] != self.max_inputs:
            if input_pattern.shape[0] < self.max_inputs:
                padding = torch.zeros(self.max_inputs - input_pattern.shape[0], device=self.device)
                input_pattern = torch.cat([input_pattern, padding], dim=0)
            else:
                input_pattern = input_pattern[:self.max_inputs]
        
        return {
            'input_embeddings': circuit_data['input_embeddings'],
            'output_embedding': circuit_data['output_embedding'],
            'desired_output': desired_output,
            'target_pattern': input_pattern,
            'num_inputs': num_inputs,
            'circuit_path': circuit_data['path']
        }


class ReverseSimulatorTrainer:
    """Trainer class for the reverse circuit simulator."""
    
    def __init__(
        self,
        model: ReverseCircuitTransformer,
        circuit_pool: List[str],
        embedding_dim: int = 128,
        max_inputs: int = 100,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        device: str = 'cpu',  # Force CPU due to CUDA compatibility issues
        save_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)  # Move model to device
        self.circuit_pool = circuit_pool
        self.embedding_dim = embedding_dim
        self.max_inputs = max_inputs
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize optimizer and loss
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.loss_fn = ReverseSimulatorLoss()
        
        # Create training environment
        self.env = ReverseSimulatorEnv(
            circuit_pool=circuit_pool,
            embedding_dim=embedding_dim,
            max_inputs=max_inputs,
            device=device
        )
        
        # Create embedding extractor for the environment
        self.env.embedding_extractor = EmbeddingExtractor()
        
        # Training metrics
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'simulation_success_rate': []
        }
    
    def train_supervised(
        self, 
        num_epochs: int = 100,
        val_split: float = 0.2,
        batch_size: int = 32
    ):
        """Train the model using supervised learning with generated data."""
        
        # Create dataset
        dataset = CircuitDataset(
            circuit_paths=self.circuit_pool,
            embedding_dim=self.embedding_dim,
            max_inputs=self.max_inputs,
            device=self.device
        )
        
        # Split dataset
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Move batch to device and detach to prevent graph reuse
                input_embeddings = batch['input_embeddings'].to(self.device).detach()
                output_embedding = batch['output_embedding'].to(self.device).detach()
                desired_output = batch['desired_output'].to(self.device).detach()
                target_pattern = batch['target_pattern'].to(self.device).detach()
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(input_embeddings, output_embedding, desired_output)
                
                # Compute loss
                loss = self.loss_fn(predictions, target_pattern)
                
                # Backward pass
                loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                
                # Calculate accuracy (only for actual inputs)
                batch_size_actual = input_embeddings.size(0)
                for i in range(batch_size_actual):
                    num_inputs = batch['num_inputs'][i].item()
                    if num_inputs > 0:
                        pred_binary = (predictions[i, :num_inputs] > 0.5).float()
                        target_binary = target_pattern[i, :num_inputs]
                        train_correct += (pred_binary == target_binary).all().item()
                        train_total += 1
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_embeddings = batch['input_embeddings'].to(self.device).detach()
                    output_embedding = batch['output_embedding'].to(self.device).detach()
                    desired_output = batch['desired_output'].to(self.device).detach()
                    target_pattern = batch['target_pattern'].to(self.device).detach()
                    
                    predictions = self.model(input_embeddings, output_embedding, desired_output)
                    loss = self.loss_fn(predictions, target_pattern)
                    
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    batch_size_actual = input_embeddings.size(0)
                    for i in range(batch_size_actual):
                        num_inputs = batch['num_inputs'][i].item()
                        if num_inputs > 0:
                            pred_binary = (predictions[i, :num_inputs] > 0.5).float()
                            target_binary = target_pattern[i, :num_inputs]
                            val_correct += (pred_binary == target_binary).all().item()
                            val_total += 1
            
            # Calculate average metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = train_correct / train_total if train_total > 0 else 0.0
            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
            
            # Update learning rate
            self.scheduler.step(avg_val_loss)
            
            # Update history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['accuracy'].append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
    
    def train_reinforcement(self, num_episodes: int = 1000):
        """Train the model using reinforcement learning with circuit simulation."""
        
        print(f"Starting RL training for {num_episodes} episodes")
        
        episode_rewards = []
        episode_successes = []
        
        for episode in tqdm(range(num_episodes)):
            # Reset environment
            obs, info = self.env.reset()
            
            # Get model prediction
            action = self.env.predict_action(obs)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Store episode results
            episode_rewards.append(reward)
            episode_successes.append(info['simulation_success'])
            
            # Update model (simplified RL update)
            if episode % 10 == 0:  # Update every 10 episodes
                self._update_model_rl(obs, action, reward)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                success_rate = np.mean(episode_successes[-100:])
                print(f"Episode {episode}: Avg Reward: {avg_reward:.3f}, Success Rate: {success_rate:.3f}")
        
        # Update training history
        self.training_history['simulation_success_rate'].extend(episode_successes)
    
    def _update_model_rl(self, obs: Dict, action: np.ndarray, reward: float):
        """Update model using reinforcement learning."""
        self.model.train()
        
        # Convert observation to tensors and detach to prevent graph reuse
        input_embeddings = torch.tensor(
            obs['input_embeddings'], 
            dtype=torch.float32, 
            device=self.device,
            requires_grad=False
        )
        
        # Ensure input_embeddings has the correct shape [batch_size, max_inputs, embedding_dim]
        if input_embeddings.dim() == 2:
            input_embeddings = input_embeddings.unsqueeze(0)  # Add batch dimension
        elif input_embeddings.dim() == 1:
            # If it's 1D, reshape to [1, max_inputs, embedding_dim]
            input_embeddings = input_embeddings.view(1, -1, self.embedding_dim)
        
        output_embedding = torch.tensor(
            obs['output_embedding'], 
            dtype=torch.float32, 
            device=self.device,
            requires_grad=False
        ).unsqueeze(0)
        
        desired_output = torch.tensor(
            obs['desired_output'], 
            dtype=torch.float32, 
            device=self.device,
            requires_grad=False
        ).unsqueeze(0)
        
        # Get model prediction
        prediction = self.model(input_embeddings, output_embedding, desired_output)
        
        # Convert action to tensor
        target = torch.tensor(action, dtype=torch.float32, device=self.device, requires_grad=False).unsqueeze(0)
        
        # Compute loss with simulation reward
        simulation_reward = torch.tensor([reward], dtype=torch.float32, device=self.device, requires_grad=False)
        loss = self.loss_fn(prediction, target, simulation_reward)
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
        else:
            torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint['training_history']
        return checkpoint['epoch']
    
    def evaluate(self, test_circuits: List[str], num_tests: int = 100) -> Dict[str, float]:
        """Evaluate the model on test circuits."""
        self.model.eval()
        
        success_count = 0
        total_tests = 0
        
        with torch.no_grad():
            for _ in range(num_tests):
                # Select random test circuit
                circuit_path = random.choice(test_circuits)
                
                try:
                    # Load circuit and get embeddings using AIG conversion
                    struct_emb, func_emb, gate_mapping, circuit_gates = self.env.embedding_extractor.extract_embeddings(circuit_path)
                    
                    # Find inputs and outputs
                    input_gates = [g for g in circuit_gates if g.type == 1 and g.nfi == 0]
                    output_gates = [g for g in circuit_gates if g.type != 0 and g.nfo == 0]
                    
                    if len(input_gates) > 0 and len(output_gates) > 0:
                        num_inputs = len(input_gates)
                        
                        # Prepare input
                        input_embeddings = func_emb[:num_inputs]
                        if num_inputs < self.max_inputs:
                            padding = torch.zeros(
                                self.max_inputs - num_inputs, 
                                self.embedding_dim,
                                device=self.device
                            )
                            input_embeddings = torch.cat([input_embeddings, padding], dim=0)
                        
                        output_embedding = func_emb[num_inputs:num_inputs+len(output_gates)][-1]
                        
                        # Generate random desired output
                        desired_output = torch.tensor([[random.randint(0, 1)]], 
                                                    dtype=torch.float32, device=self.device)
                        
                        # Get prediction
                        input_embeddings = input_embeddings.unsqueeze(0)
                        output_embedding = output_embedding.unsqueeze(0)
                        
                        prediction = self.model(input_embeddings, output_embedding, desired_output)
                        binary_prediction = (prediction > 0.5).float()
                        
                        # Simulate circuit
                        input_pattern = binary_prediction[0, :num_inputs].cpu().numpy().astype(int)
                        reward, success = self.env._simulate_circuit(input_pattern)
                        
                        if success:
                            success_count += 1
                        total_tests += 1
                        
                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    continue
        
        success_rate = success_count / total_tests if total_tests > 0 else 0.0
        
        return {
            'success_rate': success_rate,
            'total_tests': total_tests,
            'successful_tests': success_count
        }


def main():
    """Main training script."""
    # Get circuit pool
    circuit_pool = glob.glob("/home/local1/chinthana/s-imply/data/bench/arbitrary/*.bench")
    circuit_pool.extend(glob.glob("/home/local1/chinthana/s-imply/data/bench/ISCAS85/*.bench"))
    
    if not circuit_pool:
        print("No circuits found!")
        return
    
    print(f"Found {len(circuit_pool)} circuits")
    
    # Create model
    model = ReverseCircuitTransformer(
        embedding_dim=128,
        d_model=256,
        nhead=8,
        num_layers=6
    )
    
    # Create trainer
    trainer = ReverseSimulatorTrainer(
        model=model,
        circuit_pool=circuit_pool,
        learning_rate=1e-4,
        batch_size=16,
        save_dir='data/weights'
    )
    
    # Train model
    print("Starting supervised training...")
    trainer.train_supervised(num_epochs=50, batch_size=16)
    
    print("Starting reinforcement learning...")
    trainer.train_reinforcement(num_episodes=500)
    
    # Save final model
    print("Saving final model...")
    trainer.save_checkpoint(trainer.training_history['epoch'][-1] if trainer.training_history['epoch'] else 0, is_best=True)
    
    # Evaluate model
    print("Evaluating model...")
    test_circuits = circuit_pool[:5]  # Use first 5 circuits for testing
    results = trainer.evaluate(test_circuits, num_tests=50)
    print(f"Evaluation results: {results}")


if __name__ == "__main__":
    main()
