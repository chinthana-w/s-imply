"""Hybrid RL + Supervised trainer for reconvergent path justification.

This trainer uses a combination of:
1. Supervised learning: Train on known correct justifications
2. Reinforcement learning: Learn policy to satisfy FANIN_LUT constraints
   with consistency penalties

Reward structure:
- Positive reward for satisfying FANIN_LUT constraints
- Small penalty for violating FANIN_LUT constraints
- Large penalty for inconsistent start/reconv node values
"""

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict

from src.ml.reconv_lib import MultiPathTransformer
from src.ml.embedding_extractor import EmbeddingExtractor
from src.atpg.reconv_podem import load_dataset, FANIN_LUT
from src.util.io import parse_bench_file
from src.util.struct import GateType


class ReconvRLTrainer:
    """Hybrid RL + Supervised trainer for reconvergent justification."""
    
    def __init__(
        self,
        embedding_dim: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_interaction_layers: int = 2,
        dim_feedforward: int = 512,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        reward_weights: Optional[Dict[str, float]] = None,
        verbose: bool = False,
        amp: bool = False,
    ):
        """Initialize trainer.
        
        Parameters
        ----------
        embedding_dim : int
            Dimension of gate embeddings.
        nhead : int
            Number of attention heads.
        num_encoder_layers : int
            Number of encoder layers for path processing.
        num_interaction_layers : int
            Number of layers for path interaction.
        dim_feedforward : int
            Dimension of feedforward network.
        learning_rate : float
            Learning rate for optimizer.
        device : str
            Device for training ('cuda' or 'cpu').
        reward_weights : dict, optional
            Weights for reward components:
            - 'fanin_correct': reward for satisfying FANIN_LUT
            - 'fanin_wrong': penalty for violating FANIN_LUT
            - 'consistency': penalty for inconsistent start/reconv nodes
        """

        self.device = device
        self.embedding_dim = embedding_dim
        self.verbose = verbose
        self.amp = amp

        # Initialize model
        model = MultiPathTransformer(
            embedding_dim=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_interaction_layers=num_interaction_layers,
            dim_feedforward=dim_feedforward
        )
        if torch.cuda.device_count() > 1:
            if self.verbose:
                print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
            self.model = torch.nn.DataParallel(model)
            self.model = self.model.cuda()
            self.device = 'cuda'
        else:
            self.model = model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Embedding extractor
        self.embedding_extractor = EmbeddingExtractor()

        # Reward weights
        if reward_weights is None:
            reward_weights = {
                'fanin_correct': 1.0,      # Good reward for correct
                'fanin_wrong': -0.2,       # Small penalty
                'consistency': -5.0        # Large penalty for inconsistency
            }
        self.reward_weights = reward_weights

        # Training stats
        self.stats = defaultdict(list)
        if self.verbose:
            print("ReconvRLTrainer initialized:")
            print(f"  device(s)={self.device}, emb_dim={self.embedding_dim}, nhead={nhead}")
            print(f"  enc_layers={num_encoder_layers}, inter_layers={num_interaction_layers}, ff_dim={dim_feedforward}")
            print(f"  lr={learning_rate}, reward_weights={self.reward_weights}")
    
    def extract_gate_embeddings(
        self,
        circuit_path: str,
        gate_indices: List[int]
    ) -> torch.Tensor:
        """Extract embeddings for specific gates in a circuit.
        
        Parameters
        ----------
        circuit_path : str
            Path to circuit .bench file.
        gate_indices : list[int]
            Indices of gates to extract embeddings for.
        
        Returns
        -------
        torch.Tensor
            Embeddings for the specified gates.
            Shape: (num_gates, embedding_dim)
        """
        # Extract embeddings for the circuit
        struct_emb, func_emb, gate_mapping, original_circuit = \
            self.embedding_extractor.extract_embeddings(circuit_path)
        
        # Combine structural and functional embeddings
        combined_emb = torch.cat([struct_emb, func_emb], dim=-1)
        
        # Extract embeddings for specific gates
        gate_embeddings = []
        for idx in gate_indices:
            if idx < len(combined_emb):
                gate_embeddings.append(combined_emb[idx])
            else:
                # Fallback: use zero embedding
                gate_embeddings.append(
                    torch.zeros(combined_emb.shape[-1], device=self.device)
                )
        
        return torch.stack(gate_embeddings)
    
    def create_simple_gate_embedding(
        self,
        circuit,
        gate_idx: int
    ) -> torch.Tensor:
        """Create simple embedding based on gate type and structural info.
        
        Fallback method when embedding_extractor fails.
        
        Parameters
        ----------
        circuit : list[Gate]
            Circuit gates.
        gate_idx : int
            Gate index.
        
        Returns
        -------
        torch.Tensor
            Simple embedding vector.
        """
        gate = circuit[gate_idx]
        
        # One-hot encode gate type (10 types + 1 for unknown)
        gate_type_vec = torch.zeros(11, device=self.device)
        if hasattr(gate, 'type') and 0 <= gate.type < 11:
            gate_type_vec[gate.type] = 1.0
        
        # Structural features
        nfi = getattr(gate, 'nfi', 0)
        nfo = getattr(gate, 'nfo', 0)
        
        struct_vec = torch.tensor([
            nfi / 10.0,  # Normalize fanin count
            nfo / 10.0,  # Normalize fanout count
        ], device=self.device)
        
        # Combine
        embedding = torch.cat([gate_type_vec, struct_vec])
        
        # Pad or truncate to match embedding_dim
        if embedding.shape[0] < self.embedding_dim:
            padding = torch.zeros(
                self.embedding_dim - embedding.shape[0],
                device=self.device
            )
            embedding = torch.cat([embedding, padding])
        else:
            embedding = embedding[:self.embedding_dim]
        
        return embedding
    
    def prepare_batch(
        self,
        dataset_entries: List[Dict],
        target_value: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """Prepare batch of data for training.
        
        Parameters
        ----------
        dataset_entries : list[dict]
            List of dataset entries.
        target_value : int
            Target justification value (0 or 1).
        
        Returns
        -------
        path_embeddings : torch.Tensor
            Padded path embeddings.
            Shape: (batch_size, num_paths, max_seq_len, embedding_dim)
        attention_masks : torch.Tensor
            Attention masks for padded tokens.
            Shape: (batch_size, num_paths, max_seq_len)
        targets : list[dict]
            Target justifications for each entry.
        """
        batch_paths = []
        batch_masks = []
        targets = []
        
        for entry in dataset_entries:
            circuit_path = entry['file']
            info = entry['info']
            paths = info['paths']
            
            # Get target justification
            if target_value == 1:
                target_just = entry['justification_1']
            else:
                target_just = entry['justification_0']
            
            targets.append(target_just)
            
            # Load circuit
            circuit, _ = parse_bench_file(circuit_path)
            
            # Extract embeddings for each path
            path_embeddings = []
            path_masks = []
            
            for path in paths:
                # Create embeddings for gates in this path
                gate_embs = []
                for gate_idx in path:
                    try:
                        emb = self.create_simple_gate_embedding(circuit, gate_idx)
                        gate_embs.append(emb)
                    except Exception as e:
                        print(f"Warning: Failed to create embedding for gate {gate_idx}: {e}")
                        gate_embs.append(torch.zeros(self.embedding_dim, device=self.device))
                
                if gate_embs:
                    path_emb = torch.stack(gate_embs)
                    path_embeddings.append(path_emb)
                    # Create mask (True for valid tokens)
                    mask = torch.ones(len(gate_embs), dtype=torch.bool, device=self.device)
                    path_masks.append(mask)
            
            if path_embeddings:
                batch_paths.append(path_embeddings)
                batch_masks.append(path_masks)
        
        # Pad paths to same length
        max_num_paths = max(len(paths) for paths in batch_paths)
        max_seq_len = max(
            max(path.shape[0] for path in paths)
            for paths in batch_paths
        )
        
        # Create padded tensors
        padded_paths = torch.zeros(
            len(batch_paths), max_num_paths, max_seq_len, self.embedding_dim,
            device=self.device
        )
        padded_masks = torch.zeros(
            len(batch_paths), max_num_paths, max_seq_len,
            dtype=torch.bool, device=self.device
        )
        
        for i, (paths, masks) in enumerate(zip(batch_paths, batch_masks)):
            for j, (path, mask) in enumerate(zip(paths, masks)):
                seq_len = path.shape[0]
                padded_paths[i, j, :seq_len, :] = path
                padded_masks[i, j, :seq_len] = mask
        
        return padded_paths, padded_masks, targets
    
    def compute_fanin_reward(
        self,
        circuit,
        predictions: torch.Tensor,
        path_indices: List[int]
    ) -> float:
        """Compute reward based on FANIN_LUT satisfaction.
        
        Parameters
        ----------
        circuit : list[Gate]
            Circuit gates.
        predictions : torch.Tensor
            Predicted logic values for gates.
            Shape: (num_gates, 2) - logits for 0 and 1
        path_indices : list[int]
            Indices of gates in the path.
        
        Returns
        -------
        float
            Reward value.
        """
        reward = 0.0
        
        # Convert predictions to values
        pred_values = torch.argmax(predictions, dim=-1)
        
        for i, gate_idx in enumerate(path_indices):
            gate = circuit[gate_idx]
            gate_type = gate.type
            
            # Skip input gates
            if gate_type == GateType.INPT:
                continue
            
            # Get predicted output value
            if i >= len(pred_values):
                continue
            pred_output = pred_values[i].item()
            
            # Get fanin indices and their predicted values
            fanin_indices = getattr(gate, 'fin', [])
            fanin_values = []
            
            for fin_idx in fanin_indices:
                # Find position of fanin in path
                if fin_idx in path_indices:
                    pos = path_indices.index(fin_idx)
                    if pos < len(pred_values):
                        fanin_values.append(pred_values[pos].item())
            
            # Check if fanin values are valid for the gate type
            if fanin_values and gate_type in FANIN_LUT:
                # Convert pred_output to LogicValue for lookup
                from src.util.struct import LogicValue
                logic_output = LogicValue.ONE if pred_output == 1 else LogicValue.ZERO
                valid_inputs = FANIN_LUT[gate_type].get(logic_output, [])
                
                # Check if any valid input combination matches
                # For simplicity, check if fanin values are in valid_inputs
                valid_input_values = [vi.value for vi in valid_inputs]
                all_valid = all(fv in valid_input_values for fv in fanin_values)
                
                if all_valid:
                    reward += self.reward_weights['fanin_correct']
                else:
                    reward += self.reward_weights['fanin_wrong']
        
        return reward
    
    def compute_consistency_reward(
        self,
        predictions: torch.Tensor,
        info: Dict,
        attention_masks: torch.Tensor
    ) -> float:
        """Compute reward for start/reconv node consistency.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Predicted values for all paths.
            Shape: (num_paths, seq_len, 2)
        info : dict
            Reconvergent structure info with start and reconv indices.
        attention_masks : torch.Tensor
            Masks indicating valid positions.
            Shape: (num_paths, seq_len)
        
        Returns
        -------
        float
            Consistency reward (negative penalty).
        """
        penalty = 0.0
        num_paths = predictions.shape[0]
        
        # Get predictions for start node (first position in each path)
        start_preds = predictions[:, 0, :]  # Shape: (num_paths, 2)
        start_values = torch.argmax(start_preds, dim=-1)
        
        # Check if all start values are the same
        if len(torch.unique(start_values)) > 1:
            penalty += self.reward_weights['consistency']
        
        # Get predictions for reconv node (last valid position in each path)
        reconv_values = []
        for i in range(num_paths):
            # Find last valid position
            valid_positions = torch.where(attention_masks[i])[0]
            if len(valid_positions) > 0:
                last_pos = int(valid_positions[-1].item())
                reconv_pred = predictions[i, last_pos, :]
                reconv_value = torch.argmax(reconv_pred).item()
                reconv_values.append(reconv_value)
        
        # Check if all reconv values are the same
        if reconv_values and len(set(reconv_values)) > 1:
            penalty += self.reward_weights['consistency']
        
        return penalty
    
    def train_step_supervised(
        self,
        path_embeddings: torch.Tensor,
        attention_masks: torch.Tensor,
        targets: List[Dict],
        info: Dict
    ) -> Dict[str, float]:
        """Supervised training step.
        
        Parameters
        ----------
        path_embeddings : torch.Tensor
            Path embeddings (batch_size, num_paths, seq_len, embedding_dim).
        attention_masks : torch.Tensor
            Attention masks (batch_size, num_paths, seq_len).
        targets : list[dict]
            Target justifications.
        info : dict
            Reconvergent structure info.
        
        Returns
        -------
        dict
            Training metrics.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Ensure tensors are on the correct device
        path_embeddings = path_embeddings.to(self.device)
        attention_masks = attention_masks.to(self.device)
        predictions = self.model(path_embeddings, attention_masks)
        # Shape: (batch_size, num_paths, seq_len, 2)

        batch_size, num_paths, seq_len, _ = predictions.shape
        if self.verbose:
            print(f"    [SUP] preds shape: batch={batch_size}, paths={num_paths}, seq={seq_len}")

        # Create target tensor
        target_tensor = torch.full(
            (batch_size, num_paths, seq_len),
            -1,  # Padding value
            dtype=torch.long,
            device=self.device
        )

        # Fill in target values
        # Note: This is simplified - in practice you'd need proper gate-to-position mapping
        # For now, we'll use a simple approach

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            predictions.permute(0, 3, 1, 2),  # (batch, classes, paths, seq)
            target_tensor,
            ignore_index=-1
        )

        # Backward pass
        loss.backward()
        self.optimizer.step()

        metrics = {'supervised_loss': loss.item()}
        if self.verbose:
            print(f"    [SUP] loss={metrics['supervised_loss']:.6f}")
        return metrics
    
    def train_step_rl(
        self,
        path_embeddings: torch.Tensor,
        attention_masks: torch.Tensor,
        circuit,
        info: Dict
    ) -> Dict[str, float]:
        """RL training step using policy gradient.
        
        Parameters
        ----------
        path_embeddings : torch.Tensor
            Path embeddings (batch_size, num_paths, seq_len, embedding_dim).
        attention_masks : torch.Tensor
            Attention masks (batch_size, num_paths, seq_len).
        circuit : list[Gate]
            Circuit gates.
        info : dict
            Reconvergent structure info.
        
        Returns
        -------
        dict
            Training metrics.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(path_embeddings, attention_masks)
        # Shape: (batch_size, num_paths, seq_len, 2)
        
        batch_size = predictions.shape[0]
        total_reward = 0.0
        policy_loss = 0.0
        
        for b in range(batch_size):
            # Sample actions from policy
            probs = F.softmax(predictions[b], dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)
            
            # Compute reward for this sample
            num_paths = predictions.shape[1]
            reward = 0.0
            
            for p in range(num_paths):
                path_indices = info['paths'][p]
                path_preds = predictions[b, p, :len(path_indices), :]
                reward += self.compute_fanin_reward(
                    circuit, path_preds, path_indices
                )
            
            # Add consistency reward
            reward += self.compute_consistency_reward(
                predictions[b], info, attention_masks[b]
            )
            
            total_reward += reward
            
            # Policy gradient loss: -log_prob * reward
            # Sum over valid positions only
            valid_log_probs = log_probs[attention_masks[b]]
            policy_loss += -(valid_log_probs.sum() * reward)
        
        # Average over batch
        policy_loss = policy_loss / batch_size
        avg_reward = total_reward / batch_size
        
        # Backward pass
        policy_loss.backward()
        self.optimizer.step()
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'avg_reward': avg_reward
        }
        if self.verbose:
            print(f"    [RL ] policy_loss={metrics['policy_loss']:.6f}, avg_reward={float(metrics['avg_reward']):.6f}")
        return metrics
    
    def train_epoch(
        self,
        dataset: List[Dict],
        batch_size: int = 4,
        supervised_weight: float = 0.7,
        rl_weight: float = 0.3,
        amp: bool = False
    ) -> Dict[str, float]:
        """Train for one epoch using hybrid approach.
        
        Parameters
        ----------
        dataset : list[dict]
            Training dataset.
        batch_size : int
            Batch size.
        supervised_weight : float
            Weight for supervised loss.
        rl_weight : float
            Weight for RL loss.
        
        Returns
        -------
        dict
            Epoch metrics.
        """
        epoch_metrics = defaultdict(float)
        num_batches = 0
        t0 = time.time()
        
        # Shuffle dataset
        indices = torch.randperm(len(dataset))
        total_batches = max(1, (len(dataset) + batch_size - 1) // batch_size)
        
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_entries = [dataset[idx] for idx in batch_indices]
            batch_id = (i // batch_size) + 1

            # Progress bar
            progress = int(50 * batch_id / total_batches)
            bar = '[' + '#' * progress + '-' * (50 - progress) + ']'
            print(f"Epoch progress {bar} Batch {batch_id}/{total_batches}", end='\r')

            # Prepare batch (use justification_1 for training)
            path_emb, attn_mask, targets = self.prepare_batch(batch_entries, target_value=1)

            if path_emb.shape[0] == 0:
                if self.verbose:
                    print(f"  [B{batch_id:03d}/{total_batches}] empty batch, skipping")
                continue
            if self.verbose:
                print(f"  [B{batch_id:03d}/{total_batches}] path_emb={tuple(path_emb.shape)}, attn_mask={tuple(attn_mask.shape)}")
            if batch_id == total_batches:
                print()  # Newline at end of epoch

            # Get circuit and info from first entry
            circuit, _ = parse_bench_file(batch_entries[0]['file'])
            info = batch_entries[0]['info']

            # Supervised training step
            if supervised_weight > 0:
                if amp or self.amp:
                    from torch.cuda.amp import autocast, GradScaler
                    scaler = getattr(self, '_amp_scaler', None)
                    if scaler is None:
                        scaler = GradScaler()
                        self._amp_scaler = scaler
                    with autocast():
                        sup_metrics = self.train_step_supervised(
                            path_emb, attn_mask, targets, info
                        )
                        scaler.scale(sup_metrics['supervised_loss']).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()
                else:
                    sup_metrics = self.train_step_supervised(
                        path_emb, attn_mask, targets, info
                    )
                    for k, v in sup_metrics.items():
                        epoch_metrics[k] += v * supervised_weight
                    if self.verbose:
                        print(f"    [B{batch_id:03d}] sup_loss(w)={sup_metrics.get('supervised_loss', 0.0)*supervised_weight:.6f}")

            # RL training step
            if rl_weight > 0:
                if amp or self.amp:
                    from torch.cuda.amp import autocast, GradScaler
                    scaler = getattr(self, '_amp_scaler', None)
                    if scaler is None:
                        scaler = GradScaler()
                        self._amp_scaler = scaler
                    with autocast():
                        rl_metrics = self.train_step_rl(
                            path_emb, attn_mask, circuit, info
                        )
                        scaler.scale(rl_metrics['policy_loss']).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()
                else:
                    rl_metrics = self.train_step_rl(
                        path_emb, attn_mask, circuit, info
                    )
                    for k, v in rl_metrics.items():
                        epoch_metrics[k] += v * rl_weight
                    if self.verbose:
                        print(f"    [B{batch_id:03d}] rl_loss(w)={rl_metrics.get('policy_loss', 0.0)*rl_weight:.6f}, avg_reward={float(rl_metrics.get('avg_reward', 0.0)):.6f}")

            num_batches += 1
        
        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        if self.verbose:
            dt = time.time() - t0
            print(f"  [EPOCH] batches={num_batches}, time={dt:.2f}s, metrics={dict(epoch_metrics)}")
        
        return dict(epoch_metrics)
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'reward_weights': self.reward_weights
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")
        return checkpoint


def main():
    """Main training script."""
    # Load dataset
    dataset_path = "data/datasets/reconv_dataset.pkl"
    dataset = load_dataset(dataset_path)
    
    # Initialize trainer
    trainer = ReconvRLTrainer(
        embedding_dim=128,
        nhead=8,
        num_encoder_layers=4,
        num_interaction_layers=2,
        dim_feedforward=512,
        learning_rate=1e-4
    )
    
    # Training loop
    num_epochs = 50
    checkpoint_dir = "checkpoints/reconv_rl"
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train epoch
        metrics = trainer.train_epoch(
            dataset,
            batch_size=4,
            supervised_weight=0.7,
            rl_weight=0.3
        )
        
        # Print metrics
        print(f"  Metrics: {metrics}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{epoch + 1}.pth"
            )
            trainer.save_checkpoint(checkpoint_path, epoch + 1, metrics)


if __name__ == "__main__":
    main()
