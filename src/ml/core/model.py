import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Helper: Standard Positional Encoding ---
# Injects position information into the input embeddings.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant sinusoidal position encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it's not a parameter but is part of state_dict
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        # self.pe: [1, max_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# --- The Main Transformer Architecture ---


class MultiPathTransformer(nn.Module):
    """
    A Transformer architecture to analyze and predict values for multiple,
    variable-length reconvergent paths.

    Parameters
    ----------
    input_dim : int
        Dimension of incoming node embeddings (from dataset/collate)
    model_dim : int
        Internal transformer model dimension (can be larger than input_dim)
    nhead : int
        Number of attention heads (must divide model_dim)
    num_encoder_layers : int
        Layers for the shared path encoder
    num_interaction_layers : int
        Layers for the path interaction encoder
    dim_feedforward : int
        Feedforward dimension inside Transformer layers
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        nhead: int,
        num_encoder_layers: int,
        num_interaction_layers: int,
        dim_feedforward: int = 512,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)

        # Explicit Gate Type Embedding (12 types -> 64 dims)
        self.gate_type_emb = nn.Embedding(12, 64)

        # Node Identity Embedding (Topology Awareness)
        self.num_total_nodes = 20000  # Default large enough for ISCAS85/89
        self.node_emb = nn.Embedding(self.num_total_nodes, 64)

        # Input has: [Base Feature] + [GateType(64)] + [NodeID(64)]
        self.input_aug_dim = self.input_dim + 64 + 64

        # Optional input projection to expand/shrink to model_dim
        # Input is now augmented input_dim + 16
        if self.input_aug_dim != self.model_dim:
            self.input_proj = nn.Linear(self.input_aug_dim, self.model_dim)
        else:
            self.input_proj = nn.Identity()

        # 1. Shared Path Encoder
        # This single encoder processes each path independently to learn the
        # general features of a logic path.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        # Use new nested tensor API to avoid warnings
        encoder_layer.enable_nested_tensor = True
        self.shared_path_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # 2. Path Interaction Layer
        # This layer allows the fully-encoded paths to "talk" to each other.
        # It's another Transformer encoder that treats each path as a single "token".
        interaction_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        # Use new nested tensor API to avoid warnings
        interaction_layer.enable_nested_tensor = True
        self.path_interaction_layer = nn.TransformerEncoder(
            interaction_layer, num_layers=num_interaction_layers
        )

        # 3. Prediction Heads
        # A simple linear layer to predict the final logic value (0 or 1) for each node.
        self.prediction_head = nn.Linear(self.model_dim, 2)

        # New: Global Solvability Head
        # Predicts if the entire reconvergent structure is solvable (0) or unsolvable (1)
        # given the target value. Input is pooled from path summaries.
        self.solvability_head = nn.Linear(self.model_dim, 2)

        # 4. Cross-Attention Block
        # Allows path nodes (Query) to attend to all interaction-aware path summaries (Key/Value).
        self.cross_attn = nn.MultiheadAttention(self.model_dim, nhead, batch_first=True)
        self.cross_norm = nn.LayerNorm(self.model_dim)

        self.pos_encoder = PositionalEncoding(self.model_dim)

    def forward(
        self,
        path_list,
        attention_masks,
        gate_types=None,
        node_ids=None,
        seed: Optional[int] = None,
        perturb_scale: float = 0.0,
        checkpointing: bool = False,
    ):
        """
        Args:
            path_list (Tensor): [batch_size, num_paths, seq_len, input_dim]
            attention_masks (Tensor): [batch_size, num_paths, seq_len] (True for valid tokens)
            gate_types (Tensor): [batch_size, num_paths, seq_len]
            node_ids (Tensor): [batch_size, num_paths, seq_len] - Physical Gate IDs
            seed (int, optional): Random seed.
            perturb_scale (float): Noise scale.
            checkpointing (bool): Whether to use gradient checkpointing.
        """
        from torch.utils.checkpoint import checkpoint

        batch_size, num_paths, seq_len, _ = path_list.shape

        # 1. Embed Discrete Features
        features = [path_list]

        if gate_types is not None:
            gt_emb = self.gate_type_emb(gate_types.clamp(0, 11))
            features.append(gt_emb)

        if node_ids is not None:
            # Clamp to avoid out of bounds
            n_emb = self.node_emb(node_ids.clamp(0, self.num_total_nodes - 1))
            features.append(n_emb)
        else:
            # Fallback if no node_ids provided (though they should be)
            n_emb = torch.zeros(batch_size, num_paths, seq_len, 64, device=path_list.device)
            features.append(n_emb)

        # Concatenate: [B, P, L, D_total]
        x = torch.cat(features, dim=-1)

        # Apply input projection
        x = self.input_proj(x)
        # Reshape the input to process all paths in the batch at once.
        # (batch_size * num_paths, seq_len, model_dim)
        flat_paths = x.view(-1, seq_len, self.model_dim)

        # Apply Positional Encoding to learn sequential order (input -> output)
        flat_paths = self.pos_encoder(flat_paths)

        flat_masks = attention_masks.view(-1, seq_len)

        # Pass all paths through the same shared encoder.
        if checkpointing:
            # Note: src_key_padding_mask requires ~flat_masks (padded positions are True)
            encoded_paths = checkpoint(
                self.shared_path_encoder, flat_paths, None, ~flat_masks, use_reentrant=False
            )
        else:
            encoded_paths = self.shared_path_encoder(flat_paths, src_key_padding_mask=~flat_masks)

        # --- Step 2: Allow Paths to Interact ---
        # Masked Max Pooling: Aggregate features across length dimension

        # Apply mask: set invalid positions to -inf
        mask_expanded = flat_masks.unsqueeze(-1)  # (B*P, L, 1)

        # Fill invalid positions with large negative value for Max Pooling
        pooled_input = encoded_paths.masked_fill(~mask_expanded, -1e9)

        # Max Pool over L dimension: (B*P, D)
        path_summaries, _ = pooled_input.max(dim=1)

        # Group by original batch item: (batch_size, num_paths, model_dim)
        path_representations = path_summaries.view(batch_size, num_paths, self.model_dim)

        # Paths interact through a Transformer operating over the path axis.
        if checkpointing:
            interaction_aware_reps = checkpoint(
                self.path_interaction_layer, path_representations, use_reentrant=False
            )
        else:
            interaction_aware_reps = self.path_interaction_layer(path_representations)

        # --- Step 3: Combine and Predict ---
        # Reshape encoded paths back to grouped form: (batch_size, num_paths, seq_len, model_dim)
        encoded_paths = encoded_paths.view(batch_size, num_paths, seq_len, self.model_dim)

        # --- New Cross-Attention Step ---
        # Flatten paths again to treat them as independent query sequences: (B*P, L, D)
        query = encoded_paths.view(-1, seq_len, self.model_dim)

        # Prepare Key/Value
        kv = (
            interaction_aware_reps.unsqueeze(1)
            .expand(-1, num_paths, -1, -1)
            .reshape(-1, num_paths, self.model_dim)
        )

        # Cross-Attention
        if checkpointing:
            attn_out, _ = checkpoint(self.cross_attn, query, kv, kv, use_reentrant=False)
        else:
            attn_out, _ = self.cross_attn(query, kv, kv)

        # Residual Connection + Norm + Add back original interaction context
        global_context = interaction_aware_reps.unsqueeze(2)  # (B, P, 1, model_dim)
        flat_global_context = global_context.view(-1, 1, self.model_dim)

        # Combine: Original Node + CrossAttn Result + Own Summary
        final_representations = self.cross_norm(query + attn_out + flat_global_context)

        # Reshape back to (B, P, L, D) for prediction
        final_representations = final_representations.view(
            batch_size, num_paths, seq_len, self.model_dim
        )

        # Per-node logits
        per_node_logits = self.prediction_head(final_representations)  # (B, P, L, 2)

        # Global Solvability Prediction
        batch_summary = interaction_aware_reps.mean(dim=1)  # (B, D)
        solvability_logits = self.solvability_head(batch_summary)  # (B, 2)

        return per_node_logits, solvability_logits


# --- Training Logic Placeholder ---
def custom_loss_function(predictions, targets, original_lengths):
    """
    Placeholder for the full training logic.
    """
    # 1. Main Prediction Loss (e.g., Cross-Entropy)
    # This would compare predictions to the ground truth, ignoring padded values.
    main_loss = F.cross_entropy(
        predictions.permute(0, 3, 1, 2), targets, ignore_index=-1
    )  # Assuming -1 for padding

    # 2. Consistency Loss for Shared Nodes
    # Enforces that the first and last nodes of all paths in a set are the same.
    # Get predictions for the first node of each path (for all items in batch)
    first_node_preds = predictions[:, :, 0, :]  # Shape: (batch_size, num_paths, 2)
    # Get predictions for the last node of each path (using original_lengths)
    # (More complex indexing needed here based on original_lengths)

    # Calculate variance or MSE between predictions for path 0, path 1, etc.
    consistency_loss = torch.var(first_node_preds, dim=1).mean()  # Simplified example

    # 3. Combine Losses
    total_loss = main_loss + (0.5 * consistency_loss)  # Weighting factor of 0.5

    return total_loss
