import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper: Standard Positional Encoding ---
# Injects position information into the input embeddings.
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
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        # self.pe: [1, max_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
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
    def __init__(self, input_dim: int, model_dim: int, nhead: int, num_encoder_layers: int, num_interaction_layers: int, dim_feedforward: int = 512):
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        
        # Explicit Gate Type Embedding (12 types -> 64 dims)
        # We append this to the input embedding, so actual input to projection increases by 64.
        self.gate_type_emb = nn.Embedding(12, 64)
        self.input_aug_dim = self.input_dim + 64

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
            batch_first=True
        )
        self.shared_path_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 2. Path Interaction Layer
        # This layer allows the fully-encoded paths to "talk" to each other.
        # It's another Transformer encoder that treats each path as a single "token".
        interaction_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.path_interaction_layer = nn.TransformerEncoder(interaction_layer, num_layers=num_interaction_layers)
        
        # 3. Prediction Head
        # A simple linear layer to predict the final logic value (0 or 1) for each node.
        # We use embedding_dim as input and 2 as output for the two classes (0 and 1).
        self.prediction_head = nn.Linear(self.model_dim, 2)
        
        # 4. Cross-Attention Block
        # Allows path nodes (Query) to attend to all interaction-aware path summaries (Key/Value).
        self.cross_attn = nn.MultiheadAttention(self.model_dim, nhead, batch_first=True)
        self.cross_norm = nn.LayerNorm(self.model_dim)
        
        self.pos_encoder = PositionalEncoding(self.model_dim)

    def forward(self, path_list, attention_masks, gate_types=None):
        """
        Args:
            path_list (Tensor): A padded tensor of path embeddings.
                                Shape: (batch_size, num_paths, seq_len, embedding_dim)
            attention_masks (Tensor): A boolean mask to ignore padded tokens.
                                     Shape: (batch_size, num_paths, seq_len)
            gate_types (Tensor, optional): Gate types for each node.
                                           Shape: (batch_size, num_paths, seq_len)
        """
        batch_size, num_paths, seq_len, _ = path_list.shape

        if gate_types is not None:
             # Embed gate types
             # gate_types: [B, P, L] -> [B, P, L, 16]
             gt_emb = self.gate_type_emb(gate_types.clamp(0, 11))
             # Concatenate to path embeddings
             path_list = torch.cat([path_list, gt_emb], dim=-1) # [B, P, L, D+64]

        # --- Step 1: Encode Each Path Independently ---
        # Reshape the input to process all paths in the batch at once.
        # (batch_size * num_paths, seq_len, input_aug_dim)
        flat_paths = path_list.view(-1, seq_len, self.input_aug_dim)
        # Project to model dimension if needed
        flat_paths = self.input_proj(flat_paths)
        
        # Apply Positional Encoding to learn sequential order (input -> output)
        flat_paths = self.pos_encoder(flat_paths)

        flat_masks = attention_masks.view(-1, seq_len)

        # Pass all paths through the same shared encoder.
        encoded_paths = self.shared_path_encoder(flat_paths, src_key_padding_mask=~flat_masks)

        # --- Step 2: Allow Paths to Interact ---
        # Use the first token as a summary representation for each path.
        path_representations = encoded_paths[:, 0, :]  # (batch_size * num_paths, model_dim)

        # Group by original batch item: (batch_size, num_paths, model_dim)
        path_representations = path_representations.view(batch_size, num_paths, self.model_dim)

        # Paths interact through a Transformer operating over the path axis.
        interaction_aware_reps = self.path_interaction_layer(path_representations)

        # --- Step 3: Combine and Predict ---
        # Broadcast interaction context back to each node position in each path.
        # Broadcast interaction context back to each node position in each path.
        # OLD: global_context = interaction_aware_reps.unsqueeze(2)  # (B, P, 1, model_dim)

        # Reshape encoded paths back to grouped form: (batch_size, num_paths, seq_len, model_dim)
        encoded_paths = encoded_paths.view(batch_size, num_paths, seq_len, self.model_dim)
        
        # --- New Cross-Attention Step ---
        # Flatten paths again to treat them as independent query sequences: (B*P, L, D)
        query = encoded_paths.view(-1, seq_len, self.model_dim)
        
        # Prepare Key/Value: The set of all path summaries for the corresponding batch item.
        # interaction_aware_reps is (B, P, D).
        # We need to repeat this P times so that each of the P paths in a batch can see the full set of P summaries.
        # kv: (B, P, P, D) -> flatten to (B*P, P, D)
        kv = interaction_aware_reps.unsqueeze(1).expand(-1, num_paths, -1, -1).reshape(-1, num_paths, self.model_dim)
        
        # Cross-Attention: Query=Nodes, Key=PathSummaries, Value=PathSummaries
        attn_out, _ = self.cross_attn(query, kv, kv)
        
        # Residual Connection + Norm + Add back original interaction context (skip connection for "self" bias)
        # Note: We keep the explicit "own interaction context" addition as a strong bias.
        global_context = interaction_aware_reps.unsqueeze(2) # (B, P, 1, model_dim)
        # Reshape global_context to flat (B*P, 1, D) for addition
        flat_global_context = global_context.view(-1, 1, self.model_dim)
        
        # Combine: Original Node + CrossAttn Result + Own Summary
        # This gives the model maximum flexibility: use local info, look at others, or look at own summary.
        final_representations = self.cross_norm(query + attn_out + flat_global_context)
        
        # Reshape back to (B, P, L, D) for prediction (optional, but prediction_head works on last dim anyway)
        final_representations = final_representations.view(batch_size, num_paths, seq_len, self.model_dim)

        # Per-node logits
        predictions = self.prediction_head(final_representations)  # (B, P, L, 2)
        return predictions

# --- Training Logic Placeholder ---
def custom_loss_function(predictions, targets, original_lengths):
    """
    Placeholder for the full training logic.
    """
    # 1. Main Prediction Loss (e.g., Cross-Entropy)
    # This would compare predictions to the ground truth, ignoring padded values.
    main_loss = F.cross_entropy(predictions.permute(0, 3, 1, 2), targets, ignore_index=-1) # Assuming -1 for padding
    
    # 2. Consistency Loss for Shared Nodes
    # Enforces that the first and last nodes of all paths in a set are the same.
    # Get predictions for the first node of each path (for all items in batch)
    first_node_preds = predictions[:, :, 0, :] # Shape: (batch_size, num_paths, 2)
    # Get predictions for the last node of each path (using original_lengths)
    # (More complex indexing needed here based on original_lengths)
    
    # Calculate variance or MSE between predictions for path 0, path 1, etc.
    consistency_loss = torch.var(first_node_preds, dim=1).mean() # Simplified example
    
    # 3. Combine Losses
    total_loss = main_loss + (0.5 * consistency_loss) # Weighting factor of 0.5
    
    return total_loss