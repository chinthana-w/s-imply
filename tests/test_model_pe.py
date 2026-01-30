
import torch
from src.ml.reconv_lib import MultiPathTransformer, PositionalEncoding

def test_cuda_and_amp():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Tiny model
    B, P, L, D = 2, 4, 10, 32
    model = MultiPathTransformer(input_dim=D, model_dim=D, nhead=4, num_encoder_layers=1, num_interaction_layers=1, dim_feedforward=64).to(device)
    
    # Fake data
    paths = torch.randn(B, P, L, D, device=device)
    masks = torch.ones(B, P, L, dtype=torch.bool, device=device)
    
    print("Testing forward pass (float32)...")
    out = model(paths, masks)
    print(f"Output shape: {out.shape}")
    
    print("Testing AMP forward pass...")
    with torch.amp.autocast('cuda', enabled=True):
        out_amp = model(paths, masks)
        print(f"AMP Output shape: {out_amp.shape}")
        
    print("Testing PositionalEncoding standalone...")
    pe = PositionalEncoding(d_model=D).to(device)
    dummy = torch.randn(B, L, D, device=device)
    out_pe = pe(dummy)
    print(f"PE Output shape: {out_pe.shape}")
    
    print("All tests passed.")

if __name__ == "__main__":
    test_cuda_and_amp()
