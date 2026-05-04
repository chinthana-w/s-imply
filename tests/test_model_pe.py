import torch

from src.ml.core.model import MultiPathTransformer, PositionalEncoding


def test_cuda_and_amp():
    # Detect device
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            # Test if we can actually use it
            torch.randn(1, device=device)
            print(f"Using device: {device}")
        except Exception as e:
            print(f"CUDA available but failed to use: {e}. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        print("CUDA not available, using CPU.")
        device = torch.device("cpu")

    # Tiny model
    B, P, L, D = 2, 4, 10, 32
    model = MultiPathTransformer(
        input_dim=D,
        model_dim=D,
        nhead=4,
        num_encoder_layers=1,
        num_interaction_layers=1,
        dim_feedforward=64,
    ).to(device)

    # Fake data
    paths = torch.randn(B, P, L, D, device=device)
    masks = torch.ones(B, P, L, dtype=torch.bool, device=device)

    print("Testing forward pass (float32)...")
    gate_types = torch.randint(0, 12, (B, P, L), device=device)
    out, solv_out = model(paths, masks, gate_types=gate_types)
    print(f"Output shape: {out.shape}, Solvability output shape: {solv_out.shape}")

    assert out.shape == (B, P, L, 2)
    assert solv_out.shape == (B, 2)

    if device.type == "cuda":
        print("Testing AMP forward pass...")
        with torch.amp.autocast("cuda", enabled=True):
            out_amp, solv_amp = model(paths, masks, gate_types=gate_types)
            print(f"AMP Output shape: {out_amp.shape}")
            assert out_amp.shape == (B, P, L, 2)

    print("Testing PositionalEncoding standalone...")
    pe = PositionalEncoding(d_model=D).to(device)
    dummy = torch.randn(B, L, D, device=device)
    out_pe = pe(dummy)
    print(f"PE Output shape: {out_pe.shape}")

    print("All tests passed.")


if __name__ == "__main__":
    test_cuda_and_amp()
