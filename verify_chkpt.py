import torch

from src.ml.core.model import MultiPathTransformer


def test_checkpoint_compatibility():
    ckpt_path = "checkpoints/supervised_v5/best_model.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        model_state = state["state_dict"]
    elif "model" in state:
        model_state = state["model"]
    else:
        model_state = state

    # Extract config if present
    cfg_dict = state.get("config", {})
    if hasattr(cfg_dict, "__dict__"):
        cfg_dict = cfg_dict.__dict__
    
    num_encoder_layers = cfg_dict.get("num_encoder_layers", 3)
    num_interaction_layers = cfg_dict.get("num_interaction_layers", 3)
    
    # Try to dynamically get dimensions from the weights if possible
    input_proj_weight = model_state.get("input_proj.weight")
    if input_proj_weight is not None:
        model_dim = input_proj_weight.shape[0]
        input_dim = input_proj_weight.shape[1] - 64
    else:
        model_dim = 512
        input_dim = 132

    print(f"Instantiating model with input_dim={input_dim}, model_dim={model_dim}, enc={num_encoder_layers}, int={num_interaction_layers}")
    model = MultiPathTransformer(
        input_dim=input_dim,
        model_dim=model_dim,
        nhead=4,
        num_encoder_layers=num_encoder_layers,
        num_interaction_layers=num_interaction_layers,
        dim_feedforward=512,
    )
    
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print("Missing keys:")
    for k in missing: print("  -", k)
    print("Unexpected keys:")
    for k in unexpected: print("  -", k)

    if not missing and not unexpected:
        print("Checkpoint loaded perfectly with NO missing or unexpected keys!")
    else:
        print("Checkpoint loaded with some discrepancies (see above).")

if __name__ == "__main__":
    test_checkpoint_compatibility()
