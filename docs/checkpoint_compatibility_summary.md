# Checkpoint Compatibility Summary

This document summarizes the compatibility and performance of key checkpoints evaluated on Monday, May 4, 2026.

## Evaluated Checkpoints

| Checkpoint Path | Status | Environment | ITC99 Gate Coverage (1%) |
| :--- | :--- | :--- | :--- |
| `checkpoints/supervised_v4/best_model.pth` | COMPATIBLE | deepgate (CPU) | 16.30% |
| `checkpoints/supervised_v5/best_model.pth` | COMPATIBLE | deepgate (CPU) | 17.70% |
| `checkpoints/unlinked_candidate/best_model.pth` | COMPATIBLE | deepgate (CPU) | 18.17% |

## Compatibility Notes

- **Architecture**: All checkpoints are compatible with `src.ml.core.model.MultiPathTransformer`.
- **Environment**: Verified in `deepgate` conda environment.
- **Hardware**: Evaluations performed on CPU due to CUDA compatibility issues with RTX 5070 Ti (sm_120) and installed PyTorch (supports up to sm_90).
- **Inference Diversity**: Verified that `ModelPairPredictor` correctly loads these models and supports multi-candidate decoding.

## Performance Trend

The "updated" training pipeline, which includes supervised per-node CE for SAT labels and the new shard format, shows a steady improvement in no-fallback coverage on the held-out ITC99 b17 gate subset.

- **v4 to v5**: +1.40% absolute (+8.5% relative)
- **v5 to unlinked_candidate**: +0.47% absolute (+2.6% relative)
- **Overall (v4 to unlinked)**: +1.87% absolute (+11.5% relative)

These results confirm that the architectural and loss-function updates are driving better generalization on held-out circuits.
