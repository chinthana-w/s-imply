# Failure Analysis Summary

## Manual Inspection Results
Inspection of checkpoints trained with Gated Reward + Curriculum reveals:
1.  **Inversion Failure**: The model frequently fails to correctly invert values.
    - `NOT(0) -> 0` (Should be 1)
    - `NOR(1, ...) -> 1` (Should be 0)
    - `NAND(0, ...) -> 0` (Should be 1)
2.  **Buffer Behavior**: The model acts like a "buffer", copying the input value to the output regardless of gate type.
3.  **Consistency Shortcut**: By buffering values, the model often achieves global consistency (e.g., `1 -> 1 -> ... -> 1` matches `1 -> 1 -> ... -> 1`), satisfying the Reconvergence metric but failing Edge Accuracy.
4.  **Gradient Issues**: The hard reward (-1.0) penalizes these failures but provides no directional information on *how* to fix them (unlike a differentiable loss).

## Proposed Solution: Hybrid Loss
Combine the **Gated Reward** (Hard Constraint) with **Soft Edge Loss** (Differentiable Guidance).
- **Gated Reward**: Ensures that the final policy is penalized if it violates logic.
- **Soft Edge Loss**: Provides strong gradients to push probability distributions towards valid local logic (e.g., increasing P(1) if input is 0 and gate is NOT).
- **Action**: Retrain with `soft_edge_lambda=50.0` (or higher) AND `Gated Reward`.
