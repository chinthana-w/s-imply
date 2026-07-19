# Training Metrics and Hyperparameters

This document summarizes the main metrics and hyperparameters used when
training the S-Imply multi-path transformer. It covers the supervised and
logic-consistency trainer in `src.ml.train`, plus the RL fine-tuning path in
`scripts.train_rl`.

## Training Modes

S-Imply has two training stages:

1. Supervised and logic-consistency training learns node values, solvability,
   and circuit-consistent assignments from generated reconvergent path data.
2. RL fine-tuning reuses collected AI-PODEM experience and updates the same
   transformer policy from reward-weighted rollouts.

The supervised trainer should usually produce the baseline checkpoint. RL
fine-tuning is a later refinement step that starts from that checkpoint and
uses real solver outcomes as feedback.

## Core Metrics

The supervised trainer reports:

- `train_loss` and `val_loss`: total objective after all enabled loss terms.
  `best_model.pth` is selected by lowest validation loss.
- `avg_reward`: diagnostic reward derived from local consistency checks. It is
  useful for trend tracking, but it is not the final ATPG score.
- `path_acc`: fraction of valid paths whose predicted values satisfy the
  path-level checks used by the loss.
- `edge_acc`: local gate-edge consistency accuracy for adjacent nodes along
  each path.
- `c_viol`: constraint violation rate. This should decrease when the model is
  learning to respect known node values.
- `reconv_match_rate`: debug metric for whether valid paths agree at the
  reconvergent endpoint.
- `solvability_acc`: accuracy of the SAT/UNSAT solvability head when labels are
  available.

The RL trainer reports:

- `loss`: REINFORCE policy loss with entropy regularization.
- `entropy`: average policy entropy over valid path positions. Higher entropy
  means the model is still exploring; very low entropy can indicate premature
  collapse.
- `reward`: mean reward of the sampled experience batch.

Final model quality is still measured by ATPG benchmarks: detected faults,
coverage, runtime, backtracks, timeouts, and classic-relative AI contribution.
Training metrics are diagnostics, not proof that a fault will be detected.

## Main Hyperparameters

Model capacity:

- `--model-dim`: transformer hidden width. Default: `512`.
- `--nhead`: number of attention heads. Default: `4`.
- `--enc-layers`: shared per-path encoder depth. Default: `3`.
- `--int-layers`: path interaction encoder depth. Default: `3`.
- `--ffn-dim`: feedforward width inside transformer blocks. Default: `512`.
- `--max-paths`: maximum paths retained per sample. Default: `200`.
- `--max-len`: optional maximum path-length filter for curriculum runs.

Optimization and runtime:

- `--epochs`: number of passes over the training split. Default: `10`.
- `--batch-size`: supervised batch size. Default: `128`.
- `--lr`: learning rate in the training config. Default: `1e-4`.
- `--amp`: enables automatic mixed precision.
- `--grad-accum`: accumulates gradients across batches before stepping.
- `--checkpointing`: saves VRAM by recomputing activations during backward.
- `--num-workers` and `--pin-memory`: DataLoader throughput controls.
- `--shard-cache-size`: number of processed shards cached in memory.
- `--resume`: resumes full state from `<output>/resume.pth`.

Loss weights:

- `--lambda-supervised-node`: per-node label cross-entropy weight. Default:
  `2.0`.
- `--lambda-solvability`: SAT/UNSAT head loss weight. Default: `1.0`.
- `--lambda-shared-node`: repeated-node consistency weight across paths.
  Default: `1.0`.
- `--lambda-logic`: reconvergence-level logic consistency weight.
- `--lambda-full-logic`: full path gate-consistency weight.
- `--soft-edge-lambda`: local soft edge consistency weight. Default: `1.0`.
- `--entropy-beta`: entropy regularization weight.
- `--gumbel-temp`: initial Gumbel-Softmax temperature. Default: `1.0`.
- `--gumbel-anneal-rate`: per-epoch Gumbel temperature decay. Default: `0.99`.

RL fine-tuning:

- `--experience_dir`: directory containing `batch_*.pkl` experience files.
- `--model`: initial transformer checkpoint.
- `--output`: RL-tuned checkpoint path.
- `--epochs`: RL fine-tuning epochs. Default: `10`.
- `--batch_size`: RL experience batch size. Default: `256`.
- `--lr`: RL optimizer learning rate. Default: `1e-4`.
- `--max_paths`: path cap per experience sample. Default: `200`.
- `--num_workers`: RL DataLoader workers. Default: `8`.

## Practical Reading

Healthy supervised runs should show validation loss trending down while
`edge_acc`, `path_acc`, `reconv_match_rate`, and `solvability_acc` improve.
If validation loss improves but constraint violations remain high, the model may
be learning labels without respecting known circuit state. If local metrics look
good but ATPG coverage does not improve, the model may be producing assignments
that are locally consistent but not useful inside the full PODEM search.

For RL runs, reward should be interpreted together with entropy. Rising reward
with moderate entropy is usually healthier than rising reward with entropy
collapsing immediately. The final checkpoint should always be benchmarked
against classic PODEM and the supervised checkpoint before promoting it.
