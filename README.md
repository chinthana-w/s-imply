# s-imply

A Topology-Aware Justification Oracle for digital circuits using Multi-Path Transformers and 3-Valued Logic reasoning.

## 🚀 Key Features
- **3-Valued Logic Reasoning**: Explicitly handles `0`, `1`, and `X` (Don't Care) logic states.
- **Topology-Aware Embeddings**: Maps physical gate identities across reconvergent paths for global consistency.
- **Physics-Informed Training**: Incorporates differentiable logic consistency loss to enforce Boolean truth tables.
- **Hybrid AI-PODEM**: Integrates AI-based justification directly into the PODEM backtrace loop.

---

## 🛠️ Usage Guide

### 1. Data Preparation (Building Shards)
Before training, convert large pickle datasets into optimized tensor shards for fast I/O.

```bash
python -m src.ml.core.dataset \
    --input /home/local1/cache-cw/reconv_dataset.pkl \
    --out /home/local1/cache-cw/processed_reconv/ \
    --max_len 50
```

### 2. Experience Collection
Generate fresh RL experience by running AI-assisted PODEM on benchmark circuits.
Accepts benchmark directories or individual `.bench` files. The model is pre-loaded
once and shared across all circuits to avoid redundant GPU transfers.

```bash
python -m scripts.collect_experience \
    --bench_dirs data/bench/ISCAS85 data/bench/iscas89 \
    --model checkpoints/unlinked_candidate/best_model.pth \
    --max_faults 50 \
    --exploration 5
```

### 3. Model Training
Train the transformer using a combination of supervised labels and physics-informed consistency losses.

```bash
python -m src.ml.train train \
    --dataset /home/local1/cache-cw/reconv_dataset.pkl \
    --processed-dir /home/local1/cache-cw/processed_reconv/ \
    --output checkpoints/reconv_topology_3val_v1_ssd \
    --epochs 50 \
    --batch-size 3000 \
    --grad-accum 1 \
    --max-paths 256 \
    --shard-cache-size 25 \
    --checkpointing \
    --num-workers 4 \
    --amp --verbose \
    --lambda-logic 1.0 \
    --ffn-dim 2048 \
    --model-dim 512 \
    --enc-layers 3 \
    --int-layers 3
```

### 4. RL Fine-tuning
After collecting experience, fine-tune the transformer using policy gradient (REINFORCE)
on the collected episodes.

#### Fine-tune only
```bash
python -m scripts.train_rl \
    --model checkpoints/unlinked_candidate/best_model.pth \
    --output checkpoints/reconv_rl_model.pt \
    --epochs 10 \
    --batch_size 256 \
    --max_paths 200 \
    --amp
```

#### Full pipeline (collect → train → benchmark)
```bash
python scripts/run_rl_pipeline.py --all \
    --bench_dirs data/bench/ISCAS85 data/bench/iscas89 \
    --max_faults 100 \
    --exploration 5 \
    --epochs 20
```

Individual stages can be run independently with `--collect`, `--train`, or `--benchmark`.
On multi-GPU machines the collection stage is automatically parallelised across GPUs,
with the number of processes capped by available RAM (assumes ~5 GB per process).

---

### 5. AI-PODEM Inference & Benchmarking
Evaluate the model's performance on complete circuits with support for different AI integration levels.

#### Standard Benchmark (Vanilla vs AI)
Compare standard PODEM against AI-assisted versions (Activation vs Propagation).

```bash
python -m scripts.benchmark_c432_compare
```

#### Debug / Single Fault Trace
Run a deep trace on a specific fault to visualize AI justification steps.

```bash
python -m scripts.debug_ai_podem_execution \
    data/bench/ISCAS85/c17.bench \
    "10-1" \
    --model checkpoints/reconv_model/best_model.pth
```

---

## 🏗️ Project Structure

| Component | Path | Description |
|:---|:---|:---|
| **Core Logic** | `src/atpg/` | PODEM, Logic Sim, and Reconvergent Solvers |
| **Reconv Cache** | `src/atpg/reconv_cache.py` | Disk-persisted reconvergent pair topology cache |
| **Model** | `src/ml/core/model.py` | Multi-Path Transformer with Cross-Attention |
| **Loss** | `src/ml/core/loss.py` | Differentiable Logic Consistency Loss |
| **Dataset** | `src/ml/core/dataset.py` | Sharded Data Management |
| **RL Recorder** | `src/ml/rl/rl_recorder.py` | Experience collection and episode recording |
| **Scripts** | `scripts/` | RL Pipeline and Benchmarking utilities |

---

## 🧪 Environmental Setup
Ensure you are using the `deepgate` conda environment:

```bash
conda activate deepgate
```

For more detailed developer documentation, see **[GUIDE.md](GUIDE.md)**.
