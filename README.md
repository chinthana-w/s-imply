# s-imply

A Topology-Aware Justification Oracle for digital circuits using Multi-Path Transformers and 3-Valued Logic reasoning.

## 🚀 Key Features
- **3-Valued Logic Reasoning**: Explicitly handles `0`, `1`, and `X` (Don't Care) logic states.
- **Topology-Aware Embeddings**: Maps physical gate identities across reconvergent paths for global consistency.
- **Physics-Informed Training**: Incorporates differentiable logic consistency loss to enforce Boolean truth tables.
- **Hybrid AI-PODEM**: Integrates AI-based justification directly into the PODEM backtrace loop.
- **Hierarchical Reconvergence**: Specialized `HierarchicalReconvSolver` for nested logic structures.

---

## 🛠️ Usage Guide

### 1. Data Preparation (Building Shards)
Build fault-driven SAT/UNSAT samples from ISCAS85/89 only, then convert the
chunked output into optimized tensor shards.  Keep `patterns-per-fault` small at
first because sample count scales with faults × patterns × reconvergent path
pairs.

```bash
python -m scripts.build_fault_dataset \
    --bench_dirs data/bench/ISCAS85 data/bench/iscas89 \
    --output /home/local1/cache-cw/iscas85_89_fault_chunks \
    --max_faults 0 \
    --patterns-per-fault 1 \
    --sim_attempts 50 \
    --unsat-ratio 0.10 \
    --max-samples-per-circuit 0 \
    --seed 42
```

```bash
python -m src.ml.core.dataset preprocess \
    --input /home/local1/cache-cw/iscas85_89_fault_chunks \
    --out /home/local1/cache-cw/iscas85_89_shards \
    --max_len 50 \
    --shard_size 5000 \
    --dtype float16 \
    --resume
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
    --lambda-supervised-node 2.0 \
    --lambda-solvability 0.5 \
    --ffn-dim 2048 \
    --model-dim 512 \
    --enc-layers 3 \
    --int-layers 3
```

### 3.1 ITC99 Held-Out Gate
ITC99 is not used for training or validation.  Materialize the deterministic
10% gate subset once, then benchmark checkpoints against that subset before any
full ITC99 run.

```bash
python -m scripts.select_itc99_gate_faults \
    --bench data/bench/ITC99/b17.bench \
    --output data/bench/ITC99/b17_gate_10pct_faults.json \
    --fraction 0.10 \
    --seed 20260504
```

```bash
python -m scripts.benchmark_itc99_gate \
    --model checkpoints/reconv_topology_3val_v1_ssd/best_model.pth \
    --fault-list data/bench/ITC99/b17_gate_10pct_faults.json \
    --out docs/itc99_gate_report.json \
    --csv-out docs/itc99_gate_per_fault.csv \
    --manifest-out docs/itc99_gate_run_manifest.json \
    --notion-summary-out docs/notion_result_summary.md \
    --candidate-count 8 \
    --ai-attempts 2 \
    --max-backtracks 5000 \
    --coverage-target 0.8 \
    --compare-classic \
    --backtrack-target
```

The benchmark report records per-fault outcomes, baseline metadata, command
provenance, AI/classic backtrack comparison, target pass/fail fields, and
optional Notion-ready markdown.  Bounded smoke runs using `--limit-faults`
validate the benchmark/reporting path only; do not treat them as a promotion
decision for the full 6,445-fault 10% gate.  Promote to full ITC99 only after a
comparable 10% gate artifact reaches at least 80% no-fallback coverage and uses
fewer AI backtracks than classic PODEM on the same faults.

Current strict no-fallback runs are intentionally narrow.  In this mode,
`ai_podem()` does not retry clean PODEM, model prediction does not add internal
fallback candidates, strict hint backtrace raises when hints cannot complete a
PI path, and propagation backtrace failure returns `UNTESTABLE`.  The ITC99 gate
benchmark also sets `max_backtracks=0`, so a passing fault means the AI
activation/hint path solved the fault without classic backtracking.  Until a
full 6,445-fault gate artifact proves otherwise, treat no-fallback AI coverage
as limited to simple faults that classic PODEM can solve with zero backtracks.

For a repo-local quick session, use the guarded wrapper:

```bash
COVERAGE_TARGET=0.8 COMPARE_CLASSIC=1 BACKTRACK_TARGET=1 \
    AI_ATTEMPTS=2 bash scripts/train_test_session.sh
```

It builds ISCAS85/89 data, trains, and writes ITC99 gate artifacts under
`docs/session_reports/$RUN_ID/` by default.  The wrapper checks disk/RAM
minimums before each stage and can resume selected stages with `STAGES=...`.
The wrapper exposes the backtrack gate as environment controls; without the
`COMPARE_CLASSIC=1` and `BACKTRACK_TARGET=1` overrides, a wrapper test run does
not prove the fewer-backtracks target.

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
python -m scripts.run_rl_pipeline --all \
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
