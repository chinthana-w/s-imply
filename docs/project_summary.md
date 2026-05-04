# Project s-imply: Back Implication Prediction using Attention

## 1. Problem Description
The project addresses the challenge of **back implication prediction** in digital logic circuits. S-Imply (Structural Implication) focuses on **reconvergent path structures**, which are difficult for traditional ATPG algorithms because they require consistent logic assignments across multiple paths that fan out from a common stem and reconverge at a target node.

## 2. Current Approach
The project employs a **differentiable, policy-gradient-style framework** to "learn the intuition" of Boolean Satisfiability (SAT) over these specific structures. While inspired by Reinforcement Learning, it replaces the sparse reward signal of traditional RL with a rich, end-to-end differentiable loss function, enabling more stable and efficient training.

- **Objective**: Learn a policy $\pi(a|s)$ that assigns logic values (0 or 1) to every node of a reconvergent path pair such that:
    1.  **Local Consistency**: All gate logic constraints (AND, OR, NOT, etc.) are satisfied.
    2.  **Reconvergence Consistency**: The paths agree on the value at the reconvergence node.
    3.  **Stem Consistency**: The paths agree on the value at the fanout stem.

## 3. Solution Architecture

### A. Model: Multi-Path Transformer (`src/ml/core/model.py`)
The core model is a hierarchical Transformer designed to process multiple reconvergent paths simultaneously and allow them to exchange information.

1.  **Input Embeddings**: The model constructs a rich input representation by concatenating multiple feature vectors.
    -   **Base Feature Embedding**: An initial feature vector derived from the dataset, including GNN-based embeddings from DeepGate. The dimension is inferred at runtime.
    -   **Gate Type Embedding**: A learnable 64-dimensional vector (`gate_type_emb`) representing the gate's logical function (AND, OR, NOT, etc.).
    -   **Node ID Embedding**: A learnable 64-dimensional vector (`node_emb`) representing the physical ID of the gate. This gives the model a strong sense of topological identity, allowing it to distinguish between two different AND gates in the circuit.
    -   **Positional Encoding**: Standard sinusoidal encoding is applied to the final projected embedding to represent the sequential order of gates along a path.
2.  **Shared Path Encoder**:
    -   A standard Transformer Encoder (`shared_path_encoder`) that processes each path independently.
    -   Learns local sequential logic features (e.g., "inverter chain" or "control value propagation").
3.  **Path Interaction Layer**:
    -   A Transformer Encoder (`path_interaction_layer`) that operates on the **path summary tokens** (aggregated from each path via max-pooling).
    -   Enables the model to understand the relationship between different branches (e.g., "Path A produces 0, so Path B must produce 1").
4.  **Cross-Attention Mechanism**:
    -   A `MultiheadAttention` block (`cross_attn`) where:
        -   **Query**: Individual node representations from the shared path encoder.
        -   **Key/Value**: The set of interaction-aware path summaries.
    -   Allows every node in every path to attend to the global context of the entire reconvergent structure.
5.  **Prediction Heads**:
    -   **Logic Head**: A linear layer mapping final node representations to logits for Logic 0 and Logic 1.
    -   **Solvability Head**: A global linear head that predicts whether the entire structure is SAT or UNSAT (Solvable vs Impossible) based on the pooled interaction vector.

### B. Solver & ATPG Logic (`src/atpg/reconv_podem.py`)
Instead of a purely generative approach, the system relies on algorithmic solvers to identify structures and verify feasibility.

-   **Structure Identification**:
    -   `pick_reconv_pair`: Enhanced beam search with a dominance-based heuristic. Prioritizes "tight" reconvergent loops by penalizing nodes with high external fanout (Exit Lines).
    -   `find_shortest_reconv_pair_ending_at`: Backward BFS to find the closest common ancestor (stem).
    -   **Maamari Concepts (Maamari & Rajski, 1990)**:
        -   **Local Reconvergent Region (LRR)**: Formally defined as the intersection of nodes reachable from the stem and nodes that can reach the reconvergence point.
        -   **Exit Lines**: Fanouts of LRR nodes that leave the region. These are tracked as critical "logic leakage" points.
-   **Consistency Checking (`PathConsistencySolver`)**:
    -   Verifies if a target value at the reconvergence node is logically possible.
    -   **Recursive Regional Consistency**: Uses an optimized `_backtrace_assignment` that checks assignments against constraints on **Exit Lines**. This prevents the solver from accepting locally valid paths that are globally impossible due to path masking.
    -   **Performance Optimization**: Uses an `exit_map` (dictionary-based lookup) to maintain stable O(1) performance during deep recursion, even on complex circuits like `c6288`.
-   **Recursive Justification (`HierarchicalReconvSolver`)** (`src/atpg/recursive_reconv_solver.py`):
    -   Utilizes LRR boundaries to prune justification queues, keeping the solver focused on Primary Inputs (PIs) and Exit Lines that directly influence the target reconvergence result.
    -   **Just-In-Time (JIT) Backtrace**: Replaced upfront path resolution with a lazy algorithm driven by dynamic PI verification queues. Predicts logic only when trace paths topologically intersect a path pair's reconvergent terminus, scaling to wider arrays without upfront enumeration.
-   **Configurable Backtracking** (`src/atpg/ai_podem.py`):
    -   `max_backtracks` parameter limits backtracking in the PODEM main loop, preventing runaway search on unsolvable faults.
    -   Fault values use D/D-bar (DB) representation for correct stuck-at fault propagation.
-   **Reconvergent Pair Cache** (`src/atpg/reconv_cache.py`):
    -   Disk-persisted cache storing reconvergent pair topology (node IDs and paths) per `.bench` file.
    -   Pairs are a pure function of circuit topology and never change between runs; the cache avoids expensive BFS traversals on repeated collection passes.
    -   Stored in `.reconv_cache/` subdirectory alongside each `.bench` file.

### C. Training Pipeline (`src/ml/train.py`)
The training pipeline is designed for end-to-end differentiable training of the policy using the **Gumbel-Softmax estimator**. This technique allows gradients to flow through the discrete 0/1 sampling process, enabling the direct optimization of logic consistency.

-   **Loss Function (`reinforce_loss`)** (`src/ml/core/loss.py`): The total loss is a weighted sum of multiple components designed to guide the model towards logically valid assignments.
    1.  **Differentiable Logic Losses**: The primary learning signal. Instead of a sparse reward, the model is penalized for logic violations directly.
        -   **Soft Edge Loss (`soft_edge_lambda`)**: A differentiable penalty based on the Gumbel-Softmax one-hot outputs for violations of local gate logic. NOT and BUFF gate violations carry a **2x weight** to counter their deterministic nature.
        -   **Full Path Logic Loss (`lambda_full_logic`)**: An auxiliary loss that penalizes all edge-level gate logic violations along the entire path (`calculate_full_logic_loss`). NOT/BUFF also weighted 2x here.
    2.  **Reconvergence Consistency Loss**: An MSE-based penalty on the logits at the reconvergence node to ensure all paths predict the same value.
    3.  **Anchor Supervision Loss**: A standard Cross-Entropy loss that trains the model to predict a pre-verified "anchor" value at a specific node, providing a strong supervised signal for solvable (SAT) cases. Only applied to samples where `solvability_labels == 0` (SAT).
    4.  **Constrained Curriculum Loss**: A Cross-Entropy loss applied to a subset of nodes that are temporarily "constrained" with ground-truth values from a full logic simulation.
    5.  **Solvability Loss**: A weighted Cross-Entropy loss (UNSAT:SAT weight = 10:1) for the auxiliary `solvability_head`, which predicts whether a structure is SAT or UNSAT.
    6.  **Entropy Regularization (`entropy_beta`)**: Encourages exploration by penalizing overly confident (low-entropy) probability distributions.

-   **Curriculum Learning**: The training process incorporates several curriculum strategies to ease the model into the complex task.
    -   **Constrained Curriculum**: When enabled (`--constrained_curriculum`), the training follows a schedule where the first **25%** of epochs are constraint-free, after which the probability of applying ground-truth constraints ramps up linearly to `max_constraint_prob` over the remaining **75%** of epochs.
    -   **Length Curriculum**: The dataset can be filtered by maximum path length (`--max-len`), allowing the model to be trained on shorter, easier problems first.

-   **Gumbel-Softmax Temperature Annealing**: Temperature decays per epoch as `τ_t = τ₀ × α^(epoch-1)` (default α=0.99, τ₀=1.0, minimum 0.1).

-   **Memory Optimizations**:
    -   `max_paths` truncation per sample (default 200) to prevent OOM on dense circuits.
    -   Gradient checkpointing (`--checkpointing`) for VRAM-constrained runs.
    -   Shard-based lazy dataset loading with configurable LRU cache (`shard_cache_size`).
    -   AMP (`--amp`) with GradScaler and gradient clipping (max norm 1.0).

### D. DeepGate Integration (`src/ml/data/embedding.py`)
The project integrates **DeepGate**, a Graph Neural Network (GNN)-based model, to provide high-fidelity circuit embeddings.

1.  **Structural & Functional Embeddings**: DeepGate generates 128-dimensional embeddings for every node in the circuit.
    -   **Structural Embedding (`hs`)**: Captures the topological context of a gate.
    -   **Functional Embedding (`hf`)**: Captures the logical role and input-output relationships.
2.  **Environment Management**: DeepGate runs in its own Conda environment (`deepgate`) and is dynamically imported into the S-Imply pipeline via manual `sys.path` configuration.
3.  **Circuit Pre-processing**: Circuits are converted to AIG (And-Inverter Graph) format before being passed to DeepGate's `BenchParser`.
4.  **Disk Caching**: Embeddings are cached per circuit in `.deepgate_cache/` subdirectories to avoid redundant inference across runs.

### E. RL Training Pipeline (`scripts/`)
An offline RL pipeline supplements supervised training with experience collected from live AI-PODEM runs.

-   **Experience Collection** (`scripts/collect_experience.py`):
    -   Runs AI-PODEM on ISCAS85/89 benchmark circuits, recording model inputs and outcomes at each solver call.
    -   Uses `ExperienceRecorder` (`src/ml/rl/rl_recorder.py`) to buffer and periodically flush episodes to disk as `batch_*.pkl` files.
    -   **Reward shaping**: `+10 - 0.001×backtracks` for success, `-5 - 0.001×backtracks` for failure, `-10` for exceptions.
    -   Supports multi-GPU parallelism via `run_rl_pipeline.py`, which distributes benchmark files across GPUs.
    -   `HierarchicalReconvSolver` is passed the `recorder` instance so it can log step-level data. Pair topology is persisted to disk cache (`reconv_cache.py`) after each circuit.

-   **RL Model Training** (`scripts/train_rl.py`):
    -   Loads experience batches from disk via `ExperienceDataset` with LRU file-level caching.
    -   Reconstructs DeepGate embeddings from `EmbeddingRegistry` during collation.
    -   Uses classic REINFORCE (policy gradient): `loss = -log_prob(action) × advantage`.
    -   Advantages are reward-normalized per batch.
    -   Entropy bonus (`entropy_beta`) encourages exploration.
    -   Saves best checkpoint by lowest training loss.

-   **Pipeline Orchestration** (`scripts/run_rl_pipeline.py`):
    -   Three stages selectable independently: `--collect`, `--train`, `--benchmark`.
    -   `--all` runs the full pipeline end-to-end.
    -   Automatically caps parallel GPU processes based on available RAM (assumes ~5 GB per process).

---

## 4. Project Structure & Key Files

-   **`src/atpg/`**:
    -   `podem.py`: Standard PODEM implementation with SCOAP heuristics and D/DB fault propagation.
    -   `reconv_podem.py`: Beam-search reconvergent pair discovery and `PathConsistencySolver`.
    -   `ai_podem.py`: AI-augmented PODEM integrating `HierarchicalReconvSolver` into the backtrace loop. Includes `post_process_logic_gates()` for deterministic gate correction.
    -   `recursive_reconv_solver.py`: `HierarchicalReconvSolver` with JIT backtrace and PI verification queues.
    -   `reconv_cache.py`: Disk-persisted cache for reconvergent pair topology per `.bench` file.
    -   `logic_sim_three.py`: 3-valued logic simulator (0, 1, X, D, DB) for fault simulation.
-   **`src/ml/`**:
    -   `core/model.py`: PyTorch implementation of `MultiPathTransformer`.
    -   `core/loss.py`: `reinforce_loss`, `calculate_consistency_loss`, `calculate_full_logic_loss`.
    -   `core/dataset.py`: `ReconvergentPathsDataset` with sharded lazy loading, anchor injection, and constraint injection.
    -   `data/embedding.py`: DeepGate integration — AIG conversion, 128D embedding extraction, disk caching.
    -   `rl/rl_recorder.py`: `ExperienceRecorder` for buffering and saving RL episodes.
    -   `rl/env.py`: RL environment definition.
    -   `train.py`: Main supervised training loop with curriculum, AMP, gradient accumulation config, and LR scheduling.
-   **`scripts/`**:
    -   `collect_experience.py`: AI-PODEM rollout collection with exploration.
    -   `train_rl.py`: REINFORCE-based RL fine-tuning from collected experience.
    -   `run_rl_pipeline.py`: Unified pipeline orchestrator (collect → train → benchmark).
    -   `benchmark_c432_compare.py`: Vanilla vs AI-PODEM performance comparison.
-   **`data/`**:
    -   `bench/ISCAS85/`, `bench/iscas89/`, `bench/ITC99/`: Benchmark circuit netlists.
    -   `datasets/reconv_dataset.pkl`: Serialized raw dataset.
    -   `datasets/reconv_shards_v3/`: Pre-processed tensor shards for lazy loading.

## 5. Metrics & Validation
The following metrics are used to evaluate model performance:
-   **`valid_rate`**: The percentage of samples where the model generates a fully valid justification (0 edge violations, consistent stem/reconvergence).
-   **`edge_acc`**: The percentage of local gate input/output relations that are satisfied across all paths.
-   **`constraint_violation_rate`**: The percentage of nodes that violate their ground-truth values when the constrained curriculum is active.
-   **`reconv_match_rate`**: The percentage of samples where all paths predict the same value for the reconvergence node.
-   **`anchor_match_rate`**: How often the model satisfies the injected anchor constraint for solvable (SAT) cases.
-   **`solv_acc`**: Accuracy of predicting whether a target is logically solvable (SAT) or impossible (UNSAT).
-   **`false_unsat_rate`**: The frequency of incorrectly predicting UNSAT for a solvable case.
-   **`true_unsat_rate`**: The frequency of correctly identifying an UNSAT case.

## 6. Experimental Results
### A. SAT/UNSAT Consistency (Maamari Update)
**Timestamp: 2026-02-02**
Following the integration of Regional Consistency and LRR-based labeling, the model demonstrates high fidelity in identifying impossible targets:
-   **Solvability Accuracy (`solv_acc`)**: **96.9%**
-   **Path Logic Consistency (`edge_acc`)**: **91.3%**
-   **Logic Prediction Accuracy (`acc`)**: **55.2%** (Baseline improvement over 50% random-init sequence matching).
-   **Throughput**: **16 batches/sec** (Optimized `exit_map` in solver resolved previous deadlocks in complex ISCAS85 circuits).

### B. AI-Assisted PODEM Benchmarking
**Timestamp: 2026-02-03**
Integration of DeepGate embeddings and AI-assisted justification/propagation evaluated on ISCAS85:

| Circuit | Mode | Faults | FC (%) | Avg Time/Fault (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **c17** | Vanilla | 22 | 100% | 0.09 |
| **c17** | AI-All | 22 | 100% | 0.70 |
| **c2670** | Vanilla | 50 | 100% | 119.90 |
| **c2670** | AI-All | 50 | 100% | 122.37 |
| **c432** | AI-All | 50 | 8.0%* | 25.78 |

*\*Note: Low FC on c432 is a known issue in the base PODEM implementation related to XOR logic handling and D-frontier sorting, currently under investigation.*

### C. Post-Processing Logic Gate Fix
**Timestamp: 2026-02-12**
Diagnostic analysis of 185-epoch trained model revealed NOT gates as sole failure mode (40.9% error rate, 89% of all edge errors). Model outputs p≈0.50 at NOT gates due to non-autoregressive architecture unable to condition on previous predictions. Forward-propagation post-processing (`post_process_logic_gates()`) in `ai_podem.py` fixes NOT/BUFF deterministic gates:

| Metric | Before | After | Δ |
| :--- | :--- | :--- | :--- |
| **Zero-error rate** | 75.5% | **96.8%** | **+21.3%** |
| **Edge accuracy** | 98.1% | **99.8%** | +1.7% |
| **Edge errors (1280 samples)** | 518 | **52** | **-90%** |
| **Reconvergence failures** | 0 | 0 | ±0 |

### D. Training Pipeline Repair & Optimization
**Timestamp: 2026-02-17**
Resolved critical pipeline failures and implemented SSD-optimized data loading:
-   **Lazy Loading (`ReconvergentPathsDataset`)**: Implemented on-demand shard loading to handle massive datasets without exhausting RAM. The dataset now loads only metadata initially and fetches tensor data from disk-cached shards during iteration.
-   **Pipeline Stabilization**: Fixed circular dependencies and missing helper functions (`_generate_anchor`, `resolve_gate_types`) that were causing `ImportError` failures in `src.ml.train`.
-   **Performance**: Optimized `gate_mapping` conversion with integer-key pre-checks, reducing initialization time. Multi-worker data loading verified to work correctly with lazy loading logic.

### E. Just-In-Time Architecture & AI Consistency Enforcement
**Timestamp: 2026-02-20**
Rewrote `HierarchicalReconvSolver` to replace upfront path resolution with a "Just-In-Time" backtrace algorithm driven by dynamic PI verification queues.
-   **Dynamic Instantiation:** Predicts logic only when trace paths topologically intersect a path pair's reconvergent terminus, scaling effortlessly to wider arrays.
-   **Context Merging:** Plumbed `ai_podem.py` constraints implicitly through to model tensor structures (`batch_embs[128:130]`) alongside deterministic fallback verifications over XOR/XNOR topologies.
-   **Logic Tying:** Tested via benchmark validation on `c432.bench 329-0`. The backtracking model natively rejects hallucinated outputs yielding unresolvable constraints upstream, collapsing conflicting trees logically while accepting sound traces dynamically.

### F. RL Training Pipeline & PODEM Hardening
**Timestamp: 2026-02-24**
Initiated comprehensive RL training rerun and stabilized the PODEM core:
-   **Datasets**: Combined `ISCAS85` and `iscas89` benchmarks (17 circuits total).
-   **Configuration**: 1000 faults per circuit (500 per GPU), 50 training epochs, batch size 4096, max paths 250. Hardware: 2× 16 GB GPUs.
-   **PODEM Hardening**: Added configurable `max_backtracks` to the PODEM main loop. Updated fault representation to D/DB notation for correct stuck-at propagation.
-   **Reconvergent Pair Cache**: Added `reconv_cache.py` to persist pair topology to disk, eliminating repeated BFS traversals on subsequent collection passes.
-   **Goal**: Improve the model's ability to handle complex reconvergence in sequential-like structures (ISCAS89) and further reduce `✗ Convergence Path Invalid` failures observed in dense combinational logic (`c1908`).

### G. ISCAS Training / ITC99 Gate Split
**Timestamp: 2026-05-04**
Updated the training workflow to keep ITC99 fully held out while improving
fault/pattern trace supervision:
-   **Training Data**: `build_fault_dataset.py` now records compact fault,
    pattern, and path-pair provenance for ISCAS85/89 samples, supports capped
    patterns per fault, and can add corrupted-constraint UNSAT local queries.
-   **Shard Format**: Processed shards now preserve per-node SAT label masks,
    constraint tensors, solvability labels using `0=SAT, 1=UNSAT`, and compact
    fault/pattern/sample IDs without duplicating circuit objects.
-   **Training Loss**: `src/ml/core/loss.py` adds supervised per-node CE for SAT
    labels while excluding UNSAT samples from node supervision; physics losses
    remain auxiliary.
-   **Held-Out Gate**: `scripts/select_itc99_gate_faults.py` materializes a
    deterministic 10% subset of ITC99 `b17.bench` (6,445 / 64,458 faults), and
    `scripts/benchmark_itc99_gate.py` benchmarks checkpoints against that gate
    before any full ITC99 run.
-   **Inference Diversity**: `ModelPairPredictor` replaces timestamp randomness
    with deterministic multi-candidate decoding, so repeated runs are
    reproducible while still checking multiple candidate patterns per logic cone.

## 7. Current Challenges & Roadmap
-   **Handling "Don't Cares" (X)**: The current model predicts binary 0/1. Integrating explicit X prediction or X-tolerance in the loss function is an ongoing area of research.
-   **Complex Reconvergence**: Scaling from pair-wise paths to N-ary reconvergent structures.
-   **Integration with Commercial ATPG**: Using the model's predictions as high-quality initial heuristics for industry-standard ATPG tools.
-   **Remaining Edge Errors (~3.2%)**: Post-processing fixes NOT/BUFF but AND/OR/NAND/NOR inequality violations remain. Consider iterative refinement or autoregressive decoding for further improvement.
-   **XOR/XNOR Handling**: Low fault coverage on c432 due to XOR logic in the D-frontier. Under active investigation.
