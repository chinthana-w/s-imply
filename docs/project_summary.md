# Project s-imply: Back Implication Prediction using Attention

## 1. Problem Description
The project addresses the challenge of **back implication prediction** in digital logic circuits. S-Imply (Structural Implication) focuses on **reconvergent path structures**, which are difficult for traditional ATPG algorithms because they require consistent logic assignments across multiple paths that fan out from a common stem and reconverge at a target node.

## 2. Current Approach
The project employs a **hybrid Reinforcement Learning (RL) + Supervised Learning** framework designed to "learn the intuition" of Boolean Satisfiability (SAT) over these specific structures.

- **Objective**: Learn a policy $\pi(a|s)$ that assigns logic values (0 or 1) to every node of a reconvergent path pair such that:
    1.  **Local Consistency**: All gate logic constraints (AND, OR, NOT, etc.) are satisfied.
    2.  **Reconvergence Consistency**: The paths agree on the value at the reconvergence node.
    3.  **Stem Consistency**: The paths agree on the value at the fanout stem.

## 3. Solution Architecture

### A. Model: Multi-Path Transformer (`src/ml/reconv_lib.py`)
The core model is a hierarchical Transformer designed to process multiple reconvergent paths simultaneously and allow them to exchange information.

1.  **Input Embeddings**:
    -   **Base Path Embedding**: 132-dimensional vector representing the node's structural role and values.
    -   **Explicit Gate Type Embedding**: A learnable 64-dimensional vector (`gate_type_emb`) representing the gate type (AND, OR, NOT, etc.), concatenated to the base embedding.
    -   **Positional Encoding**: Standard sinusoidal encoding to represent the sequential order of gates from stem to reconvergence.
2.  **Shared Path Encoder**:
    -   A standard Transformer Encoder (`shared_path_encoder`) that processes each path independently.
    -   Learns local sequential logic features (e.g., "inverter chain" or "control value propagation").
3.  **Path Interaction Layer**:
    -   A Transformer Encoder (`path_interaction_layer`) that operates on the **path summary tokens** (the first token of each path).
    -   Enables the model to understand the relationship between different branches (e.g., "Path A produces 0, so Path B must produce 1").
4.  **Cross-Attention Mechanism**:
    -   A `MultiheadAttention` block (`cross_attn`) where:
        -   **Query**: Individual node representations.
        -   **Key/Value**: The set of interaction-aware path summaries.
    -   Allows every node in every path to attend to the global context of the reconvergent structure.
5.  **Prediction Head**:
    -   A linear layer mapping the final node representations to logits for Logic 0 and Logic 1.

### B. Solver & ATPG Logic (`src/atpg/reconv_podem.py`)
Instead of a purely generative approach, the system relies on algorithmic solvers to identify structures and verify feasibility.

-   **Structure Identification**:
    -   `pick_reconv_pair`: Beam search to find reconvergent fanout structures.
    -   `find_shortest_reconv_pair_ending_at`: Backward BFS to find the closest common ancestor (stem) for a given reconvergence node.
-   **Consistency Checking (`PathConsistencySolver`)**:
    -   Verifies if a target value at the reconvergence node is logically possible.
    -   **Recursive Backtracking**: Uses `_backtrace_assignment` to recursively justify values backwards from the target node to the stem, checking for conflicts.
    -   **Side-Input Constraint Handling**: Explicitly handles constraints on "side inputs" (inputs to gates on the path that are not part of the path itself).

### C. Training Pipeline (`src/ml/train_reconv.py`)
The training uses a REINFORCE-based policy gradient approach with auxiliary losses to guide the model towards valid assignments.

-   **Loss Function (`policy_loss_and_metrics`)**:
    1.  **REINFORCE Loss**: Maximizes expected reward.
        -   **Reward Signal**: +1.0 for valid assignments (all constraints met), negative penalty proportional to errors otherwise.
    2.  **Weighted Soft Edge Constraints**: Differentiable penalty (`soft_edge_lambda`) for violations of local gate logic.
        -   **Class Balancing**: Uses a **12.0x weight** for NOT gate violations to counter their statistical rarity (<5% of dataset) and prevent "buffer bias".
    3.  **Reconvergence Consistency**: Penalty for variance in the predicted values at the reconvergence node.
    4.  **Anchor Support**:
        -   **Anchor Injection**: Procedurally injects a "target" value (Logic 0 or 1) at the reconvergence node input to guide the model.
        -   **Anchor Reward**: Bonus for predicting the injected anchor value correctly.
    5.  **Entropy Regularization**: Encourages exploration (`entropy_beta`).

## 4. Project Structure & Key Files

-   **`src/atpg/`**:
    -   `reconv_podem.py`: Primary logic for finding reconvergent paths and the validation solver (`PathConsistencySolver`).
    -   `logic_sim_three.py`: 3-valued logic simulator (0, 1, X) for fault simulation.
-   **`src/ml/`**:
    -   `reconv_lib.py`: PyTorch implementation of the `MultiPathTransformer`.
    -   `train_reconv.py`: Main training loop, loss calculation, and metric tracking.
    -   `reconv_ds.py`: Dataset loader for pre-processed circuit paths.
    -   `embedding_extractor.py`: Converts circuit netlists (BENCH/AIG) into embedding-ready tensors.
-   **`data/`**:
    -   `bench/`: ISCAS85/89, ITC99 benchmark circuits.
    -   `datasets/`: Serialized datasets.

## 5. Metrics & Validation
The following metrics are used to evaluate model performance:
-   **`valid_rate`**: The percentage of samples where the model generates a fully valid justification (0 edge violations, consistent stem/reconvergence).
-   **`edge_acc`**: The percentage of local gate input/output relations that are satisfied.
-   **`reconv_match_rate`**: The percentage of samples where all paths predict the same value for the reconvergence node.
-   **`anchor_match_rate`**: How often the model satisfies the injected anchor constraint.

## 6. Current Challenges & Roadmap
-   **Handling "Don't Cares" (X)**: The current model predicts binary 0/1. Integrating explicit X prediction or X-tolerance in the loss function is an ongoing area of research.
-   **Complex Reconvergence**: Scaling from pair-wise paths to N-ary reconvergent structures.
-   **Integration with Commercial ATPG**: Using the model's predictions as high-quality initial heuristics for industry-standard ATPG tools.
