# S-Imply: Learning Reconvergent Back-Implication Heuristics for ATPG

**Target venue:** International Test Conference (ITC 2026)  
**Manuscript status:** Conference draft (Prism-ready Markdown)  
**Authors:** *[Author list omitted]*

---

## Abstract
Reconvergent fanout structures are a persistent source of complexity in automatic test pattern generation (ATPG), because assignments justified on one branch can be invalidated by constraints on another branch. This paper presents **S-Imply**, a learning-assisted framework for back-implication in reconvergent regions that combines (i) structural search and consistency checking derived from classical ATPG principles, and (ii) a transformer-based predictor trained with differentiable logic-consistency objectives. The method introduces explicit region-based checks over local reconvergent regions (LRRs), including exit-line-aware feasibility validation, and uses a multi-path attention architecture to model cross-branch dependencies. On benchmarked circuits, S-Imply demonstrates strong SAT/UNSAT discrimination (96.9% solvability accuracy) while maintaining high local logic consistency (91.3% edge accuracy). We also report diagnostic analysis and mitigation of deterministic-gate failure modes via post-processing, yielding substantial gains in zero-error assignment rate. These results suggest that learned reconvergent heuristics can be integrated into ATPG flows as a practical complement to algorithmic search.

**Keywords:** ATPG, reconvergent fanout, back implication, machine learning for EDA, transformer models, SAT/UNSAT classification

---

## 1. Introduction
Automatic test pattern generation remains fundamental to digital test, but scalability and robustness are challenged by structural phenomena such as reconvergent fanout. In these regions, a single stem drives diverging paths that reconverge downstream, producing strong non-local constraints that are difficult to satisfy with purely local heuristics.

Classical PODEM-style methods provide a principled search framework, yet their efficiency depends heavily on implication and justification quality. Our hypothesis is that a learned model can supply high-value guidance specifically in reconvergent regions, where branch interactions are difficult to capture with static heuristics.

This paper contributes:

1. A **region-aware consistency framework** based on local reconvergent regions (LRRs) and exit-line constraints for feasibility checking.
2. A **multi-path transformer** that jointly reasons over multiple reconvergent branches using cross-attention.
3. A **differentiable training objective** focused on logic consistency, reconvergence agreement, and solvability classification.
4. An **integration path to ATPG** demonstrating where learned back-implication can complement deterministic search.

---

## 2. Related Work
ATPG has long leveraged structural heuristics, implication engines, and SAT-based reasoning to handle hard-to-detect faults. Reconvergent structures are historically recognized as difficult due to masking effects and high interaction complexity. Recent ML-for-EDA efforts have shown promise in replacing or augmenting handcrafted heuristics, but often lack explicit mechanisms for reconvergence-specific consistency.

Our approach follows a hybrid philosophy: preserve deterministic feasibility checks while learning branch-interaction priors from data. This keeps the method aligned with ATPG correctness requirements and avoids unconstrained black-box decisions.

> **ITC style note:** Keep related work concise (problem-driven) and reserve detailed comparisons for experiments.

---

## 3. Problem Formulation
Given a reconvergent structure with a stem, multiple branch paths, and a reconvergence target, predict branch-wise logic assignments that satisfy:

- **Local gate consistency:** each gate relation on each path is valid;
- **Reconvergence consistency:** branch assignments agree at the reconvergent node;
- **Stem consistency:** branch assumptions remain compatible at the fanout stem;
- **Global feasibility:** assignments are not invalidated by region exits.

We define each sample as a structured path set with gate features, topology-aware identifiers, and a binary solvability label (SAT/UNSAT) for target assignments.

---

## 4. Method

### 4.1 Region-Aware Reconvergence Analysis
We use a structural engine to identify short reconvergent pairs and compute local reconvergent regions (LRRs). Exit lines (fanouts leaving the LRR) are explicitly tracked and checked during recursive backtrace. This prevents false feasibility caused by path-local but globally inconsistent assignments.

### 4.2 Multi-Path Transformer
The model consumes per-node features enriched with gate-type and node-identity embeddings. A shared transformer encoder captures intra-path dependencies, followed by a path-interaction encoder over pooled path summaries. A cross-attention stage lets each node representation attend to interaction-aware path context, producing:

- node-level logits for logic assignment;
- a global solvability prediction head.

### 4.3 Training Objective
Instead of sparse reward-only optimization, we use a weighted differentiable objective that penalizes logic-rule violations and inconsistency at structural anchors (stem/reconvergence), with an auxiliary solvability loss. This design improves gradient quality and training stability for discrete logic decisions.

### 4.4 ATPG Integration
The predictor is used as a guidance module within a deterministic ATPG workflow. Structural solvers remain authoritative for acceptance/rejection of assignments; learned predictions bias search toward promising justifications.

---

## 5. Experimental Setup

### 5.1 Benchmarks and Data
Experiments use standard benchmark suites including ISCAS and ITC families. Data instances are constructed from reconvergent path structures with SAT/UNSAT labeling via consistency checks.

### 5.2 Metrics
We report:

- **solv_acc** (SAT/UNSAT accuracy),
- **edge_acc** (local gate-consistency accuracy),
- **valid_rate** (fully valid assignment rate),
- **false_unsat_rate / true_unsat_rate** (UNSAT discrimination behavior),
- throughput indicators where relevant.

### 5.3 Implementation Notes
The training/data pipeline uses lazy shard loading for large datasets and multi-worker loading. This is included for reproducibility and to separate algorithmic gains from data-pipeline artifacts.

---

## 6. Results

### 6.1 Core Reconvergence Consistency
On the reported evaluation snapshot, S-Imply achieves:

- **96.9% solvability accuracy**,
- **91.3% edge accuracy**,
- improved assignment behavior over random-match baselines.

These results indicate that the model captures branch-interaction signals relevant to SAT/UNSAT distinction while preserving local logic consistency.

### 6.2 AI-Assisted ATPG Behavior
In AI-assisted PODEM tests, fault coverage and runtime behavior vary by circuit complexity and baseline sensitivity. Small circuits show parity with vanilla flows; larger designs indicate modest runtime overhead with preserved coverage in tested subsets.

### 6.3 Error Analysis and Deterministic Gate Repair
Diagnostic analysis identified NOT/BUFF stages as dominant residual error sources under non-autoregressive decoding. A lightweight forward post-processing pass for deterministic gates significantly improved zero-error assignments and reduced edge errors.

---

## 7. Discussion
The empirical trend supports a **hybrid ATPG strategy**: deterministic solvers enforce correctness while learned models prioritize likely-consistent decisions in reconvergent regions. Two caveats remain:

1. Binary 0/1 prediction does not yet fully model X (unknown) semantics.
2. Pairwise reconvergence modeling should be generalized to wider N-branch structures.

These are key directions for production deployment.

---

## 8. Threats to Validity
- **Benchmark representativeness:** subset-based evaluations may not cover all industrial logic patterns.
- **Implementation coupling:** ATPG baseline quality influences measured incremental gains.
- **Labeling assumptions:** SAT/UNSAT labels depend on solver fidelity and region definitions.

---

## 9. Conclusion
S-Imply demonstrates that reconvergence-focused learning can improve back-implication quality without relaxing ATPG correctness constraints. By combining region-aware structural checks with multi-path attention, the framework delivers strong solvability discrimination and logic consistency on benchmarked circuits. Future work targets X-aware modeling, broader reconvergence topologies, and tighter integration with industrial test flows.

---

## Acknowledgment (Optional)
*To be completed for submission.*

## References (Placeholder Format)
> Replace with BibTeX/citation keys in your Prism pipeline.

[1] J. P. Roth, “Diagnosis of automata failures: A calculus and a method,” *IBM Journal*, 1966.  
[2] P. Goel, “An implicit enumeration algorithm to generate tests for combinational logic circuits,” *IEEE TC*, 1981.  
[3] M. Abramovici, M. A. Breuer, and A. D. Friedman, *Digital Systems Testing and Testable Design*.  
[4] *Recent ITC papers on ML-guided ATPG and SAT-assisted test generation* (insert exact citations).

---

## Appendix A: ITC-Oriented Content Filtering Checklist
Use this checklist before camera-ready export:

- Keep only claims backed by reported metrics/tables.
- Move debug chronology and implementation incident logs to internal docs.
- Convert “known issue” statements into controlled limitations with evidence.
- Ensure every percentage in text appears in a figure/table or reproducibility artifact.
- Keep method section algorithmic and reproducible; avoid product-style language.
- Use neutral comparative wording unless statistical significance is shown.
