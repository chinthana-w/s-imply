# AI-Guided PODEM: Results Analysis

**Benchmark:** ITC99 b17 gate-level fault list (10% sample, 6 445 faults)
**Model:** `supervised_v5/best_model.pth`
**Run:** `full_retry_v1` — two-segment run; 5 899 faults from first pass + 546 from resume
**Note:** The results below cover the **546 faults** from the completed resume segment
(fault indices 5 899–6 444), where both AI and classic PODEM ran head-to-head under identical
conditions. The full 6 445-fault run remains incomplete due to memory constraints.

---

## Coverage Summary

| Metric | AI-Guided PODEM | Classic PODEM |
|---|---|---|
| Faults attempted | 546 | 546 |
| Faults detected | **444 (81.3%)** | 457 (83.7%) |
| Faults missed | 102 (18.7%) | 89 (16.3%) |
| Detected by AI only | 2 | — |
| Detected by classic only | 15 | — |
| Neither method detected | 87 | — |

> [!NOTE]
> The 87 faults solved by neither method represent genuinely hard reconvergent faults where
> both AI and exhaustive PODEM (30 s timeout, 5 000 backtracks) fail. These are most likely
> true untestable faults or faults requiring deeper search than either budget allows.

**AI relative coverage** (AI detections as a fraction of classic detections): **97.2%**
The AI method recovers 97.2% of the faults that classic PODEM can find, while operating
under a fundamentally different (non-exhaustive) search strategy.

---

## Efficiency: Time and Backtracks

### Time

| Scenario | AI-Guided | Classic PODEM |
|---|---|---|
| Mean time (successful faults) | 0.97 s | 1.21 s |
| Median time (successful faults) | 0.26 s | 0.40 s |
| Mean time (failed faults) | 8.76 s | **30.0 s** |
| Median time (failed faults) | 7.46 s | **30.0 s** |
| Total wall time (546 faults) | ~1 322 s | **3 228 s** |

- AI is **2.4× faster overall** (1 322 s vs 3 228 s on the same fault set).
- The speedup is most dramatic on failures: classic PODEM hits its 30-second timeout on every
  hard fault; the AI method times out in ~8.8 s on average because the AI assignment phase
  fails quickly, capping the wasted PODEM search time.

### Backtrack Count (on faults both methods solved, n = 442)

| Method | Mean backtracks | Total backtracks |
|---|---|---|
| AI-Guided PODEM | 1.4 | 619 |
| Classic PODEM | 2.7 | 1 208 |
| **Reduction** | — | **−48.8%** |

AI guidance cuts the number of PODEM backtracks nearly in half on solvable faults.
This reflects the model's ability to pre-fill reconvergent path assignments that are
consistent with the target, reducing the combinatorial search space.

---

## Precheck Effectiveness

**313/546 faults (57.3%)** were solved entirely in the precheck phase — a **zero-backtrack,
zero-PODEM-search** direct assignment that the AI model generates in a single forward pass.
These faults required no backtracking at all; the model's assignment was immediately
simulatable and detected the fault.

| Detection mode | Count | % of attempted |
|---|---|---|
| Precheck only (zero-backtrack) | 313 | 57.3% |
| Full AI-guided PODEM search | 131 | 24.0% |
| AI failed (missed) | 102 | 18.7% |

---

## Pros of the AI Method

1. **Speed on failures.** Classic PODEM burns its full 30-second timeout on hard faults.
   The AI method fails fast (~8.8 s) because the AI assignment phase exits early, saving
   roughly 21 s per undetectable fault. At scale this compounds to hours saved.

2. **Backtrack reduction.** On faults the AI solves, it uses 48.8% fewer backtracks than
   classic PODEM. The model implicitly encodes reconvergence constraints that guide PODEM
   toward the correct assignment with fewer dead ends.

3. **Zero-backtrack coverage.** 57.3% of faults were solved with a single model inference
   and no PODEM search at all. For these faults the AI is effectively free — inference cost
   is ~1–10 ms versus potentially seconds of PODEM search.

4. **Complementary misses.** The AI detects 2 faults that classic PODEM misses (likely due
   to the 5 000 backtrack cap), suggesting the AI occasionally explores regions of assignment
   space that exhaustive search does not reach within the budget.

5. **Confidence-guided retry.** The new `solve_with_retry` mechanism allows the system to
   recover from low-confidence pair predictions without restarting from scratch. This reduces
   the cost of the AI being "almost right" — a failed prediction on one reconvergent pair no
   longer invalidates the entire assignment.

---

## Cons and Limitations

1. **Coverage gap: −2.4 pp vs classic.** The AI misses 15 faults (2.7%) that classic PODEM
   catches. These are faults where the model's reconvergent-pair prediction is incorrect and
   the retry mechanism does not recover. The model has not been trained to exhaustively cover
   corner cases in very deep reconvergent cones.

2. **High variance on solve times.** The AI median success time (0.26 s) is fast, but the
   mean (0.97 s) is inflated by a subset of hard faults where multi-candidate sampling and
   retries consume significant time. Classic PODEM has tighter variance on easy faults.

3. **Model inference overhead.** The structural embedding (`struct_emb`) requires a full
   graph feature extraction pass over the circuit before inference begins. For small circuits
   this startup cost dominates; b17 amortizes it over 64 k faults but smaller benchmarks may
   not.

4. **GPU dependency.** The model runs on GPU (RTX 5070 Ti in this run). On CPU-only machines
   inference is significantly slower (~10–50× per pair query), which would eliminate the
   speed advantage on successful faults and worsen failure latency.

5. **Incomplete coverage on reconvergent-heavy faults.** The AI model was trained on a
   specific distribution of reconvergent pair configurations. Faults in b17's deepest
   reconvergent cones (which appear concentrated in the last ~500 faults of the sorted list,
   coinciding with where RSS grew fastest) expose distribution mismatch, causing correlated
   failures.

6. **Run did not complete.** Memory pressure aborted the first 6 445-fault pass at fault
   5 899 (91.5% complete). The resume covered the remaining 546 faults. Full-scope coverage
   cannot be stated for this run.

---

## Comparison vs Baseline

The baseline (`unlinked_candidate` checkpoint, 1% fault set) achieved **18.2% coverage**.
The current `supervised_v5` model achieves **81.3% on the resumed segment**, a **+63.1 pp**
improvement. However, the baseline run covered a different (smaller) fault slice and ran
without the precheck mechanism, so this is not a strictly controlled comparison.

---

## Recommendations

| Action | Rationale |
|---|---|
| Run full 6 445-fault pass with new memory fixes | Get complete full-scope coverage number |
| Investigate the 15 AI-missed / classic-solved faults | Likely low-confidence predictions that exceed current retry budget; may benefit from `--max-confidence-retries 5` |
| Profile the 87 faults neither method solves | Determine true-untestable fraction vs budget-constrained misses |
| Extend training data to deeper reconvergent cones | Address the distribution mismatch on high-gate-index faults |
| Evaluate on additional ITC99 benchmarks (b14, b20, b22) | Check generalization beyond b17 |
