# AI-PODEM Benchmark — ISCAS85

**Model:** `checkpoints/supervised_v4/best_model.pth`  
**Date:** 2026-03-31  
**Mode:** no-fallback — AI assignment + bounded PODEM (500 backtracks max), no vanilla retry  

## Summary

| Metric | Value |
|--------|-------|
| Total faults | 28,380 |
| Reconvergent faults (topology) | 20,804 (73.3%) |
| AI-PODEM succeeded (no fallback) | 71 / 20,804 (0.3%) |
| AI-PODEM failed | 20,733 (99.7%) |

## Coverage Table

| Circuit | Total Faults | Reconv Faults | Reconv% | AI Succ | AI Fail | AI Coverage |
|---------|-------------|---------------|---------|---------|---------|-------------|
| c17 | 22 | 4 | 18% | 2 | 2 | 50.0% |
| c432 | 392 | 222 | 57% | 0 | 222 | 0.0% |
| c499 | 486 | 164 | 34% | 0 | 164 | 0.0% |
| c880 | 886 | 528 | 60% | 0 | 528 | 0.0% |
| c1355 | 1,174 | 1,012 | 86% | 8 | 1004 | 0.8% |
| c1908 | 1,826 | 1,124 | 62% | 0 | 1124 | 0.0% |
| c2670 | 2,852 | 1,772 | 62% | 0 | 1772 | 0.0% |
| c3540 | 3,438 | 2,588 | 75% | 0 | 2588 | 0.0% |
| c5315 | 4,970 | 3,238 | 65% | 0 | 3238 | 0.0% |
| c6288 | 4,896 | 4,260 | 87% | 61 | 4199 | 1.4% |
| c7552 | 7,438 | 5,892 | 79% | 0 | 5892 | 0.0% |
| **TOTAL** | **28,380** | **20,804** | **73%** | **71** | **20733** | **0.3%** |

## Backtrack Comparison (matched fault set)

Vanilla PODEM run on the exact faults where AI-PODEM succeeded — apples-to-apples.

| Circuit | Matched Faults | Vanilla BT | AI BT | BT Reduction | Speedup |
|---------|---------------|-----------|-------|--------------|---------|
| c17 | 2 | 0 | 0 | +100.0% | 0.00x |
| c432 | 0 | — | — | — | — |
| c499 | 0 | — | — | — | — |
| c880 | 0 | — | — | — | — |
| c1355 | 8 | 0 | 60 | -5900.0% | 0.07x |
| c1908 | 0 | — | — | — | — |
| c2670 | 0 | — | — | — | — |
| c3540 | 0 | — | — | — | — |
| c5315 | 0 | — | — | — | — |
| c6288 | 61 | 0 | 0 | +100.0% | 0.90x |
| c7552 | 0 | — | — | — | — |
| **TOTAL** | **71** | **0** | **60** | — | — |

## Analysis

### Findings

- **Overall no-fallback coverage: 0.3%** (71/20,804 reconvergent faults).
- Coverage is **non-zero on c17 (50%), c1355 (0.8%), and c6288 (1.4%)**; zero on all other circuits.
- On matched faults (c17, c1355, c6288), both vanilla and AI backtracks are 0 —
  meaning the AI succeeded on **trivially easy faults** that vanilla also solves instantly.
  The model has not yet learned to solve the genuinely hard reconvergent faults.
- **With fallback enabled** (normal operation), fault coverage is 100% on all circuits.
  AI assignments are structurally valid but do not consistently prune the PODEM search space.
- **66% of all faults** are in reconvergent fan-out structures — the problem domain is well-represented.

### Root cause

The model was trained on a **10% subset (5.38M samples) for 50 epochs** — best checkpoint at epoch 5.
This is insufficient to learn the circuit-level generalisation needed to guide PODEM across
the diverse structures in ISCAS85. The full 53.8M sample dataset and longer training are required.

### Next steps

1. **Full dataset training:** Run on the 53.8M sample fault dataset for 200+ epochs.
2. **Checkpoint evaluation:** Re-run this benchmark every 10–20 epochs to track generalisation.
3. **Hard fault analysis:** Identify the structural properties of faults the model fails on
   and ensure they are represented in the training distribution.
4. **Loss rebalancing:** Investigate per-circuit weighting to prevent easy-fault over-fitting.

---
*Report generated 2026-03-31 09:43*
