# Benchmark Experiment Report

Run ID: `20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-1`

Date: 2026-05-11 UTC

## Scope

Code slice 1 added an activation-pattern precheck to the ITC99 gate benchmark.
The benchmark now records whether an AI-generated activation assignment directly
detects a fault with zero PODEM backtracks before falling through to the existing
no-fallback AI-PODEM path. The precheck can be disabled with
`--no-activation-precheck` for comparison runs.

## Validation

- `conda run -n deepgate python -m py_compile scripts/benchmark_itc99_gate.py`
  passed.
- `conda run -n deepgate ruff check scripts/benchmark_itc99_gate.py` passed.
- A bounded ITC99 smoke benchmark ran the first 20 faults from the deterministic
  10% gate with CPU execution, two deterministic AI attempts, classic PODEM
  comparison, the activation precheck, and the 80% coverage target.

## Artifacts

- Pre-run manifest:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-1/run_manifest.json`
- Benchmark command manifest:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-1/benchmark_run_manifest.json`
- JSON report:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-1/benchmark_report.json`
- Per-fault CSV:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-1/benchmark_report.csv`
- Notion-ready dated log:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-1/notion_result_summary.md`

## Result

The bounded smoke used
`checkpoints/iscas85_89_20260507_095012/best_model.pth` on the first 20 faults
from `data/bench/ITC99/b17_gate_10pct_faults.json`.

- Coverage: 20/20 faults detected, 100.0%.
- Coverage target: passed for the bounded smoke at the configured 80.0% target.
- Activation precheck: 0 zero-backtrack detections.
- AI backtracks: 237 total on successful faults.
- Classic PODEM backtracks: 237 total on the same successful faults.
- AI/classic backtrack ratio: 1.0.
- Backtrack target: failed, because AI did not use fewer backtracks than
  classic PODEM.
- Baseline comparison: not decision-comparable because `--limit-faults 20` is
  smaller than the documented baseline scope.

No full 10% gate or full ITC99 run was promoted from this slice because the
bounded backtrack target failed.

## Notion

The Notion-ready dated log was generated at the artifact path above. The
connector update was not completed in this run because the Notion MCP update
call was cancelled before applying the page edit.

## Next Validation Step

Use the new per-fault `precheck_success`, `precheck_pi_assignments`,
`ai_backtracks`, and `classic_backtracks` fields to focus the next code slice on
faults where classic PODEM has positive backtracks and AI currently ties it.
