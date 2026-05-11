# Benchmark Experiment Report

Run ID: `20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2`

Date: 2026-05-11 UTC

## Scope

Code slice 2 tested whether activation-derived internal assignments could guide
PODEM propagation without enabling model-driven propagation fallback. The code
adds a conservative static-hint backtracer that only chooses a fanin when the
AI activation assignment already contains a consistent value for that fanin;
otherwise it falls back to the existing simple backtrace heuristic.

## Validation

- `conda run -n deepgate python -m py_compile src/atpg/ai_podem.py scripts/benchmark_itc99_gate.py scripts/verify_train_test_reporting.py`
  passed.
- `conda run -n deepgate ruff check src/atpg/ai_podem.py scripts/benchmark_itc99_gate.py scripts/verify_train_test_reporting.py`
  passed.
- A bounded ITC99 smoke benchmark ran the first 5 faults from the deterministic
  10% gate with no fallback, `--compare-classic`, `--backtrack-target`,
  `--coverage-target 0.8`, and `--device cpu`.

## Artifacts

- Run manifest:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2/run_manifest.json`
- Benchmark command manifest:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2/benchmark_run_manifest.json`
- JSON report:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2/benchmark_report.json`
- Per-fault CSV:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2/benchmark_report.csv`
- Notion-ready dated log:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2/notion_result_summary.md`

## Result

The bounded smoke used
`checkpoints/iscas85_89_20260507_095012/best_model.pth` on the first five faults
from `data/bench/ITC99/b17_gate_10pct_faults.json`.

- Coverage: 5/5 faults detected, 100.0%.
- Coverage target: passed for this bounded smoke at the configured 80.0% target.
- AI backtracks on successful faults: 230.
- Classic PODEM backtracks on the same successful faults: 230.
- Backtrack target: failed, because AI did not use fewer backtracks than classic
  PODEM.
- Activation precheck: 0 zero-backtrack detections.
- Baseline comparison: not decision-comparable because `--limit-faults 5` is
  smaller than the full deterministic 10% ITC99 gate.

No full 10% gate or full ITC99 run was promoted from this slice because the
bounded backtrack target failed.

## Next Validation Step

Focus the next code slice on propagation-side search reduction for the two hard
PI faults in the first bounded gate. The full ITC99 benchmark should remain
blocked until a comparable gate artifact passes both coverage and
AI-backtracks-less-than-classic criteria.
