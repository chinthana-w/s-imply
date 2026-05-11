# Benchmark Experiment Report

Run ID: `20260511T150345Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2`

Date: 2026-05-11 UTC

## Scope

Code slice 2 tightened the ITC99 benchmark gate so coverage claims can be
checked against classic PODEM backtracks in the same artifact. The benchmark can
now run deterministic no-fallback AI retries, optionally probe AI propagation,
record classic PODEM comparison metrics, and fail the backtrack target without
losing per-fault provenance.

## Validation

- `conda run -n deepgate python -m py_compile scripts/benchmark_itc99_gate.py scripts/verify_train_test_reporting.py`
  passed.
- `conda run -n deepgate ruff check scripts/benchmark_itc99_gate.py scripts/verify_train_test_reporting.py`
  passed.
- `bash -n scripts/train_test_session.sh` passed.
- `conda run -n deepgate python -m scripts.verify_train_test_reporting` passed.
- Bounded ITC99 smoke artifacts were generated with JSON, CSV, benchmark
  manifest, and Notion-ready markdown.

## Benchmark Artifacts

- Run manifest:
  `runs/orchestration/20260511T150345Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2/run_manifest.json`
- Main bounded benchmark JSON:
  `runs/orchestration/20260511T150345Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2/benchmark_report.json`
- Main bounded per-fault CSV:
  `runs/orchestration/20260511T150345Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2/benchmark_report.csv`
- Benchmark command manifest:
  `runs/orchestration/20260511T150345Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2/benchmark_run_manifest.json`
- Notion-ready result summary:
  `runs/orchestration/20260511T150345Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2/notion_result_summary.md`
- AI propagation probe JSON:
  `runs/orchestration/20260511T150345Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2/propagation_probe_report.json`

## Results

The main bounded smoke used
`checkpoints/iscas85_89_20260507_095012/best_model.pth` on the first five faults
from `data/bench/ITC99/b17_gate_10pct_faults.json`, with two deterministic AI
attempts, no fallback, and classic PODEM comparison enabled.

- Coverage: 5/5 faults detected, 100.0%.
- Coverage target: passed for the bounded smoke at the configured 80.0% target.
- AI backtracks: 230 total on successful faults.
- Classic PODEM backtracks: 230 total on the same successful faults.
- Backtrack target: failed, because AI did not use fewer backtracks than
  classic PODEM.
- Baseline comparison: not decision-comparable because `--limit-faults 5` is
  smaller than the documented baseline scope.

The optional AI propagation probe was worse on the same bounded scope:
0/5 faults detected. No full 10% gate or full ITC99 run was promoted from this
slice because the bounded backtrack target failed.

## Next Validation Step

Use the activation-only path as the current candidate and focus the next code
slice on reducing backtracks before running the full 10% ITC99 gate. The full
ITC99 benchmark should remain blocked until a comparable gate artifact passes
both coverage and backtrack criteria.
