# Benchmark Experiment Report

Run ID: `20260511T150345Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-1`

Date: 2026-05-11 UTC

## Scope

Code slice 1 tightened the ITC99 gate benchmark artifact contract. The benchmark
now records deterministic AI attempts, AI backtracks, optional classic PODEM
comparison on the same faults, the backtrack target decision, and explicit
`--device` selection for CPU/CUDA reproducibility.

## Validation

- `conda run -n deepgate python -m py_compile scripts/benchmark_itc99_gate.py`
  passed.
- `conda run -n deepgate python -m ruff check scripts/benchmark_itc99_gate.py`
  passed.
- A bounded ITC99 smoke benchmark ran the first 2 faults from the deterministic
  10% gate with `--compare-classic`, `--backtrack-target`, `--ai-attempts 3`,
  and `--device cpu`.

## Artifacts

- Pre-run manifest:
  `runs/orchestration/20260511T150345Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-1/run_manifest.json`
- Benchmark command manifest:
  `runs/orchestration/20260511T150345Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-1/benchmark_run_manifest.json`
- JSON report:
  `runs/orchestration/20260511T150345Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-1/benchmark_report.json`
- Per-fault CSV:
  `runs/orchestration/20260511T150345Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-1/benchmark_report.csv`
- Notion-ready dated log:
  `runs/orchestration/20260511T150345Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-1/notion_result_summary.md`

## Result

The bounded smoke detected 2/2 faults with no fallback using
`checkpoints/iscas85_89_20260507_095012/best_model.pth`. This passed the 80%
coverage threshold for the bounded smoke only.

The backtrack target failed on this smoke: AI used 230 total backtracks and
classic PODEM used 230 total backtracks on the same two faults. The artifact
therefore does not justify a full ITC99 promotion claim.

The baseline comparison is marked not decision-comparable because the smoke used
`--limit-faults 2`, while the documented baseline is the prior
`unlinked_candidate 1% ITC99 gate` result in
`docs/checkpoint_compatibility_summary.md`.

## Next Validation Step

Use the new gate artifact fields to validate the next implementation candidate
on a larger deterministic gate slice. Promote to the full 10% ITC99 gate only
after the bounded slice passes both coverage and AI-backtracks-less-than-classic
criteria.
