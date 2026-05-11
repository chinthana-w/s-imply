# Benchmark Experiment Report

Run ID: `20260508T214226Z-make-iterative-improvements-to-the-train-test-pi-coding-agent-1`

Date: 2026-05-09 UTC

## Scope

Code slice 1 improved the train/test benchmark pipeline artifact contract. The
ITC99 benchmark script now emits command provenance, baseline comparison
metadata, per-fault CSV output, a run manifest, and a Notion-ready dated log
entry. The train/test wrapper now calls the benchmark through `python -m` and
writes the same artifacts during the `test` stage.

## Validation

- `conda run -n deepgate python -m py_compile scripts/benchmark_itc99_gate.py`
  passed.
- `conda run -n deepgate python -m ruff check scripts/benchmark_itc99_gate.py`
  passed.
- `bash -n scripts/train_test_session.sh` passed.
- No-stage wrapper smoke passed with temp paths and no training/benchmark stages.
- Bounded ITC99 smoke benchmark passed and wrote JSON, CSV, manifest, and
  Notion-ready summary artifacts.

## Benchmark Artifact

- Command provenance:
  `runs/orchestration/20260508T214226Z-make-iterative-improvements-to-the-train-test-pi-coding-agent-1/benchmark_run_manifest.json`
- JSON report:
  `runs/orchestration/20260508T214226Z-make-iterative-improvements-to-the-train-test-pi-coding-agent-1/benchmark_report.json`
- Per-fault CSV:
  `runs/orchestration/20260508T214226Z-make-iterative-improvements-to-the-train-test-pi-coding-agent-1/benchmark_report.csv`
- Notion-ready log:
  `runs/orchestration/20260508T214226Z-make-iterative-improvements-to-the-train-test-pi-coding-agent-1/notion_result_summary.md`

## Result

The bounded smoke ran the first 2 faults from the deterministic ITC99 10% gate
using `checkpoints/iscas85_89_20260506_115018/best_model.pth`. It detected 2/2
faults with no fallback. This validates the benchmark execution path and
artifact writing only. The report marks the baseline comparison as not
decision-comparable because the smoke scope is smaller than the documented
`unlinked_candidate 1% ITC99 gate` baseline in
`docs/checkpoint_compatibility_summary.md`.

## Next Validation Step

Run the configured 10% ITC99 gate without `--limit-faults` after review. Treat
that full gate artifact as the pass/fail source for promotion toward full ITC99.
