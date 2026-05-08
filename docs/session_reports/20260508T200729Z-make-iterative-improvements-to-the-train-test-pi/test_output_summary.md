# Test Output Summary

Run ID: 20260508T200729Z-make-iterative-improvements-to-the-train-test-pi
Owner: atpg_solver

Status: complete

Validation:
- Focused tests for touched ATPG behavior.
- Scoped Ruff checks for touched files.
- Small AI-vs-vanilla ATPG benchmark subset if solver behavior changes.

Commands and results:
- `conda run -n deepgate python -m pytest tests/test_ai_podem.py -q`
  - Failed before collection: `No module named pytest` in the `deepgate` environment.
- `conda run -n deepgate python -m unittest tests.test_ai_podem -v`
  - Passed: 6 tests.
- `conda run -n deepgate python -m ruff check src/atpg/ai_podem.py tests/test_ai_podem.py`
  - Passed: all checks.
- c17 smoke subset, first 4 faults, via `src.atpg.benchmark_ai_podem.run_benchmark`
  - Vanilla: 4/4 detected, 100.0% coverage.
  - AI activation: 4/4 detected, 100.0% coverage.
  - AI all: 4/4 detected, 100.0% coverage.
- c17 full fault list, via `src.atpg.benchmark_ai_podem.run_benchmark`
  - Vanilla: 22/22 detected, 100.0% coverage, 0 backtracks.
  - AI activation: 22/22 detected, 100.0% coverage, 0 backtracks.
  - AI all: 22/22 detected, 100.0% coverage, 0 backtracks.

Notes:
- The configured checkpoint `checkpoints/reconv_minimal_model.pt` was not present, so AI modes
  used random weights and the result should be treated only as a small execution-path smoke.
- The c17 full run emitted a PyTorch nested tensor prototype warning; it did not fail the run.
