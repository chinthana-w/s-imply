# Benchmark Experiment Report

Run:
`20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-1`

## Goal

Recover from the blocked code benchmark run
`20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2`
by identifying the blocking failure, coordinating with sibling artifacts, and
leaving the smallest validated workaround for the next gate retry.

## Blocking Failure

The blocked sibling run used `--limit-faults 5` on the raw deterministic ITC99
gate list. The five executed faults were all primary-input faults:

- `3/DB`
- `14/D`
- `25/D`
- `26/D`
- `29/D`

That prefix is a poor code gate for the current AI-assisted objective. AI
activation can only set the fault-site PI, and propagation remains classic when
`--enable-ai-propagation` is disabled. Three of the five faults already tie
classic at zero backtracks, so the smoke cannot demonstrate the requested
AI-backtracks-less-than-classic target.

Sibling evidence:

- Prior report:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-coding-agent-2/benchmark_report.json`
- Prior bounded result: 5/5 detected, AI backtracks 230, classic backtracks
  230, backtrack target failed.

## Workaround Implemented

Added `--exclude-primary-input-faults` to `scripts/benchmark_itc99_gate.py`.
The filter is applied before `--limit-faults`, and the report/manifest record:

- `exclude_primary_input_faults`
- `primary_input_faults_excluded`
- `primary_input_faults_excluded_count`
- `post_filter_faults`

This is only a bounded-smoke selection workaround. It does not replace the
configured 6,445-fault ITC99 10% promotion gate.

## Validation

Commands:

- `conda run -n deepgate python -m py_compile scripts/benchmark_itc99_gate.py`
- `conda run -n deepgate ruff check scripts/benchmark_itc99_gate.py`
- `conda run -n deepgate python -m scripts.benchmark_itc99_gate ... --exclude-primary-input-faults --limit-faults 5 --compare-classic --backtrack-target`
- Diagnostic only: same bounded smoke with `--enable-ai-propagation`

Artifacts:

- `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-1/run_manifest.json`
- `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-1/benchmark_run_manifest.json`
- `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-1/benchmark_report.json`
- `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-1/benchmark_report.csv`
- `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-1/notion_result_summary.md`
- `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-1/benchmark_report_ai_propagation.json`
- `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-1/benchmark_report_ai_propagation.csv`

## Results

Non-PI bounded smoke:

- Filtered out 294 primary-input faults from the 6,445-fault gate list.
- Executed first 5 non-PI faults.
- Coverage: 4/5 detected, 80.0%.
- Coverage target: passed at the configured 80.0% target.
- AI backtracks on successful faults: 68.
- Classic backtracks on the same successful faults: 68.
- Backtrack target: failed.
- Activation precheck: 0 zero-backtrack detections.

AI propagation diagnostic:

- Coverage: 0/5 detected.
- Classic comparison on the same faults: 5/5 detected.
- Failure mode: `AIBacktracer` raises no-fallback failures for the propagation
  objectives.

## Decision

Do not promote this attempt to the full 10% ITC99 gate or full ITC99 test set.
The PI-prefix workaround is valid and should be kept, but the requested
backtrack target is still blocked.

## Next Step

Fix no-fallback AI propagation/backtrace behavior before rerunning a larger
non-PI gate, then run the configured 6,445-fault ITC99 10% gate only after both
coverage and backtrack targets pass on the smaller gate.

## Notion

The Notion target page was found:
`Back Implication Prediction using Attention`.

The update call was cancelled before applying the edit. The Notion-ready dated
log is available at:

`runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-1/notion_result_summary.md`
