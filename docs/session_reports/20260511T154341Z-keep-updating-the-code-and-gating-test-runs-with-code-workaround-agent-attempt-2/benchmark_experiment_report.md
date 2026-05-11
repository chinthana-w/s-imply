# Benchmark Experiment Report

Run:
`20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-2`

## Goal

Recovery attempt 2 for the blocked ITC99 gate workflow. The target remains
80% no-fallback coverage with fewer AI backtracks than classic PODEM before
promoting to the configured 6,445-fault ITC99 10% gate or full ITC99 run.

## Sibling Blocker

The previous recovery attempt found that the raw 5-fault gate prefix contained
only primary-input faults, added `--exclude-primary-input-faults`, and then
blocked on the first 5 non-PI faults:

- Coverage: 4/5, meeting the bounded 80% coverage target.
- AI backtracks on successful faults: 68.
- Classic backtracks on the same successful faults: 68.
- Backtrack target: failed.
- AI propagation diagnostic: 0/5 coverage due to strict no-fallback
  `AIBacktracer` failures.

## Workaround Implemented

Patched `src/atpg/ai_podem.py` so `AIBacktracer` no longer treats a propagation
objective with no reconvergent pair as a strict no-fallback failure. For those
objectives, AI has no applicable reconvergent structure, so the backtracer now
uses ordinary local backtrace and still keeps strict failure behavior when an
actual reconvergent AI solve fails.

This is a narrow propagation workaround. It does not promote the model or alter
the deterministic ITC99 gate definition.

## Validation

Static checks:

- `conda run -n deepgate python -m py_compile src/atpg/ai_podem.py scripts/benchmark_itc99_gate.py`
- `conda run -n deepgate ruff check src/atpg/ai_podem.py scripts/benchmark_itc99_gate.py`

Bounded AI propagation retry:

- Command provenance:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-2/benchmark_run_manifest.json`
- JSON:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-2/benchmark_report.json`
- CSV:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-2/benchmark_report.csv`

Bounded activation-only retry with 4 deterministic attempts:

- Command provenance:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-2/benchmark_run_manifest_activation_attempts.json`
- JSON:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-2/benchmark_report_activation_attempts.json`
- CSV:
  `runs/orchestration/20260511T154341Z-keep-updating-the-code-and-gating-test-runs-with-code-workaround-agent-attempt-2/benchmark_report_activation_attempts.csv`

## Results

AI propagation retry:

- Coverage: 1/5, below the 80% bounded target.
- Baseline comparison: improved over the prior AI propagation diagnostic
  0/5, but still not promotable.
- Remaining failure mode: reconvergent propagation objectives such as gates
  5282 and 21155 still fail under strict no-fallback.

Activation-only retry:

- Coverage: 4/5, same as attempt 1.
- AI backtracks on successful faults: 68.
- Classic backtracks on the same successful faults: 68.
- AI less than classic count: 0.
- Backtrack target: failed.

## Decision

Do not promote to the full 6,445-fault ITC99 gate or full ITC99 set.

The smallest viable workaround is documented and validated: allow local
non-reconvergent propagation objectives to backtrace normally. It is not enough
to meet the gate. The next code pass needs to fix model-guided reconvergent
propagation or change the AI objective ranking so successful faults use fewer
backtracks than classic PODEM.

## Next Step

Focus on the `AIBacktracer` path for strict reconvergent propagation failures.
The next useful diagnostic is a single-fault verbose run on gate 1475 with
fault values D and DB, because those are the bounded faults where classic uses
34 backtracks and the current AI activation-only path ties classic exactly.
