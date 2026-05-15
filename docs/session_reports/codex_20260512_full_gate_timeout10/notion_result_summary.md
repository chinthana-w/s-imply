## Experiment Log - 2026-05-13 ITC99 Gate Benchmark

- Command: `/home/local1/miniconda3/envs/deepgate/bin/python -m scripts.benchmark_itc99_gate --model checkpoints/reconv_solver_fix_20260511/best_model.pth --fault-list data/bench/ITC99/b17_gate_10pct_faults.json --out docs/session_reports/codex_20260512_full_gate_timeout10/itc99_gate_report.json --csv-out docs/session_reports/codex_20260512_full_gate_timeout10/itc99_gate_per_fault.csv --manifest-out docs/session_reports/codex_20260512_full_gate_timeout10/itc99_gate_run_manifest.json --notion-summary-out docs/session_reports/codex_20260512_full_gate_timeout10/notion_result_summary.md --candidate-count 8 --ai-attempts 1 --candidate-seed-base 20260504 --max-backtracks 5000 --ai-timeout 10 --classic-timeout 5 --baseline-coverage 0.1817 --baseline-label 'unlinked_candidate 1% ITC99 gate' --baseline-source docs/checkpoint_compatibility_summary.md --coverage-target 0.8 --run-id codex_20260512_full_gate_timeout10`
- Inputs: model `checkpoints/reconv_solver_fix_20260511/best_model.pth`, fault list `data/bench/ITC99/b17_gate_10pct_faults.json`
- Artifacts: `docs/session_reports/codex_20260512_full_gate_timeout10/itc99_gate_report.json`
- Per-fault CSV: `docs/session_reports/codex_20260512_full_gate_timeout10/itc99_gate_per_fault.csv`
- Manifest: `docs/session_reports/codex_20260512_full_gate_timeout10/itc99_gate_run_manifest.json`
- Metrics: 5171/6445 faults detected (80.2327% no-fallback coverage)
- Search backtracks: AI-guided PODEM `202693`; classic not measured (`--compare-classic` was not enabled); AI/classic comparison=N/A
- Activation precheck: 116 zero-backtrack detections
- Baseline: unlinked_candidate 1% ITC99 gate at 18.1700% from `docs/checkpoint_compatibility_summary.md`
- Baseline comparison: +62.0627% absolute coverage
- Coverage target: 80.0000%; pass=True
- Backtrack target enabled: False; pass=N/A
- Result: full 10% ITC99 gate coverage target passed; AI/classic backtrack target was not evaluated in this run.
- Next step: run a separate `--compare-classic --backtrack-target` benchmark if a same-fault backtrack-efficiency claim is needed.
