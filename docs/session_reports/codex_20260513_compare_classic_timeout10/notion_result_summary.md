## Experiment Log - 2026-05-14 ITC99 Gate Benchmark

- Command: `/home/local1/miniconda3/envs/deepgate/bin/python -m scripts.benchmark_itc99_gate --model checkpoints/reconv_solver_fix_20260511/best_model.pth --device cuda --fault-list data/bench/ITC99/b17_gate_10pct_faults.json --out docs/session_reports/codex_20260513_compare_classic_timeout10/itc99_gate_report.json --csv-out docs/session_reports/codex_20260513_compare_classic_timeout10/itc99_gate_per_fault.csv --manifest-out docs/session_reports/codex_20260513_compare_classic_timeout10/itc99_gate_run_manifest.json --notion-summary-out docs/session_reports/codex_20260513_compare_classic_timeout10/notion_result_summary.md --candidate-count 8 --ai-attempts 1 --max-backtracks 5000 --ai-timeout 10 --compare-classic --classic-timeout 10 --coverage-target 0.8 --backtrack-target --run-id codex_20260513_compare_classic_timeout10`
- Inputs: model `checkpoints/reconv_solver_fix_20260511/best_model.pth`, fault list `data/bench/ITC99/b17_gate_10pct_faults.json`
- Artifacts: `docs/session_reports/codex_20260513_compare_classic_timeout10/itc99_gate_report.json`
- Per-fault CSV: `docs/session_reports/codex_20260513_compare_classic_timeout10/itc99_gate_per_fault.csv`
- Manifest: `docs/session_reports/codex_20260513_compare_classic_timeout10/itc99_gate_run_manifest.json`
- Metrics: 5171/6445 faults detected (80.2327% no-fallback coverage)
- Classic search effort: `205003` total backtracks, `17687` on AI-solved faults; AI/model backtrack comparison=N/A
- Activation precheck: 116 zero-backtrack detections
- Baseline: unlinked_candidate 1% ITC99 gate at 18.1700% from `docs/checkpoint_compatibility_summary.md`
- Baseline comparison: +62.0627% absolute coverage
- Coverage target: 80.0000%; pass=True
- Backtrack target enabled: True; pass=N/A because AI has no comparable backtrack metric
- Result: measurement artifact created; no promotion decision without reviewing the full gate target.
- Next step: validate the candidate checkpoint on the configured 10% ITC99 gate once this slice passes code review.
