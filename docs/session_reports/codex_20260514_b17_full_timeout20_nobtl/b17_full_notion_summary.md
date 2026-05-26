## Experiment Log - 2026-05-23 ITC99 Gate Benchmark

- Command: `/home/local1/miniconda3/envs/deepgate/bin/python -m scripts.benchmark_itc99_gate --model checkpoints/reconv_solver_fix_20260511/best_model.pth --device auto --fault-list data/bench/ITC99/b17_gate_10pct_faults.json --full --out docs/session_reports/codex_20260514_b17_full_timeout20_nobtl/b17_full_report.json --csv-out docs/session_reports/codex_20260514_b17_full_timeout20_nobtl/b17_full_per_fault.csv --manifest-out docs/session_reports/codex_20260514_b17_full_timeout20_nobtl/b17_full_manifest.json --notion-summary-out docs/session_reports/codex_20260514_b17_full_timeout20_nobtl/b17_full_notion_summary.md --candidate-count 8 --ai-attempts 1 --ai-timeout 20 --compare-classic --classic-timeout 20 --no-backtrack-limit --strict-ai-no-fallback --coverage-target 0.8 --checkpoint-every 100 --run-id codex_20260514_b17_full_timeout20_nobtl`
- Inputs: model `checkpoints/reconv_solver_fix_20260511/best_model.pth`, fault list `data/bench/ITC99/b17_gate_10pct_faults.json`
- Artifacts: `docs/session_reports/codex_20260514_b17_full_timeout20_nobtl/b17_full_report.json`
- Per-fault CSV: `docs/session_reports/codex_20260514_b17_full_timeout20_nobtl/b17_full_per_fault.csv`
- Manifest: `docs/session_reports/codex_20260514_b17_full_timeout20_nobtl/b17_full_manifest.json`
- Metrics: 13705/64458 faults detected (21.2619% no-fallback coverage)
- Classic search effort: `1766599` total backtracks, `0` on AI-solved faults; AI/model backtrack comparison=N/A
- Activation precheck: 783 zero-backtrack detections
- Baseline: unlinked_candidate 1% ITC99 gate at 18.1700% from `docs/checkpoint_compatibility_summary.md`
- Baseline comparison: +3.0919% absolute coverage
- Coverage target: 80.0000% of `35583` denominator faults (target is measured against faults covered by classic PODEM); observed `13705/35583` = 38.5156%; required `28467`; pass=False
- Backtrack target enabled: False; pass=N/A
- Result: measurement artifact created; no promotion decision without reviewing the full gate target.
- Next step: validate the candidate checkpoint on the configured 10% ITC99 gate once this slice passes code review.
