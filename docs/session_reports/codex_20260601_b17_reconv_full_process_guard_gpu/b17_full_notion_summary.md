## Experiment Log - 2026-06-06 ITC99 Gate Benchmark

- Command: `/home/local1/miniconda3/envs/deepgate/bin/python -m scripts.benchmark_itc99_gate --model checkpoints/reconv_solver_fix_20260511/best_model.pth --device cuda --fault-list data/bench/ITC99/b17_gate_10pct_faults.json --full --reconv-only --out docs/session_reports/codex_20260601_b17_reconv_full_process_guard_gpu/b17_full_report.json --csv-out docs/session_reports/codex_20260601_b17_reconv_full_process_guard_gpu/b17_full_per_fault.csv --manifest-out docs/session_reports/codex_20260601_b17_reconv_full_process_guard_gpu/b17_full_manifest.json --notion-summary-out docs/session_reports/codex_20260601_b17_reconv_full_process_guard_gpu/b17_full_notion_summary.md --candidate-count 8 --ai-attempts 1 --ai-timeout 20 --compare-classic --classic-timeout 20 --no-backtrack-limit --strict-ai-no-fallback --coverage-target 0.8 --torch-num-threads 1 --memory-guard-mode process --max-rss-gb 8 --flush-every 2000 --cooldown-s 0.02 --progress-every 25 --checkpoint-every 100 --run-id codex_20260601_b17_reconv_full_process_guard_gpu`
- Inputs: model `checkpoints/reconv_solver_fix_20260511/best_model.pth`, fault list `data/bench/ITC99/b17_gate_10pct_faults.json`
- Artifacts: `docs/session_reports/codex_20260601_b17_reconv_full_process_guard_gpu/b17_full_report.json`
- Per-fault CSV: `docs/session_reports/codex_20260601_b17_reconv_full_process_guard_gpu/b17_full_per_fault.csv`
- Manifest: `docs/session_reports/codex_20260601_b17_reconv_full_process_guard_gpu/b17_full_manifest.json`
- Metrics: 28572/45796 faults detected (62.3897% no-fallback coverage, scope `full_filtered_fault_set`)
- Attempted coverage: `28572/45796` = 62.3897%; full configured-scope progress: `28572/45796` = 62.3897%
- Classic search effort: `3893094` total backtracks, `0` on AI-solved faults; AI/model backtrack comparison=N/A
- Activation precheck: 28572 zero-backtrack detections
- Baseline: unlinked_candidate 1% ITC99 gate at 18.1700% from `docs/checkpoint_compatibility_summary.md`
- Baseline comparison: +44.2197% absolute coverage
- Coverage target: 80.0000% of `34223` denominator faults (target is measured against faults covered by classic PODEM); observed `28572/34223` = 83.4877%; required `27379`; pass=True
- Backtrack target enabled: False; pass=N/A
- Result: measurement artifact created; no promotion decision without reviewing the full gate target.
- Next step: validate the candidate checkpoint on the configured 10% ITC99 gate once this slice passes code review.
