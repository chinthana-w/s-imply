## Experiment Log - 2026-05-31 ITC99 Gate Benchmark

- Command: `/home/local1/miniconda3/envs/deepgate/bin/python -m scripts.benchmark_itc99_gate --model checkpoints/reconv_solver_fix_20260511/best_model.pth --device cuda --fault-list data/bench/ITC99/b17_gate_10pct_faults.json --full --reconv-only --out docs/session_reports/codex_20260529_b17_reconv_full_coveragefix_gpu/b17_full_report.json --csv-out docs/session_reports/codex_20260529_b17_reconv_full_coveragefix_gpu/b17_full_per_fault.csv --manifest-out docs/session_reports/codex_20260529_b17_reconv_full_coveragefix_gpu/b17_full_manifest.json --notion-summary-out docs/session_reports/codex_20260529_b17_reconv_full_coveragefix_gpu/b17_full_notion_summary.md --candidate-count 8 --ai-attempts 1 --ai-timeout 20 --compare-classic --classic-timeout 20 --no-backtrack-limit --strict-ai-no-fallback --coverage-target 0.8 --torch-num-threads 1 --min-available-memory-gb 16 --max-system-memory-percent 80 --max-rss-gb 24 --cooldown-s 0.02 --progress-every 25 --checkpoint-every 100 --run-id codex_20260529_b17_reconv_full_coveragefix_gpu`
- Inputs: model `checkpoints/reconv_solver_fix_20260511/best_model.pth`, fault list `data/bench/ITC99/b17_gate_10pct_faults.json`
- Artifacts: `docs/session_reports/codex_20260529_b17_reconv_full_coveragefix_gpu/b17_full_report.json`
- Per-fault CSV: `docs/session_reports/codex_20260529_b17_reconv_full_coveragefix_gpu/b17_full_per_fault.csv`
- Manifest: `docs/session_reports/codex_20260529_b17_reconv_full_coveragefix_gpu/b17_full_manifest.json`
- Metrics: 12154/45796 faults detected (59.6720% no-fallback coverage, scope `attempted_faults`)
- Attempted coverage: `12154/20368` = 59.6720%; full configured-scope progress: `12154/45796` = 26.5394%
- Classic search effort: `1847881` total backtracks, `0` on AI-solved faults; AI/model backtrack comparison=N/A
- Activation precheck: 12154 zero-backtrack detections
- Baseline: unlinked_candidate 1% ITC99 gate at 18.1700% from `docs/checkpoint_compatibility_summary.md`
- Baseline comparison: not decision-comparable: run did not complete; compare final full-scope coverage only after all configured faults are attempted
- Coverage target: 80.0000% of `14887` denominator faults (target is measured against faults covered by classic PODEM); observed `12154/14887` = 81.6417%; required `11910`; pass=False
- Backtrack target enabled: False; pass=N/A
- Result: measurement artifact created; no promotion decision without reviewing the full gate target.
- Next step: validate the candidate checkpoint on the configured 10% ITC99 gate once this slice passes code review.
