#!/usr/bin/env bash
set -euo pipefail

cd /home/local1/chinthana/s-imply
OUT_DIR=docs/session_reports/codex_20260514_b17_full_timeout20_nobtl

{
  echo "started $(date -Is)"
  echo "${BASHPID}" > "${OUT_DIR}/run.pid"
  exec /home/local1/miniconda3/envs/deepgate/bin/python -u -m scripts.benchmark_itc99_gate \
    --model checkpoints/reconv_solver_fix_20260511/best_model.pth \
    --device cuda \
    --fault-list data/bench/ITC99/b17_gate_10pct_faults.json \
    --full \
    --out "${OUT_DIR}/b17_full_report.json" \
    --csv-out "${OUT_DIR}/b17_full_per_fault.csv" \
    --manifest-out "${OUT_DIR}/b17_full_manifest.json" \
    --notion-summary-out "${OUT_DIR}/b17_full_notion_summary.md" \
    --candidate-count 8 \
    --ai-attempts 1 \
    --ai-timeout 20 \
    --compare-classic \
    --classic-timeout 20 \
    --no-backtrack-limit \
    --strict-ai-no-fallback \
    --coverage-target 0.8 \
    --checkpoint-every 100 \
    --run-id codex_20260514_b17_full_timeout20_nobtl
} 2>&1 | tee "${OUT_DIR}/run.log"
