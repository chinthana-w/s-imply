#!/usr/bin/env bash
set -euo pipefail

cd /home/local1/chinthana/s-imply
OUT_DIR=docs/session_reports/codex_20260514_b17_full_timeout20_nobtl

if [ -s "${OUT_DIR}/b17_full_per_fault.csv" ]; then
  /home/local1/miniconda3/envs/deepgate/bin/python -m scripts.summarize_b17_coverage \
    --csv "${OUT_DIR}/b17_full_per_fault.csv" \
    --out-md "${OUT_DIR}/b17_partial_summary.md" \
    --out-json "${OUT_DIR}/b17_partial_summary.json" \
    --expected-total 64458 | head -n 45
else
  echo "waiting for first 100-fault checkpoint"
  tail -n 20 "${OUT_DIR}/run.log" 2>/dev/null || true
fi
