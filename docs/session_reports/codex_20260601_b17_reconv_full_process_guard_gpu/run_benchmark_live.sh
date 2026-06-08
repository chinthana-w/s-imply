#!/usr/bin/env bash
set -euo pipefail

cd /home/local1/chinthana/s-imply
OUT_DIR=docs/session_reports/codex_20260601_b17_reconv_full_process_guard_gpu
mkdir -p "${OUT_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

{
  echo "started $(date -Is)"
  echo "${BASHPID}" > "${OUT_DIR}/run.pid"
  exec nice -n 10 /home/local1/miniconda3/envs/deepgate/bin/python -u -m scripts.benchmark_itc99_gate \
    --model checkpoints/reconv_solver_fix_20260511/best_model.pth \
    --device cuda \
    --fault-list data/bench/ITC99/b17_gate_10pct_faults.json \
    --full \
    --reconv-only \
    --reconv-fault-list-out "${OUT_DIR}/b17_reconv_faults.json" \
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
    --torch-num-threads 1 \
    --memory-guard-mode process \
    --max-rss-gb 8 \
    --flush-every 2000 \
    --cooldown-s 0.02 \
    --progress-every 25 \
    --checkpoint-every 100 \
    --run-id codex_20260601_b17_reconv_full_process_guard_gpu
} 2>&1 | tee "${OUT_DIR}/run.log"
