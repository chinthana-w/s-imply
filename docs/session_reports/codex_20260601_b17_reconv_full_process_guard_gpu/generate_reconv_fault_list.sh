#!/usr/bin/env bash
set -euo pipefail

cd /home/local1/chinthana/s-imply
OUT_DIR=docs/session_reports/codex_20260601_b17_reconv_full_process_guard_gpu
mkdir -p "${OUT_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{
  echo "started $(date -Is)"
  exec nice -n 15 /home/local1/miniconda3/envs/deepgate/bin/python -u -m scripts.benchmark_itc99_gate \
    --model checkpoints/reconv_solver_fix_20260511/best_model.pth \
    --device cpu \
    --fault-list data/bench/ITC99/b17_gate_10pct_faults.json \
    --full \
    --reconv-only \
    --reconv-filter-only \
    --reconv-fault-list-out "${OUT_DIR}/b17_reconv_faults.json" \
    --out "${OUT_DIR}/b17_reconv_filter_report.json" \
    --manifest-out "${OUT_DIR}/b17_reconv_filter_manifest.json" \
    --torch-num-threads 1 \
    --memory-guard-mode process \
    --max-rss-gb 8 \
    --flush-every 2000 \
    --progress-every 1000 \
    --run-id codex_20260601_b17_reconv_fault_list_only
} 2>&1 | tee "${OUT_DIR}/reconv_fault_list.log"
