#!/usr/bin/env bash
set -Eeuo pipefail

# Unified ISCAS85/89 train + ITC99-gate test session.
#
# Quick default:
#   bash scripts/train_test_session.sh
#
# Full run defaults:
#   FULL=1 bash scripts/train_test_session.sh
#
# Useful overrides:
#   RUN_ID=my_run EPOCHS=10 MAX_FAULTS=250 BATCH_SIZE=512 bash scripts/train_test_session.sh
#   STAGES="select_gate train test" CHECKPOINT_DIR=checkpoints/existing bash scripts/train_test_session.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  printf '\n[%s] %s\n' "$(timestamp)" "$*"
}

run_step() {
  log "Running: $*"
  "$@"
}

contains_stage() {
  local wanted="$1"
  [[ " ${STAGES} " == *" ${wanted} "* || " ${STAGES} " == *" all "* ]]
}

FULL="${FULL:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
STAGES="${STAGES:-all}"

if [[ "$FULL" == "1" ]]; then
  MAX_FAULTS="${MAX_FAULTS:-0}"
  EPOCHS="${EPOCHS:-50}"
  MAX_SAMPLES_PER_CIRCUIT="${MAX_SAMPLES_PER_CIRCUIT:-0}"
else
  MAX_FAULTS="${MAX_FAULTS:-100}"
  EPOCHS="${EPOCHS:-5}"
  MAX_SAMPLES_PER_CIRCUIT="${MAX_SAMPLES_PER_CIRCUIT:-50000}"
fi

CACHE_ROOT="${CACHE_ROOT:-/home/local1/cache-cw/simply_sessions/${RUN_ID}}"
CHUNK_DIR="${CHUNK_DIR:-${CACHE_ROOT}/iscas85_89_fault_chunks}"
SHARD_DIR="${SHARD_DIR:-${CACHE_ROOT}/iscas85_89_shards}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/iscas85_89_${RUN_ID}}"
REPORT_DIR="${REPORT_DIR:-docs/session_reports/${RUN_ID}}"

ISCAS85_DIR="${ISCAS85_DIR:-data/bench/ISCAS85}"
ISCAS89_DIR="${ISCAS89_DIR:-data/bench/iscas89}"
ITC99_BENCH="${ITC99_BENCH:-data/bench/ITC99/b17.bench}"
ITC99_FAULT_LIST="${ITC99_FAULT_LIST:-data/bench/ITC99/b17_gate_10pct_faults.json}"

DATA_SEED="${DATA_SEED:-42}"
ITC99_SEED="${ITC99_SEED:-20260504}"
PATTERNS_PER_FAULT="${PATTERNS_PER_FAULT:-1}"
SIM_ATTEMPTS="${SIM_ATTEMPTS:-50}"
UNSAT_RATIO="${UNSAT_RATIO:-0.10}"

MAX_LEN="${MAX_LEN:-50}"
SHARD_SIZE="${SHARD_SIZE:-5000}"
DTYPE="${DTYPE:-float16}"

BATCH_SIZE="${BATCH_SIZE:-1024}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_PATHS="${MAX_PATHS:-256}"
SHARD_CACHE_SIZE="${SHARD_CACHE_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MODEL_DIM="${MODEL_DIM:-512}"
FFN_DIM="${FFN_DIM:-2048}"
ENC_LAYERS="${ENC_LAYERS:-3}"
INT_LAYERS="${INT_LAYERS:-3}"
NHEAD="${NHEAD:-4}"

LAMBDA_SUPERVISED_NODE="${LAMBDA_SUPERVISED_NODE:-2.0}"
LAMBDA_SOLVABILITY="${LAMBDA_SOLVABILITY:-0.5}"
LAMBDA_SHARED_NODE="${LAMBDA_SHARED_NODE:-1.0}"
LAMBDA_LOGIC="${LAMBDA_LOGIC:-1.0}"
LAMBDA_FULL_LOGIC="${LAMBDA_FULL_LOGIC:-0.5}"

CANDIDATE_COUNT="${CANDIDATE_COUNT:-8}"
MAX_BACKTRACKS="${MAX_BACKTRACKS:-5000}"
AMP_FLAG="${AMP_FLAG:---amp}"

mkdir -p "$CACHE_ROOT" "$REPORT_DIR"

log "Session ID: ${RUN_ID}"
log "Stages: ${STAGES}"
log "Cache root: ${CACHE_ROOT}"
log "Checkpoint dir: ${CHECKPOINT_DIR}"

if contains_stage build_data; then
  build_args=(
    scripts/build_fault_dataset.py
    --bench_dirs "$ISCAS85_DIR" "$ISCAS89_DIR"
    --output "$CHUNK_DIR"
    --max_faults "$MAX_FAULTS"
    --patterns-per-fault "$PATTERNS_PER_FAULT"
    --sim_attempts "$SIM_ATTEMPTS"
    --unsat-ratio "$UNSAT_RATIO"
    --seed "$DATA_SEED"
  )
  if [[ "$MAX_SAMPLES_PER_CIRCUIT" != "0" ]]; then
    build_args+=(--max-samples-per-circuit "$MAX_SAMPLES_PER_CIRCUIT")
  fi
  run_step python "${build_args[@]}"
else
  log "Skipping build_data"
fi

if contains_stage preprocess; then
  run_step python -m src.ml.core.dataset preprocess \
    --input "$CHUNK_DIR" \
    --out "$SHARD_DIR" \
    --max_len "$MAX_LEN" \
    --shard_size "$SHARD_SIZE" \
    --dtype "$DTYPE" \
    --resume
else
  log "Skipping preprocess"
fi

if contains_stage select_gate; then
  run_step python scripts/select_itc99_gate_faults.py \
    --bench "$ITC99_BENCH" \
    --output "$ITC99_FAULT_LIST" \
    --fraction 0.10 \
    --seed "$ITC99_SEED"
else
  log "Skipping select_gate"
fi

if contains_stage train; then
  run_step python -m src.ml.train train \
    --dataset "$CHUNK_DIR" \
    --processed-dir "$SHARD_DIR" \
    --output "$CHECKPOINT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --max-paths "$MAX_PATHS" \
    --shard-cache-size "$SHARD_CACHE_SIZE" \
    --checkpointing \
    --num-workers "$NUM_WORKERS" \
    $AMP_FLAG \
    --verbose \
    --nhead "$NHEAD" \
    --model-dim "$MODEL_DIM" \
    --ffn-dim "$FFN_DIM" \
    --enc-layers "$ENC_LAYERS" \
    --int-layers "$INT_LAYERS" \
    --lambda-supervised-node "$LAMBDA_SUPERVISED_NODE" \
    --lambda-solvability "$LAMBDA_SOLVABILITY" \
    --lambda-shared-node "$LAMBDA_SHARED_NODE" \
    --lambda-logic "$LAMBDA_LOGIC" \
    --lambda-full-logic "$LAMBDA_FULL_LOGIC"
else
  log "Skipping train"
fi

if contains_stage test; then
  MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_DIR}/best_model.pth}"
  run_step python scripts/benchmark_itc99_gate.py \
    --model "$MODEL_PATH" \
    --fault-list "$ITC99_FAULT_LIST" \
    --out "${REPORT_DIR}/itc99_gate_report.json" \
    --candidate-count "$CANDIDATE_COUNT" \
    --candidate-seed-base "$ITC99_SEED" \
    --max-backtracks "$MAX_BACKTRACKS"
else
  log "Skipping test"
fi

if contains_stage iscas85_test; then
  MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_DIR}/best_model.pth}"
  run_step python scripts/benchmark_iscas85_nofallback.py \
    --model "$MODEL_PATH" \
    --bench_dir "$ISCAS85_DIR" \
    --out_dir "${CACHE_ROOT}/iscas85_bench" \
    --report "${REPORT_DIR}/iscas85_nofallback_report.md" \
    --max_backtracks "$MAX_BACKTRACKS"
else
  log "Skipping iscas85_test"
fi

log "Done."
log "Best model: ${CHECKPOINT_DIR}/best_model.pth"
log "Reports: ${REPORT_DIR}"
