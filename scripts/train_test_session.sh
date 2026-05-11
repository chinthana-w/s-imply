#!/usr/bin/env bash
set -Eeuo pipefail

# Bash reads long scripts incrementally. Re-exec a stable snapshot so edits to
# this file while a run is active cannot corrupt later stages.
if [[ -z "${TRAIN_TEST_SESSION_SNAPSHOT:-}" ]]; then
  export TRAIN_TEST_SESSION_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
  )"
  snapshot_path="$(mktemp "${TMPDIR:-/tmp}/train_test_session.XXXXXX.sh")"
  cp "${BASH_SOURCE[0]}" "$snapshot_path"
  chmod +x "$snapshot_path"
  export TRAIN_TEST_SESSION_SNAPSHOT=1
  exec bash "$snapshot_path" "$@"
fi

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
#
# Resource guards:
#   CLEAN_PREVIOUS_RUNS=1 KEEP_LATEST_RUNS=1 bash scripts/train_test_session.sh
#   MIN_CACHE_FREE_GB=50 MIN_REPO_FREE_GB=10 MIN_RAM_AVAILABLE_GB=16 bash scripts/train_test_session.sh

ROOT_DIR="${TRAIN_TEST_SESSION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  printf '\n[%s] %s\n' "$(timestamp)" "$*"
}

run_step() {
  check_resources "before $*"
  log "Running: $*"
  "$@"
  check_resources "after $*"
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
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1}"
SINGLE_GPU="${SINGLE_GPU:-0}"

LAMBDA_SUPERVISED_NODE="${LAMBDA_SUPERVISED_NODE:-2.0}"
LAMBDA_SOLVABILITY="${LAMBDA_SOLVABILITY:-0.5}"
LAMBDA_SHARED_NODE="${LAMBDA_SHARED_NODE:-1.0}"
LAMBDA_LOGIC="${LAMBDA_LOGIC:-1.0}"
LAMBDA_FULL_LOGIC="${LAMBDA_FULL_LOGIC:-0.5}"

CANDIDATE_COUNT="${CANDIDATE_COUNT:-8}"
MAX_BACKTRACKS="${MAX_BACKTRACKS:-5000}"
AMP_FLAG="${AMP_FLAG:---amp}"
ITC99_BENCHMARK_LIMIT_FAULTS="${ITC99_BENCHMARK_LIMIT_FAULTS:-0}"
BASELINE_COVERAGE="${BASELINE_COVERAGE:-0.1817}"
BASELINE_LABEL="${BASELINE_LABEL:-unlinked_candidate 1% ITC99 gate}"
BASELINE_SOURCE="${BASELINE_SOURCE:-docs/checkpoint_compatibility_summary.md}"
COVERAGE_TARGET="${COVERAGE_TARGET:-1.0}"

MIN_CACHE_FREE_GB="${MIN_CACHE_FREE_GB:-20}"
MIN_REPO_FREE_GB="${MIN_REPO_FREE_GB:-5}"
MIN_RAM_AVAILABLE_GB="${MIN_RAM_AVAILABLE_GB:-8}"
CLEAN_PREVIOUS_RUNS="${CLEAN_PREVIOUS_RUNS:-0}"
KEEP_LATEST_RUNS="${KEEP_LATEST_RUNS:-1}"

existing_path_for_df() {
  local path="$1"
  while [[ ! -e "$path" && "$path" != "/" ]]; do
    path="$(dirname "$path")"
  done
  printf '%s\n' "$path"
}

available_disk_gb() {
  local path
  path="$(existing_path_for_df "$1")"
  df -Pk "$path" | awk 'NR == 2 { printf "%.0f", $4 / 1024 / 1024 }'
}

available_ram_gb() {
  awk '/MemAvailable:/ { printf "%.0f", $2 / 1024 / 1024 }' /proc/meminfo
}

assert_disk_available() {
  local label="$1"
  local path="$2"
  local min_gb="$3"
  local available_gb
  available_gb="$(available_disk_gb "$path")"
  if (( available_gb < min_gb )); then
    log "ERROR: ${label} has ${available_gb} GiB free; need at least ${min_gb} GiB."
    log "Set CLEAN_PREVIOUS_RUNS=1 or move CACHE_ROOT/CHECKPOINT_DIR to a larger filesystem."
    exit 1
  fi
}

assert_ram_available() {
  local min_gb="$1"
  local available_gb
  available_gb="$(available_ram_gb)"
  if (( available_gb < min_gb )); then
    log "ERROR: RAM has ${available_gb} GiB available; need at least ${min_gb} GiB."
    log "Reduce BATCH_SIZE, NUM_WORKERS, SHARD_CACHE_SIZE, MODEL_DIM, FFN_DIM, or MAX_PATHS."
    exit 1
  fi
}

resource_snapshot() {
  log "Resource snapshot: cache_free=$(available_disk_gb "$CACHE_ROOT")GiB repo_free=$(available_disk_gb "$ROOT_DIR")GiB ram_available=$(available_ram_gb)GiB"
}

check_resources() {
  local context="$1"
  log "Checking resources (${context})"
  assert_disk_available "CACHE_ROOT filesystem" "$CACHE_ROOT" "$MIN_CACHE_FREE_GB"
  assert_disk_available "CHECKPOINT_DIR filesystem" "$CHECKPOINT_DIR" "$MIN_REPO_FREE_GB"
  assert_disk_available "REPORT_DIR filesystem" "$REPORT_DIR" "$MIN_REPO_FREE_GB"
  assert_ram_available "$MIN_RAM_AVAILABLE_GB"
  resource_snapshot
}

cleanup_glob_except_current() {
  local current_path="$1"
  local keep_latest="$2"
  shift 2
  local candidates=()
  local path
  for path in "$@"; do
    [[ -e "$path" ]] || continue
    [[ "$(realpath -m "$path")" == "$(realpath -m "$current_path")" ]] && continue
    candidates+=("$path")
  done
  if (( ${#candidates[@]} == 0 )); then
    return
  fi
  mapfile -t candidates < <(ls -dt "${candidates[@]}")
  if (( keep_latest > 0 && ${#candidates[@]} > keep_latest )); then
    candidates=("${candidates[@]:$keep_latest}")
  elif (( keep_latest > 0 )); then
    return
  fi
  for path in "${candidates[@]}"; do
    log "Removing previous train/test session artifact: ${path}"
    rm -rf -- "$path"
  done
}

cleanup_previous_runs() {
  log "Cleaning previous train_test_session.sh outputs"
  cleanup_glob_except_current "$CACHE_ROOT" "$KEEP_LATEST_RUNS" \
    "$(dirname "$CACHE_ROOT")"/*
  cleanup_glob_except_current "$CHECKPOINT_DIR" "$KEEP_LATEST_RUNS" \
    checkpoints/iscas85_89_*
  cleanup_glob_except_current "$REPORT_DIR" "$KEEP_LATEST_RUNS" \
    docs/session_reports/*
}

if [[ "$CLEAN_PREVIOUS_RUNS" == "1" ]]; then
  cleanup_previous_runs
fi

mkdir -p "$CACHE_ROOT" "$REPORT_DIR"

log "Session ID: ${RUN_ID}"
log "Stages: ${STAGES}"
log "Cache root: ${CACHE_ROOT}"
log "Checkpoint dir: ${CHECKPOINT_DIR}"
log "Resource minimums: cache=${MIN_CACHE_FREE_GB}GiB repo=${MIN_REPO_FREE_GB}GiB ram=${MIN_RAM_AVAILABLE_GB}GiB"
check_resources "startup"

if contains_stage build_data; then
  build_args=(
    -m scripts.build_fault_dataset
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
  run_step python -m scripts.select_itc99_gate_faults \
    --bench "$ITC99_BENCH" \
    --output "$ITC99_FAULT_LIST" \
    --fraction 0.10 \
    --seed "$ITC99_SEED"
else
  log "Skipping select_gate"
fi

if contains_stage train; then
  train_args=(
    python -m src.ml.train train
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
    --lambda-full-logic "$LAMBDA_FULL_LOGIC" \
    --gpu-ids "$TRAIN_GPU_IDS"
  )
  if [[ "$SINGLE_GPU" == "1" ]]; then
    train_args+=(--single-gpu)
  fi
  run_step "${train_args[@]}"
else
  log "Skipping train"
fi

if contains_stage test; then
  MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_DIR}/best_model.pth}"
  benchmark_args=(
    -m scripts.benchmark_itc99_gate
    --model "$MODEL_PATH"
    --fault-list "$ITC99_FAULT_LIST"
    --out "${REPORT_DIR}/itc99_gate_report.json"
    --csv-out "${REPORT_DIR}/itc99_gate_per_fault.csv"
    --manifest-out "${REPORT_DIR}/itc99_gate_run_manifest.json"
    --notion-summary-out "${REPORT_DIR}/notion_result_summary.md"
    --candidate-count "$CANDIDATE_COUNT"
    --candidate-seed-base "$ITC99_SEED"
    --max-backtracks "$MAX_BACKTRACKS"
    --baseline-coverage "$BASELINE_COVERAGE"
    --baseline-label "$BASELINE_LABEL"
    --baseline-source "$BASELINE_SOURCE"
    --coverage-target "$COVERAGE_TARGET"
    --run-id "$RUN_ID"
  )
  if [[ "$ITC99_BENCHMARK_LIMIT_FAULTS" != "0" ]]; then
    benchmark_args+=(--limit-faults "$ITC99_BENCHMARK_LIMIT_FAULTS")
  fi
  run_step python "${benchmark_args[@]}"
else
  log "Skipping test"
fi

if contains_stage full_itc99_test; then
  MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_DIR}/best_model.pth}"
  run_step python -m scripts.benchmark_itc99_gate \
    --model "$MODEL_PATH" \
    --fault-list "$ITC99_FAULT_LIST" \
    --out "${REPORT_DIR}/itc99_full_report.json" \
    --csv-out "${REPORT_DIR}/itc99_full_per_fault.csv" \
    --manifest-out "${REPORT_DIR}/itc99_full_run_manifest.json" \
    --notion-summary-out "${REPORT_DIR}/notion_full_result_summary.md" \
    --candidate-count "$CANDIDATE_COUNT" \
    --candidate-seed-base "$ITC99_SEED" \
    --max-backtracks "$MAX_BACKTRACKS" \
    --baseline-coverage "$BASELINE_COVERAGE" \
    --baseline-label "$BASELINE_LABEL" \
    --baseline-source "$BASELINE_SOURCE" \
    --coverage-target "$COVERAGE_TARGET" \
    --run-id "$RUN_ID" \
    --full
else
  log "Skipping full_itc99_test"
fi

if contains_stage iscas85_test; then
  MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_DIR}/best_model.pth}"
  run_step python -m scripts.benchmark_iscas85_nofallback \
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
