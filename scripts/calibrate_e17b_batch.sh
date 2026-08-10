#!/usr/bin/env bash
# Short VRAM / throughput calibration for E17b (seq 4096, per_layer_banks).
# Goal: maximize tokens/sec on 4x3090. Effective batch is NOT fixed — raise
# per-device microbatch (and optionally lower accum) until near-OOM, leave
# ~1-2 GiB headroom.
#
# Usage (Polonez, GPUs idle):
#   bash scripts/calibrate_e17b_batch.sh
#   ONLY_BS="4 6 8 10 12" ACCUM=1 MAX_STEPS=16 bash scripts/calibrate_e17b_batch.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

ONLY_BS="${ONLY_BS:-4 6 8 10 12}"
ACCUM="${ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-4}"
RESULTS_FILE="${RESULTS_FILE:-Cache/logs/e17b_batch_calibration_$(date +%Y%m%d_%H%M%S).tsv}"
mkdir -p "$(dirname "$RESULTS_FILE")"
echo -e "per_device\taccum\teff_batch\tpeak_mem_mib\tstatus\twall_s\tlog" | tee "$RESULTS_FILE"

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

for BS in $ONLY_BS; do
  EFF=$((BS * NUM_GPUS * ACCUM))
  echo "=== calibrating PER_DEVICE=${BS} ACCUM=${ACCUM} eff_batch=${EFF} (${NUM_GPUS} GPUs) ==="
  LOG="Cache/logs/calib_e17b_bs${BS}_a${ACCUM}_$(date +%Y%m%d_%H%M%S).log"
  MEM_LOG="Cache/logs/calib_e17b_bs${BS}_a${ACCUM}_mem_$(date +%Y%m%d_%H%M%S).csv"
  (
    echo "timestamp,gpu,mem_used_mib"
    while true; do
      ts=$(date -Is)
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -v ts="$ts" -F', ' '{gsub(/ /,"",$2); print ts","$1","$2}'
      sleep 2
    done
  ) >"$MEM_LOG" &
  MEM_PID=$!
  set +e
  START_TS=$(date +%s)
  (
    # Do NOT set WANDB_DISABLED — transformers rejects it when report_to was wandb.
    unset WANDB_DISABLED || true
    export WANDB_MODE=disabled
    export REPORT_TO=none
    export MAX_STEPS="$MAX_STEPS"
    export LOGGING_STEPS="$LOGGING_STEPS"
    export EVAL_STRATEGY=no
    export SAVE_STRATEGY=no
    export LOAD_BEST_MODEL_AT_END=False
    export AUTO_INTERVALS=0
    export SAVE_TOTAL_LIMIT=1
    export TARGET_TOKENS=1000000000
    export SKIP_PRETOKENIZE=1
    export PER_DEVICE_BATCH_SIZE="$BS"
    export GRADIENT_ACCUMULATION_STEPS="$ACCUM"
    bash scripts/launch_e17b.sh
  ) >"$LOG" 2>&1
  RC=$?
  END_TS=$(date +%s)
  set -e
  kill "$MEM_PID" 2>/dev/null || true
  wait "$MEM_PID" 2>/dev/null || true
  PEAK=$(awk -F, 'NR>1{if($3+0>m)m=$3+0}END{print m+0}' "$MEM_LOG")
  if grep -q 'CUDA out of memory\|OutOfMemoryError' "$LOG"; then
    STATUS=OOM
  elif [[ $RC -ne 0 ]]; then
    STATUS="FAIL_rc${RC}"
  elif grep -q "{'loss'" "$LOG"; then
    STATUS=OK
  else
    STATUS=UNKNOWN
  fi
  WALL=$((END_TS - START_TS))
  echo -e "${BS}\t${ACCUM}\t${EFF}\t${PEAK}\t${STATUS}\t${WALL}\t${LOG}" | tee -a "$RESULTS_FILE"
  # Stop climbing once we OOM — larger BS will too.
  if [[ "$STATUS" == "OOM" ]]; then
    echo "OOM at bs=${BS}; stopping sweep."
    break
  fi
  sleep 3
done

echo "=== calibration done; results in ${RESULTS_FILE} ==="
cat "$RESULTS_FILE"
