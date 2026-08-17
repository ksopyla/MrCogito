#!/usr/bin/env bash
# Short VRAM / throughput calibration for E17d (seq 4096, attn-residual, no token carry).
# Goal: fill 4x RTX 3090s. Default packing=none so peak VRAM is the 4096-token worst
# case (length_group underfills short buckets). Effective batch is restored after the
# sweep by lowering GRADIENT_ACCUMULATION_STEPS so 4 * bs * accum stays near 72, the
# E16b/E17/E17c token-budget invariant. TARGET_TOKENS stays 300M on the real launch.
#
# Usage (Polonez, GPUs idle):
#   bash scripts/calibrate_e17d_batch.sh
#   ONLY_BS="6 8 9 10 12" ACCUM=1 MAX_STEPS=16 bash scripts/calibrate_e17d_batch.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

ONLY_BS="${ONLY_BS:-6 8 9 10 12}"
ACCUM="${ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-4}"
PACKING="${PACKING:-none}"
RESULTS_FILE="${RESULTS_FILE:-Cache/logs/e17d_batch_calibration_$(date +%Y%m%d_%H%M%S).tsv}"
mkdir -p "$(dirname "$RESULTS_FILE")"
echo -e "per_device\taccum\teff_batch\tpeak_mem_mib\tmax_batch_len\ttok_s\tstatus\twall_s\tlog" | tee "$RESULTS_FILE"

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

for BS in $ONLY_BS; do
  EFF=$((BS * NUM_GPUS * ACCUM))
  echo "=== calibrating PER_DEVICE=${BS} ACCUM=${ACCUM} eff_batch=${EFF} packing=${PACKING} (${NUM_GPUS} GPUs) ==="
  STAMP=$(date +%Y%m%d_%H%M%S)
  LOG="Cache/logs/calib_e17d_bs${BS}_a${ACCUM}_${STAMP}.log"
  MEM_LOG="Cache/logs/calib_e17d_bs${BS}_a${ACCUM}_mem_${STAMP}.csv"
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
    export TARGET_TOKENS=300000000
    export SKIP_PRETOKENIZE=1
    export PER_DEVICE_BATCH_SIZE="$BS"
    export GRADIENT_ACCUMULATION_STEPS="$ACCUM"
    export BATCH_PACKING_MODE="$PACKING"
    bash scripts/launch_e17d.sh
  ) >"$LOG" 2>&1
  RC=$?
  END_TS=$(date +%s)
  set -e
  kill "$MEM_PID" 2>/dev/null || true
  wait "$MEM_PID" 2>/dev/null || true
  PEAK=$(awk -F, 'NR>1{if($3+0>m)m=$3+0}END{print m+0}' "$MEM_LOG")
  TOK_S=$(grep -o "perf/real_tokens_per_second': [^,}]*" "$LOG" | tail -1 | awk -F': ' '{print $2}')
  TOK_S="${TOK_S:-}"
  MAX_LEN=$(grep -o "data/mean_batch_max_length': [^,}]*" "$LOG" | tail -1 | awk -F': ' '{print $2}')
  MAX_LEN="${MAX_LEN:-}"
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
  echo -e "${BS}\t${ACCUM}\t${EFF}\t${PEAK}\t${MAX_LEN}\t${TOK_S}\t${STATUS}\t${WALL}\t${LOG}" | tee -a "$RESULTS_FILE"
  if [[ "$STATUS" == "OOM" ]]; then
    echo "OOM at bs=${BS}; stopping sweep."
    break
  fi
  sleep 3
done

echo "=== calibration done; results in ${RESULTS_FILE} ==="
cat "$RESULTS_FILE"
python3 - "$RESULTS_FILE" "$NUM_GPUS" <<'PY'
import sys
path, n_gpus = sys.argv[1], int(sys.argv[2])
rows = []
with open(path) as f:
    header = f.readline()
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 7:
            continue
        bs, accum, eff, peak, max_len, tok_s, status = parts[:7]
        if status != "OK":
            continue
        try:
            peak_i = int(float(peak))
        except ValueError:
            continue
        if peak_i > 22528:  # leave ~2 GiB on 24 GiB
            continue
        try:
            tok = float(tok_s) if tok_s else 0.0
        except ValueError:
            tok = 0.0
        rows.append((int(bs), peak_i, tok, max_len))
if not rows:
    print("No OK row with peak <= 22 GiB. Inspect the TSV.")
    sys.exit(0)
# Prefer highest microbatch that still has headroom; break ties on tok/s.
rows.sort(key=lambda r: (r[0], r[2]))
bs, peak, tok, max_len = rows[-1]
target_eff = 72
best_acc, best_err = 1, abs(bs * n_gpus * 1 - target_eff)
for acc in range(1, 9):
    err = abs(bs * n_gpus * acc - target_eff)
    if err < best_err:
        best_acc, best_err = acc, err
eff = bs * n_gpus * best_acc
print(
    f"Recommend PER_DEVICE_BATCH_SIZE={bs} GRADIENT_ACCUMULATION_STEPS={best_acc} "
    f"eff_batch={eff} (target 72, err {best_err}) peak={peak}MiB tok/s={tok:.0f} "
    f"mean_batch_max_len={max_len}"
)
print("Keep TARGET_TOKENS=300000000. Do not resume the aborted underfilled run.")
PY
