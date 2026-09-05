#!/usr/bin/env bash
# Short throughput / VRAM calibration for E17d (seq 4096, attn-residual, no token carry).
#
# Rank by **real tokens/sec** under production packing (`length_group`). Filling the
# 3090s is a constraint, not the objective: E17b showed pad-to-batch-max at bs=8
# filled VRAM and *slowed* training vs bs=3. After length grouping, raise
# per-device microbatch only while tok/s rises and 4096-token buckets stay
# ~1–2 GiB under 24 GiB. Restore effective batch ~72 via accumulation so the
# 300M token budget keeps a comparable optimizer-step count to E17c.
#
# Usage (Polonez, GPUs idle):
#   bash scripts/calibrate_e17d_batch.sh
#   ONLY_BS="3 4 6 8 10 12" PACKING=length_group MAX_STEPS=20 bash scripts/calibrate_e17d_batch.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

ONLY_BS="${ONLY_BS:-3 4 6 8 10 12}"
ACCUM="${ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-20}"
LOGGING_STEPS="${LOGGING_STEPS:-4}"
PACKING="${PACKING:-length_group}"
RESULTS_FILE="${RESULTS_FILE:-Cache/logs/e17d_batch_calibration_$(date +%Y%m%d_%H%M%S).tsv}"
mkdir -p "$(dirname "$RESULTS_FILE")"
echo -e "per_device\taccum\tpacking\teff_batch\tpeak_mem_mib\tmax_batch_len\tpad_ratio\ttok_s\tstatus\twall_s\tlog" | tee "$RESULTS_FILE"

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

_avg_last_metric() {
  local log="$1" key="$2"
  python3 - "$log" "$key" <<'PY'
import re, sys
path, key = sys.argv[1], sys.argv[2]
pat = re.compile(re.escape(key) + r"': ([0-9.eE+-]+)")
vals = []
with open(path, errors="replace") as f:
    for line in f:
        if "{'loss'" not in line and '{"loss"' not in line:
            continue
        m = pat.search(line)
        if m:
            vals.append(float(m.group(1)))
if not vals:
    print("")
    raise SystemExit
# Drop the first logging window (compile / cache warmup).
use = vals[1:] if len(vals) > 1 else vals
print(sum(use) / len(use))
PY
}

for BS in $ONLY_BS; do
  EFF=$((BS * NUM_GPUS * ACCUM))
  echo "=== calibrating PER_DEVICE=${BS} ACCUM=${ACCUM} eff_batch=${EFF} packing=${PACKING} (${NUM_GPUS} GPUs) ==="
  STAMP=$(date +%Y%m%d_%H%M%S)
  LOG="Cache/logs/calib_e17d_bs${BS}_a${ACCUM}_${PACKING}_${STAMP}.log"
  MEM_LOG="Cache/logs/calib_e17d_bs${BS}_a${ACCUM}_${PACKING}_mem_${STAMP}.csv"
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
  TOK_S=$(_avg_last_metric "$LOG" "perf/real_tokens_per_second" || true)
  MAX_LEN=$(_avg_last_metric "$LOG" "data/mean_batch_max_length" || true)
  PAD=$(_avg_last_metric "$LOG" "data/pad_ratio" || true)
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
  echo -e "${BS}\t${ACCUM}\t${PACKING}\t${EFF}\t${PEAK}\t${MAX_LEN}\t${PAD}\t${TOK_S}\t${STATUS}\t${WALL}\t${LOG}" | tee -a "$RESULTS_FILE"
  if [[ "$STATUS" == "OOM" ]]; then
    echo "OOM at bs=${BS}; stopping sweep (larger bs will too)."
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
        if len(parts) < 9:
            continue
        bs, accum, packing, eff, peak, max_len, pad, tok_s, status = parts[:9]
        if status != "OK":
            continue
        try:
            peak_i = int(float(peak))
        except ValueError:
            continue
        if peak_i > 23000:  # need ~1 GiB headroom on 24576
            print(f"skip bs={bs}: peak {peak_i} MiB too close to 24 GiB")
            continue
        try:
            tok = float(tok_s) if tok_s else 0.0
        except ValueError:
            tok = 0.0
        rows.append({
            "bs": int(bs),
            "peak": peak_i,
            "tok": tok,
            "max_len": max_len,
            "pad": pad,
            "packing": packing,
        })
if not rows:
    print("No OK row with VRAM headroom. Inspect the TSV.")
    raise SystemExit(0)

# Primary rank: real tokens/sec. GPU fill is only a feasibility filter.
rows.sort(key=lambda r: r["tok"], reverse=True)
print("Ranked by real tokens/sec (length_group production packing):")
for r in rows:
    print(
        f"  bs={r['bs']:>2}  tok/s={r['tok']:.0f}  peak={r['peak']}MiB  "
        f"mean_batch_max_len={r['max_len']}  pad_ratio={r['pad']}"
    )
best = rows[0]
bs = best["bs"]
target_eff = 72
best_acc, best_err = 1, abs(bs * n_gpus * 1 - target_eff)
for acc in range(1, 9):
    err = abs(bs * n_gpus * acc - target_eff)
    if err < best_err or (err == best_err and acc > best_acc):
        # prefer the accum that hits 72; if tied, larger accum is closer to E17c step count
        best_acc, best_err = acc, err
eff = bs * n_gpus * best_acc
print(
    f"Recommend PER_DEVICE_BATCH_SIZE={bs} GRADIENT_ACCUMULATION_STEPS={best_acc} "
    f"eff_batch={eff} (E17c target 72, |err|={best_err}) because tok/s={best['tok']:.0f} "
    f"is highest with peak={best['peak']}MiB under length_group."
)
print("Keep TARGET_TOKENS=300000000. Do not resume the aborted underfilled run.")
PY
