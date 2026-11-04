#!/bin/bash
# E33n speed / memory check: one short training run per ITEM on the visible GPU, results in OUT/summary.tsv.
# Used by stage0.sh on the H100 and directly on Odra / Polonez (free GPUs) before any paid run.
#
# Usage: bash infra/aws/jobs/speed_check.sh DATA OUT ITEM...      ITEM = size:arm:phase:tokens:micro
#   e.g. CUDA_VISIBLE_DEVICES=0 bash infra/aws/jobs/speed_check.sh ~/data /tmp/sc s30:e33n:trunk:4e6:4
# A decay item starts from the trunk item of the same size and arm when one ran earlier in OUT.
# tokens/s = mean of the last 5 logged `perf/real_tokens_per_second`; peak memory = nvidia-smi every 2 s.
# AFTER_EACH (optional): a command run after every item (stage0.sh copies results to S3 with it).
set -uo pipefail
DATA="$1"; OUT="$2"; shift 2
export PATH="$HOME/.local/bin:$PATH" WANDB_MODE=offline TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_DIR="${WANDB_DIR:-$OUT/wandb}"
mkdir -p "$OUT" "$WANDB_DIR"
[ -f "$OUT/summary.tsv" ] || printf 'size\tarm\tphase\tmicro\tstatus\ttok_per_s\tpeak_mem_gb\tseconds\n' > "$OUT/summary.tsv"
GPU="${CUDA_VISIBLE_DEVICES:-0}"; GPU="${GPU%%,*}"

for item in "$@"; do
  IFS=: read -r size arm phase tokens micro <<< "$item"
  log="$OUT/${size}_${arm}_${phase}_m${micro}.log"
  init=""
  if [ "$phase" = decay ]; then
    init=$(ls -d "$OUT/${size}_${arm}"/trunk/*/final 2>/dev/null | tail -1)
  fi
  CMD=$(uv run python scripts/run_e33n.py cmd --arm "$arm" --size "$size" --phase "$phase" --data "$DATA" \
        --out "$OUT/${size}_${arm}" --tokens "$tokens" --micro "$micro" ${init:+--init "$init"})
  echo "== $(date -u +%H:%M:%S) $size $arm $phase micro $micro"
  t0=$(date +%s)
  nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > "$log.gpu" &
  smi=$!
  ( eval "$CMD" ) > "$log" 2>&1; rc=$?
  kill $smi 2>/dev/null
  tps=$(grep -oE "'perf/real_tokens_per_second': [0-9.]+" "$log" | tail -5 | awk '{s+=$2;n++} END {if(n) printf "%.0f", s/n}')
  mem=$(awk -F', ' '{if ($1+0>m) m=$1+0} END {printf "%.1f", m/1024}' "$log.gpu")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$size" "$arm" "$phase" "$micro" "$([ $rc = 0 ] && echo ok || echo FAIL)" \
    "${tps:-?}" "$mem" "$(( $(date +%s) - t0 ))" >> "$OUT/summary.tsv"
  [ $rc = 0 ] || tail -25 "$log"
  [ -n "${AFTER_EACH:-}" ] && eval "$AFTER_EACH"
done
cat "$OUT/summary.tsv"
