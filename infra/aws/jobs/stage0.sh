#!/bin/bash
# E33n stage 0 (one H100): speed and memory of every arm × size on short runs, so the budget becomes hours.
#   trunk (4k rows): s30 × 4 arms (~80 steps), m100 and l300 × {dense, e33n} (~30 steps)
#   decay (16k rows): s30 × {dense, e33n} and m100 e33n (~20 steps) — memory of the long rows
# Results: s3://BUCKET/runs/stage0/ (logs, per-run GPU memory trace, summary.tsv).
set -uo pipefail
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" WANDB_MODE=offline WANDB_DIR="$MC_DISK/wandb" TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_IMPLICIT_TOKEN=1
D="$MC_DISK/data"; R="$MC_DISK/runs/stage0"
mkdir -p "$D/fineweb_edu_gpt2" "$R" "$WANDB_DIR"
aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/finewebedu_val_000000.bin" "$D/fineweb_edu_gpt2/"
for i in 01 02 03; do
  aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/finewebedu_train_0000$i.bin" "$D/fineweb_edu_gpt2/"
done
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/tokenizer_gpt2/" "$D/tokenizer_gpt2/"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/longdocs_lite_gpt2/" "$D/longdocs_lite_gpt2/" --exclude "*" \
  --include "manifest.json" --include "*_train_000000.bin" --include "*_val_000000.bin" || true
printf 'size\tarm\tphase\tmicro\tstatus\ttok_per_s\tpeak_mem_gb\tseconds\n' > "$R/summary.tsv"

sync_up() {
  aws s3 sync --only-show-errors "$R/" "s3://$MC_BUCKET/runs/stage0/" --exclude "*/checkpoint-*" \
    --exclude "*.safetensors" --exclude "*optimizer.pt"
}

one() {  # size arm phase tokens micro
  local size=$1 arm=$2 phase=$3 tokens=$4 micro=$5 init="" t0 rc log tps mem smi
  log="$R/${size}_${arm}_${phase}.log"
  if [ "$phase" = decay ]; then
    init=$(ls -d "$R/${size}_${arm}"/trunk/*/final 2>/dev/null | tail -1)
    [ -n "$init" ] && init="--init $init"
  fi
  CMD=$(uv run python scripts/run_e33n.py cmd --arm "$arm" --size "$size" --phase "$phase" --data "$D" \
        --out "$R/${size}_${arm}" --tokens "$tokens" --micro "$micro" $init)
  echo "== $(date -u +%H:%M:%S) $size $arm $phase micro $micro"
  t0=$(date +%s)
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > "$log.gpu" &
  smi=$!
  ( eval "$CMD" ) > "$log" 2>&1; rc=$?
  kill $smi 2>/dev/null
  tps=$(grep -oE "'perf/real_tokens_per_second': [0-9.]+" "$log" | tail -5 | awk '{s+=$2;n++} END {if(n) printf "%.0f", s/n}')
  mem=$(awk -F', ' '{if ($1+0>m) m=$1+0} END {printf "%.1f", m/1024}' "$log.gpu")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$size" "$arm" "$phase" "$micro" "$([ $rc = 0 ] && echo ok || echo FAIL)" \
    "${tps:-?}" "$mem" "$(( $(date +%s) - t0 ))" >> "$R/summary.tsv"
  [ $rc = 0 ] || tail -25 "$log"
  sync_up
}

for arm in window dense e33 e33n; do one s30 "$arm" trunk 2.1e7 16; done
for arm in dense e33n; do one m100 "$arm" trunk 1.6e7 8; done
for arm in dense e33n; do one l300 "$arm" trunk 2.1e7 8; done
# long rows: wait (≤ 40 min) for LongDocs-lite from the data node
for _ in $(seq 1 40); do
  aws s3 ls "s3://$MC_BUCKET/data/longdocs_lite_gpt2/manifest.json" >/dev/null 2>&1 && break
  sleep 60
done
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/longdocs_lite_gpt2/" "$D/longdocs_lite_gpt2/" --exclude "*" \
  --include "manifest.json" --include "*_train_000000.bin" --include "*_val_000000.bin" || true
one s30 dense decay 2.6e6 4
one s30 e33n decay 2.6e6 4
one m100 e33n decay 2.6e6 2
cat "$R/summary.tsv"
sync_up
