#!/bin/bash
# E33n stage 0 (one H100): speed and memory of every arm × size on short runs, so the budget becomes hours.
#   trunk (4k rows): s30 × 4 arms (~80 steps), m100 and l300 × {dense, e33n} (~30 steps)
#   decay (16k rows): s30 × {dense, e33n} and m100 e33n (~20 steps) — memory of the long rows
# Results: s3://BUCKET/runs/stage0/ (logs, per-run GPU memory trace, summary.tsv). Measurement: speed_check.sh.
set -uo pipefail
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" WANDB_DIR="$MC_DISK/wandb"
D="$MC_DISK/data"; R="$MC_DISK/runs/stage0"
mkdir -p "$D/fineweb_edu_gpt2" "$R" "$WANDB_DIR"
aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/finewebedu_val_000000.bin" "$D/fineweb_edu_gpt2/"
for i in 01 02 03; do
  aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/finewebedu_train_0000$i.bin" "$D/fineweb_edu_gpt2/"
done
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/tokenizer_gpt2/" "$D/tokenizer_gpt2/"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/longdocs_lite_gpt2/" "$D/longdocs_lite_gpt2/" --exclude "*" \
  --include "manifest.json" --include "*_train_000000.bin" --include "*_val_000000.bin"

export AFTER_EACH="aws s3 sync --only-show-errors '$R/' 's3://$MC_BUCKET/runs/stage0/' --exclude '*/checkpoint-*' \
  --exclude '*.safetensors' --exclude '*optimizer.pt'"
bash infra/aws/jobs/speed_check.sh "$D" "$R" \
  s30:window:trunk:2.1e7:16 s30:dense:trunk:2.1e7:16 s30:e33:trunk:2.1e7:16 s30:e33n:trunk:2.1e7:16 \
  m100:dense:trunk:1.6e7:8 m100:e33n:trunk:1.6e7:8 l300:dense:trunk:2.1e7:8 l300:e33n:trunk:2.1e7:8 \
  s30:dense:decay:2.6e6:4 s30:e33n:decay:2.6e6:4 m100:e33n:decay:2.6e6:2
eval "$AFTER_EACH"
