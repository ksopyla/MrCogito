#!/bin/bash
# E33n stage-0 learning-rate check (spec §Plan stage 0): the window control at SIZE, TOKENS per peak learning rate,
# one peak per GPU in parallel, warmup 5 % then constant then a 1−sqrt decay over the last 20 % (so the final losses
# are annealed and comparable). Decides the trunk peak learning rate. Free on Odra / Polonez.
# Usage (repo root, in tmux): bash infra/aws/jobs/lr_check.sh DATA OUT SIZE TOKENS MICRO LR...
#   e.g. bash infra/aws/jobs/lr_check.sh ~/data Cache/e33n_lr s30 1e8 4 5e-4 1e-3 2e-3 4e-3
# Result: OUT/lr_summary.tsv (peak lr, final train loss, final validation loss).
set -uo pipefail
D="$1"; OUT="$2"; SIZE="$3"; TOKENS="$4"; MICRO="$5"; shift 5
export PATH="$HOME/.local/bin:$PATH" WANDB_MODE=offline TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_DIR="$OUT/wandb"
mkdir -p "$OUT" "$WANDB_DIR"
ROWS=$(uv run python -c "import sys; sys.path.insert(0, '.'); from scripts.run_e33n import SIZES; print(SIZES['$SIZE']['rows'])")
STEPS=$(python3 -c "print(max(1, round(float('$TOKENS') / ($ROWS * 4096))))")
WARM=$(python3 -c "print(max(1, int(0.05 * $STEPS)))")
DEC=$(python3 -c "print(max(1, int(0.2 * $STEPS)))")
g=0
for lr in "$@"; do
  CMD=$(uv run python scripts/run_e33n.py cmd --arm window --size "$SIZE" --phase trunk --data "$D" \
        --out "$OUT/lr$lr" --tokens "$TOKENS" --micro "$MICRO" --lr "$lr")
  CMD="$CMD --lr_scheduler_type warmup_stable_decay --warmup_steps $WARM --lr_scheduler_kwargs \
'{\"num_decay_steps\": $DEC, \"decay_type\": \"1-sqrt\", \"min_lr_ratio\": 0.1}' --eval_strategy steps --eval_steps $STEPS"
  echo "$CMD" > "$OUT/lr$lr.cmd"
  echo "== $(date -u +%H:%M:%S) lr $lr on GPU $g ($STEPS steps, warmup $WARM, decay $DEC)"
  ( export CUDA_VISIBLE_DEVICES=$g; eval "$CMD" ) > "$OUT/lr$lr.log" 2>&1 &
  g=$((g + 1))
done
wait
printf 'lr\ttrain_loss_last\teval_loss\n' > "$OUT/lr_summary.tsv"
for lr in "$@"; do
  tl=$(grep -oE "'loss': [0-9.]+" "$OUT/lr$lr.log" | tail -1 | awk '{print $2}')
  el=$(grep -oE "'eval_loss': [0-9.]+" "$OUT/lr$lr.log" | tail -1 | awk '{print $2}')
  printf '%s\t%s\t%s\n' "$lr" "${tl:-?}" "${el:-?}" >> "$OUT/lr_summary.tsv"
done
cat "$OUT/lr_summary.tsv"
echo "lr check done $(date -u)"
