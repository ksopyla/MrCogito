#!/bin/bash
# List fine-tune test (root-cause work, 2026-11-23): a 150M model from its final weights, on 16k rows of which a
# share comes from one constructed set (a list set: questions that need several far values; or a lookup set: the
# control, every question one value) and the rest is the 16k phase's own sources (replay). The 64k phase's rates
# (Muon 0.001, AdamW part 6e-5, decaying to a tenth), 2 rows per card and optimizer step x cards.
# Usage: bash scripts/run_list_ft.sh NAME INIT REPLAY_MIX EXTRA_DIR SET GPUS [TOKENS] [WEIGHT] [ARM]
#   GPUS: comma-separated card numbers (e.g. 0,1,3). LR, ADAMW_LR, EVAL_EVERY, EXTRA (more run_e33n.py arguments).
# Everything lands in Cache/list_ft/NAME/ (events.log in UTC, train.log, decay/*/final).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
NAME="$1"; INIT="$2"; REPLAY="$3"; EXTRA_DIR="$4"; SET="$5"; GPUS="$6"; TOKENS="${7:-6e7}"; WEIGHT="${8:-0.5}"
ARM="${9:-e37a}"
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled
R="$PWD/Cache/list_ft/$NAME"
mkdir -p "$R"
say() { echo "[list_ft $NAME $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$R/events.log"; }
# REPLAY_PHASE (default decay: rows of 16k; long: rows of 64k) picks the replay mix's phase and so the row length;
# ROWS_PER_CARD (default 2; use 1 at 64k) rows per card and optimizer step. SET and EXTRA_DIR may be comma lists.
uv run --no-sync python scripts/make_list_ft_mix.py --replay "$REPLAY" --extra "$EXTRA_DIR" --set "$SET" \
  --out "$R/mix" --weight "$WEIGHT" --tokens "$TOKENS" --phase "${REPLAY_PHASE:-decay}" >> "$R/events.log" 2>&1 \
  || { say "the mix failed"; exit 1; }
n=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
# shellcheck disable=SC2086
CMD=$(uv run --no-sync python scripts/run_e33n.py cmd --arm "$ARM" --size m120t --phase decay --data "$R" --out "$R" \
  --init "$INIT" --mix "$R/mix" --mix_lenient --rows $(( ${ROWS_PER_CARD:-2} * n )) --micro 1 --gpus "$n" --grad_ckpt \
  --optimizer muon --lr "${LR:-0.001}" --muon_adamw_lr "${ADAMW_LR:-6e-5}" --eval_every "${EVAL_EVERY:-100}" \
  --save_every 200 --fast ${EXTRA:-}) || { say "the command failed"; exit 1; }
echo "$CMD" > "$R/train.cmd"
say "start on cards $GPUS: $TOKENS tokens, $SET at $WEIGHT of the rows, from $INIT"
CUDA_VISIBLE_DEVICES="$GPUS" bash -c "$CMD" > "$R/train.log" 2>&1
rc=$?
say "end exit $rc: $(ls -d "$R"/decay/*/final 2>/dev/null | tail -1)"
