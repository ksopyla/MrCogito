#!/bin/bash
# E41 screen at raised rates (2026-11-27): the retrofit's 16k part for several arms, one after another on one card,
# at Muon 3e-3 and AdamW 2e-4 (constant) instead of the released model's final 1e-4 / 6e-6. Why: at the final rates
# the token layers barely move in a screen, so every arm looks like the closed-book rows alone, while at the cloud
# runs' rates the reasoner arms fell behind by 0.06. Same rows, order and seed for every arm
# (scripts/e40_train.sh), 600 steps of 4 rows of 16,384 (39M tokens), evaluation every 120 steps.
# Usage (home server, repo root):
#   setsid nohup bash scripts/e41_rate_screen.sh GPU "ARM ARM ..." [OUT=Cache/e41/rate] > /dev/null 2>&1 &
#   MIX: the E40 home mix (default: scripts/e40_train.sh's). Events: OUT/<arm>/events.log (UTC).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
GPU="$1"; ARMS="$2"; OUT="${3:-Cache/e41/rate}"
export DECAY_STEPS="${DECAY_STEPS:-600}" DECAY_EVAL="${DECAY_EVAL:-120}" STOP_AFTER=decay
export MUON_LR="${MUON_LR:-3e-3}" ADAMW_LR="${ADAMW_LR:-2e-4}" TREE_LR="${TREE_LR:-2e-4}"
for arm in $ARMS; do
  [ -f "$OUT/$arm/decay.rc" ] && continue   # done before
  bash scripts/e40_train.sh "$arm" "$GPU" "$OUT/$arm"
done
