#!/usr/bin/env bash
# Text checks · local calibration round 1e (lab tier, Apple M5 Max), 2026-10-09.
#
# Findings so far: at 100M tokens no local model (dense, no-memory, 8-question) learned to copy a
# passage it had just seen, so no task could be learned. On a pure copy task the same model breaks
# out of a long flat stretch only by chance (one seed at ~700 steps, another not within 1,200): the
# budget was too short, nothing was broken.
# Question: how many text tokens until the 16M dense model learns to copy, and do the tasks follow?
# Round 1d (300M tokens): quoting a sign is learned (55 % vs 25 % guessing at 512 tokens), finding a
# person's home is not. One dense run to 1B tokens (fresh data), same probes: tokens to pass per task.
# repeats); copy probe and short-document exams along the way.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=Cache/text_checks/lab_r1/v1q8_1b
D=Cache/text_checks/lab_v1q8_1100m
stamp() { date -u "+%Y-%m-%d %H:%M:%S UTC"; }

echo "== data $(stamp)"
[ -f $D/text_checks_meta.json ] || uv run python scripts/build_text_checks_data.py --stories full --max_train_files 4 \
  --seq_len 1024 --tokenizer Cache/text_checks/v0_hub/tokenizer --target_tokens 1.1e9 --num_proc 12 \
  --eval_lengths 1024 2048 4096 --extra_lengths 1024 --eval_items 200 --world_version v1 --train_questions 8 --out_dir $D
echo "== train $(stamp)"
uv run python scripts/run_text_checks.py plan --phase train --tier lab --mode local --gpus 0 --arches dense \
  --tokens 1e9 --data $D --out $OUT
bash $OUT/launch/train_start.sh

echo "== along training $(stamp)"
for ck in $(ls -d $OUT/jobs/train_dense/train/*/checkpoint-* | awk -F- '{print $NF" "$0}' | sort -n | awk 'NR%4==0{print $2}'); do
  echo "-- $(basename $ck)"
  uv run python scripts/probe_copy.py --checkpoint $ck --device mps --gaps 0 100 --text Cache/text_checks/lab_v1 2>&1 | grep gap
  uv run python evaluation/text_checks_eval.py --checkpoint $ck --items Cache/text_checks/lab_v1/eval/id_short.jsonl \
    --tokenizer $D/tokenizer --lengths 512 --no_removed --out $OUT/short_$(basename $ck).json 2>&1 | grep -E "^id"
done
echo "== done $(stamp)"
