#!/usr/bin/env bash
# Text checks · local calibration round 1b (lab tier, Apple M5 Max), 2026-10-08.
#
# Finding of round 1 (dense, v1 data, 100M tokens at 1k): every task at the guessing rate, even on
# 256-token documents; the model learned the answer format (copy an invented place) but not which
# person it belongs to. One ~3-token answer per ~550-token document is ~0.5 % task signal.
# Question: does more answer signal per token make the tasks learnable at this size?
#   q8   v1, 8 questions per training document (different cast members), stories 45 % of tokens
#   q8w  the same with stories 20 % of tokens (more world documents)
# Dense first on both (same step size 2e-3, same 100M tokens); exams unchanged (one question).
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=Cache/text_checks/lab_r1
RUN="uv run python scripts/run_text_checks.py"
stamp() { date -u "+%Y-%m-%d %H:%M:%S UTC"; }

# wait for the data and for round 1's v1 queue (one job at a time on the Apple GPU)
until [ -f Cache/text_checks/lab_v1q8/text_checks_meta.json ] && [ -f Cache/text_checks/lab_v1q8w/text_checks_meta.json ]; do sleep 30; done
until [ -f $OUT/v1/jobs/eval_local/DONE ]; do sleep 60; done

for d in v1q8 v1q8w; do
  echo "== train dense $d $(stamp)"
  $RUN plan --phase train --tier lab --mode local --gpus 0 --arches dense --data Cache/text_checks/lab_$d --out $OUT/$d
  bash $OUT/$d/launch/train_start.sh
  uv run python evaluation/text_checks_eval.py --checkpoint "$(cat $OUT/$d/jobs/train_dense/model_path)" \
    --items Cache/text_checks/lab_v1/eval/id_short.jsonl --tokenizer Cache/text_checks/lab_$d/tokenizer \
    --out $OUT/$d/jobs/eval_dense/short.json
done
echo "== done $(stamp)"
