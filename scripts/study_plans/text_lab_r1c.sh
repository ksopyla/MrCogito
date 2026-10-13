#!/usr/bin/env bash
# Text checks · local calibration round 1c (lab tier, Apple M5 Max), 2026-10-09.
#
# Finding of rounds 1/1b: the 16M dense model after 100M tokens cannot copy a passage it has just
# seen (real text or random tokens), so no task can be learned; 8 questions per document did not help.
# Question: does the E31 input layer (hashed n-gram features + value embeddings, which make local
# prediction cheap) delay the copy ability? Diagnostic control: the same dense model without it,
# same data (v1, one question), step size and tokens. Copy probe on its checkpoints and the dense ones.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=Cache/text_checks/lab_r1
stamp() { date -u "+%Y-%m-%d %H:%M:%S UTC"; }
until ! pgrep -f text_lab_r1b.sh >/dev/null; do sleep 60; done   # one job at a time on the Apple GPU

echo "== train dense_plain v1 $(stamp)"
bash $OUT/v1_plain/launch/train_start.sh
uv run python evaluation/text_checks_eval.py --checkpoint "$(cat $OUT/v1_plain/jobs/train_dense_plain/model_path)" \
  --items Cache/text_checks/lab_v1/eval/id_short.jsonl --tokenizer Cache/text_checks/lab_v1/tokenizer \
  --out $OUT/v1_plain/jobs/eval_dense_plain/short.json

echo "== copy probe over training $(stamp)"
for run in v1/jobs/train_dense v1_plain/jobs/train_dense_plain; do
  for ck in $(ls -d $OUT/$run/train/*/checkpoint-* | awk -F- '{print $NF" "$0}' | sort -n | awk 'NR%4==0{print $2}'); do
    echo "-- $run $(basename $ck)"
    uv run python scripts/probe_copy.py --checkpoint $ck --device mps --gaps 0 100 --text Cache/text_checks/lab_v1 2>&1 | grep gap
  done
done
echo "== done $(stamp)"
