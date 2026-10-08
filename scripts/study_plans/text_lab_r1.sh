#!/usr/bin/env bash
# Text checks · local data calibration, round 1 (lab tier on the Apple M5 Max), 2026-10-08.
#
# Question: on data v1 (shortcuts fixed) vs v0 (the published draft), which tasks does a 16M dense model
# learn in 100M tokens at 1k, how fast, and which does the no-long-memory control also pass?
#   1. step size: dense and local, 5e-4, 1e-3 and 2e-3, 12M tokens on v1; chosen by dev loss only
#   2. dense, local on v1 (100M tokens) + exams at 1k/2k/4k
#   3. dense, local on v0 (same step sizes) + exams
#   4. e31c on v1 (notebook model) + exams, if the first rows look sane
# Every job is resumable (DONE markers); re-running this script skips finished work.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=Cache/text_checks/lab_r1
RUN="uv run python scripts/run_text_checks.py"
stamp() { date -u "+%Y-%m-%d %H:%M:%S UTC"; }

echo "== tune $(stamp)"
$RUN plan --phase tune --tier lab --mode local --gpus 0 --arches dense local --data Cache/text_checks/lab_v1 --out $OUT/tune
bash $OUT/tune/launch/tune_start.sh
$RUN select --out $OUT/tune --tier lab --write

for v in v1 v0; do
  echo "== train $v $(stamp)"
  $RUN plan --phase train --tier lab --mode local --gpus 0 --arches dense local --data Cache/text_checks/lab_$v --out $OUT/$v
  bash $OUT/$v/launch/train_start.sh
done

echo "== train e31c v1 $(stamp)"
$RUN plan --phase train --tier lab --mode local --gpus 0 --arches e31c --data Cache/text_checks/lab_v1 --out $OUT/v1_e31c
bash $OUT/v1_e31c/launch/train_start.sh
echo "== all done $(stamp)"
