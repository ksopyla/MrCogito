#!/bin/bash
# E33p stage 1, one arm on an AWS H100 node (a launch.sh job): the home-server recipe — trunk, 16k decay, long-document
# loss (notes / opened tokens on and off), RULER at 4k and 16k, core benchmarks — with the instrumentation on.
# Usage (launch.sh JOB words): infra/aws/jobs/e33p_s1.sh ARM [SEED]
#   SEED (default 1) is the trainer seed; the home servers run seed 42, so AWS adds a second seed per arm.
# Results: s3://BUCKET/runs/e33p/e33n_s30_<ARM>_seed<SEED>/ (W&B offline, synced from the laptop later).
set -uo pipefail
ARM="${1:?arm}"; SEED="${2:-1}"
# micro-batch 8 / 2: twice the home servers' 4 / 1, whose peak was ~21 GB of 24 (stage 0); the driver's H100
# default (16 / 4) was sized before E33p and would need ~80 GB. Same tokens per optimizer step either way.
export E33N_EXPERIMENT=E33p E33N_FLOW=1 E33N_SEED="$SEED" E33N_RULER_LENGTHS=4096,16384 \
  E33N_MICRO_TRUNK=8 E33N_MICRO_DECAY=2
exec bash "$(dirname "$0")/e33n_run.sh" "$ARM" s30
