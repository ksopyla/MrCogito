#!/bin/bash
# E33p stage 1 on one multi-GPU AWS node (a launch.sh job, e.g. g6e.12xlarge = 4 × L40S 48 GB, spot): the arms one
# after another, each model on all the node's GPUs (data parallel), the three evals side by side on separate GPUs.
# Same recipe as the home servers (e33p_s1.sh): trunk, 16k decay, long-document loss, RULER 4k/16k, core benchmarks,
# instrumentation on; the same rows per optimizer step, split as micro 4 × accumulation 4 × 4 GPUs (trunk) and
# 1 × 4 × 4 (decay) — the home servers' micro-batches, which fit in 24 GB.
# Usage (launch.sh JOB words): infra/aws/jobs/e33p_s1_multi.sh SEED ARM [ARM ...]
# Checkpoints go to S3 (E33N_CKPT_S3), so after a spot interruption the same launch resumes where it stopped;
# arms whose evals are already in S3 are skipped.
# Results: s3://BUCKET/runs/e33p/e33n_s30_<ARM>_seed<SEED>/ (W&B offline, synced from the laptop later).
set -uo pipefail
SEED="${1:?seed}"; shift
G=$(nvidia-smi -L | wc -l)
# E33N_EXPERIMENT / E33N_RULER_LENGTHS may be set by a wrapper (E33t stage A: e33t_a.sh)
export E33N_EXPERIMENT="${E33N_EXPERIMENT:-E33p}" E33N_FLOW=1 E33N_SEED="$SEED" \
  E33N_RULER_LENGTHS="${E33N_RULER_LENGTHS:-4096,16384}" E33N_GPUS="$G" \
  E33N_CKPT_S3=1 E33N_MICRO_TRUNK=4 E33N_MICRO_DECAY=1
source /etc/mrcogito.env
for ARM in "$@"; do
  EXP=$(echo "$E33N_EXPERIMENT" | tr "[:upper:]" "[:lower:]")
  if aws s3 ls "s3://$MC_BUCKET/runs/$EXP/e33n_s30_${ARM}_seed${SEED}/eval_core.json" >/dev/null 2>&1; then
    echo "[e33p_s1_multi] $ARM seed $SEED already has its evals in S3, skipped"; continue
  fi
  echo "[e33p_s1_multi] $(date -u +%Y-%m-%dT%H:%M:%SZ) $ARM seed $SEED on $G GPUs"
  bash "$(dirname "$0")/e33n_run.sh" "$ARM" s30 || echo "[e33p_s1_multi] $ARM ended with code $?"
done
