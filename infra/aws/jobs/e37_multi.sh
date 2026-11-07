#!/bin/bash
# E37 stage A on one multi-GPU AWS node (a launch.sh job, e.g. g6e.12xlarge = 4 × L40S 48 GB, spot): the arms one
# after another at 100M with the screening budget, each model on all the node's GPUs, the three evals side by side.
# Screening budget (spec E37, stage A): FineWeb-Edu 1.0B tokens at 4k, then LongDocs-lite 0.2B tokens at 16k.
# Usage (launch.sh JOB words): infra/aws/jobs/e37_multi.sh SEED ARM [ARM ...]
#   E37_TRUNK_TOKENS / E37_DECAY_TOKENS override the budget; E37_MICRO_TRUNK / E37_MICRO_DECAY the micro-batch.
# Checkpoints go to S3 (E33N_CKPT_S3), so after a spot interruption the same launch resumes where it stopped;
# arms whose evals are already in S3 are skipped.
# Results: s3://BUCKET/runs/e37/e33n_m100_<ARM>_tok<T>_seed<SEED>/ (W&B offline, synced from the laptop later).
set -uo pipefail
SEED="${1:?seed}"; shift
G=$(nvidia-smi -L | wc -l)
TT="${E37_TRUNK_TOKENS:-1.0e9}"
export E33N_EXPERIMENT=E37 E33N_FLOW=1 E33N_SEED="$SEED" E33N_RULER_LENGTHS=4096,16384 E33N_GPUS="$G" \
  E33N_CKPT_S3=1 E33N_DECAY_TOKENS="${E37_DECAY_TOKENS:-2.0e8}" \
  E33N_MICRO_TRUNK="${E37_MICRO_TRUNK:-4}" E33N_MICRO_DECAY="${E37_MICRO_DECAY:-1}"
source /etc/mrcogito.env
for ARM in "$@"; do
  if aws s3 ls "s3://$MC_BUCKET/runs/e37/e33n_m100_${ARM}_tok${TT}_seed${SEED}/eval_core.json" >/dev/null 2>&1; then
    echo "[e37_multi] $ARM seed $SEED already has its evals in S3, skipped"; continue
  fi
  echo "[e37_multi] $(date -u +%Y-%m-%dT%H:%M:%SZ) $ARM seed $SEED on $G GPUs, trunk $TT"
  bash "$(dirname "$0")/e33n_run.sh" "$ARM" m100 "" "$TT" || echo "[e37_multi] $ARM ended with code $?"
done
