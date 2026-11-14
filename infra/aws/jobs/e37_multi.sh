#!/bin/bash
# E37 stage A on one multi-GPU AWS node (a launch.sh job, e.g. g6e.12xlarge = 4 × L40S 48 GB, spot): the arms one
# after another at 100M with the screening budget, each model on all the node's GPUs; then its evals spread over
# all the GPUs (e37_evals.sh: RULER 16k split by task, so no GPU idles while one runs it for hours).
# Screening budget (spec E37, stage A): FineWeb-Edu 1.0B tokens at 4k, then LongDocs-lite 0.2B tokens at 16k.
# The concept arms train with the length scale on (concept_sum_len = 16,384, scripts/run_e33n.py; inert in training)
# and are also scored with it off (eval_longdoc_nosum.json).
# Usage (launch.sh JOB words): infra/aws/jobs/e37_multi.sh SEED ARM [ARM ...]   (most important arm first)
#   E37_TRUNK_TOKENS / E37_DECAY_TOKENS override the budget; E37_MICRO_TRUNK / E37_MICRO_DECAY the micro-batch.
# Checkpoints go to S3 (E33N_CKPT_S3), so after a spot interruption the same launch resumes where it stopped;
# arms whose eval_core.json is in S3 (written last, when all evals are done) are skipped.
# Results: s3://BUCKET/runs/e37/e33n_m100_<ARM>_tok<T>_seed<SEED>/ (W&B offline, synced from the laptop later).
set -uo pipefail
SEED="${1:?seed}"; shift
G=$(nvidia-smi -L | wc -l)
TT="${E37_TRUNK_TOKENS:-1.0e9}"
export E33N_EXPERIMENT=E37 E33N_FLOW=1 E33N_SEED="$SEED" E33N_GPUS="$G" E33N_CKPT_S3=1 E33N_SKIP_EVAL=1 \
  E33N_DECAY_TOKENS="${E37_DECAY_TOKENS:-2.0e8}" \
  E33N_MICRO_TRUNK="${E37_MICRO_TRUNK:-4}" E33N_MICRO_DECAY="${E37_MICRO_DECAY:-1}"
source /etc/mrcogito.env
HERE="$(cd "$(dirname "$0")" && pwd)"
for ARM in "$@"; do
  RUN="e33n_m100_${ARM}_tok${TT}_seed${SEED}"
  R="$MC_DISK/runs/$RUN"; S3R="s3://$MC_BUCKET/runs/e37/$RUN"
  if aws s3 ls "$S3R/eval_core.json" >/dev/null 2>&1; then
    echo "[e37_multi] $ARM seed $SEED already has its evals in S3, skipped"; continue
  fi
  echo "[e37_multi] $(date -u +%Y-%m-%dT%H:%M:%SZ) $ARM seed $SEED on $G GPUs, trunk $TT"
  bash "$HERE/e33n_run.sh" "$ARM" m100 "" "$TT" || { echo "[e37_multi] $ARM training ended with code $?"; continue; }
  DECAY=$(ls -d "$R"/decay/*/final 2>/dev/null | tail -1)
  [ -n "$DECAY" ] || { echo "[e37_multi] $ARM has no decay model"; continue; }
  echo "[e37_multi] $(date -u +%Y-%m-%dT%H:%M:%SZ) $ARM evals"
  bash "$HERE/e37_evals.sh" "$DECAY" "$R" "$MC_DISK/data" || echo "[e37_multi] $ARM evals incomplete"
  aws s3 sync --only-show-errors "$R/" "$S3R/" || true
  echo "[e37_multi] $(date -u +%Y-%m-%dT%H:%M:%SZ) $ARM done"
done
