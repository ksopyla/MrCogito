#!/bin/bash
# E38: the 100M models on the m21mix_v1 data mix (SmolLM2 tokenizer; docs/experiments_specs/ahead/E38_long_budget_mix.md),
# author 2026-11-14: "Train two best models 100M with a longer data budget, with a better tokenizer and data mix".
# Three phases from the mix's own budget (scripts/run_e33n.py --mix): trunk 8.0B tokens in rows of 4,096, decay 3.0B
# in rows of 16,384, long 0.5B in rows of 65,536; then the tests of the 64k model (e37_evals_long.sh: long-document
# loss by position 4k-128k, RULER 4k-32k with cached decoding, core) and the long-document loss of the 16k model.
# Speed settings of 2026-11-14 (--fast: fused loss, compiled concept grouping) with larger micro-batches.
# Each arm trains on all the node's GPUs, one after another; checkpoints go to S3, so after a spot interruption the
# same launch resumes; an arm whose eval_core_long.json is in S3 is skipped. A launch.sh job cannot pass environment
# variables to the node, hence this wrapper.
# Usage (launch.sh JOB words): infra/aws/jobs/e38_full.sh SEED ARM [ARM ...]
# Results: s3://BUCKET/runs/e38/e33n_m100_<ARM>_tok8.0e9_seed<SEED>/
set -uo pipefail
SEED="${1:?seed}"; shift
G=$(nvidia-smi -L | wc -l)
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
source /etc/mrcogito.env
MIX="${E38_MIX:-m21mix_v1}"; TT="${E38_TRUNK_TOKENS:-8.0e9}"; D="$MC_DISK/data"
# rows per pass and card with the fused loss (L40S 48 GB: eng_check_node.sh, 2026-11-14; 80 GB cards take twice that)
if [ "${MEM:-0}" -ge 70000 ]; then MT=16; MD=4; else MT=8; MD=2; fi
export E33N_EXPERIMENT=E38 E33N_SEED="$SEED" E33N_GPUS="$G" E33N_CKPT_S3=1 E33N_SKIP_EVAL=1 E33N_MIX="$MIX" \
  E33N_LONG=1 E33N_GRAD_CKPT=1 E33N_MICRO_TRUNK="${E38_MICRO_TRUNK:-$MT}" E33N_MICRO_DECAY="${E38_MICRO_DECAY:-$MD}" \
  E33N_MICRO_LONG=1 E33N_CMD_EXTRA="--fast --save_every ${E38_SAVE_EVERY:-400}"
unset E33N_FLOW
HERE="$(cd "$(dirname "$0")" && pwd)"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
for ARM in "$@"; do
  RUN="e33n_m100_${ARM}_tok${TT}_seed${SEED}"
  R="$MC_DISK/runs/$RUN"; S3R="s3://$MC_BUCKET/runs/e38/$RUN"
  if aws s3 ls "$S3R/eval_core_long.json" >/dev/null 2>&1; then
    echo "[e38_full] $ARM seed $SEED already has its tests in S3, skipped"; continue
  fi
  echo "[e38_full] $(ts) $ARM seed $SEED on $G GPUs, mix $MIX, trunk $TT"
  bash "$HERE/e33n_run.sh" "$ARM" m100 "" "$TT" || { echo "[e38_full] $(ts) $ARM training ended with code $?"; continue; }
  DECAY=$(ls -d "$R"/decay/*/final 2>/dev/null | tail -1)
  LONG=$(ls -d "$R"/long/*/final 2>/dev/null | tail -1)
  [ -n "$LONG" ] || { echo "[e38_full] $ARM has no 64k model"; continue; }
  export E33N_TOKENIZER="$D/$MIX/tokenizer" E33N_LONGDOCS="$D/$MIX" \
    E33N_LONGDOC_ARGS="--sources books,arxiv,doab,usgpo,hansard,pdfs,code_repos --eot 0 --tokenizer $D/$MIX/tokenizer"
  echo "[e38_full] $(ts) $ARM tests"
  if [ -n "$DECAY" ] && [ ! -f "$R/eval_longdoc_decay.json" ]; then   # what the 64k phase changed: the 16k model's loss
    ( CUDA_VISIBLE_DEVICES=0 uv run python evaluation/long_doc_loss.py --checkpoint "$DECAY" --data "$D/$MIX" \
        --lengths 4096,16384,65536 --rows 8 $E33N_LONGDOC_ARGS --out "$R/eval_longdoc_decay.json" \
        > "$R/eval_longdoc_decay.log" 2>&1 || echo "[e38_full] $ARM decay long-document loss FAILED" ) &
  fi
  bash "$HERE/e37_evals_long.sh" "$LONG" "$R" "$D" "$S3R" || echo "[e38_full] $(ts) $ARM tests incomplete"
  wait
  aws s3 sync --only-show-errors "$R/" "$S3R/" || true
  echo "[e38_full] $(ts) $ARM done"
done
