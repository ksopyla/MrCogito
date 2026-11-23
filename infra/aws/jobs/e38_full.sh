#!/bin/bash
# E38: the 100M models on the m21mix_v1 data mix (SmolLM2 tokenizer; docs/experiments_specs/ahead/E38_long_budget_mix.md),
# author 2026-11-14: "Train two best models 100M with a longer data budget, with a better tokenizer and data mix".
# Three phases from the mix's own budget (scripts/run_e33n.py --mix): trunk 8.0B tokens in rows of 4,096, decay 3.0B
# in rows of 16,384, long 0.5B in rows of 65,536; then the tests of the 64k model (e37_evals_long.sh: long-document
# loss by position 4k-128k, RULER 4k-32k with cached decoding, core) and the long-document loss of the 16k model.
# Speed settings of 2026-11-14 (--fast: fused loss, compiled concept grouping), the E37 runs' micro-batches.
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
SIZE="${E38_SIZE:-m100}"   # e38_board.sh: m120t, the Tiny-ML leaderboard size (shared input/output table)
# E38_OPT: adam (default) or muon:PEAK (Muon on the matrices; trunk and decay at PEAK, the 64k phase at a tenth)
OPT="${E38_OPT:-adam}"; OPT_FLAGS=""; TAG=""
case "$OPT" in
  adam) ;;
  muon:*) OPT_FLAGS="--optimizer muon --muon_peak ${OPT#muon:}"; TAG="muon${OPT#muon:}" ;;
  *) echo "[e38_full] unknown E38_OPT=$OPT"; exit 2 ;;
esac
# rows per pass and card: the E37 runs' own (4 at 4k, 1 at 16k on a 48 GB card; twice that on 80 GB). On one L40S
# with --fast, 4 / 6 / 8 / 10 rows at 4k all give 29-30.5k tokens/s (old settings: 20.3k) and 1 / 2 rows at 16k
# both ~20.3k (old: 15.2k), so larger passes only cost memory (eng_check_node.sh, us-west-2, 2026-11-14)
if [ "${MEM:-0}" -ge 70000 ]; then MT=8; MD=2; else MT=4; MD=1; fi
# E38_EXPERIMENT (e39_board.sh: E39): the W&B experiment id and the S3 folder runs/<lower-case id>/
EXPT="${E38_EXPERIMENT:-E38}"; EXPL=$(echo "$EXPT" | tr "[:upper:]" "[:lower:]")
export E33N_EXPERIMENT="$EXPT" E33N_SEED="$SEED" E33N_GPUS="$G" E33N_CKPT_S3=1 E33N_SKIP_EVAL=1 E33N_MIX="$MIX" \
  E33N_LONG=1 E33N_GRAD_CKPT=1 E33N_MICRO_TRUNK="${E38_MICRO_TRUNK:-$MT}" E33N_MICRO_DECAY="${E38_MICRO_DECAY:-$MD}" \
  E33N_MICRO_LONG=1 E33N_CMD_EXTRA="--fast --save_every ${E38_SAVE_EVERY:-800} $OPT_FLAGS" E33N_TAG="$TAG"
unset E33N_FLOW
HERE="$(cd "$(dirname "$0")" && pwd)"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
# plain names for the W&B board (author 2026-11-14: models named plainly on dashboards, codes as a secondary label)
case "$SIZE" in m120t) SZP="150M";; m100) SZP="100M";; s30) SZP="30M";; *) SZP="$SIZE";; esac
case "$OPT" in adam) OPTP="AdamW";; *) OPTP="Muon";; esac
for ARM in "$@"; do
  RUN="e33n_${SIZE}_${ARM}${TAG:+_$TAG}_tok${TT}_seed${SEED}"
  case "$ARM" in e37a) ARMP="concept memory";; lsa) ARMP="LSA port";; catalogue_reuse_n4) ARMP="catalogue";;
    dense_rope) ARMP="fair full attention";; dense) ARMP="full attention";; window) ARMP="window only";;
    living) ARMP="living concepts";; living_spans) ARMP="living spans";; *) ARMP="$ARM";; esac
  export E33N_PLAIN="$SZP $ARMP, $OPTP"
  export MC_WANDB_TAGS="$EXPT,$SIZE,$MIX,$ARM,opt-$OPT,seed$SEED${E38_BOARD:+,leaderboard}"
  export MC_WANDB_NOTES="$SZP $ARMP, $OPTP${E38_BOARD:+, for the Tiny-ML leaderboard}. Code: arm $ARM, size $SIZE, mix $MIX, optimizer $OPT, seed $SEED, run $RUN."
  R="$MC_DISK/runs/$RUN"; S3R="s3://$MC_BUCKET/runs/$EXPL/$RUN"
  if aws s3 ls "$S3R/eval_core_long.json" >/dev/null 2>&1; then
    echo "[e38_full] $ARM seed $SEED already has its tests in S3, skipped"; continue
  fi
  echo "[e38_full] $(ts) $ARM seed $SEED on $G GPUs, size $SIZE, mix $MIX, optimizer $OPT, trunk $TT"
  bash "$HERE/e33n_run.sh" "$ARM" "$SIZE" "" "$TT" || { echo "[e38_full] $(ts) $ARM training ended with code $?"; continue; }
  DECAY=$(ls -d "$R"/decay/*/final 2>/dev/null | tail -1)
  LONG=$(ls -d "$R"/long/*/final 2>/dev/null | tail -1)
  [ -n "$LONG" ] || { echo "[e38_full] $ARM has no 64k model"; continue; }
  export E33N_TOKENIZER="$D/$MIX/tokenizer" E33N_LONGDOCS="$D/$MIX" \
    E33N_LONGDOC_ARGS="--sources books,arxiv,doab,usgpo,hansard,pdfs,code_repos --eot 0 --tokenizer $D/$MIX/tokenizer"
  echo "[e38_full] $(ts) $ARM tests"
  if [ -n "$DECAY" ] && [ ! -f "$R/eval_longdoc_decay.json" ]; then   # what the 64k phase changed: the 16k model's loss
    # before the main tests and alone on its card (about a quarter of an hour): two evaluations on one card can
    # run it out of memory, and the main tests' marker must be the last file written
    CUDA_VISIBLE_DEVICES=0 uv run python evaluation/long_doc_loss.py --checkpoint "$DECAY" --data "$D/$MIX" \
      --lengths 4096,16384,65536 --rows 8 $E33N_LONGDOC_ARGS --out "$R/eval_longdoc_decay.json" \
      > "$R/eval_longdoc_decay.log" 2>&1 || echo "[e38_full] $ARM decay long-document loss FAILED"
    aws s3 cp --only-show-errors "$R/eval_longdoc_decay.json" "$S3R/eval_longdoc_decay.json" 2>/dev/null || true
  fi
  bash "$HERE/e37_evals_long.sh" "$LONG" "$R" "$D" "$S3R" || echo "[e38_full] $(ts) $ARM tests incomplete"
  aws s3 sync --only-show-errors "$R/" "$S3R/" || true
  echo "[e38_full] $(ts) $ARM done"
done
