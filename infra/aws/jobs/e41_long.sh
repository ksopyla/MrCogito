#!/bin/bash
# E41 (docs/experiments_specs/ahead/E41_tree_concept_model.md): the stage on rows of 262,144 tokens, read after the
# 64k stage. No row was ever longer than 65,536 tokens because the concept read scores every concept of the row for
# every query; the top-down tree read (tree_sparse) scores only the children of followed nodes, so its cost per
# token does not grow with the row. Mix m21mix_xl1 (single documents that long, planted-fact recall rows over
# several books, packed shorter documents), arm tree_xl (bounded and cut reads only, by a node's own estimate).
#
# This is e33n_run.sh's `long` phase on another mix: the run's folder is seeded beforehand from the laptop (admin
# profile: the node's role may not copy between S3 folders) with the model it continues, as BOTH trunk/<x>/final
# and decay/<x>/final, so the runner finds stages 1 and 2 done and starts at `long` from decay's final weights.
# Usage (launch.sh JOB words): infra/aws/jobs/e41_long.sh [ARM] [TAG] [PEAK]
#   ARM default tree_xl (also tree_xl_wide); TAG default xl: a word in the run name; PEAK default 0.01: the Muon
#   peak the stage's rate starts at a tenth of (1e-3, where the 64k stage started; a step holds four rows, 1M
#   tokens, four times the 64k stage's).
# Results: s3://BUCKET/runs/e41/e33n_m120t_<ARM>_muon<PEAK>-<TAG>_tok8.0e9_seed42/
set -uo pipefail
source /etc/mrcogito.env
ARM="${1:-tree_xl}"; RTAG="${2:-xl}"; PEAK="${3:-0.01}"
DST="s3://$MC_BUCKET/runs/e41/e33n_m120t_${ARM}_muon${PEAK}-${RTAG}_tok8.0e9_seed42"
for ph in trunk decay; do
  have=$(aws s3 ls --recursive "$DST/$ph/" 2>/dev/null || true)
  case "$have" in
    *final/model.safetensors*) echo "[e41-long] the model to continue is in $DST/$ph/" ;;
    *) echo "[e41-long] no weights under $DST/$ph/ (seed the folder from the laptop first)"; exit 3 ;;
  esac
done
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU" in *H100*) export E38_SAVE_EVERY="${E38_SAVE_EVERY:-200}" ;; *) export E38_SAVE_EVERY="${E38_SAVE_EVERY:-100}" ;; esac
export E38_EXPERIMENT=E41 E38_MIX=m21mix_xl1 E38_TAG="$RTAG" E38_PLAIN_NOTE="on 256k rows" E33N_LONG_STAGE="stage 4"
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/e38_board.sh" 42 "muon:$PEAK" "$ARM"
RC=$?
# One number hidden in a very long text (scripts/e41_long_needle.py), here because a row of 1M tokens does not fit a
# 24 GB home card: the trained model at 64k … 1M by its own read and by 16 followed per level, and the model it
# continued at 1M through the same top-down read (its shorter lengths are measured at home). After the tests'
# marker, so a spot interruption here is not relaunched: the results are then measured at home up to 512k.
RUN="e33n_m120t_${ARM}_muon${PEAK}-${RTAG}_tok8.0e9_seed42"
R="$MC_DISK/runs/$RUN"; D="$MC_DISK/data/m21mix_xl1"
LONG=$(ls -d "$R"/long/*/final 2>/dev/null | tail -1)
SEED=$(ls -d "$R"/decay/*/final 2>/dev/null | tail -1)
if [ -n "$LONG" ] && [ ! -f "$R/needle_long.json" ]; then
  echo "[e41-long] $(date -u +%Y-%m-%dT%H:%M:%SZ) needle test of the trained model"
  ( CUDA_VISIBLE_DEVICES=0 uv run python scripts/e41_long_needle.py --checkpoint "$LONG" --mix "$D" \
      --lengths 65536,131072,262144,524288,1048576 --depths 8 --modes on,descent16 --top_down \
      --out "$R/needle_long.json" > "$R/needle_long.log" 2>&1
    aws s3 cp --only-show-errors "$R/needle_long.json" "$DST/needle_long.json" || true ) &
  if [ -n "$SEED" ]; then
    ( uv run python scripts/e41_tree_view.py view "$SEED" "$R/seed_treeview" > "$R/needle_seed.log" 2>&1 \
      && CUDA_VISIBLE_DEVICES=1 uv run python scripts/e41_long_needle.py --checkpoint "$R/seed_treeview" --mix "$D" \
        --lengths 262144,1048576 --depths 8 --modes descent16,descent64 --top_down \
        --out "$R/needle_seed.json" >> "$R/needle_seed.log" 2>&1
      aws s3 cp --only-show-errors "$R/needle_seed.json" "$DST/needle_seed.json" || true ) &
  fi
  wait
  aws s3 cp --only-show-errors "$R/needle_long.log" "$DST/needle_long.log" || true
  aws s3 cp --only-show-errors "$R/needle_seed.log" "$DST/needle_seed.log" || true
  echo "[e41-long] $(date -u +%Y-%m-%dT%H:%M:%SZ) needle tests done"
fi
exit "$RC"
