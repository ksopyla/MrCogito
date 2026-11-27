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
exec bash "$(cd "$(dirname "$0")" && pwd)/e38_board.sh" 42 "muon:$PEAK" "$ARM"
