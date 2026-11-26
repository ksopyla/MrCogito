#!/bin/bash
# E41 (docs/experiments_specs/ahead/E41_tree_concept_model.md, author 2026-11-26 "Build and train 150M"): the tree
# concept model, stages 2 and 3 on m21mix_v4 from concept memory's stage-1 weights (the E38 recall run's recipe,
# so the comparison with it is the architecture alone). Every new part starts as an identity (the reasoner's
# outputs are zero, the tree read with every node followed is the concept read), so step 0 is concept memory's
# stage 1; the reasoner's weights are new.
# The run's folder is seeded with the stage-1 final weights beforehand from the laptop (admin profile: the node's
# role may not copy between S3 folders); the copy here is only a fallback, as in e38_lists.sh.
# Usage (launch.sh JOB words): infra/aws/jobs/e41_tree.sh [ARM] [TAG]
#   ARM default tree (also tree_noaux, tree_read); TAG: one more word in the run name (a second run of an arm).
# Results: s3://BUCKET/runs/e41/e33n_m120t_<ARM>_muon0.01[-TAG]_tok8.0e9_seed42/
set -uo pipefail
source /etc/mrcogito.env
ARM="${1:-tree}"; RTAG="${2:-}"
SRC="s3://$MC_BUCKET/runs/e38/e33n_m120t_e37a_muon0.01_tok8.0e9_seed42"
DST="s3://$MC_BUCKET/runs/e41/e33n_m120t_${ARM}_muon0.01${RTAG:+-$RTAG}_tok8.0e9_seed42"
have=$(aws s3 ls --recursive "$DST/trunk/" 2>/dev/null || true)
case "$have" in
  *final/model.safetensors*) ;;
  *) echo "[e41] $(date -u +%Y-%m-%dT%H:%M:%SZ) stage-1 weights from $SRC"
     aws s3 sync --only-show-errors --copy-props none "$SRC/trunk/" "$DST/trunk/" --exclude "*/checkpoint-*" || true
     have=$(aws s3 ls --recursive "$DST/trunk/" 2>/dev/null || true) ;;
esac
case "$have" in
  *final/model.safetensors*) echo "[e41] stage-1 weights are in $DST/trunk/" ;;
  *) echo "[e41] no stage-1 weights under $DST/trunk/"; exit 3 ;;
esac
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU" in *H100*) export E38_SAVE_EVERY="${E38_SAVE_EVERY:-400}" ;; esac
export E38_EXPERIMENT=E41 E38_MIX=m21mix_v4 E38_PLAIN_NOTE="with recall questions"
[ -n "$RTAG" ] && export E38_TAG="$RTAG"
exec bash "$(cd "$(dirname "$0")" && pwd)/e38_board.sh" 42 muon:0.01 "$ARM"
