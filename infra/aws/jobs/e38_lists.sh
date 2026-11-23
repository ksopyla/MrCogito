#!/bin/bash
# E38 on the mix with list questions (m21mix_v3, 2026-11-23): stages 2 and 3 of a 150M model again, from the
# stage-1 weights its E38 run already has in S3. Stage 1 (rows of 4,096) reads no constructed rows, so it is the
# same for both mixes; v3 adds the list sets to stages 2 and 3 (3 % and 5 % of the rows, from the pdfs share).
# The run gets its own folder (…-lists…), seeded with the stage-1 final weights (server side, no checkpoints), so
# infra/aws/jobs/e33n_run.sh starts at stage 2; tests and resume behaviour as every E38 board run.
# Usage (launch.sh JOB words): infra/aws/jobs/e38_lists.sh [ARM]        (default e37a, concept memory)
# Results: s3://BUCKET/runs/e38/e33n_m120t_<ARM>_muon0.01-lists_tok8.0e9_seed42/
set -uo pipefail
source /etc/mrcogito.env
ARM="${1:-e37a}"
SRC="s3://$MC_BUCKET/runs/e38/e33n_m120t_${ARM}_muon0.01_tok8.0e9_seed42"
DST="s3://$MC_BUCKET/runs/e38/e33n_m120t_${ARM}_muon0.01-lists_tok8.0e9_seed42"
# The listing goes into a variable and is matched there: `aws s3 ls | grep -q` under pipefail fails whenever grep
# stops reading at its first match (2026-11-23 18:44 UTC: "Broken pipe", the weights were there). The node's role
# may not copy between S3 folders either (no object-tag permission), so the stage-1 weights are put there
# beforehand from the laptop (the sync below, run with the admin profile); here the copy is only a fallback.
have=$(aws s3 ls --recursive "$DST/trunk/" 2>/dev/null || true)
case "$have" in
  *final/model.safetensors*) ;;
  *) echo "[e38_lists] $(date -u +%Y-%m-%dT%H:%M:%SZ) stage-1 weights from $SRC"
     aws s3 sync --only-show-errors --copy-props none "$SRC/trunk/" "$DST/trunk/" --exclude "*/checkpoint-*" || true
     have=$(aws s3 ls --recursive "$DST/trunk/" 2>/dev/null || true) ;;
esac
case "$have" in
  *final/model.safetensors*) echo "[e38_lists] stage-1 weights are in $DST/trunk/" ;;
  *) echo "[e38_lists] no stage-1 weights under $DST/trunk/"; exit 3 ;;
esac
export E38_MIX=m21mix_v3 E38_TAG=lists E38_PLAIN_NOTE="with list questions"
exec bash "$(cd "$(dirname "$0")" && pwd)/e38_board.sh" 42 muon:0.01 "$ARM"
