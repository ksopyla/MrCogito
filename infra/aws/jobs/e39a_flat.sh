#!/bin/bash
# E39a (docs/experiments_specs/ahead/E39a_living_flat.md, author 2026-11-26 "Rerun stages 2-3"): stages 2 and 3 of
# E39's living concepts again, with every visible span competing for the opened tokens in training (arm
# living_flat), from E39's stage-1 weights. At 4k rows (stage 1) a reader already read all of a document's concepts,
# so the first level cost nothing there and stage 1 stays as it is. Stages 2-3 on m21mix_v4, as E39 and the E38
# recall run, so the comparison with both is one change each.
# The run's folder is seeded with E39's stage-1 final weights beforehand from the laptop (admin profile: the node's
# role may not copy between S3 folders); the copy here is only a fallback, as in e38_lists.sh.
# Usage (launch.sh JOB words): infra/aws/jobs/e39a_flat.sh
# Results: s3://BUCKET/runs/e39a/e33n_m120t_living_flat_muon0.01_tok8.0e9_seed42/
set -uo pipefail
source /etc/mrcogito.env
SRC="s3://$MC_BUCKET/runs/e39/e33n_m120t_living_muon0.01_tok8.0e9_seed42"
DST="s3://$MC_BUCKET/runs/e39a/e33n_m120t_living_flat_muon0.01_tok8.0e9_seed42"
have=$(aws s3 ls --recursive "$DST/trunk/" 2>/dev/null || true)
case "$have" in
  *final/model.safetensors*) ;;
  *) echo "[e39a] $(date -u +%Y-%m-%dT%H:%M:%SZ) stage-1 weights from $SRC"
     aws s3 sync --only-show-errors --copy-props none "$SRC/trunk/" "$DST/trunk/" --exclude "*/checkpoint-*" || true
     have=$(aws s3 ls --recursive "$DST/trunk/" 2>/dev/null || true) ;;
esac
case "$have" in
  *final/model.safetensors*) echo "[e39a] stage-1 weights are in $DST/trunk/" ;;
  *) echo "[e39a] no stage-1 weights under $DST/trunk/"; exit 3 ;;
esac
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU" in *H100*) export E38_SAVE_EVERY="${E38_SAVE_EVERY:-400}" ;; esac
export E38_EXPERIMENT=E39a E38_MIX=m21mix_v4 E38_PLAIN_NOTE="with recall questions"
exec bash "$(cd "$(dirname "$0")" && pwd)/e38_board.sh" 42 muon:0.01 living_flat
