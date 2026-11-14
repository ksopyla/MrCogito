#!/bin/bash
# mc: no-sync
# E38 data job (CPU node): the constructed far-context rows of m21mix_v1 — multi-hop "libraries" from
# 2WikiMultihopQA's training split and planted-fact lookups in Gutenberg books, at 16k and 64k rows
# (scripts/build_reasoning_rows.py; SmolLM2 tokenizer, anonymous Hub access). The four sets land next to the other
# sources in s3://$MC_BUCKET/data/m21mix_v1/ (shards first, done.json last); infra/aws/jobs/m21_data_build.sh picks
# their done.json up when it writes mix.json, so run this before that job ends, or run that job once more after
# (it skips every finished source and only rewrites mix.json).
#   AWS node:     bash infra/aws/launch.sh m21-reason 2 infra/aws/jobs/m21_reasoning_build.sh BRANCH c7i.4xlarge
#   home server:  M21_DATA=DIR bash infra/aws/jobs/m21_reasoning_build.sh [SCALE]
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
SCALE="${1:-1}"
[ -f /etc/mrcogito.env ] && source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
SUFFIX=""; [ "$SCALE" != 1 ] && SUFFIX="_scale$SCALE"
if [ -n "${M21_DATA:-}" ]; then
  D="$M21_DATA"; S3="${M21_S3:-}"
elif [ -n "${MC_DISK:-}" ] && [ -n "${MC_BUCKET:-}" ]; then
  D="$MC_DISK/data/m21mix_v1$SUFFIX"; S3="s3://$MC_BUCKET/data/m21mix_v1$SUFFIX"
else
  echo "[m21r] set M21_DATA=DIR, or run on a node that has /etc/mrcogito.env"; exit 2
fi
SETS=(multihop_16k multihop_64k lookup_16k lookup_64k)
PY=(uv run --no-project --python 3.12 --with "numpy>=2,<3" --with "pyarrow>=18" --with "tokenizers>=0.21,<0.23"
    --with "huggingface_hub>=0.34,<1" --with "tiktoken==0.12.0" --with "requests>=2.31")
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$D/_logs"
if [ -n "$S3" ]; then   # a relaunch: sets already in S3 are skipped
  for n in "${SETS[@]}"; do
    [ -f "$D/$n.done.json" ] || aws s3 cp --only-show-errors "$S3/$n.done.json" "$D/$n.done.json" 2>/dev/null || true
  done
fi
echo "[m21r] $(now) building ${SETS[*]} at scale $SCALE into $D (S3 ${S3:-none})"
"${PY[@]}" python scripts/build_reasoning_rows.py --out "$D" --scale "$SCALE" 2>&1 | tee -a "$D/_logs/reasoning.log"
rc=${PIPESTATUS[0]}
if [ "$rc" = 0 ] && [ -n "$S3" ]; then
  for n in "${SETS[@]}"; do
    ok=1   # file by file (no folder walk, see m21_data_build.sh)
    for f in "$D/${n}"_train_*.bin "$D/${n}"_val_*.bin; do
      [ -f "$f" ] || continue
      aws s3 cp --only-show-errors "$f" "$S3/$(basename "$f")" || ok=0
    done
    [ "$ok" = 1 ] && aws s3 cp --only-show-errors "$D/$n.done.json" "$S3/$n.done.json" || { echo "[m21r] $(now) !! copy of $n FAILED"; rc=3; }
  done
  aws s3 cp --only-show-errors "$D/_logs/reasoning.log" "$S3/_logs/reasoning.log" || true
fi
du -sh "$D" | sed "s/^/[m21r] folder size: /"
[ "$rc" = 0 ] && echo "[m21r] $(now) reasoning rows done" || echo "[m21r] $(now) !! NOT COMPLETE (exit $rc)"
exit "$rc"
