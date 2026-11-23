#!/bin/bash
# mc: no-sync
# m21mix_v3 (2026-11-23) on a CPU node: m21mix_v2 plus the two list sets — questions whose answer needs several
# far values (scripts/build_reasoning_rows.py --only list_16k,list_64k; SmolLM2 tokenizer, anonymous Hub access).
# Everything v2 has is copied into s3://BUCKET/data/m21mix_v3/ server side, the list sets are built here and copied
# (shards first, done.json last), then m21_data_build.sh writes mix.json for the v3 recipe (it pulls the sources to
# count their rows, builds nothing: every source has its done.json).
#   bash infra/aws/launch.sh m21v3-data 3 infra/aws/jobs/m21v3_data_build.sh BRANCH c7i.8xlarge
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH"
V2="s3://$MC_BUCKET/data/m21mix_v2"; V3="s3://$MC_BUCKET/data/m21mix_v3"; D="$MC_DISK/data/m21mix_v3"
SETS=(list_16k list_64k)
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$D/_logs"
echo "[m21v3] $(now) copying $V2 to $V3 (server side)"
aws s3 sync --only-show-errors "$V2/" "$V3/" --exclude "mix.json" --exclude "_logs/*" --exclude "_raw*" \
  || { echo "[m21v3] $(now) !! the copy failed"; exit 3; }
for n in "${SETS[@]}"; do   # a relaunch: a set already in S3 is skipped
  [ -f "$D/$n.done.json" ] || aws s3 cp --only-show-errors "$V3/$n.done.json" "$D/$n.done.json" 2>/dev/null || true
done
echo "[m21v3] $(now) building ${SETS[*]} into $D"
(
  export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
  uv run --no-project --python 3.12 --with "numpy>=2,<3" --with "pyarrow>=18" --with "tokenizers>=0.21,<0.23" \
    --with "huggingface_hub>=0.34,<1" --with "tiktoken==0.12.0" --with "requests>=2.31" \
    python scripts/build_reasoning_rows.py --out "$D" --only "$(IFS=,; echo "${SETS[*]}")" 2>&1 \
    | tee -a "$D/_logs/list_rows.log"
  exit "${PIPESTATUS[0]}"
) || { echo "[m21v3] $(now) !! the list sets failed"; exit 1; }
for n in "${SETS[@]}"; do
  for f in "$D/${n}"_train_*.bin "$D/${n}"_val_*.bin; do
    [ -f "$f" ] || continue
    aws s3 cp --only-show-errors "$f" "$V3/$(basename "$f")" || { echo "[m21v3] $(now) !! copy of $f failed"; exit 3; }
  done
  aws s3 cp --only-show-errors "$D/$n.done.json" "$V3/$n.done.json" || { echo "[m21v3] $(now) !! copy of $n failed"; exit 3; }
done
aws s3 cp --only-show-errors "$D/_logs/list_rows.log" "$V3/_logs/list_rows.log" || true
echo "[m21v3] $(now) the list sets are in $V3; writing mix.json for the v3 recipe"
source infra/aws/jobs/hf_env.sh
export M21_GATED_OK=1 M21_PULL=1 M21_RECIPE=data/mix_recipes/m21_mix_v3.json M21_DATA="$D" M21_S3="$V3"
exec bash infra/aws/jobs/m21_data_build.sh 1
