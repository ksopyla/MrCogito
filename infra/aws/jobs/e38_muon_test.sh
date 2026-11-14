#!/bin/bash
# E38 Muon test (author 2026-11-14: "Ok start them" … "My past experience with Muon was really good"): concept memory
# at 100M, the web stage of the m21mix_v1 data for 1.05B tokens (2,000 steps of 128 rows of 4,096) with Muon on the
# matrices, everything else as the running E38 concept-memory run (seed 42, warm-up 152 steps, constant rate after,
# --fast, the same evaluation rows every 500 steps). Its AdamW twin is that run's own trunk log at the same steps, so
# the baseline costs nothing. A one-GPU node keeps 128 rows per step by accumulation (data order differs from the
# four-GPU run; same rows on average). Checkpoints go to S3, so a relaunch after a spot interruption resumes.
# Usage (launch.sh JOB words): infra/aws/jobs/e38_muon_test.sh LR [TOKENS]
# Results: s3://BUCKET/runs/e38/e38_muon_e37a_lr<LR>_tok<TOKENS>_seed42/{trunk.log,events.log}
set -uo pipefail
LR="${1:?Muon learning rate}"; TT="${2:-1.05e9}"
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_PROJECT=MrCogito EXPERIMENT_ID=E38
cd "$(dirname "$0")/../../.." || exit 1
source infra/aws/jobs/wandb_env.sh
D="$MC_DISK/data"; RUN="e38_muon_e37a_lr${LR}_tok${TT}_seed42"; R="$MC_DISK/runs/$RUN"; S3R="s3://$MC_BUCKET/runs/e38/$RUN"
export WANDB_DIR="$R/wandb"; mkdir -p "$R" "$WANDB_DIR"
log() { echo "[e38_muon $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$R/events.log"; }
up() { aws s3 sync --only-show-errors "$R/" "$S3R/" --exclude "wandb/*" || true; }
aws s3 sync --only-show-errors "$S3R/" "$R/" || true
G=$(nvidia-smi -L | wc -l); MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
M=4; [ "${MEM:-0}" -ge 70000 ] && M=8
log "start $RUN on ${MC_NAME:-?} ($G GPU, $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)), code $(git log --oneline -1 | cut -c1-60)"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/m21mix_v1/" "$D/m21mix_v1/"
CMD=$(uv run python scripts/run_e33n.py cmd --arm e37a --size m100 --phase trunk --data "$D" --out "$R" --tokens "$TT" \
      --warmup 152 --micro "$M" --gpus "$G" --eval_every 500 --fast --optimizer muon --lr "$LR" --mix "$D/m21mix_v1" \
      --seed 42 --wandb)
echo "$CMD" > "$R/trunk.cmd"
ck=$(ls -d "$R"/trunk/*/checkpoint-* 2>/dev/null | awk -F'checkpoint-' '{print $NF" "$0}' | sort -rn | head -1 | cut -d' ' -f2)
[ -n "$ck" ] && [ -f "$ck/trainer_state.json" ] && log "resumes from ${ck##*/}" || ck=""
( while sleep 900; do up; done ) & SYNC=$!
( export RUN_ID_SUFFIX="e37a_muon_lr${LR}"; eval "$CMD${ck:+ --resume_from_checkpoint $ck}" ) >> "$R/trunk.log" 2>&1
rc=$?
kill "$SYNC" 2>/dev/null
log "trunk ended with code $rc"
[ "$rc" = 0 ] && touch "$R/DONE"
up
