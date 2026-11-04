#!/bin/bash
# E33n, one arm × size end to end on one H100 (spec docs/experiments_specs/ahead/E33n_lsa_testbed_topk_notes.md):
#   1. trunk  — FineWeb-Edu GPT-2 shards, rows of 4,096, warmup then constant learning rate
#   2. decay  — from the trunk's final weights, fresh optimizer, LongDocs-lite rows of 16,384, 1−sqrt decay
#   3. evals  — long-document loss by position at 4k–128k (notebook on / off), RULER (13 tasks, 4k–32k), core benchmarks
# Everything lands in s3://BUCKET/runs/e33n/<run>/ (weights of the two final models, logs, W&B offline runs, results).
# Usage (as a launch.sh job): "infra/aws/jobs/e33n_run.sh ARM SIZE [LR] [TRUNK_TOKENS]"
set -uo pipefail
ARM="$1"; SIZE="$2"; LR="${3:-}"; TT="${4:-}"
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline WANDB_PROJECT=MrCogito EXPERIMENT_ID=E33n
D="$MC_DISK/data"
RUN="e33n_${SIZE}_${ARM}${LR:+_lr$LR}${TT:+_tok$TT}"
R="$MC_DISK/runs/$RUN"; S3R="s3://$MC_BUCKET/runs/e33n/$RUN"
export WANDB_DIR="$R/wandb"
mkdir -p "$D/fineweb_edu_gpt2" "$R" "$WANDB_DIR"
log() { echo "[e33n $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$R/events.log"; }
sync_up() {
  aws s3 sync --only-show-errors "$R/" "$S3R/" --exclude "*/checkpoint-*" --exclude "*optimizer.pt" \
    --exclude "*rng_state*" --exclude "*.pth" || true
}
( while sleep 900; do sync_up; done ) &
SYNC=$!

log "start $RUN on $MC_IID"
NSH=$(uv run python -c "import sys; sys.path.insert(0, '.'); from scripts.run_e33n import SIZES; print(int(float('${TT:-0}') or SIZES['$SIZE']['trunk']) // 100_000_000 + 2)")
aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/finewebedu_val_000000.bin" "$D/fineweb_edu_gpt2/"
for i in $(seq 1 "$NSH"); do
  f=$(printf "finewebedu_train_%06d.bin" "$i")
  aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/$f" "$D/fineweb_edu_gpt2/" || true
done
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/tokenizer_gpt2/" "$D/tokenizer_gpt2/"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/longdocs_lite_gpt2/" "$D/longdocs_lite_gpt2/"
log "data ready: $NSH FineWeb shards, LongDocs-lite"

CMD=$(uv run python scripts/run_e33n.py cmd --arm "$ARM" --size "$SIZE" --phase trunk --data "$D" --out "$R" \
      ${LR:+--lr $LR} ${TT:+--tokens $TT} --eval_every 500 --wandb)
echo "$CMD" > "$R/trunk.cmd"
log "trunk start"
( eval "$CMD" ) > "$R/trunk.log" 2>&1 || { log "trunk FAILED"; tail -40 "$R/trunk.log"; sync_up; kill $SYNC; exit 1; }
TRUNK=$(ls -d "$R"/trunk/*/final | tail -1)
log "trunk done: $TRUNK"
sync_up

CMD=$(uv run python scripts/run_e33n.py cmd --arm "$ARM" --size "$SIZE" --phase decay --data "$D" --out "$R" \
      ${LR:+--lr $LR} --init "$TRUNK" --eval_every 200 --wandb)
echo "$CMD" > "$R/decay.cmd"
log "decay start"
( eval "$CMD" ) > "$R/decay.log" 2>&1 || { log "decay FAILED"; tail -40 "$R/decay.log"; sync_up; kill $SYNC; exit 1; }
DECAY=$(ls -d "$R"/decay/*/final | tail -1)
log "decay done: $DECAY"
sync_up

log "eval long-document loss"
uv run python evaluation/long_doc_loss.py --checkpoint "$DECAY" --data "$D/longdocs_lite_gpt2" \
  --lengths 4096,16384,32768,65536,131072 --rows 8 --notes_off --out "$R/eval_longdoc.json" > "$R/eval_longdoc.log" 2>&1 \
  || log "long-document eval FAILED (see eval_longdoc.log)"
sync_up
log "eval RULER"
uv run --with wonderwords python evaluation/run_lsa_suite.py --checkpoint "$DECAY" --tokenizer "$D/tokenizer_gpt2" \
  --suite ruler --lengths 4096,8192,16384,32768 --limit 100 --batch 8 --out "$R/eval_ruler.json" > "$R/eval_ruler.log" 2>&1 \
  || log "RULER FAILED (see eval_ruler.log)"
sync_up
log "eval core"
uv run python evaluation/run_lsa_suite.py --checkpoint "$DECAY" --tokenizer "$D/tokenizer_gpt2" --suite core \
  --limit 0 --batch 32 --out "$R/eval_core.json" > "$R/eval_core.log" 2>&1 || log "core FAILED (see eval_core.log)"
kill $SYNC 2>/dev/null
log "done"
sync_up
