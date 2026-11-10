#!/bin/bash
# E33n, one arm × size end to end on one GPU (spec docs/experiments_specs/done_failed/E33n_lsa_testbed_topk_notes.md):
#   1. trunk  — FineWeb-Edu GPT-2 shards, rows of 4,096, warmup then constant learning rate
#   2. decay  — from the trunk's final weights, fresh optimizer, LongDocs-lite rows of 16,384, 1−sqrt decay
#   3. evals  — long-document loss by position at 4k–128k (notebook on / off), RULER (13 tasks, 4k–32k), core benchmarks
#
# AWS node (a launch.sh job; data from S3, results to s3://BUCKET/runs/e33n/<run>/, W&B offline):
#     infra/aws/jobs/e33n_run.sh ARM SIZE [LR] [TRUNK_TOKENS]
# Home server (Odra / Polonez, in tmux; data from data_build.sh, W&B as the server is set up):
#     E33N_DATA=DIR E33N_RUNS=DIR CUDA_VISIBLE_DEVICES=0 bash infra/aws/jobs/e33n_run.sh ARM SIZE [LR] [TRUNK_TOKENS]
#   E33N_BURST_H=5.5 E33N_COOL_MIN=20: train in bursts of at most 5.5 h with a 20 min cooldown between them
#   (Polonez heat rule); each burst ends with a checkpoint and the next resumes from it.
# E33N_MICRO_TRUNK / E33N_MICRO_DECAY: micro-batch per GPU (default: the driver's, sized for an 80 GB H100).
# E33N_CORE_BATCH (default 32) / E33N_RULER_BATCH (default 8): eval batch; on a 24 GB GPU use core 8 (batch 32 ×
#   2,048 tokens × the 50k vocabulary needs a 12 GB log-softmax).
# E33N_CORE_BACKEND (default sdpa): the core benchmarks' many short, ragged batches made flex attention's compiler
#   fail on Polonez (ptxas segfault, then "PassManager::run failed", 2026-11-05); SDPA needs no compile.
# E33N_RULER_LENGTHS (default 4096,8192,16384,32768): RULER lengths, 100 documents each (uncached decoding: the cost
#   grows with length; 4096 is LSA's headline column).
# A rerun skips the phases that already have a final model.
# Smoke tests: E33N_CMD_EXTRA="--local" (laptop trainer settings), E33N_SKIP_EVAL=1 (stop after the decay).
# E33p: E33N_EXPERIMENT=E33p (W&B / run id, S3 prefix runs/e33p/), E33N_FLOW=1 (instrumentation + weight
#   snapshots, plan §8). E33N_SEED=N: trainer seed N (default 42), run folder gets _seedN.
# E33N_GPUS=N: one model on N GPUs (data parallel, torchrun; same rows per optimizer step), and the three evals run
#   side by side on separate GPUs. On a home server pair it with CUDA_VISIBLE_DEVICES=0,1,2.
# E33N_CKPT_S3=1 (AWS spot nodes): checkpoints go to S3 with the rest, and a relaunched node first pulls the run
#   folder back, so an interrupted run resumes from its newest checkpoint instead of starting over.
set -uo pipefail
ARM="$1"; SIZE="$2"; LR="${3:-}"; TT="${4:-}"
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_PROJECT=MrCogito EXPERIMENT_ID=${E33N_EXPERIMENT:-E33n}
if [ -f /etc/mrcogito.env ]; then
  source /etc/mrcogito.env
  D="$MC_DISK/data"; RUNS="$MC_DISK/runs"; export WANDB_MODE=offline
else
  D="${E33N_DATA:?set E33N_DATA}"; RUNS="${E33N_RUNS:?set E33N_RUNS}"; MC_BUCKET=""
fi
RUN="e33n_${SIZE}_${ARM}${LR:+_lr$LR}${TT:+_tok$TT}${E33N_SEED:+_seed$E33N_SEED}"
EXP=$(echo "$EXPERIMENT_ID" | tr "[:upper:]" "[:lower:]")
R="$RUNS/$RUN"; S3R="${MC_BUCKET:+s3://$MC_BUCKET/runs/$EXP/$RUN}"
export WANDB_DIR="$R/wandb"
mkdir -p "$R" "$WANDB_DIR"
log() { echo "[e33n $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$R/events.log"; }
sync_up() {
  [ -n "$S3R" ] || return 0
  if [ -n "${E33N_CKPT_S3:-}" ]; then
    aws s3 sync --only-show-errors "$R/" "$S3R/" || true
  else
    aws s3 sync --only-show-errors "$R/" "$S3R/" --exclude "*/checkpoint-*" --exclude "*optimizer.pt" \
      --exclude "*rng_state*" --exclude "*.pth" || true
  fi
}
if [ -n "$S3R" ] && [ -n "${E33N_CKPT_S3:-}" ]; then   # a relaunched spot node: take the run folder back first
  aws s3 sync --only-show-errors "$S3R/" "$R/" || true
fi
G="${E33N_GPUS:-1}"
IFS=, read -ra GL <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((G - 1)))}"   # the GPUs this run may use
SYNC=""
if [ -n "$S3R" ]; then ( while sleep 900; do sync_up; done ) & SYNC=$!; fi
finish() { [ -n "$SYNC" ] && kill "$SYNC" 2>/dev/null; sync_up; }

# newest complete checkpoint of a phase (trainer_state.json is written last; a half-copied one from S3 has none)
newest_ck() {
  local d
  for d in $(ls -d "$R/$1"/*/checkpoint-* 2>/dev/null | awk -F'checkpoint-' '{print $NF" "$0}' | sort -rn | cut -d' ' -f2); do
    [ -f "$d/trainer_state.json" ] && { echo "$d"; return; }
  done
}
# the last invocation stopped on its wall-clock budget, even if exit code 3 was lost at shutdown
# (Polonez dense arm, 2026-11-06 15:00 UTC: budget line logged and checkpoint saved, yet rc != 3)
budget_hit() { awk '/STARTING TRAINING/{f=0} /Wall-clock budget of/{f=1} END{exit !f}' "$R/$1.log"; }

# phase NAME CMD: run a training phase; with E33N_BURST_H, in bursts that resume from the newest checkpoint.
# A relaunch after a failure also continues from this phase's newest checkpoint instead of starting over.
phase() {
  local name=$1 cmd=$2 ck rc
  ck=$(newest_ck "$name")
  [ -n "$ck" ] && log "$name restarts from ${ck##*/}"
  while true; do
    ( export RUN_ID_SUFFIX="${ARM}_${SIZE}_${name}"   # arms launched in the same second get distinct W&B runs
      eval "$cmd${E33N_BURST_H:+ --max_train_hours $E33N_BURST_H}${ck:+ --resume_from_checkpoint $ck}" ) \
      >> "$R/$name.log" 2>&1
    rc=$?
    [ "$rc" = 0 ] && return 0
    [ "$rc" = 3 ] || budget_hit "$name" || return "$rc"
    ck=$(newest_ck "$name")
    log "$name burst ended at ${ck##*/}; cooldown ${E33N_COOL_MIN:-20} min"
    sync_up
    sleep $(( ${E33N_COOL_MIN:-20} * 60 ))
    log "$name resumes from ${ck##*/}"
  done
}

log "start $RUN on ${MC_IID:-$(hostname)} GPU ${CUDA_VISIBLE_DEVICES:-all} ($G per model)"
if [ -n "$MC_BUCKET" ]; then
  mkdir -p "$D/fineweb_edu_gpt2"
  NSH=$(uv run python -c "import sys; sys.path.insert(0, '.'); from scripts.run_e33n import SIZES; print(int(float('${TT:-0}') or SIZES['$SIZE']['trunk']) // 100_000_000 + 2)")
  aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/finewebedu_val_000000.bin" "$D/fineweb_edu_gpt2/"
  for i in $(seq 1 "$NSH"); do
    f=$(printf "finewebedu_train_%06d.bin" "$i")
    aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/$f" "$D/fineweb_edu_gpt2/" || true
  done
  aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/tokenizer_gpt2/" "$D/tokenizer_gpt2/"
  aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/longdocs_lite_gpt2/" "$D/longdocs_lite_gpt2/"
fi
log "data: $(ls "$D"/fineweb_edu_gpt2/finewebedu_train_*.bin | wc -l) FineWeb shards, $(ls "$D"/longdocs_lite_gpt2/*_train_*.bin | wc -l) LongDocs-lite shards"

TRUNK=$(ls -d "$R"/trunk/*/final 2>/dev/null | tail -1)
if [ -z "$TRUNK" ]; then
  CMD=$(uv run python scripts/run_e33n.py cmd --arm "$ARM" --size "$SIZE" --phase trunk --data "$D" --out "$R" \
        ${LR:+--lr $LR} ${TT:+--tokens $TT} ${E33N_MICRO_TRUNK:+--micro $E33N_MICRO_TRUNK} --eval_every 500 --wandb ${E33N_FLOW:+--flow} ${E33N_SEED:+--seed $E33N_SEED} --gpus "$G" ${E33N_CMD_EXTRA:-})
  echo "$CMD" > "$R/trunk.cmd"
  log "trunk start"
  phase trunk "$CMD" || { log "trunk FAILED"; tail -40 "$R/trunk.log"; finish; exit 1; }
  TRUNK=$(ls -d "$R"/trunk/*/final | tail -1)
fi
log "trunk done: $TRUNK"
sync_up

DECAY=$(ls -d "$R"/decay/*/final 2>/dev/null | tail -1)
if [ -z "$DECAY" ]; then
  CMD=$(uv run python scripts/run_e33n.py cmd --arm "$ARM" --size "$SIZE" --phase decay --data "$D" --out "$R" \
        ${LR:+--lr $LR} ${E33N_MICRO_DECAY:+--micro $E33N_MICRO_DECAY} --init "$TRUNK" --eval_every 200 --wandb ${E33N_FLOW:+--flow} ${E33N_SEED:+--seed $E33N_SEED} --gpus "$G" ${E33N_CMD_EXTRA:-})
  echo "$CMD" > "$R/decay.cmd"
  log "decay start"
  phase decay "$CMD" || { log "decay FAILED"; tail -40 "$R/decay.log"; finish; exit 1; }
  DECAY=$(ls -d "$R"/decay/*/final | tail -1)
fi
log "decay done: $DECAY"
sync_up
[ -n "${E33N_SKIP_EVAL:-}" ] && { log "evals skipped"; finish; exit 0; }

ev_longdoc() {
  log "eval long-document loss"
  uv run python evaluation/long_doc_loss.py --checkpoint "$DECAY" --data "$D/longdocs_lite_gpt2" \
    --lengths 4096,16384,32768,65536,131072 --rows 8 --notes_off --open_off --gist_off --open_reuse --out "$R/eval_longdoc.json" > "$R/eval_longdoc.log" 2>&1 \
    || log "long-document eval FAILED (see eval_longdoc.log)"
  sync_up
}
ev_ruler() {
  log "eval RULER"
  uv run --with wonderwords python evaluation/run_lsa_suite.py --checkpoint "$DECAY" --tokenizer "$D/tokenizer_gpt2" \
    --suite ruler --lengths "${E33N_RULER_LENGTHS:-4096,8192,16384,32768}" --limit 100 --batch "${E33N_RULER_BATCH:-8}" --out "$R/eval_ruler.json" > "$R/eval_ruler.log" 2>&1 \
    || log "RULER FAILED (see eval_ruler.log)"
  sync_up
}
ev_core() {
  log "eval core"
  uv run python evaluation/run_lsa_suite.py --checkpoint "$DECAY" --tokenizer "$D/tokenizer_gpt2" --suite core \
    --limit 0 --batch "${E33N_CORE_BATCH:-32}" --backend "${E33N_CORE_BACKEND:-sdpa}" --out "$R/eval_core.json" > "$R/eval_core.log" 2>&1 || log "core FAILED (see eval_core.log)"
}
n=${#GL[@]}
if [ "$G" -gt 1 ] && [ "$n" -gt 2 ]; then   # one eval per GPU, side by side
  ( export CUDA_VISIBLE_DEVICES="${GL[0]}"; ev_longdoc ) & P1=$!
  ( export CUDA_VISIBLE_DEVICES="${GL[1]}"; ev_ruler ) & P2=$!
  ( export CUDA_VISIBLE_DEVICES="${GL[2]}"; ev_core ) & P3=$!
  wait "$P1" "$P2" "$P3"   # not a bare wait: the S3 sync loop runs in the background until finish
elif [ "$G" -gt 1 ] && [ "$n" -eq 2 ]; then   # the long-document loss keeps a card to itself (128k: ~19 GB of 24)
  ( export CUDA_VISIBLE_DEVICES="${GL[0]}"; ev_longdoc ) & P1=$!
  ( export CUDA_VISIBLE_DEVICES="${GL[1]}"; ev_ruler; ev_core ) & P2=$!
  wait "$P1" "$P2"
else
  ev_longdoc; ev_ruler; ev_core
fi
log "done"
finish
