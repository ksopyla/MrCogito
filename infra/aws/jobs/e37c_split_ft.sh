#!/bin/bash
# E37c — two-agent split training (docs/experiments_specs/done_failed/E37c_two_agent_split_training.md; approved by the
# author on 2026-11-14: "Yes, spec it", to "spec and run the short split fine-tune (about $15)").
# One 100M model's 16k-stage final weights are fine-tuned on LongDocs-lite rows of 16,384 that are cut at random
# points (a local layer reads its own side of a cut only, the reading layer the whole document), then examined as two
# agents with the eval-only probe of 2026-11-13 (same settings, same documents).
#   1. pre-flight: the cut loss and the long-document loss of the untuned model (kill rule K2, the no-harm reference)
#   2. training:   scripts/run_e33n.py --phase split (0.2B tokens, 32 rows a step, 6.4e-5 -> 6.4e-6), checkpoints to S3
#   3. exams:      cut loss, two-agent probe at 4k / 16k / 32k (both haystacks) and 64k (number haystack), long-document loss
# A launch.sh job for one L40S (more GPUs: the same rows per step, data parallel):
#   MC_SPOT=1 bash infra/aws/launch.sh e37c-split 9 "infra/aws/jobs/e37c_split_ft.sh e37a" BRANCH g6e.xlarge
# Arguments: ARM (e37a | lsa | dense: the m100 full-budget runs). Environment: E37C_SEED (1), E37C_TOKENS, E37C_LR,
#   E37C_MICRO (1: an L40S holds one 16k row), E37C_CUT ("32 4096"), E37C_TAG (a suffix of the run name).
# Results: s3://BUCKET/runs/e37c/<run>/ — its own folder; the shared E37 run folder is only read. A relaunched spot
# node pulls the folder back first: finished steps are skipped and training resumes from its newest checkpoint.
set -uo pipefail
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_PROJECT=MrCogito WANDB_MODE=offline EXPERIMENT_ID=E37c
ARM="${1:?arm}"; SIZE=m100; SEED="${E37C_SEED:-1}"
read -r CUT_MIN CUT_MAX <<< "${E37C_CUT:-32 4096}"
RUN="e37c_${SIZE}_${ARM}_seed${SEED}${E37C_TAG:+_$E37C_TAG}"
BASE="s3://$MC_BUCKET/runs/e37/e33n_m100_${ARM}_tok3.0e9_seed42"
S3R="s3://$MC_BUCKET/runs/e37c/$RUN"; R="$MC_DISK/runs/$RUN"; D="$MC_DISK/data"; CK0="$MC_DISK/base/$ARM"
export WANDB_DIR="$R/wandb" RUN_ID_SUFFIX="${ARM}_${SIZE}_split"
mkdir -p "$R" "$WANDB_DIR" "$D" "$CK0"
G=$(nvidia-smi -L | wc -l)
log() { echo "[e37c $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$R/events.log"; }
sync_up() { aws s3 sync --only-show-errors "$R/" "$S3R/" || true; }
aws s3 sync --only-show-errors "$S3R/" "$R/" || true   # a relaunched spot node: take the run folder back first
( while sleep 600; do sync_up; done ) &
SYNC=$!
finish() { kill "$SYNC" 2>/dev/null; sync_up; }
log "start $RUN on $MC_NAME ${MC_IID:-} ($G x $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)); code $(git log --oneline -1 | cut -c1-70)"

key=$(aws s3 ls --recursive "$BASE/decay/" | grep "/final/config.json" | awk '{print $4}' | head -1)
[ -n "$key" ] || { log "$ARM: no final checkpoint under $BASE/decay/"; finish; exit 1; }
aws s3 sync --only-show-errors "s3://$MC_BUCKET/$(dirname "$key")/" "$CK0/" --exclude "*.bin" --exclude "optimizer*"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/tokenizer_gpt2/" "$D/tokenizer_gpt2/"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/longdocs_lite_gpt2/" "$D/longdocs_lite_gpt2/"
log "base: $(dirname "$key") ($(du -sh "$CK0" | cut -f1)); data: $(ls "$D"/longdocs_lite_gpt2/*_train_*.bin | wc -l) LongDocs-lite shards"

cut_loss() {   # NAME CHECKPOINT
  [ -f "$R/cut_loss_$1.json" ] && return 0
  uv run python scripts/a2a_cut_loss.py --checkpoint "$2" --data "$D/longdocs_lite_gpt2" --train_gaps "$CUT_MIN,$CUT_MAX" \
    --out "$R/cut_loss_$1.json" > "$R/cut_loss_$1.log" 2>&1
  log "cut loss ($1): exit $? $(tail -2 "$R/cut_loss_$1.log" | head -1 | cut -c1-260)"
}
longdoc() {    # NAME CHECKPOINT
  [ -f "$R/eval_longdoc_$1.json" ] && return 0
  uv run python evaluation/long_doc_loss.py --checkpoint "$2" --data "$D/longdocs_lite_gpt2" \
    --lengths 4096,16384,65536 --rows 8 --out "$R/eval_longdoc_$1.json" > "$R/eval_longdoc_$1.log" 2>&1
  log "long-document loss ($1): exit $?"
}
probe() {      # NAME CHECKPOINT TASKS LENGTHS LIMIT TAILS
  [ -f "$R/split_$1.done" ] && return 0
  uv run --with wonderwords python scripts/a2a_split_probe.py --checkpoint "$2" --tokenizer "$D/tokenizer_gpt2" \
    --tasks "$3" --lengths "$4" --limit "$5" --limit_long 20 --bits 8,4 --bits_docs 15 --tails "$6" --new 12 \
    --out "$R/split_$1.json" > "$R/split_$1.log" 2>&1
  rc=$?
  [ "$rc" = 0 ] && touch "$R/split_$1.done"
  log "two-agent probe ($1): exit $rc, $(grep -c '^{"task"' "$R/split_$1.log") cells"
  sync_up
}

# 1. pre-flight on the untuned model
cut_loss base "$CK0"
longdoc base "$CK0"
sync_up

# 2. training on cut rows (resumes from this run's newest complete checkpoint)
FT=$(ls -d "$R"/split/*/final 2>/dev/null | tail -1)
if [ -z "$FT" ]; then
  CMD=$(uv run python scripts/run_e33n.py cmd --arm "$ARM" --size "$SIZE" --phase split --data "$D" --out "$R" \
        --init "$CK0" --micro "${E37C_MICRO:-1}" --eval_every 100 --wandb --seed "$SEED" --gpus "$G" \
        --cut_min "$CUT_MIN" --cut_max "$CUT_MAX" ${E37C_TOKENS:+--tokens $E37C_TOKENS} ${E37C_LR:+--lr $E37C_LR})
  echo "$CMD" > "$R/split.cmd"
  ck=""
  for d in $(ls -d "$R"/split/*/checkpoint-* 2>/dev/null | awk -F'checkpoint-' '{print $NF" "$0}' | sort -rn | cut -d' ' -f2); do
    [ -f "$d/trainer_state.json" ] && { ck="$d"; break; }
  done
  log "training start${ck:+ (resumes from ${ck##*/})}"
  eval "$CMD${ck:+ --resume_from_checkpoint $ck}" >> "$R/split.log" 2>&1
  rc=$?
  FT=$(ls -d "$R"/split/*/final 2>/dev/null | tail -1)
  [ "$rc" = 0 ] && [ -n "$FT" ] || { log "training FAILED (exit $rc)"; tail -40 "$R/split.log"; finish; exit 1; }
fi
log "training done: $FT"
sync_up

# 3. exams of the tuned model, the pass-mark cells first
cut_loss tuned "$FT"
probe tuned "$FT" niah_single_1,niah_single_2 4096,16384,32768 30 16
longdoc tuned "$FT"
probe tuned_64k "$FT" niah_single_1 65536 20 16,32,64
log "done"
finish
