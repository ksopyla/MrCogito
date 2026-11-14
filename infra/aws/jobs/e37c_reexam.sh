#!/bin/bash
# E37c follow-up (approved by the author on 2026-11-14: "Yes, run it", to "run the no-cut control and a same-documents
# re-exam"): the two-agent probe of the untuned 100M model and the E37c cut-tuned model on documents drawn from one
# seed, the cells of the E37c exam (4k / 16k / 32k both haystacks, 64k number haystack). The no-cut control model is
# examined with the same seed by its own job (infra/aws/jobs/e37c_split_ft.sh e37a nocut SEED), so all three models
# meet the same documents. 4k–32k of both models run at once on one L40S (the probe's steps are small), 64k one after the other.
#   MC_SPOT=1 bash infra/aws/launch.sh e37c-reexam 4 "infra/aws/jobs/e37c_reexam.sh e37a 2026" BRANCH g6e.xlarge
# Arguments: ARM (e37a) PROBE_SEED. Results: s3://BUCKET/runs/e37c/e37c_reexam_<ARM>_seed<PROBE_SEED>/ (read-only
# use of the E37 and E37c run folders).
set -uo pipefail
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARM="${1:?arm}"; PSEED="${2:?probe seed}"
RUN="e37c_reexam_${ARM}_seed${PSEED}"
S3R="s3://$MC_BUCKET/runs/e37c/$RUN"; R="$MC_DISK/runs/$RUN"; D="$MC_DISK/data"
mkdir -p "$R" "$D"
log() { echo "[e37c-reexam $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$R/events.log"; }
sync_up() { aws s3 sync --only-show-errors "$R/" "$S3R/" || true; }
aws s3 sync --only-show-errors "$S3R/" "$R/" || true
( while sleep 300; do sync_up; done ) &
SYNC=$!
log "start $RUN on $MC_NAME ${MC_IID:-} ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)); code $(git log --oneline -1 | cut -c1-70)"

pull() {  # NAME S3_PREFIX_TO_SEARCH
  local key
  key=$(aws s3 ls --recursive "$2" | grep "/final/config.json" | awk '{print $4}' | head -1)
  [ -n "$key" ] || { log "$1: no final checkpoint under $2"; return 1; }
  aws s3 sync --only-show-errors "s3://$MC_BUCKET/$(dirname "$key")/" "$MC_DISK/ck/$1/" --exclude "*.bin" --exclude "optimizer*"
  log "$1: $(dirname "$key") ($(du -sh "$MC_DISK/ck/$1" | cut -f1))"
}
pull base "s3://$MC_BUCKET/runs/e37/e33n_m100_${ARM}_tok3.0e9_seed42/decay/" || { kill $SYNC; sync_up; exit 1; }
pull cut "s3://$MC_BUCKET/runs/e37c/e37c_m100_${ARM}_seed1/split/" || { kill $SYNC; sync_up; exit 1; }
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/tokenizer_gpt2/" "$D/tokenizer_gpt2/"

probe() {  # NAME CHECKPOINT TASKS LENGTHS LIMIT TAILS
  [ -f "$R/split_$1.done" ] && return 0
  uv run --with wonderwords python scripts/a2a_split_probe.py --checkpoint "$2" --tokenizer "$D/tokenizer_gpt2" \
    --tasks "$3" --lengths "$4" --limit "$5" --limit_long 20 --bits 8,4 --bits_docs 15 --tails "$6" --new 12 \
    --seed "$PSEED" --out "$R/split_$1.json" > "$R/split_$1.log" 2>&1
  rc=$?
  [ "$rc" = 0 ] && touch "$R/split_$1.done"
  log "two-agent probe ($1): exit $rc, $(grep -c '^{"task"' "$R/split_$1.log") cells"
}
short() { probe "$1" "$MC_DISK/ck/$1" niah_single_1,niah_single_2 4096,16384,32768 30 16; }
short base & P1=$!
short cut & P2=$!
wait $P1 $P2
sync_up
for m in base cut; do   # one at a time: two 64k documents at once may not fit the card
  probe "${m}_64k" "$MC_DISK/ck/$m" niah_single_1 65536 20 16,32,64
done
log "done"
kill $SYNC 2>/dev/null; sync_up
