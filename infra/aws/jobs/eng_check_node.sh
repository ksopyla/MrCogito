#!/bin/bash
# Engineering check on one L40S (g6e.xlarge), 2026-11-14 (author: "implement the engineering improvements ... Test it
# properly, and compare to past runs"): the training speed of the 100M models with the old settings and with the new
# ones, on the card type every 100M run to date used, and the read-once / answer-many timing next to the cached and
# uncached decoding of 2026-11-13.
#   MC_SPOT=1 bash infra/aws/launch.sh eng-check 3 "infra/aws/jobs/eng_check_node.sh" eng-100m g6e.xlarge
#   bash infra/aws/launch.sh eng-check-h100 2 "infra/aws/jobs/eng_check_node.sh h100" eng-100m p5.4xlarge   (on demand)
# Part 1, training (real trainer, real shards, 8 rows of 4k / 2 rows of 16k per optimizer step, 60 steps each):
#   old   = the settings of every run to 2026-11-14 (chunked loss; micro-batch 4 at 4k and 1 at 16k per card)
#   fast  = scripts/run_e33n.py --fast (fused loss, compiled concept grouping) at several micro-batches
# Part 2, decoding: evaluation/bench_cached_decoding.py --questions 5 on the 100M checkpoints (seed 42, 3.0B).
# Results: s3://BUCKET/runs/eng_check/<node>/{speed_summary.tsv, *.json, *.log}, uploaded as they finish.
set -uo pipefail
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# cached decoding compiles one attention graph per prompt length per call; past dynamo's limit it falls back to
# the eager kernel, which ran out of memory at 128k (e37-needles 806b655, 2026-11-14)
export MC_DYNAMO_CACHE_LIMIT=${MC_DYNAMO_CACHE_LIMIT:-2048}
D="$MC_DISK/data"; W="$MC_DISK/eng_check"; OUT="s3://$MC_BUCKET/runs/eng_check/$MC_NAME"
HERE="$(cd "$(dirname "$0")" && pwd)"
ARMS=(${ENG_ARMS:-e37a lsa})
# first JOB word: the card profile. l40s (default, 48 GB): micro-batches up to 10 / 3; h100 (80 GB): up to 20 / 6,
# training speed only (the decoding was timed on the L40S)
PROFILE="${1:-l40s}"
if [ "$PROFILE" = h100 ]; then TM=(8 12 16 20); DM=(2 4 6); ENG_SKIP_DECODE=1; else TM=(4 6 8 10); DM=(1 2 3); fi
mkdir -p "$D/fineweb_edu_gpt2" "$W"
log() { echo "[eng_check $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$W/node.log"; }
up() { aws s3 sync --only-show-errors "$W/" "$OUT/" --exclude "*/checkpoint-*" --exclude "*.safetensors" --exclude "*.pt" --exclude "ckpt/*" --exclude "*.gpu" || true; }
log "start on $MC_NAME ${MC_IID:-} ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)); code $(git log --oneline -1 | cut -c1-70)"
for f in finewebedu_val_000000.bin finewebedu_train_000001.bin finewebedu_train_000002.bin; do
  aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/$f" "$D/fineweb_edu_gpt2/"
done
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/tokenizer_gpt2/" "$D/tokenizer_gpt2/"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/longdocs_lite_gpt2/" "$D/longdocs_lite_gpt2/"

# one(arm, phase, micro, rows, label, extra flags): 60 optimizer steps through speed_check.sh
one() {
  local arm=$1 phase=$2 micro=$3 rows=$4 label=$5 extra=$6
  local seq=$([ "$phase" = trunk ] && echo 4096 || echo 16384)
  SPEED_LABEL="$label" SPEED_EXTRA="--rows $rows $extra" \
    bash "$HERE/speed_check.sh" "$D" "$W/speed" "m100:$arm:$phase:$(( rows * seq * 60 )):$micro" > "$W/speed_${arm}_${phase}_$label.out" 2>&1
  rm -rf "$W/speed/m100_${arm}_$label" 2>/dev/null
  log "$arm $phase $label: $(tail -1 "$W/speed/summary.tsv")"
  cp "$W/speed/summary.tsv" "$W/speed_summary.tsv"; up
}
if [ -z "${ENG_SKIP_SPEED:-}" ]; then
  for arm in "${ARMS[@]}"; do
    one "$arm" trunk 4 8 old_m4 ""
    one "$arm" trunk 4 8 old_m4_flow "--flow"       # as the E37 runs were launched (E33p instrumentation on)
    for m in "${TM[@]}"; do one "$arm" trunk "$m" $(( 2 * m )) "fast_m$m" "--fast"; done
    one "$arm" decay 1 2 old_m1 ""
    for m in "${DM[@]}"; do one "$arm" decay "$m" $(( 2 * m )) "fast_m$m" "--fast"; done
  done
fi

if [ -z "${ENG_SKIP_DECODE:-}" ]; then
  B="s3://$MC_BUCKET/runs/e37"
  for arm in ${ENG_DECODE_ARMS:-e37a dense lsa}; do
    run="e33n_m100_${arm}_tok3.0e9_seed42"
    key=$(aws s3 ls --recursive "$B/$run/decay/" | grep "/final/config.json" | awk '{print $4}' | head -1)
    [ -n "$key" ] || { log "$arm: no final checkpoint under $B/$run/decay/"; continue; }
    ck="$W/ckpt/$arm"
    aws s3 sync --only-show-errors "s3://$MC_BUCKET/$(dirname "$key")/" "$ck/" --exclude "*.bin" --exclude "optimizer*"
    uv run python evaluation/bench_cached_decoding.py --checkpoint "$ck" --tokenizer "$D/tokenizer_gpt2" \
      --lengths "${ENG_LENGTHS:-16384,65536,131072}" --new 32 --skip_uncached_above 16384 --questions 5 \
      --out "$W/decode_$arm.json" > "$W/decode_$arm.log" 2>&1
    log "$arm decode: exit $? $(grep -c '"length"' "$W/decode_$arm.log") lengths"
    up
  done
fi
log "done"; up
