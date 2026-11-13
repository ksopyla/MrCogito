#!/bin/bash
# Two agents from one trained 100M model, eval only (2026-11-13, author: "you can even start testing by using the AWS
# machines ... check if this is even possible"): scripts/a2a_split_probe.py on one model's final checkpoint (agent A
# reads a RULER needle document, agent B gets only the question and reads A's memory), and optionally
# scripts/cramming_probe.py on the same model (how many bits K input vectors carry into it, the frozen receiver).
# A launch.sh job for one L40S (g6e.xlarge):
#   MC_SPOT=1 bash infra/aws/launch.sh a2a-e37a 4 "infra/aws/jobs/a2a_probe_node.sh e37a 4096,16384,32768" BRANCH g6e.xlarge
# Arguments: ARM (e37a | lsa | dense: the m100 full-budget runs) LENGTHS [cram]. Environment: A2A_TASKS, A2A_LIMIT (30),
#   A2A_LIMIT_LONG (20, past 16k), A2A_BITS (8,4), A2A_BITS_DOCS (15), A2A_TAILS (16), A2A_NEW (12), A2A_TAG (a suffix of the output names, so
#   two runs of one arm can share a node: several of these jobs fit one L40S, the steps are small and CPU-bound).
# Results: s3://BUCKET/runs/a2a_probe/<node>/ (split_<arm>.json after every task and length, logs every 5 minutes).
set -uo pipefail
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARM="${1:?arm}"; LENGTHS="${2:?lengths}"; CRAM="${3:-}"; N="${1}${A2A_TAG:+_$A2A_TAG}"
B="s3://$MC_BUCKET/runs/e37"; OUT="s3://$MC_BUCKET/runs/a2a_probe/$MC_NAME"; W="$MC_DISK/a2a"
mkdir -p "$W/out"
log() { echo "[a2a_probe $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$W/out/node.log"; }
log "$N: start on $MC_NAME ${MC_IID:-} ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)); code $(git log --oneline -1 | cut -c1-60)"
run="e33n_m100_${ARM}_tok3.0e9_seed42"
key=$(aws s3 ls --recursive "$B/$run/decay/" | grep "/final/config.json" | awk '{print $4}' | head -1)
[ -n "$key" ] || { log "$ARM: no final checkpoint under $B/$run/decay/"; exit 1; }
ck="$W/ckpt/$ARM"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/$(dirname "$key")/" "$ck/" --exclude "*.bin" --exclude "optimizer*"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/tokenizer_gpt2/" "$W/tok/"
log "$ARM: $(dirname "$key") -> $(du -sh "$ck" | cut -f1)"
( while sleep 300; do aws s3 sync --only-show-errors "$W/out/" "$OUT/" || true; done ) &
SYNC=$!
uv run --with wonderwords python scripts/a2a_split_probe.py --checkpoint "$ck" --tokenizer "$W/tok" \
  --tasks "${A2A_TASKS:-niah_single_1,niah_single_2}" --lengths "$LENGTHS" \
  --limit "${A2A_LIMIT:-30}" --limit_long "${A2A_LIMIT_LONG:-20}" --bits "${A2A_BITS:-8,4}" \
  --bits_docs "${A2A_BITS_DOCS:-15}" --tails "${A2A_TAILS:-16}" --new "${A2A_NEW:-12}" --out "$W/out/split_${N}.json" > "$W/out/split_${N}.log" 2>&1
log "split probe: exit $? $(grep -c '^{"task"' "$W/out/split_${N}.log") cells"
aws s3 sync --only-show-errors "$W/out/" "$OUT/" || true
if [ "$CRAM" = cram ]; then
  aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/finewebedu_val_000000.bin" "$W/val.bin"
  for spec in "1:16 32 64 128" "4:64 128 256 512" "16:256 512 1024"; do
    k=${spec%%:*}
    # shellcheck disable=SC2086
    uv run python scripts/cramming_probe.py --checkpoint "$ck" --bin "$W/val.bin" --vectors "$k" \
      --lengths ${spec#*:} --n_texts 8 --steps "${A2A_CRAM_STEPS:-2000}" \
      --out "$W/out/cramming_${ARM}_k${k}.json" > "$W/out/cramming_${ARM}_k${k}.log" 2>&1
    log "cramming K=$k: exit $?"
    aws s3 sync --only-show-errors "$W/out/" "$OUT/" || true
  done
fi
kill "$SYNC" 2>/dev/null
log "done"
aws s3 sync --only-show-errors "$W/out/" "$OUT/" || true
