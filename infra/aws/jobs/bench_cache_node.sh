#!/bin/bash
# Cached vs uncached greedy decoding on one GPU (2026-11-13, author: "Yes, on AWS (~$1)"): the five 100M models
# (dense, LSA port, catalogue n4, E37's concept read e37a, E33p's notes-open), each at LENGTHS, 32 answer tokens,
# through evaluation/bench_cached_decoding.py (same tokens both ways? wall time of each, peak GPU memory).
# A launch.sh job for one L40S (g6e.xlarge), branch kv-cache:
#   MC_SPOT=1 bash infra/aws/launch.sh kvcache-bench 2 "infra/aws/jobs/bench_cache_node.sh" kv-cache g6e.xlarge
# Arguments (optional): ARM... (default: all five). LENGTHS / UNCACHED_MAX override the lengths and the longest
# length also timed without the cache (default 16k, 32k, 64k both ways; 128k cached only).
# Results: s3://BUCKET/runs/kvcache_bench/<node>/<arm>.{json,log}, uploaded after every model.
set -uo pipefail
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B="s3://$MC_BUCKET/runs/e37"; OUT="s3://$MC_BUCKET/runs/kvcache_bench/$MC_NAME"; W="$MC_DISK/bench"
LENGTHS="${LENGTHS:-16384,32768,65536,131072}"; UNCACHED_MAX="${UNCACHED_MAX:-65536}"
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(dense lsa catalogue_reuse_n4 e37a notes_open)
mkdir -p "$W"
log() { echo "[bench_cache $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$W/node.log"; }
log "start on $MC_NAME ${MC_IID:-} ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)); code $(git log --oneline -1 | cut -c1-60)"
for arm in "${ARMS[@]}"; do
  run="e33n_m100_${arm}_tok3.0e9_seed42"
  key=$(aws s3 ls --recursive "$B/$run/decay/" | grep "/final/config.json" | awk '{print $4}' | head -1)
  [ -n "$key" ] || { log "$arm: no final checkpoint under $B/$run/decay/"; continue; }
  ck="$W/ckpt/$arm"
  aws s3 sync --only-show-errors "s3://$MC_BUCKET/$(dirname "$key")/" "$ck/" --exclude "*.bin" --exclude "optimizer*"
  log "$arm: $(dirname "$key") -> $(du -sh "$ck" | cut -f1)"
  uv run python evaluation/bench_cached_decoding.py --checkpoint "$ck" --lengths "$LENGTHS" --new 32 \
    --skip_uncached_above "$UNCACHED_MAX" --out "$W/$arm.json" > "$W/$arm.log" 2>&1
  log "$arm: exit $? $(grep -c '"length"' "$W/$arm.log") lengths"
  aws s3 cp --only-show-errors "$W/$arm.log" "$OUT/$arm.log"
  [ -f "$W/$arm.json" ] && aws s3 cp --only-show-errors "$W/$arm.json" "$OUT/$arm.json"
done
aws s3 cp --only-show-errors "$W/node.log" "$OUT/node.log"
log "done"
