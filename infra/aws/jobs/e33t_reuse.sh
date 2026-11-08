#!/bin/bash
# E33t, the catalogue with the pass exclusion off (arm catalogue_reuse: a loop pass may re-open notes that earlier
# passes opened), seed 1, the stage-A recipe (e33t_a.sh: trunk, 16k decay, long-document loss with the eval switches,
# core benchmarks, RULER at 4k), then the planted-password probe on it (flow_probe.py --passkey: 4k × 40 rows and
# 16k × 20 rows, the rows the catalogue, the LSA port, notes-open and dense were probed on, 2026-11-07).
# Why: that probe found the catalogue opens the key on pass 1 (94 % at 4k) but may not return to it on passes 2–4,
# where the LSA port and dense put 37–41 % of their read on it; switching the exclusion off only at test time
# (open_override "reuse") lifted answers from 10 % to 20 %, so the passes have to learn it in training.
# Usage (launch.sh / chain_job.sh JOB words): infra/aws/jobs/e33t_reuse.sh
# Results: s3://BUCKET/runs/e33t/e33n_s30_catalogue_reuse_seed1/ (probe: probe_passkey_<len>.json in the same folder).
set -uo pipefail
source /etc/mrcogito.env
bash "$(dirname "$0")/e33t_a.sh" catalogue_reuse
R="$MC_DISK/runs/e33n_s30_catalogue_reuse_seed1"
S3R="s3://$MC_BUCKET/runs/e33t/e33n_s30_catalogue_reuse_seed1"
F=$(ls -d "$R"/decay/*/final 2>/dev/null | tail -1)
[ -n "$F" ] || { echo "[e33t_reuse] no decay checkpoint under $R: probe skipped"; exit 1; }
cd "$(dirname "$0")/../../.." || exit 1
export PATH="$HOME/.local/bin:$PATH" CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for L in 4096:40 16384:20; do
  len=${L%%:*}; rows=${L##*:}
  uv run python evaluation/flow_probe.py --checkpoint "$F" --data "$MC_DISK/data/longdocs_lite_gpt2" --length "$len" \
    --rows "$rows" --passkey --out "$R/probe_passkey_$len.json" > "$R/probe_passkey_$len.log" 2>&1
  echo "rc $?" >> "$R/probe_passkey_$len.log"
  echo "[e33n $(date -u +%Y-%m-%dT%H:%M:%SZ)] planted-password probe $len done" | tee -a "$R/events.log"
done
aws s3 sync --only-show-errors "$R/" "$S3R/" --exclude "*/checkpoint-*" --exclude "*optimizer.pt" \
  --exclude "*rng_state*" --exclude "*.pth" || true
