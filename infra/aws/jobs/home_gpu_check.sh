#!/bin/bash
# E33n GPU check on a home server (Odra / Polonez, RTX 3090 24 GB): small data from public sources, then three
# speed_check.sh streams in parallel, one per GPU. Free; run before paying for H100 hours. Proves the flex top-k path
# compiles and trains on a GPU, and gives speed and memory per arm (an H100 is ~3× a 3090 for these sizes).
# Usage (repo root, in tmux): bash infra/aws/jobs/home_gpu_check.sh OUT_DIR
set -uo pipefail
OUT="$1"; D="$OUT/data"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUT"
uv sync -q || exit 1
[ -f "$D/longdocs_lite_gpt2/manifest.json" ] || bash infra/aws/jobs/data_build.sh "$D" 2 20e6 8 || exit 1
CUDA_VISIBLE_DEVICES=0 bash infra/aws/jobs/speed_check.sh "$D" "$OUT/gpu0" \
  s30:window:trunk:2.1e7:4 s30:dense:trunk:2.1e7:4 s30:dense:decay:6.5e6:1 > "$OUT/gpu0.out" 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash infra/aws/jobs/speed_check.sh "$D" "$OUT/gpu1" \
  s30:e33:trunk:2.1e7:4 s30:e33n:trunk:2.1e7:4 s30:e33n:decay:6.5e6:1 > "$OUT/gpu1.out" 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash infra/aws/jobs/speed_check.sh "$D" "$OUT/gpu2" \
  m100:e33n:trunk:5.2e6:2 m100:dense:trunk:5.2e6:2 > "$OUT/gpu2.out" 2>&1 &
wait
cat "$OUT"/gpu*/summary.tsv
echo "home GPU check done $(date -u)"
