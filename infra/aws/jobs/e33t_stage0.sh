#!/bin/bash
# E33t stage 0 on one free GPU of a home server (plan §7): the GPU tests (flex = reference read for the catalogue
# arms, forward and backward), the tiny pipeline check of both catalogue arms with the instrumentation, and the s30
# speed / memory check against LSA-native on the same GPU (spec K4: ≤ 1.3 × LSA's time).
#
# Usage (from the repo root, in tmux):  E33N_DATA=DIR CUDA_VISIBLE_DEVICES=<gpu> bash infra/aws/jobs/e33t_stage0.sh OUT
# Results: OUT/stage0.log (one line per step), OUT/speed/summary.tsv, OUT/*.log.
set -uo pipefail
OUT="${1:?usage: E33N_DATA=DIR CUDA_VISIBLE_DEVICES=<gpu> bash infra/aws/jobs/e33t_stage0.sh OUT}"
D="${E33N_DATA:?set E33N_DATA}"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUT/pipeline"
log() { echo "[e33t-s0 $(date -u +%FT%TZ)] $*" | tee -a "$OUT/stage0.log"; }
log "start $(git rev-parse --short HEAD) on GPU ${CUDA_VISIBLE_DEVICES:-all}"
uv run pytest tests/test_note_open_cuda.py tests/test_catalogue_read.py tests/test_note_open.py tests/test_flow_stats.py \
  -q -s > "$OUT/cuda_tests.log" 2>&1 && log "CUDA tests passed" || log "CUDA tests FAILED (see cuda_tests.log)"
E33N_ARMS="catalogue catalogue_2l" bash infra/aws/jobs/pipeline_check.sh "$D" "$OUT/pipeline" --flow \
  > "$OUT/pipeline.log" 2>&1 && log "pipeline check passed" || log "pipeline check FAILED (see pipeline.log)"
bash infra/aws/jobs/speed_check.sh "$D" "$OUT/speed" s30:catalogue:trunk:2.1e7:4 s30:catalogue:decay:6.5e6:1 \
  s30:catalogue_2l:decay:6.5e6:1 s30:lsa:decay:6.5e6:1 > "$OUT/speed.log" 2>&1 && log "speed check done" \
  || log "speed check FAILED (see speed.log)"
column -t -s $'\t' "$OUT/speed/summary.tsv" | tee -a "$OUT/stage0.log"
log "all done"
