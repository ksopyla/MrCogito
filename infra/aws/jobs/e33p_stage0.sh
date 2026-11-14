#!/bin/bash
# E33p stage 0 on one home server with three free RTX 3090s (plan §7): GPU tests, the tiny pipeline check of every
# arm with the instrumentation, and the s30 speed / memory check of the new arms and their controls.
#
# Usage (from the repo root, in tmux):  E33N_DATA=DIR bash infra/aws/jobs/e33p_stage0.sh OUT
#   GPU 0: CUDA tests → tiny pipeline check (all arms, --flow) → pages_open trunk + decay speed
#   GPU 1: e33 trunk → notes_open trunk → notes_open decay speed
#   GPU 2: lsa trunk → lsa decay → dense trunk speed (dense now loops 4 rounds, like every arm)
# Results: OUT/summary.tsv (speed_check rows from all three queues), OUT/*.log; a line per finished queue in
# OUT/stage0.log. Each speed item trains 2.1e7 tokens (trunk, micro 4) or 6.5e6 (decay, micro 1).
set -uo pipefail
OUT="${1:?usage: E33N_DATA=DIR bash infra/aws/jobs/e33p_stage0.sh OUT}"
D="${E33N_DATA:?set E33N_DATA}"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUT"
log() { echo "[e33p-s0 $(date -u +%FT%TZ)] $*" | tee -a "$OUT/stage0.log"; }

q0() {
  log "gpu0 CUDA tests"
  CUDA_VISIBLE_DEVICES=0 uv run pytest tests/test_note_open_cuda.py tests/test_note_open.py tests/test_lsa_read.py \
    tests/test_flow_stats.py -q -s > "$OUT/cuda_tests.log" 2>&1 && log "gpu0 CUDA tests passed" \
    || log "gpu0 CUDA tests FAILED (see cuda_tests.log)"
  log "gpu0 tiny pipeline check"
  mkdir -p "$OUT/pipeline"
  CUDA_VISIBLE_DEVICES=0 bash infra/aws/jobs/pipeline_check.sh "$D" "$OUT/pipeline" --flow > "$OUT/pipeline.log" 2>&1 \
    && log "gpu0 pipeline check passed" || log "gpu0 pipeline check FAILED (see pipeline.log)"
  CUDA_VISIBLE_DEVICES=0 bash infra/aws/jobs/speed_check.sh "$D" "$OUT/speed" s30:pages_open:trunk:2.1e7:4 \
    s30:pages_open:decay:6.5e6:1 > "$OUT/speed_q0.log" 2>&1
  log "gpu0 done"
}
q1() {
  CUDA_VISIBLE_DEVICES=1 bash infra/aws/jobs/speed_check.sh "$D" "$OUT/speed" s30:e33:trunk:2.1e7:4 \
    s30:notes_open:trunk:2.1e7:4 s30:notes_open:decay:6.5e6:1 > "$OUT/speed_q1.log" 2>&1
  log "gpu1 done"
}
q2() {
  CUDA_VISIBLE_DEVICES=2 bash infra/aws/jobs/speed_check.sh "$D" "$OUT/speed" s30:lsa:trunk:2.1e7:4 \
    s30:lsa:decay:6.5e6:1 s30:dense:trunk:2.1e7:4 > "$OUT/speed_q2.log" 2>&1
  log "gpu2 done"
}
log "start $(git rev-parse --short HEAD)"
q0 & q1 & q2 &
wait
log "all done"
column -t -s $'\t' "$OUT/speed/summary.tsv" | tee -a "$OUT/stage0.log"
