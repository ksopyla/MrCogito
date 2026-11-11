#!/bin/bash
# E33t stage C evaluations past 128k (plan §10.8), one model on one GPU, with the list read (no [B, S, G] table):
#   1. long-document loss at 256k (every held-out row that long) and 1M (one document), notebook on / off
#   2. the planted password at 128k and 1M (20 depths in held-out documents), notebook on / off
#   3. with E33T_TWO_LEVEL=P (catalogue models): both again with the two-level search (P pages first) as an eval switch
# The 4k–128k long-document loss, RULER and core come from e33n_run.sh (E33N_LONG=1 for the 64k models).
#
# Home server (in tmux, from the repo root):
#   E33N_DATA=DIR CUDA_VISIBLE_DEVICES=2 bash infra/aws/jobs/e33t_c_evals.sh CHECKPOINT OUT_DIR TAG
# E33T_LOSS_LENGTHS (default 262144,1048576), E33T_KEY_LENGTHS (default 131072,1048576), E33T_KEY_DEPTHS (default 20).
# Results: OUT_DIR/c_<TAG>_longdoc.json, c_<TAG>_passkey.json (and _2l variants); a rerun skips finished files.
set -uo pipefail
CK="$1"; OUT="$2"; TAG="$3"
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D="${E33N_DATA:?set E33N_DATA}"
mkdir -p "$OUT"
log() { echo "[e33t-c $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$OUT/c_${TAG}.events.log"; }
run() {  # NAME CMD...: skip when NAME.json exists
  local name=$1; shift
  [ -f "$OUT/$name.json" ] && { log "$name exists, skipped"; return 0; }
  log "$name start"
  "$@" --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1 && log "$name done" || log "$name FAILED (see $name.log)"
}
log "start $TAG on $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-all}: $CK"
for lv in 0 ${E33T_TWO_LEVEL:-}; do
  sfx=""; extra=()
  if [ "$lv" != 0 ]; then sfx="_2l"; extra=(--two_level "$lv"); fi
  run "c_${TAG}_longdoc$sfx" uv run python evaluation/long_doc_loss.py --checkpoint "$CK" --data "$D/longdocs_lite_gpt2" \
    --lengths "${E33T_LOSS_LENGTHS:-262144,1048576}" --rows 16 --notes_off --open_off --open_impl list "${extra[@]}"
  run "c_${TAG}_passkey$sfx" uv run python evaluation/long_passkey.py --checkpoint "$CK" --data "$D/longdocs_lite_gpt2" \
    --tokenizer "$D/tokenizer_gpt2" --lengths "${E33T_KEY_LENGTHS:-131072,1048576}" --depths "${E33T_KEY_DEPTHS:-20}" \
    --notes_off --open_impl list "${extra[@]}"
done
log "done"
