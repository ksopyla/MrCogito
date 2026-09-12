#!/bin/bash
# Perceiver AR (E18 family) evaluation suite: reasoning (lm-evaluation-harness) +
# long-context (teacher-forced RULER-lite). One call per checkpoint; failure tolerant;
# lm-eval and the long-context suite run in parallel when LMEVAL_GPU != LONGCTX_GPU.
#
# Usage (on Polonez/Odra, inside byobu):
#   bash scripts/eval_perceiver_ar_suite.sh <checkpoint_dir> <tag>
#   HF_MODEL=HuggingFaceTB/SmolLM2-135M bash scripts/eval_perceiver_ar_suite.sh - smollm2_135m   # reference row (lm-eval only)
#
# Knobs (env):
#   LMEVAL_GPU=0 LONGCTX_GPU=1     GPU per half (same index -> sequential)
#   TIER=core|full  LIMIT=         lm-eval tier / examples-per-task cap (smoke)
#   MAX_LENGTH=2048 BATCH_SIZE=16  lm-eval context cap (loglikelihood tasks are short) / batch
#   CONTEXT_LENGTHS=8192,32768     RULER-lite lengths (add 65536,131072 for the long sweep)
#   BUCKETS=8192,32768             perplexity length buckets over MANIFEST rows
#   TRIALS=8 MAX_ROWS=64           synthetic trials per (probe, length) / manifest rows per bucket
#   PROBES=passkey,multikey,vt,fwe,buckets   (E21 checkpoints: append ,message for the paired
#                                  real/none/swapped/raw message ablation; MESSAGE_SPANS=512,4096)
#   MANIFEST=...                   pretokenized manifest for filler + buckets
#                                  (default: $DATASETS_TOK_DIR/../datasets_tok_smollm3_32k/e18b_lm_ret05_manifest.json)
#   ATTN_BACKEND=flex              long-context backend (flex for >= 8k; sdpa for short)
#   SKIP_HEALTH=1 SKIP_LMEVAL=1 SKIP_LONGCTX=1 SKIP_REACH=1
#
# Outputs:
#   Cache/eval/<tag>/health.log, longctx_suite.json, reach.json
#   Cache/Evaluation_reports/lm_eval/<tag>.json  (+ summary.csv row upsert)
#   Cache/logs/eval_<tag>_<stamp>.log
set -uo pipefail

CKPT="${1:?checkpoint dir (or '-' with HF_MODEL)}"
TAG="${2:?tag}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/remote_paths.sh
source "${SCRIPT_DIR}/remote_paths.sh"
cd "${PROJECT_ROOT}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LMEVAL_GPU="${LMEVAL_GPU:-0}"
LONGCTX_GPU="${LONGCTX_GPU:-1}"
TIER="${TIER:-core}"
LIMIT="${LIMIT:-}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-8192,32768}"
BUCKETS="${BUCKETS:-8192,32768}"
TRIALS="${TRIALS:-8}"
MAX_ROWS="${MAX_ROWS:-64}"
PROBES="${PROBES:-passkey,multikey,vt,fwe,buckets}"
MESSAGE_SPANS="${MESSAGE_SPANS:-512,4096}"   # E21 message probe CE spans after the boundary
ATTN_BACKEND="${ATTN_BACKEND:-flex}"
MANIFEST="${MANIFEST:-${DATASETS_TOK_DIR}/../datasets_tok_smollm3_32k/e18b_lm_ret05_manifest.json}"
HF_MODEL="${HF_MODEL:-}"

OUT="Cache/eval/${TAG}"
REPORTS="Cache/Evaluation_reports/lm_eval"
mkdir -p "$OUT" "$REPORTS" Cache/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="Cache/logs/eval_${TAG}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== perceiver_ar eval suite: ${TAG} @ ${STAMP} ==="
echo "HEAD=$(git rev-parse --short HEAD) $(git log -1 --oneline)"
echo "CKPT=$CKPT HF_MODEL=${HF_MODEL:-<none>}"
echo "LMEVAL_GPU=$LMEVAL_GPU LONGCTX_GPU=$LONGCTX_GPU TIER=$TIER LIMIT=${LIMIT:-all}"
echo "CONTEXT_LENGTHS=$CONTEXT_LENGTHS BUCKETS=$BUCKETS TRIALS=$TRIALS PROBES=$PROBES"
echo "MANIFEST=$MANIFEST"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv || true

if [ -z "$HF_MODEL" ] && [ ! -f "$CKPT/config.json" ]; then
  echo "FATAL: $CKPT has no config.json"; exit 2
fi
if [ -n "$HF_MODEL" ]; then
  SKIP_LONGCTX=1; SKIP_HEALTH=1; SKIP_REACH=1
fi

STATUS_DIR="$(mktemp -d)"
run_step() {   # run_step <name> <gpu> <cmd...>   -> writes $STATUS_DIR/<name>
  local name="$1" gpu="$2"; shift 2
  echo; echo "======== $name (GPU $gpu) $(date +%H:%M:%S) ========"
  local t0=$SECONDS
  CUDA_VISIBLE_DEVICES="$gpu" "$@"
  local code=$?
  echo "$code" > "$STATUS_DIR/$name"
  echo "======== $name exit=$code in $((SECONDS - t0))s ========"
}

# --- Tier 0: weights + forward/loss health (fast, on the lm-eval GPU) ---
health() {
  uv run python analysis/check_model_health.py --model_path "$CKPT" --model_type perceiver_ar \
    > "$OUT/health.log" 2>&1
  local code=$?
  grep -E "^\[|Loss value|Logits range|Unique predictions|FINAL|MODEL" "$OUT/health.log" | tail -12
  return $code
}

# --- Reasoning: lm-evaluation-harness ---
lmeval() {
  local src=(--checkpoint "$CKPT")
  [ -n "$HF_MODEL" ] && src=(--hf_model "$HF_MODEL")
  local extra=()
  [ -n "$LIMIT" ] && extra+=(--limit "$LIMIT")
  uv run python evaluation/run_lm_eval_suite.py "${src[@]}" --tag "$TAG" --tier "$TIER" \
    --max_length "$MAX_LENGTH" --batch_size "$BATCH_SIZE" --out_dir "$REPORTS" "${extra[@]}"
}

# --- Long context: RULER-lite suite from one model load ---
longctx() {
  uv run python evaluation/long_context_probes.py --checkpoint "$CKPT" --probe suite --suite "$PROBES" \
    --attn_backend "$ATTN_BACKEND" --manifest "$MANIFEST" --context_lengths "$CONTEXT_LENGTHS" \
    --buckets "$BUCKETS" --trials "$TRIALS" --max_rows "$MAX_ROWS" --message_spans "$MESSAGE_SPANS" \
    --out "$OUT/longctx_suite.json"
}

# --- Reach: which window the full layers actually use (positive control for the global read) ---
reach() {
  uv run python evaluation/long_context_probes.py --checkpoint "$CKPT" --probe reach \
    --attn_backend "$ATTN_BACKEND" --manifest "$MANIFEST" --context_lengths "$CONTEXT_LENGTHS" \
    --trials "$TRIALS" --out "$OUT/reach.json"
}

lmeval_half() {
  [ "${SKIP_HEALTH:-0}" = 1 ] || run_step health "$LMEVAL_GPU" health
  [ "${SKIP_LMEVAL:-0}" = 1 ] || run_step lmeval "$LMEVAL_GPU" lmeval
}
longctx_half() {
  [ "${SKIP_LONGCTX:-0}" = 1 ] || run_step longctx "$LONGCTX_GPU" longctx
  [ "${SKIP_REACH:-0}" = 1 ] || run_step reach "$LONGCTX_GPU" reach
}

if [ "$LMEVAL_GPU" = "$LONGCTX_GPU" ]; then
  lmeval_half; longctx_half
else
  # Explicit PIDs: with `exec > >(tee ...)` a bare `wait` (bash >= 5.1) also waits for the
  # tee process substitution, which only exits when this script does -> deadlock.
  lmeval_half & P_LM=$!
  longctx_half & P_LC=$!
  wait "$P_LM" "$P_LC"
fi

echo; echo "=== summary for ${TAG} ==="
FAILED=()
for f in "$STATUS_DIR"/*; do
  [ -e "$f" ] || continue
  n="$(basename "$f")"; c="$(cat "$f")"
  printf '  %-8s exit=%s\n' "$n" "$c"
  [ "$c" = 0 ] || FAILED+=("$n:$c")
done
rm -rf "$STATUS_DIR"
uv run python evaluation/summarize_eval_suite.py --tags "$TAG" --eval_root Cache/eval \
  --lm_csv "$REPORTS/summary.csv" --out "$OUT/summary.md" || true
echo "FAILED: ${FAILED[*]:-none}"
echo "LOG=$LOG"
[ "${#FAILED[@]}" -eq 0 ]
