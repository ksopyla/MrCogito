#!/bin/bash
# E10 pre-launch gate: wait for frozen 8K eval data, rerun paired Stage 0,
# build the 2K Gemma training manifest, then launch the concept arm.
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
# shellcheck source=scripts/remote_paths.sh
source "${SCRIPT_DIR}/remote_paths.sh"

EVAL_CACHE="${EVAL_CACHE:-${HF_HOME}/datasets_tok_gemma_e10_eval8k_12k_v2}"
EVAL_MANIFEST="${EVAL_MANIFEST:-${EVAL_CACHE}/e10_frozen_eval8k_manifest.json}"
STAGE0_OUTPUT="${PROJECT_ROOT}/Cache/e10_stage0_frozen_heldout.json"
TRAIN_CACHE="${HF_HOME}/datasets_tok_gemma"
TRAIN_MANIFEST="${TRAIN_CACHE}/smollm3_inspired_2k_e05_gemma_manifest.json"
RAW_ARCHIVE="/nas/ml_data/mrcogito/hf_datasets/raw"

echo "Waiting for frozen E10 eval manifest: ${EVAL_MANIFEST}"
while [ ! -s "$EVAL_MANIFEST" ]; do
    if ! pgrep -f "pretokenize_mix.py.*--eval_only" >/dev/null; then
        echo "ERROR: eval pretokenizer stopped before writing ${EVAL_MANIFEST}"
        exit 1
    fi
    sleep 60
done

echo "=== E10 paired held-out Stage 0 ==="
uv run python analysis/run_e10_stage0.py \
    --seq_lens 2048 8192 \
    --num_docs 64 \
    --batch_size 1 \
    --eval_manifest "$EVAL_MANIFEST" \
    --output "$STAGE0_OUTPUT"

uv run python -c '
import json, sys
r = json.load(open(sys.argv[1]))
g2 = r["seq_lens"]["2048"]["G_beyond_1024"]
g8 = r["seq_lens"]["8192"]["G_beyond_1024"]
action = r["recommendation"]["action"]
print(f"Stage-0 gate: G2K={g2:.4f} G8K={g8:.4f} action={action}")
if g2 < 0.05 or g8 < 0.05 or action != "KEEP_SPEC":
    raise SystemExit("E10 Stage-0 gate failed; training will not launch.")
' "$STAGE0_OUTPUT"

echo "=== E10 2K Gemma training pretokenization ==="
uv run python scripts/pretokenize_mix.py \
    --mix smollm3_inspired_2k_e05 \
    --tokenizer google/gemma-3-1b-pt \
    --max_seq_length 2048 \
    --cache_dir "$TRAIN_CACHE" \
    --raw_dir "$RAW_ARCHIVE" \
    --manifest "$TRAIN_MANIFEST" \
    --objective causal_lm \
    --seed 42 \
    --train_num_proc 8 \
    --test_num_proc 4 \
    --download_workers 8 \
    --jobs 1 \
    --raw_archive_dir "$RAW_ARCHIVE"

echo "=== E10 Stage 0 passed; launching concept arm ==="
export DATASETS_TOK_DIR="$TRAIN_CACHE"
export DATASETS_RAW_DIR="$RAW_ARCHIVE"
export MANIFEST="$TRAIN_MANIFEST"
export SKIP_PRETOKENIZE=1
export EXPERIMENT_ID="${EXPERIMENT_ID:-E10}"
exec bash scripts/launch_e10.sh
