#!/bin/bash
# E14 — E10e architecture on forced four-block delayed recall.
# Builds no model fork: this pins the approved data/masking/budget profile and
# delegates to launch_e10.sh, which delegates to the canonical generic trainer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/ksopyla/dev/MrCogito"
[ -d "$PROJECT_ROOT" ] || PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DATASETS_TOK_DIR="${DATASETS_TOK_DIR:-${PROJECT_ROOT}/../hf_home/datasets_tok_gemma}"
export EXPERIMENT_ID=E14
export PRETOKENIZE_MIX=e14_delayed_recall
export MANIFEST="${MANIFEST:-${DATASETS_TOK_DIR}/e14_delayed_recall_gemma_manifest.json}"
export SKIP_PRETOKENIZE=1
export TARGET_TOKENS="${TARGET_TOKENS:-9437184}"

export PRESERVE_PRECOMPUTED_LABELS=true
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export READ_CONCEPT_NORM=true
export READ_GATE_INIT=0.01
export WRITE_GATE_INIT=0.01
export CONCEPT_MEMORY_LR=3e-4
export WARMUP_STEPS=50
export LOGGING_STEPS="${LOGGING_STEPS:-20}"

# Effective batch 6 on Odra: checkpoint 164 follows 2,015,232 input tokens.
export AUTO_INTERVALS=0
export EVAL_STEPS=164
export SAVE_STEPS=164
export SAVE_TOTAL_LIMIT=3
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-256}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: E14 manifest not found: $MANIFEST"
    echo "Build it first with scripts/build_delayed_recall_dataset.py (see the E14 plan)."
    exit 1
fi

echo "=== E14 forced delayed-recall launch ==="
echo "  manifest=$MANIFEST target_tokens=$TARGET_TOKENS checkpoint_gate_step=$SAVE_STEPS"
echo "  sparse_labels=$PRESERVE_PRECOMPUTED_LABELS batch_per_gpu=$PER_DEVICE_BATCH_SIZE accum=$GRADIENT_ACCUMULATION_STEPS"

exec bash "${SCRIPT_DIR}/launch_e10.sh"
