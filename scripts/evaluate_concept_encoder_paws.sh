#!/bin/bash
# Evaluate Concept Encoder on PAWS (adversarial paraphrase detection).
#
# PAWS tests whether concepts encode semantics, not surface word overlap.
# ("Flights from NYC to LA" vs "Flights from LA to NYC" — same words, different meaning.)
#
# Usage:
#   bash scripts/evaluate_concept_encoder_paws.sh
#
# MODEL_PATH is set directly in this file — update it when a new model is trained.

set -o pipefail

echo "=== PAWS Evaluation for Concept Encoder ==="

PROJECT_ROOT="/home/ksopyla/dev/MrCogito"
if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(pwd)"
fi

export HF_HOME="${PROJECT_ROOT}/../hf_home"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/../hf_home/datasets"
export TOKENIZERS_PARALLELISM=false

# Python runs via `uv run`; ensure uv is on PATH for non-interactive SSH.
if [ -d "$HOME/.local/bin" ]; then export PATH="$HOME/.local/bin:$PATH"; fi

# =============================================================================
# MODEL TO EVALUATE — update this when a new denoising checkpoint is trained
# =============================================================================
MODEL_PATH="${MODEL_PATH_OVERRIDE:-${PROJECT_ROOT}/Cache/Training/REPLACE_WITH_PERCEIVER_DENOISE_CHECKPOINT}"
MODEL_TYPE="${MODEL_TYPE_OVERRIDE:-perceiver_denoise}"
# =============================================================================

echo "  Model Type: $MODEL_TYPE"
echo "  Model Path: $MODEL_PATH"
echo ""

if command -v uv > /dev/null 2>&1; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python3"
fi

$PYTHON_CMD evaluation/evaluate_on_benchmark.py \
    --benchmark paws \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name "$MODEL_PATH" \
    --batch_size 96 \
    --epochs 5 \
    --learning_rate 1e-5

echo ""
echo "PAWS evaluation completed!"
