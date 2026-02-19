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

# =============================================================================
# MODEL TO EVALUATE — update this when a new model is trained
# =============================================================================
# perceiver_mlm L6 + combined+kendall_gal (Feb 19 2026, eff. rank 95.5%)
MODEL_PATH="${PROJECT_ROOT}/Cache/Training/perceiver_mlm_H512L6C128_20260219_105435/perceiver_mlm_H512L6C128_20260219_105435"
# L6 baseline — no concept losses (eff. rank 4%)
# MODEL_PATH="${PROJECT_ROOT}/Cache/Training/perceiver_mlm_H512L6C128_20260208_211633/perceiver_mlm_H512L6C128_20260208_211633"
# =============================================================================

# Auto-detect MODEL_TYPE from MODEL_PATH name
if echo "$MODEL_PATH" | grep -q "perceiver_posonly_mlm"; then
    MODEL_TYPE="perceiver_posonly_mlm"
elif echo "$MODEL_PATH" | grep -q "perceiver_mlm"; then
    MODEL_TYPE="perceiver_mlm"
elif echo "$MODEL_PATH" | grep -q "weighted_mlm"; then
    MODEL_TYPE="weighted_mlm"
else
    MODEL_TYPE="perceiver_mlm"
fi

echo "  Model Type: $MODEL_TYPE"
echo "  Model Path: $MODEL_PATH"
echo ""

python evaluation/evaluate_on_benchmark.py \
    --benchmark paws \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name "$MODEL_PATH" \
    --batch_size 96 \
    --epochs 5 \
    --learning_rate 1e-5

echo ""
echo "PAWS evaluation completed!"
