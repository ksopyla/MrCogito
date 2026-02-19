#!/bin/bash
# Evaluate Concept Encoder on PAWS (adversarial paraphrase detection)
#
# PAWS contains sentence pairs that share almost all words but may have
# different meanings ("Flights from NYC to LA" vs "Flights from LA to NYC").
# Bag-of-words models fail badly. If the concept bottleneck captures
# genuine semantic structure, it should handle these adversarial pairs.
#
# Usage:
#   bash scripts/evaluate_concept_encoder_paws.sh                          # default model
#   bash scripts/evaluate_concept_encoder_paws.sh <model_path>             # custom model
#   bash scripts/evaluate_concept_encoder_paws.sh <model_path> <model_type>

set -o pipefail

echo "=== PAWS Evaluation for Concept Encoder ==="

PROJECT_ROOT="/home/ksopyla/dev/MrCogito"
if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(pwd)"
fi

export HF_HOME="${PROJECT_ROOT}/../hf_home"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/../hf_home/datasets"
export TOKENIZERS_PARALLELISM=false

# Default: perceiver_mlm L6 + combined+kendall_gal concept losses (Feb 19 2026, eff. rank 95.5%)
DEFAULT_MODEL_PATH="${PROJECT_ROOT}/Cache/Training/perceiver_mlm_H512L6C128_20260219_105435/perceiver_mlm_H512L6C128_20260219_105435"
# L6 baseline (no concept losses): perceiver_mlm_H512L6C128_20260208_211633
MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"

# Auto-detect model type
if [ -n "$2" ]; then
    MODEL_TYPE="$2"
elif echo "$MODEL_PATH" | grep -q "perceiver_posonly_mlm"; then
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
