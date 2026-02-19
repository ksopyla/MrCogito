#!/bin/bash
# Evaluate Concept Encoder on SICK (relatedness + entailment)
#
# SICK tests two properties critical for concept encoders:
#   - Relatedness (regression): Do concepts preserve semantic similarity?
#   - Entailment (3-class): Do concepts preserve compositional meaning?
#
# Usage:
#   bash scripts/evaluate_concept_encoder_sick.sh                          # default model, both tasks
#   bash scripts/evaluate_concept_encoder_sick.sh <model_path>             # custom model, both tasks
#   bash scripts/evaluate_concept_encoder_sick.sh <model_path> sick_relatedness  # relatedness only
#   bash scripts/evaluate_concept_encoder_sick.sh <model_path> sick_entailment   # entailment only
#   bash scripts/evaluate_concept_encoder_sick.sh <model_path> sick_all <model_type>

set -o pipefail

echo "=== SICK Evaluation for Concept Encoder ==="

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
BENCHMARK="${2:-sick_all}"

# Auto-detect model type
if [ -n "$3" ]; then
    MODEL_TYPE="$3"
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
echo "  Benchmark:  $BENCHMARK"
echo ""

python evaluation/evaluate_on_benchmark.py \
    --benchmark "$BENCHMARK" \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name "$MODEL_PATH" \
    --batch_size 96 \
    --epochs 10 \
    --learning_rate 1e-5

echo ""
echo "SICK evaluation completed!"
