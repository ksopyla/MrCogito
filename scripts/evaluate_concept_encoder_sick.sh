#!/bin/bash
# Evaluate Concept Encoder on SICK (relatedness + entailment).
#
# SICK tests two properties critical for concept encoders:
#   - Relatedness (regression): Do concepts preserve semantic similarity?
#   - Entailment (3-class): Do concepts preserve compositional meaning?
#
# Usage:
#   bash scripts/evaluate_concept_encoder_sick.sh                  # both tasks (default)
#   bash scripts/evaluate_concept_encoder_sick.sh sick_relatedness # relatedness only
#   bash scripts/evaluate_concept_encoder_sick.sh sick_entailment  # entailment only
#   bash scripts/evaluate_concept_encoder_sick.sh sick_all         # both explicitly
#
# MODEL_PATH is set directly in this file — update it when a new model is trained.

set -o pipefail

echo "=== SICK Evaluation for Concept Encoder ==="

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
# perceiver_mlm L6 + fixed0.1 combined (Feb 20 2026, eff. rank 12.5%)
MODEL_PATH="${PROJECT_ROOT}/Cache/Training/perceiver_mlm_H512L6C128_20260220_184029/perceiver_mlm_H512L6C128_20260220_184029"
# perceiver_mlm L6 + combined+kendall_gal (Feb 19 2026, eff. rank 95.5%)
# MODEL_PATH="${PROJECT_ROOT}/Cache/Training/perceiver_mlm_H512L6C128_20260219_105435/perceiver_mlm_H512L6C128_20260219_105435"
# L6 baseline — no concept losses (eff. rank 4%)
# MODEL_PATH="${PROJECT_ROOT}/Cache/Training/perceiver_mlm_H512L6C128_20260208_211633/perceiver_mlm_H512L6C128_20260208_211633"
# =============================================================================

# Benchmark: optional $1, defaults to "sick_all" (both relatedness + entailment)
BENCHMARK="${1:-sick_all}"

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
