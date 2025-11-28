#!/bin/bash
# Bash script to evaluate the Concept Encoder model on the GLUE benchmark on Linux (Odra)
# Based on scripts/evaluate_concept_encoder_glue.ps1

# Exit on error
set -e

echo "=== GLUE Evaluation Script for Concept Encoder ==="

# --- Configuration ---

# Project root - automatically detect or set hardcoded
# Adapting to Odra server path structure
PROJECT_ROOT="/home/ksopyla/dev/MrCogito"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Warning: Project root $PROJECT_ROOT not found. Using current directory."
    PROJECT_ROOT="$(pwd)"
fi

# Set HuggingFace cache directories (Consistent with training script)
export HF_HOME="${PROJECT_ROOT}/../hf_home"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/../hf_home/datasets"

# Unset deprecated variable to avoid warnings
unset TRANSFORMERS_CACHE

# Default Model Path 
# Using the path provided in the user query
DEFAULT_MODEL_PATH="${PROJECT_ROOT}/Cache/Training/weighted_mlm_H512L2C128_20251123_213949/weighted_mlm_H512L2C128_20251123_213949"

# Allow overriding model path via first argument
MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"

# Task Configuration
# Default task is mrpc, can be overridden
TASK="${2:-mrpc}" 
# Task list: cola, mnli-matched, mnli-mismatched, mrpc, qnli, qqp, rte, sst2, stsb, wnli

# Tokenizer - should match what was used in training (bert-base-cased per training script)
TOKENIZER_NAME="bert-base-cased"

echo "Configuration:"
echo "  - Project Root: $PROJECT_ROOT"
echo "  - HF Cache: $HF_HOME"
echo "  - Model Path: $MODEL_PATH"
echo "  - Task: $TASK"
echo "  - Tokenizer: $TOKENIZER_NAME"
echo ""

# Check if model path exists
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "WARNING: Model path does not seem to exist or contain config.json: $MODEL_PATH"
    echo "Please check the path."
fi

echo "Starting evaluation..."

# Execute evaluation script
# Assumes python environment is already activated or python is accessible
# Added --visualize flag
# Added --save_model flag
# IMPORTANT: Using 'weighted_mlm' model type triggers the new Weighted Classification Head
python training/evaluate_model_on_glue.py \
    --model_type "weighted_mlm" \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --task "$TASK" \
    --batch_size 32 \
    --epochs 5 \
    --learning_rate 2e-5 \
    --visualize \
    --save_model

echo ""
echo "GLUE evaluation completed!"
