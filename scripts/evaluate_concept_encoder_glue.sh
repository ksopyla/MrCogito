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

# Model Configuration - set both together!
#
# MODEL_TYPE options for concept encoders:
#   - "weighted_mlm": Weighted attention pooling approach
#   - "perceiver_mlm": Perceiver IO with Input+Position decoder queries
#   - "perceiver_posonly_mlm": Perceiver IO with Position-only decoder queries (pure Perceiver IO)
#
# Note: perceiver_mlm and perceiver_posonly_mlm use the same classification head
# (the difference is only in how the MLM decoder works during pretraining)

# Weighted MLM:
# DEFAULT_MODEL_PATH="${PROJECT_ROOT}/Cache/Training/weighted_mlm_H512L2C128_20260117_153544/weighted_mlm_H512L2C128_20260117_153544"
# MODEL_TYPE="weighted_mlm"

# Perceiver MLM (comment out above, uncomment below):
DEFAULT_MODEL_PATH="${PROJECT_ROOT}/Cache/Training/perceiver_mlm_H512L2C128_20260118_172328/perceiver_mlm_H512L2C128_20260118_172328"
MODEL_TYPE="perceiver_mlm"

# Perceiver Position-Only MLM (comment out above, uncomment below):
# DEFAULT_MODEL_PATH="${PROJECT_ROOT}/Cache/Training/"
# MODEL_TYPE="perceiver_posonly_mlm"

# Allow overriding model path via first argument
MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"

# Task Configuration
# Default task is mrpc, can be overridden
TASK="${2:-mrpc}" 
# Task list: cola, mnli-matched, mnli-mismatched, mrpc, qnli, qqp, rte, sst2, stsb, wnli

# Tokenizer - use the tokenizer saved alongside the trained model (best match)
TOKENIZER_NAME="$MODEL_PATH"

echo "Configuration:"
echo "  - Project Root: $PROJECT_ROOT"
echo "  - HF Cache: $HF_HOME"
echo "  - Model Type: $MODEL_TYPE"
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

python training/evaluate_model_on_glue.py \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --task "$TASK" \
    --batch_size 96 \
    --epochs 20 \
    --learning_rate 1e-5 \
    --visualize \
    --save_model

echo ""
echo "GLUE evaluation completed!"
