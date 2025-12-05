#!/bin/bash
# Script to evaluate HuggingFace and Custom Tokenizers
# Run from project root: ./scripts/evaluate_tokenizers.sh

set -e

# --- Configuration ---
PROJECT_ROOT="/home/ksopyla/dev/MrCogito"

# If running locally on Windows/Other, fallback to current dir
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Warning: Polonez project root not found. Using current directory."
    PROJECT_ROOT="$(pwd)"
else
    # Set HuggingFace cache directories to the dedicated storage
    # INFO: Ensure PROJECT_ROOT resides on the fast 2TB NVMe drive for best performance!
    export HF_HOME="${PROJECT_ROOT}/../hf_home"
    export HF_DATASETS_CACHE="${PROJECT_ROOT}/../hf_home/datasets"
    echo "Configured HF_HOME: $HF_HOME"
fi

# Ensure python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify HF Login status (needed for Llama tokenizer)
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "Error: You are not logged in to Hugging Face."
    echo "Please run 'huggingface-cli login' and paste your Write token."
    exit 1
fi

# Run the evaluation script
# Default: 10k samples for compression, 10k for perplexity, 100 steps training
# Use --skip_ppl to skip slow perplexity training
echo "Starting Tokenizer Evaluation..."
python analysis/evaluate_tokenizers_comprehensive.py \
    --minipile_samples 50000 \
    "$@"

echo "Done!"

