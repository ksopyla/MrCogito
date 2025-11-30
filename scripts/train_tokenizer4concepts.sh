#!/bin/bash
# Script to train tokenizer on Polonez server with correct environment paths
# Run from project root: ./scripts/train_tokenizer_polonez.sh

set -e

# --- Configuration for Polonez/Odra ---
PROJECT_ROOT="/home/ksopyla/dev/MrCogito"

# If running locally on Windows/Other, fallback to current dir but warn
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Warning: Polonez project root not found. Using current directory."
    PROJECT_ROOT="$(pwd)"
else
    # Set HuggingFace cache directories to the dedicated storage
    # INFO: Ensure PROJECT_ROOT resides on the fast 2TB NVMe drive for best performance!
    export HF_HOME="${PROJECT_ROOT}/../hf_home"
    export HF_DATASETS_CACHE="${PROJECT_ROOT}/../hf_home/datasets"
    echo "Configured HF_HOME: $HF_HOME (Ensure this is on your fast NVMe!)"
fi

# Ensure python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify HF Login status
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "Error: You are not logged in to Hugging Face."
    echo "Please run 'huggingface-cli login' and paste your Write token."
    exit 1
else
    echo "HF Login verified: $(huggingface-cli whoami | head -n 1)"
fi

# Run the training script
# Uses 1M samples, creates 32k and 64k vocabs, pushes to Hub
echo "Starting Tokenizer Training..."
python training/train_tokenizer_custom.py \
    --dataset "JeanKaddour/minipile" \
    --sample_size 1000000 \
    --vocab_sizes 32000 64000 \
    --push_to_hub \
    --user_handle "ksopyla"

echo "Done!"

