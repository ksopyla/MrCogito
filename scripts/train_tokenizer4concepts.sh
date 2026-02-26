#!/bin/bash
# Script to train tokenizer on Polonez server with correct environment paths
# Run from project root: ./scripts/train_tokenizer4concepts.sh

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
    echo "Configured HF_HOME: $HF_HOME"
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

# 2. Run the training script
# Uses 100k samples, creates 32k/50k/64k vocabs
# Vocab sizes: 32768 (standard), 50368 (ModernBERT), 65536 (power of 2)
# Algorithm options: "unigram", "bpe", or "both" (for fair comparison)

# Option A: Train Unigram only (morphological awareness)
echo "Step 2a: Training Unigram Tokenizers..."
python training/train_tokenizer_custom.py \
    --dataset "JeanKaddour/minipile" \
    --sample_size 100000 \
    --vocab_sizes 32768 50368 65536 \
    --algorithm unigram \
    --push_to_hub \
    --user_handle "ksopyla"

# Option B: Train BPE only (industry standard for code)
echo "Step 2b: Training BPE Tokenizers..."
python training/train_tokenizer_custom.py \
    --dataset "JeanKaddour/minipile" \
    --sample_size 100000 \
    --vocab_sizes 32768 50368 65536 \
    --algorithm bpe \
    --push_to_hub \
    --user_handle "ksopyla"

# Option C: Train both (recommended for fair comparison)
# Uncomment to train both algorithms:
# echo "Step 2: Training Both Unigram and BPE Tokenizers..."
# python training/train_tokenizer_custom.py \
#     --dataset "JeanKaddour/minipile" \
#     --sample_size 100000 \
#     --vocab_sizes 32768 50368 65536 \
#     --algorithm both \
#     --push_to_hub \
#     --user_handle "ksopyla"

echo "Done!"
