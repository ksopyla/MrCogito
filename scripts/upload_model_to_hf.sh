#!/bin/bash
# Upload Concept Encoder trained models to Hugging Face Hub
# Run this script on Polonez or Odra after training and evaluation.
#
# Usage:
#   bash scripts/upload_model_to_hf.sh
#
#   # List models only (no upload prompt):
#   bash scripts/upload_model_to_hf.sh --list-only
#
# Prerequisites:
#   - Models in Cache/Training
#   - Evaluation reports in Cache/Evaluation_reports
#   - Hugging Face: run 'huggingface-cli login' once
#
# The script will show a numbered list of models with evaluation metrics.
# Enter the model number to upload it to your HF space (ksopyla/concept-encoder-*).

set -e

# --- Configuration ---
PROJECT_ROOT="/home/ksopyla/dev/MrCogito"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Project root not found: $PROJECT_ROOT"
    echo "Using current directory: $(pwd)"
    PROJECT_ROOT="$(pwd)"
fi

cd "$PROJECT_ROOT"

# Check HF login
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo ""
    echo "Hugging Face not logged in. Run: huggingface-cli login"
    echo "Then paste your Write token from https://huggingface.co/settings/tokens"
    echo ""
    exit 1
fi

echo ""
echo "=== MrCogito: Upload Model to Hugging Face ==="
echo "  Project: $PROJECT_ROOT"
echo "  Host:    $(hostname)"
echo ""

# Run the Python upload script (passes --list-only, --training-dir, etc.)
if command -v poetry > /dev/null 2>&1; then
    poetry run python scripts/upload_model_to_hf.py "$@"
else
    python scripts/upload_model_to_hf.py "$@"
fi
