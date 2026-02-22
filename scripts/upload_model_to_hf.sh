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

# Load HF_TOKEN from .env if present (so non-interactive SSH runs work)
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env" 2>/dev/null || true
    set +a
fi

# Also accept HUGGINGFACE_TOKEN as an alias for HF_TOKEN
if [ -z "$HF_TOKEN" ] && [ -n "$HUGGINGFACE_TOKEN" ]; then
    export HF_TOKEN="$HUGGINGFACE_TOKEN"
fi

# Check HF login: try token-based auth first, then cached credentials
if [ -n "$HF_TOKEN" ]; then
    if ! huggingface-cli whoami --token "$HF_TOKEN" > /dev/null 2>&1; then
        echo ""
        echo "HF_TOKEN is set but huggingface-cli whoami failed."
        echo "Check that the token is valid: https://huggingface.co/settings/tokens"
        echo ""
        exit 1
    fi
elif ! huggingface-cli whoami > /dev/null 2>&1; then
    echo ""
    echo "Hugging Face not logged in. Run: huggingface-cli login"
    echo "Or set HF_TOKEN in $PROJECT_ROOT/.env"
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
