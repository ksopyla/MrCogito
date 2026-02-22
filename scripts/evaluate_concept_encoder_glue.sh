#!/bin/bash
# Evaluate the Concept Encoder model on the GLUE benchmark.
#
# MODEL_PATH is set directly in this file — update it when a new model is trained.
# MODEL_TYPE is auto-detected from MODEL_PATH (no need to change it manually).
#
# Usage:
#   bash scripts/evaluate_concept_encoder_glue.sh            # all semantic tasks (default)
#   bash scripts/evaluate_concept_encoder_glue.sh all        # all semantic tasks
#   bash scripts/evaluate_concept_encoder_glue.sh all-glue   # all GLUE tasks
#   bash scripts/evaluate_concept_encoder_glue.sh mrpc       # single task
#
# Task list: all, all-glue, cola, mrpc, stsb, sst2, qnli, qqp, rte, mnli-matched, mnli-mismatched
#
# Model history:
#   perceiver_mlm L6 + combined+kendall_gal (Feb 19 2026, eff. rank 95.5%):
#     perceiver_mlm_H512L6C128_20260219_105435  ← CURRENT
#   perceiver_mlm L6 baseline (Feb 08 2026, eff. rank 4%):
#     perceiver_mlm_H512L6C128_20260208_211633
#   weighted_mlm L2 (Jan 17 2026, MRPC 82.2% F1):
#     weighted_mlm_H512L2C128_20260117_153544

set -o pipefail  # Catch errors in piped commands

echo "=== GLUE Evaluation Script for Concept Encoder ==="

# Initialize pyenv/poetry PATH for non-interactive SSH sessions
if [ -d "$HOME/.pyenv" ]; then
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
    eval "$(pyenv init - 2>/dev/null)" || true
fi
if [ -d "$HOME/.local/share/pypoetry" ]; then
    export PATH="$HOME/.local/share/pypoetry/venv/bin:$PATH"
fi

# Load .env for HF_TOKEN (enables HF Hub model download without manual login)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$(dirname "$SCRIPT_DIR")/.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE" 2>/dev/null || true; set +a
fi

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

# Suppress tokenizer parallelism fork warning (harmless, but noisy)
export TOKENIZERS_PARALLELISM=false

# Model Configuration - set both together!
#
# MODEL_TYPE options for concept encoders:
#   - "weighted_mlm": Weighted attention pooling approach
#   - "perceiver_mlm": Perceiver IO with Input+Position decoder queries
#   - "perceiver_posonly_mlm": Perceiver IO with Position-only decoder queries (pure Perceiver IO)
#
# Note: perceiver_mlm and perceiver_posonly_mlm use the same classification head
# (the difference is only in how the MLM decoder works during pretraining)

# =============================================================================
# MODEL TO EVALUATE — update this when a new model is trained
# Can be overridden non-interactively via environment variables:
#   MODEL_PATH_OVERRIDE="ksopyla/concept-encoder-..." MODEL_TYPE_OVERRIDE="perceiver_decoder_cls" bash ...
# =============================================================================

# Default model (last trained canonical run)
# perceiver_mlm L6 + fixed0.1 combined (Feb 20 2026, eff. rank 12.5%)
DEFAULT_MODEL_PATH="${PROJECT_ROOT}/Cache/Training/perceiver_mlm_H512L6C128_20260220_184029/perceiver_mlm_H512L6C128_20260220_184029"

# Previous models (uncomment to set a different default):
# perceiver_mlm L6 baseline — no concept losses (eff. rank 4%) — uploaded to HF Hub
# DEFAULT_MODEL_PATH="${PROJECT_ROOT}/Cache/Training/perceiver_mlm_H512L6C128_20260208_211633/perceiver_mlm_H512L6C128_20260208_211633"
# perceiver_mlm L6 + combined+kendall_gal (Feb 19 2026, eff. rank 95.5%, QQP/MNLI regressed)
# DEFAULT_MODEL_PATH="${PROJECT_ROOT}/Cache/Training/perceiver_mlm_H512L6C128_20260219_105435/perceiver_mlm_H512L6C128_20260219_105435"
# HF Hub model (auto-downloads to Cache/Models on first use):
# DEFAULT_MODEL_PATH="ksopyla/concept-encoder-perceiver_mlm_H512L6C128_20260208_211633"

# Apply env-var overrides (for non-interactive SSH / CI use)
MODEL_PATH="${MODEL_PATH_OVERRIDE:-$DEFAULT_MODEL_PATH}"

# =============================================================================

# Task: optional $1, defaults to "all" (semantic-relevant subset)
# Task list: all, all-glue, cola, mrpc, stsb, sst2, qnli, qqp, rte, mnli-matched, mnli-mismatched
TASK="${1:-all}"

# Auto-detect MODEL_TYPE from MODEL_PATH name.
# MODEL_TYPE_OVERRIDE skips auto-detection (required for perceiver_decoder_cls
# and HF Hub model IDs that don't embed the type in the path).
if [ -n "$MODEL_TYPE_OVERRIDE" ]; then
    MODEL_TYPE="$MODEL_TYPE_OVERRIDE"
    echo "MODEL_TYPE overridden: $MODEL_TYPE"
elif echo "$MODEL_PATH" | grep -q "perceiver_posonly_mlm"; then
    MODEL_TYPE="perceiver_posonly_mlm"
elif echo "$MODEL_PATH" | grep -q "perceiver_mlm"; then
    MODEL_TYPE="perceiver_mlm"
elif echo "$MODEL_PATH" | grep -q "weighted_mlm"; then
    MODEL_TYPE="weighted_mlm"
else
    MODEL_TYPE="perceiver_mlm"
    echo "WARNING: Could not auto-detect MODEL_TYPE from path. Defaulting to: $MODEL_TYPE"
    echo "  Use MODEL_TYPE_OVERRIDE env var to set explicitly."
fi

# Tokenizer: use the model path itself (works for both local and HF Hub IDs)
TOKENIZER_NAME="${TOKENIZER_NAME_OVERRIDE:-$MODEL_PATH}"

# Task-specific epoch count
# Standard GLUE fine-tuning: fewer epochs for larger datasets
# Small (< 10k): 20 epochs - cola, mrpc, stsb, rte, wnli
# Medium (10k-100k): 5 epochs - sst2
# Large (> 100k): 3 epochs - qnli, qqp, mnli-matched, mnli-mismatched
get_task_epochs() {
    local task=$1
    case $task in
        sst2)       echo 5 ;;
        qnli)       echo 3 ;;
        qqp)        echo 3 ;;
        mnli-matched)    echo 3 ;;
        mnli-mismatched) echo 3 ;;
        *)          echo 20 ;;  # cola, mrpc, stsb, rte, wnli
    esac
}

echo "Configuration:"
echo "  - Project Root: $PROJECT_ROOT"
echo "  - HF Cache: $HF_HOME"
echo "  - Model Type: $MODEL_TYPE (auto-detected)"
echo "  - Model Path: $MODEL_PATH"
echo "  - Task: $TASK"
echo "  - Tokenizer: $TOKENIZER_NAME"
echo ""

# Check if model path exists
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "WARNING: Model path does not seem to exist or contain config.json: $MODEL_PATH"
    echo "Please check the path."
fi

# --- Evaluation Function ---
run_single_task() {
    local task=$1
    local epochs=$(get_task_epochs "$task")

    echo "------------------------------------------------------------"
    echo "  Task: $task | Epochs: $epochs | Model: $MODEL_TYPE"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"

    if command -v poetry > /dev/null 2>&1; then
        PYTHON_CMD="poetry run python"
    elif command -v python3 > /dev/null 2>&1; then
        PYTHON_CMD="python3"
    else
        echo "ERROR: neither poetry nor python3 found in PATH"; return 1
    fi

    $PYTHON_CMD evaluation/evaluate_model_on_glue.py \
        --model_type "$MODEL_TYPE" \
        --model_name_or_path "$MODEL_PATH" \
        --tokenizer_name "$TOKENIZER_NAME" \
        --task "$task" \
        --batch_size 96 \
        --epochs "$epochs" \
        --learning_rate 1e-5 \
        --visualize \
        --save_model

    echo ""
    echo "  Completed: $task at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

# --- Execute ---
if [ "$TASK" = "all-glue" ]; then
    # Run ALL GLUE tasks (WNLI excluded - unreliable)
    ALL_TASKS=("cola" "rte" "mrpc" "stsb" "sst2" "qnli" "qqp" "mnli-matched" "mnli-mismatched")
    TOTAL=${#ALL_TASKS[@]}
elif [ "$TASK" = "all" ]; then
    # Concept-relevant tasks only (skip CoLA/RTE/SST-2 — architectural ceiling or low signal)
    # MRPC, QQP: semantic similarity (concept strength)
    # STS-B: continuous similarity regression (direct concept quality measure)
    # MNLI: compositional entailment (tests if concepts preserve meaning)
    ALL_TASKS=("mrpc" "stsb" "qqp" "mnli-matched" "mnli-mismatched")
    TOTAL=${#ALL_TASKS[@]}
    SUCCEEDED=0
    FAILED=0
    FAILED_TASKS=()

    echo "Running ALL GLUE tasks ($TOTAL tasks) for model: $MODEL_TYPE"
    echo ""
    START_TIME=$(date +%s)

    for i in "${!ALL_TASKS[@]}"; do
        task="${ALL_TASKS[$i]}"
        echo "============================================================"
        echo "  [$((i+1))/$TOTAL] $task"
        echo "============================================================"

        if run_single_task "$task"; then
            SUCCEEDED=$((SUCCEEDED + 1))
        else
            FAILED=$((FAILED + 1))
            FAILED_TASKS+=("$task")
            echo "  FAILED: $task - continuing with next task..."
            echo ""
        fi
    done

    END_TIME=$(date +%s)
    DURATION=$(( (END_TIME - START_TIME) / 60 ))

    echo "============================================================"
    echo "  ALL TASKS COMPLETE"
    echo "============================================================"
    echo "  Model:     $MODEL_TYPE"
    echo "  Succeeded: $SUCCEEDED/$TOTAL"
    echo "  Failed:    $FAILED/$TOTAL"
    echo "  Duration:  ${DURATION} minutes"
    if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
        echo "  Failed tasks: ${FAILED_TASKS[*]}"
    fi
    echo "  Wandb: https://wandb.ai/ksopyla/MrCogito"
    echo "============================================================"
else
    echo "Starting evaluation..."
    run_single_task "$TASK"
    echo "GLUE evaluation completed!"
fi
