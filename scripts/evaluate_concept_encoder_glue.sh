#!/bin/bash
# Evaluate the Concept Encoder model on the GLUE benchmark.
#
# MODEL_PATH is set directly in this file — update it when a new denoising checkpoint is trained.
# MODEL_TYPE is explicit; no path-based auto-detection remains.
#
# Usage:
#   bash scripts/evaluate_concept_encoder_glue.sh            # all semantic tasks (default)
#   bash scripts/evaluate_concept_encoder_glue.sh all        # all semantic tasks
#   bash scripts/evaluate_concept_encoder_glue.sh all-glue   # all GLUE tasks
#   bash scripts/evaluate_concept_encoder_glue.sh mrpc       # single task
#
# Task list: all, all-glue, cola, mrpc, stsb, sst2, qnli, qqp, rte, mnli-matched, mnli-mismatched
#
# Recommended order for a new perceiver checkpoint:
#   1. analysis/run_concept_analysis.py
#   2. evaluation/evaluate_on_benchmark.py --benchmark stsb_zero_shot
#   3. this GLUE script for MRPC / STS-B / QQP / MNLI

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
# MODEL_TYPE options for maintained concept encoders:
#   - "perceiver_denoise": canonical denoising perceiver stack
#   - "diffusion_mlm"
#   - "prefix_diffusion"
#   - "weighted_mlm"

# =============================================================================
# MODEL TO EVALUATE — update this when a new model is trained
# Can be overridden non-interactively via environment variables:
#   MODEL_PATH_OVERRIDE="ksopyla/concept-encoder-..." MODEL_TYPE_OVERRIDE="perceiver_denoise" bash ...
# =============================================================================

# Default model path placeholder — replace with a fresh perceiver denoising checkpoint.
DEFAULT_MODEL_PATH="${PROJECT_ROOT}/Cache/Training/REPLACE_WITH_PERCEIVER_DENOISE_CHECKPOINT"

MODEL_PATH="${MODEL_PATH_OVERRIDE:-$DEFAULT_MODEL_PATH}"
MODEL_TYPE="${MODEL_TYPE_OVERRIDE:-perceiver_denoise}"

# =============================================================================

# Task: optional $1, defaults to "all" (semantic-relevant subset)
# Task list: all, all-glue, cola, mrpc, stsb, sst2, qnli, qqp, rte, mnli-matched, mnli-mismatched
TASK="${1:-all}"

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
