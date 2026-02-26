#!/bin/bash
# Train Recursive Concept Encoder (TRM-style weight-tied) on multi-GPU Linux server.
#
# This trains recursive_mlm: 1 shared ConceptEncoderLayer applied K times,
# with the same Perceiver IO decoder as perceiver_mlm.
#
# Key differences from train_mlm_multigpu_perceiver.sh:
#   - MODEL_TYPE="recursive_mlm" (uses RecursiveConceptEncoderForMaskedLM)
#   - ~42M params instead of ~61M (47% fewer encoder params)
#   - Same decoder, same loss manager, same training pipeline
#   - num_hidden_layers controls how many iterations the shared layer is applied
#
# Usage:
#   bash scripts/train_recursive_mlm.sh
#
# Warm-start from standard perceiver_mlm:
#   Edit MODEL_NAME_OR_PATH below to point to an existing checkpoint.
#   The script loads layer-0 weights into the shared layer, skips layers 1-N.

set -e

echo "=== Recursive Concept Encoder MLM Training ==="
echo ""

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

if [ $NUM_GPUS -gt 0 ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
else
    echo "ERROR: No GPUs detected!"
    exit 1
fi
echo ""

export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NVIDIA_TF32_OVERRIDE=1

# =============================================================================
# RECURSIVE MODEL CONFIGURATION
# =============================================================================
# Same architecture dimensions as standard L6, but the encoder uses 1 shared
# layer applied NUM_LAYERS times. The decoder is NOT shared.
#
# Parameter comparison (H512, L6/K6, C128, intermediate=2048):
#   perceiver_mlm:   encoder 37.6M + decoder ~24M = ~61M total
#   recursive_mlm:   encoder  6.3M + decoder ~24M = ~42M total  (-31%)
# =============================================================================

MODEL_TYPE="recursive_mlm"
HIDDEN_SIZE=512
TOKEN_EMBEDDING_DIM=0
NUM_LAYERS=6                      # Number of iterations (K) for the shared layer
CONCEPT_NUM=128
INTERMEDIATE_SIZE=2048
CONCEPT_POSITION_TYPE="none"

# Warm-start: uncomment to load from existing perceiver_mlm checkpoint
# The recursive model loads encoder.layers.0.* → encoder.shared_layer.*
# MODEL_NAME_OR_PATH=""

# --- Data ---
DATASET_NAME="JeanKaddour/minipile"
DATASET_SUBSET=""
TOKENIZER_NAME="answerdotai/ModernBERT-base"
MAX_SEQ_LENGTH=512
MLM_PROBABILITY=0.15
TEST_SIZE_PERCENT=0.1

# --- Training ---
PER_DEVICE_BATCH_SIZE=64
EVAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=2e-4                # Slightly lower than standard 3e-4 for gradient accumulation through K iters
NUM_EPOCHS=20
WARMUP_STEPS=1500
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# --- Loss ---
CONCEPT_LOSSES="combined"
LOSS_WEIGHTING="fixed"
LOSS_WEIGHT=0.1

# --- Compile ---
TORCH_COMPILE_DYNAMIC=False

# --- Logging ---
LOGGING_STEPS=1000
EVAL_STRATEGY="steps"
EVAL_STEPS=5000
SAVE_STRATEGY="steps"
SAVE_STEPS=5000

# --- Paths ---
PROJECT_ROOT="/home/ksopyla/dev/MrCogito"
OUTPUT_DIR="$PROJECT_ROOT/Cache/Training"
LOGGING_DIR="$PROJECT_ROOT/Cache/logs"
export LOG_DIR="$LOGGING_DIR"
export HF_HOME="${PROJECT_ROOT}/../hf_home"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/../hf_home/datasets"
DATASET_CACHE_DIR="${HF_DATASETS_CACHE}"
SEED=42

echo "Model Configuration:"
echo "  - Model Type: $MODEL_TYPE (1 shared layer, $NUM_LAYERS iterations)"
echo "  - Hidden Size: $HIDDEN_SIZE"
echo "  - Concept Num: $CONCEPT_NUM"
echo "  - Intermediate Size: $INTERMEDIATE_SIZE"
echo ""
echo "Training Configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Per-device Batch: $PER_DEVICE_BATCH_SIZE"
echo "  - Effective Batch: $((PER_DEVICE_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
echo "  - LR: $LEARNING_RATE"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Concept Losses: $CONCEPT_LOSSES ($LOSS_WEIGHTING, weight=$LOSS_WEIGHT)"
echo ""

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR" "$DATASET_CACHE_DIR"

SHELL_LOG="${LOGGING_DIR}/shell_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).log"
echo "Starting training... (log: $SHELL_LOG)"
echo ""

# Build optional args
OPTIONAL_ARGS=""
if [ -n "$MODEL_NAME_OR_PATH" ]; then
    OPTIONAL_ARGS="--model_name_or_path $MODEL_NAME_OR_PATH"
    echo "Warm-starting from: $MODEL_NAME_OR_PATH"
fi

accelerate launch \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --multi_gpu \
    training/mlm_training.py \
    --model_type "$MODEL_TYPE" \
    --hidden_size "$HIDDEN_SIZE" \
    --token_embedding_dim "$TOKEN_EMBEDDING_DIM" \
    --num_hidden_layers "$NUM_LAYERS" \
    --concept_num "$CONCEPT_NUM" \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --concept_position_type "$CONCEPT_POSITION_TYPE" \
    --mlm_probability "$MLM_PROBABILITY" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --dataset_name "$DATASET_NAME" \
    --dataset_name_subset "$DATASET_SUBSET" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --test_size_percent "$TEST_SIZE_PERCENT" \
    --dataset_cache_dir "$DATASET_CACHE_DIR" \
    --concept_losses "$CONCEPT_LOSSES" \
    --loss_weighting "$LOSS_WEIGHTING" \
    --loss_weight "$LOSS_WEIGHT" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --warmup_steps "$WARMUP_STEPS" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --logging_steps "$LOGGING_STEPS" \
    --eval_strategy "$EVAL_STRATEGY" \
    --eval_steps "$EVAL_STEPS" \
    --save_strategy "$SAVE_STRATEGY" \
    --save_steps "$SAVE_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$LOGGING_DIR" \
    --seed "$SEED" \
    --bf16 \
    --ddp_backend "nccl" \
    --ddp_find_unused_parameters False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
    --gradient_checkpointing False \
    --optim "adamw_torch_fused" \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --save_safetensors False \
    --overwrite_output_dir True \
    --remove_unused_columns True \
    --disable_tqdm True \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --torch_compile False \
    --torch_compile_dynamic "$TORCH_COMPILE_DYNAMIC" \
    $OPTIONAL_ARGS \
    2>&1 | python scripts/clean_tee.py "$SHELL_LOG"

echo ""
echo "Recursive MLM training completed!"
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOGGING_DIR"
