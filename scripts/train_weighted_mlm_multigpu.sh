#!/bin/bash
# Bash script to train the weighted MLM model on multi-GPU Linux server
# Uses accelerate for distributed training on 4x RTX 3090 GPUs
#
# Usage:
#   bash scripts/train_weighted_mlm_multigpu.sh

set -e  # Exit on error

echo "=== Multi-GPU Training Script for Concept Encoder ==="
echo ""

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN  # Change to INFO for debugging
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8  # Adjust based on CPU cores available
# Add near the top of the environment variables section:
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NVIDIA_TF32_OVERRIDE=1

# Training configuration
MODEL_TYPE="weighted_mlm"
HIDDEN_SIZE=512
NUM_LAYERS=2
CONCEPT_NUM=128
INTERMEDIATE_SIZE=1024

# Data configuration
DATASET_NAME="Salesforce/wikitext"
DATASET_SUBSET="wikitext-103-raw-v1"
TOKENIZER_NAME="bert-base-cased"
MAX_SEQ_LENGTH=512
MLM_PROBABILITY=0.15
TEST_SIZE_PERCENT=0.1

# Training hyperparameters optimized for 4x RTX 3090 (24GB each)
PER_DEVICE_BATCH_SIZE=96        # 48 per GPU = 192 total
GRADIENT_ACCUMULATION_STEPS=1    # Effective batch = 192 * 2 = 384
LEARNING_RATE=5e-4
NUM_EPOCHS=5
WARMUP_STEPS=2000
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# Logging and evaluation
LOGGING_STEPS=500
EVAL_STRATEGY="steps"
EVAL_STEPS=2000
SAVE_STRATEGY="steps"
SAVE_STEPS=10000

# Paths (adjust these for your server)
OUTPUT_DIR="$HOME/dev/MrCogito/Cache/Training"
LOGGING_DIR="$HOME/dev/MrCogito/Cache/logs"

# Default HF dataset cache
DATASET_CACHE_DIR="${HF_DATASETS_CACHE:-.cache/huggingface/datasets}"

# Seeds for reproducibility
SEED=42

echo "Model Configuration:"
echo "  - Model Type: $MODEL_TYPE"
echo "  - Hidden Size: $HIDDEN_SIZE"
echo "  - Num Layers: $NUM_LAYERS"
echo "  - Concept Num: $CONCEPT_NUM"
echo "  - Intermediate Size: $INTERMEDIATE_SIZE"
echo ""
echo "Training Configuration:"
echo "  - Per-device Batch Size: $PER_DEVICE_BATCH_SIZE"
echo "  - Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  - Effective Batch Size: $((PER_DEVICE_BATCH_SIZE * 4 * GRADIENT_ACCUMULATION_STEPS))"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Epochs: $NUM_EPOCHS"
echo ""

# Create output directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGGING_DIR"
mkdir -p "$DATASET_CACHE_DIR"

# Launch training with accelerate
echo "Starting training..."
echo ""

accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --multi_gpu \
    training/mlm_training.py \
    --model_type "$MODEL_TYPE" \
    --hidden_size "$HIDDEN_SIZE" \
    --num_hidden_layers "$NUM_LAYERS" \
    --concept_num "$CONCEPT_NUM" \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --mlm_probability "$MLM_PROBABILITY" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --dataset_name "$DATASET_NAME" \
    --dataset_name_subset "$DATASET_SUBSET" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --test_size_percent "$TEST_SIZE_PERCENT" \
    --dataset_cache_dir "$DATASET_CACHE_DIR" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_BATCH_SIZE" \
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
    --lr_scheduler_type "linear" \
    --report_to "wandb" \
    --save_safetensors False \
    --overwrite_output_dir True \
    --remove_unused_columns True \
    --disable_tqdm False \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False

echo ""
echo "Training completed successfully!"
echo "Output saved to: $OUTPUT_DIR"
echo "Logs saved to: $LOGGING_DIR"
