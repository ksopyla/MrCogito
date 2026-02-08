#!/bin/bash
# Bash script to train Concept Encoder MLM models on multi-GPU Linux server
# Supports all 3 model types: weighted_mlm, perceiver_mlm, perceiver_posonly_mlm
# Uses accelerate for distributed training on multiple GPUs
#
# Usage:
#   # Edit MODEL_TYPE below, then run:
#   bash scripts/train_mlm_multigpu_perceiver.sh
#
#   # Or train all 3 sequentially:
#   for TYPE in weighted_mlm perceiver_posonly_mlm perceiver_mlm; do
#     sed -i "s/^MODEL_TYPE=.*/MODEL_TYPE=\"$TYPE\"/" scripts/train_mlm_multigpu_perceiver.sh
#     bash scripts/train_mlm_multigpu_perceiver.sh
#   done
#
# IMPORTANT: RTX 5090 / CUDA 12.8 COMPATIBILITY ISSUE
# ======================================================
# CUDA 12.8 has a KNOWN COMPILER BUG affecting RTX 5090 (SM120/Blackwell architecture)
# that causes "illegal memory access" errors during distributed training.
# This bug is FIXED in CUDA 12.9.1 or later.

set -e  # Exit on error

echo "=== Multi-GPU Training Script for Concept Encoder (Perceiver) ==="
echo ""

# Detect number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

# Set CUDA_VISIBLE_DEVICES to use all detected GPUs
if [ $NUM_GPUS -gt 0 ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
else
    echo "ERROR: No GPUs detected!"
    exit 1
fi
echo ""

# Set environment variables for optimal performance
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600  # 1 hour - increased for larger model + grad accum
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8

# Additional NCCL settings for stability
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NVIDIA_TF32_OVERRIDE=1

# =============================================================================
# SCALED-UP MODEL CONFIGURATION (v2, 2026-02-06)
# =============================================================================
# Previous config: H512, L2, C128, intermediate=1024 (~34-36M params)
# New config:      H512, L6, C128, intermediate=2048 (~58-61M params)
#
# Key changes from v1:
#   - Depth: 2 -> 6 layers (most impactful, enables deeper syntactic learning)
#   - FFN: 1024 -> 2048 (more capacity per layer)
#   - Epochs: 20 -> 30 (more pretraining to compensate for small dataset)
#   - LR: 5e-4 -> 3e-4 (lower for larger model stability)
#   - Warmup: 2000 -> 3000 (longer warmup for deeper model)
#   - Batch: 48 per device (reduced from 64 for memory)
#   - Grad accum: 2 (effective batch = 48*NUM_GPUs*2)
#
# Estimated params: ~85M (weighted), ~88M (perceiver variants)
# Estimated training time: ~36-48h on 4x RTX 3090 for 30 epochs
# Target MLM loss: < 3.0 (vs 4.0 for L2 models)
#
# MODEL_TYPE options (change this for each training run):
#   - "weighted_mlm": Weighted concept combination decoder
#   - "perceiver_mlm": Perceiver IO with Input+Position decoder queries
#   - "perceiver_posonly_mlm": Perceiver IO with Position-only queries (pure Perceiver IO)
#
# Run all 3 models sequentially:
#   for TYPE in weighted_mlm perceiver_posonly_mlm perceiver_mlm; do
#     sed -i "s/^MODEL_TYPE=.*/MODEL_TYPE=\"$TYPE\"/" scripts/train_mlm_multigpu_perceiver.sh
#     bash scripts/train_mlm_multigpu_perceiver.sh
#   done
# =============================================================================

# --- Model Architecture ---
#MODEL_TYPE="weighted_mlm"        # DONE 2026-02-08, 10h09m, best loss 3.415
MODEL_TYPE="perceiver_posonly_mlm" # NEXT
#MODEL_TYPE="perceiver_mlm"
HIDDEN_SIZE=512
NUM_LAYERS=6                      # Scaled from 2 -> 6 (key change)
CONCEPT_NUM=128
INTERMEDIATE_SIZE=2048            # Scaled from 1024 -> 2048

# --- Data Configuration ---
DATASET_NAME="JeanKaddour/minipile"
DATASET_SUBSET="" 
TOKENIZER_NAME="answerdotai/ModernBERT-base"
MAX_SEQ_LENGTH=512
MLM_PROBABILITY=0.15
TEST_SIZE_PERCENT=0.1

# --- Training Hyperparameters ---
# Memory: ~61-83M params, fits RTX 3090 (24GB) with batch=64 and NO gradient checkpointing
# Effective batch: 64 * 4 GPUs * 2 accum = 512 (good for MLM pretraining)
# For OOM: reduce to PER_DEVICE_BATCH_SIZE=48 (effective batch = 384)
PER_DEVICE_BATCH_SIZE=64
EVAL_BATCH_SIZE=16              # Eval computes full logits [B, L, V], keep smaller
GRADIENT_ACCUMULATION_STEPS=2   # Effective batch = 64 * NUM_GPUs * 2
LEARNING_RATE=3e-4              # Lower than v1 (5e-4) for deeper model stability
NUM_EPOCHS=40                   # Increased from 20 for more pretraining
WARMUP_STEPS=3000               # ~3.8% of total steps, standard range
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# --- Loss Configuration ---
# Options for concept_losses: orthogonality, soft_orthogonality, uniformity, vicreg, combined, none
# Options for loss_weighting: fixed, learnable, kendall_gal
CONCEPT_LOSSES="none"
LOSS_WEIGHTING="kendall_gal"
LOSS_WEIGHT=0.1  # Only used with loss_weighting=fixed

# --- Logging and Evaluation ---
LOGGING_STEPS=1000              # More frequent logging for longer training
EVAL_STRATEGY="steps"
EVAL_STEPS=5000
SAVE_STRATEGY="steps"
SAVE_STEPS=50000                # Save less frequently (models are larger)

# Paths are dependent on the server setup:
# runpod: /workspace/MrCogito
# odra: $HOME/dev/MrCogito
# polone: $HOME/dev/MrCogito
PROJECT_ROOT="/home/ksopyla/dev/MrCogito"

OUTPUT_DIR="$PROJECT_ROOT/Cache/Training"
LOGGING_DIR="$PROJECT_ROOT/Cache/logs"

# Optional: Set HF_HOME and HF_DATASETS_CACHE 
export HF_HOME="${PROJECT_ROOT}/../hf_home"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/../hf_home/datasets"

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
echo "  - Number of GPUs: $NUM_GPUS"
echo "  - Per-device Batch Size: $PER_DEVICE_BATCH_SIZE"
echo "  - Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  - Effective Batch Size: $((PER_DEVICE_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
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
    --num_processes=$NUM_GPUS \
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
    --concept_losses "$CONCEPT_LOSSES" \
    --loss_weighting "$LOSS_WEIGHTING" \
    --loss_weight "$LOSS_WEIGHT" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-$PER_DEVICE_BATCH_SIZE}" \
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
    --dataloader_num_workers 2 \
    --gradient_checkpointing False \
    --optim "adamw_torch_fused" \
    --lr_scheduler_type "linear" \
    --report_to "wandb" \
    --save_safetensors True \
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
