#!/bin/bash
# Script to pretrain Concept Encoder (Perceiver) on Minipile dataset
# Aligned with train_perceiver_mlm_multigpu.sh for Polonez/Odra server
# Run from project root: ./scripts/train_perceiver_minipile.sh

set -e  # Exit on error

echo "=== Multi-GPU Pretraining Script for Concept Encoder (Perceiver) on Minipile ==="
echo ""

# --- Environment Setup ---
# Detect number of available GPUs
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

# Set environment variables for optimal performance (Polonez specific)
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8

# Additional NCCL settings for stability (Matching reference script)
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NVIDIA_TF32_OVERRIDE=1

# --- Project & Data Paths ---
# Adapting to Odra/Polonez server path structure
if [ -d "/home/ksopyla/dev/MrCogito" ]; then
    PROJECT_ROOT="/home/ksopyla/dev/MrCogito"
    export HF_HOME="${PROJECT_ROOT}/../hf_home"
    export HF_DATASETS_CACHE="${PROJECT_ROOT}/../hf_home/datasets"
    echo "Running on Server: Configured HF Cache at $HF_HOME"
else
    PROJECT_ROOT="$(pwd)"
    echo "Running Locally: Using current directory as root"
fi

export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# --- Training Configuration ---
MODEL_TYPE="perceiver_mlm"
DATASET="JeanKaddour/minipile"
# Using the newly trained robust tokenizer
TOKENIZER="ksopyla/minipile-english-unigram-32k"
SEQ_LEN=512

# Model Params
CONCEPT_NUM=128
HIDDEN_SIZE=512
LAYERS=4

# Hyperparameters (Adjusted for Multi-GPU)
# Total Batch = PER_DEVICE * NUM_GPUS * GRAD_ACC
# Target: ~128-256 effective batch size
PER_DEVICE_BATCH_SIZE=32 
GRAD_ACC=1   # With 4 GPUs, 32*4 = 128 effective. 
LR=1e-4      # Perceiver usually needs stable LR
EPOCHS=2
WARMUP_STEPS=2000
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# Output Paths (Consistent with other scripts)
OUTPUT_DIR="${PROJECT_ROOT}/Cache/Training/minipile_perceiver_H${HIDDEN_SIZE}C${CONCEPT_NUM}"
LOGGING_DIR="${PROJECT_ROOT}/Cache/logs"
DATASET_CACHE_DIR="${HF_DATASETS_CACHE:-.cache/huggingface/datasets}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGGING_DIR"

echo "Configuration:"
echo "  - Model: $MODEL_TYPE"
echo "  - Dataset: $DATASET"
echo "  - Tokenizer: $TOKENIZER"
echo "  - GPUs: $NUM_GPUS"
echo "  - Per Device Batch: $PER_DEVICE_BATCH_SIZE"
echo "  - Effective Batch: $((PER_DEVICE_BATCH_SIZE * NUM_GPUS * GRAD_ACC))"
echo "  - Output Dir: $OUTPUT_DIR"
echo ""

# --- Run Training with Accelerate ---
echo "Starting training..."

# Using accelerate for distributed training
# Note: --mixed_precision=bf16 is preferred for Ampere GPUs (RTX 3090)
accelerate launch \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --multi_gpu \
    training/mlm_training.py \
    --model_type "$MODEL_TYPE" \
    --dataset_name "$DATASET" \
    --tokenizer_name "$TOKENIZER" \
    --max_seq_length "$SEQ_LEN" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --learning_rate "$LR" \
    --num_train_epochs "$EPOCHS" \
    --concept_num "$CONCEPT_NUM" \
    --hidden_size "$HIDDEN_SIZE" \
    --num_hidden_layers "$LAYERS" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$LOGGING_DIR" \
    --dataset_cache_dir "$DATASET_CACHE_DIR" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --eval_strategy "steps" \
    --eval_steps 5000 \
    --logging_steps 100 \
    --warmup_steps "$WARMUP_STEPS" \
    --mlm_probability 0.15 \
    --bf16 \
    --ddp_backend "nccl" \
    --ddp_find_unused_parameters False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
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
    --greater_is_better False \
    --seed 42

echo ""
echo "Training completed successfully!"
echo "Output saved to: $OUTPUT_DIR"
