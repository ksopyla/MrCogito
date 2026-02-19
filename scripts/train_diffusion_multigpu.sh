#!/bin/bash
# Multi-GPU training script for Concept Encoder + Masked Diffusion Decoder
# Run on Polonez (4x RTX 3090) or Odra (3x RTX 3090)
#
# Training objective: Masked Discrete Diffusion
#   - Noise level t ~ Uniform(t_min=0.05, 1.0) sampled per batch
#   - Each token masked independently with probability t
#   - Model predicts clean tokens at masked positions using concepts + context
#   - Variable mask rate forces the encoder to build richer concept representations
#     (vs MLM's fixed 15% which allows local-context shortcuts)
#
# Usage:
#   bash scripts/train_diffusion_multigpu.sh
#
# Architecture comparison:
#   MLM model (mlm_training.py):
#     Encoder: ConceptEncoder L6  + Perceiver decoder (1 cross-attn layer)
#     Loss:    fixed 15% masking, sparse CE
#   Diffusion model (train_diffusion.py):
#     Encoder: ConceptEncoder L6  (same encoder — reuse MLM pretrained weights!)
#     Decoder: ConceptDiffusionDecoder (D=2 transformer layers, concept cross-attn)
#     Loss:    variable t% masking, CE over all masked positions

set -e

echo "=== Multi-GPU Training: Concept Encoder + Masked Diffusion ==="
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

# Performance environment
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NVIDIA_TF32_OVERRIDE=1

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
# Encoder is identical to the MLM models (H512 L6 C128).
# You can warm-start from an existing MLM checkpoint (set MODEL_NAME_OR_PATH).
HIDDEN_SIZE=512
TOKEN_EMBEDDING_DIM=0         # 0 = same as HIDDEN_SIZE
NUM_ENCODER_LAYERS=6          # Same depth as best MLM model
CONCEPT_NUM=128
INTERMEDIATE_SIZE=2048
CONCEPT_POSITION_TYPE="none"
DECODER_LAYERS=2              # Diffusion decoder layers (keep 1-4, larger = slower)
T_MIN=0.05                    # Minimum masking rate (avoids trivial t≈0 batches)

# =============================================================================
# DATA
# =============================================================================
DATASET_NAME="JeanKaddour/minipile"
DATASET_SUBSET=""
TOKENIZER_NAME="answerdotai/ModernBERT-base"
MAX_SEQ_LENGTH=512
TEST_SIZE_PERCENT=0.1

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
# Memory: similar to MLM perceiver (~88M params)
# Effective batch: 64 * NUM_GPUS * 2 = 512
PER_DEVICE_BATCH_SIZE=64
EVAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=3e-4
NUM_EPOCHS=20
WARMUP_STEPS=1500
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# =============================================================================
# CONCEPT LOSSES
# =============================================================================
# "combined" = variance + uniformity (recommended)
# "t_regs_mst" = MST-based uniformity (best at detecting dimensional collapse)
# Combine both: CONCEPT_LOSSES="combined t_regs_mst"
CONCEPT_LOSSES="combined"
LOSS_WEIGHTING="kendall_gal"
LOSS_WEIGHT=0.1

# =============================================================================
# TORCH COMPILE
# =============================================================================
# Diffusion training does NOT have the variable-shape issue of MLM because
# masking happens inside model.forward() uniformly, so all tensors have fixed
# shape [B, L] at the collator level.  However, the sparse CE loss inside
# forward() still has variable M (masked count).  Use dynamic=True to be safe.
TORCH_COMPILE_DYNAMIC=False   # Enable once you verify training is stable

# =============================================================================
# LOGGING
# =============================================================================
LOGGING_STEPS=500
EVAL_STRATEGY="steps"
EVAL_STEPS=2500
SAVE_STRATEGY="steps"
SAVE_STEPS=10000
SEED=42

# =============================================================================
# PATHS (update for your server)
# runpod:  /workspace/MrCogito        (ssh root@<pod-ip> -p <pod-port>)
# odra:    /home/ksopyla/dev/MrCogito  (ssh odra    — see .cursor/rules/computing-environments-remote.mdc)
# polonez: /home/ksopyla/dev/MrCogito  (ssh polonez — see .cursor/rules/computing-environments-remote.mdc)
# =============================================================================
PROJECT_ROOT="/home/ksopyla/dev/MrCogito"
OUTPUT_DIR="$PROJECT_ROOT/Cache/Training"
LOGGING_DIR="$PROJECT_ROOT/Cache/logs"
export LOG_DIR="$LOGGING_DIR"
export HF_HOME="${PROJECT_ROOT}/../hf_home"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/../hf_home/datasets"
DATASET_CACHE_DIR="${HF_DATASETS_CACHE:-.cache/huggingface/datasets}"

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR" "$DATASET_CACHE_DIR"

echo "Model: ConceptEncoder-H${HIDDEN_SIZE}L${NUM_ENCODER_LAYERS}C${CONCEPT_NUM} + DiffusionDecoder-D${DECODER_LAYERS}"
echo "Dataset: $DATASET_NAME"
echo "Effective batch: $((PER_DEVICE_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
echo "Concept losses: $CONCEPT_LOSSES (weighting: $LOSS_WEIGHTING)"
echo ""

SHELL_LOG="${LOGGING_DIR}/shell_diffusion_$(date +%Y%m%d_%H%M%S).log"
echo "Starting training... (log: $SHELL_LOG)"

accelerate launch \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --multi_gpu \
    training/train_diffusion.py \
    --hidden_size "$HIDDEN_SIZE" \
    --token_embedding_dim "$TOKEN_EMBEDDING_DIM" \
    --num_hidden_layers "$NUM_ENCODER_LAYERS" \
    --concept_num "$CONCEPT_NUM" \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --concept_position_type "$CONCEPT_POSITION_TYPE" \
    --decoder_layers "$DECODER_LAYERS" \
    --t_min "$T_MIN" \
    --torch_compile_dynamic "$TORCH_COMPILE_DYNAMIC" \
    --dataset_name "$DATASET_NAME" \
    --dataset_name_subset "$DATASET_SUBSET" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --test_size_percent "$TEST_SIZE_PERCENT" \
    --dataset_cache_dir "$DATASET_CACHE_DIR" \
    --concept_losses "$CONCEPT_LOSSES" \
    --loss_weighting "$LOSS_WEIGHTING" \
    --loss_weight "$LOSS_WEIGHT" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
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
    --torch_compile False \
    --ddp_backend "nccl" \
    --ddp_find_unused_parameters False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
    --gradient_checkpointing False \
    --optim "adamw_torch_fused" \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --save_safetensors True \
    --overwrite_output_dir True \
    --remove_unused_columns False \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    2>&1 | tee -a "$SHELL_LOG"

echo ""
echo "Training completed! Output: $OUTPUT_DIR"
echo "Logs: $SHELL_LOG"
