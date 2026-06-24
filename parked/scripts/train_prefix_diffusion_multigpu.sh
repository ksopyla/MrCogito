#!/bin/bash
# Multi-GPU training script for easier prefix diffusion.
# Run on Polonez (4x RTX 3090) or Odra (3x RTX 3090)
#
# Goal: keep the diffusion objective unchanged while making the prefix task
# easier to learn:
#   - WikiText-103 instead of MiniPile
#   - longer observed prefix (70-80%)
#   - sentence-boundary splitting
#   - longer default training horizon
#
# Usage:
#   bash scripts/train_prefix_diffusion_multigpu.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/remote_paths.sh
source "${SCRIPT_DIR}/../../scripts/remote_paths.sh"

echo "=== Multi-GPU Training: Prefix Diffusion ==="
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
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"   # Optional warm-start from encoder checkpoint

HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
TOKEN_EMBEDDING_DIM="${TOKEN_EMBEDDING_DIM:-64}"
NUM_ENCODER_LAYERS="${NUM_ENCODER_LAYERS:-6}"
CONCEPT_NUM="${CONCEPT_NUM:-128}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"
CONCEPT_POSITION_TYPE="${CONCEPT_POSITION_TYPE:-none}"
USE_BIXT="${USE_BIXT:-True}"
BIXT_TOKEN_FFN="${BIXT_TOKEN_FFN:-True}"
DECODER_LAYERS="${DECODER_LAYERS:-2}"
T_MIN="${T_MIN:-0.3}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
ELBO_WEIGHT="${ELBO_WEIGHT:-True}"

# =============================================================================
# DATA
# =============================================================================
DATASET_NAME="${DATASET_NAME:-Salesforce/wikitext}"
DATASET_SUBSET="${DATASET_SUBSET:-wikitext-103-v1}"
TOKENIZER_NAME="${TOKENIZER_NAME:-answerdotai/ModernBERT-base}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
TEST_SIZE_PERCENT="${TEST_SIZE_PERCENT:-0.1}"
PREFIX_RATIO_MIN="${PREFIX_RATIO_MIN:-0.7}"
PREFIX_RATIO_MAX="${PREFIX_RATIO_MAX:-0.8}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-sentence_boundary}"
MIN_PREFIX_CONTENT="${MIN_PREFIX_CONTENT:-8}"
MIN_SUFFIX_CONTENT="${MIN_SUFFIX_CONTENT:-16}"
MIN_TOTAL_CONTENT_TOKENS="${MIN_TOTAL_CONTENT_TOKENS:-32}"

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"                 # Keep overridable; default to a longer run
WARMUP_STEPS="${WARMUP_STEPS:-3000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# =============================================================================
# CONCEPT LOSSES (start with none for clean baseline)
# =============================================================================
CONCEPT_LOSSES="${CONCEPT_LOSSES:-none}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-fixed}"
LOSS_WEIGHT="${LOSS_WEIGHT:-0.02}"
CONCEPT_LOSS_WARMUP_STEPS="${CONCEPT_LOSS_WARMUP_STEPS:-0}"

# =============================================================================
# TORCH COMPILE
# =============================================================================
TORCH_COMPILE_DYNAMIC="${TORCH_COMPILE_DYNAMIC:-False}"

# =============================================================================
# LOGGING
# =============================================================================
LOGGING_STEPS="${LOGGING_STEPS:-1000}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_STEPS="${EVAL_STEPS:-10000}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
SEED="${SEED:-42}"

SHELL_LOG="${LOGGING_DIR}/shell_prefix_diffusion_$(date +%Y%m%d_%H%M%S).log"

BIXT_LABEL=""
if [ "$USE_BIXT" = "True" ]; then
    BIXT_LABEL="+BiXT"
fi

echo "Model: ConceptEncoder-H${HIDDEN_SIZE}L${NUM_ENCODER_LAYERS}C${CONCEPT_NUM}${BIXT_LABEL} + PrefixDiffusionDecoder-D${DECODER_LAYERS}"
echo "Token embedding dim: $TOKEN_EMBEDDING_DIM"
echo "Dataset: $DATASET_NAME"
echo "Dataset subset: $DATASET_SUBSET"
echo "Prefix ratio: [${PREFIX_RATIO_MIN}, ${PREFIX_RATIO_MAX}]"
echo "Split strategy: $SPLIT_STRATEGY"
echo "Diffusion t range: [${T_MIN}, 1.0]"
echo "Epochs: $NUM_EPOCHS"
echo "Effective batch: $((PER_DEVICE_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
echo ""

echo "HF_HOME=${HF_HOME}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "Starting training... (log: $SHELL_LOG)"

accelerate launch \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --multi_gpu \
    training/train_prefix_diffusion.py \
    --hidden_size "$HIDDEN_SIZE" \
    --token_embedding_dim "$TOKEN_EMBEDDING_DIM" \
    --num_hidden_layers "$NUM_ENCODER_LAYERS" \
    --concept_num "$CONCEPT_NUM" \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --concept_position_type "$CONCEPT_POSITION_TYPE" \
    --use_bixt "$USE_BIXT" \
    --bixt_token_ffn "$BIXT_TOKEN_FFN" \
    --decoder_layers "$DECODER_LAYERS" \
    --t_min "$T_MIN" \
    --label_smoothing "$LABEL_SMOOTHING" \
    --elbo_weight "$ELBO_WEIGHT" \
    --torch_compile_dynamic "$TORCH_COMPILE_DYNAMIC" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset_name "$DATASET_NAME" \
    --dataset_name_subset "$DATASET_SUBSET" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --test_size_percent "$TEST_SIZE_PERCENT" \
    --prefix_ratio_min "$PREFIX_RATIO_MIN" \
    --prefix_ratio_max "$PREFIX_RATIO_MAX" \
    --split_strategy "$SPLIT_STRATEGY" \
    --min_prefix_content "$MIN_PREFIX_CONTENT" \
    --min_suffix_content "$MIN_SUFFIX_CONTENT" \
    --min_total_content_tokens "$MIN_TOTAL_CONTENT_TOKENS" \
    --dataset_cache_dir "$HF_DATASETS_CACHE" \
    --concept_losses "$CONCEPT_LOSSES" \
    --loss_weighting "$LOSS_WEIGHTING" \
    --loss_weight "$LOSS_WEIGHT" \
    --concept_loss_warmup_steps "$CONCEPT_LOSS_WARMUP_STEPS" \
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
    --load_best_model_at_end False \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    2>&1 | python scripts/clean_tee.py "$SHELL_LOG"

echo ""
echo "Training completed! Output: $OUTPUT_DIR"
echo "Logs: $SHELL_LOG"
