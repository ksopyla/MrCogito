#!/bin/bash

set -euo pipefail

echo "=== Perceiver Denoising Multi-GPU Training ==="
echo "Default profile: Odra A1 reconstruction baseline"

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "$NUM_GPUS" -le 0 ]; then
    echo "ERROR: No GPUs detected."
    exit 1
fi

GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NCCL_DEBUG=WARN
# Allow callers to override the allocator config (e.g. expandable_segments:True to
# cut fragmentation / reserved-but-unallocated waste on tight-memory runs).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/remote_paths.sh
source "${SCRIPT_DIR}/remote_paths.sh"

SHELL_LOG="${LOGGING_DIR}/shell_perceiver_denoise_$(date +%Y%m%d_%H%M%S).log"

echo "HF_HOME=${HF_HOME}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOGGING_DIR=${LOGGING_DIR}"

# Default experiment profile:
# A1 on Odra = clean perceiver_denoise reconstruction baseline
# H512 / T512 / L6 / C128 / D3, BiXT on, no concept losses.
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
TOKEN_EMBEDDING_DIM="${TOKEN_EMBEDDING_DIM:-512}"
NUM_LAYERS="${NUM_LAYERS:-6}"
CONCEPT_NUM="${CONCEPT_NUM:-128}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"
DECODER_NUM_LAYERS="${DECODER_NUM_LAYERS:-3}"
# Decoder family + modern blocks (E01). Defaults reproduce the perceiver_denoise baseline.
DECODER_TYPE="${DECODER_TYPE:-perceiver_posonly}"   # | causal_ar
DECODER_POS_TYPE="${DECODER_POS_TYPE:-learned}"     # | rope (causal_ar)
DECODER_WORD_DROPOUT="${DECODER_WORD_DROPOUT:-0.0}"
# E05: sliding-window causal context for causal_ar (last-K tokens). Empty = full causal.
DECODER_CONTEXT_WINDOW="${DECODER_CONTEXT_WINDOW:-}"
HIDDEN_ACT="${HIDDEN_ACT:-gelu}"                    # | silu (SwiGLU)
NORM_TYPE="${NORM_TYPE:-layernorm}"                 # | rmsnorm
# E03 — concept de-collapse via a frozen-encoder hidden-state anchor (causal_ar + reconstruction).
# Default OFF reproduces E01 exactly (the matched control arm = ANCHOR_LOSS=false).
ANCHOR_LOSS="${ANCHOR_LOSS:-false}"                 # | true
ANCHOR_MODEL="${ANCHOR_MODEL:-HuggingFaceTB/SmolLM2-135M}"
ANCHOR_LOSS_WEIGHT="${ANCHOR_LOSS_WEIGHT:-0.5}"
ANCHOR_STANDARDIZE="${ANCHOR_STANDARDIZE:-true}"
ANCHOR_HEAD_LAYERS="${ANCHOR_HEAD_LAYERS:-2}"
DATASET_NAME="${DATASET_NAME:-JeanKaddour/minipile}"
DATASET_SUBSET="${DATASET_SUBSET:-}"
# Registered multi-dataset mix name (e.g. long_2k_base_v1). Empty = single dataset above.
DATASET_MIX="${DATASET_MIX:-}"
# Preferred recipe-based mix config (path or id under data/mix_recipes/).
DATASET_MIX_RECIPE="${DATASET_MIX_RECIPE:-}"
TOKENIZER_NAME="${TOKENIZER_NAME:-answerdotai/ModernBERT-base}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
DELETION_RATE="${DELETION_RATE:-0.6}"
OBJECTIVE_VARIANT="${OBJECTIVE_VARIANT:-reconstruction}"
PREFIX_RATIO_MIN="${PREFIX_RATIO_MIN:-0.3}"
PREFIX_RATIO_MAX="${PREFIX_RATIO_MAX:-0.5}"
MIN_PREFIX_CONTENT="${MIN_PREFIX_CONTENT:-5}"
MIN_SUFFIX_CONTENT="${MIN_SUFFIX_CONTENT:-10}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-sentence_boundary}"
CONCEPT_LOSSES="${CONCEPT_LOSSES:-none}"
LOSS_WEIGHT="${LOSS_WEIGHT:-0.02}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-1500}"
LOGGING_STEPS="${LOGGING_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
SAVE_SAFETENSORS="${SAVE_SAFETENSORS:-True}"
SEED="${SEED:-42}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
# FineWeb-Edu one-off tokenization: use Polonez headroom (32–48); keep Odra defaults modest.
TRAIN_NUM_PROC="${TRAIN_NUM_PROC:-8}"
TEST_NUM_PROC="${TEST_NUM_PROC:-4}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
# DDP collective timeout (seconds). Raise well above the 30-min default when the first
# epoch tokenizes a large corpus under main_process_first (e.g. FineWeb-Edu), otherwise
# non-main ranks time out at the preprocessing barrier (NCCL SeqNum=1 ALLREDUCE).
DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"
# Export so setup_distributed() can apply it to the FIRST process group too
# (created before TrainingArguments parsing; --ddp_timeout alone is too late
# to protect the first-time preprocessing barrier).
export DDP_TIMEOUT

RESUME_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

# E03 anchor args are only passed when enabled; the control arm (ANCHOR_LOSS=false) passes nothing,
# so anchor_loss defaults to False and the run is byte-for-byte E01.
# E05: only pass --decoder_context_window when set, so the default stays None (full causal).
WINDOW_ARGS=()
if [ -n "$DECODER_CONTEXT_WINDOW" ]; then
    WINDOW_ARGS+=(--decoder_context_window "$DECODER_CONTEXT_WINDOW")
fi

# E05: pass --dataset_mix only when set (overrides the single dataset path).
MIX_ARGS=()
if [ -n "$DATASET_MIX" ]; then
    MIX_ARGS+=(--dataset_mix "$DATASET_MIX")
fi
if [ -n "$DATASET_MIX_RECIPE" ]; then
    MIX_ARGS+=(--dataset_mix_recipe "$DATASET_MIX_RECIPE")
fi

ANCHOR_ARGS=()
if [ "$ANCHOR_LOSS" = "true" ]; then
    ANCHOR_ARGS+=(
        --anchor_loss
        --anchor_model_name "$ANCHOR_MODEL"
        --anchor_loss_weight "$ANCHOR_LOSS_WEIGHT"
        --anchor_standardize "$ANCHOR_STANDARDIZE"
        --anchor_head_layers "$ANCHOR_HEAD_LAYERS"
    )
fi

accelerate launch \
    --num_processes="$NUM_GPUS" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --multi_gpu \
    training/train_perceiver_denoise.py \
    --hidden_size "$HIDDEN_SIZE" \
    --token_embedding_dim "$TOKEN_EMBEDDING_DIM" \
    --num_hidden_layers "$NUM_LAYERS" \
    --concept_num "$CONCEPT_NUM" \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --decoder_num_layers "$DECODER_NUM_LAYERS" \
    --decoder_type "$DECODER_TYPE" \
    --decoder_pos_type "$DECODER_POS_TYPE" \
    --decoder_word_dropout "$DECODER_WORD_DROPOUT" \
    --hidden_act "$HIDDEN_ACT" \
    --norm_type "$NORM_TYPE" \
    --use_bixt \
    --deletion_rate "$DELETION_RATE" \
    --objective_variant "$OBJECTIVE_VARIANT" \
    --prefix_ratio_min "$PREFIX_RATIO_MIN" \
    --prefix_ratio_max "$PREFIX_RATIO_MAX" \
    --min_prefix_content "$MIN_PREFIX_CONTENT" \
    --min_suffix_content "$MIN_SUFFIX_CONTENT" \
    --split_strategy "$SPLIT_STRATEGY" \
    --dataset_name "$DATASET_NAME" \
    --dataset_name_subset "$DATASET_SUBSET" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --dataset_cache_dir "$HF_DATASETS_CACHE" \
    --train_num_proc "$TRAIN_NUM_PROC" \
    --test_num_proc "$TEST_NUM_PROC" \
    --concept_losses "$CONCEPT_LOSSES" \
    --loss_weight "$LOSS_WEIGHT" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --warmup_steps "$WARMUP_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --eval_strategy "steps" \
    --eval_steps "$EVAL_STEPS" \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$LOGGING_DIR" \
    --seed "$SEED" \
    --bf16 \
    --ddp_backend "nccl" \
    --ddp_timeout "$DDP_TIMEOUT" \
    --ddp_find_unused_parameters False \
    --dataloader_pin_memory True \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --gradient_checkpointing False \
    --optim "adamw_torch_fused" \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --save_safetensors "$SAVE_SAFETENSORS" \
    --overwrite_output_dir True \
    --remove_unused_columns True \
    --disable_tqdm True \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    "${WINDOW_ARGS[@]}" \
    "${MIX_ARGS[@]}" \
    "${ANCHOR_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    2>&1 | python scripts/clean_tee.py "$SHELL_LOG"
