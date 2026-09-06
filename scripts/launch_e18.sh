#!/bin/bash
# E18 — Perceiver AR v2 pilot launcher (Polonez, 4×3090). Thin wrapper: pins the pilot protocol
# and delegates to the generic launcher. Spec: docs/experiments_specs/ahead/E18_perceiver_ar_v2_baseline.md
#
#   bash scripts/launch_e18.sh                                    # stage A: 125M, seq 8k, 2B tokens
#   E18_STAGE=32k RESUME_FROM_CHECKPOINT=<ckpt> bash scripts/launch_e18.sh   # stage B: seq 32k, 0.5B
#   PAR_MODE=dense bash scripts/launch_e18.sh                     # matched dense control (stage A)
#   E18_TASK=copy bash scripts/launch_e18.sh                      # P2 copy task (6 layers, seq 32k)
#
# First run tokenizes the mix (SKIP_PRETOKENIZE=0); later runs reuse the manifest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_ID="${EXPERIMENT_ID:-E18}"
export MODEL_FAMILY=perceiver_ar
export OBJECTIVE_VARIANT=causal_lm
export DECODER_TYPE=causal_ar          # ignored by the family; keeps legacy arg validation quiet
export TOKENIZER_NAME="${TOKENIZER_NAME:-HuggingFaceTB/SmolLM3-3B}"

E18_STAGE="${E18_STAGE:-8k}"
E18_TASK="${E18_TASK:-lm}"

# --- model (125M dense-equivalent) ---
export HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
export INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"
export TOKEN_EMBEDDING_DIM="${TOKEN_EMBEDDING_DIM:-256}"
export NUM_LAYERS="${NUM_LAYERS:-12}"            # stack layers (window N)
export NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
export HEAD_DIM="${HEAD_DIM:-128}"
export PAR_MODE="${PAR_MODE:-perceiver}"
export PAR_PRE_LAYERS="${PAR_PRE_LAYERS:-1}"
export PAR_PRE_WINDOW="${PAR_PRE_WINDOW:-512}"
export PAR_GLOBAL_LAYERS="${PAR_GLOBAL_LAYERS:-1}"
export PAR_BLOCK="${PAR_BLOCK:-2048}"
export PAR_NGRAM_BUCKETS="${PAR_NGRAM_BUCKETS:-65536}"
export PAR_VALUE_EMBED_LAYERS="${PAR_VALUE_EMBED_LAYERS:-0,4,8}"
export PAR_NOPE_EVERY="${PAR_NOPE_EVERY:-4}"
export ATTN_BACKEND="${ATTN_BACKEND:-flex}"
export ATTN_PAD_MULTIPLE="${ATTN_PAD_MULTIPLE:-2048}"
export CHUNKED_CE_BLOCK_SIZE="${CHUNKED_CE_BLOCK_SIZE:-2048}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"

# --- optimizer (E05/E16b-calibrated Muon triple) ---
export OPTIMIZER="${OPTIMIZER:-muon}"
export LEARNING_RATE="${LEARNING_RATE:-0.01}"
export MUON_ADAMW_LR="${MUON_ADAMW_LR:-2e-4}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant_with_warmup}"
export WARMUP_STEPS="${WARMUP_STEPS:-500}"
export LOGGING_STEPS="${LOGGING_STEPS:-20}"
export AUTO_INTERVALS="${AUTO_INTERVALS:-1}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-6}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-256}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Low-padding sortish batching (remote sampler, 2026-08): PG-19 8k rows next to short web rows.
export BATCH_PACKING_MODE="${BATCH_PACKING_MODE:-length_group}"

# --- data: one tokenized tree per (tokenizer, seq length) so TARGET_TOKENS counts real tokens ---
PROJECT_ROOT_HINT="/home/ksopyla/dev/MrCogito"
[ -d "$PROJECT_ROOT_HINT" ] || PROJECT_ROOT_HINT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ "$E18_TASK" = "copy" ] || [ "$E18_STAGE" = "32k" ]; then
    TOK_TREE="datasets_tok_smollm3_32k"
else
    TOK_TREE="datasets_tok_smollm3_8k"
fi
export DATASETS_TOK_DIR="${DATASETS_TOK_DIR:-${PROJECT_ROOT_HINT}/../hf_home/${TOK_TREE}}"
export TRAIN_NUM_PROC="${TRAIN_NUM_PROC:-32}"
export TEST_NUM_PROC="${TEST_NUM_PROC:-8}"

if [ "$E18_TASK" = "copy" ]; then
    # P2: mirrored copy at 32k, small 6-layer model, manifest from scripts/build_copy_task_dataset.py
    export NUM_LAYERS="${COPY_NUM_LAYERS:-6}"
    export MAX_SEQ_LENGTH=32768
    export PRETOKENIZED_MANIFEST="${PRETOKENIZED_MANIFEST:-${DATASETS_TOK_DIR}/copy_32k_manifest.json}"
    export PRESERVE_PRECOMPUTED_LABELS=true
    export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
    export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
    export NUM_EPOCHS="${NUM_EPOCHS:-2}"
    export EVAL_STEPS="${EVAL_STEPS:-500}"
    export SAVE_STEPS="${SAVE_STEPS:-500}"
    export AUTO_INTERVALS=0
    export PAR_VALUE_EMBED_LAYERS="0,3"
    export PRETOKENIZE_MIX=""
else
    export PRETOKENIZE_MIX="${PRETOKENIZE_MIX:-e18_pilot_longdoc_v1}"
    export MANIFEST="${MANIFEST:-${DATASETS_TOK_DIR}/${PRETOKENIZE_MIX}_manifest.json}"
    export SKIP_PRETOKENIZE="${SKIP_PRETOKENIZE:-0}"
    if [ "$E18_STAGE" = "32k" ]; then
        export MAX_SEQ_LENGTH=32768
        export TARGET_TOKENS="${TARGET_TOKENS:-500000000}"
        export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
        export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
        export WARMUP_STEPS="${WARMUP_STEPS:-100}"
    else
        export MAX_SEQ_LENGTH=8192
        export TARGET_TOKENS="${TARGET_TOKENS:-2000000000}"
        export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
        export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
    fi
fi
echo "=== E18 Perceiver AR v2 pilot — task=${E18_TASK} stage=${E18_STAGE} mode=${PAR_MODE} ==="
echo "  H=${HIDDEN_SIZE} stack=${NUM_LAYERS} pre=${PAR_PRE_LAYERS}@${PAR_PRE_WINDOW} global=${PAR_GLOBAL_LAYERS} N=${PAR_BLOCK}"
echo "  seq=${MAX_SEQ_LENGTH} backend=${ATTN_BACKEND} tok_dir=${DATASETS_TOK_DIR}"
echo "  muon lr=${LEARNING_RATE} adamw=${MUON_ADAMW_LR} wd=${WEIGHT_DECAY} batch/gpu=${PER_DEVICE_BATCH_SIZE} accum=${GRADIENT_ACCUMULATION_STEPS}"

exec bash "${SCRIPT_DIR}/train_concept_pretraining_multigpu.sh"
