#!/bin/bash
# E10 — pretrained-backbone concept memory (Gemma-3-1B graft, Design C).
#   bash scripts/launch_e10.sh                    # concept arm (C=128)
#   CONCEPT_NUM=0 bash scripts/launch_e10.sh      # matched no-concept control arm
#
# Pins the E10 protocol (frozen gemma-3-1b-pt + LoRA, block K=512 = Gemma's sliding window,
# Gemma-tokenized mix at seq 2048, plain causal_lm objective) and delegates to the GENERIC
# launcher (train_perceiver_denoise_multigpu.sh), which owns training defaults + the gated
# pretokenize phase + the accelerate invocation. Override any knob by exporting it first.
#
# Arm invariant: both arms share backbone/LoRA/masks/mix/seed/effective-batch/epochs; ONLY
# CONCEPT_NUM differs. Spec: docs/experiments_specs/E10_gemma_backbone_concept_memory.md
# (run Stage 0 — analysis/run_e10_stage0.py — BEFORE training; gap G >= 0.05 nats gates the run).
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/ksopyla/dev/MrCogito"
[ -d "$PROJECT_ROOT" ] || PROJECT_ROOT="$(pwd)"

# ---- Data path overrides (set BEFORE sourcing remote_paths.sh; that script only
#      assigns DATASETS_*_DIR when unset, so this pre-set wins). E10 re-tokenizes
#      the E05 mix with the Gemma tokenizer into its OWN tree (never collides with
#      the SmolLM2 datasets_tok), and reads raw shards straight from the NAS archive
#      (populated by the E05 pass) to avoid re-downloading ~150 GB. ----
export DATASETS_TOK_DIR="${DATASETS_TOK_DIR:-${PROJECT_ROOT}/../hf_home/datasets_tok_gemma}"
# Re-tokenize straight from the NAS raw archive. The pretokenize script detects
# raw_dir == raw_archive_dir and skips the post-tokenize unlink that would
# otherwise delete the archive shards themselves.
export DATASETS_RAW_DIR="${DATASETS_RAW_DIR:-/nas/ml_data/mrcogito/hf_datasets/raw}"
export RAW_ARCHIVE_DIR="${RAW_ARCHIVE_DIR:-/nas/ml_data/mrcogito/hf_datasets/raw}"

# shellcheck source=scripts/remote_paths.sh
source "${SCRIPT_DIR}/remote_paths.sh"

# ---- Identity + E10 protocol pins ----
export EXPERIMENT_ID="${EXPERIMENT_ID:-E10}"
export BACKBONE_MODEL="${BACKBONE_MODEL:-google/gemma-3-1b-pt}"
export OBJECTIVE_VARIANT=causal_lm
export CONCEPT_NUM="${CONCEPT_NUM:-128}"
export CONCEPT_BLOCK=512                       # MUST equal the backbone's sliding window
export CONCEPT_IO_MODE="${CONCEPT_IO_MODE:-global_kv}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export TOKENIZER_NAME=google/gemma-3-1b-pt
export MAX_SEQ_LENGTH=2048
export SEED=42

# ---- Pretokenize pins ----
export PRETOKENIZE_MIX="${PRETOKENIZE_MIX:-smollm3_inspired_2k_e05}"
export MANIFEST="${MANIFEST:-${DATASETS_TOK_DIR}/${PRETOKENIZE_MIX}_gemma_manifest.json}"

# ---- Optimization (LoRA-typical; the backbone is frozen) ----
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-500}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export GRADIENT_CHECKPOINTING=True             # 1B backbone at seq 2048 on 24 GB cards
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export EVAL_STEPS="${EVAL_STEPS:-2000}"
export SAVE_STEPS="${SAVE_STEPS:-2000}"        # multiple of EVAL_STEPS (load_best_model_at_end)
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"   # full checkpoints carry the 1B backbone
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export DDP_TIMEOUT="${DDP_TIMEOUT:-14400}"

# ---- Token-matched arms: identical effective batch 24 on both servers ----
#   Odra   (3 GPU): 4 x 3 x 2 = 24
#   Polonez (4 GPU): 3 x 4 x 2 = 24
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "$NUM_GPUS" -eq 4 ]; then
    export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-3}"
else
    export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
fi
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

# ---- Budget: ~2B tokens (LoRA fine-tune, ICAE/RMT scale). MUST match across arms. ----
export NUM_EPOCHS="${NUM_EPOCHS:-0.1}"

echo "=== E10 launch (backbone=${BACKBONE_MODEL}, C=${CONCEPT_NUM}, ${NUM_GPUS} GPUs, effective batch $((PER_DEVICE_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))) ==="
echo "  arm=$([ "${CONCEPT_NUM}" = "0" ] && echo control || echo concept)  io=${CONCEPT_IO_MODE}  K=${CONCEPT_BLOCK}  lora_r=${LORA_R}"
echo "  mix=${PRETOKENIZE_MIX} (Gemma tokenizer) seq=${MAX_SEQ_LENGTH}  LR=${LEARNING_RATE}  epochs=${NUM_EPOCHS} (MUST match the other arm)"
echo "  DATASETS_TOK_DIR=${DATASETS_TOK_DIR}"
echo "  DATASETS_RAW_DIR=${DATASETS_RAW_DIR}"
echo "  MANIFEST=${MANIFEST}"

# Delegate everything else (defaults, pretokenize, accelerate launch) to the generic launcher.
exec bash "${SCRIPT_DIR}/train_perceiver_denoise_multigpu.sh"
