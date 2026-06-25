#!/bin/bash
# E05 windowed prefix→suffix warmup on Odra (smollm3_inspired_2k_e05warmup, K=128, seq 2K).
# Two phases:
#   1) pretokenize: parallel per-source parquet download + tokenize + save_to_disk
#      (scripts/pretokenize_mix.py). Cached under ~/dev/hf_home/datasets_tok.
#   2) train: DDP, loads the pretokenized manifest via load_from_disk (instant).
# Re-run training only (cache warm): SKIP_PRETOKENIZE=1
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/remote_paths.sh
source "${SCRIPT_DIR}/remote_paths.sh"

export EXPERIMENT_ID="${EXPERIMENT_ID:-E05}"
export DECODER_TYPE=causal_ar
export DECODER_CONTEXT_WINDOW="${DECODER_CONTEXT_WINDOW:-128}"
export MAX_SEQ_LENGTH=2048
export OBJECTIVE_VARIANT=prefix_suffix
export HIDDEN_SIZE=768
export TOKEN_EMBEDDING_DIM=256
export NUM_LAYERS=6
export CONCEPT_NUM=128
export DECODER_NUM_LAYERS=4
export INTERMEDIATE_SIZE=2048
export HIDDEN_ACT=silu
export NORM_TYPE=rmsnorm
export DECODER_POS_TYPE=rope
export TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M
export SEED=42
export NUM_EPOCHS="${NUM_EPOCHS:-0.3}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export EVAL_BATCH_SIZE=4
export LEARNING_RATE=3e-4
export WARMUP_STEPS=500
export LOGGING_STEPS=100
export EVAL_STEPS=1000
export SAVE_STEPS=1000
export SAVE_TOTAL_LIMIT=3
export DDP_TIMEOUT="${DDP_TIMEOUT:-14400}"
export TRAIN_NUM_PROC="${TRAIN_NUM_PROC:-8}"
export TEST_NUM_PROC="${TEST_NUM_PROC:-4}"
export DATALOADER_NUM_WORKERS=4

MIX_RECIPE="${MIX_RECIPE:-smollm3_inspired_2k_e05}"
DATASETS_TOK_DIR="${HF_HOME}/../datasets_tok"
MANIFEST="${MANIFEST:-${DATASETS_TOK_DIR}/${MIX_RECIPE}_manifest.json}"
# Archive raw parquet/zst to NAS after tokenizing (tokenizer-agnostic, survives a
# future tokenizer switch). Set RAW_ARCHIVE_DIR= to disable. NAS is slow but write-once.
RAW_ARCHIVE_DIR="${RAW_ARCHIVE_DIR:-/nas/ml_data/mrcogito/hf_datasets/raw}"

echo "=== E05 Odra launch ==="
echo "EXPERIMENT_ID=${EXPERIMENT_ID}  DECODER_CONTEXT_WINDOW=${DECODER_CONTEXT_WINDOW:-<full causal>}"
echo "MIX_RECIPE=${MIX_RECIPE}  MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH}"
echo "MANIFEST=${MANIFEST}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} × GPUs × GRAD_ACCUM=${GRADIENT_ACCUMULATION_STEPS}"

if [ "${SKIP_PRETOKENIZE:-0}" != "1" ]; then
    echo "=== Phase 1: pretokenize mix (parallel download + tokenize) ==="
    uv run python scripts/pretokenize_mix.py \
        --mix "${MIX_RECIPE}" \
        --tokenizer "${TOKENIZER_NAME}" \
        --max_seq_length "${MAX_SEQ_LENGTH}" \
        --cache_dir "${DATASETS_TOK_DIR}" \
        --raw_dir "${HF_HOME}/../datasets_raw" \
        --manifest "${MANIFEST}" \
        --objective "${OBJECTIVE_VARIANT}" \
        --seed "${SEED}" \
        --train_num_proc "${TRAIN_NUM_PROC}" \
        --test_num_proc "${TEST_NUM_PROC}" \
        --download_workers "${DOWNLOAD_WORKERS:-8}" \
        --jobs "${PRETOK_JOBS:-1}" \
        ${RAW_ARCHIVE_DIR:+--raw_archive_dir "${RAW_ARCHIVE_DIR}"}
else
    echo "=== Skipping pretokenize (SKIP_PRETOKENIZE=1) ==="
fi

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: manifest not found at ${MANIFEST} — pretokenize failed?"
    exit 1
fi

echo "=== W&B identity preflight ==="
uv run python - <<'PY'
import os
from training.utils_training import build_perceiver_wandb_identity

def e(name, default):
    return os.environ.get(name, default)

identity = build_perceiver_wandb_identity(
    decoder_type=e("DECODER_TYPE", "causal_ar"),
    objective_variant=e("OBJECTIVE_VARIANT", "prefix_suffix"),
    hidden_size=int(e("HIDDEN_SIZE", "768")),
    num_hidden_layers=int(e("NUM_LAYERS", "6")),
    concept_num=int(e("CONCEPT_NUM", "128")),
    decoder_num_layers=int(e("DECODER_NUM_LAYERS", "4")),
    checkpoint_family="concept_ar",
    pretraining_objective="ar_prefix_suffix_generation",
    use_bixt=True,
    experiment_id=os.environ.get("EXPERIMENT_ID"),
)
print("group:", identity.group)
print("job_type:", identity.job_type)
print("tags:", ", ".join(identity.tags))
PY

# Training uses the pretokenized manifest (load_from_disk, instant) — no dataset_mix needed.
export PRETOKENIZED_MANIFEST="${MANIFEST}"
unset DATASET_MIX_RECIPE
unset DATASET_MIX

echo "=== Phase 2: starting training (pretokenized manifest) ==="
exec bash "${SCRIPT_DIR}/train_perceiver_denoise_multigpu.sh"
