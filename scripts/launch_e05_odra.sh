#!/bin/bash
# E05 windowed prefix→suffix warmup on Odra (smollm3_inspired_2k, K=128, seq 2K).
# Pretokenizes once into HF_DATASETS_CACHE, then launches DDP training (cache reused).
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/remote_paths.sh
source "${SCRIPT_DIR}/remote_paths.sh"

export EXPERIMENT_ID="${EXPERIMENT_ID:-E05}"
export DECODER_TYPE=causal_ar
export DECODER_CONTEXT_WINDOW="${DECODER_CONTEXT_WINDOW:-128}"
export DATASET_MIX_RECIPE="${DATASET_MIX_RECIPE:-smollm3_inspired_2k}"
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

echo "=== E05 Odra launch ==="
echo "EXPERIMENT_ID=${EXPERIMENT_ID}"
echo "DECODER_CONTEXT_WINDOW=${DECODER_CONTEXT_WINDOW:-<full causal>}"
echo "DATASET_MIX_RECIPE=${DATASET_MIX_RECIPE}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} × GPUs × GRAD_ACCUM=${GRADIENT_ACCUMULATION_STEPS}"

if [ "${SKIP_PRETOKENIZE:-0}" != "1" ]; then
    echo "=== Pretokenizing mix (single process, warms HF map cache) ==="
    uv run python - <<'PY'
import os
from transformers import AutoTokenizer

from data.dataset_preprocess import load_and_preprocess_dataset_mix
from training.train_perceiver_denoise import resolve_append_eos_token_id

cache = os.environ["HF_DATASETS_CACHE"]
recipe = os.environ["DATASET_MIX_RECIPE"]
max_len = int(os.environ["MAX_SEQ_LENGTH"])
seed = int(os.environ["SEED"])
train_proc = int(os.environ.get("TRAIN_NUM_PROC", "8"))
test_proc = int(os.environ.get("TEST_NUM_PROC", "4"))
objective = os.environ["OBJECTIVE_VARIANT"]

tokenizer = AutoTokenizer.from_pretrained(os.environ["TOKENIZER_NAME"])
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
append_eos = resolve_append_eos_token_id(
    objective, is_causal_ar=True, eos_token_id=tokenizer.eos_token_id
)

train_ds, test_ds = load_and_preprocess_dataset_mix(
    tokenizer,
    recipe,
    max_seq_length=max_len,
    dataset_cache_dir=cache,
    train_num_proc=train_proc,
    test_num_proc=test_proc,
    append_eos_token_id=append_eos,
    split_seed=seed,
    interleave_seed=seed,
)
print(f"Pretokenize OK: train={len(train_ds):,} eval={len(test_ds):,} cache={cache}")
PY
else
    echo "=== Skipping pretokenize (SKIP_PRETOKENIZE=1) ==="
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

echo "=== Starting training ==="
exec bash "${SCRIPT_DIR}/train_perceiver_denoise_multigpu.sh"
