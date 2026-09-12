#!/bin/bash
# E22 — Perceiver Concept LM pilot launcher (Odra 3×3090 / Polonez 4×3090). Thin wrapper: pins
# the pilot protocol and delegates to the generic launcher.
# Spec: docs/experiments_specs/done_failed/E22_perceiver_concept_lm.md
#
#   bash scripts/launch_e22.sh                       # arm A: the bet (encoder → concepts → latent → decoder)
#   E22_ARM=C bash scripts/launch_e22.sh             # arm C: same model, decoder never reads the array
#   E22_ARM=dense bash scripts/launch_e22.sh         # dense control: perceiver_ar PAR_MODE=dense, 18 full layers
#   E22_SMOKE=1 bash scripts/launch_e22.sh           # 30-step smoke / batch calibration on the same data
#
# Data: the E22 long-document manifest (write once per server with scripts/write_manifest_variant.py
# from the E18b merged manifest); every row packed to 32k with doc_ids.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_ID="${EXPERIMENT_ID:-E22}"
E22_ARM="${E22_ARM:-A}"
E22_SMOKE="${E22_SMOKE:-0}"
export OBJECTIVE_VARIANT=causal_lm
export DECODER_TYPE=causal_ar          # ignored by the family; keeps legacy arg validation quiet
export TOKENIZER_NAME="${TOKENIZER_NAME:-HuggingFaceTB/SmolLM3-3B}"

# --- shared widths (E18 pilot geometry) ---
export HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
export INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"
export TOKEN_EMBEDDING_DIM="${TOKEN_EMBEDDING_DIM:-256}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
export HEAD_DIM="${HEAD_DIM:-128}"
export PAR_NGRAM_BUCKETS="${PAR_NGRAM_BUCKETS:-65536}"
export PAR_VALUE_EMBED_DIM="${PAR_VALUE_EMBED_DIM:-64}"
export ATTN_BACKEND="${ATTN_BACKEND:-flex}"
export ATTN_PAD_MULTIPLE="${ATTN_PAD_MULTIPLE:-2048}"
export CHUNKED_CE_BLOCK_SIZE="${CHUNKED_CE_BLOCK_SIZE:-2048}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"

case "$E22_ARM" in
  A|C)
    export MODEL_FAMILY=perceiver_concept
    export PCL_ENC_LAYERS="${PCL_ENC_LAYERS:-6}"
    export PCL_ENC_WINDOW="${PCL_ENC_WINDOW:-512}"
    export PCL_CONCEPT_RATIO="${PCL_CONCEPT_RATIO:-16}"
    export PCL_CONCEPT_SLOTS="${PCL_CONCEPT_SLOTS:-1}"
    export PCL_LATENT_LAYERS="${PCL_LATENT_LAYERS:-4}"
    export PCL_LATENT_REPEATS="${PCL_LATENT_REPEATS:-1}"
    export PCL_DEC_LAYERS="${PCL_DEC_LAYERS:-8}"
    export PCL_DEC_SEGMENT="${PCL_DEC_SEGMENT:-1024}"
    export PCL_DEC_LOCAL="${PCL_DEC_LOCAL:-block}"
    export PCL_CONCEPT_MODE="$([ "$E22_ARM" = "C" ] && echo none || echo "${PCL_CONCEPT_MODE:-full}")"
    export PCL_CONCEPT_XATTN_SCOPE="${PCL_CONCEPT_XATTN_SCOPE:-causal}"   # E23: exclusive
    export NUM_LAYERS="${PCL_ENC_LAYERS}"   # legacy arg; the family ignores it
    ;;
  dense)
    # Matched dense control: same width, 18 full-causal layers (= enc 6 + latent 4 + dec 8).
    export MODEL_FAMILY=perceiver_ar
    export PAR_MODE=dense
    export PAR_PRE_LAYERS=0
    export PAR_GLOBAL_LAYERS=0
    export NUM_LAYERS="${NUM_LAYERS:-18}"
    export PAR_BLOCK=2048
    export PAR_VALUE_EMBED_LAYERS="${PAR_VALUE_EMBED_LAYERS:-0,3,10}"
    export PAR_NOPE_EVERY=0
    ;;
  *) echo "E22_ARM must be A, C or dense"; exit 2 ;;
esac

# --- optimizer (E05/E18-calibrated Muon triple), cosine to 10% ---
export OPTIMIZER="${OPTIMIZER:-muon}"
export LEARNING_RATE="${LEARNING_RATE:-0.01}"
export MUON_ADAMW_LR="${MUON_ADAMW_LR:-2e-4}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-128}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DDP_TIMEOUT="${DDP_TIMEOUT:-10800}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export LENGTH_CACHE_NUM_PROC="${LENGTH_CACHE_NUM_PROC:-4}"

# --- data: packed 32k rows over the E22 long-document manifest ---
PROJECT_ROOT_HINT="$(cd "${SCRIPT_DIR}/.." && pwd)"          # the servers keep the repo next to hf_home
export DATASETS_TOK_DIR="${DATASETS_TOK_DIR:-${PROJECT_ROOT_HINT}/../hf_home/datasets_tok_smollm3_32k}"
export PRETOKENIZED_MANIFEST="${PRETOKENIZED_MANIFEST:-${DATASETS_TOK_DIR}/e22_longmix_32k_manifest.json}"
export PRETOKENIZE_MIX=""
export MAX_SEQ_LENGTH=32768
export BATCH_PACKING_MODE="${BATCH_PACKING_MODE:-pack}"
export LOSS_SPAN_MARKERS="${LOSS_SPAN_MARKERS:-128103,128104}"   # E18b keyed-recall rows: labels inside START..END
export TRAIN_NUM_PROC="${TRAIN_NUM_PROC:-8}"
export TEST_NUM_PROC="${TEST_NUM_PROC:-4}"
# Effective batch fixed at 24 packed 32k rows (~0.79M tokens/step) on either server:
# 2 rows/GPU × 3 GPUs × accum 4 (Odra) or × 4 GPUs × accum 3 (Polonez). Calibrated 2026-09-12 on a
# 3090: arm A at B=2 peaks at 11.4 GiB and ~19.8k tok/s/GPU; arm C 8.0 GiB / ~29k tok/s/GPU.
_E22_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"; [ "${_E22_GPUS:-0}" -ge 1 ] || _E22_GPUS=1
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
_E22_ROWS="${E22_EFFECTIVE_ROWS:-24}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$(( _E22_ROWS / (PER_DEVICE_BATCH_SIZE * _E22_GPUS) ))}"
[ "$GRADIENT_ACCUMULATION_STEPS" -ge 1 ] || export GRADIENT_ACCUMULATION_STEPS=1
export AUTO_INTERVALS=0
if [ "$E22_SMOKE" = "1" ]; then
    export MAX_STEPS="${MAX_STEPS:-30}"
    export NUM_EPOCHS=1
    export EVAL_STEPS="${EVAL_STEPS:-15}"
    export SAVE_STEPS="${SAVE_STEPS:-15}"
    export REPORT_TO="${REPORT_TO:-none}"
    export MAX_EVAL_SAMPLES=8
else
    # 0.5B tokens ≈ 640 optimizer steps at ~0.79M packed tokens / step.
    export TARGET_TOKENS="${TARGET_TOKENS:-500000000}"
    export EVAL_STEPS="${EVAL_STEPS:-80}"
    export SAVE_STEPS="${SAVE_STEPS:-80}"
fi

if [ ! -f "$PRETOKENIZED_MANIFEST" ]; then
    echo "ERROR: E22 manifest not found at $PRETOKENIZED_MANIFEST"
    echo "  write it with: uv run python scripts/write_manifest_variant.py --src <e18b_lm_ret05_manifest.json> --dst $PRETOKENIZED_MANIFEST --token_share '{...}' --mix_id e22_longmix_32k"
    exit 1
fi

echo "=== E22 Perceiver Concept LM — arm=${E22_ARM} family=${MODEL_FAMILY} smoke=${E22_SMOKE} ==="
if [ "$MODEL_FAMILY" = "perceiver_concept" ]; then
    echo "  enc=${PCL_ENC_LAYERS}@swa${PCL_ENC_WINDOW} r=${PCL_CONCEPT_RATIO} c=${PCL_CONCEPT_SLOTS} latent=${PCL_LATENT_LAYERS}x${PCL_LATENT_REPEATS} dec=${PCL_DEC_LAYERS}@${PCL_DEC_LOCAL}${PCL_DEC_SEGMENT} concepts=${PCL_CONCEPT_MODE}"
else
    echo "  dense control: layers=${NUM_LAYERS} block=${PAR_BLOCK}"
fi
echo "  seq=${MAX_SEQ_LENGTH} pack=${BATCH_PACKING_MODE} backend=${ATTN_BACKEND} manifest=${PRETOKENIZED_MANIFEST}"
echo "  muon lr=${LEARNING_RATE} adamw=${MUON_ADAMW_LR} wd=${WEIGHT_DECAY} batch/gpu=${PER_DEVICE_BATCH_SIZE} accum=${GRADIENT_ACCUMULATION_STEPS} sched=${LR_SCHEDULER_TYPE}"

exec bash "${SCRIPT_DIR}/train_concept_pretraining_multigpu.sh"
