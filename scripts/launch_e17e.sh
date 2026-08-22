#!/bin/bash
# E17e — starve the local window (K=256) on the E17d cell.
# Pins E17d's attn-residual / no-carry protocol plus CONCEPT_BLOCK=256, then
# delegates through E10's Gemma protocol to the shared multi-GPU launcher.
# Caller overrides remain available for calibration controls such as MAX_STEPS,
# REPORT_TO, and SAVE/EVAL_STRATEGY.
#
# Token budget is 300M non-padding tokens. Do not override to 300B; that is not
# runnable on 4× RTX 3090. Do not launch 1B unless the 300M late-half gate passes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_ID="${EXPERIMENT_ID:-E17e}"
export CONCEPT_BLOCK=256
export CONCEPT_IO_MODE=per_layer_banks
export CONCEPT_READ_MODE=dedicated
export CONCEPT_READ_PLACEMENT=attn_residual
export TIE_CONCEPT_WRITER=false
export CONCEPT_WRITE_MODE=additive
export WRITE_GATE_INIT=0.1
export MEMORY_CARRY_DROPOUT=1.0
export INFERENCE_CARRY_POLICY=drop_after_first
export MEMORY_PRESSURE_TOKENS=0
export MEMORY_PRESSURE_WEIGHT=1.0
export READ_CONCEPT_NORM=true
export READ_GATE_INIT=0.1

# Match E16b/E17/E17d data, optimization, and token-budget protocol.
export MAX_SEQ_LENGTH=4096
export PRETOKENIZE_MIX=e16b_long_4k_v1
export BATCH_PACKING_MODE="${BATCH_PACKING_MODE:-length_group}"
export LENGTH_GROUP_MEGA_BATCH_MULT="${LENGTH_GROUP_MEGA_BATCH_MULT:-20}"
# Parallel Hugging Face datasets.map workers for the length sidecar (rank 0).
export LENGTH_CACHE_NUM_PROC="${LENGTH_CACHE_NUM_PROC:-32}"
# First full run is the 300M mechanism-verdict budget (100M kill / 300M primary
# late-half Δpermutation). Override to 1000000000 only after that gate passes; a
# later 1B run uses its own cosine, not a continuation of this scheduler.
export TARGET_TOKENS="${TARGET_TOKENS:-300000000}"
export SKIP_PRETOKENIZE="${SKIP_PRETOKENIZE:-1}"
export OPTIMIZER=muon
export LEARNING_RATE="${LEARNING_RATE:-0.01}"
export MUON_ADAMW_LR="${MUON_ADAMW_LR:-2e-4}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export CONCEPT_MEMORY_LR=
export WARMUP_STEPS="${WARMUP_STEPS:-500}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
export AUTO_INTERVALS="${AUTO_INTERVALS:-1}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-12}"
export LOGGING_STEPS="${LOGGING_STEPS:-20}"

# Reuse the immutable Gemma-tokenized 4K tree. Start from E17d's bs=8 accum=2;
# recalibrate by real tok/s only if K=256 OOM or underfills.
PROJECT_ROOT_HINT="/home/ksopyla/dev/[REDACTED]"
[ -d "$PROJECT_ROOT_HINT" ] || PROJECT_ROOT_HINT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export DATASETS_TOK_DIR="${DATASETS_TOK_DIR:-${PROJECT_ROOT_HINT}/../hf_home/datasets_tok_gemma_4k}"
export MANIFEST="${MANIFEST:-${DATASETS_TOK_DIR}/${PRETOKENIZE_MIX}_gemma_manifest.json}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

exec bash "${SCRIPT_DIR}/launch_e10.sh"
