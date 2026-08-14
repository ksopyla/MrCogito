#!/bin/bash
# E17c — depth-private gated working memory under causal carry pressure.
# Pins only the registered E17c differences, then delegates through E10's Gemma
# protocol to the shared multi-GPU launcher. Caller overrides remain available for
# calibration controls such as MAX_STEPS, REPORT_TO, and SAVE/EVAL_STRATEGY.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_ID="${EXPERIMENT_ID:-E17c}"
export CONCEPT_IO_MODE=per_layer_banks
export CONCEPT_READ_MODE=dedicated
export TIE_CONCEPT_WRITER=false
export CONCEPT_WRITE_MODE=gated_replace
export WRITE_UPDATE_GATE_INIT=0.25
export MEMORY_CARRY_DROPOUT=0.5
export MEMORY_PRESSURE_TOKENS=64
export MEMORY_PRESSURE_WEIGHT=4.0
export READ_CONCEPT_NORM=true
export READ_GATE_INIT=0.1

# Match E16b/E17/E17b data, optimization, and token-budget protocol.
export MAX_SEQ_LENGTH=4096
export PRETOKENIZE_MIX=e16b_long_4k_v1
export TARGET_TOKENS="${TARGET_TOKENS:-1000000000}"
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

# Reuse the immutable Gemma-tokenized 4K tree and effective batch 72 used by E16b/E17.
PROJECT_ROOT_HINT="/home/ksopyla/dev/[REDACTED]"
[ -d "$PROJECT_ROOT_HINT" ] || PROJECT_ROOT_HINT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export DATASETS_TOK_DIR="${DATASETS_TOK_DIR:-${PROJECT_ROOT_HINT}/../hf_home/datasets_tok_gemma_4k}"
export MANIFEST="${MANIFEST:-${DATASETS_TOK_DIR}/${PRETOKENIZE_MIX}_gemma_manifest.json}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-3}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-6}"

exec bash "${SCRIPT_DIR}/launch_e10.sh"
