#!/usr/bin/env bash
# E17 — 4-bank per-global-layer concept memory (the E16b write-path structural fix).
# Spec: docs/experiments_specs/ahead/E17_four_bank_concept_memory.md (+ _plan.md).
#
# Per-layer concept banks make each write "selfish" (a layer reads the bank it wrote last
# block), giving the write gate a clean gradient the shared topology denies it. Identical
# to E16b except CONCEPT_IO_MODE=per_layer_banks (4 private banks, tied ConceptWriteHead,
# identical machinery; only the bank initializations differ).
#
# Polonez 4x RTX 3090, eff batch 72 (per_device 3 x accum 6 x 4), 1B tokens (matching E16b),
# warmup 500, report @100M. No early kill — divergence only (see spec Kill criteria).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_ID="${EXPERIMENT_ID:-E17}"
export CONCEPT_IO_MODE=per_layer_banks
export READ_CONCEPT_NORM=true
export READ_GATE_INIT=0.01
export WRITE_GATE_INIT=0.01
export OPTIMIZER=muon
export LEARNING_RATE="${LEARNING_RATE:-0.01}"
export MUON_ADAMW_LR="${MUON_ADAMW_LR:-2e-4}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export CONCEPT_MEMORY_LR=
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
export PRETOKENIZE_MIX="${PRETOKENIZE_MIX:-e16b_long_4k_v1}"
export TARGET_TOKENS="${TARGET_TOKENS:-1000000000}"
export WARMUP_STEPS="${WARMUP_STEPS:-500}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
export AUTO_INTERVALS="${AUTO_INTERVALS:-1}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-12}"
export SKIP_PRETOKENIZE="${SKIP_PRETOKENIZE:-1}"
export LOGGING_STEPS="${LOGGING_STEPS:-20}"
# e16b_long_4k_v1 lives in the 4k tokenized dir (launch_e10.sh defaults to the non-4k dir).
SCRIPT_DIR_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_HINT="/home/ksopyla/dev/MrCogito"
[ -d "$PROJECT_ROOT_HINT" ] || PROJECT_ROOT_HINT="$(cd "${SCRIPT_DIR_ABS}/.." && pwd)"
export DATASETS_TOK_DIR="${PROJECT_ROOT_HINT}/../hf_home/datasets_tok_gemma_4k"
export MANIFEST="${DATASETS_TOK_DIR}/${PRETOKENIZE_MIX}_gemma_manifest.json"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-3}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-6}"
echo "=== E17 4-bank per-global-layer concept memory (Polonez 4x3090) ==="
echo "  io=${CONCEPT_IO_MODE} seq=${MAX_SEQ_LENGTH} mix=${PRETOKENIZE_MIX} target_tokens=${TARGET_TOKENS}"
echo "  lr=${LEARNING_RATE} muon_adamw_lr=${MUON_ADAMW_LR} wd=${WEIGHT_DECAY} eff_batch=$((PER_DEVICE_BATCH_SIZE*4*GRADIENT_ACCUMULATION_STEPS))"
exec bash "${SCRIPT_DIR}/launch_e10.sh"
