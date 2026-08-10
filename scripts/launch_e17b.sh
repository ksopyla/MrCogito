#!/usr/bin/env bash
# E17b — E17 per-layer banks with mid write-gate init (WRITE_GATE_INIT=0.1).
# Spec: docs/experiments_specs/ahead/E17b_per_layer_mid_write_init.md (+ _plan.md).
#
# Single variable vs E17: WRITE_GATE_INIT 0.01 → 0.1. Self-contained so it does not
# depend on editing launch_e17.sh defaults.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_ID="${EXPERIMENT_ID:-E17b}"
export CONCEPT_IO_MODE="${CONCEPT_IO_MODE:-per_layer_banks}"
export READ_CONCEPT_NORM="${READ_CONCEPT_NORM:-true}"
export READ_GATE_INIT="${READ_GATE_INIT:-0.01}"
export WRITE_GATE_INIT="${WRITE_GATE_INIT:-0.1}"
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
SCRIPT_DIR_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_HINT="$(cd "${SCRIPT_DIR_ABS}/.." && pwd)"
export DATASETS_TOK_DIR="${DATASETS_TOK_DIR:-${PROJECT_ROOT_HINT}/../hf_home/datasets_tok_gemma_4k}"
export MANIFEST="${MANIFEST:-${DATASETS_TOK_DIR}/${PRETOKENIZE_MIX}_gemma_manifest.json}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
echo "=== ${EXPERIMENT_ID} per-layer + mid write init ${WRITE_GATE_INIT} ==="
echo "  io=${CONCEPT_IO_MODE} write_gate_init=${WRITE_GATE_INIT} seq=${MAX_SEQ_LENGTH} mix=${PRETOKENIZE_MIX}"
echo "  batch/gpu=${PER_DEVICE_BATCH_SIZE} accum=${GRADIENT_ACCUMULATION_STEPS} (Polonez 2026-08-10 calib: bs8 peak~21.5GiB, best samples/s; bs10 ok but slower; bs12 OOM)"
exec bash "${SCRIPT_DIR}/launch_e10.sh"
