#!/bin/bash
# E16b — bold long-context Muon scale-up of the E16 shared-depth workspace.
# 4K sequences, long-document mix, Muon, 1B non-padding tokens.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_ID="${EXPERIMENT_ID:-E16b}"
export CONCEPT_IO_MODE=shared_depth_recurrent
export READ_CONCEPT_NORM=true
export READ_GATE_INIT=0.01
export WRITE_GATE_INIT=0.01

export OPTIMIZER=muon
export LEARNING_RATE="${LEARNING_RATE:-0.01}"
export MUON_ADAMW_LR="${MUON_ADAMW_LR:-2e-4}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export CONCEPT_MEMORY_LR=""

export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
export PRETOKENIZE_MIX="${PRETOKENIZE_MIX:-smollm3_inspired_4k_e16b}"
export TARGET_TOKENS="${TARGET_TOKENS:-1000000000}"
export WARMUP_STEPS="${WARMUP_STEPS:-500}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
export AUTO_INTERVALS="${AUTO_INTERVALS:-1}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-12}"
export SKIP_PRETOKENIZE="${SKIP_PRETOKENIZE:-1}"
export LOGGING_STEPS="${LOGGING_STEPS:-20}"

# 4K default microbatch is conservative; override after Odra calibration.
# Effective batch target remains 72: per_device * num_gpus * accum.
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-6}"

echo "=== E16b Muon long-context launch ==="
echo "  seq=${MAX_SEQ_LENGTH} mix=${PRETOKENIZE_MIX} target_tokens=${TARGET_TOKENS}"
echo "  lr=${LEARNING_RATE} muon_adamw_lr=${MUON_ADAMW_LR} wd=${WEIGHT_DECAY}"
echo "  batch/gpu=${PER_DEVICE_BATCH_SIZE} accum=${GRADIENT_ACCUMULATION_STEPS}"

exec bash "${SCRIPT_DIR}/launch_e10.sh"
