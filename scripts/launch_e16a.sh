#!/bin/bash
# E16a — matched optimizer A/B on the implemented E16 shared-depth workspace.
# Pins the common 2K/100M protocol and delegates to launch_e10.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_ID="${EXPERIMENT_ID:-E16a}"
export CONCEPT_IO_MODE=shared_depth_recurrent
export READ_CONCEPT_NORM=true
export READ_GATE_INIT=0.01
export WRITE_GATE_INIT=0.01
export TARGET_TOKENS="${TARGET_TOKENS:-100000000}"
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
export AUTO_INTERVALS="${AUTO_INTERVALS:-1}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-12}"
export SKIP_PRETOKENIZE="${SKIP_PRETOKENIZE:-1}"

OPTIMIZER="${OPTIMIZER:-adam}"
export OPTIMIZER
case "$OPTIMIZER" in
    adam)
        export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
        export CONCEPT_MEMORY_LR="${CONCEPT_MEMORY_LR:-3e-4}"
        export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
        ;;
    muon)
        export LEARNING_RATE="${LEARNING_RATE:-0.01}"
        export MUON_ADAMW_LR="${MUON_ADAMW_LR:-2e-4}"
        export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
        export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
        # Differential role-based LR is intentionally unsupported by Muon.
        export CONCEPT_MEMORY_LR=""
        ;;
    *)
        echo "ERROR: E16a OPTIMIZER must be 'adam' or 'muon'; got '$OPTIMIZER'." >&2
        exit 2
        ;;
esac

echo "=== E16a ${OPTIMIZER} arm ==="
echo "  target_tokens=${TARGET_TOKENS} matrix_or_lora_lr=${LEARNING_RATE} weight_decay=${WEIGHT_DECAY}"
if [ "$OPTIMIZER" = "muon" ]; then
    echo "  muon_adamw_lr=${MUON_ADAMW_LR} momentum=${MUON_MOMENTUM}"
else
    echo "  concept_memory_lr=${CONCEPT_MEMORY_LR}"
fi

exec bash "${SCRIPT_DIR}/launch_e10.sh"
