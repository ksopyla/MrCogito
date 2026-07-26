#!/bin/bash
# E16a unattended matched pair: Adam control first, then Muon treatment.
# Run only after the separate sustained-LR Muon calibration passes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

TARGET_TOKENS="${TARGET_TOKENS:-100000000}"
MUON_MATRIX_LR="${MUON_MATRIX_LR:-0.01}"

echo "__E16A_ADAM_START__"
EXPERIMENT_ID=E16a OPTIMIZER=adam TARGET_TOKENS="$TARGET_TOKENS" \
    bash "${SCRIPT_DIR}/launch_e16a.sh"
echo "__E16A_ADAM_COMPLETE__"

echo "__E16A_MUON_START__"
EXPERIMENT_ID=E16a OPTIMIZER=muon LEARNING_RATE="$MUON_MATRIX_LR" \
    TARGET_TOKENS="$TARGET_TOKENS" \
    bash "${SCRIPT_DIR}/launch_e16a.sh"
echo "__E16A_MUON_COMPLETE__"
