#!/bin/bash
# Compatibility wrapper for the historical generic launcher name.
# New commands should use scripts/train_concept_pretraining_multigpu.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_concept_pretraining_multigpu.sh" "$@"
