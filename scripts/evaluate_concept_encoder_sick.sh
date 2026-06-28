#!/bin/bash
# Evaluate Concept Encoder on SICK (relatedness + entailment).
#
# SICK tests two properties critical for concept encoders:
#   - Relatedness (regression): Do concepts preserve semantic similarity?
#   - Entailment (3-class): Do concepts preserve compositional meaning?
#
# Usage:
#   bash scripts/evaluate_concept_encoder_sick.sh                  # both tasks (default)
#   bash scripts/evaluate_concept_encoder_sick.sh sick_relatedness # relatedness only
#   bash scripts/evaluate_concept_encoder_sick.sh sick_entailment  # entailment only
#   bash scripts/evaluate_concept_encoder_sick.sh sick_all         # both explicitly
#
# MODEL_PATH is set directly in this file — update it when a new model is trained.

set -o pipefail

echo "=== SICK Evaluation for Concept Encoder ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/remote_paths.sh
source "${SCRIPT_DIR}/remote_paths.sh"

export TOKENIZERS_PARALLELISM=false

# Python runs via `uv run`; ensure uv is on PATH for non-interactive SSH.
if [ -d "$HOME/.local/bin" ]; then export PATH="$HOME/.local/bin:$PATH"; fi

# =============================================================================
# MODEL TO EVALUATE — update this when a new denoising checkpoint is trained
# =============================================================================
MODEL_PATH="${MODEL_PATH_OVERRIDE:-${PROJECT_ROOT}/Cache/Training/REPLACE_WITH_PERCEIVER_DENOISE_CHECKPOINT}"
MODEL_TYPE="${MODEL_TYPE_OVERRIDE:-perceiver_denoise}"
# =============================================================================

# Benchmark: optional $1, defaults to "sick_all" (both relatedness + entailment)
BENCHMARK="${1:-sick_all}"

echo "  Project Root: $PROJECT_ROOT"
echo "  HF Cache: $HF_HOME"
echo "  Model Type: $MODEL_TYPE"
echo "  Model Path: $MODEL_PATH"
echo "  Benchmark:  $BENCHMARK"
echo ""

if command -v uv > /dev/null 2>&1; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python3"
fi

$PYTHON_CMD evaluation/evaluate_on_benchmark.py \
    --benchmark "$BENCHMARK" \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name "$MODEL_PATH" \
    --batch_size 96 \
    --epochs 10 \
    --learning_rate 1e-5

echo ""
echo "SICK evaluation completed!"
