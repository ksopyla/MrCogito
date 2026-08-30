#!/bin/bash
# E17e 300M evaluation suite (experiment-evaluate tiers).
# Run on Polonez in byobu: byobu new-session -s EVAL_E17e
# Best = checkpoint-2660, last = checkpoint-2668.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/remote_paths.sh
source "${SCRIPT_DIR}/remote_paths.sh"
cd "${PROJECT_ROOT}"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ID="${RUN_ID:-backbone_concept_gemma_3_1b_pt_K256_concept_20260822_120601}"
RUN_ROOT="${RUN_ROOT:-Cache/Training/${RUN_ID}}"
BEST="${BEST:-${RUN_ROOT}/checkpoint-2660}"
LAST="${LAST:-${RUN_ROOT}/checkpoint-2668}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/../hf_home/datasets_tok_gemma_4k/e16b_long_4k_v1_gemma_manifest.json}"
REPORTS="${PROJECT_ROOT}/Cache/Evaluation_reports"
mkdir -p "$REPORTS" "$REPORTS/compute_audit" Cache/logs

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="Cache/logs/eval_E17e_${STAMP}.log"
exec > >(python3 "${SCRIPT_DIR}/clean_tee.py" "$LOG") 2>&1

echo "=== E17e eval ${STAMP} ==="
echo "HEAD=$(git rev-parse --short HEAD) $(git log -1 --oneline)"
echo "BEST=$BEST"
echo "LAST=$LAST"
echo "MANIFEST=$MANIFEST"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
test -d "$BEST"
test -d "$LAST"
test -f "$MANIFEST"

FAILED=()
run_cmd() {
  local name="$1"; shift
  echo
  echo "======== $name ========"
  set +e
  "$@"
  local code=$?
  set -e
  if [ "$code" -ne 0 ]; then
    FAILED+=("$name:$code")
    echo "FAILED $name exit=$code"
  else
    echo "OK $name"
  fi
}

# --- Compute audit (run-level, no GPU) ---
run_cmd compute_audit uv run python analysis/run_compute_audit.py \
  --run-id "$RUN_ID" \
  --out-dir Cache/Evaluation_reports/compute_audit/

tier1() {
  local tag="$1" ckpt="$2" minlen="${3:-0}"
  if [ "$minlen" -gt 0 ]; then
    uv run python analysis/run_concept_analysis.py \
      --model_path "$ckpt" \
      --model_type backbone_concept \
      --eval_source pretokenized \
      --pretokenized_manifest "$MANIFEST" \
      --output_json "${REPORTS}/${RUN_ID}_${tag}_late_bin_gate.json" \
      --num_batches 24 --batch_size 2 --max_seq_length 4096 \
      --min_seq_length "$minlen" \
      --length_buckets 3072 \
      --ablation_batches 24 \
      --no_generation_eval
  else
    uv run python analysis/run_concept_analysis.py \
      --model_path "$ckpt" \
      --model_type backbone_concept \
      --eval_source pretokenized \
      --pretokenized_manifest "$MANIFEST" \
      --output_json "${REPORTS}/${RUN_ID}_${tag}_concept_analysis.json" \
      --num_batches 24 --batch_size 2 --max_seq_length 4096 \
      --length_buckets 1024,2048,3072 \
      --ablation_batches 24 \
      --no_generation_eval
  fi
}

# --- Best ---
run_cmd health_best uv run python -c "from analysis.check_model_health import inspect_weights_detailed; import sys; sys.exit(0 if inspect_weights_detailed('$BEST') is not False else 1)"

run_cmd t1_geom_best tier1 best "$BEST" 0
run_cmd t1_late_best tier1 best "$BEST" 2048

run_cmd t15_best uv run python analysis/run_generation_quality.py \
  --model_path "$BEST" \
  --model_type backbone_concept \
  --no_suffix_ce \
  --free_generation_max_new_tokens 512 \
  --length_cutoffs 64 128 256 512 \
  --prompt_styles continuation chat \
  --concept_modes real zero shuffle static \
  --max_prompts 6 \
  --output_json "${REPORTS}/${RUN_ID}_best_generation_quality.json"

run_cmd assess_best uv run python analysis/run_e16b_generation_assessment.py \
  --e16b_path "$BEST" \
  --output_json "${REPORTS}/${RUN_ID}_ckpt2660_generation_assessment.json" \
  --max_prompts 6 \
  --skip_context_sweep

run_cmd stsb_best uv run python evaluation/evaluate_on_benchmark.py \
  --benchmark stsb_zero_shot \
  --model_type backbone_concept \
  --model_name_or_path "$BEST" --tokenizer_name google/gemma-3-1b-pt \
  --batch_size 32 --max_length 128

run_cmd sick_best env MODEL_PATH_OVERRIDE="$BEST" MODEL_TYPE_OVERRIDE=backbone_concept \
  TOKENIZER_NAME_OVERRIDE=google/gemma-3-1b-pt \
  bash scripts/evaluate_concept_encoder_sick.sh sick_all

run_cmd paws_best env MODEL_PATH_OVERRIDE="$BEST" MODEL_TYPE_OVERRIDE=backbone_concept \
  TOKENIZER_NAME_OVERRIDE=google/gemma-3-1b-pt \
  bash scripts/evaluate_concept_encoder_paws.sh

run_cmd glue_best env MODEL_PATH_OVERRIDE="$BEST" MODEL_TYPE_OVERRIDE=backbone_concept \
  TOKENIZER_NAME_OVERRIDE=google/gemma-3-1b-pt \
  bash scripts/evaluate_concept_encoder_glue.sh all

# --- Last ---
run_cmd health_last uv run python -c "from analysis.check_model_health import inspect_weights_detailed; import sys; sys.exit(0 if inspect_weights_detailed('$LAST') is not False else 1)"

run_cmd t1_geom_last tier1 last "$LAST" 0
run_cmd t1_late_last tier1 last "$LAST" 2048

run_cmd t15_last uv run python analysis/run_generation_quality.py \
  --model_path "$LAST" \
  --model_type backbone_concept \
  --no_suffix_ce \
  --free_generation_max_new_tokens 512 \
  --length_cutoffs 64 128 256 512 \
  --prompt_styles continuation chat \
  --concept_modes real zero shuffle static \
  --max_prompts 6 \
  --output_json "${REPORTS}/${RUN_ID}_last_generation_quality.json"

run_cmd assess_last uv run python analysis/run_e16b_generation_assessment.py \
  --e16b_path "$LAST" \
  --output_json "${REPORTS}/${RUN_ID}_ckpt2668_generation_assessment.json" \
  --max_prompts 6 \
  --skip_context_sweep

echo
echo "FAILED: ${FAILED[*]:-none}"
echo "LOG=$LOG"
