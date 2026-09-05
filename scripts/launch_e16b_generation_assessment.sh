#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
cd /home/ksopyla/dev/MrCogito
export HF_HOME=/home/ksopyla/dev/hf_home
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
mkdir -p Cache/logs Cache/Evaluation_reports
LOG="Cache/logs/eval_e16b_generation_assessment_$(date +%Y%m%d_%H%M%S).log"
echo "uv=$(command -v uv)" | tee "$LOG"
# Scale: 6 prompts x (e16b real/zero + base) continuation, 4 chat, 2 sample,
# plus 2-doc context sweep at 128/512/1024/2048 with 128 new tokens.
uv run python analysis/run_e16b_generation_assessment.py \
  --e16b_path Cache/Training/backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850/checkpoint-7900 \
  --output_json Cache/Evaluation_reports/e16b_ckpt7900_generation_assessment_scale.json \
  --max_new_tokens 256 \
  --ctx_max_new_tokens 128 \
  --max_prompts 6 \
  --n_ctx_docs 2 \
  --sample \
  2>&1 | tee -a "$LOG"
