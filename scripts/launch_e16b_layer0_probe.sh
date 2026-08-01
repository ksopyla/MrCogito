#!/usr/bin/env bash
# E16b Layer-0 generation probe — extends the 2026-08-01 assessment with the two
# conditions that report recommended but had NOT yet tested:
#   1. repetition_penalty (the decode band-aid) — does penalising seen tokens
#      break the greedy fixed-point loops?
#   2. concept_mode "frozen" — encode the prompt into z WITH writes, then decode
#      read-only (no writes from the model's own tokens). If frozen ≫ real on
#      diversity, it confirms self-generated concept writes poison free-run and
#      defines the right inference contract (and motivates the Layer-1 training bet).
#
# Same prompt bank + base Gemma control + prompt-context sweep as the 2026-08-01
# scale run, so results are directly comparable to
# Cache/Evaluation_reports/e16b_ckpt7900_generation_assessment_scale.json.
#
# Generation axis is extended to 1024 tokens (cutoffs 32..1024) for a longer
# collapse/recovery view. Chat probe is skipped (already shown to fail on both
# base and E16b in the prior report).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
cd /home/ksopyla/dev/MrCogito
export HF_HOME=/home/ksopyla/dev/hf_home
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
mkdir -p Cache/logs Cache/Evaluation_reports
LOG="Cache/logs/eval_e16b_layer0_probe_$(date +%Y%m%d_%H%M%S).log"
echo "uv=$(command -v uv)" | tee "$LOG"
echo "Layer-0 probe: rp=1.2, modes=real/zero/frozen, +sample, +base, gen->1024" | tee -a "$LOG"
uv run python analysis/run_e16b_generation_assessment.py \
  --e16b_path Cache/Training/backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850/checkpoint-7900 \
  --output_json Cache/Evaluation_reports/e16b_ckpt7900_layer0_probe.json \
  --max_new_tokens 1024 \
  --ctx_max_new_tokens 256 \
  --max_prompts 4 \
  --n_ctx_docs 2 \
  --skip_chat \
  --sample \
  --repetition_penalty 1.2 \
  --extra_concept_modes frozen \
  2>&1 | tee -a "$LOG"
echo "DONE: Cache/Evaluation_reports/e16b_ckpt7900_layer0_probe.json" | tee -a "$LOG"
