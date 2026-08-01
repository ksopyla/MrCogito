#!/usr/bin/env bash
# E16b write-init falsification — DIAGNOSTIC pre-check (Odra). Not an experiment spec; a
# knob ablation sanctioned to separate two causes of E16b's dead write gates.
#
# Fresh shared_depth_recurrent run, IDENTICAL to E16b except WRITE_GATE_INIT 0.01 -> 0.3
# (read gate init unchanged at 0.01). ~100M tokens on Odra (3x3090), eff batch 72 (matches
# E16b). Uses the EXISTING shared_depth_recurrent code (no E17 changes) — runs concurrently
# with E17 implementation.
#
# Decisive read (depth_alphas / write gates, logged to W&B as concept_gates/write_*):
#   - STAY OPEN (tanh ~0.29 holds/grows) -> the 0.01 cold-start init was starving the writes
#     (cause 2a); higher init alone is a partial fix.
#   - COLLAPSE back toward ~0 -> the shared-depth topology is the cause (cause 2b), which
#     is exactly what E17's per-layer banks target.
#
# Either outcome informs E17; the run is cheap (~100M, ~12 GPU-h).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
cd /home/ksopyla/dev/MrCogito
export HF_HOME=/home/ksopyla/dev/hf_home
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
mkdir -p Cache/logs
LOG="Cache/logs/train_e16b_writegate_falsification_$(date +%Y%m%d_%H%M%S).log"
echo "uv=$(command -v uv)" | tee "$LOG"
echo "write-init falsification: WRITE_GATE_INIT=0.3 (E16b=0.01), fresh shared_depth_recurrent, 100M tok, Odra 3x3090" | tee -a "$LOG"

export EXPERIMENT_ID=E16b_fg
export CONCEPT_IO_MODE=shared_depth_recurrent
export READ_CONCEPT_NORM=true
export READ_GATE_INIT=0.01
export WRITE_GATE_INIT=0.3
export OPTIMIZER=muon
export LEARNING_RATE=0.01
export MUON_ADAMW_LR=2e-4
export MUON_MOMENTUM=0.95
export WEIGHT_DECAY=0.1
export CONCEPT_MEMORY_LR=
export MAX_SEQ_LENGTH=4096
export PRETOKENIZE_MIX=e16b_long_4k_v1
export TARGET_TOKENS=100000000
export WARMUP_STEPS=50
export AUTO_INTERVALS=1
export SAVE_TOTAL_LIMIT=6
export SKIP_PRETOKENIZE=1
export PER_DEVICE_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=6
bash scripts/launch_e10.sh 2>&1 | tee -a "$LOG"
echo "DONE write-init falsification" | tee -a "$LOG"
