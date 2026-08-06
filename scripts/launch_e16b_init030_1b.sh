#!/usr/bin/env bash
# E16b + WRITE_GATE_INIT=0.3 — finish to 1B (resume from the 100M falsification checkpoint).
#
# The 100M write-init falsification (run ...20260801_203613) OPENED the write gates (0.20–0.32) and
# recovered free-run diversity (distinct-1 0.04→0.29). This resumes it from checkpoint-791 (~100M
# tokens) and trains to 1B total tokens (matching E16b's budget) on a fresh 1B schedule, producing
# the full init-0.3 model for a proper generation-quality assessment. Same shared_depth_recurrent
# architecture as E16b; only the write-gate init differs (0.3 vs 0.01).
#
# Measured throughput (2026-08-03, W&B; corrected after a like-for-like comparison):
#   This run (Odra resumed segment, step 800->4000): ~37.5 s/step.
#   E16b original (Odra resumed segment, step 4000->7900): ~34.7 s/step — within ~8%.
#   E16b's summary "17.45 s/step" was a whole-run average contaminated by a faster pre-resume
#   segment (it itself resumed from checkpoint-3950) — NOT comparable; there is NO 2.15x
#   regression and NO thermal throttle. init-0.3's GPUs are fully utilized (~347 W / 82C); E16b's
#   were under-utilized/idle-stalled (40-51% power, 70-79C). Configs identical (bf16,
#   grad-checkpointing, batch 4 x accum 6, seq 4096, muon, 3 GPUs). ~37.5 s/step is the normal
#   Odra resumed-segment rate -> completion ~Aug 5 (~11:00Z). Steps for 1B = 7905 (unchanged).
#
# Resume semantics: HF Trainer recreates the LR scheduler for the new TARGET_TOKENS at init, then
# loads last_epoch=791 from the checkpoint — so the LR continues at the 1B-schedule value (near
# peak at step 791), not the decayed 100M value. Optimizer (Muon) state is loaded from the checkpoint.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_ID="${EXPERIMENT_ID:-E16b_init030}"
export CONCEPT_IO_MODE=shared_depth_recurrent
export READ_CONCEPT_NORM=true
export READ_GATE_INIT=0.01
export WRITE_GATE_INIT=0.3
export OPTIMIZER=muon
export LEARNING_RATE="${LEARNING_RATE:-0.01}"
export MUON_ADAMW_LR="${MUON_ADAMW_LR:-2e-4}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export CONCEPT_MEMORY_LR=
export MAX_SEQ_LENGTH=4096
export PRETOKENIZE_MIX=e16b_long_4k_v1
export TARGET_TOKENS=1000000000
export WARMUP_STEPS=500
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
export AUTO_INTERVALS=1
export SAVE_TOTAL_LIMIT=12
export SKIP_PRETOKENIZE=1
export LOGGING_STEPS=20
SCRIPT_DIR_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_HINT="/home/ksopyla/dev/MrCogito"
[ -d "$PROJECT_ROOT_HINT" ] || PROJECT_ROOT_HINT="$(cd "${SCRIPT_DIR_ABS}/.." && pwd)"
export DATASETS_TOK_DIR="${PROJECT_ROOT_HINT}/../hf_home/datasets_tok_gemma_4k"
export MANIFEST="${DATASETS_TOK_DIR}/${PRETOKENIZE_MIX}_gemma_manifest.json"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-${PROJECT_ROOT_HINT}/Cache/Training/backbone_concept_gemma_3_1b_pt_K512_concept_20260801_203613/checkpoint-791}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-6}"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
echo "=== E16b init-0.3 -> 1B (resume from checkpoint; host GPUs=${NGPU}) ==="
echo "  io=${CONCEPT_IO_MODE} write_gate_init=${WRITE_GATE_INIT} seq=${MAX_SEQ_LENGTH} target_tokens=${TARGET_TOKENS} warmup=${WARMUP_STEPS}"
echo "  resume=${RESUME_FROM_CHECKPOINT}  per_device=${PER_DEVICE_BATCH_SIZE} accum=${GRADIENT_ACCUMULATION_STEPS} eff_batch=$((PER_DEVICE_BATCH_SIZE*NGPU*GRADIENT_ACCUMULATION_STEPS)) (${NGPU} GPUs)"
exec bash "${SCRIPT_DIR}/launch_e10.sh"
