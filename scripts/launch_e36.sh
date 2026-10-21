#!/usr/bin/env bash
# E36 — a trained writer for a frozen reader. Thin wrapper over the shared launcher
# (scripts/train_concept_pretraining_multigpu.sh, MODEL_FAMILY=frozen_reader) and evaluation/frozen_reader_eval.py.
# Note: docs/4_Research_Notes/e36_frozen_reader_writer_20261018.md
#
#   S0, reader only (floor / ceiling / Cramming probe, no training):
#     RUN=Cache/e36/s0_smollm2_360m S0_READER=HuggingFaceTB/SmolLM2-360M GPU_IDS=3 bash scripts/launch_e36.sh
#   S1, task-free writer (continuation + 20 % reconstruction), then its evaluation:
#     RUN=Cache/e36/s1_input_s512 GPU_IDS=0,1 NUM_GPUS=2 FR_WAY=input bash scripts/launch_e36.sh
#
# Resumable: a finished training leaves TRAIN_DONE, a finished evaluation its JSON; a re-run skips both.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=.
source scripts/remote_paths.sh >/dev/null 2>&1 || true
RUN="${RUN:?set RUN=<run folder>}"
MANIFEST="${MANIFEST:-$DATASETS_TOK_DIR/e36_longdoc_4k6_v1_manifest.json}"
EVAL_ROWS="${EVAL_ROWS:-256}"          # held-out rows per source for the continuation measure
SQUAD_ITEMS="${SQUAD_ITEMS:-500}"
EVAL_EXTRA="${EVAL_EXTRA:-}"
export NUM_GPUS="${NUM_GPUS:-1}" GPU_IDS="${GPU_IDS:-$(seq -s, 0 $(( ${NUM_GPUS:-1} - 1 )))}"
mkdir -p "$RUN"
stamp() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }

if [ -n "${S0_READER:-}" ]; then
    OUT="$RUN/s0.json"
    [ -f "$OUT" ] && { echo "$(stamp) S0 done: $OUT"; exit 0; }
    echo "$(stamp) S0 reader=$S0_READER gpu=$GPU_IDS" | tee -a "$RUN/launch.log"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" uv run python evaluation/frozen_reader_eval.py --reader "$S0_READER" \
        --manifest "$MANIFEST" --rows_per_source "$EVAL_ROWS" --squad_items "$SQUAD_ITEMS" --cramming \
        --out "$OUT" $EVAL_EXTRA 2>&1 | tee -a "$RUN/s0.log"
    echo "$(stamp) S0 finished" | tee -a "$RUN/launch.log"
    exit 0
fi

export MODEL_FAMILY=frozen_reader OBJECTIVE_VARIANT=causal_lm DECODER_TYPE=causal_ar
export FR_READER="${FR_READER:-HuggingFaceTB/SmolLM2-360M}" FR_WRITER_INIT="${FR_WRITER_INIT:-HuggingFaceTB/SmolLM2-135M}"
export FR_WRITER_LAYERS="${FR_WRITER_LAYERS:-3}" FR_SPAN="${FR_SPAN:-512}" FR_MEM_SPANS="${FR_MEM_SPANS:-8}"
export FR_WAY="${FR_WAY:-input}" FR_RECON_FRAC="${FR_RECON_FRAC:-0.2}" FR_ENTRY_POSITIONS="${FR_ENTRY_POSITIONS:-span_mid}"
export FR_WRITER_POOL="${FR_WRITER_POOL:-mean}" FR_CURRICULUM="${FR_CURRICULUM:-}"
export FR_OBJECTIVE="${FR_OBJECTIVE:-continuation}" FR_INIT_CHECKPOINT="${FR_INIT_CHECKPOINT:-}"
export PRETOKENIZED_MANIFEST="$MANIFEST" TOKENIZER_NAME="$FR_READER" MAX_SEQ_LENGTH=4610
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}" GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}" MAX_STEPS="${MAX_STEPS:-6000}" WARMUP_STEPS="${WARMUP_STEPS:-200}"
export LR_SCHEDULER_TYPE=cosine WEIGHT_DECAY=0.01 MAX_GRAD_NORM=1.0 OPTIMIZER=adam
export BATCH_PACKING_MODE="${BATCH_PACKING_MODE:-none}" DATALOADER_NUM_WORKERS=2 EVAL_BATCH_SIZE=8 GRADIENT_CHECKPOINTING=False
export EVAL_STEPS="${EVAL_STEPS:-500}" SAVE_STEPS="${SAVE_STEPS:-500}" SAVE_TOTAL_LIMIT=3 LOGGING_STEPS=25
export MAX_EVAL_SAMPLES=256 LOAD_BEST_MODEL_AT_END=False SEED="${SEED:-0}"
export TRAIN_OUTPUT_DIR="$RUN/train" EXPERIMENT_ID="E36-$(basename "$RUN")" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$RUN/TRAIN_DONE" ]; then
    echo "$(stamp) S1 train way=$FR_WAY span=$FR_SPAN M=$FR_MEM_SPANS layers=$FR_WRITER_LAYERS gpus=$GPU_IDS" \
        | tee -a "$RUN/launch.log"
    bash scripts/train_concept_pretraining_multigpu.sh >> "$RUN/train.log" 2>&1
    FINAL=$(find "$RUN/train" -maxdepth 3 -type d -name final | head -1)
    [ -n "$FINAL" ] || { echo "$(stamp) training left no final checkpoint" | tee -a "$RUN/launch.log"; exit 1; }
    touch "$RUN/TRAIN_DONE"
    echo "$(stamp) S1 train finished: $FINAL" | tee -a "$RUN/launch.log"
fi
FINAL=$(find "$RUN/train" -maxdepth 3 -type d -name final | head -1)
OUT="$RUN/eval_final.json"
if [ ! -f "$OUT" ]; then
    echo "$(stamp) S1 eval $FINAL" | tee -a "$RUN/launch.log"
    CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" uv run python evaluation/frozen_reader_eval.py --checkpoint "$FINAL" \
        --manifest "$MANIFEST" --rows_per_source "$EVAL_ROWS" --squad_items "$SQUAD_ITEMS" --out "$OUT" $EVAL_EXTRA \
        2>&1 | tee -a "$RUN/eval.log"
    echo "$(stamp) S1 eval finished: $OUT" | tee -a "$RUN/launch.log"
fi
