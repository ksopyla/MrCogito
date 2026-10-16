#!/usr/bin/env bash
# E35 — latent messages between agents: one training run, then its exams. Thin wrapper over the shared launcher
# (scripts/train_concept_pretraining_multigpu.sh, MODEL_FAMILY=a2a). Spec: docs/experiments_specs/ahead/E35_latent_messages.md
#
#   RUN=Cache/e35/k16 GPU_IDS=0,1,2 NUM_GPUS=3 GRADIENT_ACCUMULATION_STEPS=4 A2A_MSG_SLOTS=16 bash scripts/launch_e35_a2a.sh
#
# Resumable: a finished training leaves TRAIN_DONE, a finished exam its JSON; a re-run skips both.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=.
source scripts/remote_paths.sh >/dev/null 2>&1 || true
RUN="${RUN:?set RUN=<run folder>}"
DATA="${DATA:-$DATASETS_TOK_DIR/text_checks_v2q8b4_a2a}"
EXAMS="${EXAMS:-$DATASETS_TOK_DIR/text_checks_v2q8/eval}"
INIT="${INIT:-Cache/e35/init}"
EXAM_LENGTHS="${EXAM_LENGTHS:-1024 4096 16384}"
CONTROL_ITEMS="${CONTROL_ITEMS:-100}"   # items per cell for the empty / mismatched-message controls
EXAM_EXTRA="${EXAM_EXTRA:-}"            # extra exam arguments for every exam, e.g. "--tasks lookup keyed --max_items_per_cell 100"
export NUM_GPUS="${NUM_GPUS:-1}" GPU_IDS="${GPU_IDS:-$(seq -s, 0 $(( ${NUM_GPUS:-1} - 1 )))}"

export MODEL_FAMILY=a2a OBJECTIVE_VARIANT=causal_lm DECODER_TYPE=causal_ar
export A2A_SENDER_INIT="${A2A_SENDER_INIT:-$INIT/sender}" A2A_RECEIVER_INIT="${A2A_RECEIVER_INIT:-$INIT/receiver}"
export A2A_MSG_SLOTS="${A2A_MSG_SLOTS:-16}" A2A_TX_LAYERS="${A2A_TX_LAYERS:-2}" A2A_TX_DIM="${A2A_TX_DIM:-512}"
export A2A_TX_ORDER="${A2A_TX_ORDER:-rank}"
export PRETOKENIZED_MANIFEST="$DATA/manifest.json" TOKENIZER_NAME="$DATA/tokenizer" MAX_SEQ_LENGTH=4096
# receiver shape (read by the launcher for its logs; the model itself comes from the receiver checkpoint)
export HIDDEN_SIZE=576 TOKEN_EMBEDDING_DIM=256 HEAD_DIM=64 NUM_ATTENTION_HEADS=9 NUM_KV_HEADS=1 PAR_MODE=dense
export PAR_NGRAM_ORDERS='' PAR_VALUE_EMBED_LAYERS='' ATTN_BACKEND="${ATTN_BACKEND:-flex}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}" GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export LEARNING_RATE="${LEARNING_RATE:-5e-4}" MAX_STEPS="${MAX_STEPS:-3000}" WARMUP_STEPS="${WARMUP_STEPS:-60}"
export LR_SCHEDULER_TYPE=cosine WEIGHT_DECAY=0.1 MAX_GRAD_NORM=1.0 OPTIMIZER=adam LOGIT_SOFTCAP=30.0 Z_LOSS=0.0001
export BATCH_PACKING_MODE=length_group CHUNKED_CE_BLOCK_SIZE=2048 DATALOADER_NUM_WORKERS=2 EVAL_BATCH_SIZE=8
export EVAL_STEPS="${EVAL_STEPS:-500}" SAVE_STEPS="${SAVE_STEPS:-500}" SAVE_TOTAL_LIMIT=3 LOGGING_STEPS=25
export MAX_EVAL_SAMPLES=256 LOAD_BEST_MODEL_AT_END=False SEED="${SEED:-0}"
export TRAIN_OUTPUT_DIR="$RUN/train" EXPERIMENT_ID="E35-$(basename "$RUN")" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN"
stamp() { echo "$1 $(date '+%Y-%m-%d %H:%M %Z')" | tee -a "$RUN/events.log"; }

if [ ! -f "$RUN/TRAIN_DONE" ]; then
    CK=$(ls -d "$RUN"/train/*/checkpoint-* 2>/dev/null | awk -F- '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2- || true)
    export RESUME_FROM_CHECKPOINT="${CK:-}"
    stamp "train start (K=$A2A_MSG_SLOTS, order=$A2A_TX_ORDER, resume=${CK:-none})"
    if bash scripts/train_concept_pretraining_multigpu.sh > "$RUN/train.log" 2>&1 && ls -d "$RUN"/train/*/final >/dev/null 2>&1; then
        touch "$RUN/TRAIN_DONE"; stamp "train done"
    else
        stamp "train FAILED (see train.log)"; exit 1
    fi
fi
FINAL=$(ls -d "$RUN"/train/*/final | head -1)
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
exam() {  # exam <gpu> <message> <extra args...>
    local g=$1 m=$2; shift 2
    [ -f "$RUN/exam_$m.json" ] && return 0
    CUDA_VISIBLE_DEVICES=$g uv run python evaluation/text_checks_eval.py --checkpoint "$FINAL" --tokenizer "$TOKENIZER_NAME" \
        --items "$EXAMS/id.jsonl" "$EXAMS/steps.jsonl" --lengths $EXAM_LENGTHS --a2a_message "$m" $EXAM_EXTRA "$@" \
        --out "$RUN/exam_$m.json" > "$RUN/exam_$m.log" 2>&1 && stamp "exam $m done" || stamp "exam $m FAILED"
}
stamp "exams start"
if [ "${EXAM_SERIAL:-0}" = 1 ]; then  # one GPU shared with another job: one exam at a time
    exam "${GPUS[0]}" real
    exam "${GPUS[0]}" none --max_items_per_cell "$CONTROL_ITEMS"
    exam "${GPUS[0]}" swapped --max_items_per_cell "$CONTROL_ITEMS"
else
    exam "${GPUS[0]}" real &
    exam "${GPUS[1]:-${GPUS[0]}}" none --max_items_per_cell "$CONTROL_ITEMS" &
    exam "${GPUS[2]:-${GPUS[0]}}" swapped --max_items_per_cell "$CONTROL_ITEMS" &
    wait
fi
stamp "exams done"
touch "$RUN/DONE"
