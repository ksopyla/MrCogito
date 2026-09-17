#!/bin/bash
# E28 — exclusive r=16 on Hub CogitoProbe-bits (1k then 4k; fixed vs scaled).
# Spec: docs/experiments_specs/done_failed/E28_exclusive_cogitoprobe_bits.md
# Prefer load_dataset("ksopyla/cogito-probe-bits"). Local generate is fallback only.
# Does NOT launch 32k and does NOT set BATCH_PACKING_MODE=length_group.
#
#   SEQ=1024 VARIANT=fixed PAR_MODE=perceiver bash scripts/launch_e28.sh
#   SEQ=1024 VARIANT=fixed PAR_MODE=dense bash scripts/launch_e28.sh
# Copy Wave A compressor flags via env (PAR_MESSAGE_IDENTITY_SLOTS / prefix AE / key_spans).
# Dense S0 forces those knobs off — message boundary requires perceiver mode.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/remote_paths.sh
source "${SCRIPT_DIR}/remote_paths.sh"

SEQ="${SEQ:-1024}"
VARIANT="${VARIANT:-fixed}"
FAMILY="${FAMILY:-bits}"
# shellcheck source=scripts/skip_e29_guard.sh
source "${SCRIPT_DIR}/skip_e29_guard.sh"
e29_park_if_bind
COGITO_PROBE_ID="${COGITO_PROBE_ID:-ksopyla/cogito-probe-${FAMILY}}"
PROBE_ROOT="${PROBE_ROOT:-$ROOT/Cache/concept_probes/full_1k4k}"
PROBE_SOURCE="${PROBE_SOURCE:-hub}"   # hub | local

if [ "$SEQ" = "32768" ] || [ "$SEQ" = "16384" ] || [ "$SEQ" = "8192" ]; then
  echo "E28 Wave B does not train seq=$SEQ (1k then 4k only). Refuse 8k/16k/32k." >&2
  exit 2
fi

LOCAL_TRAIN="$PROBE_ROOT/hf/$FAMILY/seq${SEQ}/$VARIANT/train"
LOCAL_VAL="$PROBE_ROOT/hf/$FAMILY/seq${SEQ}/$VARIANT/validation"
LOCAL_TEST="$PROBE_ROOT/hf/$FAMILY/seq${SEQ}/$VARIANT/test"

export EXPERIMENT_ID="${EXPERIMENT_ID:-E28}"
export MODEL_FAMILY=perceiver_ar
export OBJECTIVE_VARIANT=causal_lm
export DECODER_TYPE=causal_ar
export TOKENIZER_NAME="${TOKENIZER_NAME:-HuggingFaceTB/SmolLM3-3B}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
export TOKEN_EMBEDDING_DIM="${TOKEN_EMBEDDING_DIM:-256}"
export INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-512}"
export NUM_LAYERS="${NUM_LAYERS:-2}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-1}"
export HEAD_DIM="${HEAD_DIM:-32}"
export NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-8}"
export PAR_MODE="${PAR_MODE:-perceiver}"
export PAR_PRE_LAYERS="${PAR_PRE_LAYERS:-1}"
export PAR_PRE_WINDOW="${PAR_PRE_WINDOW:-512}"
export PAR_GLOBAL_LAYERS="${PAR_GLOBAL_LAYERS:-1}"
export PAR_GLOBAL_LOGIT_SCALE="${PAR_GLOBAL_LOGIT_SCALE:-log}"
export PAR_BLOCK="${PAR_BLOCK:-1024}"
export PAR_NGRAM_BUCKETS="${PAR_NGRAM_BUCKETS:-1024}"
export PAR_VALUE_EMBED_LAYERS="${PAR_VALUE_EMBED_LAYERS:-0,1}"
export PAR_VALUE_EMBED_DIM="${PAR_VALUE_EMBED_DIM:-16}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export ATTN_PAD_MULTIPLE="${ATTN_PAD_MULTIPLE:-1}"
export MAX_SEQ_LENGTH="$SEQ"
export PRESERVE_PRECOMPUTED_LABELS=true
export BATCH_PACKING_MODE=none
export PRETOKENIZE_MIX=""
export SKIP_PRETOKENIZE=1
export NUM_EPOCHS="${NUM_EPOCHS:-12}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export WARMUP_STEPS="${WARMUP_STEPS:-50}"
export LOGGING_STEPS="${LOGGING_STEPS:-20}"
export EVAL_STEPS="${EVAL_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
export AUTO_INTERVALS=0
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

if [ "$PROBE_SOURCE" = "local" ]; then
  if [ ! -d "$LOCAL_TRAIN" ] || [ ! -d "$LOCAL_VAL" ]; then
    echo "PROBE_SOURCE=local but missing $LOCAL_TRAIN" >&2
    exit 1
  fi
  MANIFEST_DIR="${PROBE_ROOT}/manifests"
  mkdir -p "$MANIFEST_DIR"
  MANIFEST="$MANIFEST_DIR/${FAMILY}_seq${SEQ}_${VARIANT}.json"
  python3 - "$MANIFEST" "$LOCAL_TRAIN" "$LOCAL_VAL" "$FAMILY" "$SEQ" "$VARIANT" <<'PY'
import json, sys
from pathlib import Path
man, train, val, family, seq, variant = sys.argv[1:]
Path(man).write_text(json.dumps({
    "mix_id": f"cogito_probe_{family}_{seq}_{variant}",
    "objective": "causal_lm",
    "max_seq_length": int(seq),
    "seed": 20260916,
    "label_policy": "answer_only",
    "sources": [{
        "name": f"{family}_seq{seq}_{variant}",
        "train_path": train,
        "eval_path": val,
        "weight": 1.0,
        "in_eval": True,
    }],
}, indent=2))
print("wrote", man)
PY
  export PRETOKENIZED_MANIFEST="$MANIFEST"
  unset COGITO_PROBE_ID || true
else
  export COGITO_PROBE_ID
  export COGITO_PROBE_SEQ_LEN="$SEQ"
  export COGITO_PROBE_VARIANT="$VARIANT"
  unset PRETOKENIZED_MANIFEST || true
  echo "E28 data: load_dataset($COGITO_PROBE_ID) seq=$SEQ variant=$VARIANT (Hub; no 32k)"
fi

if [ "$PAR_MODE" = "dense" ]; then
  export PAR_MESSAGE_BOUNDARY_TOKEN_ID=-1
  export PAR_MESSAGE_IDENTITY_SLOTS=False
  export PAR_MESSAGE_GLOBAL_ANCHORS=none
  export PAR_MESSAGE_PREFIX_AE=False
  export PAR_MESSAGE_PREFIX_AE_WEIGHT=0.0
  export PAR_MESSAGE_SLOTS_INPLACE=False
else
  if [ -z "${PAR_MESSAGE_BOUNDARY_TOKEN_ID:-}" ] || [ "${PAR_MESSAGE_BOUNDARY_TOKEN_ID}" = "-1" ]; then
    ATOM="${PROBE_ROOT}/atom_table.json"
    if [ -f "$ATOM" ]; then
      PAR_MESSAGE_BOUNDARY_TOKEN_ID="$(python3 -c "import json; print(json.load(open('$ATOM'))['markers']['query']['id'])")"
    else
      PAR_MESSAGE_BOUNDARY_TOKEN_ID="$(uv run python - <<'PY'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B", use_fast=True)
ids = t.encode("Q", add_special_tokens=False)
if len(ids) != 1:
    raise SystemExit(f"marker Q is not 1 token: {ids}")
print(ids[0])
PY
)"
    fi
    export PAR_MESSAGE_BOUNDARY_TOKEN_ID
  fi
  export PAR_MESSAGE_COMPRESS_RATIO="${PAR_MESSAGE_COMPRESS_RATIO:-16}"
  export PAR_MESSAGE_SLOTS_INPLACE="${PAR_MESSAGE_SLOTS_INPLACE:-True}"
fi
bash "$SCRIPT_DIR/train_concept_pretraining_multigpu.sh"

RUN="$(ls -td "$ROOT/Cache/Training/"* | head -1)"
CKPT="$RUN"
BEST="$(ls -td "$RUN"/checkpoint-* 2>/dev/null | head -1 || true)"
[ -n "${BEST:-}" ] && CKPT="$BEST"
mkdir -p "$ROOT/Cache/Evaluation_reports"
EVAL_OUT="$ROOT/Cache/Evaluation_reports/${EXPERIMENT_ID}_${FAMILY}_seq${SEQ}_${VARIANT}_$(basename "$RUN").json"
if [ "$PROBE_SOURCE" = "local" ] && [ -d "$LOCAL_TEST" ]; then
  uv run python "$ROOT/evaluation/evaluate_cogito_probe.py" \
    --checkpoint "$CKPT" --data "$LOCAL_TEST" --out "$EVAL_OUT" \
    || echo "${EXPERIMENT_ID} eval failed (training artifacts kept)"
else
  uv run python "$ROOT/evaluation/evaluate_cogito_probe.py" \
    --checkpoint "$CKPT" --data "$COGITO_PROBE_ID" \
    --seq_len "$SEQ" --variant "$VARIANT" --split test \
    --out "$EVAL_OUT" \
    || echo "${EXPERIMENT_ID} eval failed (training artifacts kept)"
fi
