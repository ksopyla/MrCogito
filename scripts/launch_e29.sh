#!/bin/bash
# E29 — exclusive slots on Hub CogitoProbe-bind (hop vs gist). Props is a
# gist-control eval on the same checkpoint, not a second train.
# Spec: docs/experiments_specs/ahead/E29_exclusive_cogitoprobe_bind.md
#   SEQ=1024 bash scripts/launch_e29.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export EXPERIMENT_ID="${EXPERIMENT_ID:-E29}"
export FAMILY=bind
export VARIANT="${VARIANT:-fixed}"
export PAR_MODE="${PAR_MODE:-perceiver}"
export PAR_MESSAGE_IDENTITY_SLOTS="${PAR_MESSAGE_IDENTITY_SLOTS:-True}"
export PAR_MESSAGE_GLOBAL_ANCHORS="${PAR_MESSAGE_GLOBAL_ANCHORS:-none}"
export COGITO_PROBE_ID="${COGITO_PROBE_ID:-ksopyla/cogito-probe-bind}"
export PROBE_SOURCE="${PROBE_SOURCE:-hub}"
bash "$SCRIPT_DIR/launch_e28.sh"

SEQ="${SEQ:-1024}"
RUN="$(ls -td "$ROOT/Cache/Training/"* | head -1)"
CKPT="$(ls -td "$RUN"/checkpoint-* 2>/dev/null | head -1 || echo "$RUN")"
mkdir -p "$ROOT/Cache/Evaluation_reports"
TAG="$(basename "$RUN")"
EVAL_PY="$ROOT/evaluation/evaluate_cogito_probe.py"

eval_probe() {
  local data="$1" task="$2" suffix="$3"
  shift 3
  echo "E29 eval $suffix data=$data task=${task:-*} ckpt=$CKPT"
  uv run python "$EVAL_PY" \
    --checkpoint "$CKPT" --data "$data" \
    --seq_len "$SEQ" --variant "$VARIANT" --split test \
    ${task:+--task "$task"} \
    --out "$ROOT/Cache/Evaluation_reports/e29_${suffix}_${TAG}.json" \
    "$@" \
    || echo "E29 eval $suffix failed (training artifacts kept)"
}

# hops vs gist on bind (launch_e28 also writes an overall bind JSON with by_task)
eval_probe ksopyla/cogito-probe-bind hop_friend_place "bind_hops_seq${SEQ}"
eval_probe ksopyla/cogito-probe-bind attr_color "bind_gist_attr_seq${SEQ}"
eval_probe ksopyla/cogito-probe-bind who_place "bind_gist_who_seq${SEQ}"
# props filler-shuffle control: gold answers must hold after permuting filler
eval_probe ksopyla/cogito-probe-props prop_color "props_control_seq${SEQ}"
eval_probe ksopyla/cogito-probe-props prop_color "props_filler_shuffle_seq${SEQ}" --shuffle_filler
