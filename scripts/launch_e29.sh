#!/bin/bash
# E29 — exclusive slots on Hub CogitoProbe-bind (hop vs gist). Props is a
# gist-control eval on the same checkpoint, not a second train.
# Spec: docs/experiments_specs/ahead/E29_exclusive_cogitoprobe_bind.md
#   SEQ=1024 bash scripts/launch_e29.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_ID="${EXPERIMENT_ID:-E29}"
export FAMILY=bind
export VARIANT="${VARIANT:-fixed}"
export COGITO_PROBE_ID="${COGITO_PROBE_ID:-ksopyla/cogito-probe-bind}"
export PROBE_SOURCE="${PROBE_SOURCE:-hub}"
bash "$SCRIPT_DIR/launch_e28.sh"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SEQ="${SEQ:-1024}"
RUN="$(ls -td "$ROOT/Cache/Training/"* | head -1)"
CKPT="$(ls -td "$RUN"/checkpoint-* 2>/dev/null | head -1 || echo "$RUN")"
uv run python "$ROOT/evaluation/evaluate_cogito_probe.py" \
  --checkpoint "$CKPT" --data ksopyla/cogito-probe-props \
  --seq_len "$SEQ" --variant "$VARIANT" --split test \
  --out "$ROOT/Cache/Evaluation_reports/e29_props_control_$(basename "$RUN").json" \
  || echo "E29 props control eval failed"
uv run python "$ROOT/evaluation/evaluate_cogito_probe.py" \
  --checkpoint "$CKPT" --data ksopyla/cogito-probe-bind \
  --seq_len "$SEQ" --variant "$VARIANT" --split test \
  --out "$ROOT/Cache/Evaluation_reports/e29_bind_seq${SEQ}_$(basename "$RUN").json" \
  || echo "E29 bind eval failed"
