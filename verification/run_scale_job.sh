#!/usr/bin/env bash
# Tiny wrapper so tmux send-keys cannot wrap and split paths.
set -euo pipefail
cd /workspace
name="$1"
shift
outdir="${SCALE_OUT_DIR:-/opt/cursor/artifacts/scale}"
mkdir -p "$outdir"
uv run python verification/symbolic_channel_probe.py "$@" --run_name "$name" \
  --out "$outdir/${name}.json" \
  2>&1 | tee "$outdir/${name}.log"
