#!/usr/bin/env bash
# Tiny wrapper so tmux send-keys cannot wrap and split paths.
set -euo pipefail
cd /workspace
name="$1"
shift
# Prefer the workspace disk. /opt/cursor/artifacts is a FUSE store that can
# go size-0 and drop a finished cell's JSON (seq1024 A hit 95.9% then failed
# to write /opt/cursor/artifacts/scale_hard/reach_seq1024_A.json).
outdir="${SCALE_OUT_DIR:-/workspace/Cache/scale_hard}"
mkdir -p "$outdir"
uv run python verification/symbolic_channel_probe.py "$@" --run_name "$name" \
    --out "$outdir/${name}.json" \
  2>&1 | tee "$outdir/${name}.log"
