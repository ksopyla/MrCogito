#!/usr/bin/env bash
# Tiny wrapper so tmux send-keys cannot wrap and split paths.
set -euo pipefail
cd /workspace
name="$1"
shift
uv run python verification/symbolic_channel_probe.py "$@" --run_name "$name" \
  --out "/opt/cursor/artifacts/scale/${name}.json" \
  2>&1 | tee "/opt/cursor/artifacts/scale/${name}.log"
