#!/bin/bash
# Thin E24 S0/S1 hunt wrapper for Odra/Polonez.
# Usage: bash scripts/e24_bapo_hunt.sh NAME GPU [probe args...]
# Runs verification/bapo_capability_probe.py with PYTHONPATH = this checkout
# (override with E24_WORKTREE). Does not hardcode --recipe.
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "usage: $0 NAME GPU [probe args...]" >&2
  exit 2
fi
NAME="$1"
GPU="$2"
shift 2
export PATH="$HOME/.local/bin:$PATH"

WT="${E24_WORKTREE:-$(cd "$(dirname "$0")/.." && pwd)}"
if [ ! -f "$WT/verification/bapo_capability_probe.py" ]; then
  echo "probe not found under $WT" >&2
  exit 1
fi

find_py() {
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    echo "${VIRTUAL_ENV}/bin/python"
    return 0
  fi
  if [ -x "$WT/.venv/bin/python" ]; then
    echo "$WT/.venv/bin/python"
    return 0
  fi
  local parent d
  parent="$(dirname "$WT")"
  for d in "$parent"/*/; do
    if [ -x "${d}.venv/bin/python" ] && [ -f "${d}pyproject.toml" ]; then
      echo "${d}.venv/bin/python"
      return 0
    fi
  done
  return 1
}

PY="$(find_py)" || {
  echo "no project venv found next to $WT" >&2
  exit 1
}

export PYTHONPATH="$WT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
cd "$WT"
mkdir -p "$WT/Cache/logs" "$WT/Cache/bapo_s0/$NAME"
LOG="$WT/Cache/logs/e24_s0_${NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "log=$LOG gpu=$GPU name=$NAME py=$PY wt=$WT args=$*" | tee "$LOG"
set +e
"$PY" verification/bapo_capability_probe.py \
  --eval_every 50 --early_stop_acc 0.99 --max_params 100000000 \
  --threads 4 --seed 0 --out "$WT/Cache/bapo_s0/$NAME" \
  "$@" 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "DONE name=$NAME exit=$ec" | tee -a "$LOG"
exit "$ec"
