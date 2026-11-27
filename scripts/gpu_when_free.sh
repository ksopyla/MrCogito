#!/bin/bash
# Run one command on a home card once nothing computes on it (a queue of one).
#   setsid nohup bash scripts/gpu_when_free.sh GPU LOG CMD [ARG...] > /dev/null 2>&1 < /dev/null &
# The command runs from the repository root with CUDA_VISIBLE_DEVICES=GPU; its output and exit code go to LOG.
# WAIT_FIRST=seconds: sleep before the first look (two queued on one card: give the second a later start, so the
# first has taken the card by then).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1
G="$1"; LOG="$2"; shift 2
mkdir -p "$(dirname "$LOG")"
sleep "${WAIT_FIRST:-0}"
while [ -n "$(nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do sleep 30; done
echo "[$(date -u +%FT%TZ)] GPU $G free: $*" >> "$LOG"
CUDA_VISIBLE_DEVICES="$G" "$@" >> "$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] exit $?" >> "$LOG"
