#!/bin/bash
# Run one command on a home card once nothing computes on it. Several may be queued on one card: they run one at a
# time (a per-card lock held while a command runs), in no fixed order.
#   setsid nohup bash scripts/gpu_when_free.sh GPU LOG CMD [ARG...] > /dev/null 2>&1 < /dev/null &
# The command runs from the repository root with CUDA_VISIBLE_DEVICES=GPU; its output and exit code go to LOG.
# WAIT_FIRST=seconds: sleep before the first look.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1
G="$1"; LOG="$2"; shift 2
mkdir -p "$(dirname "$LOG")"
LOCK="/tmp/gpu_when_free_${USER}_$G.lock"          # a symbolic link to the holder's PID (made in one step)
sleep "${WAIT_FIRST:-0}"
while true; do
  owner=$(readlink "$LOCK" 2>/dev/null || true)
  if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then rm -f "$LOCK"; owner=""; fi     # its holder is gone
  if [ -z "$owner" ] && [ -z "$(nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] \
     && ln -s "$$" "$LOCK" 2>/dev/null; then
    sleep 1
    [ "$(readlink "$LOCK" 2>/dev/null)" = "$$" ] && break     # still ours (another waiter may have cleared a stale one)
  fi
  sleep $((25 + RANDOM % 10))
done
trap 'rm -f "$LOCK"' EXIT
echo "[$(date -u +%FT%TZ)] GPU $G free: $*" >> "$LOG"
CUDA_VISIBLE_DEVICES="$G" "$@" >> "$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] exit $?" >> "$LOG"
