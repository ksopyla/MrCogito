#!/bin/bash
# Keep a home server's GPUs under a temperature (Odra rule: under 80 °C) by pausing our OWN GPU jobs when their card
# gets hot and resuming them when it has cooled (2026-11-24, E40: three RTX 3090s at full training load reached
# 79 °C in six minutes). A job is ours when its working folder is inside this checkout; each is paused alone, by its
# process id (SIGSTOP: its GPU work stops after the kernels in flight; single-card jobs only, no collective to time
# out) and resumed with SIGCONT. Other sessions' jobs are never touched: they are only reported.
# Usage (repo root): setsid nohup bash scripts/heat_guard.sh > /dev/null 2>&1 < /dev/null &
#   HOT (default 79) pauses, COOL (default 72) resumes; events (UTC) in Cache/heat_guard.log; PID in Cache/heat_guard.pid
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd -P)"
HOT="${HOT:-79}"; COOL="${COOL:-72}"; EVERY="${EVERY:-10}"
LOG="${HEAT_LOG:-Cache/heat_guard.log}"
mkdir -p "$(dirname "$LOG")"
echo $$ > Cache/heat_guard.pid
say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
declare -A paused   # pid -> card
resume_all() { for p in "${!paused[@]}"; do kill -CONT "$p" 2>/dev/null; done; say "guard stopped, all resumed"; exit 0; }
trap resume_all TERM INT
say "guard start: pause at >= $HOT C, resume at <= $COOL C, every $EVERY s, jobs under $ROOT"
while :; do
  while IFS=', ' read -r idx temp; do
    [[ -z $idx ]] && continue
    for p in $(nvidia-smi -i "$idx" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      cwd=$(readlink "/proc/$p/cwd" 2>/dev/null) || continue
      [[ $cwd == "$ROOT"* ]] || continue
      if [[ -z ${paused[$p]:-} ]] && (( temp >= HOT )); then
        kill -STOP "$p" 2>/dev/null && { paused[$p]=$idx; say "GPU $idx at $temp C: paused pid $p"; }
      elif [[ -n ${paused[$p]:-} ]] && (( temp <= COOL )); then
        kill -CONT "$p" 2>/dev/null; unset "paused[$p]"; say "GPU $idx at $temp C: resumed pid $p"
      fi
    done
  done < <(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
  for p in "${!paused[@]}"; do [[ -d /proc/$p ]] || unset "paused[$p]"; done   # ended while paused
  sleep "$EVERY"
done
