#!/usr/bin/env bash
# Run capability-suite launch folders one burst after another, with a cooldown between bursts.
# Each burst = every <host>_gpu*.sh script in one launch folder, in parallel; the next burst starts
# after all of them exit and the cooldown has passed (Polonez heat rule: rest 10–20 min after a
# 5–6 h burst). Finished jobs are skipped by the GPU scripts themselves (DONE markers), so a re-run resumes.
#
#   bash scripts/run_bursts.sh --host polonez --cooldown_min 20 \
#       Cache/capability/li_seed2_30m/launch Cache/capability/e33a_lookup2k_lr5e-5/launch
#
# Start it inside byobu/tmux; progress goes to stdout and each GPU script's own log.
set -uo pipefail

HOST=polonez
COOLDOWN_MIN=20
DIRS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --cooldown_min) COOLDOWN_MIN="$2"; shift 2 ;;
    *) DIRS+=("$1"); shift ;;
  esac
done
[ ${#DIRS[@]} -gt 0 ] || { echo "usage: $0 [--host H] [--cooldown_min M] <launch dir>..."; exit 1; }

temps() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | paste -sd' '; }

n=0
for d in "${DIRS[@]}"; do
  n=$((n + 1))
  if [ $n -gt 1 ]; then
    echo "COOLDOWN ${COOLDOWN_MIN} min $(date '+%Y-%m-%dT%H:%M:%S') · GPU °C: $(temps)"
    sleep $((COOLDOWN_MIN * 60))
  fi
  echo "BURST $n $d $(date '+%Y-%m-%dT%H:%M:%S') · GPU °C: $(temps)"
  pids=()
  # one burst may join several launch folders with '+' (e.g. a study queue on GPUs 0–1 and a suite run on 2–3)
  for s in $(for part in ${d//+/ }; do ls "$part"/"${HOST}"_*gpu*.sh; done); do
    bash "$s" > /dev/null 2>&1 &   # each script tees its own log into the launch folder
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "BURST $n done $(date '+%Y-%m-%dT%H:%M:%S') · GPU °C: $(temps)"
done
echo "ALL BURSTS DONE $(date '+%Y-%m-%dT%H:%M:%S')"
