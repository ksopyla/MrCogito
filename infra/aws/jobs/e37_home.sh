#!/bin/bash
# E37 at 30M on a home server, E33p's exact stage-1 protocol (1.5B FineWeb-Edu at 4k, 0.2B LongDocs-lite at 16k,
# seed 42, instrumentation on, RULER at 4k and 16k), so the runs compare directly with E33p's seed-42 controls.
# One model on the GPUs given (data parallel when several). Polonez: heat bursts of 5.5 h with 20 min cooldowns.
# Usage (on the server, from ~/dev/MrCogito-arch): bash infra/aws/jobs/e37_home.sh ARM GPU_LIST [polonez]
#   e.g. bash infra/aws/jobs/e37_home.sh e37a 0,1 polonez        (in tmux e37-s30-e37a)
# Data: the stage-1 copy in ~/dev/MrCogito-lsa/Cache/e33n_data (read only). Runs: Cache/e37_runs/ of this worktree.
set -uo pipefail
ARM="${1:?arm}"; GPUS="${2:?gpu list}"; HOSTRULE="${3:-}"
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export E33N_DATA="$HOME/dev/MrCogito-lsa/Cache/e33n_data" E33N_RUNS="$PWD/Cache/e37_runs" E33N_EXPERIMENT=E37 \
  E33N_FLOW=1 E33N_MICRO_TRUNK=4 E33N_MICRO_DECAY=1 E33N_CORE_BATCH=8 E33N_RULER_BATCH=8 \
  E33N_RULER_LENGTHS=4096,16384 E33N_GPUS="$N" CUDA_VISIBLE_DEVICES="$GPUS"
if [ "$HOSTRULE" = polonez ]; then export E33N_BURST_H=5.5 E33N_COOL_MIN=20; fi
mkdir -p "$E33N_RUNS"
exec bash infra/aws/jobs/e33n_run.sh "$ARM" s30
