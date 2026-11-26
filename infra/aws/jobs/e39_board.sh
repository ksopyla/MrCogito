#!/bin/bash
# E39 living concepts (docs/experiments_specs/ahead/E39_living_concepts.md): the 150M run on the E38 board protocol
# (e38_board.sh: size m120t, the m21mix_v2 data, the three phases, the same tests of the 64k model), tagged E39 in
# W&B, results in s3://BUCKET/runs/e39/. On an H100 node it saves every 400 steps (800 there would be about two hours
# of work; the spec's compute rule), elsewhere every 800 as E38.
# Stages 2 and 3 read m21mix_v4 (v2 plus the recall rows: 3 % of stage 2, 5 % of stage 3), the author's choice of
# 2026-11-24 ("wait, and then switch to check and compare"), so E39 compares with the E38 recall run on the same
# recipe. Stage 1 is the same under both mixes (same phase, sources and shards), so a node that started on v2
# resumes stage 1 from its checkpoint unchanged.
# Usage (launch.sh JOB words): infra/aws/jobs/e39_board.sh SEED OPT ARM [ARM ...]    e.g. 42 muon:0.01 living
set -uo pipefail
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU" in *H100*) export E38_SAVE_EVERY="${E38_SAVE_EVERY:-400}" ;; esac
export E38_EXPERIMENT=E39 E38_MIX="${E38_MIX:-m21mix_v4}" E33N_LATE_NOTE="${E33N_LATE_NOTE-with recall questions}"
exec bash "$(cd "$(dirname "$0")" && pwd)/e38_board.sh" "$@"
