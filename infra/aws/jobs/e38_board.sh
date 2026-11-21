#!/bin/bash
# E38 Tiny-ML Leaderboard runs (author 2026-11-14: submit to huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard;
# "SmolLM2, bigger body"): the m120t size (shared input/output table, 141.5M total for concept memory, LSA port and
# full attention, 145.6M for the catalogue; the board caps at 150M) on the m21mix_v2 data (Nemotron-CC web half,
# SmolLM2 tokens), the same three phases, tests and resume behaviour as e38_full.sh.
# Usage (launch.sh JOB words): infra/aws/jobs/e38_board.sh SEED OPT ARM [ARM ...]    OPT = adam | muon:PEAK
# Results: s3://BUCKET/runs/e38/e33n_m120t_<ARM>[_muon<PEAK>]_tok8.0e9_seed<SEED>/
set -uo pipefail
SEED="${1:?seed}"; OPT="${2:?adam or muon:PEAK}"; shift 2
export E38_SIZE=m120t E38_MIX="${E38_MIX:-m21mix_v2}" E38_OPT="$OPT"
exec bash "$(cd "$(dirname "$0")" && pwd)/e38_full.sh" "$SEED" "$@"
