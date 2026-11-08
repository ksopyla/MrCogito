#!/bin/bash
# E33t stage B on one multi-GPU AWS node (a launch.sh job, g6e.12xlarge spot): note and page sizes around the
# stage-A winner catalogue_reuse (256-token pages, 32 notes = 8 tokens per note, pass exclusion off), seed 1, the
# stage-A recipe (e33t_a.sh: trunk, 16k decay, long-document loss with the eval switches, core benchmarks, RULER at
# 4k), each model on all the node's GPUs:
#   catalogue_reuse_n4    256-token pages, 64 notes (4 tokens per note)
#   catalogue_reuse_n16   256-token pages, 16 notes (16 tokens per note; the filing cap 16 is exact)
#   catalogue_reuse_p512  512-token pages, 64 notes (8 tokens per note)
# Approved by the author 2026-11-08 (~17:30 UTC) with stage C. About 4.7 h per model.
# Usage (launch.sh JOB words): infra/aws/jobs/e33t_b.sh [ARM ...]
# Results: s3://BUCKET/runs/e33t/e33n_s30_<ARM>_seed1/ (the same seed-1 controls as stage A, under runs/e33p/).
set -uo pipefail
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(catalogue_reuse_n4 catalogue_reuse_n16 catalogue_reuse_p512)
exec bash "$(dirname "$0")/e33t_a.sh" "${ARMS[@]}"
