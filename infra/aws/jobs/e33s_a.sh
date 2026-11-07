#!/bin/bash
# E33s stage A on one multi-GPU AWS node (a launch.sh job): the catalogue arm and its two-level twin, seed 1, the
# E33p recipe (trunk, 16k decay, long-document loss with the eval switches, core benchmarks, RULER at 4k only: the
# 16k column took hours per note model in E33p and sits at the floor at 30M), each model on all the node's GPUs.
# Usage (launch.sh JOB words): infra/aws/jobs/e33s_a.sh [ARM ...]   (default: catalogue catalogue_2l)
# Results: s3://BUCKET/runs/e33s/e33n_s30_<ARM>_seed1/. Compared with the AWS seed-1 controls under runs/e33p/.
set -uo pipefail
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(catalogue catalogue_2l)
export E33N_EXPERIMENT=E33s E33N_RULER_LENGTHS=4096
exec bash "$(dirname "$0")/e33p_s1_multi.sh" 1 "${ARMS[@]}"
