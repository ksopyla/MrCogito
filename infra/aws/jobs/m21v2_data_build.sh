#!/bin/bash
# mc: no-sync
# E38 data, version 2 (author 2026-11-14, after storing the Hugging Face token): build m21mix_v2 on a CPU node. The
# sources shared with v1 are copied into s3://BUCKET/data/m21mix_v2/ beforehand (server-side) and pulled here; the
# three gated Nemotron-CC sources are built with the token from Secrets Manager; mix.json is written for the v2 recipe.
#   bash infra/aws/launch.sh m21v2-data 4 infra/aws/jobs/m21v2_data_build.sh BRANCH c7i.8xlarge
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source /etc/mrcogito.env
source infra/aws/jobs/hf_env.sh
export M21_GATED_OK=1 M21_PULL=1 M21_RECIPE=data/mix_recipes/m21_mix_v2.json \
  M21_DATA="$MC_DISK/data/m21mix_v2" M21_S3="s3://$MC_BUCKET/data/m21mix_v2"
exec bash infra/aws/jobs/m21_data_build.sh 1
