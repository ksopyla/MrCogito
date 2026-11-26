#!/bin/bash
# mc: no-sync
# m21mix_v4 (2026-11-23) on a CPU node: m21mix_v2 plus the two recall sets (recall_16k, recall_64k: list questions,
# half of the items written as a statement to complete). Same steps as m21v3_data_build.sh. Copy v2 into
# s3://BUCKET/data/m21mix_v4/ from the laptop first (the node's role may not copy between S3 folders).
#   bash infra/aws/launch.sh m21v4-data 3 infra/aws/jobs/m21v4_data_build.sh BRANCH c7i.8xlarge
export M21_NEW=m21mix_v4 M21_NEW_SETS="recall_16k recall_64k" M21_NEW_RECIPE=data/mix_recipes/m21_mix_v4.json
exec bash "$(cd "$(dirname "$0")" && pwd)/m21v3_data_build.sh"
