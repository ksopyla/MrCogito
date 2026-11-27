#!/bin/bash
# mc: no-sync
# m21mix_xl1 (2026-11-27, E41's stage on rows of 262,144 tokens) on a CPU node: the long sources of m21mix_v2
# (shard for shard) plus recall_256k, planted-fact recall rows over several books in a row
# (scripts/build_reasoning_rows.py --only recall_256k); one phase, `long`, at 262,144 tokens
# (data/mix_recipes/m21_mix_xl1.json). Same steps as m21v3_data_build.sh. Copy the sources into
# s3://BUCKET/data/m21mix_xl1/ from the laptop first (the node's role may not copy between S3 folders).
#   bash infra/aws/launch.sh m21xl1-data 3 infra/aws/jobs/m21xl1_data_build.sh BRANCH c7i.8xlarge
export M21_NEW=m21mix_xl1 M21_NEW_SETS="recall_256k" M21_NEW_RECIPE=data/mix_recipes/m21_mix_xl1.json
export M21_COPY_ONLY="code_repos books arxiv doab usgpo hansard pdfs multihop_64k lookup_64k"
exec bash "$(cd "$(dirname "$0")" && pwd)/m21v3_data_build.sh"
