#!/bin/bash
# mc: no-sync
# E33n data job (CPU node, c7i.4xlarge): build the data (data_build.sh: 60 FineWeb-Edu GPT-2 shards, the GPT-2
# tokenizer, LongDocs-lite at 300M tokens per source) and copy it to s3://BUCKET/data/.
# Ran once on 2026-11-04 (11 min); the data in S3 is the run's input, so this job is only for a rebuild.
set -euo pipefail
source /etc/mrcogito.env
D="$MC_DISK/data"
bash infra/aws/jobs/data_build.sh "$D" 60 300e6 16
aws s3 sync --only-show-errors "$D/fineweb_edu_gpt2/" "s3://$MC_BUCKET/data/fineweb_edu_gpt2/" --exclude ".cache/*"
aws s3 sync --only-show-errors "$D/tokenizer_gpt2/" "s3://$MC_BUCKET/data/tokenizer_gpt2/"
aws s3 sync --only-show-errors "$D/longdocs_lite_gpt2/" "s3://$MC_BUCKET/data/longdocs_lite_gpt2/"
echo "data job done $(date -u)"
