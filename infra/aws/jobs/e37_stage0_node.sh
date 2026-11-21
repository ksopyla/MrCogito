#!/bin/bash
# E37 stage 0 as a launch.sh job (one GPU node): a slice of the LSA-testbed data from S3, then
# infra/aws/jobs/e37_stage0.sh (GPU tests: flex = reference concept read; 100M speed / memory at 4k and 16k for lsa,
# e37a, e37b, dense), results to s3://BUCKET/runs/e37/stage0/. The node powers off after (node_run.sh).
set -uo pipefail
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_IMPLICIT_TOKEN=1
D="$MC_DISK/data"; OUT="$MC_DISK/runs/e37_stage0"
mkdir -p "$D/fineweb_edu_gpt2" "$OUT"
aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/finewebedu_val_000000.bin" "$D/fineweb_edu_gpt2/"
aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/finewebedu_train_000001.bin" "$D/fineweb_edu_gpt2/"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/tokenizer_gpt2/" "$D/tokenizer_gpt2/"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/longdocs_lite_gpt2/" "$D/longdocs_lite_gpt2/"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash infra/aws/jobs/e37_stage0.sh "$D" "$OUT"
aws s3 sync --only-show-errors "$OUT/" "s3://$MC_BUCKET/runs/e37/stage0/" --exclude "*/checkpoint-*" --exclude "*.safetensors" \
  --exclude "*optimizer.pt" --exclude "*.pth"
