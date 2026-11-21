#!/bin/bash
# E37 stage B: the full m100 protocol instead of stage A's screening budget — FineWeb-Edu 3.0B tokens at 4k, then
# LongDocs-lite 0.4B tokens at 16k (scripts/run_e33n.py SIZES["m100"]: ~21 tokens per parameter of the 162M model,
# 5,722 trunk steps of 128 rows). The arms run one after another on all the node's GPUs, as in stage A (e37_multi.sh:
# checkpoints in S3, a relaunch resumes, arms with eval_core.json in S3 are skipped). Works on a 4 × L40S node and on a
# one-H100 node (p5.4xlarge): with 80 GB cards the trunk micro-batch is 8 rows instead of 4 (the decay stays at 1).
# A launch.sh job cannot pass environment variables to the node, hence this wrapper.
# Usage (launch.sh JOB words): infra/aws/jobs/e37_full.sh SEED ARM [ARM ...]   (most important arm first)
# Results: s3://BUCKET/runs/e37/e33n_m100_<ARM>_tok3.0e9_seed<SEED>/
export E37_TRUNK_TOKENS=3.0e9 E37_DECAY_TOKENS=4.0e8
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "${MEM:-0}" -ge 70000 ] && export E37_MICRO_TRUNK=8
exec bash "$(dirname "$0")/e37_multi.sh" "$@"
