#!/bin/bash
# AWS node job (chained after E33p stage 1's dense seed 1): E33t stage A first, then the rest of E33p stage 1 seed 1
# (the window control and LSA-native), so the first E33t result does not wait for the last controls.
# The two controls were held back from the stage-1 queue with S3 "skip" markers (eval_core.json files that say
# "NOT a result"); they are removed here before those arms run. Real results are never touched.
# Usage (launch.sh / chain_job.sh JOB words): infra/aws/jobs/e33t_queue.sh
set -uo pipefail
source /etc/mrcogito.env
for a in window lsa; do
  f="s3://$MC_BUCKET/runs/e33p/e33n_s30_${a}_seed1/eval_core.json"
  if aws s3 cp --only-show-errors "$f" - 2>/dev/null | grep -q "NOT a result"; then
    aws s3 rm --only-show-errors "$f" && echo "[e33t_queue] removed the skip marker of $a seed 1"
  fi
done
bash "$(dirname "$0")/e33t_a.sh" catalogue catalogue_2l
bash "$(dirname "$0")/e33p_s1_multi.sh" 1 window lsa
