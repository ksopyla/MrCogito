#!/bin/bash
# Runs ON a training node: start another job when the current one ends, without powering off in between.
# Usage (on the node, repo root, in tmux):  bash infra/aws/chain_job.sh HOURS JOB [ARGS...]
#   Before the current job ends, create $MC_DISK/KEEP_ALIVE so node_run.sh does not power off. This script then
#   1. waits until the current node_run.sh has exited;
#   2. keeps that job's spend record as spend/<instance-id>-part<time>.json (status.sh sums every file there);
#   3. replaces the node's hard stop with HOURS from now (shutdown -c, shutdown -h +minutes);
#   4. removes KEEP_ALIVE and runs node_run.sh JOB (heartbeat to S3, the job, power-off at its end).
# Needs no AWS sign-in on the laptop: the node's own role reaches S3.
set -uo pipefail
source /etc/mrcogito.env
HOURS="${1:?hours}"; shift
[ -f "$MC_DISK/KEEP_ALIVE" ] || { echo "[chain] $MC_DISK/KEEP_ALIVE missing: the current job would power off"; exit 1; }
echo "[chain] $(date -u +%FT%TZ) waiting for the current node_run.sh to exit"
# anchored: the tmux server's own command line also contains "node_run.sh" and outlives the job
while pgrep -f "^bash infra/aws/node_run.sh" >/dev/null; do sleep 60; done
TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 600")
IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
aws s3 cp --only-show-errors "s3://$MC_BUCKET/spend/$IID.json" "s3://$MC_BUCKET/spend/$IID-part$(date +%s).json" || true
sudo shutdown -c 2>/dev/null || true
sudo shutdown -h "+$((HOURS * 60))" "chained job: max lifetime reached" || true
rm -f "$MC_DISK/KEEP_ALIVE"
echo "[chain] $(date -u +%FT%TZ) hard stop in $HOURS h; starting $*"
git pull -q --ff-only || true
bash infra/aws/node_run.sh "$@" 2>&1 | tee -a "$MC_DISK/job.log"
