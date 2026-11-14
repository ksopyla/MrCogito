#!/bin/bash
# Runs ON the training node (started by launch.sh's user data, inside tmux "job", from the repo root).
#  1. every 5 min: spend heartbeat to s3://BUCKET/spend/<instance-id>.json and the job log to s3://BUCKET/logs/<name>/;
#  2. uv sync; runs the job script given as $1;
#  3. final sync, then powers off (the instance terminates: shutdown behaviour = terminate).
# Set KEEP_ALIVE=1 in the job's environment file to skip the power-off (debugging only; MaxHours still applies).
set -uo pipefail
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$MC_DISK/hf" UV_CACHE_DIR="$MC_DISK/uv-cache" TMPDIR="$MC_DISK/tmp"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
JOB="$1"; shift   # the job script; any further words are its arguments (e.g. "jobs/e33n_run.sh e33n s30")
PRICE="${MC_PRICE_PER_HOUR:-6.88}"
TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
md() { curl -s -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1"; }
IID=$(md instance-id); ITYPE=$(md instance-type)
case "$ITYPE" in p5.4xlarge) PRICE=6.88 ;; c7i.4xlarge) PRICE=0.714 ;; c7i.2xlarge) PRICE=0.357 ;; esac
START=$(date -u +%s)
export MC_IID="$IID" MC_ITYPE="$ITYPE"

heartbeat() {
  local now hours
  now=$(date -u +%s)
  hours=$(python3 -c "print(round(($now-$START)/3600+0.05,3))")   # +3 min for boot before this script
  printf '{"name":"%s","id":"%s","type":"%s","start":%s,"now":%s,"hours":%s,"usd":%s,"job":"%s"}\n' \
    "$MC_NAME" "$IID" "$ITYPE" "$START" "$now" "$hours" "$(python3 -c "print(round($hours*$PRICE,2))")" "$JOB" \
    > "$MC_DISK/spend.json"
  aws s3 cp --only-show-errors "$MC_DISK/spend.json" "s3://$MC_BUCKET/spend/$IID.json" || true
  aws s3 cp --only-show-errors "$MC_DISK/job.log" "s3://$MC_BUCKET/logs/$MC_NAME/job.log" || true
}
( while true; do heartbeat; sleep 300; done ) &
HB=$!

echo "[node_run] $(date -u) $MC_NAME $IID $ITYPE job=$JOB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
if grep -q '^# mc: no-sync' "$JOB"; then
  echo "[node_run] job asks for no uv sync (CPU-only job)"
else
  uv sync --frozen 2>&1 | tail -3 || uv sync 2>&1 | tail -3
fi
bash "$JOB" "$@"
RC=$?
echo "[node_run] $(date -u) job exit code $RC"
kill "$HB" 2>/dev/null
heartbeat
if [ "${KEEP_ALIVE:-0}" != "1" ] && [ ! -f "$MC_DISK/KEEP_ALIVE" ]; then
  echo "[node_run] powering off (instance terminates)"
  sudo shutdown -h now
fi
