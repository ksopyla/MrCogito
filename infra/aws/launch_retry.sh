#!/bin/bash
# Retry launch.sh every INTERVAL seconds until it gets capacity or ATTEMPTS run out (on-demand H100 comes and goes).
# Usage: bash infra/aws/launch_retry.sh ATTEMPTS INTERVAL NAME MAX_HOURS JOB [BRANCH] [TYPE]
ATTEMPTS="$1"; INTERVAL="$2"; shift 2
for i in $(seq 1 "$ATTEMPTS"); do
  echo "[$(date -u +%H:%M:%S)] attempt $i/$ATTEMPTS"
  bash "$(dirname "$0")/launch.sh" "$@"
  rc=$?
  [ "$rc" = 0 ] && exit 0
  [ "$rc" != 2 ] && { echo "launch failed with code $rc (not a capacity error) — stopping"; exit "$rc"; }
  sleep "$INTERVAL"
done
echo "no capacity after $ATTEMPTS attempts"
exit 2
