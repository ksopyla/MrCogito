#!/bin/bash
# Terminate training nodes by Name tag or instance id (only instances tagged Project=mrcogito), in every region of
# MC_REGIONS. bash 3.2 compatible.
# Usage: bash infra/aws/terminate.sh NAME_OR_ID [...]    |    bash infra/aws/terminate.sh --all
set -euo pipefail
source "$(dirname "$0")/env.sh"
live() {  # REGION + extra filters → instance ids (one per line)
  local region="$1"; shift
  aws ec2 describe-instances --region "$region" --filters "Name=tag:Project,Values=${MC_PROJECT_TAG}" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" "$@" \
    --query 'Reservations[].Instances[].InstanceId' --output text | tr '\t' '\n' | grep '^i-' || true
}
ANY=""
for region in $MC_REGIONS; do
  IDS=""
  if [ "${1:-}" = "--all" ]; then
    IDS=$(live "$region")
  else
    for a in "$@"; do
      if [[ "$a" == i-* ]]; then IDS="$IDS $(live "$region" "Name=instance-id,Values=$a")"
      else IDS="$IDS $(live "$region" "Name=tag:Name,Values=$a")"; fi
    done
  fi
  IDS=$(echo $IDS)
  [ -z "$IDS" ] && continue
  ANY=1
  aws ec2 terminate-instances --region "$region" --instance-ids $IDS \
    --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) terminate $region $IDS" >> "$(dirname "$0")/runs.log"
done
[ -z "$ANY" ] && echo "nothing to terminate"
exit 0
