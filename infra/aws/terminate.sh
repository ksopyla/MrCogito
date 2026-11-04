#!/bin/bash
# Terminate training nodes by Name tag or instance id (only instances tagged Project=mrcogito). bash 3.2 compatible.
# Usage: bash infra/aws/terminate.sh NAME_OR_ID [...]    |    bash infra/aws/terminate.sh --all
set -euo pipefail
source "$(dirname "$0")/env.sh"
live() {  # extra filters → instance ids (one per line)
  aws ec2 describe-instances --filters "Name=tag:Project,Values=${MC_PROJECT_TAG}" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" "$@" \
    --query 'Reservations[].Instances[].InstanceId' --output text | tr '\t' '\n' | grep '^i-' || true
}
IDS=""
if [ "${1:-}" = "--all" ]; then
  IDS=$(live)
else
  for a in "$@"; do
    if [[ "$a" == i-* ]]; then IDS="$IDS $(live "Name=instance-id,Values=$a")"
    else IDS="$IDS $(live "Name=tag:Name,Values=$a")"; fi
  done
fi
IDS=$(echo $IDS)
[ -z "$IDS" ] && { echo "nothing to terminate"; exit 0; }
aws ec2 terminate-instances --instance-ids $IDS --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) terminate $IDS" >> "$(dirname "$0")/runs.log"
