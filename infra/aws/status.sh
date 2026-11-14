#!/bin/bash
# Read-only: what is running, for how long, and what it has cost so far (instance hours × on-demand price).
# Usage: bash infra/aws/status.sh [--account]   (--account also prints quotas, network and p5 offerings)
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "account ${MC_ACCOUNT} · region ${AWS_REGION} · bucket ${MC_BUCKET}"
if [ "${1:-}" = "--account" ]; then
  echo "== quotas (vCPU)"
  for q in L-417A185B L-1216C47A; do
    aws service-quotas get-service-quota --service-code ec2 --quota-code "$q" \
      --query 'Quota.[QuotaName,Value]' --output text
  done
  echo "== ${MC_INSTANCE_TYPE} offered in"
  aws ec2 describe-instance-type-offerings --location-type availability-zone \
    --filters "Name=instance-type,Values=${MC_INSTANCE_TYPE}" --query 'InstanceTypeOfferings[].Location' --output text
  echo "== default VPC and subnets"
  aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[].VpcId' --output text
  aws ec2 describe-subnets --filters Name=default-for-az,Values=true \
    --query 'Subnets[].[AvailabilityZone,SubnetId]' --output text
  echo "== EBS encryption by default"
  aws ec2 get-ebs-encryption-by-default --query EbsEncryptionByDefault --output text
  echo "== bucket"
  aws s3api head-bucket --bucket "$MC_BUCKET" >/dev/null 2>&1 && echo "exists" || echo "missing"
fi

echo "== instances tagged ${MC_PROJECT_TAG} (not terminated) in ${MC_REGIONS}"
for region in $MC_REGIONS; do
  aws ec2 describe-instances --region "$region" \
    --filters "Name=tag:Project,Values=${MC_PROJECT_TAG}" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value,InstanceId,InstanceType,State.Name,LaunchTime,PublicIpAddress]' \
    --output text | while read -r name id type state launched ip; do
      [ -z "${id:-}" ] && continue
      start=$(date -u -j -f "%Y-%m-%dT%H:%M:%S" "${launched%%+*}" +%s 2>/dev/null || date -u -d "$launched" +%s)
      hours=$(python3 -c "import time;print(f'{(time.time()-$start)/3600:.2f}')")
      cost=$(python3 -c "print(f'{$hours*$MC_PRICE_PER_HOUR:.2f}')")
      echo "$region $name $id $type $state up ${hours} h ≈ \$${cost} ip ${ip}"
    done
done
echo "== spend so far (every node's heartbeat in s3://${MC_BUCKET}/spend/, updated every 5 min)"
TMP=$(mktemp -d)
if aws s3 cp --only-show-errors --recursive "s3://${MC_BUCKET}/spend/" "$TMP/" 2>/dev/null; then
  python3 - "$TMP" "$MC_BUDGET_USD" <<'EOF'
import json, pathlib, sys
recs = [json.loads(p.read_text()) for p in pathlib.Path(sys.argv[1]).glob("*.json")]
tot = sum(r["usd"] for r in recs)
for r in sorted(recs, key=lambda r: r["start"]):
    print(f'  {r["name"]:<28} {r["type"]:<12} {r["hours"]:>7.2f} h  ${r["usd"]:>8.2f}  {r["job"]}')
print(f"total ≈ ${tot:.2f} of the ${float(sys.argv[2]):.0f} cap ({100*tot/float(sys.argv[2]):.1f} %)")
EOF
else
  echo "no heartbeats yet"
fi
rm -rf "$TMP"
