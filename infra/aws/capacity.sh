#!/bin/bash
# Read-only: H100 Capacity Block offers (reservable, paid upfront) and current spot prices of GPU types we have
# spot quota for. Usage: bash infra/aws/capacity.sh [HOURS]
set -uo pipefail
source "$(dirname "$0")/env.sh"
HOURS="${1:-24}"
START=$(python3 -c "import datetime as d;print((d.datetime.now(d.UTC)+d.timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ'))")
END=$(python3 -c "import datetime as d;print((d.datetime.now(d.UTC)+d.timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ'))")
echo "== Capacity Block offers: 1 × p5.4xlarge for ${HOURS} h, starting within 7 days"
aws ec2 describe-capacity-block-offerings --instance-type p5.4xlarge --instance-count 1 \
  --capacity-duration-hours "$HOURS" --start-date-range "$START" --end-date-range "$END" \
  --query 'CapacityBlockOfferings[].[AvailabilityZone,StartDate,EndDate,UpfrontFee,CurrencyCode]' --output text 2>&1 | head -12
echo "== cheapest spot price per type now (Linux)"
aws ec2 describe-spot-price-history --instance-types g6e.xlarge g6e.2xlarge g6e.4xlarge g6e.12xlarge g6.xlarge p5.4xlarge \
  --product-descriptions "Linux/UNIX" --start-time "$START" \
  --query 'SpotPriceHistory[].[InstanceType,AvailabilityZone,SpotPrice]' --output text | sort -k1,1 -k3,3n | awk '!s[$1]++'
echo "== GPU quotas (vCPU)"
aws service-quotas list-service-quotas --service-code ec2 \
  --query "Quotas[?contains(QuotaName,'G and VT') || contains(QuotaName,' P ')].[QuotaName,Value]" --output text
