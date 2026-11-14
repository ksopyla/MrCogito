#!/bin/bash
# Keep one spot job alive until it is done: (re)launch NAME whenever no node of that name is pending or running and
# the job's done-marker is not in S3. AWS takes spot nodes back with two minutes' notice; the training jobs keep
# their checkpoints in S3 and resume (e33n_run.sh with E33N_CKPT_S3=1), so the same launch continues the run.
# Budget guard: before every launch, the spend so far (status.sh: the sum of every node's heartbeat) plus what the
# nodes running now and this launch can still cost (their remaining MaxHours x price) must stay under MC_BUDGET_USD;
# otherwise the loop stops and says so. Runs on the laptop (like launch_retry.sh); one line per event in LOG.
# Usage: MC_SPOT=1 MC_BUDGET_USD=4000 bash infra/aws/keep_running.sh NAME MAX_HOURS "JOB" BRANCH TYPE DONE_KEY [INTERVAL_S]
#   DONE_KEY: an S3 key in the bucket (e.g. runs/e38/<run>/eval_core_long.json); when it exists the loop ends.
#   KEEP_MAX_LAUNCHES (default 12): give up after this many launches (a job that keeps dying is not a spot problem).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
NAME="$1"; MAX_HOURS="$2"; JOB="$3"; BRANCH="$4"; TYPE="$5"; DONE_KEY="$6"; INTERVAL="${7:-600}"
LOG="${KEEP_LOG:-$HERE/keep_$NAME.log}"
say() { echo "[keep $NAME $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
launches=0
alive() {
  for r in $MC_REGIONS; do
    aws ec2 describe-instances --region "$r" --filters "Name=tag:Name,Values=$NAME" \
      "Name=instance-state-name,Values=pending,running" --query 'Reservations[].Instances[].InstanceId' --output text
  done | grep -q .
}
committed() {  # spend so far + what running nodes may still cost, in dollars (status.sh prints both)
  bash "$HERE/status.sh" 2>/dev/null | python3 -c "
import re, sys
t = sys.stdin.read()
m = re.search(r'total ≈ \\\$([0-9.]+)', t)
print(m.group(1) if m else 'nan')"
}
while true; do
  if aws s3 ls "s3://$MC_BUCKET/$DONE_KEY" >/dev/null 2>&1; then say "done: $DONE_KEY is in S3"; exit 0; fi
  if alive; then sleep "$INTERVAL"; continue; fi
  if [ "$launches" -ge "${KEEP_MAX_LAUNCHES:-12}" ]; then say "STOP: $launches launches and still not done"; exit 3; fi
  spent=$(committed)
  price=$(aws ec2 describe-spot-price-history --instance-types "$TYPE" --product-descriptions Linux/UNIX \
    --start-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --query 'max(SpotPriceHistory[].to_number(SpotPrice))' --output text)
  ok=$(python3 -c "
spent, price, hours, cap = float('$spent'), float('$price'), float('$MAX_HOURS'), float('$MC_BUDGET_USD')
print(int(spent == spent and spent + price * hours <= cap))")
  if [ "$ok" != 1 ]; then
    say "STOP (budget): spent \$$spent + up to \$$price/h x $MAX_HOURS h would pass the \$$MC_BUDGET_USD cap"; exit 4
  fi
  say "launching (launch $((launches + 1)); spent \$$spent of \$$MC_BUDGET_USD; $TYPE spot up to \$$price/h)"
  if bash "$HERE/launch.sh" "$NAME" "$MAX_HOURS" "$JOB" "$BRANCH" "$TYPE" >> "$LOG" 2>&1; then
    launches=$((launches + 1)); say "launched"
  else
    say "no capacity or quota; next try in $INTERVAL s"
  fi
  sleep "$INTERVAL"
done
