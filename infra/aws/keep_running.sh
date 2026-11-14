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
#   KEEP_ALT_TYPE (optional): a second instance type to try when TYPE has no capacity, but only once every key in
#     KEEP_ALT_AFTER_KEYS exists (e.g. a one-H100 node first, the shared 4-GPU slot only after the runs queued on
#     it are done). KEEP_ALT_SPOT (default 1) and KEEP_ALT_MAX_HOURS (default MAX_HOURS) apply to that type.
#     TYPE itself launches spot or on demand as MC_SPOT says (H100 nodes: on demand; the P spot quota is 8 vCPU).
#   KEEP_WAIT_KEYS (optional): space-separated S3 keys that must all exist before the FIRST launch — another
#     queue's done-markers, so this job takes a shared slot only after those runs have finished.
#   KEEP_WAIT_UNTIL (optional, with KEEP_WAIT_KEYS): a UTC time "YYYY-MM-DDTHH:MM" after which the loop stops waiting
#     and exits with code 5 instead of launching (someone has to look at why the other runs are late).
# Run it from a frozen copy (bash reads a script while it runs, so editing the file under a live keeper corrupts it):
#   mkdir -p Cache/keepers && cp infra/aws/keep_running.sh Cache/keepers/NAME.sh
#   KEEP_HERE=$PWD/infra/aws nohup bash Cache/keepers/NAME.sh ... &
set -uo pipefail
HERE="${KEEP_HERE:-$(cd "$(dirname "$0")" && pwd)}"
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
if [ -n "${KEEP_WAIT_KEYS:-}" ]; then
  say "waiting for: $KEEP_WAIT_KEYS${KEEP_WAIT_UNTIL:+ (until $KEEP_WAIT_UNTIL UTC)}"
  while true; do
    missing=""
    for k in $KEEP_WAIT_KEYS; do aws s3 ls "s3://$MC_BUCKET/$k" >/dev/null 2>&1 || missing="$missing $k"; done
    [ -z "$missing" ] && { say "all wait keys are in S3"; break; }
    if [ -n "${KEEP_WAIT_UNTIL:-}" ] && [ "$(date -u +%Y-%m-%dT%H:%M)" \> "$KEEP_WAIT_UNTIL" ]; then
      say "STOP (waited until $KEEP_WAIT_UNTIL UTC): still missing$missing"; exit 5
    fi
    sleep "$INTERVAL"
  done
fi
while true; do
  if aws s3 ls "s3://$MC_BUCKET/$DONE_KEY" >/dev/null 2>&1; then say "done: $DONE_KEY is in S3"; exit 0; fi
  if alive; then sleep "$INTERVAL"; continue; fi
  if [ "$launches" -ge "${KEEP_MAX_LAUNCHES:-12}" ]; then say "STOP: $launches launches and still not done"; exit 3; fi
  # try_type TYPE SPOT HOURS: budget check, then one launch attempt; 0 = launched, 1 = no capacity, exits on budget
  try_type() {
    local type=$1 spot=$2 hours=$3 price spent ok
    spent=$(committed)
    if [ "$spot" = 1 ]; then
      price=$(aws ec2 describe-spot-price-history --instance-types "$type" --product-descriptions Linux/UNIX \
        --start-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --query 'max(SpotPriceHistory[].to_number(SpotPrice))' --output text)
    else   # on-demand Linux us-east-1, as launch.sh
      case "$type" in p5.4xlarge) price=6.88 ;; g6e.12xlarge) price=10.493 ;; g6e.xlarge) price=1.861 ;;
        g6e.2xlarge) price=2.242 ;; *) price=$MC_PRICE_PER_HOUR ;; esac
    fi
    ok=$(python3 -c "
spent, price, hours, cap = float('$spent'), float('$price'), float('$hours'), float('$MC_BUDGET_USD')
print(int(spent == spent and spent + price * hours <= cap))")
    if [ "$ok" != 1 ]; then
      say "STOP (budget): spent \$$spent + up to \$$price/h x $hours h would pass the \$$MC_BUDGET_USD cap"; exit 4
    fi
    say "launching $type (launch $((launches + 1)); spent \$$spent of \$$MC_BUDGET_USD; $([ "$spot" = 1 ] && echo spot || echo on-demand) up to \$$price/h)"
    if MC_SPOT=$spot bash "$HERE/launch.sh" "$NAME" "$hours" "$JOB" "$BRANCH" "$type" >> "$LOG" 2>&1; then
      launches=$((launches + 1)); say "launched on $type"; return 0
    fi
    say "no $type capacity or quota"; return 1
  }
  if ! try_type "$TYPE" "${MC_SPOT:-0}" "$MAX_HOURS" && [ -n "${KEEP_ALT_TYPE:-}" ]; then
    alt_ok=1
    for k in ${KEEP_ALT_AFTER_KEYS:-}; do aws s3 ls "s3://$MC_BUCKET/$k" >/dev/null 2>&1 || alt_ok=0; done
    [ "$alt_ok" = 1 ] && try_type "$KEEP_ALT_TYPE" "${KEEP_ALT_SPOT:-1}" "${KEEP_ALT_MAX_HOURS:-$MAX_HOURS}" || true
  fi
  sleep "$INTERVAL"
done
