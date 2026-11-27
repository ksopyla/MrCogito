#!/bin/bash
# Watch one keeper-run job from the laptop and act on its marks by itself (E39a's waiter, made reusable for E41).
# Every 5 min it reads the run's stage log and the nodes' spend heartbeats from S3:
#   speed  - median tokens/s of stage-log lines 4-33 (about steps 40-330): logged; below SPEED_MIN -> exit 10
#            (a report, nothing is stopped).
#   loss   - the 5th held-out evaluation of the stage (step ~1,000 at eval_every 200): logged; above LOSS_MAX ->
#            stops the keeper by its PID, terminates the nodes named NAME (infra/aws/terminate.sh), exit 2.
#   money  - the heartbeats of nodes named NAME sum to CAP_USD or more -> stops the keeper and the nodes, exit 7.
#   done   - DONE_KEY in S3 -> exit 0. Keeper gone without it -> exit 5. No stage log 3 h after the start -> exit 4.
#            MAX_H hours -> exit 3. S3 unreadable for 2 h (login?) -> exit 6.
# Usage (from the worktree root; state and log in Cache/ops/):
#   bash infra/aws/watch_run.sh NAME S3_RUN_PREFIX KEEPER_PID_FILE LOSS_MAX SPEED_MIN CAP_USD [STAGE=decay] [MAX_H=40]
#   e.g. bash infra/aws/watch_run.sh e41-tree runs/e41/e33n_m120t_tree_muon0.01_tok8.0e9_seed42 \
#          Cache/keepers/e41-tree.pid 2.227 48600 220
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source infra/aws/env.sh
NAME="$1"; RUNP="$2"; PIDF="$3"; LOSS_MAX="$4"; SPEED_MIN="$5"; CAP="$6"; STAGE="${7:-decay}"; MAX_H="${8:-40}"
S3R="s3://$MC_BUCKET/$RUNP"
mkdir -p Cache/ops
EV="Cache/ops/${NAME}_events.log"; TMP="Cache/ops/${NAME}_${STAGE}_peek.log"; SP="Cache/ops/${NAME}_spend"
say() { echo "[watch $NAME $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$EV"; }
stop_all() {
  kill "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null
  bash infra/aws/terminate.sh "$NAME" 2>&1 | tee -a "$EV"
}
t0=$(date +%s); did_speed=""; did_loss=""; bad=0; last_usd=""; seen=""
# WATCH_SPEED_LINES (default 33): stage-log lines the speed's median needs (a stage of few steps logs fewer: the
# 256k stage has 28 lines in all)
SPL="${WATCH_SPEED_LINES:-33}"
say "watching $S3R (stage $STAGE): stop above loss $LOSS_MAX at the 5th evaluation, report below $SPEED_MIN tokens/s, cap \$$CAP"
while true; do
  now=$(date +%s)
  if aws s3 ls "$S3R/eval_core_long.json" >/dev/null 2>&1; then say "DONE: eval_core_long.json is in S3"; exit 0; fi
  # money: this run's nodes only
  rm -rf "$SP"; mkdir -p "$SP"
  if aws s3 cp --only-show-errors --recursive "s3://$MC_BUCKET/spend/" "$SP/" 2>/dev/null; then
    usd=$(python3 - "$SP" "$NAME" <<'PY'
import json, pathlib, sys
tot = 0.0
for p in pathlib.Path(sys.argv[1]).glob("*.json"):
    try:
        r = json.loads(p.read_text())
    except Exception:
        continue
    if r.get("name") == sys.argv[2]:
        tot += float(r.get("usd", 0.0))
print(f"{tot:.2f}")
PY
)
    if [ "$usd" != "$last_usd" ]; then last_usd="$usd"; echo "[watch $NAME $(date -u +%Y-%m-%dT%H:%M:%SZ)] spend \$$usd of \$$CAP" >> "$EV"; fi
    if python3 -c "import sys; sys.exit(0 if float('$usd') >= float('$CAP') else 1)"; then
      say "MONEY STOP: \$$usd reached the cap \$$CAP; stopping the keeper and the nodes"; stop_all; exit 7
    fi
  fi
  if aws s3 cp --only-show-errors "$S3R/$STAGE.log" "$TMP" 2>/dev/null; then
    bad=0; seen=1
    read -r n sp ev5 evl < <(python3 - "$TMP" "$SPL" <<'PY'
import re, statistics, sys
t = open(sys.argv[1], errors="replace").read()
k = int(sys.argv[2])
sp = [float(x) for x in re.findall(r"'perf/real_tokens_per_second': ([0-9.]+)", t)]
ev = [float(x) for x in re.findall(r"'eval_loss': ([0-9.]+)", t)]
n = len(re.findall(r"\{'loss'", t))
print(n, f"{statistics.median(sp[3:k]):.0f}" if len(sp) >= k else "-", f"{ev[4]:.4f}" if len(ev) >= 5 else "-",
      f"{ev[-1]:.4f}" if ev else "-")
PY
)
    if [ -z "$did_speed" ] && [ "$sp" != "-" ]; then
      did_speed=1; say "speed: median $sp tokens/s over $STAGE log lines 4-$SPL (report below $SPEED_MIN)"
      [ "$sp" -lt "$SPEED_MIN" ] && { say "SPEED BELOW THE MARK: report before going on"; exit 10; }
    fi
    if [ -z "$did_loss" ] && [ "$ev5" != "-" ]; then
      did_loss=1; say "held-out loss at the 5th evaluation of $STAGE: $ev5 (stop above $LOSS_MAX)"
      if python3 -c "import sys; sys.exit(0 if float('$ev5') > float('$LOSS_MAX') else 1)"; then
        say "LOSS STOP: stopping the keeper and terminating $NAME"; stop_all; exit 2
      fi
    fi
  else
    bad=$((bad + 1))
    if ! aws sts get-caller-identity >/dev/null 2>&1 && [ "$bad" -ge 24 ]; then say "S3 unreadable 2 h (login?)"; exit 6; fi
    # only when the stage log was never read: one failed read later is not a missing log (run 4's watcher left on
    # such a read at 08:25 UTC on 2026-11-27, its speed never read because the stage logs fewer than 33 lines)
    [ -z "$seen" ] && [ $((now - t0)) -ge 10800 ] && { say "no $STAGE log 3 h after the start"; exit 4; }
  fi
  kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null || {
    aws s3 ls "$S3R/eval_core_long.json" >/dev/null 2>&1 && { say "DONE"; exit 0; }
    say "keeper gone without the done marker"; exit 5; }
  [ $((now - t0)) -ge $((MAX_H * 3600)) ] && { say "$MAX_H h passed"; exit 3; }
  sleep 300
done
