#!/bin/bash
# E40 on Odra, hands-off (docs/experiments_specs/done_failed/E40_depth_mixed_read.md): the three arms of the retrofit
# (scripts/e40_train.sh), the halfway kill mark K2 on the depth-mixed read, and the release tests of each arm
# (scripts/release_evals.sh, long-document modes off, summaries off and summaries only).
#
# Two cards at a time (2026-11-24 19:10 UTC): with card 1 busy, card 2 climbs from 69 to 81 C within five seconds of
# work, so no guard can hold it under 80 C; with card 1 idle it ran at 72 C. Schedule:
#   A  control on card 0, depth-mixed read on card 1 (card 2 idle)
#   B  2-DCA everywhere on card 2 (card 1 idle beside it); the release tests of the control, then of the read, on card 0
#   C  the release tests of 2-DCA on cards 0 and 2
#   K2 at the 4th held-out evaluation of part 1 (step 1,144 = 75M tokens, the halfway point): the depth-mixed read's
#      held-out loss more than 0.02 above the control's -> that arm is stopped (its own process group, by PID).
# Re-entrant: an arm already training (its train.pid alive) is waited for, a finished arm is not trained again, a
# tested arm is not tested again. scripts/heat_guard.sh runs alongside (pauses our jobs on a hot card).
# Usage (Odra, repo root): setsid nohup bash scripts/e40_pipeline.sh > /dev/null 2>&1 < /dev/null &
#   events (UTC): Cache/e40/pipeline_events.log; per arm Cache/e40/runs/<arm>/events.log; tests Cache/e40/evals/<arm>/.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
R=Cache/e40/runs
EV=Cache/e40/pipeline_events.log
DATA="${LONGDOCS:-$HOME/dev/MrCogito-lsa/Cache/e33n_data/longdocs_lite_gpt2}"
export LONGCTX="${LONGCTX:-$HOME/dev/MrCogito-eng/Cache/e38_board/longctx}"   # retokenised documents + 64k spans
mkdir -p "$R" Cache/e40/evals
echo $$ > Cache/e40/pipeline.pid
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$EV"; }
if ! { [[ -f Cache/heat_guard.pid ]] && kill -0 "$(cat Cache/heat_guard.pid)" 2>/dev/null; }; then
  HOT=78 COOL=70 EVERY=5 setsid nohup bash scripts/heat_guard.sh > /dev/null 2>&1 < /dev/null &
fi

running() { [[ -f $R/$1/train.pid ]] && kill -0 "$(cat "$R/$1/train.pid")" 2>/dev/null; }
finished() { grep -q "all done" "$R/$1/events.log" 2>/dev/null; }
launch() {  # launch <arm> <gpu>
  local arm=$1 g=$2
  if running "$arm"; then say "$arm: already training"; return; fi
  if finished "$arm"; then say "$arm: trained before"; return; fi
  mkdir -p "$R/$arm"
  bash scripts/e40_train.sh "$arm" "$g" "$R/$arm" > /dev/null 2>&1 &
  echo $! > "$R/$arm/train.pid"
  say "$arm: training on GPU $g"
}
wait_arm() { while running "$1"; do sleep 60; done; say "$1: $(tail -n 1 "$R/$1/events.log" 2>/dev/null)"; }
evals() {  # evals <arm> <gpus>
  local ck
  ck=$(ls -d "$R/$1"/long/*/final 2>/dev/null | tail -1)
  if [[ -z $ck ]]; then say "$1: no final weights, no release tests"; return; fi
  if grep -q "all jobs done" "Cache/e40/evals/$1/logs/events.log" 2>/dev/null; then say "$1: tested before"; return; fi
  say "$1: release tests start on GPUs $2 ($ck)"
  bash scripts/release_evals.sh "$ck" "Cache/e40/evals/$1" "$DATA" "$2" off,nosum,sumonly "$1"
  say "$1: release tests done"
}
nth_eval() {  # the n-th held-out loss in a trainer log (empty until it exists)
  grep -o "'eval_loss': [0-9.]*" "$1" 2>/dev/null | sed -n "${2}p" | awk '{print $2}'
}
k2_watch() {
  local c a d pg
  while :; do
    c=$(nth_eval "$R/e37a/decay.log" 4); a=$(nth_eval "$R/e40_read/decay.log" 4)
    if [[ -n $c && -n $a ]]; then
      d=$(python3 -c "print(f'{$a - $c:+.4f}')")
      say "K2 (halfway, step 1144): control held-out loss $c, depth-mixed read $a, difference $d (stop above +0.02)"
      if python3 -c "import sys; sys.exit(0 if $a - $c > 0.02 else 1)"; then
        pg=$(cat "$R/e40_read/decay.pgid" 2>/dev/null)
        say "K2 TRIPPED: stopping the depth-mixed read (process group $pg)"
        [[ -n $pg ]] && kill -TERM -- "-$pg" 2>/dev/null
        touch "$R/e40_read/K2_STOPPED"
      fi
      return 0
    fi
    if { [[ -z $c ]] && ! running e37a; } || { [[ -z $a ]] && ! running e40_read; }; then
      say "K2: an arm stopped before the halfway evaluation"
      return 0
    fi
    sleep 120
  done
}

say "pipeline start (two cards at a time)"
# A
launch e37a 0
launch e40_read 1
KPID=""
grep -q "K2 (halfway" "$EV" 2>/dev/null || { k2_watch & KPID=$!; }
wait_arm e37a
wait_arm e40_read
[[ -n $KPID ]] && kill "$KPID" 2>/dev/null
# B
launch e40_dca2 2
( evals e37a 0; evals e40_read 0 ) &
EPID=$!
wait_arm e40_dca2
wait "$EPID"
# C
evals e40_dca2 0,2
say "all done"
