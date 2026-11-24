#!/bin/bash
# E40 on Odra, hands-off (docs/experiments_specs/ahead/E40_depth_mixed_read.md): the three arms of the retrofit, one per
# card (scripts/e40_train.sh), the halfway kill mark K2 on the depth-mixed read, then the release tests of each arm on
# all three cards (scripts/release_evals.sh, long-document modes off, summaries off and summaries only).
#   K2: at the 4th held-out evaluation of part 1 (step 1,144 = 75M tokens, the halfway point) the depth-mixed read's
#       held-out loss more than 0.02 above the control's -> that arm is stopped (its own process group, by PID).
# Usage (Odra, repo root): setsid nohup bash scripts/e40_pipeline.sh > /dev/null 2>&1 < /dev/null &
#   events (UTC): Cache/e40/pipeline_events.log; per arm Cache/e40/runs/<arm>/events.log; tests Cache/e40/evals/<arm>/.
#   ONLY_EVALS=1 skips training (the tests of finished arms).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
R=Cache/e40/runs
EV=Cache/e40/pipeline_events.log
DATA="${LONGDOCS:-$HOME/dev/MrCogito-lsa/Cache/e33n_data/longdocs_lite_gpt2}"
export LONGCTX="${LONGCTX:-$HOME/dev/MrCogito-eng/Cache/e38_board/longctx}"   # retokenised documents + 64k spans
mkdir -p "$R" Cache/e40/evals
echo $$ > Cache/e40/pipeline.pid
if ! { [[ -f Cache/heat_guard.pid ]] && kill -0 "$(cat Cache/heat_guard.pid)" 2>/dev/null; }; then
  setsid nohup bash scripts/heat_guard.sh > /dev/null 2>&1 < /dev/null &   # pauses our jobs on a card >= 79 C
fi
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$EV"; }

nth_eval() {  # nth_eval <log> <n>: the n-th held-out loss in a trainer log (empty until it exists)
  grep -o "'eval_loss': [0-9.]*" "$1" 2>/dev/null | sed -n "${2}p" | awk '{print $2}'
}

k2_watch() {
  local c a
  while :; do
    c=$(nth_eval "$R/e37a/decay.log" 4); a=$(nth_eval "$R/e40_read/decay.log" 4)
    if [[ -n $c && -n $a ]]; then
      local d; d=$(python3 -c "print(f'{$a - $c:+.4f}')")
      say "K2 (halfway, step 1144): control held-out loss $c, depth-mixed read $a, difference $d (stop above +0.02)"
      if python3 -c "import sys; sys.exit(0 if $a - $c > 0.02 else 1)"; then
        local pg; pg=$(cat "$R/e40_read/decay.pgid" 2>/dev/null)
        say "K2 TRIPPED: stopping the depth-mixed read (process group $pg)"
        [[ -n $pg ]] && kill -TERM -- "-$pg" 2>/dev/null
        touch "$R/e40_read/K2_STOPPED"
      fi
      return 0
    fi
    # an arm whose part 1 ended before its 4th evaluation: nothing to compare
    if { [[ -z $c ]] && grep -q "decay end" "$R/e37a/events.log" 2>/dev/null; } || \
       { [[ -z $a ]] && grep -q "decay end" "$R/e40_read/events.log" 2>/dev/null; }; then
      say "K2: an arm's part 1 ended before the halfway evaluation"
      return 0
    fi
    sleep 120
  done
}

if [[ -z ${ONLY_EVALS:-} ]]; then
  # the heaviest arm on the coolest card (2026-11-24: card 2 ran hottest); the heat guard pauses a hot card's job
  say "training start: 2-DCA everywhere on GPU 0, depth-mixed read on GPU 1, control on GPU 2"
  pids=()
  for a in e40_dca2:0 e40_read:1 e37a:2; do
    bash scripts/e40_train.sh "${a%%:*}" "${a##*:}" "$R/${a%%:*}" > /dev/null 2>&1 &
    pids+=($!)
  done
  k2_watch &
  KPID=$!
  for p in "${pids[@]}"; do wait "$p"; done
  kill "$KPID" 2>/dev/null
  for arm in e37a e40_read e40_dca2; do say "$arm: $(tail -n 1 "$R/$arm/events.log")"; done
fi

for arm in e37a e40_read e40_dca2; do
  ck=$(ls -d "$R/$arm"/long/*/final 2>/dev/null | tail -1)
  if [[ -z $ck ]]; then say "$arm: no final weights, no release tests"; continue; fi
  say "$arm: release tests start ($ck)"
  bash scripts/release_evals.sh "$ck" "Cache/e40/evals/$arm" "$DATA" 0,1,2 off,nosum,sumonly "$arm"
  say "$arm: release tests done ($(tail -n 1 "Cache/e40/evals/$arm/logs/events.log"))"
done
say "all done"
