#!/usr/bin/env bash
# Write per-GPU scripts that read saved study models (<job dir>/<arch>.pt, jobs with `save`) at longer lengths with
# verification/length_ladder.py → <job dir>/ladder.json. Read-only: nothing is trained.
#
# The GPUs share one work list: a job dir is claimed when its training has finished (DONE; FAILED jobs are skipped),
# so a list may name runs that are still training — the scripts come back for them until every dir is done.
# Each GPU waits for its own queue logs (--wait_for, '{g}' = GPU id) to print QUEUE-END and for the GPU to be idle.
# A re-run skips dirs with LADDER_DONE or LADDER_FAILED (delete the marker and .ladder_claim to retry).
#
#   bash scripts/ladder_study_run.sh --host odra --gpus "0 1 2" --launch Cache/study/e33_ladders/launch \
#       --wait_for "Cache/study/e33_reasoning_fair/launch/odra_fair_tasks_gpu{g}.log" <job dir>...
#
# Lengths default to the capability checks' ladder (1k → 128k); dense reads are capped at 32k (quadratic cost).
# Exams with several candidates are also scored on the picked candidate (length_ladder --candidate auto).
set -euo pipefail

HOST=odra
GPUS="0"
LAUNCH=""
MAX_LEN="dense=32768"
LENGTHS="1024 2048 4096 8192 16384 32768 65536 131072"
ROWS=64
WAIT=()
RUNS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --launch) LAUNCH="$2"; shift 2 ;;
    --max_len) MAX_LEN="$2"; shift 2 ;;
    --lengths) LENGTHS="$2"; shift 2 ;;
    --rows) ROWS="$2"; shift 2 ;;
    --wait_for) WAIT+=("$2"); shift 2 ;;
    *) RUNS+=("$(realpath -m "$1")"); shift ;;
  esac
done
[ -n "$LAUNCH" ] && [ ${#RUNS[@]} -gt 0 ] || { echo "usage: $0 --launch DIR [--host H] [--gpus '0 1'] [--wait_for LOG] <job dir>..."; exit 1; }

REPO=$(pwd)
mkdir -p "$LAUNCH"
LAUNCH=$(realpath "$LAUNCH")
printf '%s\n' "${RUNS[@]}" > "$LAUNCH/runs.txt"
starter="$LAUNCH/${HOST}_start.sh"
printf '#!/usr/bin/env bash\nbyobu has-session -t study 2>/dev/null || byobu new-session -d -s study\n' > "$starter"
for g in $GPUS; do
  s="$LAUNCH/${HOST}_gpu${g}.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "# length ladder over saved study models · host $HOST · GPU $g · work list $LAUNCH/runs.txt"
    echo "set -uo pipefail"
    echo "cd $REPO"
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$g"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    echo "exec > >(tee -a $LAUNCH/${HOST}_gpu${g}.log) 2>&1"
    for w in ${WAIT[@]+"${WAIT[@]}"}; do
      echo "until grep -q QUEUE-END ${w//\{g\}/$g} 2>/dev/null; do sleep 120; done"
    done
    cat <<EOF
while true; do
  pending=0
  while read -r d; do
    [ -f "\$d/LADDER_DONE" ] || [ -f "\$d/LADDER_FAILED" ] || [ -f "\$d/FAILED" ] && continue
    [ -f "\$d/DONE" ] || { pending=1; continue; }
    mkdir "\$d/.ladder_claim" 2>/dev/null || continue
    while nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader < /dev/null | grep -q .; do sleep 60; done
    echo "LADDER \$d \$(date '+%Y-%m-%dT%H:%M:%S')"
    uv run python verification/length_ladder.py --ckpt "\$d" --lengths $LENGTHS --rows $ROWS --max_len "$MAX_LEN" \\
      < /dev/null > "\$d/ladder.log" 2>&1
    rc=\$?; echo "EXIT \$d \$rc"
    if [ \$rc -eq 0 ]; then touch "\$d/LADDER_DONE"; else touch "\$d/LADDER_FAILED"; fi
  done < $LAUNCH/runs.txt
  [ \$pending -eq 0 ] && break
  sleep 300
done
echo "QUEUE-END \$(date +%Y-%m-%dT%H:%M:%S)"
EOF
  } > "$s"
  chmod +x "$s"
  echo "byobu new-window -t study -n ladder_g$g 'bash $s'" >> "$starter"
  echo "$s"
done
chmod +x "$starter"
echo "start: bash $starter"
