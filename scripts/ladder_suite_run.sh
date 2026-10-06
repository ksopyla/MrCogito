#!/usr/bin/env bash
# Write per-GPU scripts that read every saved suite model (<job dir>/<arch>.pt, from a run launched with
# --extra "--save_ckpt @out") at longer lengths with verification/length_ladder.py → <job dir>/ladder.json.
# Read-only: nothing is trained. Job dirs that already have ladder.json are skipped, so a re-run resumes.
# The scripts are named <host>_gpu<g>.sh so scripts/run_bursts.sh can chain them after training bursts.
#
#   bash scripts/ladder_suite_run.sh --host polonez --gpus "0 1 2 3" --launch Cache/capability/ladders_2026-10-04 \
#       Cache/capability/li_seed2_30m Cache/capability/li_ckpt_s01_30m
#
# Dense reads are capped at 32k (quadratic cost); latent-memory models read to 128k.
set -euo pipefail

HOST=polonez
GPUS="0"
LAUNCH=""
MAX_LEN="dense=32768"
RUNS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --launch) LAUNCH="$2"; shift 2 ;;
    --max_len) MAX_LEN="$2"; shift 2 ;;
    *) RUNS+=("$1"); shift ;;
  esac
done
[ -n "$LAUNCH" ] && [ ${#RUNS[@]} -gt 0 ] || { echo "usage: $0 --launch DIR [--host H] [--gpus '0 1'] <run dir>..."; exit 1; }

REPO=$(pwd)
mkdir -p "$LAUNCH"
read -r -a G <<< "$GPUS"
n=${#G[@]}
for i in "${!G[@]}"; do
  g=${G[$i]}
  s="$LAUNCH/${HOST}_gpu${g}.sh"
  cat > "$s" <<EOF
#!/usr/bin/env bash
# length ladder over saved suite models · host $HOST · GPU $g (share $i of $n) · runs: ${RUNS[*]}
set -uo pipefail
cd $REPO
export PATH="\$HOME/.local/bin:\$PATH" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$g
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec > >(tee -a $LAUNCH/${HOST}_gpu${g}.log) 2>&1
k=0
for d in \$(find ${RUNS[*]} -name '*.pt' -printf '%h\n' | sort -u); do
  k=\$((k + 1)); [ \$(( (k - 1) % $n )) -eq $i ] || continue
  if [ -f "\$d/ladder.json" ]; then echo "skip \$d (done)"; continue; fi
  echo "LADDER \$d \$(date '+%Y-%m-%dT%H:%M:%S')"
  uv run python verification/length_ladder.py --ckpt "\$d" --max_len "$MAX_LEN" > "\$d/ladder.log" 2>&1
  echo "EXIT \$d \$?"
done
EOF
  chmod +x "$s"
  echo "$s"
done
