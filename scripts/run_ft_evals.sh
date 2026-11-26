#!/bin/bash
# Tests of one checkpoint for the root-cause work (2026-11-23), on three cards of this machine at once:
#   card A: the exam on held-out list rows, then on held-out lookup rows (scripts/concept_use_probe.py rows);
#   card B: the forced-next-value probe on RULER's two several-values tasks at 8k and 4k, then the first-token probe
#           at 16k, 8k and 4k;
#   card C: the forced-next-value probe at 16k.
# Usage: bash scripts/run_ft_evals.sh NAME CHECKPOINT LIST_VAL_SHARD LOOKUP_VAL_SHARD CARD_A CARD_B CARD_C
# Results: Cache/rows_exam/NAME_{list,lookup}.json, Cache/use_probe/NAME_*k.json, NAME_start_*k.json; events in
# Cache/rows_exam/events.log and Cache/use_probe/events.log (UTC).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
NAME="$1"; CK="$2"; LV="$3"; KV="$4"; GA="$5"; GB="$6"; GC="$7"
PY=.venv/bin/python
mkdir -p Cache/rows_exam Cache/use_probe
(
  for s in "list:$LV" "lookup:$KV"; do
    n=${s%%:*}; f=${s#*:}
    CUDA_VISIBLE_DEVICES=$GA $PY scripts/concept_use_probe.py rows --checkpoint "$CK" --shard "$f" --limit 48 \
      --out "Cache/rows_exam/${NAME}_$n.json" > "Cache/rows_exam/${NAME}_$n.out" 2> "Cache/rows_exam/${NAME}_$n.log"
    echo "[rows $(date -u +%FT%TZ)] ${NAME}_$n done $?" >> Cache/rows_exam/events.log
  done
) &
(
  bash scripts/run_use_probe.sh "$GB" "$NAME" "$CK" 50 8192 4096
  EXTRA="--mode start" bash scripts/run_use_probe.sh "$GB" "${NAME}_start" "$CK" 50 16384 8192 4096
) &
( bash scripts/run_use_probe.sh "$GC" "$NAME" "$CK" 50 16384 ) &
wait
echo "[ft_evals $(date -u +%FT%TZ)] $NAME done" >> Cache/use_probe/events.log
