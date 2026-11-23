#!/bin/bash
# Root-cause probe queue on one card: scripts/concept_use_probe.py force for one checkpoint at several lengths,
# one after another. Results in Cache/use_probe/NAME_<length>k.{json,pt,out,log}; one line per event in
# Cache/use_probe/events.log (UTC).
# Usage: bash scripts/run_use_probe.sh GPU NAME CHECKPOINT LIMIT LENGTH [LENGTH ...]
#   TASKS (default niah_multivalue,niah_multiquery), EXTRA (more arguments for the probe, e.g. "--free base,ior")
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
GPU="$1"; NAME="$2"; CK="$3"; LIMIT="$4"; shift 4
UV="${UV:-$HOME/.local/bin/uv}"
TASKS="${TASKS:-niah_multivalue,niah_multiquery}"
mkdir -p Cache/use_probe
for L in "$@"; do
  tag="${NAME}_$((L / 1024))k"
  echo "[use_probe $(date -u +%FT%TZ)] start $tag on GPU $GPU" >> Cache/use_probe/events.log
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU" "$UV" run --no-sync --with wonderwords python scripts/concept_use_probe.py force \
    --checkpoint "$CK" --length "$L" --limit "$LIMIT" --tasks "$TASKS" --out "Cache/use_probe/$tag" ${EXTRA:-} \
    > "Cache/use_probe/$tag.out" 2> "Cache/use_probe/$tag.log"
  echo "[use_probe $(date -u +%FT%TZ)] end $tag exit $?" >> Cache/use_probe/events.log
done
