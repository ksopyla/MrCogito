#!/bin/bash
# E41 rows of 262,144 tokens built at home, no AWS (2026-11-27): a small-scale m21mix_xl1
# (data/mix_recipes/m21_mix_xl1.json) for the home screen of the 256k stage.
#   * single long documents: books, doab and usgpo rebuilt at a larger scale than the E40 home mix holds (one book in
#     twenty is 262,144 tokens long), by scripts/build_m21_mix.py in the release build's own file order with the same
#     held-out documents;
#   * recall_256k: planted-fact recall rows over several books in a row (scripts/build_reasoning_rows.py);
#   * the packed sources (pdfs, code_repos, arxiv, multihop_64k, lookup_64k): the E40 home mix's shards, linked.
#
# Usage (Odra, from the repo root):  setsid nohup bash scripts/e41_xl_data_build.sh [OUT] > /dev/null 2>&1 &
#   OUT default Cache/e41/m21xl_home; E40 mix from E40_MIX (default ~/dev/MrCogito-e40/Cache/e40/m21v2_e40);
#   progress OUT/_logs/events.log (UTC), the last line "mix.json written".
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
D="${1:-Cache/e41/m21xl_home}"
SRC="${E40_MIX:-$HOME/dev/MrCogito-e40/Cache/e40/m21v2_e40}"
mkdir -p "$D/_logs"
D="$(cd "$D" && pwd)"
EV="$D/_logs/events.log"
stamp() { echo "[$(date -u +%FT%TZ)] $*" >> "$EV"; }
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1
PY=(.venv/bin/python)
RECIPE=data/mix_recipes/m21_mix_xl1.json
WORKERS="${E41_WORKERS:-3}"
SOURCES=(books:${E41_BOOKS_SCALE:-0.25} usgpo:${E41_USGPO_SCALE:-0.2} doab:${E41_DOAB_SCALE:-0.2})
RECALL_SCALE="${E41_RECALL_SCALE:-0.5}"

stamp "start: $D, recipe $RECIPE, packed sources linked from $SRC"
df -h "$D" | tail -1 >> "$EV"
for n in pdfs code_repos arxiv multihop_64k lookup_64k; do
  [ -f "$SRC/$n.done.json" ] || { stamp "FAILED: $SRC has no $n"; exit 1; }
  for f in "$SRC/$n.done.json" "$SRC/$n"_train_*.bin "$SRC/$n"_val_*.bin; do ln -sf "$f" "$D/"; done
done
[ -d "$D/tokenizer" ] || cp -r "$SRC/tokenizer" "$D/tokenizer"

one() {  # one source at its own scale; skipped when its done.json exists (resume)
  local n="${1%%:*}" s="${1##*:}" rc
  if [ -f "$D/$n.done.json" ]; then stamp "$n: done before, skipped"; return 0; fi
  stamp "$n: start at scale $s"
  "${PY[@]}" scripts/build_m21_mix.py --recipe "$RECIPE" --out "$D" --scale "$s" --only "$n" --no-mix \
      --workers "$WORKERS" >> "$D/_logs/$n.log" 2>&1
  rc=$?
  stamp "$n: exit $rc"
  return "$rc"
}
( stamp "recall_256k: start at scale $RECALL_SCALE"
  "${PY[@]}" scripts/build_reasoning_rows.py --out "$D" --scale "$RECALL_SCALE" --only recall_256k --val_rows 8 \
      --tokenizer "$D/tokenizer" >> "$D/_logs/recall_256k.log" 2>&1
  stamp "recall_256k: exit $?" ) &
RPID=$!
XRC=0
for s in "${SOURCES[@]}"; do one "$s" & done
wait
wait "$RPID" 2>/dev/null
for s in "${SOURCES[@]}"; do [ -f "$D/${s%%:*}.done.json" ] || XRC=1; done
"${PY[@]}" scripts/build_m21_mix.py --recipe "$RECIPE" --out "$D" --mix-only >> "$D/_logs/mix.log" 2>&1
stamp "mix.json written (exit $?); sources missing: $XRC; folder $(du -shL "$D" | cut -f1)"
