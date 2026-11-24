#!/bin/bash
# E40 training rows built at home, no AWS (2026-11-24): the sources of m21mix_v2's 16k phase (stage 2, "decay") and
# 64k phase (stage 3, "long"), each at a small scale of its full-build target, from the public Hugging Face datasets
# (scripts/build_m21_mix.py; Nemotron-CC-v2 is gated: HF_TOKEN from ~/.cache/huggingface/token). The builder reads
# every source's files in the release build's own order and holds out the same documents (a property of each
# document), so these rows are a subset of what the release run read and never a LongDocs-lite validation document.
# Scales: enough rows for ~90M tokens of 16k rows and ~60M tokens of 64k rows in the phases' own proportions, with a
# margin (from the 1 % check build's supply report, Cache/e38_check/m21_check). The constructed sets (multi-hop
# libraries, planted-fact lookups at 16k and 64k) come from scripts/build_reasoning_rows.py at one scale.
#
# Usage (Odra, from the repo root):  setsid nohup bash scripts/e40_data_build.sh [OUT] > /dev/null 2>&1 &
#   OUT default Cache/e40/m21v2_e40; progress OUT/_logs/events.log (UTC); per source OUT/_logs/<source>.log.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
D="${1:-Cache/e40/m21v2_e40}"
mkdir -p "$D/_logs"
EV="$D/_logs/events.log"
stamp() { echo "[$(date -u +%FT%TZ)] $*" >> "$EV"; }
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1
[ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ] && HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
export HF_TOKEN
PY=(.venv/bin/python)
RECIPE=data/mix_recipes/m21_mix_v2.json
WORKERS="${E40_WORKERS:-2}"
# source:scale (fraction of the full build's target); largest downloads first
SOURCES=(code_repos:0.10 pdfs:0.05 books:0.04 arxiv:0.06 nemotron_cc_hq:0.01 dclm_edu:0.01 infinibyte:0.04 chat:0.04
         doab:0.02 usgpo:0.02 hansard:0.02)
REASON_SCALE="${E40_REASON_SCALE:-0.12}"
JOBS="${E40_JOBS:-5}"

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
export -f one stamp
export D EV RECIPE WORKERS

stamp "start: $D, recipe $RECIPE, $JOBS sources at once x $WORKERS tokenizer threads, reasoning sets at $REASON_SCALE"
df -h "$D" | tail -1 >> "$EV"
( stamp "reasoning sets: start"
  "${PY[@]}" scripts/build_reasoning_rows.py --out "$D" --scale "$REASON_SCALE" \
      --only multihop_16k,multihop_64k,lookup_16k,lookup_64k >> "$D/_logs/reasoning.log" 2>&1
  stamp "reasoning sets: exit $?" ) &
RPID=$!
printf '%s\n' "${SOURCES[@]}" | PY="${PY[*]}" xargs -P "$JOBS" -I{} bash -c 'PY=($PY); one "$@"' _ {}
XRC=$?
wait "$RPID"
"${PY[@]}" scripts/build_m21_mix.py --recipe "$RECIPE" --out "$D" --mix-only >> "$D/_logs/mix.log" 2>&1
stamp "mix.json written (exit $?); sources exit $XRC; folder $(du -sh "$D" | cut -f1)"
