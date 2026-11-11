#!/bin/bash
# E37, the later RULER pass: LSA's lengths past the node set (8k, 32k, 64k; the nodes score 4k and 16k) on one trained
# checkpoint, one lm-eval call per (length, task), the pairs dealt out to the GPUs given (shortest length first). A
# finished pair (later/ruler_<L>_<task>.json) is skipped on a relaunch, so what a lane finished survives.
# Merged into OUT_DIR/eval_ruler_later.json in run_lsa_suite.py's layout (results[L][task]; mean of the 13 tasks per
# length when all 13 are there). Runs on a home server or a node.
# Usage: bash infra/aws/jobs/e37_ruler_later.sh CHECKPOINT OUT_DIR DATA_DIR [LENGTHS]   (default 8192,32768,65536)
#   CHECKPOINT: our model's final/ folder (perceiver_ar, run_lsa_suite.py in this repo's uv env), or with
#     E37_LSA_ORIG=1 the original LSA's torchtitan step folder (evaluation/lsa_orig_eval.py in their venv:
#     LSA_PY = its python, LSA_OV = the folder with lm-eval 0.4.11 put ahead of theirs, as lsa_orig_run.sh does).
#   DATA_DIR holds tokenizer_gpt2/ (and fineweb_edu_gpt2/, longdocs_lite_gpt2/ for the original LSA's loader).
#   GPUs: CUDA_VISIBLE_DEVICES (default: all). Rows per batch: E37_RULER_ROWS_TOKENS / length, 1..8 (default 65536:
#   8 at 8k, 2 at 32k, 1 at 64k, for 24 GB cards; decoding is uncached and projects only the current position).
#   E37_RULER_LIMIT: documents per task (default 100, as the nodes). E37_BURST_MIN / E37_COOL_MIN (home heat rule):
#   after BURST_MIN minutes no lane takes a new pair; once all lanes are idle the GPUs rest COOL_MIN minutes, then the
#   next burst starts (a pair started just before the deadline still finishes).
set -uo pipefail
CK="${1:?checkpoint}"; O="${2:?out dir}"; D="${3:?data dir}"; LENGTHS="${4:-8192,32768,65536}"
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../../.." || exit 1
P="$O/later"; mkdir -p "$P"
IFS=, read -ra GL <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}"
LIM=${E37_RULER_LIMIT:-100}; RT=${E37_RULER_ROWS_TOKENS:-65536}
TASKS=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery
       niah_multivalue ruler_vt ruler_cwe ruler_fwe ruler_qa_squad ruler_qa_hotpot)
IFS=, read -ra LS <<< "$LENGTHS"
ITEMS=(); for L in "${LS[@]}"; do for t in "${TASKS[@]}"; do ITEMS+=("$L:$t"); done; done
log() { echo "[e37_ruler_later $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$O/events.log"; }

one() {  # GPU LENGTH TASK BATCH
  local f="$P/ruler_${2}_${3}"
  if [ -n "${E37_LSA_ORIG:-}" ]; then
    CUDA_VISIBLE_DEVICES=$1 PYTHONPATH="${LSA_OV:?}${PYTHONPATH:+:$PYTHONPATH}" "${LSA_PY:?}" evaluation/lsa_orig_eval.py \
      ruler --checkpoint "$CK" --data "$D" --lengths "$2" --tasks "$3" --limit "$LIM" --batch_tokens $(( $4 * $2 )) \
      --out "$f.json" > "$f.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$1 uv run --with wonderwords python evaluation/run_lsa_suite.py --checkpoint "$CK" \
      --tokenizer "$D/tokenizer_gpt2" --suite ruler --lengths "$2" --tasks "$3" --limit "$LIM" --batch "$4" \
      --out "$f.json" > "$f.log" 2>&1
  fi
}
lane() {  # GPU DEADLINE(epoch, 0 = none)
  local g=$1 dl=$2 it L t B s
  for it in "${ITEMS[@]}"; do
    L=${it%%:*}; t=${it##*:}
    [ -f "$P/ruler_${L}_${t}.json" ] && continue
    [ "$dl" -gt 0 ] && [ "$(date -u +%s)" -ge "$dl" ] && return 0
    mkdir "$P/claim/${L}_${t}" 2>/dev/null || continue          # another lane took it
    B=$(( RT / L )); [ "$B" -lt 1 ] && B=1; [ "$B" -gt 8 ] && B=8
    log "GPU $g: RULER $L $t (batch $B)"; s=$(date -u +%s)
    if one "$g" "$L" "$t" "$B" && [ -f "$P/ruler_${L}_${t}.json" ]; then
      log "GPU $g: RULER $L $t done in $(( $(date -u +%s) - s )) s"
    else
      log "GPU $g: RULER $L $t FAILED after $(( $(date -u +%s) - s )) s (see later/ruler_${L}_${t}.log)"
    fi
  done
}
left() { local n=0 it; for it in "${ITEMS[@]}"; do [ -f "$P/ruler_${it%%:*}_${it##*:}.json" ] || n=$((n + 1)); done; echo $n; }

log "RULER $LENGTHS on $CK, GPUs ${GL[*]}, limit $LIM, ${E37_LSA_ORIG:+original LSA, }$(left) pairs left"
burst=0
while :; do
  rm -rf "$P/claim"; mkdir -p "$P/claim"                       # claims are per burst: a FAILED pair is retried once
  burst=$((burst + 1)); n0=$(left)
  DL=0; [ -n "${E37_BURST_MIN:-}" ] && DL=$(( $(date -u +%s) + E37_BURST_MIN * 60 ))
  PIDS=(); for g in "${GL[@]}"; do ( lane "$g" "$DL" ) & PIDS+=($!); sleep 3; done
  wait "${PIDS[@]}"
  n=$(left)
  [ "$n" = 0 ] && break
  [ "$n" = "$n0" ] && { log "burst $burst finished nothing ($n pairs left): stopping"; break; }
  [ -z "${E37_BURST_MIN:-}" ] && { [ "$burst" -ge 2 ] && break; continue; }
  log "burst $burst done, $n pairs left: GPUs rest ${E37_COOL_MIN:-20} min"; sleep $(( ${E37_COOL_MIN:-20} * 60 ))
done

python3 - "$O" "$CK" "$LIM" <<'PY'
import json, sys
from pathlib import Path
O, CK, LIM = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
res = {}
for f in sorted((O / "later").glob("ruler_*.json")):
    part = json.loads(f.read_text())
    for L, r in part["results"].items():
        res.setdefault(L, {}).update(r)
mean = {}
for L, r in res.items():
    vals = [x.get(f"{L},none") for x in r.values() if isinstance(x, dict)]
    vals = [v for v in vals if isinstance(v, (int, float)) and v >= 0]
    mean[L] = sum(vals) / len(vals) if len(vals) == 13 else None
    print(f"RULER {L}: {len(vals)} tasks, mean {mean[L]}")
out = {"checkpoint": CK, "suite": "ruler", "lengths": sorted(int(x) for x in res), "limit": LIM, "results": res,
       "mean": mean, "tasks": {L: len(r) for L, r in res.items()},
       "note": "e37_ruler_later.sh: one lm-eval call per (length, task), merged; mean only with all 13 tasks"}
(O / "eval_ruler_later.json").write_text(json.dumps(out, indent=2, default=str))
PY
log "pass ended: $(left) pairs left"
