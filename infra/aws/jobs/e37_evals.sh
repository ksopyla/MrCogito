#!/bin/bash
# E37: the evals of one trained model spread over all the GPUs given (called by e37_multi.sh per arm on an AWS node;
# runs on a home server too). e33n_run.sh's own eval step runs the three evals side by side on three GPUs, and RULER
# at 16k (uncached decoding) then keeps one GPU busy for hours while the others idle; here it is split by task.
#   lane 0 (first GPU): long-document loss by position at 4k–128k (as trained; a concept model also with the length
#           scale off, eval_longdoc_nosum.json), the core benchmarks, then RULER at 4k (13 tasks);
#   lanes 1..: RULER at 16k, the 13 tasks in three groups spread over the other GPUs; merged into eval_ruler.json.
# eval_core.json is written LAST and only when every eval is there: e37_multi.sh skips an arm whose eval_core.json is
# in S3, so it must mean "all evals done" (a failed eval leaves core.json, and a relaunch redoes the evals).
# Usage: bash infra/aws/jobs/e37_evals.sh CHECKPOINT OUT_DIR DATA_DIR
#   DATA_DIR holds tokenizer_gpt2/ and longdocs_lite_gpt2/. GPUs: CUDA_VISIBLE_DEVICES (default: all).
#   E37_EVAL_QUICK=1: a pipeline check (4k only, 1 row, 2 RULER documents, RULER "16k" at 8k: below ~4k RULER's
#   hotpot QA generator never fits its documents and spins forever; 8k keeps the long key apart from lane 0's 4k).
set -uo pipefail
CK="${1:?checkpoint}"; O="${2:?out dir}"; D="${3:?data dir}"
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p "$O"
IFS=, read -ra GL <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}"
N=${#GL[@]}
log() { echo "[e37_evals $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$O/events.log"; }
if [ -n "${E37_EVAL_QUICK:-}" ]; then
  LD=4096; ROWS=1; LIM=2; RL=8192; CLIM=10
else
  LD=4096,16384,32768,65536,131072; ROWS=8; LIM=100; RL=16384; CLIM=0
fi
CONCEPT=$(python3 -c "import json,sys; print(int(json.load(open(sys.argv[1] + '/config.json')).get('par_global_read') == 'concept'))" "$CK")
# RULER task groups for the long length. Not named GROUPS: bash reserves that (the user's group ids) and silently
# ignores assignments to it, so the lanes were handed group ids as task names.
RGROUPS=("niah_single_1,niah_multikey_2,niah_multivalue,ruler_cwe,ruler_qa_hotpot:long_a"
         "niah_single_2,niah_multikey_3,niah_multiquery,ruler_fwe:long_b"
         "niah_single_3,niah_multikey_1,ruler_vt,ruler_qa_squad:long_c")

ruler() {  # GPU LENGTH TASKS NAME
  CUDA_VISIBLE_DEVICES=$1 uv run --with wonderwords python evaluation/run_lsa_suite.py --checkpoint "$CK" \
    --tokenizer "$D/tokenizer_gpt2" --suite ruler --lengths "$2" ${3:+--tasks "$3"} --limit "$LIM" --batch 8 \
    --out "$O/ruler_$4.json" > "$O/ruler_$4.log" 2>&1 || log "RULER $4 FAILED (see ruler_$4.log)"
}
longdoc() {  # NAME [extra args]
  local name=$1; shift
  uv run python evaluation/long_doc_loss.py --checkpoint "$CK" --data "$D/longdocs_lite_gpt2" --lengths "$LD" \
    --rows "$ROWS" --notes_off --open_off --gist_off "$@" --out "$O/$name.json" > "$O/$name.log" 2>&1 \
    || log "$name FAILED (see $name.log)"
}
lane0() {
  export CUDA_VISIBLE_DEVICES=${GL[0]}
  longdoc eval_longdoc
  [ "$CONCEPT" = 1 ] && longdoc eval_longdoc_nosum --concept_sum_len 0
  uv run python evaluation/run_lsa_suite.py --checkpoint "$CK" --tokenizer "$D/tokenizer_gpt2" --suite core \
    --limit "$CLIM" --batch "${E33N_CORE_BATCH:-32}" --backend "${E33N_CORE_BACKEND:-sdpa}" --out "$O/core.json" \
    > "$O/eval_core.log" 2>&1 || log "core FAILED (see eval_core.log)"
  ruler "${GL[0]}" 4096 "" 4k
}

log "evals of $CK on GPUs ${GL[*]} (concept read: $CONCEPT${E37_EVAL_QUICK:+, quick check})"
if [ "$N" -gt 1 ]; then
  declare -a LANE
  for i in 0 1 2; do L=$(( 1 + i % (N - 1) )); LANE[$L]="${LANE[$L]:-} ${RGROUPS[$i]}"; done
  PIDS=()
  ( lane0 ) & PIDS+=($!)
  for L in $(seq 1 $((N - 1))); do
    [ -n "${LANE[$L]:-}" ] || continue
    ( for g in ${LANE[$L]}; do ruler "${GL[$L]}" "$RL" "${g%%:*}" "${g##*:}"; done ) & PIDS+=($!)
  done
  wait "${PIDS[@]}"
else
  lane0
  for g in "${RGROUPS[@]}"; do ruler "${GL[0]}" "$RL" "${g%%:*}" "${g##*:}"; done
fi

# RULER: 4k from lane 0 and the three long groups, one file in run_lsa_suite.py's layout (mean of the 13 tasks)
python3 - "$O" "$RL" "$CK" "$LIM" <<'PY' || log "RULER merge FAILED"
import json, sys
from pathlib import Path
O, RL, CK, LIM = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
part = lambda k: json.loads((O / f"ruler_{k}.json").read_text()) if (O / f"ruler_{k}.json").exists() else None
res = {}
if part("4k"):
    res["4096"] = part("4k")["results"]["4096"]
long = {}
for k in ("long_a", "long_b", "long_c"):
    if part(k):
        long.update(part(k)["results"][RL])
if long:
    res[RL] = long
mean = {}
for L, r in res.items():
    vals = [x.get(f"{L},none") for x in r.values() if isinstance(x, dict)]
    vals = [v for v in vals if isinstance(v, (int, float)) and v >= 0]
    mean[L] = sum(vals) / len(vals) if vals else None
    print(f"RULER {L}: {len(vals)} tasks, mean {mean[L]}")
out = {"checkpoint": CK, "suite": "ruler", "lengths": [int(x) for x in res], "limit": LIM, "notes_off": False,
       "results": res, "mean": mean, "tasks": {L: len(r) for L, r in res.items()},
       "note": "e37_evals.sh: 4k on lane 0, the long length in three task groups on the other GPUs, merged"}
(O / "eval_ruler.json").write_text(json.dumps(out, indent=2, default=str))
PY

ok=$(python3 - "$O" "$RL" <<'PY'
import json, sys
from pathlib import Path
O, RL = Path(sys.argv[1]), sys.argv[2]
try:
    r = json.loads((O / "eval_ruler.json").read_text())
    full = all(r["tasks"].get(L) == 13 for L in ("4096", RL))
    print(int(full and (O / "eval_longdoc.json").exists() and (O / "core.json").exists()))
except Exception:
    print(0)
PY
)
if [ "$ok" = 1 ]; then
  mv "$O/core.json" "$O/eval_core.json"
  log "evals done"
else
  log "evals INCOMPLETE (eval_core.json not written; a relaunch redoes the evals)"
  exit 1
fi
