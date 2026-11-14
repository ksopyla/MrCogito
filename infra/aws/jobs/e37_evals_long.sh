#!/bin/bash
# E37 / E33t stage D+: the tests of one 64k-stage model, the same tests as the stage-B model had (e37_evals.sh plus the
# later pass's 8k and 32k), spread over all the GPUs given. One list of items, dealt out longest first: each GPU takes
# the next item nobody has claimed.
#   long-document loss by position at 4k-128k, notebook on / off, opened tokens off, notes masked (rows 8);
#   RULER, 13 tasks at 4k, 8k, 16k and 32k, the first 100 documents of each (batch 8 to 16k, 2 at 32k, as before);
#   core benchmarks (limit 0, batch 32, SDPA).
# RULER decodes with the cached path (kv-cache, nn/incremental.py: one prefill, then one position a step; same tokens
# as the old loop on the five 100M models, 25-28x faster at 64k). The stage-B / later-pass scores came from the old
# loop, so one item re-runs RULER at 4k with the old loop (PERCEIVER_AR_GEN_CACHE=0) on this very model:
# long_evals/check_ruler_4096_oldloop.json, compared task by task in eval_ruler_long.json["cache_check"].
# Item files in OUT_DIR/long_evals/ (a finished item is skipped on a relaunch; a failed one is retried once). Merged
# into OUT_DIR/eval_longdoc_long.json, eval_ruler_long.json (run_lsa_suite.py's layout, results[L][task], the mean of
# the 13 tasks per length) and eval_core_long.json, written LAST and only when every test is there (e37_long.sh skips a
# model whose eval_core_long.json is in S3). With S3_DIR, OUT_DIR/long_evals/ goes to S3 every 10 minutes.
# Usage: bash infra/aws/jobs/e37_evals_long.sh CHECKPOINT OUT_DIR DATA_DIR [S3_DIR]
#   DATA_DIR holds tokenizer_gpt2/ and longdocs_lite_gpt2/. GPUs: CUDA_VISIBLE_DEVICES (default: all).
#   E37_EVAL_QUICK=1: a pipeline check (4k loss on 1 row, RULER 2 documents at 4k and 8k only, core 10 documents).
set -uo pipefail
CK="${1:?checkpoint}"; O="${2:?out dir}"; D="${3:?data dir}"; S3D="${4:-}"
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../../.." || exit 1
P="$O/long_evals"; mkdir -p "$P"
# E38 (a built data mix): E33N_TOKENIZER = its tokenizer dir, E33N_LONGDOCS = the dir with <source>_val_*.bin,
# E33N_LONGDOC_ARGS = extra long_doc_loss.py arguments (--sources, --eot, --tokenizer). Defaults: the GPT-2 data.
TOK="${E33N_TOKENIZER:-$D/tokenizer_gpt2}"; LDD="${E33N_LONGDOCS:-$D/longdocs_lite_gpt2}"
IFS=, read -ra GL <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}"
log() { echo "[e37_evals_long $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$O/events.log"; }
ALL13="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery"
ALL13="$ALL13,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot"
if [ -n "${E37_EVAL_QUICK:-}" ]; then
  LD=4096; ROWS=1; LIM=2; CLIM=10; L32=(); L16=(); L8=8192
else
  LD=4096,16384,32768,65536,131072; ROWS=8; LIM=100; CLIM=0; L8=8192
  L32=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery
       niah_multivalue ruler_vt ruler_cwe ruler_fwe ruler_qa_squad ruler_qa_hotpot)
  # 16k in e37_evals.sh's three task groups (each ~1.4 h for the catalogue model on one L40S)
  L16=("niah_single_1,niah_multikey_2,niah_multivalue,ruler_cwe,ruler_qa_hotpot:a"
       "niah_single_2,niah_multikey_3,niah_multiquery,ruler_fwe:b"
       "niah_single_3,niah_multikey_1,ruler_vt,ruler_qa_squad:c")
fi
# ITEM = NAME|KIND|LENGTH|TASKS|BATCH, longest first (the catalogue model, one L40S: loss 4k-128k ~3,000 s, core
# ~1,500 s, RULER 4k with the old loop ~900 s; cached RULER: 16k group and 32k task a few hundred s each)
ITEMS=("longdoc|longdoc|||" "core|core|||" "check_ruler_4096_oldloop|ruler_old|4096|${ALL13}|8")
for g in "${L16[@]}"; do ITEMS+=("ruler_16384_${g##*:}|ruler|16384|${g%%:*}|8"); done
for t in "${L32[@]}"; do ITEMS+=("ruler_32768_${t}|ruler|32768|${t}|2"); done
ITEMS+=("ruler_${L8}_all|ruler|${L8}|${ALL13}|8" "ruler_4096_all|ruler|4096|${ALL13}|8")

run_item() {  # GPU ITEM
  local g=$1 name kind L tasks B cache=1
  IFS='|' read -r name kind L tasks B <<< "$2"
  [ "$kind" = ruler_old ] && { kind=ruler; cache=0; }
  case "$kind" in
    ruler) CUDA_VISIBLE_DEVICES=$g PERCEIVER_AR_GEN_CACHE=$cache uv run --with wonderwords python \
             evaluation/run_lsa_suite.py --checkpoint "$CK" \
             --tokenizer "$TOK" --suite ruler --lengths "$L" --tasks "$tasks" --limit "$LIM" --batch "$B" \
             --out "$P/$name.json" > "$P/$name.log" 2>&1 ;;
    longdoc) CUDA_VISIBLE_DEVICES=$g uv run python evaluation/long_doc_loss.py --checkpoint "$CK" \
             --data "$LDD" --lengths "$LD" --rows "$ROWS" --notes_off --open_off --gist_off ${E33N_LONGDOC_ARGS:-} \
             --out "$P/$name.json" > "$P/$name.log" 2>&1 ;;
    core) CUDA_VISIBLE_DEVICES=$g uv run python evaluation/run_lsa_suite.py --checkpoint "$CK" \
             --tokenizer "$TOK" --suite core --limit "$CLIM" --batch "${E33N_CORE_BATCH:-32}" \
             --backend "${E33N_CORE_BACKEND:-sdpa}" --out "$P/$name.json" > "$P/$name.log" 2>&1 ;;
  esac
}
lane() {  # GPU
  local g=$1 it name s
  for it in "${ITEMS[@]}"; do
    name=${it%%|*}
    [ -f "$P/$name.json" ] && continue
    mkdir "$P/claim/$name" 2>/dev/null || continue          # another GPU took it
    log "GPU $g: $name"; s=$(date -u +%s)
    if run_item "$g" "$it" && [ -f "$P/$name.json" ]; then
      log "GPU $g: $name done in $(( $(date -u +%s) - s )) s"
    else
      log "GPU $g: $name FAILED after $(( $(date -u +%s) - s )) s (see long_evals/$name.log)"
    fi
  done
}
left() { local n=0 it; for it in "${ITEMS[@]}"; do [ -f "$P/${it%%|*}.json" ] || n=$((n + 1)); done; echo $n; }

SYNC=""
if [ -n "$S3D" ]; then
  ( while sleep 600; do aws s3 sync --only-show-errors "$P/" "$S3D/long_evals/" --exclude "claim/*" || true; done ) &
  SYNC=$!
fi
log "tests of $CK on GPUs ${GL[*]}: ${#ITEMS[@]} items, $(left) left${E37_EVAL_QUICK:+ (quick check)}"
for pass in 1 2; do                                           # pass 2 retries what failed in pass 1
  rm -rf "$P/claim"; mkdir -p "$P/claim"
  PIDS=(); for g in "${GL[@]}"; do ( lane "$g" ) & PIDS+=($!); sleep 3; done
  wait "${PIDS[@]}"
  [ "$(left)" = 0 ] && break
  log "pass $pass ended with $(left) items left"
done
[ -n "$SYNC" ] && kill "$SYNC" 2>/dev/null

ok=$(python3 - "$O" "$CK" "$LIM" "${E37_EVAL_QUICK:-}" <<'PY'
import json, shutil, sys
from pathlib import Path
O, CK, LIM, QUICK = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), bool(sys.argv[4])
P = O / "long_evals"
res = {}
for f in sorted(P.glob("ruler_*.json")):
    for L, r in json.loads(f.read_text())["results"].items():
        res.setdefault(L, {}).update(r)
mean = {}
for L, r in res.items():
    vals = [x.get(f"{L},none") for x in r.values() if isinstance(x, dict)]
    vals = [v for v in vals if isinstance(v, (int, float)) and v >= 0]
    mean[L] = sum(vals) / len(vals) if len(vals) == 13 else None
    print(f"RULER {L}: {len(vals)} tasks, mean {mean[L]}", file=sys.stderr)
out = {"checkpoint": CK, "suite": "ruler", "lengths": sorted(int(x) for x in res), "limit": LIM, "notes_off": False,
       "results": res, "mean": mean, "tasks": {L: len(r) for L, r in res.items()},
       "limit_per_length": {L: [LIM] for L in res}, "decoding": "cached (nn/incremental.py, kv-cache ee681d1)",
       "note": "e37_evals_long.sh: the 64k-stage model; 16k in three task groups, 32k one call per task, 4k and 8k "
               "one call each; merged; mean only with all 13 tasks"}
chk = P / "check_ruler_4096_oldloop.json"
if chk.exists() and "4096" in res:
    old = json.loads(chk.read_text())["results"]["4096"]
    pair = {t: [res["4096"][t].get("4096,none"), old[t].get("4096,none")] for t in res["4096"] if t in old}
    out["cache_check"] = {"length": 4096, "cached_vs_old_loop": pair,
                          "same": all(a == b for a, b in pair.values()) and len(pair) == 13}
    print(f"cache check at 4k: same scores on all 13 tasks = {out['cache_check']['same']}", file=sys.stderr)
(O / "eval_ruler_long.json").write_text(json.dumps(out, indent=2, default=str))
if (P / "longdoc.json").exists():
    shutil.copy(P / "longdoc.json", O / "eval_longdoc_long.json")
need = ("4096", "8192") if QUICK else ("4096", "8192", "16384", "32768")
full = all(out["tasks"].get(L) == 13 for L in need) and (P / "longdoc.json").exists() and (P / "core.json").exists()
print(int(full))
PY
)
if [ "$ok" = 1 ]; then
  cp "$P/core.json" "$O/eval_core_long.json"
  log "tests done"
  rc=0
else
  log "tests INCOMPLETE ($(left) items missing; eval_core_long.json not written, a relaunch redoes what is missing)"
  rc=1
fi
if [ -n "$S3D" ]; then
  aws s3 sync --only-show-errors "$P/" "$S3D/long_evals/" --exclude "claim/*" || true
  for f in eval_longdoc_long.json eval_ruler_long.json eval_core_long.json events.log; do
    [ -f "$O/$f" ] && aws s3 cp --only-show-errors "$O/$f" "$S3D/$f"
  done
fi
exit $rc
