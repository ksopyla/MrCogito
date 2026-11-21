#!/bin/bash
# E38 short web-stage test of concept memory at 100M, compared head to head with the running E38 run at the same step.
# First use (author 2026-11-14, after storing the Hugging Face token): the data A/B, m21mix_v2 (Nemotron-CC web half)
# against m21mix_v1, both with AdamW and the E38 settings (seed 42, 128 rows of 4,096 per step, warm-up 152 steps,
# constant rate after, --fast). After training, both models are scored on the same yardsticks on this node: the seven
# quizzes and Wikipedia-text bits per byte (core suite), and bits per byte at 4k on the long-document sources the two
# mixes share (books, arXiv, open-access books, US government, UK parliament, PDFs, code repositories).
# Usage (launch.sh JOB words): infra/aws/jobs/e38_trunk_test.sh MIX STEPS [OPT] [LR] [COMPARE_RUN]
#   MIX = m21mix_v2 ...; STEPS must be a checkpoint step of COMPARE_RUN (E38 saves every 800: 800, 1600, 2400 ...)
# Results: s3://BUCKET/runs/e38/e38_trunk_e37a_<MIX>_<OPT>_s<STEPS>_seed42/{trunk.log,eval_ab.json,DONE}
set -uo pipefail
MIX="${1:?mix}"; STEPS="${2:?steps}"; OPT="${3:-adam}"; LR="${4:-}"; CMP="${5:-e33n_m100_e37a_tok8.0e9_seed42}"
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_PROJECT=MrCogito EXPERIMENT_ID=E38 MC_DYNAMO_CACHE_LIMIT=2048
cd "$(dirname "$0")/../../.." || exit 1
source infra/aws/jobs/wandb_env.sh
D="$MC_DISK/data"; RUN="e38_trunk_e37a_${MIX}_${OPT}_s${STEPS}_seed42"; R="$MC_DISK/runs/$RUN"; S3R="s3://$MC_BUCKET/runs/e38/$RUN"
export WANDB_DIR="$R/wandb"; mkdir -p "$R" "$WANDB_DIR"
log() { echo "[e38_trunk $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$R/events.log"; }
up() { aws s3 sync --only-show-errors "$R/" "$S3R/" --exclude "wandb/*" --exclude "cmp/*" || true; }
aws s3 sync --only-show-errors "$S3R/" "$R/" --exclude "cmp/*" || true
G=$(nvidia-smi -L | wc -l); MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
M=4; [ "${MEM:-0}" -ge 70000 ] && M=8
TT=$(( STEPS * 128 * 4096 ))
log "start $RUN on ${MC_NAME:-?} ($G GPU), code $(git log --oneline -1 | cut -c1-60)"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/$MIX/" "$D/$MIX/"
OPTS=(); [ "$OPT" = muon ] && OPTS=(--optimizer muon --lr "$LR")
if ! ls -d "$R"/trunk/*/final >/dev/null 2>&1; then
  CMD=$(uv run python scripts/run_e33n.py cmd --arm e37a --size m100 --phase trunk --data "$D" --out "$R" --tokens "$TT" \
        --warmup 152 --micro "$M" --gpus "$G" --eval_every 400 --fast --mix "$D/$MIX" --seed 42 --wandb "${OPTS[@]}")
  echo "$CMD" > "$R/trunk.cmd"
  ck=$(ls -d "$R"/trunk/*/checkpoint-* 2>/dev/null | awk -F'checkpoint-' '{print $NF" "$0}' | sort -rn | head -1 | cut -d' ' -f2)
  [ -n "$ck" ] && [ -f "$ck/trainer_state.json" ] && log "resumes from ${ck##*/}" || ck=""
  ( while sleep 900; do up; done ) & SYNC=$!
  ( export RUN_ID_SUFFIX="e37a_${MIX}_${OPT}"; eval "$CMD${ck:+ --resume_from_checkpoint $ck}" ) >> "$R/trunk.log" 2>&1
  rc=$?; kill "$SYNC" 2>/dev/null
  log "trunk ended with code $rc"; up
  [ "$rc" = 0 ] || exit "$rc"
fi
OURS=$(ls -d "$R"/trunk/*/final | tail -1)
# the run it is compared with: its checkpoint at the same step, weights and config only
key=$(aws s3 ls --recursive "s3://$MC_BUCKET/runs/e38/$CMP/trunk/" | awk '{print $4}' | grep "/checkpoint-$STEPS/config.json" | head -1)
[ -n "$key" ] || { log "no checkpoint-$STEPS of $CMP in S3"; exit 1; }
THEIRS="$R/cmp/checkpoint-$STEPS"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/$(dirname "$key")/" "$THEIRS/" --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"
TOK="$D/$MIX/tokenizer"; SRC="books,arxiv,doab,usgpo,hansard,pdfs,code_repos"
score() {  # NAME CHECKPOINT
  uv run python evaluation/run_lsa_suite.py --checkpoint "$2" --tokenizer "$TOK" --suite core --limit 0 --batch 32 \
    --backend sdpa --out "$R/ab_core_$1.json" > "$R/ab_core_$1.log" 2>&1 || log "core of $1 FAILED"
  uv run python evaluation/long_doc_loss.py --checkpoint "$2" --data "$D/$MIX" --lengths 4096 --rows 16 --sources "$SRC" \
    --eot 0 --tokenizer "$TOK" --out "$R/ab_longdoc_$1.json" > "$R/ab_longdoc_$1.log" 2>&1 || log "long-document loss of $1 FAILED"
}
log "scoring $MIX ($OURS) and $CMP checkpoint-$STEPS"
score this "$OURS"; score cmp "$THEIRS"
python3 - "$R" "$MIX" "$CMP" "$STEPS" <<'PY'
import json, sys
from pathlib import Path
R, mix, cmp, steps = Path(sys.argv[1]), *sys.argv[2:]
def j(p):
    try: return json.loads(p.read_text())
    except Exception: return None
out = {"steps": int(steps), "this": mix, "compared_with": f"{cmp} checkpoint-{steps}", "models": {}}
for who in ("this", "cmp"):
    c, l = j(R / f"ab_core_{who}.json"), j(R / f"ab_longdoc_{who}.json")
    row = {}
    if c:
        r = c["results"]; v = [r[t].get("acc_norm,none", r[t].get("acc,none")) for t in
             ("hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "lambada_openai", "sciq") if t in r]
        row["quiz_mean"] = sum(v) / len(v) if v else None
        row["wikitext_bits_per_byte"] = r.get("wikitext", {}).get("bits_per_byte,none")
    if l:
        L = l["lengths"].get("4096", {})
        row["longdoc_bits_per_byte_4k"] = {s: v.get("notes_on", {}).get("bits_per_byte") for s, v in L.items() if isinstance(v, dict)}
    out["models"][who] = row
(R / "eval_ab.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
PY
touch "$R/DONE"; up
log "done"
