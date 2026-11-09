#!/bin/bash
# E37 stage B, the original LSA model (scripts/lsa_orig.py; their trainer, our m100 data / budget / evals) on the node
# set up by lsa_orig_node.sh: trunk -> decay -> evals -> S3. Called by lsa_orig_node.sh on GO; by hand (in tmux) it
# resumes: a phase with a DONE marker is skipped, a phase with checkpoints continues from the newest (torchtitan
# resumes the optimizer, the schedule and our loader's position), and a node without the run folder first pulls it
# back from S3. Checkpoints go to S3 every 15 min (the newest two kept by the trainer, mirrored with --delete).
#   LSA_SMOKE=1: the smoke test — trunk 200 steps, decay 100 steps from its weights, quick evals; folder
#   $MC_DISK/runs/lsa_orig_smoke, S3 runs/e37/lsa_orig_smoke/.
# Evals (eval_longdoc.json, eval_ruler.json, eval_core.json, eval_val.json): evaluation/lsa_orig_eval.py, two lanes
# on the one GPU (RULER 64k alone; the rest in turn), lm-eval 0.4.11 (ours) ahead of their 0.4.13 on PYTHONPATH.
# Results: s3://BUCKET/runs/e37/lsa_orig_m100_tok3.0e9_seed42/
set -uo pipefail
source /etc/mrcogito.env
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$REPO" || exit 1
VENV="$MC_DISK/lsa-venv"; PY="$VENV/bin/python"; D="$MC_DISK/data"
export LSA_REPO="$MC_DISK/lsa" LSA_ROOT="$MC_DISK/lsa_root" HF_HOME="$MC_DISK/hf" TMPDIR="$MC_DISK/tmp" \
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH="$VENV/bin:$HOME/.local/bin:$PATH"
CUDA_DIR=$(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -V | tail -1)
[ -n "$CUDA_DIR" ] && export CUDA_HOME="$CUDA_DIR" PATH="$CUDA_DIR/bin:$PATH"   # their inline kernels build with nvcc
mkdir -p "$TMPDIR" "$LSA_ROOT"
if [ -n "${LSA_SMOKE:-}" ]; then
  RUN=lsa_orig_smoke; TS=(--steps 200); DS=(--steps 100); TRUNK_STEP=200; DECAY_STEP=100
  LD=4096,131072; ROWS=1; LIM=3; RL_SHORT=4096; RL_LONG=65536; CLIM=20
else
  RUN=lsa_orig_m100_tok3.0e9_seed42; TS=(); DS=(); TRUNK_STEP=5722; DECAY_STEP=763
  LD=4096,16384,32768,65536,131072; ROWS=8; LIM=100; RL_SHORT=4096,8192,16384,32768; RL_LONG=65536; CLIM=0
fi
R="$MC_DISK/runs/$RUN"; S3R="s3://$MC_BUCKET/runs/e37/$RUN"
mkdir -p "$R"
log() { echo "[lsa_orig $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$R/events.log"; }
sync_up() {
  aws s3 sync --only-show-errors "$R/" "$S3R/" --exclude "*/checkpoint/*" --exclude "eval_work/*" || true
  for ph in trunk decay; do
    [ -d "$R/$ph/checkpoint" ] && aws s3 sync --only-show-errors --delete "$R/$ph/checkpoint/" "$S3R/$ph/checkpoint/" || true
  done
}
if [ ! -d "$R/trunk" ] && aws s3 ls "$S3R/events.log" >/dev/null 2>&1; then
  log "pulling the run folder back from S3"
  aws s3 sync --only-show-errors "$S3R/" "$R/" || true
  for d in "$R"/*/checkpoint/step-*; do   # a step folder copied while it was being written has no .metadata
    [ -d "$d" ] && [ ! -f "$d/.metadata" ] && { log "dropping incomplete ${d#$R/}"; rm -rf "$d"; }
  done
fi
( while sleep 900; do sync_up; done ) & SYNC=$!
finish() { kill "$SYNC" 2>/dev/null; sync_up; }
log "start $RUN on ${MC_NAME:-?} ${MC_IID:-} ($(nvidia-smi --query-gpu=name --format=csv,noheader)); code $(git log --oneline -1 | cut -c1-60); their repo $(git -C "$LSA_REPO" rev-parse HEAD)"

phase() {  # NAME FINAL_STEP ARGS...
  local name=$1 final=$2; shift 2
  if [ -f "$R/$name/DONE" ]; then log "$name done before"; return 0; fi
  log "$name start (resumes from $(ls -d "$R/$name"/checkpoint/step-* 2>/dev/null | sort -V | tail -1 | xargs -r basename || echo none))"
  "$VENV/bin/torchrun" --standalone --nproc_per_node=1 scripts/lsa_orig.py train --phase "$name" --data "$D" --out "$R" "$@" \
    >> "$R/$name.log" 2>&1
  local rc=$?
  if [ "$rc" = 0 ] && [ -f "$R/$name/checkpoint/step-$final/.metadata" ]; then
    touch "$R/$name/DONE"; log "$name done (step-$final)"; sync_up; return 0
  fi
  log "$name FAILED (rc $rc; tail of $name.log below)"; tail -30 "$R/$name.log" | tee -a "$R/events.log"; return 1
}
phase trunk "$TRUNK_STEP" ${TS[@]+"${TS[@]}"} || { finish; exit 1; }
phase decay "$DECAY_STEP" --init "$R/trunk/checkpoint/step-$TRUNK_STEP" ${DS[@]+"${DS[@]}"} || { finish; exit 1; }
CK="$R/decay/checkpoint/step-$DECAY_STEP"; TK="$R/trunk/checkpoint/step-$TRUNK_STEP"

# evals: lm-eval 0.4.11 (our runs' version) shadows their 0.4.13 for these processes only
OV="$MC_DISK/lmeval011"
[ -d "$OV/lm_eval" ] || uv pip install -q --python "$PY" --target "$OV" --no-deps "lm-eval==0.4.11" >> "$R/events.log" 2>&1 \
  || log "lm-eval 0.4.11 overlay FAILED: the evals use their 0.4.13"
E() { PYTHONPATH="$OV${PYTHONPATH:+:$PYTHONPATH}" "$PY" evaluation/lsa_orig_eval.py "$@"; }
ev() {  # NAME ARGS...  (skipped when NAME.json exists)
  local name=$1; shift
  [ -f "$R/$name.json" ] && { log "eval $name done before"; return 0; }
  log "eval $name start"
  E "$@" --checkpoint "$CK" --data "$D" --out "$R/$name.json.part" > "$R/$name.log" 2>&1 \
    && mv "$R/$name.json.part" "$R/$name.json" && log "eval $name done" || log "eval $name FAILED (see $name.log)"
  sync_up
}
( ev ruler_long ruler --lengths "$RL_LONG" --limit "$LIM" ) & LANE=$!
ev eval_longdoc longdoc --lengths "$LD" --rows "$ROWS" --open_off
if [ ! -f "$R/eval_val.json" ]; then
  E val --checkpoint "$CK" --name decay_final --data "$D" --out "$R/eval_val.json.part" > "$R/eval_val.log" 2>&1 \
    && E val --checkpoint "$TK" --name trunk_final --data "$D" --out "$R/eval_val.json.part" >> "$R/eval_val.log" 2>&1 \
    && mv "$R/eval_val.json.part" "$R/eval_val.json" && log "eval val done" || log "eval val FAILED (see eval_val.log)"
fi
ev eval_core core --limit "$CLIM"
ev ruler_short ruler --lengths "$RL_SHORT" --limit "$LIM"
wait "$LANE"

# RULER: one file in run_lsa_suite.py's layout (results / mean / tasks per length)
"$PY" - "$R" "$CK" "$LIM" <<'PY' || log "RULER merge FAILED"
import json, sys
from pathlib import Path
R, CK, LIM = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
parts = [json.loads((R / f"{k}.json").read_text()) for k in ("ruler_short", "ruler_long") if (R / f"{k}.json").exists()]
out = {"checkpoint": CK, "suite": "ruler", "lengths": [], "limit": LIM, "notes_off": False, "results": {}, "mean": {},
       "tasks": {}, "seconds_per_length": {}, "lm_eval": parts[0].get("lm_eval") if parts else None,
       "model": "lsa_orig (their trainer)", "note": "lsa_orig_run.sh: RULER 4k-32k and 64k in two lanes, merged"}
for p in parts:
    for k in ("results", "mean", "tasks", "seconds_per_length"):
        out[k].update(p.get(k, {}))
out["lengths"] = sorted(int(L) for L in out["results"])
for L in out["lengths"]:
    print(f"RULER {L}: {out['tasks'].get(str(L))} tasks, mean {out['mean'].get(str(L))}")
(R / "eval_ruler.json").write_text(json.dumps(out, indent=2, default=str))
PY
log "done: $(ls "$R"/eval_*.json 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
finish
touch "$R/RUN_DONE"; aws s3 cp --only-show-errors "$R/events.log" "$S3R/events.log" || true
