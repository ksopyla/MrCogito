#!/bin/bash
# E37 stage B, the original LSA model (scripts/lsa_orig.py; their trainer, our m100 data / budget / evals) on the node
# set up by lsa_orig_node.sh: trunk -> decay -> evals -> S3. Called by lsa_orig_node.sh on GO; by hand (in tmux) it
# resumes: a phase with a DONE marker is skipped, a phase with checkpoints continues from the newest (torchtitan
# resumes the optimizer, the schedule and our loader's position), and a node without the run folder first pulls it
# back from S3. Checkpoints go to S3 every 15 min (the newest two kept by the trainer, mirrored with --delete).
# Hardware: one H100 (p5.4xlarge) or a 4 x L40S node (g6e.12xlarge), from nvidia-smi:
#   * all the node's GPUs train one model (torchrun, their FSDP data parallel); every optimizer step is still the same
#     128 x 4k rows (the loader deals micro-batches out to the ranks in our trainer's global order);
#   * micro-batch per GPU: 80 GB cards trunk 8 / decay 2 rows, smaller cards 4 / 1 (gradient accumulation fills the step);
#   * on sm_90 their CUDA kernels (cuda_sparse, lsa_rank) run the LSA layer; elsewhere their own Triton kernels
#     (gather_sparse, lsa_select; their code picks them); the Gated DeltaNet is attn-gym's Triton path (sm_80+).
#   LSA_SMOKE=1: the smoke test — trunk 200 steps, decay 100 steps from its weights, the evals with tiny limits on the
#   same lanes; folder $MC_DISK/runs/lsa_orig_smoke_<gpu> (h100, l40s), S3 runs/e37/lsa_orig_smoke_<gpu>/.
# Evals (eval_longdoc.json, eval_ruler.json, eval_core.json, eval_val.json): evaluation/lsa_orig_eval.py, one lane per
# GPU (1 GPU: RULER 64k beside everything else; 4 GPUs: long-document + val + core + RULER 4k-16k | 32k | 64k
# in two task groups),
# lm-eval 0.4.11 (ours) ahead of their 0.4.13 on PYTHONPATH.
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
G=$(nvidia-smi -L | wc -l)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [ "${MEM:-0}" -ge 70000 ]; then MT=8; MD=2; BT=131072; else MT=4; MD=1; BT=65536; fi
if [ -n "${LSA_SMOKE:-}" ]; then
  TAG=$(echo "$GPU" | grep -oiE "h100|h200|b200|l40s|a100|l4" | head -1 | tr '[:upper:]' '[:lower:]')
  RUN=lsa_orig_smoke_${TAG:-gpu}; TS=(--steps 200); DS=(--steps 100); TRUNK_STEP=200; DECAY_STEP=100
  LD=4096,131072; ROWS=1; LIM=3; CLIM=20
else
  RUN=lsa_orig_m100_tok3.0e9_seed42; TS=(); DS=(); TRUNK_STEP=5722; DECAY_STEP=763
  LD=4096,16384,32768,65536,131072; ROWS=8; LIM=100; CLIM=0
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
printf '{"gpus": %s, "name": "%s", "mem_mib": %s, "compute_cap": "%s", "micro_trunk": %s, "micro_decay": %s, "node": "%s", "id": "%s"}\n' \
  "$G" "$GPU" "$MEM" "$CC" "$MT" "$MD" "${MC_NAME:-}" "${MC_IID:-}" > "$R/hw.json"
( while sleep 900; do sync_up; done ) & SYNC=$!
finish() { kill "$SYNC" 2>/dev/null; sync_up; }
log "start $RUN on ${MC_NAME:-?} ${MC_IID:-} ($G x $GPU, sm $CC); code $(git log --oneline -1 | cut -c1-60); their repo $(git -C "$LSA_REPO" rev-parse HEAD)"

phase() {  # NAME FINAL_STEP ARGS...
  local name=$1 final=$2; shift 2
  if [ -f "$R/$name/DONE" ]; then log "$name done before"; return 0; fi
  log "$name start on $G GPU(s) (resumes from $(ls -d "$R/$name"/checkpoint/step-* 2>/dev/null | sort -V | tail -1 | xargs -r basename || echo none))"
  "$VENV/bin/torchrun" --standalone --nproc_per_node="$G" scripts/lsa_orig.py train --phase "$name" --data "$D" --out "$R" "$@" \
    >> "$R/$name.log" 2>&1
  local rc=$?
  if [ "$rc" = 0 ] && [ -f "$R/$name/checkpoint/step-$final/.metadata" ]; then
    touch "$R/$name/DONE"; log "$name done (step-$final)"; sync_up; return 0
  fi
  log "$name FAILED (rc $rc; tail of $name.log below)"; tail -30 "$R/$name.log" | tee -a "$R/events.log"; return 1
}
phase trunk "$TRUNK_STEP" --micro "$MT" ${TS[@]+"${TS[@]}"} || { finish; exit 1; }
phase decay "$DECAY_STEP" --micro "$MD" --init "$R/trunk/checkpoint/step-$TRUNK_STEP" ${DS[@]+"${DS[@]}"} || { finish; exit 1; }
CK="$R/decay/checkpoint/step-$DECAY_STEP"; TK="$R/trunk/checkpoint/step-$TRUNK_STEP"

# evals: lm-eval 0.4.11 (our runs' version) shadows their 0.4.13 for these processes only
OV="$MC_DISK/lmeval011"
[ -d "$OV/lm_eval" ] || uv pip install -q --python "$PY" --target "$OV" --no-deps "lm-eval==0.4.11" >> "$R/events.log" 2>&1 \
  || log "lm-eval 0.4.11 overlay FAILED: the evals use their 0.4.13"
E() { PYTHONPATH="$OV${PYTHONPATH:+:$PYTHONPATH}" "$PY" evaluation/lsa_orig_eval.py "$@"; }
ev() {  # GPU NAME ARGS...  (skipped when NAME.json exists)
  local gpu=$1 name=$2; shift 2
  [ -f "$R/$name.json" ] && { log "eval $name done before"; return 0; }
  log "eval $name start (GPU $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu E "$@" --checkpoint "$CK" --data "$D" --out "$R/$name.json.part" > "$R/$name.log" 2>&1 \
    && mv "$R/$name.json.part" "$R/$name.json" && log "eval $name done" || log "eval $name FAILED (see $name.log)"
  sync_up
}
ev_val() {  # GPU
  [ -f "$R/eval_val.json" ] && return 0
  CUDA_VISIBLE_DEVICES=$1 E val --checkpoint "$CK" --name decay_final --data "$D" --out "$R/eval_val.json.part" > "$R/eval_val.log" 2>&1 \
    && CUDA_VISIBLE_DEVICES=$1 E val --checkpoint "$TK" --name trunk_final --data "$D" --out "$R/eval_val.json.part" >> "$R/eval_val.log" 2>&1 \
    && mv "$R/eval_val.json.part" "$R/eval_val.json" && log "eval val done" || log "eval val FAILED (see eval_val.log)"
}
main_lane() {  # GPU
  ev "$1" eval_longdoc longdoc --lengths "$LD" --rows "$ROWS" --open_off
  ev_val "$1"
  ev "$1" eval_core core --limit "$CLIM"
}
RB=(--limit "$LIM" --batch_tokens "$BT")
if [ "$G" -ge 4 ]; then
  # RULER cost ~ length (uncached decoding): 4k+8k+16k ~ 7, 32k ~ 8, 64k ~ 16 units -> 64k split by task over 2 GPUs
  R64A=niah_single_1,niah_multikey_2,niah_multivalue,ruler_cwe,ruler_qa_hotpot,niah_single_3,niah_multikey_1
  R64B=niah_single_2,niah_multikey_3,niah_multiquery,ruler_fwe,ruler_vt,ruler_qa_squad
  echo '{"lanes": [["eval_longdoc", "eval_val", "eval_core", "ruler_a"], ["ruler_b"], ["ruler_c1"], ["ruler_c2"]]}' > "$R/lanes.json"
  ( ev 1 ruler_b ruler --lengths 32768 "${RB[@]}" ) & L1=$!
  ( ev 2 ruler_c1 ruler --lengths 65536 --tasks "$R64A" "${RB[@]}" ) & L2=$!
  ( ev 3 ruler_c2 ruler --lengths 65536 --tasks "$R64B" "${RB[@]}" ) & L3=$!
  main_lane 0
  ev 0 ruler_a ruler --lengths 4096,8192,16384 "${RB[@]}"
  wait "$L1" "$L2" "$L3"
else
  echo '{"lanes": [["eval_longdoc", "eval_val", "eval_core", "ruler_a"], ["ruler_c"]]}' > "$R/lanes.json"
  ( ev 0 ruler_c ruler --lengths 65536 "${RB[@]}" ) & L3=$!
  main_lane 0
  ev 0 ruler_a ruler --lengths 4096,8192,16384,32768 "${RB[@]}"
  wait "$L3"
fi

# RULER: one file in run_lsa_suite.py's layout (results / mean / tasks per length)
"$PY" - "$R" "$CK" "$LIM" <<'PY' || log "RULER merge FAILED"
import json, sys
from pathlib import Path
R, CK, LIM = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
parts = [json.loads(p.read_text()) for p in sorted(R.glob("ruler_*.json"))]
out = {"checkpoint": CK, "suite": "ruler", "lengths": [], "limit": LIM, "notes_off": False, "results": {}, "mean": {},
       "tasks": {}, "seconds_per_length": {}, "lm_eval": parts[0].get("lm_eval") if parts else None,
       "model": "lsa_orig (their trainer)", "note": "lsa_orig_run.sh: RULER lengths in lanes (ruler_*.json), merged"}
for p in parts:   # a length may come in task groups from several lanes: merge the tasks, then recompute the mean
    for L, r in p.get("results", {}).items():
        out["results"].setdefault(L, {}).update(r)
    for L, sec in p.get("seconds_per_length", {}).items():
        out["seconds_per_length"][L] = max(sec, out["seconds_per_length"].get(L, 0))
for L, r in out["results"].items():
    vals = [x.get(f"{L},none") for x in r.values() if isinstance(x, dict)]
    vals = [v for v in vals if isinstance(v, (int, float)) and v >= 0]
    out["mean"][L] = sum(vals) / len(vals) if vals else None
    out["tasks"][L] = len(vals)
out["lengths"] = sorted(int(L) for L in out["results"])
for L in out["lengths"]:
    print(f"RULER {L}: {out['tasks'].get(str(L))} tasks, mean {out['mean'].get(str(L))}")
(R / "eval_ruler.json").write_text(json.dumps(out, indent=2, default=str))
PY
log "done: $(ls "$R"/eval_*.json 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
finish
touch "$R/RUN_DONE"; aws s3 cp --only-show-errors "$R/events.log" "$S3R/events.log" || true
