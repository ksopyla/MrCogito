#!/bin/bash
# E37 stage B, the later RULER pass on AWS (author 2026-11-12: "Yes, 32k and 64k"): RULER at 32k and 64k for the three
# full-budget models (e37a = the notebook model, lsa = our LSA port, lsa_orig = the original LSA). The 8k part ran at
# home (2026-11-11); the home cards were busy after it. A launch.sh job sized for one-L40S spot nodes (g6e.xlarge), one
# share of the work each; any GPU count works (e37_ruler_later.sh deals the pairs out to all the node's GPUs).
# Arguments: ARM:LENGTHS items, run in turn. Example:
#   MC_SPOT=1 bash infra/aws/launch.sh e37-later-e37a-64k 20 "infra/aws/jobs/e37_ruler_later_node.sh e37a:65536" \
#     arch-10m g6e.xlarge
# Per item:
#   1. the checkpoint from S3 (ours: the decay's final/; the original LSA: decay/checkpoint/step-763);
#   2. the run's later/ folder from S3 (pairs another node or a home server finished are skipped; the 8k files merge in);
#   3. infra/aws/jobs/e37_ruler_later.sh LENGTHS on all GPUs (100 documents per task, rows = 65536 / length);
#   4. every 10 min and at the end: later/ruler_*.{json,log} up to the run's S3 folder (no deletes; nothing else of the
#      run is touched) and the pass log as events_later_<node>.log;
#   5. at the end: later/ pulled again (other nodes' lengths of the same model), merged into eval_ruler_later.json,
#      uploaded; marker later/DONE_<node> (all of this item's pairs scored) or later/INCOMPLETE_<node>.
# A spot node taken back and relaunched under the same name resumes: step 2 skips what it uploaded.
# The original LSA runs in their venv: lsa_orig_node.sh's step 1 without causal-conv1d (only GLM's layers use it; the
# home pass on Odra ran the same way), lm-eval 0.4.11 first on PYTHONPATH as lsa_orig_run.sh's evals.
set -uo pipefail
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1
CUDA_DIR=$(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -V | tail -1)
[ -n "$CUDA_DIR" ] && export CUDA_HOME="$CUDA_DIR" PATH="$CUDA_DIR/bin:$PATH"   # their inline kernels build with nvcc
REPO="$(pwd)"
B="s3://$MC_BUCKET/runs/e37"; D="$MC_DISK/data"; W="$MC_DISK/later"
mkdir -p "$D" "$W"
log() { echo "[e37_later_node $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$W/node.log"; }
[ $# -ge 1 ] || { log "no ARM:LENGTHS items given"; exit 1; }
s3run() { if [ "$1" = lsa_orig ]; then echo "$B/lsa_orig_m100_tok3.0e9_seed42"; else echo "$B/e33n_m100_$1_tok3.0e9_seed42"; fi; }
up() {  # ARM: this node's pair files and pass log to the run's S3 folder
  local R; R=$(s3run "$1")
  [ -d "$W/$1/later" ] || return 0
  aws s3 sync --only-show-errors "$W/$1/later/" "$R/later/" --exclude "*" --include "ruler_*.json" --include "ruler_*.log" || true
  [ -f "$W/$1/events.log" ] && aws s3 cp --only-show-errors "$W/$1/events.log" "$R/events_later_${MC_NAME}.log" || true
}
( while sleep 600; do for it in "$@"; do up "${it%%:*}"; done; done ) & SYNC=$!
log "start on $MC_NAME ${MC_IID:-} ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ' ')); items: $*; code $(git -C "$REPO" log --oneline -1 | cut -c1-60)"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/tokenizer_gpt2/" "$D/tokenizer_gpt2/"

setup_orig() {  # their venv at the pinned commit (as lsa_orig_node.sh), the lm-eval 0.4.11 overlay
  local LSA="$MC_DISK/lsa" FULL_SHA=5ebea732d639576d2a7579b0de40e6252f9001f5 CUDA_DIR
  [ -x "$VENV/bin/python" ] && [ -d "$OV/lm_eval" ] && return 0
  [ -d "$LSA/.git" ] || git clone -q https://github.com/Prefrontal-AI/latent-sparse-attention.git "$LSA"
  git -C "$LSA" cat-file -e "$FULL_SHA^{commit}" 2>/dev/null || git -C "$LSA" fetch -q origin "$FULL_SHA"
  if ! git -C "$LSA" -c advice.detachedHead=false checkout -q --detach "$FULL_SHA^{commit}" \
     || [ "$(git -C "$LSA" rev-parse HEAD)" != "$FULL_SHA" ]; then
    log "pin mismatch: their repo is at $(git -C "$LSA" rev-parse HEAD 2>/dev/null), wanted $FULL_SHA"; return 1
  fi
  ( cd "$LSA" && UV_PROJECT_ENVIRONMENT="$VENV" uv sync --locked --extra training --no-install-package causal-conv1d ) \
    > "$MC_DISK/lsa_sync.log" 2>&1 || { log "their uv sync FAILED (tail below)"; tail -20 "$MC_DISK/lsa_sync.log" | tee -a "$W/node.log"; return 1; }
  uv pip install -q --python "$VENV/bin/python" --target "$OV" --no-deps "lm-eval==0.4.11" >> "$MC_DISK/lsa_sync.log" 2>&1 \
    || { log "lm-eval 0.4.11 overlay FAILED"; return 1; }
  "$VENV/bin/hf" download openai-community/gpt2 --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
    --include "*.json" "*.txt" > /dev/null 2>&1 || log "hub snapshot of gpt2 FAILED (their loader falls back to $D/tokenizer_gpt2)"
  PYTHONPATH="$OV" "$VENV/bin/python" -c "import torch, torchtitan, fla, lm_eval; print('their venv: torch', torch.__version__, 'fla', fla.__version__, 'lm_eval', lm_eval.__version__, lm_eval.__file__)" 2>&1 | tee -a "$W/node.log"
}
VENV="$MC_DISK/lsa-venv"; OV="$MC_DISK/lmeval011"

RC=0
for it in "$@"; do
  A=${it%%:*}; LS=${it#*:}; R=$(s3run "$A"); O="$W/$A"; T0=$(date -u +%s)
  mkdir -p "$O/later"
  log "$A: RULER $LS from $R"
  if [ "$A" = lsa_orig ]; then
    CK="$MC_DISK/ck/lsa_orig/decay/checkpoint/step-763"
    aws s3 sync --only-show-errors "$R/decay/checkpoint/step-763/" "$CK/"
    [ -f "$CK/.metadata" ] || { log "$A: no checkpoint (.metadata missing)"; RC=1; continue; }
    setup_orig || { RC=1; continue; }
  else
    PFX=$(aws s3 ls --recursive "$R/decay/" | awk '{print $4}' | grep '/final/model.safetensors$' | sed 's#/model.safetensors$##' | head -1)
    [ -n "$PFX" ] || { log "$A: no final/ under $R/decay/"; RC=1; continue; }
    CK="$MC_DISK/ck/$A/final"
    aws s3 sync --only-show-errors "s3://$MC_BUCKET/$PFX/" "$CK/" --exclude training_args.bin
  fi
  aws s3 sync --only-show-errors "$R/later/" "$O/later/" --exclude "DONE_*" --exclude "INCOMPLETE_*"
  log "$A: checkpoint $(du -sh "$CK" | cut -f1), $(ls "$O"/later/ruler_*.json 2>/dev/null | wc -l) pair files from S3; setup $(( $(date -u +%s) - T0 )) s"
  pass() {
    if [ "$A" = lsa_orig ]; then
      LSA_REPO="$MC_DISK/lsa" LSA_ROOT="$MC_DISK/lsa_root" PATH="$VENV/bin:$PATH" E37_LSA_ORIG=1 LSA_PY="$VENV/bin/python" \
        LSA_OV="$OV" bash "$REPO/infra/aws/jobs/e37_ruler_later.sh" "$CK" "$O" "$D" "$LS"
    else
      bash "$REPO/infra/aws/jobs/e37_ruler_later.sh" "$CK" "$O" "$D" "$LS"
    fi
  }
  mkdir -p "$MC_DISK/lsa_root"
  pass >> "$W/node.log" 2>&1
  up "$A"
  # merge with what the other nodes uploaded for this model (a rerun with nothing left only merges)
  aws s3 sync --only-show-errors "$R/later/" "$O/later/" --exclude "DONE_*" --exclude "INCOMPLETE_*"
  pass >> "$W/node.log" 2>&1
  up "$A"
  aws s3 cp --only-show-errors "$O/eval_ruler_later.json" "$R/eval_ruler_later.json" || true
  miss=0
  for L in ${LS//,/ }; do
    n=$(ls "$O"/later/ruler_"${L}"_*.json 2>/dev/null | wc -l); [ "$n" -ge 13 ] || miss=$((miss + 13 - n))
  done
  if [ "$miss" = 0 ]; then M=DONE; else M=INCOMPLETE; RC=1; fi
  echo "$M $(date -u +%FT%TZ) $A $LS, $miss pairs missing, $(( ($(date -u +%s) - T0) / 60 )) min" | aws s3 cp - "$R/later/${M}_${MC_NAME}" || true
  log "$A: $M ($miss pairs missing) after $(( ($(date -u +%s) - T0) / 60 )) min; $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['mean'])" "$O/eval_ruler_later.json" 2>&1)"
done
kill "$SYNC" 2>/dev/null
for it in "$@"; do up "${it%%:*}"; done
aws s3 cp --only-show-errors "$W/node.log" "s3://$MC_BUCKET/logs/$MC_NAME/node.log" || true
exit $RC
