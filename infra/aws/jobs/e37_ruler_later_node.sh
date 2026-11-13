#!/bin/bash
# E37 stage B, the later RULER pass on AWS (author 2026-11-12: "Yes, 32k and 64k"): RULER at 32k and 64k for the three
# full-budget models (e37a = the notebook model, lsa = our LSA port, lsa_orig = the original LSA). The 8k part ran at
# home (2026-11-11); the home cards were busy after it. A launch.sh job sized for one-L40S spot nodes (g6e.xlarge), one
# share of the work each; any GPU count works (e37_ruler_later.sh deals the pairs out to all the node's GPUs).
# Arguments: ARM:LENGTHS items (or ARM:val, LSA's validation loss: val_item below), run in turn. Our models' code comes
# from the node's branch: a model whose code is on another branch (the catalogue model, e37-catalogue) runs from a
# branch that is its branch plus these files (e37-catalogue-later). Example:
#   MC_SPOT=1 bash infra/aws/launch.sh e37-later-e37a-64k 20 "infra/aws/jobs/e37_ruler_later_node.sh e37a:65536" \
#     arch-10m g6e.xlarge
# Per item:
#   1. the checkpoint from S3 (ours: the decay's final/; the original LSA: decay/checkpoint/step-763);
#   2. the run's later/ folder from S3 (pairs another node or a home server finished are skipped; the 8k files merge in);
#   3. infra/aws/jobs/e37_ruler_later.sh LENGTHS on all GPUs (100 documents per task, rows = 65536 / length);
#   4. every 10 min and at the end: later/ruler_*.{json,log} up to the run's S3 folder (no deletes; nothing else of the
#      run is touched) and the pass log as events_later_<node>.log;
#   5. at the end: later/ pulled again (other nodes' lengths of the same model), merged into eval_ruler_later.json,
#      uploaded. After the last item: marker later/DONE_<node> (every item's pairs scored, val written) or
#      later/INCOMPLETE_<node>, in each of the node's runs.
# ARM:LENGTHS@N scores the first N documents per task (default 100, the nodes' 4k / 16k protocol). 64k ran at @25
# (2026-11-12): on one L40S our models' uncached 64k decoding takes 1.5-3 min a document (cost ~ length^2 there),
# 100 documents x 13 tasks would be 30-60 h per model.
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
# the pair folder and merged file (e37_ruler_later.sh); e37_needles_node.sh sets needles / eval_ruler_needles.json
SUB=${E37_RULER_SUB:-later}; MERGED=${E37_RULER_MERGED:-eval_ruler_later.json}; export E37_RULER_SUB=$SUB E37_RULER_MERGED=$MERGED
mkdir -p "$D" "$W"
log() { echo "[e37_later_node $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$W/node.log"; }
[ $# -ge 1 ] || { log "no ARM:LENGTHS items given"; exit 1; }
s3run() { if [ "$1" = lsa_orig ]; then echo "$B/lsa_orig_m100_tok3.0e9_seed42"; else echo "$B/e33n_m100_$1_tok3.0e9_seed42"; fi; }
up() {  # ARM: this node's pair files and pass log to the run's S3 folder
  local R; R=$(s3run "$1")
  [ -d "$W/$1/$SUB" ] || return 0
  aws s3 sync --only-show-errors "$W/$1/$SUB/" "$R/$SUB/" --exclude "*" --include "ruler_*.json" --include "ruler_*.log" || true
  [ -f "$W/$1/events.log" ] && aws s3 cp --only-show-errors "$W/$1/events.log" "$R/events_${SUB}_${MC_NAME}.log" || true
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
fetch_final() {  # S3_PHASE_DIR LOCAL: our checkpoint's final/ of that phase (trunk/ or decay/)
  local PFX
  PFX=$(aws s3 ls --recursive "$1" | awk '{print $4}' | grep '/final/model.safetensors$' | sed 's#/model.safetensors$##' | head -1)
  [ -n "$PFX" ] || { log "no final/ under $1"; return 1; }
  aws s3 sync --only-show-errors "s3://$MC_BUCKET/$PFX/" "$2/" --exclude training_args.bin
}
val_item() {  # ARM: LSA's validation loss (evaluation/val_loss.py, as run at home for e37a / lsa): trunk final on
  # FineWeb-Edu val 4k, decay final on FineWeb-Edu val 4k + LongDocs-lite val 16k -> the run's eval_val.json (kept if
  # already in S3). Data: the val shards, one FineWeb-Edu train shard (the manifest loader opens the train side too),
  # LongDocs-lite.
  local A=$1 R O; R=$(s3run "$1"); O="$W/$1"; mkdir -p "$O"
  if aws s3 ls "$R/eval_val.json" >/dev/null 2>&1; then log "$A: eval_val.json is in S3 already; val skipped"; return 0; fi
  mkdir -p "$D/fineweb_edu_gpt2"
  for f in finewebedu_val_000000.bin finewebedu_train_000001.bin; do
    aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/$f" "$D/fineweb_edu_gpt2/"
  done
  aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/longdocs_lite_gpt2/" "$D/longdocs_lite_gpt2/"
  fetch_final "$R/trunk/" "$MC_DISK/ck/$A/trunk_final" && fetch_final "$R/decay/" "$MC_DISK/ck/$A/final" || return 1
  log "$A: validation loss"
  if uv run python evaluation/val_loss.py --checkpoint "$MC_DISK/ck/$A/trunk_final" --name trunk_final --no_longdocs \
       --data "$D" --out "$O/eval_val.json" >> "$O/eval_val.log" 2>&1 \
     && uv run python evaluation/val_loss.py --checkpoint "$MC_DISK/ck/$A/final" --name decay_final \
       --data "$D" --out "$O/eval_val.json" >> "$O/eval_val.log" 2>&1; then
    aws s3 cp --only-show-errors "$O/eval_val.json" "$R/eval_val.json"
    aws s3 cp --only-show-errors "$O/eval_val.log" "$R/eval_val.log"
    log "$A: val done: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: round(v['fineweb_val_4k']['token_mean'], 4) for k, v in d.items() if isinstance(v, dict)})" "$O/eval_val.json")"
  else
    log "$A: val FAILED (tail below)"; tail -15 "$O/eval_val.log" | tee -a "$W/node.log"
    aws s3 cp --only-show-errors "$O/eval_val.log" "$R/eval_val_${MC_NAME}_failed.log"; return 1
  fi
}

ntk_item() {  # ARM: the long-document loss (as e37_evals.sh: 4k-128k, rows 8, the same flags) at an eval-time
  # RoPE base on every layer (E37_NTK_THETA, default 4.28e6 = NTK-aware 500k x 8^(64/62) for 16k -> 128k, head_dim 64)
  # -> the run's eval_longdoc_ntk.json (kept if already in S3)
  local A=$1 R O TH=${E37_NTK_THETA:-4.28e6}; R=$(s3run "$1"); O="$W/$1"; mkdir -p "$O"
  if aws s3 ls "$R/eval_longdoc_ntk.json" >/dev/null 2>&1; then log "$A: eval_longdoc_ntk.json is in S3 already"; return 0; fi
  aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/longdocs_lite_gpt2/" "$D/longdocs_lite_gpt2/"
  fetch_final "$R/decay/" "$MC_DISK/ck/$A/final" || return 1
  log "$A: long-document loss at RoPE base $TH"
  if uv run python evaluation/long_doc_loss.py --checkpoint "$MC_DISK/ck/$A/final" --data "$D/longdocs_lite_gpt2" \
       --lengths 4096,16384,32768,65536,131072 --rows 8 --notes_off --open_off --gist_off --rope_theta "$TH" \
       --out "$O/eval_longdoc_ntk.json" > "$O/eval_longdoc_ntk.log" 2>&1 && ! grep -q '"error"' "$O/eval_longdoc_ntk.json"; then
    aws s3 cp --only-show-errors "$O/eval_longdoc_ntk.json" "$R/eval_longdoc_ntk.json"
    aws s3 cp --only-show-errors "$O/eval_longdoc_ntk.log" "$R/eval_longdoc_ntk.log"
    log "$A: long-document loss at base $TH done"
  else
    log "$A: long-document loss at base $TH FAILED (tail below)"; tail -15 "$O/eval_longdoc_ntk.log" | tee -a "$W/node.log"
    aws s3 cp --only-show-errors "$O/eval_longdoc_ntk.log" "$R/eval_longdoc_ntk_${MC_NAME}_failed.log"; return 1
  fi
}
cached_or_refuse() {  # CHECKPOINT: with E37_REQUIRE_CACHE=1, the model must take the cached decoding (nn/incremental.py
  # raises NotImplementedError for reads it does not cover, and the harness would fall back to the slow loop silently)
  [ -n "${E37_REQUIRE_CACHE:-}" ] || return 0
  uv run python -c "import sys; sys.path.insert(0, '.')
from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval
from nn.incremental import decode_kind
print('cached decoding:', decode_kind(load_perceiver_ar_for_eval(sys.argv[1], device='cpu', attn_backend='sdpa')))" "$1" 2>&1 | tail -1
  return "${PIPESTATUS[0]}"
}

RC=0; MISS=0; ARMS=""
for it in "$@"; do
  A=${it%%:*}; LS=${it#*:}; R=$(s3run "$A"); O="$W/$A"; T0=$(date -u +%s)
  if [ "$LS" = val ]; then ARMS="$ARMS $A"; val_item "$A" || { RC=1; MISS=$((MISS + 1)); }; continue; fi
  if [ "$LS" = ntk ]; then ARMS="$ARMS $A"; ntk_item "$A" || { RC=1; MISS=$((MISS + 1)); }; continue; fi
  LIM=100; case "$LS" in *@*) LIM=${LS#*@}; LS=${LS%@*} ;; esac
  ARMS="$ARMS $A"
  mkdir -p "$O/$SUB"
  log "$A: RULER $LS ($LIM documents per task) from $R"
  if [ "$A" = lsa_orig ]; then
    CK="$MC_DISK/ck/lsa_orig/decay/checkpoint/step-763"
    aws s3 sync --only-show-errors "$R/decay/checkpoint/step-763/" "$CK/"
    [ -f "$CK/.metadata" ] || { log "$A: no checkpoint (.metadata missing)"; RC=1; continue; }
    setup_orig || { RC=1; continue; }
  else
    CK="$MC_DISK/ck/$A/final"
    fetch_final "$R/decay/" "$CK" || { RC=1; continue; }
    if ! c=$(cached_or_refuse "$CK"); then
      log "$A: no cached decoding for this model ($c); item skipped"; RC=1; MISS=$((MISS + 13)); continue
    fi
    [ -n "$c" ] && log "$A: $c"
  fi
  aws s3 sync --only-show-errors "$R/$SUB/" "$O/$SUB/" --exclude "DONE_*" --exclude "INCOMPLETE_*"
  log "$A: checkpoint $(du -sh "$CK" | cut -f1), $(ls "$O"/$SUB/ruler_*.json 2>/dev/null | wc -l) pair files from S3; setup $(( $(date -u +%s) - T0 )) s"
  pass() {
    if [ "$A" = lsa_orig ]; then
      LSA_REPO="$MC_DISK/lsa" LSA_ROOT="$MC_DISK/lsa_root" PATH="$VENV/bin:$PATH" E37_LSA_ORIG=1 LSA_PY="$VENV/bin/python" \
        LSA_OV="$OV" E37_RULER_LIMIT="$LIM" bash "$REPO/infra/aws/jobs/e37_ruler_later.sh" "$CK" "$O" "$D" "$LS"
    else
      E37_RULER_LIMIT="$LIM" bash "$REPO/infra/aws/jobs/e37_ruler_later.sh" "$CK" "$O" "$D" "$LS"
    fi
  }
  mkdir -p "$MC_DISK/lsa_root"
  pass >> "$W/node.log" 2>&1
  up "$A"
  # merge with what the other nodes uploaded for this model (a rerun with nothing left only merges)
  aws s3 sync --only-show-errors "$R/$SUB/" "$O/$SUB/" --exclude "DONE_*" --exclude "INCOMPLETE_*"
  pass >> "$W/node.log" 2>&1
  up "$A"
  aws s3 cp --only-show-errors "$O/$MERGED" "$R/$MERGED" || true
  miss=0
  for L in ${LS//,/ }; do
    n=$(ls "$O"/$SUB/ruler_"${L}"_*.json 2>/dev/null | wc -l); [ "$n" -ge 13 ] || miss=$((miss + 13 - n))
  done
  MISS=$((MISS + miss)); [ "$miss" = 0 ] || RC=1
  log "$A: RULER $LS ($LIM documents) $miss pairs missing after $(( ($(date -u +%s) - T0) / 60 )) min; $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['mean'])" "$O/$MERGED" 2>&1)"
done
kill "$SYNC" 2>/dev/null
for it in "$@"; do up "${it%%:*}"; done
# the node's marker, once all its items ran (the watcher relaunches a node that is gone without one)
if [ "$MISS" = 0 ]; then M=DONE; else M=INCOMPLETE; fi
for A in $(echo "$ARMS" | tr ' ' '\n' | sort -u); do
  echo "$M $(date -u +%FT%TZ) $MC_NAME items: $*; $MISS pairs missing" | aws s3 cp - "$(s3run "$A")/$SUB/${M}_${MC_NAME}" || true
done
log "node $M ($MISS pairs missing)"
aws s3 cp --only-show-errors "$W/node.log" "s3://$MC_BUCKET/logs/$MC_NAME/node.log" || true
exit $RC
