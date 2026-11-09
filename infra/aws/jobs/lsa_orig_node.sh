#!/bin/bash
# mc: no-sync
# E37 stage B, the ORIGINAL Latent Sparse Attention model (Kubicek et al., arXiv 2610.42147; their code
# github.com/Prefrontal-AI/latent-sparse-attention @ 5ebea732d639 (pinned, full SHA below), trainer `lsa native train`) at ~160M on one H100,
# with our data, token budget and evals. A launch.sh job (on demand p5.4xlarge):
#   1. their stack in its own venv ($MC_DISK/lsa-venv, from their uv.lock: torch 2.13.0 cu129, torchtitan @56a721b,
#      flash-linear-attention 0.5.2, causal-conv1d 1.7.0 built here, FlashAttention-3 / -2 prebuilt wheels);
#      our uv env is not needed (no uv sync of this repo: the line above);
#   2. our shards from S3 (the m100 trunk's 32 FineWeb-Edu shards + val, LongDocs-lite, GPT-2 tokenizer);
#   3. development over ssh (smoke tests): waits for $MC_DISK/GO, at most DEV_HOURS (default 6, from the job's
#      start; `echo <epoch> > $MC_DISK/DEV_UNTIL` moves the deadline); no GO by then -> the job ends, the node
#      powers off (never an idle H100);
#   4. on GO: git pull of this branch, then infra/aws/jobs/lsa_orig_run.sh (trunk -> decay -> evals -> S3); when it
#      returns the job ends and node_run.sh powers the node off.
# Usage: bash infra/aws/launch.sh lsa-orig-h100 50 infra/aws/jobs/lsa_orig_node.sh arch-10m p5.4xlarge
set -uo pipefail
source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1
REPO="$(pwd)"
LSA="$MC_DISK/lsa"; VENV="$MC_DISK/lsa-venv"; D="$MC_DISK/data"
# their HEAD 5ebea73 (2026-11-09), resolved once from GitHub; the job refuses any other commit
FULL_SHA=5ebea732d639576d2a7579b0de40e6252f9001f5
T0=$(date -u +%s)
log() { echo "[lsa_orig_node $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$MC_DISK/lsa_setup.log"; }

log "setup on $MC_NAME ${MC_IID:-} ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null))"
# 1. their code at the pinned commit (a failed checkout or another HEAD stops the job), then their locked environment
[ -d "$LSA/.git" ] || git clone -q https://github.com/Prefrontal-AI/latent-sparse-attention.git "$LSA"
git -C "$LSA" cat-file -e "$FULL_SHA^{commit}" 2>/dev/null || git -C "$LSA" fetch -q origin "$FULL_SHA"
if ! git -C "$LSA" -c advice.detachedHead=false checkout -q --detach "$FULL_SHA^{commit}" \
   || [ "$(git -C "$LSA" rev-parse HEAD)" != "$FULL_SHA" ]; then
  log "pin mismatch: their repo is at $(git -C "$LSA" rev-parse HEAD 2>/dev/null), wanted $FULL_SHA"; exit 1
fi
log "their repo at $(git -C "$LSA" log --oneline -1) ($FULL_SHA)"
CUDA_DIR=$(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -V | tail -1)
[ -n "$CUDA_DIR" ] && export CUDA_HOME="$CUDA_DIR" PATH="$CUDA_DIR/bin:$PATH"
log "nvcc: $(nvcc --version 2>/dev/null | tail -2 | tr '\n' ' ') (CUDA_HOME=${CUDA_HOME:-none})"
export UV_PROJECT_ENVIRONMENT="$VENV" TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=16
( cd "$LSA" && uv --version && uv sync --locked --extra training ) > "$MC_DISK/lsa_sync.log" 2>&1
if [ $? -ne 0 ]; then
  log "uv sync with causal-conv1d FAILED (tail of lsa_sync.log below); retrying without it (only GLM's layers use it)"
  tail -20 "$MC_DISK/lsa_sync.log" | tee -a "$MC_DISK/lsa_setup.log"
  ( cd "$LSA" && uv sync --locked --extra training --no-install-package causal-conv1d ) >> "$MC_DISK/lsa_sync.log" 2>&1 \
    || log "uv sync FAILED again (see $MC_DISK/lsa_sync.log)"
fi
"$VENV/bin/python" -c "import torch, torchtitan, fla, flash_attn_interface, lm_eval; print('torch', torch.__version__, torch.version.cuda, 'fla', fla.__version__, 'lm_eval', lm_eval.__version__)" 2>&1 | tee -a "$MC_DISK/lsa_setup.log"
"$VENV/bin/python" -c "import causal_conv1d; print('causal_conv1d', causal_conv1d.__version__)" 2>&1 | tail -1 | tee -a "$MC_DISK/lsa_setup.log"

# 2. data, exactly the m100 runs' set (e33n_run.sh: FineWeb-Edu shards 1..32 for a 3.0B trunk, the val shard)
mkdir -p "$D/fineweb_edu_gpt2"
aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/finewebedu_val_000000.bin" "$D/fineweb_edu_gpt2/"
for i in $(seq 1 32); do
  f=$(printf "finewebedu_train_%06d.bin" "$i")
  aws s3 cp --only-show-errors "s3://$MC_BUCKET/data/fineweb_edu_gpt2/$f" "$D/fineweb_edu_gpt2/" || true
done
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/tokenizer_gpt2/" "$D/tokenizer_gpt2/"
aws s3 sync --only-show-errors "s3://$MC_BUCKET/data/longdocs_lite_gpt2/" "$D/longdocs_lite_gpt2/"
log "data: $(ls "$D"/fineweb_edu_gpt2/finewebedu_train_*.bin | wc -l) FineWeb shards, $(ls "$D"/longdocs_lite_gpt2/*_train_*.bin | wc -l) LongDocs-lite shards"
# their trainer reads GPT-2's tokenizer from the pinned hub snapshot (src/lsa/assets/models/gpt2.yaml)
"$VENV/bin/hf" download openai-community/gpt2 --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --include "*.json" "*.txt" > /dev/null 2>&1 || log "hub snapshot of gpt2 FAILED (the eval/training driver falls back to $D/tokenizer_gpt2)"
log "setup done in $(( ($(date -u +%s) - T0) / 60 )) min"
touch "$MC_DISK/LSA_READY"

# 3. development window: wait for GO, never past the deadline
DEADLINE=$(( T0 + ${DEV_HOURS:-6} * 3600 ))
while [ ! -f "$MC_DISK/GO" ]; do
  [ -f "$MC_DISK/DEV_UNTIL" ] && DEADLINE=$(cat "$MC_DISK/DEV_UNTIL")
  if [ "$(date -u +%s)" -ge "$DEADLINE" ]; then log "no GO by the deadline; the job ends (node powers off)"; exit 0; fi
  sleep 60
done

# 4. the run
log "GO: $(cat "$MC_DISK/GO" 2>/dev/null)"
git -C "$REPO" pull -q --ff-only || log "git pull FAILED; running the checkout as it is"
log "code at $(git -C "$REPO" log --oneline -1)"
bash "$REPO/infra/aws/jobs/lsa_orig_run.sh"
log "run script returned $?"
