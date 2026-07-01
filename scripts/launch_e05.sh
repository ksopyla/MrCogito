#!/bin/bash
# E05 windowed prefix→suffix — the ONE entrypoint for both A/B arms:
#   OPTIMIZER=adam   bash scripts/launch_e05.sh   # Adam arm (Odra)
#   OPTIMIZER=muon   bash scripts/launch_e05.sh   # Muon arm (Polonez) — fresh, token-matched to Adam
#
# This wrapper pins the E05 protocol (model config, mix, objective, K=128, seq 2K) and the
# token-matched effective batch, then delegates to the GENERIC launcher (train_perceiver_denoise_
# multigpu.sh), which owns ALL training defaults + the accelerate invocation + the gated
# pretokenize phase. Override any knob by exporting it before invocation.
#
# A/B invariant: both arms share seed/model/mix/effective-batch/epochs; ONLY the optimizer (and its
# LR) differ. NUM_EPOCHS MUST match the live Adam arm so both see identical tokens — confirm it.
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/remote_paths.sh
source "${SCRIPT_DIR}/remote_paths.sh"

# ---- Identity + E05 architecture pins (the protocol; generic launcher supplies the rest) ----
export EXPERIMENT_ID="${EXPERIMENT_ID:-E05}"
export DECODER_TYPE=causal_ar
export DECODER_CONTEXT_WINDOW="${DECODER_CONTEXT_WINDOW:-128}"   # K=128 fixed (coherence window; never scale to N)
export DECODER_POS_TYPE=rope
export OBJECTIVE_VARIANT=prefix_suffix
export MAX_SEQ_LENGTH=2048
export HIDDEN_SIZE=768
export TOKEN_EMBEDDING_DIM=256
export NUM_LAYERS=6
export CONCEPT_NUM=128
export DECODER_NUM_LAYERS=4
export INTERMEDIATE_SIZE=2048
export HIDDEN_ACT=silu
export NORM_TYPE=rmsnorm
export TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M
export SEED=42

# ---- Data: SmolLM3-inspired 2K long-context mix. The generic launcher pretokenizes it (gated by
#       PRETOKENIZE_MIX) then trains from the manifest. Re-run training only: SKIP_PRETOKENIZE=1.
#       Polonez can bump pretokenize speed: TRAIN_NUM_PROC=32 TEST_NUM_PROC=8 (output is identical). ----
export PRETOKENIZE_MIX="${PRETOKENIZE_MIX:-smollm3_inspired_2k_e05}"

# ---- Optimizer branch (the A/B single variable) ----
OPTIMIZER="${OPTIMIZER:-adam}"
export OPTIMIZER
if [ "$OPTIMIZER" = "muon" ]; then
    # Muon. 2026-07-01: DIVERGED on E05 at LR 0.02 (onset ~step 3k) AND 0.01 (onset ~step 4.5k) —
    # delayed-onset grad_norm explosion. Root cause (confirmed): full-rank orthogonalized updates grow
    # weight spectral norms (esp. in the Q·K + lm_head bilinear couplings) with NO weight decay to curb
    # it, compounded by an over-hot lm_head fallback LR. Mitigations baked in below per Moonlight
    # (arXiv:2502.16982): weight_decay 0.1 (was 0.0) + MUON_ADAMW_LR 2e-4 (was 2e-3). LR 0.01 (0.02
    # diverged faster). MUST re-confirm stability with a constant_with_warmup calibration that sustains
    # peak LR past the onset region. Full root-cause + mitigations: run report
    # e05_muon_divergence_rootcause_20260701.md.
    export LEARNING_RATE="${LEARNING_RATE:-0.01}"
    export MUON_ADAMW_LR="${MUON_ADAMW_LR:-2e-4}"      # was 2e-3 — over-updates lm_head (output logits)
    export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"         # Moonlight: wd=0.1 is Muon's long-horizon stabilizer
else
    # Adam (E05 attempt-3 retune): LR 5e-5, max_grad_norm 0.5 — stable to 0.5 ep (eval_loss 3.83).
    # attempt 2 diverged at step 40k under LR 1e-4 / clip 1.0.
    export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
fi

# ---- Stability (both arms) ----
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
export WARMUP_STEPS="${WARMUP_STEPS:-2000}"
export EVAL_STEPS="${EVAL_STEPS:-4000}"
export SAVE_STEPS="${SAVE_STEPS:-4000}"             # must be a multiple of EVAL_STEPS (HF load_best_model_at_end)
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-3}"
export DDP_TIMEOUT="${DDP_TIMEOUT:-14400}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"   # cut fragmentation OOM at seq 2048
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

# ---- Token-matched A/B: identical effective batch 72 on both servers (→ identical tokens/step) ----
#   Odra   (3 GPU): 8 × 3 × 3 = 72   (8 is the known-stable ceiling at seq 2048; 10/12 OOM'd)
#   Polonez (4 GPU): 6 × 4 × 3 = 72   (6 keeps the same effective batch on 4 GPUs)
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "$NUM_GPUS" -eq 4 ]; then
    export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-6}"
else
    export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
fi

# ---- NUM_EPOCHS: MUST match the live Adam arm so both arms see identical tokens. ----
# The default is the committed E05 scope; the resumed Odra run may be 1 epoch — CONFIRM and
# override (e.g. NUM_EPOCHS=1) so the Muon arm matches exactly.
export NUM_EPOCHS="${NUM_EPOCHS:-0.5}"

echo "=== E05 launch (optimizer=${OPTIMIZER}, ${NUM_GPUS} GPUs, effective batch $((PER_DEVICE_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))) ==="
echo "  mix=${PRETOKENIZE_MIX}  seq=${MAX_SEQ_LENGTH}  K=${DECODER_CONTEXT_WINDOW}"
echo "  LR=${LEARNING_RATE}  max_grad_norm=${MAX_GRAD_NORM}  epochs=${NUM_EPOCHS}  (epochs MUST match the Adam arm)"

# ---- W&B identity preflight (confirm the run is discoverable before a long training run) ----
uv run python - <<'PY'
import os
from training.utils_training import build_perceiver_wandb_identity

def e(name, default):
    return os.environ.get(name, default)

identity = build_perceiver_wandb_identity(
    decoder_type=e("DECODER_TYPE", "causal_ar"),
    objective_variant=e("OBJECTIVE_VARIANT", "prefix_suffix"),
    hidden_size=int(e("HIDDEN_SIZE", "768")),
    num_hidden_layers=int(e("NUM_LAYERS", "6")),
    concept_num=int(e("CONCEPT_NUM", "128")),
    decoder_num_layers=int(e("DECODER_NUM_LAYERS", "4")),
    checkpoint_family="concept_ar",
    pretraining_objective="ar_prefix_suffix_generation",
    use_bixt=True,
    experiment_id=os.environ.get("EXPERIMENT_ID"),
)
print("group:", identity.group)
print("job_type:", identity.job_type)
print("tags:", ", ".join(identity.tags), f"optim-{os.environ.get('OPTIMIZER','adam')}")
PY

# Delegate everything else (defaults, pretokenize, accelerate launch) to the generic launcher.
exec bash "${SCRIPT_DIR}/train_perceiver_denoise_multigpu.sh"
