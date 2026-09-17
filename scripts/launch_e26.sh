#!/bin/bash
# E26 — weak prefix AE write objective on exclusive r=16 slots (Wave A).
# Spec: docs/experiments_specs/ahead/E26_prefix_ae_exclusive_slots.md
# Do NOT pass --message_identity_slots (u/delta move under AE only).
# Usage: GPU=0 bash scripts/launch_e26.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/remote_paths.sh
source "${SCRIPT_DIR}/remote_paths.sh"
NAME="${NAME:-e26_prefix_ae_slots}"
GPU="${GPU:-0}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-E26}"
set +e
bash "$SCRIPT_DIR/e24_bapo_hunt.sh" "$NAME" "$GPU" \
  --scale bridge --seq_len 512 --recipe recall_single \
  --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --message_ratio 16 --message_slots_inplace \
  --message_prefix_ae --message_prefix_ae_weight 1.0 \
  --steps 800 --k1_mult 4 \
  --experiment_id E26 --wandb \
  "$@"
hunt_ec=$?
set -e
WT="${E24_WORKTREE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# Probe exits 2 when a rung is uncalibrated; MATCH JSON + RankMe still need scoring.
if [ -d "$WT/Cache/bapo_s0/$NAME" ]; then
  bash "$SCRIPT_DIR/e21_queue_eval.sh" "$WT/Cache/bapo_s0/$NAME" "$GPU" || true
fi
exit "$hunt_ec"
