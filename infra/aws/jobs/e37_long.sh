#!/bin/bash
# E37 / E33t stage D+: the 64k training stage at 100M for one full-budget E37 model, then its tests. Approved by the
# author on 2026-11-13 in the 10M-context session ("Yes, run it", for catalogue_reuse_n4 and the LSA port only).
# Recipe = the 30M stage C's, scaled by scripts/run_e33n.py SIZES["m100"]: from the decay's final weights, LongDocs-lite
# rows of 65,536 (manifest seed 2042, trainer seed 1 for both arms, so both see the same rows), 0.1B tokens
# (E33N_LONG_TOKENS=1e8), micro 1 x accumulation 2 x 4 GPUs = 8 rows a step, 191 steps, 6.4e-5 -> 6.4e-6 (1-sqrt),
# each layer recomputed in the backward (E33N_GRAD_CKPT). No --flow (the driver refuses it with --grad_ckpt).
# The run folder is the stage-B one (s3://BUCKET/runs/e37/e33n_m100_<ARM>_tok3.0e9_seed<SEED>/): e33n_run.sh pulls it
# back, finds trunk and decay done, and starts at the 64k phase (checkpoints to S3, so a spot relaunch resumes).
# Tests: e37_evals_long.sh on the 64k final, files with a _long suffix (the stage-B / later-pass files stay).
# Usage (launch.sh JOB words, on a 4-GPU node): infra/aws/jobs/e37_long.sh SEED ARM
set -uo pipefail
SEED="${1:?seed}"; ARM="${2:?arm}"
G=$(nvidia-smi -L | wc -l)
[ "$G" -ge 4 ] || { echo "[e37_long] needs 4 GPUs (recipe: micro 1 x accum 2 x 4 GPUs), found $G"; exit 1; }
TT=3.0e9
export E33N_EXPERIMENT=E37 E33N_SEED="$SEED" E33N_GPUS=4 E33N_CKPT_S3=1 E33N_SKIP_EVAL=1 \
  E33N_LONG=1 E33N_LONG_TOKENS=1e8 E33N_GRAD_CKPT=1
unset E33N_FLOW
source /etc/mrcogito.env
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="e33n_m100_${ARM}_tok${TT}_seed${SEED}"
R="$MC_DISK/runs/$RUN"; S3R="s3://$MC_BUCKET/runs/e37/$RUN"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
if aws s3 ls "$S3R/eval_core_long.json" >/dev/null 2>&1; then
  echo "[e37_long] $ARM already has its 64k-stage tests in S3, nothing to do"; exit 0
fi
aws s3 ls "$S3R/decay/" >/dev/null 2>&1 || { echo "[e37_long] no decay folder at $S3R"; exit 1; }
echo "[e37_long] $(ts) $ARM seed $SEED: 64k stage on 4 GPUs, then tests"
bash "$HERE/e33n_run.sh" "$ARM" m100 "" "$TT" || { echo "[e37_long] $(ts) $ARM training ended with code $?"; exit 1; }
LONG=$(ls -d "$R"/long/*/final 2>/dev/null | tail -1)
[ -n "$LONG" ] || { echo "[e37_long] $ARM has no 64k-stage model"; exit 1; }
echo "[e37_long] $(ts) $ARM tests of $LONG"
bash "$HERE/e37_evals_long.sh" "$LONG" "$R" "$MC_DISK/data" "$S3R" || echo "[e37_long] $(ts) $ARM tests incomplete"
aws s3 sync --only-show-errors "$R/" "$S3R/" || true
echo "[e37_long] $(ts) $ARM done"
