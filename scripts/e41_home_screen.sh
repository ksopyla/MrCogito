#!/bin/bash
# E41: one home screen of the long stage (scripts/e40_train.sh, part `long` alone) with what follows it: the copier
# of its checkpoints to the NAS and the planted-number test of every checkpoint (scripts/e41_ck_watch.sh).
#   bash scripts/e41_home_screen.sh NAME ARM GPU EVAL_GPU
# Trains in Cache/e41/retro/NAME on GPU; tests on EVAL_GPU into Cache/e41/evals/NAME/needle.log (16 numbers at
# 64k, 128k and 256k per checkpoint, the model's own read).
# env, as scripts/e40_train.sh: INIT (the model it continues; default the recall run's final weights, fetched to
# Cache/e41/aws_models/recall_run_final on 2026-11-28; every screen before that date continued the RELEASED
# concept-memory model, ~/dev/MrCogito-eng/Cache/e38_board/e37a_m120t_final, which this default pointed at), MIX (default
# the 128k mix built at home), LONG_STEPS (default 720 = 360 rows of 131,072), LONG_EVAL, MUON_LR, ADAMW_LR, SCHED.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
NAME="$1"; ARM="$2"; GPU="$3"; EGPU="$4"
RUN="Cache/e41/retro/$NAME"; NAS="/nas/ml_data/mrcogito/tmp_e41/$NAME"; EV="Cache/e41/evals/$NAME"
[ -e "$RUN" ] && { echo "$RUN exists: another name, or remove it"; exit 2; }
export INIT="${INIT:-$PWD/Cache/e41/aws_models/recall_run_final}" MIX="${MIX:-$PWD/Cache/e41/m21xl128_home}"
export ONLY_PART=long LONG_STEPS="${LONG_STEPS:-720}" LONG_EVAL="${LONG_EVAL:-60}"
setsid nohup bash scripts/e40_train.sh "$ARM" "$GPU" "$RUN" > /dev/null 2>&1 < /dev/null &
setsid nohup bash scripts/e41_ck_watch.sh copy "$RUN/long" "$NAS" > /dev/null 2>&1 < /dev/null &
NEEDLE_MIX="${NEEDLE_MIX:-$HOME/dev/MrCogito-e40/Cache/e40/m21v2_e40}" NEEDLE_DEPTHS="${NEEDLE_DEPTHS:-16}" \
  NEEDLE_MODES="${NEEDLE_MODES:-on}" setsid nohup bash scripts/e41_ck_watch.sh eval "$NAS" "$EV" "$EGPU" \
  Cache/e41/cm_treeview 65536,131072,262144 > /dev/null 2>&1 < /dev/null &
sleep 25
cat "$RUN/events.log" 2>/dev/null | cut -c1-200
