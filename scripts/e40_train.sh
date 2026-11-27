#!/bin/bash
# E40 retrofit (docs/experiments_specs/done_failed/E40_depth_mixed_read.md): one arm on one card, from the released
# concept-memory weights, the same rows, order, seed, schedule and tokens for every arm:
#   part 1  rows of 16,384 from m21mix_v2's stage-2 mix  (DECAY_STEPS x 4 rows = 65,536 tokens per step)
#   part 2  rows of 65,536 from m21mix_v2's stage-3 mix  (LONG_STEPS x 1 row), from part 1's final weights
# Rates held constant (no warm-up, no decay): Muon 1e-4 on the matrices and AdamW 6e-6 on the rest (the released
# stage 3's final rates), the depth mixes' own AdamW group at DEPTH_LR (constant, no weight decay).
# Arms: e37a (the control: nothing added), e40_read (mixes on the reading layer's runs), e40_dca2 (2-DCA everywhere).
#
# Usage (Odra, repo root): setsid nohup bash scripts/e40_train.sh ARM GPU [OUT] > /dev/null 2>&1 < /dev/null &
#   events (UTC): OUT/events.log; logs: OUT/{decay,long}.log; weights: OUT/{decay,long}/<run>/final
#   Waits until card GPU runs no compute process (other sessions' jobs included), checking once a minute.
#   STOP_AFTER=decay ends after part 1; ONLY_PART=long runs part 2 alone from INIT (speed checks). TOKENS_SCALE
#   (default 1) scales both parts (pipeline checks only).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ARM="$1"; GPU="$2"; OUT="${3:-Cache/e40/runs/$ARM}"
INIT="${INIT:-$HOME/dev/MrCogito-eng/Cache/e38_board/e37a_m120t_final}"
MIX="${MIX:-$PWD/Cache/e40/m21v2_e40}"   # scripts/e40_data_build.sh
DECAY_STEPS="${DECAY_STEPS:-1430}"; DECAY_EVAL="${DECAY_EVAL:-286}"   # 93.7M tokens; K2 = step 1,144 (75M tokens)
LONG_STEPS="${LONG_STEPS:-920}"; LONG_EVAL="${LONG_EVAL:-230}"        # 60.3M tokens
DEPTH_LR="${DEPTH_LR:-1e-3}"
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled CONCEPT_COMPILE=1 DEPTH_MIX_COMPILE=1
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
EV="$OUT/events.log"
say() { echo "[$(date -u +%FT%TZ)] $ARM: $*" | tee -a "$EV"; }
said=0
while [[ -n $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; do
  (( said )) || { say "GPU $GPU busy with another job, waiting"; said=1; }
  sleep 60
done
EXTRA=()
case "$ARM" in
  e40_*) EXTRA=(--depth_mix_lr "$DEPTH_LR") ;;
  # E41 retrofits (docs/experiments_specs/ahead/E41_tree_concept_model.md): the tree's reasoner is new, so it gets
  # its own AdamW rate (the released recipe's peak); tree_read has no new weights
  tree_read|tree_read_step) ;;
  tree*) EXTRA=(--tree_reasoner_lr "${TREE_LR:-6e-4}") ;;
esac

part() {  # part <phase> <init> <steps> <rows per step> <eval every>
  local ph="$1" init="$2" steps="$3" rows="$4" ev="$5" seq tokens cmd rc
  seq=$([[ $ph == decay ]] && echo 16384 || echo 65536)
  tokens=$(( steps * rows * seq ))
  cmd=$(.venv/bin/python scripts/run_e33n.py cmd --arm "$ARM" --size m120t --phase "$ph" --data "$OUT" --out "$OUT" \
        --init "$init" --mix "$MIX" --mix_lenient --tokens "$tokens" --rows "$rows" --micro 1 --gpus 1 --grad_ckpt \
        --optimizer muon --lr 1e-4 --muon_adamw_lr 6e-6 --constant_lr --eval_every "$ev" --save_every "$ev" \
        --fast --workers "${WORKERS:-2}" "${EXTRA[@]}" 2>> "$OUT/$ph.log") || { say "$ph: the command failed"; return 1; }
  echo "$cmd" > "$OUT/$ph.cmd"
  say "$ph start on GPU $GPU: $steps steps x $rows rows of $seq = $tokens tokens, from $init"
  # its own process group (the pipeline's kill mark stops it by that id); the exit code lands in a file
  rm -f "$OUT/$ph.rc"
  CUDA_VISIBLE_DEVICES="$GPU" setsid bash -c 'echo $$ > "$1"; bash "$2"; echo $? > "$3"' _ "$OUT/$ph.pgid" \
      "$OUT/$ph.cmd" "$OUT/$ph.rc" >> "$OUT/$ph.log" 2>&1 &
  wait $!
  sleep 2
  local pg; pg=$(cat "$OUT/$ph.pgid" 2>/dev/null)
  while [[ -n $pg ]] && kill -0 "$pg" 2>/dev/null; do sleep 30; done
  rc=$(cat "$OUT/$ph.rc" 2>/dev/null || echo 143)
  say "$ph end, exit $rc: $(ls -d "$OUT/$ph"/*/final 2>/dev/null | tail -1)"
  return $rc
}
S=${TOKENS_SCALE:-1}
ds=$(python3 -c "print(max(1, round($DECAY_STEPS * $S)))"); ls_=$(python3 -c "print(max(1, round($LONG_STEPS * $S)))")
de=$(python3 -c "print(max(1, round($DECAY_EVAL * $S)))"); le=$(python3 -c "print(max(1, round($LONG_EVAL * $S)))")
if [[ ${ONLY_PART:-} == long ]]; then   # speed checks: the 64k part alone, from INIT
  part long "$INIT" "$ls_" 1 "$le"; exit $?
fi
part decay "$INIT" "$ds" 4 "$de" || exit 1
[[ ${STOP_AFTER:-} == decay ]] && exit 0
D_FINAL=$(ls -d "$OUT"/decay/*/final 2>/dev/null | tail -1)
[[ -n $D_FINAL ]] || { say "no final weights from part 1"; exit 1; }
part long "$D_FINAL" "$ls_" 1 "$le" || exit 1
say "all done"
