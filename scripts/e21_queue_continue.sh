#!/bin/bash
# After a Wave A Byobu session ends: E28 1k fixed vs scaled from Hub CogitoProbe,
# then E29, then E19 only if a load-bearing S1 exists AND message_slot_loops exists.
# Never 32k / length_group. Prefer load_dataset; local generate only if Hub is down.
#
#   WAIT_SESSION=E27 bash scripts/e21_queue_continue.sh
#   WAIT_SESSION=E26 SKIP_E28=1 bash scripts/e21_queue_continue.sh   # other box; no second E28
#   SKIP_E29=1 is the default: park after bits (Polonez already runs Hub bind).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/remote_paths.sh
source "${SCRIPT_DIR}/remote_paths.sh"
WAIT_SESSION="${WAIT_SESSION:-}"
export PROBE_SOURCE="${PROBE_SOURCE:-hub}"
SKIP_E28="${SKIP_E28:-0}"
# Polonez owns Hub bind. Default skip so this queue parks after bits.
SKIP_E29="${SKIP_E29:-1}"
# shellcheck source=scripts/skip_e29_guard.sh
source "${SCRIPT_DIR}/skip_e29_guard.sh"

if [ -n "$WAIT_SESSION" ]; then
  echo "waiting for byobu session $WAIT_SESSION to exit"
  while byobu has-session -t "$WAIT_SESSION" 2>/dev/null; do
    sleep 60
  done
  echo "session $WAIT_SESSION gone"
fi

s1_pass() {
  local f="$1/s1.txt"
  [ -f "$f" ] && [ "$(tr -d '[:space:]' < "$f")" = "PASS" ]
}

WAVE_A_S1=0
s1_pass "$ROOT/Cache/bapo_s0/e27_hybrid_key_anchors" && WAVE_A_S1=1
s1_pass "$ROOT/Cache/bapo_s0/e26_prefix_ae_slots" && WAVE_A_S1=1

if [ "$PROBE_SOURCE" != "hub" ]; then
  PROBE_ROOT="${PROBE_ROOT:-$ROOT/Cache/concept_probes/full_1k4k}"
  if [ ! -d "$PROBE_ROOT/hf/bits/seq1024/fixed/train" ]; then
    echo "PROBE_SOURCE=$PROBE_SOURCE and local shards missing — generate 1k+4k only (no 32k)"
    BUILDER="${PROBE_BUILDER:-$ROOT/scripts/build_concept_probe_datasets.py}"
    if [ ! -f "$BUILDER" ]; then
      echo "no local generator at $BUILDER; refuse to train 32k and refuse a silent skip" >&2
      exit 1
    fi
    (
      cd "$ROOT"
      uv run python "$BUILDER" \
        --scale full --seed 20260916 --lengths 1024 4096 \
        --families bits bind arith props \
        --out_dir "$PROBE_ROOT" --overwrite --skip_cards
    )
  fi
fi

if s1_pass "$ROOT/Cache/bapo_s0/e27_hybrid_key_anchors"; then
  export PAR_MESSAGE_IDENTITY_SLOTS=True
  export PAR_MESSAGE_GLOBAL_ANCHORS=key_spans
elif s1_pass "$ROOT/Cache/bapo_s0/e26_prefix_ae_slots"; then
  export PAR_MESSAGE_PREFIX_AE=True
  export PAR_MESSAGE_PREFIX_AE_WEIGHT=1.0
  export PAR_MESSAGE_IDENTITY_SLOTS=False
else
  export PAR_MESSAGE_IDENTITY_SLOTS=True
fi

run_e28() {
  local seq="$1" variant="$2" mode="$3" eid="$4"
  SEQ="$seq" VARIANT="$variant" FAMILY=bits \
    COGITO_PROBE_ID=ksopyla/cogito-probe-bits \
    PAR_MODE="$mode" EXPERIMENT_ID="$eid" \
    bash "$SCRIPT_DIR/launch_e28.sh"
}

if [ "$SKIP_E28" != "1" ]; then
  echo "E28 seq=1024 fixed (dense S0 then exclusive) from Hub ksopyla/cogito-probe-bits"
  run_e28 1024 fixed dense E28_s1024_fixed_dense
  run_e28 1024 fixed perceiver E28_s1024_fixed_excl
  run_e28 1024 scaled perceiver E28_s1024_scaled_excl
else
  echo "SKIP_E28=1 — do not start a second E28 on this box"
fi

E28_JSON="$(ls -t "$ROOT/Cache/Evaluation_reports/"*bits_seq1024_fixed_*.json 2>/dev/null | head -1 || true)"
E28_S1=0
if [ -n "${E28_JSON:-}" ]; then
  if python3 - "$E28_JSON" <<'PY'
import json, sys
p = json.loads(open(sys.argv[1]).read())
real = (p.get("overrides") or {}).get("real") or {}
sys.exit(0 if real.get("acc", 0) >= 0.5 and real.get("recovered_bits", 0) > 0 else 1)
PY
  then
    E28_S1=1
  fi
fi

if [ "$SKIP_E28" != "1" ] && [ "$E28_S1" = "1" ]; then
  echo "E28 1k looked load-bearing — 4k fixed vs scaled (still no 32k)"
  run_e28 4096 fixed dense E28_s4096_fixed_dense
  run_e28 4096 fixed perceiver E28_s4096_fixed_excl
  run_e28 4096 scaled perceiver E28_s4096_scaled_excl
elif [ "$SKIP_E28" = "1" ]; then
  echo "SKIP_E28=1 — skip 4k as well"
else
  echo "E28 1k did not look load-bearing — skip 4k/8k/32k"
fi

if [ "$SKIP_E28" = "1" ]; then
  echo "SKIP_E28=1 — E29/E19 stay on the Wave B box (Hub ids, no 32k)"
  exit 0
fi

if e29_skip_requested; then
  echo "SKIP_E29: park after E28 bits (dense S0 → exclusive 1k → 4k if load-bearing)."
  echo "Do not start Hub bind or E19 on this session — Polonez already runs E29."
  exit 0
fi

echo "E29 bind @1024 from Hub ksopyla/cogito-probe-bind"
SEQ=1024 bash "$SCRIPT_DIR/launch_e29.sh"

if [ "$WAVE_A_S1" = "1" ] || [ "$E28_S1" = "1" ]; then
  if python3 -c "from nn.perceiver_ar_lm import PerceiverARConfig; c=PerceiverARConfig(); assert hasattr(c,'message_slot_loops')" 2>/dev/null; then
    echo "E19 slot loops present — launch EXPERIMENT_ID=E19 FAMILY=arith SEQ=1024"
    EXPERIMENT_ID=E19 FAMILY=arith COGITO_PROBE_ID=ksopyla/cogito-probe-arith \
      SEQ=1024 VARIANT=fixed PAR_MODE=perceiver bash "$SCRIPT_DIR/launch_e28.sh"
  else
    echo "E19 gated: message_slot_loops not in PerceiverARConfig — do not launch extra-hop stand-in"
  fi
else
  echo "E19 gated: no load-bearing Wave A/E28 S1"
fi
