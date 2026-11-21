#!/bin/bash
# E33n / E33p pipeline check: every arm (E33N_ARMS to choose), trunk then decay from the trunk's weights, a few
# steps at the tiny size. Pass --flow to exercise the E33p instrumentation.
# Local:  bash infra/aws/jobs/pipeline_check.sh DATA_DIR OUT_DIR --local
# Node:   bash infra/aws/jobs/pipeline_check.sh /opt/dlami/nvme/data /opt/dlami/nvme/runs/check
set -uo pipefail
DATA="$1"; OUT="$2"; shift 2
EXTRA=("$@")
STATUS=0
for arm in ${E33N_ARMS:-window dense e33 e33n notes_open pages_open lsa}; do
  echo "== $arm trunk $(date -u +%H:%M:%S)"
  CMD=$(uv run python scripts/run_e33n.py cmd --arm "$arm" --size tiny --phase trunk --data "$DATA" \
        --out "$OUT/$arm" --tokens 6e4 "${EXTRA[@]}")
  if ! eval "$CMD" > "$OUT/$arm.trunk.log" 2>&1; then echo "FAIL $arm trunk"; tail -30 "$OUT/$arm.trunk.log"; STATUS=1; continue; fi
  grep -E "'loss'|train_loss|real_tokens_per_second" "$OUT/$arm.trunk.log" | tail -3
  FINAL=$(ls -d "$OUT/$arm"/trunk/*/final | tail -1)
  echo "== $arm decay $(date -u +%H:%M:%S)"
  CMD=$(uv run python scripts/run_e33n.py cmd --arm "$arm" --size tiny --phase decay --data "$DATA" \
        --out "$OUT/$arm" --tokens 3e4 --init "$FINAL" "${EXTRA[@]}")
  if ! eval "$CMD" > "$OUT/$arm.decay.log" 2>&1; then echo "FAIL $arm decay"; tail -30 "$OUT/$arm.decay.log"; STATUS=1; continue; fi
  grep -E "'loss'|train_loss" "$OUT/$arm.decay.log" | tail -2
done
exit $STATUS
