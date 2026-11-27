#!/bin/bash
# E41: the planted-number battery of one model on one home card (scripts/e41_long_needle.py, the model's own read):
# 32 numbers at 64k, 128k and 256k, each text in one forward; then 8 numbers at 512k and 1M, read window by window
# (a 1M row then needs ~6 GB). Results: OUT/single.json, OUT/stream_long.json; progress: OUT/battery.log.
#   bash scripts/e41_needle_battery.sh MODEL_DIR OUT_DIR GPU [EXTRA...]
# EXTRA goes to the needle script as is (e.g. --scan: tokens opened from the best concepts of the whole row,
# whatever the model's config says). NEEDLE_MIX: the built mix whose held-out books make the texts (default: the
# home copy the checkpoint evaluator uses). NEEDLE_LONG=0 skips the 512k and 1M part.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
CK="$1"; OUT="$2"; GPU="$3"; shift 3
MIX="${NEEDLE_MIX:-$HOME/dev/MrCogito-e40/Cache/e40/m21v2_e40}"
mkdir -p "$OUT"
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$OUT/battery.log"; }
say "start: $CK on GPU $GPU, extra: $*"
CUDA_VISIBLE_DEVICES="$GPU" uv run python scripts/e41_long_needle.py --checkpoint "$CK" --mix "$MIX" \
  --lengths 65536,131072,262144 --depths 32 --modes on --top_down "$@" \
  --out "$OUT/single.json" > "$OUT/single.log" 2>&1
say "one forward (64k, 128k, 256k; 32 numbers): exit $?"
if [ "${NEEDLE_LONG:-1}" = 1 ]; then
  CUDA_VISIBLE_DEVICES="$GPU" uv run python scripts/e41_long_needle.py --checkpoint "$CK" --mix "$MIX" \
    --lengths 524288,1048576 --depths 8 --modes on --top_down --stream_from 524288 "$@" \
    --out "$OUT/stream_long.json" > "$OUT/stream_long.log" 2>&1
  say "window by window (512k, 1M; 8 numbers): exit $?"
fi
say "done"
