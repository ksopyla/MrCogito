#!/usr/bin/env bash
# Tiny (default) or scaled BAPO capability probe for E18 vs dense vs encoder-decoder.
# See docs/engineering_specs/bapo_capability_ladder.md
set -euo pipefail
SCALE="${SCALE:-tiny}"
OUT="${OUT:-Cache/bapo_${SCALE}}"
ARCH="${ARCH:-dense e18 encdec e18_local}"
# shellcheck disable=SC2086
uv run python verification/bapo_capability_probe.py --scale "$SCALE" --arch $ARCH --out "$OUT" "$@"
uv run python analysis/plot_bapo_capability.py --in_dir "$OUT" --out_dir "$OUT/plots"
echo "bundles: $OUT   plots: $OUT/plots"
