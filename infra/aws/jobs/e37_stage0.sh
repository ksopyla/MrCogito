#!/bin/bash
# E37 stage 0 on one GPU (home server or AWS node): the GPU tests (flex = reference concept read), then a speed and
# memory check of the 100M arms at 4k (trunk) and 16k (decay) against the LSA arm and dense.
# Usage: CUDA_VISIBLE_DEVICES=N bash infra/aws/jobs/e37_stage0.sh DATA OUT
#   E37_SPEED_ITEMS overrides the speed items (size:arm:phase:tokens:micro, space-separated; on a 24 GB card the
#   100M decay at 16k may not fit, so measure the decay at 30M there).
# Results: OUT/pytest.log, OUT/speed/summary.tsv
set -uo pipefail
DATA="$1"; OUT="$2"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUT"
echo "== $(date -u +%H:%M:%S) GPU tests"
uv run pytest -q tests/test_concept_memory_cuda.py tests/test_concept_memory.py -s > "$OUT/pytest.log" 2>&1
echo "pytest rc $?" >> "$OUT/pytest.log"
tail -5 "$OUT/pytest.log"
echo "== $(date -u +%H:%M:%S) speed / memory"
ITEMS="${E37_SPEED_ITEMS:-m100:lsa:trunk:2e6:2 m100:e37a:trunk:2e6:2 m100:e37b:trunk:2e6:2 m100:dense:trunk:2e6:2 \
m100:lsa:decay:1e6:1 m100:e37a:decay:1e6:1 m100:e37b:decay:1e6:1 m100:dense:decay:1e6:1}"
# shellcheck disable=SC2086
bash infra/aws/jobs/speed_check.sh "$DATA" "$OUT/speed" $ITEMS
