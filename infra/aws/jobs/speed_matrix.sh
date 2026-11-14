#!/bin/bash
# Training-speed matrix on one GPU: the same short run (real trainer, real shards) with the old settings and with
# each speed switch of scripts/run_e33n.py, one line per variant in OUT/summary.tsv (tokens/s, peak memory).
#   base     — as every 100M run to 2026-11-14 (attention kernel compiled, chunked loss)
#   liger    — Liger's fused linear + cross-entropy
#   static   — the whole model through torch.compile(dynamic=False)
#   both     — static + liger
# Usage: CUDA_VISIBLE_DEVICES=0 bash infra/aws/jobs/speed_matrix.sh DATA OUT ITEM [VARIANT ...]
#   ITEM = size:arm:phase:rows:micro:steps   e.g. m100:e37a:trunk:8:2:100 (8 rows per optimizer step, 100 steps)
# The loss of the last logged step is printed per variant: same seed and rows, so the variants must agree closely.
set -uo pipefail
DATA="$1"; OUT="$2"; ITEM="$3"; shift 3
VARIANTS=("$@"); [ ${#VARIANTS[@]} -gt 0 ] || VARIANTS=(base liger static both)
IFS=: read -r size arm phase rows micro steps <<< "$ITEM"
export PATH="$HOME/.local/bin:$PATH"
seq=$([ "$phase" = trunk ] && echo 4096 || { [ "$phase" = decay ] && echo 16384 || echo 65536; })
tokens=$(( rows * seq * steps ))
HERE="$(cd "$(dirname "$0")" && pwd)"
for v in "${VARIANTS[@]}"; do
  case "$v" in
    base)   extra="" ;;
    liger)  extra="--liger" ;;
    static) extra="--compile static" ;;
    both)   extra="--compile static --liger" ;;
    dynamic) extra="--compile dynamic" ;;
    *) echo "unknown variant $v"; exit 2 ;;
  esac
  SPEED_LABEL="$v" SPEED_EXTRA="--rows $rows $extra ${SPEED_MORE:-}" \
    bash "$HERE/speed_check.sh" "$DATA" "$OUT" "$size:$arm:$phase:$tokens:$micro" > "$OUT.$arm.$phase.$v.out" 2>&1
  log="$OUT/${size}_${arm}_${phase}_m${micro}_$v.log"
  echo "$v: last losses $(grep -oE "'loss': [0-9.]+" "$log" | tail -3 | awk '{printf "%s ", $2}')"
  rm -rf "$OUT/${size}_${arm}_$v"/*/*/checkpoint-* 2>/dev/null   # speed runs keep no weights
done
cat "$OUT/summary.tsv"
