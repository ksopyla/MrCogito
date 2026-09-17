#!/bin/bash
# After a Wave A probe JSON exists: print MATCH / ablation / RankMe, and if S1
# passes run the INDEX don't-regress (far_copy seq=1024 r=16). Does not launch 32k.
# Usage: bash scripts/e21_queue_eval.sh Cache/bapo_s0/e27_hybrid_key_anchors [GPU]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
JSON_DIR="${1:?usage: $0 <bapo_s0 dir> [GPU]}"
GPU="${2:-0}"
cd "$ROOT"
python3 - "$JSON_DIR" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
files = sorted(p for p in d.glob("*.json") if p.name != "wandb_run.txt")
if not files:
    print(f"no json in {d}", file=sys.stderr)
    sys.exit(2)
task_files = [p for p in files if p.name != "summary.json"]
prefer = [p for p in task_files if "recall" in p.name]
bundle_path = prefer[-1] if prefer else (task_files[-1] if task_files else files[-1])
bundle = json.loads(bundle_path.read_text())
rungs = bundle.get("rungs") or [bundle]
s1 = False
for b in rungs:
    rec = b.get("recovered_bits")
    results = b.get("results") if isinstance(b.get("results"), dict) else {}
    if not rec and results:
        rec = {
            a: (r.get("info") or {}).get("recovered_bits")
            for a, r in results.items()
            if isinstance(r, dict)
        }
    rec = rec or {}
    print("task", b.get("task") or b.get("recipe"))
    print("recovered_bits", rec)
    e18 = rec.get("e18") or 0
    e21 = rec.get("e21") or 0
    dense = rec.get("dense") or 0
    gate = 0.75 * e18 if e18 else 0.75 * dense
    s1 = bool(e21 >= gate and e18 > 0)
    print(f"S1_gate={gate:.2f} S1={'PASS' if s1 else 'FAIL'}")
    e21r = results.get("e21") or {}
    print("ablations", e21r.get("channel_ablations"))
    print("slot_geometry", e21r.get("slot_geometry"))
    print("prefix_ae", e21r.get("prefix_ae"))
(d / "s1.txt").write_text("PASS" if s1 else "FAIL")
print("s1_file", d / "s1.txt")
PY
S1=$(tr -d '[:space:]' < "$JSON_DIR/s1.txt")
NAME="$(basename "$JSON_DIR")_index"
if [ "$S1" = "PASS" ]; then
  echo "S1 passed — INDEX don't-regress far_copy seq=1024 r=16"
  FLAGS=()
  case "$JSON_DIR" in
    *e27*) FLAGS+=(--message_identity_slots --message_global_anchors key_spans) ;;
    *e26*) FLAGS+=(--message_prefix_ae --message_prefix_ae_weight 1.0) ;;
    *) FLAGS+=(--message_identity_slots) ;;
  esac
  bash "$SCRIPT_DIR/e24_bapo_hunt.sh" "$NAME" "$GPU" \
    --scale bridge_1k --seq_len 1024 --recipe far_copy \
    --arch dense e18 e21 e18_local \
    --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
    --message_ratio 16 --message_slots_inplace \
    "${FLAGS[@]}" \
    --steps 800 --k1_mult 4 \
    --experiment_id "${EXPERIMENT_ID:-E21}_index" --wandb
else
  echo "S1 miss — skip INDEX. Do not 4k/8k/32k."
fi
