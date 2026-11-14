#!/bin/bash
# E38 pipeline check on a home server, before any paid run: a small build of the m21mix_v1 data (every source, the
# constructed rows included), then the production path end to end on one GPU with a 30M model and tiny budgets —
# infra/aws/jobs/e33n_run.sh with E33N_MIX (trunk 4k -> decay 16k -> long 64k, --fast, evaluation inside each
# phase), then the tests of the 64k model in their quick form (long-document loss with bits per byte, RULER and
# core with the SmolLM2 tokenizer).
# Usage: CUDA_VISIBLE_DEVICES=2 bash infra/aws/jobs/e38_pipeline_check.sh [SCALE] [ARM]     (default 0.01, e37a)
# Everything lands in Cache/e38_check/; the last line says PASS or names what failed.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
SCALE="${1:-0.01}"; ARM="${2:-e37a}"
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
C="$PWD/Cache/e38_check"; MIX="m21_check"; D="$C/$MIX"
mkdir -p "$C"
say() { echo "[e38_check $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$C/check.log"; }
fail() { say "FAILED: $*"; exit 1; }

say "1/4 data at scale $SCALE into $D"
if [ ! -f "$D/lookup_64k.done.json" ]; then
  uv run python scripts/build_reasoning_rows.py --out "$D" --scale "$SCALE" > "$C/reasoning.log" 2>&1 \
    || fail "build_reasoning_rows.py (see $C/reasoning.log)"
fi
M21_DATA="$D" bash infra/aws/jobs/m21_data_build.sh "$SCALE" > "$C/mix.log" 2>&1 || fail "m21_data_build.sh (see $C/mix.log)"
uv run python - "$D" <<'PY' || fail "mix.json is missing sources or a phase has a source with no row"
import json, sys
sys.path.insert(0, ".")
from pathlib import Path
from data.bin_token_dataset import load_bin_manifest
from scripts.run_e33n import mix_manifests
d = Path(sys.argv[1])
mix = json.loads((d / "mix.json").read_text())
assert not mix.get("missing") and not mix.get("external_missing"), (mix.get("missing"), mix.get("external_missing"))
paths, _ = mix_manifests(d, d.parent / "manifests", lenient=True)   # a small build: few shards, few 64k documents
for ph, p in paths.items():
    tr, ev, m = load_bin_manifest(str(p))
    print(f"{ph}: {len(tr)} train rows of {m['seq_len']} {tr.rows_per_source}; {len(ev)} eval rows")
PY

say "2/4 training, $ARM at 30M: trunk -> decay -> long"
export E33N_DATA="$C" E33N_RUNS="$C/runs" E33N_MIX="$MIX" E33N_EXPERIMENT=E38check E33N_SKIP_EVAL=1 E33N_LONG=1 \
  E33N_GRAD_CKPT=1 E33N_MICRO_TRUNK=2 E33N_MICRO_DECAY=1 E33N_MICRO_LONG=1 E33N_DECAY_TOKENS=8e5 E33N_LONG_TOKENS=2.8e6 \
  E33N_CMD_EXTRA="--fast --rows 2 --eval_every 20 --save_every 20 --mix_lenient" WANDB_MODE=offline
bash infra/aws/jobs/e33n_run.sh "$ARM" s30 "" 4e5 > "$C/train.log" 2>&1 || fail "e33n_run.sh (see $C/train.log and $C/runs/*/*.log)"
R=$(ls -d "$C"/runs/e33n_s30_"$ARM"_tok4e5* | tail -1)
LONG=$(ls -d "$R"/long/*/final 2>/dev/null | tail -1)
[ -n "$LONG" ] || fail "no 64k model under $R"
for ph in trunk decay long; do
  n=$(grep -c "eval_loss" "$R/$ph.log" || true)
  say "  $ph: $(grep -oE "'eval_loss': [0-9.]+" "$R/$ph.log" | tail -1) ($n evaluations), last train loss $(grep -oE "'loss': [0-9.]+" "$R/$ph.log" | tail -1 | cut -d' ' -f2)"
  [ "$n" -ge 1 ] || fail "$ph ran no evaluation"
done

say "3/4 resume: the trunk again from its newest checkpoint must do nothing new"
bash infra/aws/jobs/e33n_run.sh "$ARM" s30 "" 4e5 > "$C/train_again.log" 2>&1 || fail "the second e33n_run.sh call"
grep -q "trunk done" "$R/events.log" || fail "events.log has no 'trunk done'"

say "4/4 tests of the 64k model, quick form"
# core at batch 8: batch 32 x 2,048 tokens x the vocabulary needs a 12 GB log-softmax, too much next to the model
# on a 24 GB home card (e33n_run.sh's own note); the 48 GB AWS cards keep the default 32
export E33N_TOKENIZER="$D/tokenizer" E33N_LONGDOCS="$D" E37_EVAL_QUICK=1 E33N_CORE_BATCH="${E33N_CORE_BATCH:-8}" \
  E33N_LONGDOC_ARGS="--sources books,arxiv,doab,usgpo,hansard,pdfs,code_repos --eot 0 --tokenizer $D/tokenizer"
bash infra/aws/jobs/e37_evals_long.sh "$LONG" "$R" "$C" > "$C/evals.log" 2>&1 || fail "e37_evals_long.sh (see $C/evals.log, $R/long_evals/*.log)"
uv run python - "$R" <<'PY' || fail "the test files are incomplete"
import json, sys
from pathlib import Path
R = Path(sys.argv[1])
ld = json.loads((R / "eval_longdoc_long.json").read_text())
L = next(iter(ld["lengths"].values()))
bpb = {s: v.get("notes_on", {}).get("bits_per_byte") for s, v in L.items() if isinstance(v, dict) and v.get("rows", 1)}
print("bits per byte at 4k:", bpb)
assert any(isinstance(x, float) for x in bpb.values()), "no bits per byte in the long-document loss"
core = json.loads((R / "eval_core_long.json").read_text())
ruler = json.loads((R / "eval_ruler_long.json").read_text())
print("RULER mean:", ruler.get("mean"), "| core keys:", list(core)[:6])
PY
say "PASS: data, three training phases with evaluation, resume, and the tests ran end to end ($R)"
