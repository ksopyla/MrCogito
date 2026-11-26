#!/bin/bash
# E41: the test stage of infra/aws/jobs/e38_pipeline_check.sh again, on the check's trained 64k model (after a fix
# to an evaluation path; the three training stages need no rerun).
# Usage (home server, repo root): CUDA_VISIBLE_DEVICES=2 bash scripts/e41_check_tests.sh [ARM=tree] [SIZE=m120t]
# Output: Cache/e38_check/evals_again.log, last line PASS or FAILED.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ARM="${1:-tree}"; SIZE="${2:-m120t}"
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
C="$PWD/Cache/e38_check"; D="$C/m21_check"
R=$(ls -d "$C"/runs/e33n_"$SIZE"_"$ARM"_tok4e5* | tail -1)
LONG=$(ls -d "$R"/long/*/final 2>/dev/null | tail -1)
[ -n "$LONG" ] || { echo "FAILED: no 64k model under $R" | tee "$C/evals_again.log"; exit 1; }
rm -f "$R/long_evals/claim" "$R"/eval_ruler_long.json "$R"/eval_core_long.json
export E33N_TOKENIZER="$D/tokenizer" E33N_LONGDOCS="$D" E37_EVAL_QUICK=1 E33N_CORE_BATCH="${E33N_CORE_BATCH:-8}" \
  E33N_LONGDOC_ARGS="--sources books,arxiv,doab,usgpo,hansard,pdfs,code_repos --eot 0 --tokenizer $D/tokenizer"
if bash infra/aws/jobs/e37_evals_long.sh "$LONG" "$R" "$C" > "$C/evals_again.log" 2>&1 \
    && [ -f "$R/eval_ruler_long.json" ] && [ -f "$R/eval_core_long.json" ]; then
  echo "PASS: the tests ran end to end ($R)" >> "$C/evals_again.log"
else
  echo "FAILED: see $C/evals_again.log and $R/long_evals/*.log" >> "$C/evals_again.log"
fi
tail -1 "$C/evals_again.log"
