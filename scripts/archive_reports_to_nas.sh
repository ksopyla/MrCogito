#!/usr/bin/env bash
# Archive every report-like folder in every MrCogito checkout on a GPU server to the NAS: evaluation
# reports, eval suites, probe outputs, logs and loose result JSONs. Model checkpoints are skipped
# (Cache/Training goes to /nas/ml_data/mrcogito/checkpoints via experiment-run), so are datasets,
# HF caches, models and tokenizers. Capability-suite and study folders are archived (and put in the
# committed ledger) by scripts/pull_capability_results.sh instead.
#
#   bash scripts/archive_reports_to_nas.sh odra
#   bash scripts/archive_reports_to_nas.sh polonez
#
# Lands at /nas/ml_data/mrcogito/results/reports/<host>/<checkout>/<folder>/ (rsync -rlt, additive,
# never deletes; safe to re-run). Runs entirely on the server; nothing is copied to or from this machine.
set -euo pipefail

HOST="${1:?host (odra|polonez)}"
NAS=/nas/ml_data/mrcogito/results/reports/$HOST

ssh "$HOST" bash -s -- "$NAS" <<'REMOTE'
set -euo pipefail
NAS="$1"
mountpoint -q /nas/ml_data || { echo "NAS not mounted at /nas/ml_data" >&2; exit 1; }
SKIP='^(Training|Models|Tokenizers|Morfessor|HF.*|hf_home|Datasets|datasets.*|wandb|capability|study|jobs)$'
for repo in ~/dev/*/; do
  [ -d "$repo/Cache" ] || continue
  co=$(basename "$repo")
  for item in "$repo"Cache/*; do
    name=$(basename "$item")
    [[ "$name" =~ $SKIP ]] && continue
    if [ -d "$item" ]; then
      mkdir -p "$NAS/$co/$name"
      rsync -rlt --exclude='*.pt' --exclude='*.safetensors' --exclude='*.bin' --exclude='*.ckpt' \
        "$item/" "$NAS/$co/$name/"
    elif [[ "$name" =~ \.(json|csv|md|log|txt)$ ]]; then
      mkdir -p "$NAS/$co"
      rsync -rlt "$item" "$NAS/$co/"
    else
      continue
    fi
    echo "$co/Cache/$name"
  done
done
echo "archive: $(du -sh "$NAS" | cut -f1) at $NAS"
REMOTE
