#!/usr/bin/env bash
# Copy the final evaluation outputs of one run into the committed registry, so the docs can cite
# them after the server's Cache/ is gone. Cache/ stays the scratch area the evaluation scripts write to.
# Rules: docs/2_Experiments_Registry/results/README.md
#
#   bash scripts/save_eval_results.sh odra  E22_pilot  Cache/eval/e22_a Cache/Evaluation_reports/lm_eval/e22_a.json
#   bash scripts/save_eval_results.sh local E16b      Cache/Evaluation_reports/e16b_ckpt7900_generation_quality.json
#
# <host>  odra | polonez | local
# <name>  folder under results/evaluations/ — the experiment id plus a short run tag (e.g. E22_pilot)
# <path>  files or folders, relative to the repo checkout on that host (~/dev/MrCogito) or absolute / ~/...
# Only text results are kept: .json .csv .md .txt, each under 1 MB. Everything else (checkpoints, plots,
# logs, binaries, big dumps) is dropped with a warning: archive it on the NAS
# (scripts/archive_reports_to_nas.sh) and cite the NAS path. Commit the folder afterwards.
set -euo pipefail

HOST="${1:?host (odra|polonez|local)}"
NAME="${2:?results name, e.g. E22_pilot}"
shift 2
[[ $# -gt 0 ]] || { echo "give at least one result file or folder" >&2; exit 1; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO/docs/2_Experiments_Registry/results/evaluations/$NAME"
mkdir -p "$DEST"
EXCL=(--exclude='*.pt' --exclude='*.bin' --exclude='*.safetensors' --exclude='*.ckpt')

for P in "$@"; do
  if [[ "$HOST" == local ]]; then
    (cd "$REPO" && tar -czf - "${EXCL[@]}" -C "$(dirname "$P")" "$(basename "$P")") | tar -xzf - -C "$DEST"
  else
    ssh "$HOST" "cd ~/dev/MrCogito && tar -czf - ${EXCL[*]} -C \"\$(dirname $P)\" \"\$(basename $P)\"" | tar -xzf - -C "$DEST"
  fi
  echo "saved: $P"
done

while IFS= read -r f; do
  echo "dropped (not a small text result; keep it on the NAS): ${f#$REPO/}" >&2
  rm -f "$f"
done < <(find "$DEST" -type f \( -size +1M -o ! \( -name '*.json' -o -name '*.csv' -o -name '*.md' -o -name '*.txt' \) \))
find "$DEST" -type d -empty -delete

echo "results: ${DEST#$REPO/} ($(du -sh "$DEST" | cut -f1)) — commit it and cite these paths in the report"
