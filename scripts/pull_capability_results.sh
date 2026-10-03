#!/usr/bin/env bash
# Pull one capability-suite or study folder from a GPU server into the committed ledger, and
# archive the full raw folder (logs, rung JSONs, ladders, checkpoints) to the NAS.
# Process: docs/engineering_specs/capability_checks.md (skill `capability-checks`).
#
#   bash scripts/pull_capability_results.sh odra  ~/dev/MrCogito-e31/Cache/study/e30_vs_e31
#   bash scripts/pull_capability_results.sh odra  ~/dev/MrCogito/Cache/capability/e33a_full_30m
#   bash scripts/pull_capability_results.sh polonez ~/dev/MrCogito/Cache/capability/li_full_30m --no-archive
#
# Writes docs/2_Experiments_Registry/results/capability/<suite|study>/<folder>.<host>.json (commit it):
# one compact text JSON (refused above 1 MB). Checkpoints, logs and raw files go only to the NAS.
# The collector is streamed over ssh stdin (stdlib only), so the server checkout needs no update.
# Archiving copies results server → NAS on the server itself (rsync -rlt: no owner/group on the NFS share; additive, never deletes).
set -euo pipefail

HOST="${1:?host (odra|polonez)}"
ROOT="${2:?remote result folder (absolute or ~/...)}"
ARCHIVE=1
[[ "${3:-}" == "--no-archive" ]] && ARCHIVE=0

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LEDGER="$REPO/docs/2_Experiments_Registry/results/capability"
NAS=/nas/ml_data/mrcogito/results
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

ssh "$HOST" "python3 - collect $ROOT --host $HOST" < "$REPO/analysis/capability_ledger.py" > "$TMP"
KIND="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["kind"])' "$TMP")"
NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$TMP")"
SUB=$([[ "$KIND" == suite ]] && echo capability || echo study)
DEST="$LEDGER/$KIND/$NAME.$HOST.json"
# The repo gets compact text only: refuse anything that is not small, valid JSON.
python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$TMP" || { echo "collector output is not JSON" >&2; exit 1; }
SIZE=$(wc -c < "$TMP")
if (( SIZE > 1048576 )); then
  echo "ledger file would be $((SIZE / 1024)) KB (> 1 MB): not written; slim the collector instead" >&2; exit 1
fi
mkdir -p "$(dirname "$DEST")"
mv "$TMP" "$DEST"
echo "ledger: ${DEST#$REPO/}"

if [[ "$ARCHIVE" == 1 ]]; then
  ssh "$HOST" "test -d $NAS || mkdir -p $NAS; mkdir -p $NAS/$SUB/$NAME.$HOST && \
    rsync -rlt $ROOT/ $NAS/$SUB/$NAME.$HOST/ && du -sh $NAS/$SUB/$NAME.$HOST | cut -f1"
  echo "archive: $HOST:$NAS/$SUB/$NAME.$HOST/"
fi
