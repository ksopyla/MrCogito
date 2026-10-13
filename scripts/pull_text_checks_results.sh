#!/usr/bin/env bash
# Pull the compact ledger of text-checks runs from a server into the committed results folder, archive the
# run folders (without intermediate checkpoints) to the NAS, and redraw the text capability board.
#
#   bash scripts/pull_text_checks_results.sh polonez r1_smoke r1_screen [--no-archive]
#
# Results only (JSON over ssh) — code reaches servers through git, never this script.
# REMOTE_REPO: the server checkout running the text checks (default ~/dev/MrCogito-text).
set -euo pipefail
HOST="${1:?host (odra|polonez)}"
shift
ARCHIVE=1
RUNS=()
for a in "$@"; do
    if [[ "$a" == "--no-archive" ]]; then ARCHIVE=0; else RUNS+=("$a"); fi
done
[[ ${#RUNS[@]} -gt 0 ]] || { echo "name at least one run folder under Cache/text_checks/" >&2; exit 2; }
REMOTE_REPO="${REMOTE_REPO:-~/dev/MrCogito-text}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LEDGER="$REPO/docs/2_Experiments_Registry/results/capability/text"
NAS=/nas/ml_data/mrcogito/results/text_checks
mkdir -p "$LEDGER"
for RUN in "${RUNS[@]}"; do
    TMP="$(mktemp)"
    ssh "$HOST" "cd $REMOTE_REPO && .venv/bin/python analysis/text_checks_ledger.py --in_dir Cache/text_checks/$RUN --host $HOST" > "$TMP"
    python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$TMP" || { echo "collector output for $RUN is not JSON" >&2; rm -f "$TMP"; exit 1; }
    SIZE=$(wc -c < "$TMP")
    if (( SIZE > 2097152 )); then
        echo "ledger for $RUN would be $((SIZE / 1024)) KB (> 2 MB): not written; slim the collector" >&2; rm -f "$TMP"; exit 1
    fi
    mv "$TMP" "$LEDGER/$RUN.$HOST.json"
    echo "ledger: ${LEDGER#$REPO/}/$RUN.$HOST.json"
    if [[ "$ARCHIVE" == 1 ]]; then
        ssh "$HOST" "mkdir -p $NAS/$RUN.$HOST && rsync -rlt --exclude 'checkpoint-*' $REMOTE_REPO/Cache/text_checks/$RUN/ $NAS/$RUN.$HOST/ && du -sh $NAS/$RUN.$HOST | cut -f1" \
            && echo "archive: $HOST:$NAS/$RUN.$HOST/" || echo "archive skipped (NAS not reachable from $HOST)"
    fi
done
cd "$REPO" && uv run python analysis/text_board.py
