#!/usr/bin/env bash
# Sync Evaluation Reports between Polonez (remote) and the local machine.
# One-way download by default; use flags for upload or two-way.
#
# Usage:
#   scripts/sync_evaluation_reports.sh                # download new reports
#   scripts/sync_evaluation_reports.sh --upload       # upload local-only reports
#   scripts/sync_evaluation_reports.sh --two-way      # download then upload
#   scripts/sync_evaluation_reports.sh --dry-run      # show plan only
#   SSH_HOST=odra scripts/sync_evaluation_reports.sh  # sync against odra instead
#
# Prerequisites:
#   - ~/.ssh/config entry for "polonez" / "odra" (see .cursor/rules/remote-servers.mdc)
#   - SSH public key auth set up on the remote
#   - rsync installed locally (preinstalled on macOS)

set -euo pipefail

SSH_HOST="${SSH_HOST:-polonez}"
REMOTE_PATH="${REMOTE_PATH:-/home/ksopyla/dev/MrCogito/Cache/Evaluation_reports}"

# Resolve local path relative to the repo root (parent of this script's dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_PATH="${LOCAL_PATH:-${REPO_ROOT}/Cache/Evaluation_reports}"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
RSYNC_BASE=(rsync -av --progress --human-readable --include='*.csv' --include='*/' --exclude='*' \
            -e "ssh ${SSH_OPTS[*]}")

mode="download"
dry_run=0
for arg in "$@"; do
    case "$arg" in
        --upload)   mode="upload" ;;
        --two-way)  mode="two-way" ;;
        --dry-run)  dry_run=1 ;;
        -h|--help)
            sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $arg" >&2; exit 2 ;;
    esac
done

if (( dry_run )); then
    RSYNC_BASE+=(--dry-run)
fi

mkdir -p "${LOCAL_PATH}"

cyan() { printf '\033[36m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }

cyan "Evaluation Reports Sync"
echo  "  Local : ${LOCAL_PATH}"
echo  "  Remote: ${SSH_HOST}:${REMOTE_PATH}"
echo  "  Mode  : ${mode}$( ((dry_run)) && echo ' (dry-run)' )"

download() {
    green "=== Downloading new reports from ${SSH_HOST} ==="
    "${RSYNC_BASE[@]}" --update "${SSH_HOST}:${REMOTE_PATH}/" "${LOCAL_PATH}/"
}

upload() {
    green "=== Uploading new reports to ${SSH_HOST} ==="
    "${RSYNC_BASE[@]}" --update "${LOCAL_PATH}/" "${SSH_HOST}:${REMOTE_PATH}/"
}

case "$mode" in
    download) download ;;
    upload)   upload ;;
    two-way)  download; upload ;;
esac

green "Done."
