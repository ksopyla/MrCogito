#!/bin/bash
# Run on Polonez with sudo: sudo bash scripts/cleanup_polonez_users_sudo.sh
#
# Deletes inactive user accounts and their NVMe homes. No NAS archive.
# Also removes mistaken partial copies under /data/mrcogito/home_archive/.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run as root: sudo bash $0"
    exit 1
fi

USERS=(jlewalski mwrobel bmielczarek kropiak sidziniak kfuchsig)
STALE_SDB_ARCHIVE=/data/mrcogito/home_archive

for u in "${USERS[@]}"; do
    echo "=== $u ==="
    if ! id "$u" &>/dev/null; then
        echo "skip: user $u does not exist"
        continue
    fi
    userdel -r "$u"
    echo "removed user $u (account + home)"
done

if [[ -d "$STALE_SDB_ARCHIVE" ]]; then
    rm -rf "$STALE_SDB_ARCHIVE"
    echo "removed stale sdb archive: $STALE_SDB_ARCHIVE"
fi

echo "Done. Review with: df -h /home /data/mrcogito ; ls /data/mrcogito/"
