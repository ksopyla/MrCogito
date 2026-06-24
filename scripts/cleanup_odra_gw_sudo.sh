#!/bin/bash
# Run on Odra with sudo: sudo bash scripts/cleanup_odra_gw_sudo.sh
#
# Removes inactive Odra user accounts and their /home dirs. Keeps ksopyla and pgorecki.
# gw NAS archive must already be at /nas/ml_data/archive_odra/home/gw/ (no rsync/mv here).
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run as root: sudo bash $0"
    exit 1
fi

KEEP_USERS=(ksopyla pgorecki)
NAS_GW=/nas/ml_data/archive_odra/home/gw

remove_user() {
    local u="$1"
    echo "=== $u ==="
    if ! id "$u" &>/dev/null; then
        echo "skip: user $u does not exist"
        return 0
    fi
    userdel -r "$u"
    echo "removed user $u (account + home)"
}

for u in $(getent passwd | awk -F: '$3 >= 1000 && $1 != "nobody" { print $1 }' | sort); do
    keep=false
    for k in "${KEEP_USERS[@]}"; do
        if [[ "$u" == "$k" ]]; then
            keep=true
            break
        fi
    done
    if $keep; then
        echo "=== $u (keep) ==="
        continue
    fi

    if [[ "$u" == "gw" ]]; then
        if [[ ! -d "$NAS_GW" ]] || [[ -z "$(ls -A "$NAS_GW" 2>/dev/null)" ]]; then
            echo "ERROR: refusing to remove gw — NAS archive missing or empty at $NAS_GW"
            exit 1
        fi
        echo "gw NAS archive OK at $NAS_GW"
    fi

    remove_user "$u"
done

echo "Done. Remaining homes:"
ls /home
df -h /home
