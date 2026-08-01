#!/usr/bin/env bash
# One-shot local helper: create a dedicated Cloud Agent SSH key, optionally
# install the public key on the GPU hosts (via local SSH aliases), and print
# Dashboard Secrets steps.
#
# Does NOT upload secrets to Cursor — there is no API for persistent Cloud Agent
# Secrets. Paste values into https://cursor.com/dashboard/cloud-agents
#
# Network identifiers (hostnames, ports, LAN IPs) must come from your local
# ~/.ssh/config and the gitignored remote-servers skill — never from this repo.
#
# Usage:
#   bash .cursor/scripts/cloud-agent-ssh-setup.sh                 # keygen + instructions
#   bash .cursor/scripts/cloud-agent-ssh-setup.sh --install        # ssh-copy-id via local aliases
#   bash .cursor/scripts/cloud-agent-ssh-setup.sh --print-secret   # print private key for Dashboard
#   bash .cursor/scripts/cloud-agent-ssh-setup.sh --print-config   # print SSH_CONFIG from local config
set -euo pipefail

KEY="${CURSOR_CLOUD_SSH_KEY:-$HOME/.ssh/cursor_cloud_agents_ed25519}"
PUB="${KEY}.pub"
# Local SSH aliases only (no HostName/Port here — resolved by ~/.ssh/config).
HOST_ALIASES_DEFAULT="odra polonez"
HOST_ALIASES="${SSH_HOST_ALIASES:-$HOST_ALIASES_DEFAULT}"
LOCAL_SSH_CONFIG="${SSH_CONFIG_FILE:-$HOME/.ssh/config}"
DO_INSTALL=0
DO_PRINT=0
DO_PRINT_CONFIG=0

for arg in "$@"; do
  case "$arg" in
    --install) DO_INSTALL=1 ;;
    --print-secret) DO_PRINT=1 ;;
    --print-config) DO_PRINT_CONFIG=1 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$KEY" ]]; then
  echo "Generating dedicated Cloud Agent key at $KEY"
  ssh-keygen -t ed25519 -f "$KEY" -C "cursor-cloud-agents@mrcogito" -N ""
else
  echo "Reusing existing key: $KEY"
fi

echo
echo "=== Public key (authorize on servers) ==="
cat "$PUB"
echo

if [[ "$DO_INSTALL" -eq 1 ]]; then
  echo "Installing public key via local SSH aliases: $HOST_ALIASES"
  for host in $HOST_ALIASES; do
    ssh-copy-id -i "$PUB" -o IdentitiesOnly=yes "$host"
    echo "  OK: $host"
  done
  echo
  echo "Smoke test:"
  for host in $HOST_ALIASES; do
    ssh -i "$KEY" -o IdentitiesOnly=yes "$host" 'hostname; whoami'
  done
  echo
fi

echo "=== Cursor Dashboard (required — not automatable via CURSOR_API_KEY) ==="
echo "1. Open https://cursor.com/dashboard/cloud-agents → Secrets"
echo "2. Add Runtime Secret SSH_PRIVATE_KEY = contents of $KEY"
echo "3. Add Runtime Secret SSH_CONFIG = Host blocks for your GPU aliases"
echo "   (copy from $LOCAL_SSH_CONFIG; set IdentityFile to ~/.ssh/id_ed25519)."
echo "   Tip: re-run with --print-config to emit a paste-ready block."
echo "4. Optional: SSH_KNOWN_HOSTS, WANDB_API_KEY, HF_TOKEN"
echo "5. Network allowlist: the HostName(s) from that SSH_CONFIG (or Allow all)"
echo "6. Restart / relaunch any Cloud Agent after adding secrets"
echo
echo "Repo install script (.cursor/scripts/cloud-agent-ssh-install.sh) materializes"
echo "these secrets into ~/.ssh inside the Cloud Agent VM."
echo

if [[ "$DO_PRINT_CONFIG" -eq 1 ]]; then
  if [[ ! -f "$LOCAL_SSH_CONFIG" ]]; then
    echo "No local SSH config at $LOCAL_SSH_CONFIG" >&2
    exit 1
  fi
  echo "=== SSH_CONFIG value (paste into Dashboard; edit IdentityFile to ~/.ssh/id_ed25519) ==="
  # Emit only the named Host blocks; do not invent HostName/Port here.
  awk -v aliases="$HOST_ALIASES" '
    BEGIN {
      n = split(aliases, a, /[ \t]+/)
      for (i = 1; i <= n; i++) want[a[i]] = 1
    }
    tolower($1) == "host" {
      keep = 0
      for (i = 2; i <= NF; i++) if ($i in want) keep = 1
    }
    keep { print }
  ' "$LOCAL_SSH_CONFIG"
  echo
fi

if [[ "$DO_PRINT" -eq 1 ]]; then
  echo "=== SSH_PRIVATE_KEY value (paste into Dashboard, then clear terminal scrollback) ==="
  cat "$KEY"
  echo
fi

echo "Done. Do not commit $KEY or any host/port material into the public repo."
