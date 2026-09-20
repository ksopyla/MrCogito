#!/usr/bin/env bash
# LOCAL helper (run on the laptop, never in a cloud session).
#
# Emits a paste-ready `.env`-format block for the Claude Code cloud environment at
# claude.ai/code -> environment settings -> Environment variables, escaping the
# multi-line values (private key, ssh config, known_hosts) that the form cannot take
# verbatim. The counterpart bootstrap
# (.claude/scripts/cloud-session-ssh-install.sh) un-escapes them on the VM.
#
# Reuses the dedicated key created by .cursor/scripts/cloud-agent-ssh-setup.sh, so
# Cursor Cloud Agents and Claude cloud sessions share one restricted key.
#
# Usage:
#   bash .claude/scripts/cloud-env-print.sh              # config + known_hosts only
#   bash .claude/scripts/cloud-env-print.sh --with-key   # also print the private key
#
# Host/port values come from your local ~/.ssh/config and the gitignored
# remote-servers skill. Nothing here is committed. Clear your scrollback afterwards.
set -euo pipefail

KEY="${CLOUD_SSH_KEY:-$HOME/.ssh/cursor_cloud_agents_ed25519}"
HOST_ALIASES="${SSH_HOST_ALIASES:-odra polonez}"
LOCAL_SSH_CONFIG="${SSH_CONFIG_FILE:-$HOME/.ssh/config}"
WITH_KEY=0

for arg in "$@"; do
  case "$arg" in
    --with-key) WITH_KEY=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Collapse a file into a single .env-safe line with literal \n escapes.
escape_env() { sed 's/\\/\\\\/g' "$1" | awk 'BEGIN{ORS=""} {print (NR>1 ? "\\n" : "") $0}'; }

[[ -f "$LOCAL_SSH_CONFIG" ]] || { echo "No SSH config at $LOCAL_SSH_CONFIG" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Extract only the named Host blocks; never invent HostName/Port here.
awk -v aliases="$HOST_ALIASES" '
  BEGIN { n = split(aliases, a, /[ \t]+/); for (i = 1; i <= n; i++) want[a[i]] = 1 }
  tolower($1) == "host" { keep = 0; for (i = 2; i <= NF; i++) if ($i in want) keep = 1 }
  keep { print }
' "$LOCAL_SSH_CONFIG" > "$TMP/config"

[[ -s "$TMP/config" ]] || { echo "No Host blocks for '$HOST_ALIASES' in $LOCAL_SSH_CONFIG" >&2; exit 1; }

echo "# ---- paste into claude.ai/code -> environment -> Environment variables ----"
echo "# Reminder: these values are readable by anyone who can use the environment."
echo
echo "SSH_CONFIG=\"$(escape_env "$TMP/config")\""
echo

# known_hosts must be precomputed: ssh-keyscan needs port 22, blocked in the cloud.
: > "$TMP/known_hosts"
awk '
  tolower($1) == "hostname" { host = $2 }
  tolower($1) == "port" { port = $2 }
  host != "" { if (port == "") port = 22; print host, port; host = ""; port = "" }
' "$TMP/config" | sort -u | while read -r host port; do
  ssh-keyscan -p "$port" -T 5 "$host" 2>/dev/null >> "$TMP/known_hosts" || true
done

if [[ -s "$TMP/known_hosts" ]]; then
  echo "SSH_KNOWN_HOSTS=\"$(escape_env "$TMP/known_hosts")\""
else
  echo "# SSH_KNOWN_HOSTS: ssh-keyscan returned nothing (host down, or reachable"
  echo "# only over the 443 transport). Run it by hand and paste the lines."
fi
echo

if [[ "$WITH_KEY" -eq 1 ]]; then
  [[ -f "$KEY" ]] || { echo "No key at $KEY — run .cursor/scripts/cloud-agent-ssh-setup.sh first" >&2; exit 1; }
  echo "SSH_PRIVATE_KEY=\"$(escape_env "$KEY")\""
  echo
  echo "# ^ Clear your terminal scrollback now."
else
  echo "# Re-run with --with-key to print SSH_PRIVATE_KEY (from $KEY)."
fi

echo
echo "# Also set: WANDB_API_KEY (the W&B MCP server needs it; there is no .env in a"
echo "# fresh clone), and optionally HF_TOKEN."
echo "# Network access: Custom -> add the HostName(s) above, keep the default list checked."
