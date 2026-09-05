#!/usr/bin/env bash
# Cloud Agent install helper: materialize SSH secrets into ~/.ssh.
# Safe to re-run (idempotent).
#
# Required Runtime Secrets (Dashboard → Cloud Agents → Secrets):
#   SSH_PRIVATE_KEY  — OpenSSH private key material
#   SSH_CONFIG       — full ~/.ssh/config body (aliases, HostName, User, Port, …)
# Optional:
#   SSH_KNOWN_HOSTS  — known_hosts lines (skips ssh-keyscan when set)
#
# This script must not hardcode hostnames, ports, users, or LAN addresses —
# those stay out of the public repository.
set -euo pipefail

SSH_DIR="${HOME}/.ssh"
KEY_PATH="${SSH_DIR}/id_ed25519"
CONFIG_PATH="${SSH_DIR}/config"
KNOWN_HOSTS="${SSH_DIR}/known_hosts"

mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

if [[ -z "${SSH_PRIVATE_KEY:-}" || -z "${SSH_CONFIG:-}" ]]; then
  echo "cloud-agent-ssh-install: SSH_PRIVATE_KEY and/or SSH_CONFIG unset — skipping SSH materialize."
  echo "Add both as Runtime Secrets at https://cursor.com/dashboard/cloud-agents"
  echo "(host/port/user values come from secrets, not from this repo)."
  exit 0
fi

# Write key (handle secrets that may contain literal \n)
if printf '%s' "$SSH_PRIVATE_KEY" | grep -q '\\n'; then
  printf '%s' "$SSH_PRIVATE_KEY" | sed 's/\\n/\n/g' > "$KEY_PATH"
else
  printf '%s\n' "$SSH_PRIVATE_KEY" > "$KEY_PATH"
fi
chmod 600 "$KEY_PATH"

if printf '%s' "$SSH_CONFIG" | grep -q '\\n'; then
  printf '%s' "$SSH_CONFIG" | sed 's/\\n/\n/g' > "$CONFIG_PATH"
else
  printf '%s\n' "$SSH_CONFIG" > "$CONFIG_PATH"
fi
chmod 600 "$CONFIG_PATH"

# Ensure IdentityFile points at the key we just wrote when the secret omits it.
if ! grep -q 'IdentityFile' "$CONFIG_PATH"; then
  # Append to each Host block is fragile; document that secrets should include IdentityFile.
  echo "cloud-agent-ssh-install: warning — SSH_CONFIG has no IdentityFile; defaulting ssh to ${KEY_PATH}"
fi

if [[ -n "${SSH_KNOWN_HOSTS:-}" ]]; then
  if printf '%s' "$SSH_KNOWN_HOSTS" | grep -q '\\n'; then
    printf '%s' "$SSH_KNOWN_HOSTS" | sed 's/\\n/\n/g' > "$KNOWN_HOSTS"
  else
    printf '%s\n' "$SSH_KNOWN_HOSTS" > "$KNOWN_HOSTS"
  fi
  chmod 644 "$KNOWN_HOSTS"
else
  # Best-effort trust from HostName/Port in config; ignore failures (DNS/egress).
  if command -v ssh-keyscan >/dev/null 2>&1 && command -v awk >/dev/null 2>&1; then
    awk '
      tolower($1)=="host" { next }
      tolower($1)=="hostname" { host=$2 }
      tolower($1)=="port" { port=$2 }
      host != "" && port != "" {
        cmd = sprintf("ssh-keyscan -p %s -T 5 %s 2>/dev/null", port, host)
        system(cmd)
        host=""; port=""
      }
    ' "$CONFIG_PATH" >> "$KNOWN_HOSTS" 2>/dev/null || true
    [[ -f "$KNOWN_HOSTS" ]] && chmod 644 "$KNOWN_HOSTS"
  fi
fi

echo "cloud-agent-ssh-install: wrote ${KEY_PATH} and ${CONFIG_PATH}"
