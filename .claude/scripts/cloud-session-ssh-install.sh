#!/usr/bin/env bash
# Materialize SSH credentials for a Claude Code **cloud session** so the agent can
# reach the GPU servers (odra / polonez) with plain `ssh odra` / `ssh polonez`.
#
# Counterpart of .cursor/scripts/cloud-agent-ssh-install.sh (Cursor Cloud Agents).
# Wired in as a SessionStart hook via .claude/settings.json, so it runs on every
# cloud session including resumed ones (cloud VMs start with an empty ~/.ssh).
#
# Environment variables (set on the cloud environment at claude.ai/code):
#   SSH_PRIVATE_KEY  — OpenSSH private key material (dedicated key, see SECURITY)
#   SSH_CONFIG       — full ~/.ssh/config body (Host blocks: HostName, User, Port, …)
#   SSH_KNOWN_HOSTS  — known_hosts lines. STRONGLY recommended in the cloud:
#                      ssh-keyscan needs port 22, which the sandbox blocks.
#
# SECURITY: cloud-environment variables are readable by anyone who can use the
# environment, and by the agent itself. Use a DEDICATED key that is restricted on
# the server side (its own user, or command="…" / restrict in authorized_keys) —
# never the author's personal key.
#
# This script must not hardcode hostnames, ports, users, or LAN addresses; those
# stay out of the public repository (same rule as the Cursor side).
#
# Safe to re-run. Never overwrites a ~/.ssh/config this script did not write.
set -euo pipefail

MARKER="# managed-by: MrCogito .claude/scripts/cloud-session-ssh-install.sh"
SSH_DIR="${HOME}/.ssh"
KEY_PATH="${SSH_DIR}/id_ed25519"
CONFIG_PATH="${SSH_DIR}/config"
KNOWN_HOSTS="${SSH_DIR}/known_hosts"

log() { echo "cloud-session-ssh: $*"; }

# --- Only act inside a cloud session -----------------------------------------
# A local macOS session already has a real ~/.ssh; touching it would be wrong.
# CCR_* vars and /root/.ccr are set by the cloud runtime. CLOUD_SSH_BOOTSTRAP=1
# forces the bootstrap on for testing.
if [[ "${CLOUD_SSH_BOOTSTRAP:-0}" != "1" ]]; then
  if [[ -z "${CCR_AGENT_PROXY_ENABLED:-}" && ! -d /root/.ccr ]]; then
    exit 0  # local session — nothing to do, stay silent
  fi
fi

if [[ -z "${SSH_PRIVATE_KEY:-}" || -z "${SSH_CONFIG:-}" ]]; then
  log "SSH_PRIVATE_KEY and/or SSH_CONFIG unset — skipping SSH setup."
  log "Set both as environment variables on the cloud environment at claude.ai/code."
  log "Remote GPU work is unavailable in this session; see AGENTS.md."
  exit 0
fi

# --- Ensure an ssh client exists ---------------------------------------------
# Cloud VMs ship without openssh-client. Prefer installing it in the environment's
# *setup script* (cached, runs as root before Claude starts); this is the fallback.
if ! command -v ssh >/dev/null 2>&1; then
  log "ssh client not found — installing openssh-client"
  if [[ "$(id -u)" -eq 0 ]]; then
    (apt-get update -qq && apt-get install -y -qq openssh-client) >/dev/null 2>&1 \
      || log "WARNING: openssh-client install failed; add it to the environment setup script"
  else
    log "WARNING: not root, cannot install openssh-client; add it to the environment setup script"
  fi
fi

mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

# Secrets pasted through web forms often arrive with literal \n instead of newlines.
unescape_to() {
  local value="$1" dest="$2"
  if printf '%s' "$value" | grep -q '\\n'; then
    printf '%s' "$value" | sed 's/\\n/\n/g' > "$dest"
  else
    printf '%s' "$value" > "$dest"
  fi
  # Guarantee a trailing newline so appended/concatenated files stay well-formed.
  [[ -s "$dest" ]] && [[ "$(tail -c1 "$dest" | wc -l)" -eq 0 ]] && echo >> "$dest"
  return 0
}

# --- Private key --------------------------------------------------------------
unescape_to "$SSH_PRIVATE_KEY" "$KEY_PATH"
chmod 600 "$KEY_PATH"

# --- Config -------------------------------------------------------------------
# Refuse to clobber a config we do not own (protects a real ~/.ssh if the cloud
# guard above is ever bypassed).
if [[ -f "$CONFIG_PATH" ]] && ! grep -qF "$MARKER" "$CONFIG_PATH"; then
  CONFIG_PATH="${SSH_DIR}/config.mrcogito"
  log "WARNING: ~/.ssh/config exists and was not written by this script."
  log "Wrote the injected config to ${CONFIG_PATH} instead; use: ssh -F ${CONFIG_PATH} odra"
fi
{
  echo "$MARKER"
  echo "# Generated at session start. Do not edit; edit the SSH_CONFIG env var instead."
} > "$CONFIG_PATH"
unescape_to "$SSH_CONFIG" "${CONFIG_PATH}.body"
cat "${CONFIG_PATH}.body" >> "$CONFIG_PATH"
rm -f "${CONFIG_PATH}.body"
chmod 600 "$CONFIG_PATH"

if ! grep -qi 'IdentityFile' "$CONFIG_PATH"; then
  log "WARNING: SSH_CONFIG has no IdentityFile; ssh will fall back to ${KEY_PATH}"
fi

# --- known_hosts --------------------------------------------------------------
# No ssh-keyscan fallback here (unlike the Cursor script): it dials port 22, which
# the cloud sandbox blocks, so it would only hang and then silently produce an
# empty file. Provide SSH_KNOWN_HOSTS explicitly.
if [[ -n "${SSH_KNOWN_HOSTS:-}" ]]; then
  unescape_to "$SSH_KNOWN_HOSTS" "$KNOWN_HOSTS"
  chmod 644 "$KNOWN_HOSTS"
else
  log "WARNING: SSH_KNOWN_HOSTS unset — host verification will fail on first connect."
  log "Generate it locally with: ssh-keyscan -p <port> <hostname>"
fi

log "wrote ${KEY_PATH} and ${CONFIG_PATH}"
log "Transport reminder: port 22 is blocked in cloud sessions. The Host blocks in"
log "SSH_CONFIG must reach the servers over an allowlisted hostname on 443 — see AGENTS.md."
