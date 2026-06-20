#!/usr/bin/env bash
# Bridge Cursor (stdio MCP) to W&B's hosted MCP server.
#
# Why hosted instead of local uvx wandb_mcp_server:
# - wandb-mcp-server 0.3.6 requires wandb>=0.27.1
# - wandb 0.27.1+ routes Public API calls through wandb-core, which currently returns
#   "relogin required" for legacy API keys that still work on wandb 0.27.0 / 0.23.x
# - Hosted MCP authenticates with Bearer token on W&B's side (no local wandb-core)
#
# Credentials: project .env (also loaded by mcp.json envFile).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${ROOT}/.env"

cd "${ROOT}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set. Add it to ${ENV_FILE} (see .env.example)." >&2
  exit 1
fi

# mcp-remote/Cursor mangle args that contain spaces; keep the full header in env.
export WANDB_AUTH_HEADER="Bearer ${WANDB_API_KEY}"

exec npx -y mcp-remote@latest \
  https://mcp.withwandb.com/mcp \
  --header "Authorization:${WANDB_AUTH_HEADER}"
