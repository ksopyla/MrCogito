#!/usr/bin/env bash
# Launch the W&B MCP server with credentials from the project .env file.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${ROOT}/.env"

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

# wandb-mcp-server 0.3.6+ (on PyPI and git HEAD) requires wandb>=0.27.1 and
# resolved the wandb_gql import issue. No upper bound needed.
exec uvx --python 3.12 \
  --with 'wandb[workspaces]>=0.27.1' \
  --with 'wandb-workspaces>=0.3.9' \
  --from git+https://github.com/wandb/wandb-mcp-server \
  wandb_mcp_server
