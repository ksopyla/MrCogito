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

# wandb 0.27+ removed vendored wandb_gql; current wandb-mcp-server HEAD still
# imports it. Pin wandb<0.27.1 until upstream v0.3.6+ lands on PyPI/git tags.
# See: https://github.com/wandb/wandb-mcp-server/pull/93
exec uvx --python 3.12 \
  --with 'wandb[workspaces]>=0.25.1,<0.27.1' \
  --with 'wandb-workspaces>=0.3.9' \
  --from git+https://github.com/wandb/wandb-mcp-server \
  wandb_mcp_server
