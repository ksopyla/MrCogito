# Canonical project paths for Odra/Polonez launchers.
# See .cursor/skills/remote-servers/SKILL.md
#
# Single source of truth for the HF_HOME tree. Every launcher and Python script
# MUST consume these variables; do NOT re-derive HF cache paths inline. The three
# HF_HOME subdirs have distinct roles:
#   - HF_DATASETS_CACHE ($HF_HOME/datasets)  : raw load_dataset() cache
#   - DATASETS_TOK_DIR   ($HF_HOME/datasets_tok) : pre-tokenized corpora (pretokenize_mix.py output)
#   - DATASETS_RAW_DIR   ($HF_HOME/datasets_raw) : transient raw parquet/zst downloads
#
# Usage (from any script in scripts/ or parked/scripts/):
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   # shellcheck source=scripts/remote_paths.sh
#   source "${SCRIPT_DIR}/remote_paths.sh"
#   # or from parked/scripts/: source "${SCRIPT_DIR}/../../scripts/remote_paths.sh"

PROJECT_ROOT="${PROJECT_ROOT:-/home/ksopyla/dev/MrCogito}"
if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(pwd)"
fi

# Load .env (HF_TOKEN, WANDB_API_KEY, and optionally HF_HOME) if present. Existing
# env vars take precedence (load_dotenv-style: we only set unset vars), so explicit
# exports in launchers always win. Sourced here so every child process (accelerate,
# training entrypoint, eval scripts) inherits HF_TOKEN without each one needing its
# own load_dotenv() call.
if [ -f "${PROJECT_ROOT}/.env" ]; then
    while IFS='=' read -r key val; do
        # skip comments and blank lines
        case "$key" in
            ''|\#*) continue ;;
        esac
        if [ -z "${!key:-}" ]; then
            export "$key=$val"
        fi
    done < "${PROJECT_ROOT}/.env"
fi

# Only set HF_HOME/HF_DATASETS_CACHE if not already inherited from the environment
# (e.g. when a launcher re-sources this after .env loaded it). `:=` would clobber an
# intentional empty value, so guard explicitly.
if [ -z "${HF_HOME:-}" ]; then
    export HF_HOME="${PROJECT_ROOT}/../hf_home"
fi
if [ -z "${HF_DATASETS_CACHE:-}" ]; then
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
fi

# Pre-tokenized corpora and transient raw downloads live as siblings of datasets/.
# A tokenizer switch (e.g. E10's Gemma) overrides DATASETS_TOK_DIR in its launcher
# BEFORE sourcing this script (or re-exports after); we only set the default here.
if [ -z "${DATASETS_TOK_DIR:-}" ]; then
    export DATASETS_TOK_DIR="${HF_HOME}/datasets_tok"
fi
if [ -z "${DATASETS_RAW_DIR:-}" ]; then
    export DATASETS_RAW_DIR="${HF_HOME}/datasets_raw"
fi

OUTPUT_DIR="${PROJECT_ROOT}/Cache/Training"
LOGGING_DIR="${PROJECT_ROOT}/Cache/logs"

unset TRANSFORMERS_CACHE

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$DATASETS_TOK_DIR" "$DATASETS_RAW_DIR"
