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

PROJECT_ROOT="/home/ksopyla/dev/MrCogito"
if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(pwd)"
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
# Tokenizer-specific trees (e.g. datasets_tok_gemma) are a launcher override of
# DATASETS_TOK_DIR; do not hardcode them here.
export DATASETS_TOK_DIR="${HF_HOME}/datasets_tok"
export DATASETS_RAW_DIR="${HF_HOME}/datasets_raw"

OUTPUT_DIR="${PROJECT_ROOT}/Cache/Training"
LOGGING_DIR="${PROJECT_ROOT}/Cache/logs"

unset TRANSFORMERS_CACHE

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$DATASETS_TOK_DIR" "$DATASETS_RAW_DIR"
