# Canonical project paths for Odra/Polonez launchers.
# See .cursor/skills/remote-servers/SKILL.md
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

export HF_HOME="${PROJECT_ROOT}/../hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

OUTPUT_DIR="${PROJECT_ROOT}/Cache/Training"
LOGGING_DIR="${PROJECT_ROOT}/Cache/logs"

unset TRANSFORMERS_CACHE

mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"
