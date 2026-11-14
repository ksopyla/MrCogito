#!/bin/bash
# mc: no-sync
# E38 data job (CPU node): build the m21mix_v1 token shards from public Hugging Face datasets, no human step.
#   scripts/build_m21_mix.py + data/mix_recipes/m21_mix_v1.json (SmolLM2 tokenizer, anonymous Hub access).
# Several sources are built at once, one process each (JOBS processes x WORKERS tokenizer threads). On a node each
# source is copied to s3://$MC_BUCKET/data/m21mix_v1/ as soon as it is complete (its shards first, its done.json
# last), and a source whose done.json is already in S3 is skipped: after a spot interruption the same launch
# command continues with the sources that were not finished. At the end mix.json (every source's numbers + the
# supply report per phase) and the tokenizer are written and copied too.
#
#   AWS node:     bash infra/aws/launch.sh m21-data 4 infra/aws/jobs/m21_data_build.sh BRANCH c7i.8xlarge
#   home server:  M21_DATA=/data/m21mix_v1 bash infra/aws/jobs/m21_data_build.sh          (a folder, no S3)
#   smoke:        M21_DATA=/tmp/m21_smoke bash infra/aws/jobs/m21_data_build.sh 0.001      (every target x 0.001)
#
# Arguments: [SCALE] [ONLY]   SCALE multiplies every token target (default 1); ONLY = comma-separated source names.
# Environment (all optional):
#   M21_DATA     output folder; when set there is no S3 unless M21_S3 names one (s3://bucket/prefix)
#   M21_JOBS     sources built at once            (default: cores / WORKERS, at most RAM / 4 GB)
#   M21_WORKERS  tokenizer threads per source     (default 4: one process gains little beyond ~6 threads)
#   M21_RECIPE   recipe file                      (default data/mix_recipes/m21_mix_v1.json)
#   M21_PY       python launcher                  (default: uv, its own small environment, no torch;
#                                                  "uv run python" uses the project's environment)
# Exit code: 0 when every source is built (or was already); non-zero when a source failed or is missing.
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/../../.."

ONE=""
if [ "${1:-}" = "--one" ]; then   # internal: one source, started by the xargs line below
  ONE="$2"
else
  export M21_SCALE="${1:-${M21_SCALE:-1}}" M21_ONLY="${2:-${M21_ONLY:-}}"
fi
[ -f /etc/mrcogito.env ] && source /etc/mrcogito.env
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN   # every source is public: nothing is sent with a token

RECIPE="${M21_RECIPE:-data/mix_recipes/m21_mix_v1.json}"
SCALE="$M21_SCALE"; ONLY="$M21_ONLY"
SUFFIX=""; [ "$SCALE" != 1 ] && SUFFIX="_scale$SCALE"   # a scaled build never lands in the full build's folder
if [ -n "${M21_DATA:-}" ]; then
  D="$M21_DATA"; S3="${M21_S3:-}"
elif [ -n "${MC_DISK:-}" ] && [ -n "${MC_BUCKET:-}" ]; then
  D="$MC_DISK/data/m21mix_v1$SUFFIX"; S3="${M21_S3:-s3://$MC_BUCKET/data/m21mix_v1$SUFFIX}"
else
  echo "[m21] set M21_DATA=DIR (a folder, no S3), or run on a node that has /etc/mrcogito.env"; exit 2
fi
WORKERS="${M21_WORKERS:-4}"
if [ -n "${M21_PY:-}" ]; then
  read -r -a PY <<< "$M21_PY"
else
  PY=(uv run --no-project --python 3.12 --with "numpy>=2,<3" --with "pyarrow>=18" --with "zstandard>=0.23"
      --with "tokenizers>=0.21,<0.23" --with "transformers>=4.57,<5" --with "huggingface_hub>=0.34,<1"
      --with "tiktoken==0.12.0" --with "jinja2>=3.1")
fi
BUILD=("${PY[@]}" python scripts/build_m21_mix.py --recipe "$RECIPE" --out "$D" --scale "$SCALE")
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

one_source() {   # build one source; on success copy it to S3 (shards, then done.json)
  local n="$1" rc
  if [ -n "$S3" ] && [ ! -f "$D/$n.done.json" ] \
     && aws s3 cp --only-show-errors "$S3/$n.done.json" "$D/$n.done.json" 2>/dev/null; then
    echo "[m21] $(now) $n: complete in $S3 — skipped"
    return 0
  fi
  "${BUILD[@]}" --only "$n" --no-mix --workers "$WORKERS" 2>&1 | tee -a "$D/_logs/$n.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" = 0 ] && [ -n "$S3" ]; then
    # file by file, never `aws s3 sync` over the folder: sync walks _raw/ too, and a download file that another
    # source deletes meanwhile makes it exit 2 after copying everything (five sources lost their done.json that
    # way on 2026-11-14 and were repaired by hand)
    local f ok=1
    for f in "$D/${n}"_train_*.bin "$D/${n}"_val_*.bin; do
      [ -f "$f" ] || continue
      aws s3 cp --only-show-errors "$f" "$S3/$(basename "$f")" || ok=0
    done
    if [ "$ok" = 1 ] && aws s3 cp --only-show-errors "$D/$n.done.json" "$S3/$n.done.json"; then
      echo "[m21] $(now) $n: copied to $S3"
    else
      echo "[m21] $(now) $n: !! built, but the copy to $S3 FAILED"; rc=3
    fi
  fi
  [ "$rc" = 0 ] || echo "[m21] $(now) $n: !! FAILED (exit $rc), log $D/_logs/$n.log"
  return "$rc"
}

mkdir -p "$D/_logs"
if [ -n "$ONE" ]; then one_source "$ONE"; exit $?; fi

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo 2>/dev/null)
[ -n "$RAM_GB" ] || RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
JOBS="${M21_JOBS:-$(( NPROC / WORKERS ))}"
[ "$JOBS" -gt $(( RAM_GB / 4 )) ] && JOBS=$(( RAM_GB / 4 ))
[ "$JOBS" -lt 1 ] && JOBS=1
echo "[m21] $(now) recipe $RECIPE, scale $SCALE, folder $D, S3 ${S3:-none}; $NPROC cores, $RAM_GB GB RAM:" \
     "$JOBS sources at once x $WORKERS tokenizer threads"
df -h "$D" | tail -1

# access problems show up here, in a minute, before the long build (the sources that answer are built anyway)
"${BUILD[@]}" --list ${ONLY:+--only "$ONLY"} || echo "[m21] $(now) !! a source is NOT reachable without a token (see above)"

NAMES=$("${BUILD[@]}" --names ${ONLY:+--only "$ONLY"}) || { echo "[m21] the recipe could not be read"; exit 2; }
echo "[m21] $(now) sources, largest first: $(echo $NAMES)"
printf '%s\n' $NAMES | xargs -P "$JOBS" -n 1 bash "$SELF" --one
XRC=$?

if [ -n "$S3" ]; then   # sources built by other scripts (kind "external"): their numbers go into mix.json when in S3
  for n in $("${BUILD[@]}" --names external); do
    [ -f "$D/$n.done.json" ] || aws s3 cp --only-show-errors "$S3/$n.done.json" "$D/$n.done.json" 2>/dev/null || true
  done
fi
"${BUILD[@]}" --mix-only ${ONLY:+--only "$ONLY"} 2>&1 | tee -a "$D/_logs/mix.log"
MRC=${PIPESTATUS[0]}
if [ -n "$S3" ]; then
  aws s3 cp --only-show-errors "$D/mix.json" "$S3/mix.json" || MRC=3
  aws s3 sync --only-show-errors "$D/tokenizer/" "$S3/tokenizer/" || MRC=3
  aws s3 sync --only-show-errors "$D/_logs/" "$S3/_logs/" || true
fi
du -sh "$D" | sed "s/^/[m21] folder size: /"
if [ "$XRC" != 0 ] || [ "$MRC" != 0 ]; then
  echo "[m21] $(now) !! NOT COMPLETE: a source failed or is missing (see the '!!' lines above); the same command continues"
  exit 1
fi
echo "[m21] $(now) m21mix data job done"
