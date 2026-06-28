#!/bin/bash
# Run on Polonez after goodwrite_ml rsync to /data/mrcogito/goodwrite_ml/ completes.
set -euo pipefail

SRC=/data/mrcogito/goodwrite_ml
DST=/home/ksopyla/dev/goodwrite_ml

if [[ ! -d "$SRC/hf_output" ]]; then
    echo "ERROR: $SRC not populated yet"
    exit 1
fi

du -sh "$SRC" "$DST"

if [[ -L "$DST" ]]; then
    echo "Already symlinked: $DST -> $(readlink "$DST")"
    exit 0
fi

mv "$DST" "${DST}.bak"
ln -s "$SRC" "$DST"
echo "Symlink: $DST -> $SRC"
echo "Verify, then: rm -rf ${DST}.bak"
