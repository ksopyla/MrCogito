#!/bin/bash
# E41: follow a home screen as it trains. The server that trains copies each new checkpoint to the NAS (`copy`); a
# server with a free card runs the long needle test on every checkpoint that appears there (`eval`).
#   setsid nohup bash scripts/e41_ck_watch.sh copy RUN_LONG_DIR NAS_DIR > /dev/null 2>&1 &
#   setsid nohup bash scripts/e41_ck_watch.sh eval NAS_DIR OUT_DIR GPU TOKENIZER_DIR [LENGTHS] > /dev/null 2>&1 &
# Both end once the run's `final` folder is done. Results: OUT_DIR/needle.log (one line per checkpoint, UTC) and
# OUT_DIR/needle_<checkpoint>.json. The evaluation takes its card through scripts/gpu_when_free.sh (one job at a
# time per card: two evaluators and a queued test once started together on one card and one ran out of memory). A
# checkpoint is done when its line is in needle.log (the JSON is written length by length).
# NEEDLE_MIX (held-out text: a mix folder with *_val_*.bin), NEEDLE_DEPTHS (default 8), NEEDLE_MODES (on,descent16).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1
MODE="$1"; shift
if [ "$MODE" = copy ]; then
  SRC="$1"; DST="$2"; mkdir -p "$DST"
  while true; do
    for d in "$SRC"/*/checkpoint-* "$SRC"/*/final; do
      [ -f "$d/model.safetensors" ] || continue
      n=$(basename "$d"); [ -f "$DST/$n/.ok" ] && continue
      sleep 30   # the save may still be writing
      [ -f "$d/model.safetensors" ] || continue
      mkdir -p "$DST/$n"
      for f in config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do
        [ -f "$d/$f" ] && cp "$d/$f" "$DST/$n/"
      done
      touch "$DST/$n/.ok"
      echo "[$(date -u +%FT%TZ)] copied $n" >> "$DST/copy.log"
    done
    [ -f "$DST/final/.ok" ] && break
    sleep 60
  done
elif [ "$MODE" = eval ]; then
  SRC="$1"; OUT="$2"; GPU="$3"; TOK="$4"; LENGTHS="${5:-65536,131072,262144}"; mkdir -p "$OUT"
  while true; do
    for d in "$SRC"/checkpoint-* "$SRC"/final; do
      [ -f "$d/.ok" ] || continue
      n=$(basename "$d"); grep -q "\] $n: " "$OUT/needle.log" 2>/dev/null && continue
      if [ ! -f "$d/tokenizer.json" ]; then
        for f in tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do
          [ -f "$TOK/$f" ] && cp "$TOK/$f" "$d/"
        done
      fi
      rm -f "$OUT/needle_$n.log"
      bash scripts/gpu_when_free.sh "$GPU" "$OUT/needle_$n.log" uv run python scripts/e41_long_needle.py \
        --checkpoint "$d" --mix "${NEEDLE_MIX:-Cache/e40/m21v2_e40}" --lengths "$LENGTHS" \
        --depths "${NEEDLE_DEPTHS:-8}" --modes "${NEEDLE_MODES:-on,descent16}" --out "$OUT/needle_$n.json"
      D="${NEEDLE_DEPTHS:-8}"
      echo "[$(date -u +%FT%TZ)] $n: $(grep -E "^L=.*depth $D/$D" "$OUT/needle_$n.log" | tr '\n' ' ')" >> "$OUT/needle.log"
    done
    grep -q "\] final: " "$OUT/needle.log" 2>/dev/null && break
    sleep 60
  done
else
  echo "copy | eval"; exit 2
fi
