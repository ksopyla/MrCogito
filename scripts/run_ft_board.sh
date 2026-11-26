#!/bin/bash
# Board-style tests of one checkpoint that need no long-document data, on three cards at once (root-cause work,
# 2026-11-23): card A the exam on held-out list rows, then the first-token probe at 16k, 8k and 4k; card B RULER
# (13 tasks, 100 prompts, the board's settings) at 4k and 8k; card C RULER at 16k.
# Usage: bash scripts/run_ft_board.sh NAME CHECKPOINT LIST_VAL_SHARD CARD_A CARD_B CARD_C
# Results: Cache/rows_exam/NAME_list.json, Cache/use_probe/NAME_start_*k.json, Cache/list_ft_evals/NAME/ruler/*.json
# (the layout scripts/compare_release_evals.py reads); one line per event in Cache/list_ft_evals/NAME/events.log.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
NAME="$1"; CK="$2"; LV="$3"; GA="$4"; GB="$5"; GC="$6"
export PATH="$HOME/.local/bin:$PATH"
O="Cache/list_ft_evals/$NAME"
mkdir -p "$O/ruler" Cache/rows_exam Cache/use_probe
say() { echo "[ft_board $NAME $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$O/events.log"; }
ruler() {  # ruler CARD TAG LENGTHS BATCH
  CUDA_VISIBLE_DEVICES=$1 uv run --no-sync --with wonderwords python evaluation/run_lsa_suite.py --checkpoint "$CK" \
    --tokenizer "$CK" --suite ruler --lengths "$3" --batch "$4" --answers_out "$O/ruler/ruler_$2_answers.jsonl" \
    --out "$O/ruler/ruler_$2.json" > "$O/ruler_$2.out" 2> "$O/ruler_$2.log"
  say "RULER $2 done $?"
}
say "start: $CK on cards $GA $GB $GC"
(
  CUDA_VISIBLE_DEVICES=$GA .venv/bin/python scripts/concept_use_probe.py rows --checkpoint "$CK" --shard "$LV" \
    --limit 48 --out "Cache/rows_exam/${NAME}_list.json" > "Cache/rows_exam/${NAME}_list.out" \
    2> "Cache/rows_exam/${NAME}_list.log"
  say "list exam done $?"
  EXTRA="--mode start" bash scripts/run_use_probe.sh "$GA" "${NAME}_start" "$CK" 50 16384 8192 4096
  say "first-token probe done"
) &
( ruler "$GB" 4k_8k 4096,8192 8 ) &
( ruler "$GC" 16k 16384 4 ) &
wait
say "all done"
