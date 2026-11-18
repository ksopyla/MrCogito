#!/usr/bin/env bash
# The model-release evaluation protocol (.cursor/skills/model-release, 2026-11-18) for ONE perceiver_ar checkpoint,
# on three GPUs: the Tiny-ML board tier (BLiMP 67 sets, ARC-Easy, WikiText-2), the 7 core lm-eval tasks, sampled
# text, the cross-model long-document test (our row), our own long-context tests at 16k / 64k / 128k, and RULER
# 4k–32k with each task's own answer length and every reply saved. Same settings as the 150M concept-memory card
# (docs/release/hf_card_concept_memory_150m); reference models are scored separately (they need no rerun).
#
# Usage, from a MrCogito checkout with a synced .venv (uv sync --frozen):
#   setsid nohup bash scripts/release_evals.sh <checkpoint> <out_dir> <longdocs_lite_dir> [gpus=0,1,2] \
#       [memory_modes=off] [tag=model] > /dev/null 2>&1 < /dev/null &
# memory_modes: what long_context_use.py switches off on the full span: "off,nosum" for the concept read,
# "off" for the LSA read (it has no summaries). Progress: <out_dir>/logs/events.log (UTC stamps).
# Every job waits until its GPU runs no compute process (other sessions' jobs included), checking once a minute.
# LONGCTX=<dir> reuses an existing retokenised-document cache + spans (default Cache/e38_board/longctx).
# HF_EXPORT=<dir> (a scripts/export_hub_model.py folder) adds the board tier through lm-eval's stock `hf` model on the
# from_pretrained export: the same tests the way a reader (and every reference model) runs them.
set -euo pipefail
CK=$1; OUT=$2; DATA=$3; GPUS=${4:-0,1,2}; MODES=${5:-off}; TAG=${6:-model}
IFS=, read -r G0 G1 G2 <<< "$GPUS"
export PATH="$HOME/.local/bin:$PATH"
PY=(.venv/bin/python)
RULER=(uv run --no-sync --with wonderwords python evaluation/run_lsa_suite.py --checkpoint "$CK" --tokenizer "$CK"
       --suite ruler)
LONGCTX=${LONGCTX:-Cache/e38_board/longctx}  # retokenised documents + the 55 spans (document text: never published)
SPANS=$LONGCTX/spans_64k.json.gz
SOURCES=gutenberg,arxiv,doab,usgpo,hansard
mkdir -p "$OUT"/{eval_results,long_context,ruler,samples,logs} "$LONGCTX"
EV="$OUT/logs/events.log"

stamp() { echo "[$(date -u +%FT%TZ)] $*" >> "$EV"; }
wait_gpu() {  # until GPU $1 runs no compute process
  local said=0
  while [[ -n $(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; do
    (( said )) || { stamp "GPU $1: busy with another job, waiting"; said=1; }
    sleep 60
  done
}
job() {  # job <gpu> <name> <stdout file or -> <command...>
  local g=$1 n=$2 so=$3; shift 3
  [[ $so == - ]] && so="$OUT/logs/$n.out"
  wait_gpu "$g"
  stamp "GPU $g: $n start"
  local t0=$SECONDS rc=0
  CUDA_VISIBLE_DEVICES=$g "$@" > "$so" 2> "$OUT/logs/$n.log" || rc=$?
  if (( rc == 0 )); then stamp "GPU $g: $n done in $((SECONDS - t0)) s"
  else stamp "GPU $g: $n FAILED (exit $rc) after $((SECONDS - t0)) s"; fi
}

stamp "start: $TAG, checkpoint $CK, GPUs $GPUS, memory modes $MODES"
# Spans first (CPU; it builds the retokenised cache once, so the lanes never race on writing it).
if [[ ! -f $SPANS ]]; then
  stamp "spans_prepare start"
  CUDA_VISIBLE_DEVICES= "${PY[@]}" evaluation/long_context_compare.py prepare --tokenizer "$CK" --data "$DATA" \
      --cache "$LONGCTX" --sources "$SOURCES" --out "$SPANS" > "$OUT/logs/spans_prepare.out" 2> "$OUT/logs/spans_prepare.log" \
      && stamp "spans_prepare done" || { stamp "spans_prepare FAILED: no spans, stopping"; exit 1; }
fi

lane0() {
  job "$G0" board - "${PY[@]}" evaluation/run_lm_eval_suite.py --checkpoint "$CK" --tag "${TAG}_final" \
      --tier leaderboard --out_dir "$OUT/eval_results"
  job "$G0" core7 - "${PY[@]}" evaluation/run_lm_eval_suite.py --checkpoint "$CK" --tag "${TAG}_core7" \
      --tasks arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,sciq,winogrande,wikitext \
      --out_dir "$OUT/eval_results"
  if [[ -n ${HF_EXPORT:-} ]]; then
    job "$G0" board_hf - "${PY[@]}" evaluation/run_lm_eval_suite.py --hf_model "$HF_EXPORT" --dtype bfloat16 \
        --tag "${TAG}_final_hf" --tier leaderboard --out_dir "$OUT/eval_results"
  fi
  job "$G0" samples - "${PY[@]}" evaluation/sample_text.py --model "$CK" --out "$OUT/samples/samples_ours.json"
  job "$G0" compare "$OUT/long_context/compare_ours.json" "${PY[@]}" evaluation/long_context_compare.py score \
      --spans "$SPANS" --swap --model "$CK"
  job "$G0" ruler_32k_c - "${RULER[@]}" --lengths 32768 --batch 2 --tasks ruler_cwe,ruler_fwe,ruler_qa_hotpot,ruler_qa_squad \
      --answers_out "$OUT/ruler/ruler_32k_c_answers.jsonl" --out "$OUT/ruler/ruler_32k_c.json"
  job "$G0" ruler_32k_a - "${RULER[@]}" --lengths 32768 --batch 2 \
      --tasks niah_multikey_1,niah_multikey_2,niah_single_1,niah_single_2,niah_single_3 \
      --answers_out "$OUT/ruler/ruler_32k_a_answers.jsonl" --out "$OUT/ruler/ruler_32k_a.json"
}
lane1() {
  local use=("${PY[@]}" evaluation/long_context_use.py --checkpoint "$CK" --data "$DATA" --cache "$LONGCTX" --swap
             --modes "$MODES")
  job "$G1" real_docs_64k "$OUT/long_context/real_docs_64k.json" "${use[@]}" --sources "$SOURCES" --span 65536 \
      --contexts 1024,4096,16384 --docs 16
  job "$G1" real_docs_16k "$OUT/long_context/real_docs_16k.json" "${use[@]}" --sources "$SOURCES" --span 16384 \
      --contexts 1024,4096 --docs 24
  job "$G1" real_docs_128k "$OUT/long_context/real_docs_128k.json" "${use[@]}" --sources "$SOURCES" --span 131072 \
      --contexts 4096,16384,61440 --docs 8
  job "$G1" ruler_32k_b - "${RULER[@]}" --lengths 32768 --batch 2 \
      --tasks niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt \
      --answers_out "$OUT/ruler/ruler_32k_b_answers.jsonl" --out "$OUT/ruler/ruler_32k_b.json"
}
lane2() {
  job "$G2" ruler_4k_8k - "${RULER[@]}" --lengths 4096,8192 --batch 8 \
      --answers_out "$OUT/ruler/ruler_4k_8k_answers.jsonl" --out "$OUT/ruler/ruler_4k_8k.json"
  job "$G2" ruler_16k - "${RULER[@]}" --lengths 16384 --batch 4 \
      --answers_out "$OUT/ruler/ruler_16k_answers.jsonl" --out "$OUT/ruler/ruler_16k.json"
}

lane0 & lane1 & lane2 &
wait
stamp "all lanes done"
