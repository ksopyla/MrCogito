#!/bin/bash
# E37 SFT run (docs/experiments_specs/ahead/E37_sft_sensitivity.md) on a home server: one lane = one GPU = one arm,
# end to end, resumable (every step skips when its output exists).
#
#   wait   until this card is free for 15 min: no process, no claim by another lane, and no capability-check queue
#          still running on it (its log has no QUEUE-END yet); Polonez: fewer than 3 busy cards counted the same way
#          (the heat rule); Odra: every card under 78 °C
#   mix    build the instruction mix once per server (scripts/build_sft_mix.py; a lock keeps two lanes from racing)
#   base   the base model: held-out instruction loss + FineWeb/LongDocs val, `extra` (BLiMP, SWDE, BABILong 1-3)
#   lr     wait for Cache/e37_sft/lr.txt (the pre-flight's pick) unless E37_SFT_LR is set
#   sft    instruction tuning from the decay-final weights (Polonez: 5.5 h bursts, 20 min cooldowns)
#   post   the tuned model: val, core, RULER 4k + 16k, extra, long-document loss 4k / 16k / 32k
#
# Usage (from ~/dev/MrCogito-sft, in tmux; one lane per GPU):
#   bash infra/aws/jobs/e37_sft_home.sh lane ARM GPU polonez|odra
#        ARM: lsa | catalogue_reuse_n4 | e37a | notes_open (ours) | lsa_orig (theirs; Odra's ~/dev/lsa-venv)
#   E37_SFT_PREFLIGHT_FIRST=1 ... lane lsa GPU polonez   the first free card also runs the three LR pre-flights
#        (E37_SFT_PREFLIGHT_LRS, default 5e-5 1.5e-4 4.5e-4; the LSA port, E37_SFT_PREFLIGHT_TOKENS each) and the pick
#   bash infra/aws/jobs/e37_sft_home.sh preflight LR GPU polonez|odra   (one pre-flight on its own card)
#   bash infra/aws/jobs/e37_sft_home.sh pick    (lowest held-out answer loss of the finished pre-flights -> lr.txt)
# Base checkpoints: Cache/e37_sft/base/<arm>/ (ours: the decay `final` folder; lsa_orig: decay/checkpoint/step-763),
# pulled from S3 runs/e37/ with presigned links. Data: ~/dev/MrCogito-lsa/Cache/e33n_data (read only).
set -uo pipefail
MODE=${1:?mode}; shift
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME="$HOME/dev/hf_home" WANDB_PROJECT=MrCogito EXPERIMENT_ID=E37
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$REPO" || exit 1
D="${E37_SFT_DATA:-$HOME/dev/MrCogito-lsa/Cache/e33n_data}"
W="$REPO/Cache/e37_sft"; MIX="$W/data/sft_mix"; MAN="$MIX/manifest_sft.json"
LR_FILE="$W/lr.txt"; CLAIMS="$W/claims"
PF_TOKENS="${E37_SFT_PREFLIGHT_TOKENS:-20e6}"; SFT_TOKENS="${E37_SFT_TOKENS:-200e6}"
VENV="$HOME/dev/lsa-venv"; OV="$HOME/dev/lmeval011"
export LSA_REPO="$HOME/dev/lsa" LSA_ROOT="$HOME/dev/lsa_root"
mkdir -p "$W/runs" "$CLAIMS"
log() { echo "[e37_sft $(date -u +%Y-%m-%dT%H:%M:%SZ) ${LANE:-}] $*" | tee -a "$W/events.log"; }

# ---------------------------------------------------------------------------------------------------- GPU etiquette
# Another session's capability-check queue holds a card until its log ends with "QUEUE-END" (its jobs rest at 0 % use
# or leave the card empty for up to 20 min between jobs and heat rests, so an idle card is not a free card; advice of
# the capability session, 2026-11-12 20:50 UTC). A card without such a log has no queue.
CAP_DIR="${E37_SFT_CAP_DIR:-$HOME/dev/MrCogito-e33tcap/Cache/study/e33t_core/launch}"
cap_running() {
  local f="$CAP_DIR/$(hostname -s)_core_e33t_catalogue_n4_gpu$1.log"
  [ -f "$f" ] && ! grep -q "QUEUE-END" "$f"
}
gpu_busy() {
  cap_running "$1" || [ -f "$CLAIMS/gpu$1" ] \
    || nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .
}
busy_count() {
  local n=0 i
  for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do gpu_busy "$i" && n=$((n + 1)); done
  echo "$n"
}
max_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1; }
wait_gpu() {  # GPU HOSTRULE: 3 checks 5 min apart, then claim the GPU (released when the lane ends)
  local gpu=$1 rule=$2 ok=0
  while true; do
    if ! gpu_busy "$gpu" && { [ "$rule" != polonez ] || [ "$(busy_count)" -lt 3 ]; } \
       && { [ "$rule" != odra ] || [ "$(max_temp)" -lt 78 ]; }; then ok=$((ok + 1)); else ok=0; fi
    if [ "$ok" -ge 3 ]; then
      ( flock 9; if ! gpu_busy "$gpu" && { [ "$rule" != polonez ] || [ "$(busy_count)" -lt 3 ]; }; then
          echo "$$ $LANE" > "$CLAIMS/gpu$gpu"; exit 0; fi; exit 1 ) 9> "$CLAIMS/.lock" && break
      ok=0
    fi
    sleep 300
  done
  trap 'rm -f "$CLAIMS/gpu'"$gpu"'"' EXIT
  LAST_BREAK=$(date +%s)
  log "GPU $gpu claimed ($(busy_count) busy cards on $(hostname -s))"
}
heat_break() {  # Polonez heat rule: a 20 min rest after every 5.5 h of work (training bursts rest on their own)
  [ "${RULE:-}" = polonez ] || return 0
  if [ $(( $(date +%s) - ${LAST_BREAK:-$(date +%s)} )) -ge 19800 ]; then
    log "heat rest 20 min"; sleep 1200; LAST_BREAK=$(date +%s)
  fi
}

# --------------------------------------------------------------------------------------------------------- the mix
build_mix() {
  [ -f "$MAN" ] && return 0
  ( flock 8
    [ -f "$MAN" ] && exit 0
    log "building the instruction mix"
    uv run python scripts/build_sft_mix.py --out "$MIX" --tokenizer "$D/tokenizer_gpt2" --seq_len 16384 \
      --tokens 180e6 --seed 42 --replay_glob "$D/longdocs_lite_gpt2/*_train_*.bin" --replay_share 0.1 \
      > "$W/build_mix.log" 2>&1 || { log "mix build FAILED (see build_mix.log)"; exit 1; }
    log "mix built: $(tr -d '\n' < "$MIX/stats.json" | cut -c1-300)" ) 8> "$W/.mix.lock"
}

# ------------------------------------------------------------------------------------------------- our models' steps
ours_ck() { ls -d "$1"/*/final 2>/dev/null | tail -1; }
ev_ours() {  # CK TAG  -> $R/<TAG>_{val,extra_short,extra_babilong,core,ruler,longdoc}.json (each skipped when present)
  local ck=$1 t=$2 base=${3:-}
  heat_break; [ -f "$R/${t}_val.json" ] || { log "$t val"; uv run python evaluation/val_loss.py --checkpoint "$ck" --name "$t" \
      --data "$D" --sft_manifest "$MAN" --out "$R/${t}_val.json" > "$R/${t}_val.log" 2>&1 || log "$t val FAILED"; }
  heat_break; [ -f "$R/${t}_extra_short.json" ] || { log "$t BLiMP + SWDE"; uv run python evaluation/run_lsa_suite.py \
      --checkpoint "$ck" --tokenizer "$D/tokenizer_gpt2" --suite extra --extra_parts blimp,swde --backend sdpa \
      --batch 8 --out "$R/${t}_extra_short.json.part" > "$R/${t}_extra_short.log" 2>&1 \
      && mv "$R/${t}_extra_short.json.part" "$R/${t}_extra_short.json" || log "$t BLiMP + SWDE FAILED"; }
  heat_break; [ -f "$R/${t}_extra_babilong.json" ] || { log "$t BABILong"; uv run python evaluation/run_lsa_suite.py \
      --checkpoint "$ck" --tokenizer "$D/tokenizer_gpt2" --suite extra --extra_parts babilong \
      --extra_lengths 2048,4096,8192,16384 --limit 100 --batch "${E37_SFT_GEN_BATCH:-4}" \
      --out "$R/${t}_extra_babilong.json.part" > "$R/${t}_extra_babilong.log" 2>&1 \
      && mv "$R/${t}_extra_babilong.json.part" "$R/${t}_extra_babilong.json" || log "$t BABILong FAILED"; }
  [ -n "$base" ] && return 0   # the base models' core / RULER 4k+16k / long-document loss are the AWS runs'
  heat_break; [ -f "$R/${t}_core.json" ] || { log "$t core"; uv run python evaluation/run_lsa_suite.py --checkpoint "$ck" \
      --tokenizer "$D/tokenizer_gpt2" --suite core --limit 0 --batch 8 --backend sdpa --out "$R/${t}_core.json.part" \
      > "$R/${t}_core.log" 2>&1 && mv "$R/${t}_core.json.part" "$R/${t}_core.json" || log "$t core FAILED"; }
  heat_break; [ -f "$R/${t}_ruler.json" ] || { log "$t RULER 4k + 16k"; uv run --with wonderwords python evaluation/run_lsa_suite.py \
      --checkpoint "$ck" --tokenizer "$D/tokenizer_gpt2" --suite ruler --lengths 4096,16384 --limit 100 \
      --batch "${E37_SFT_GEN_BATCH:-4}" --out "$R/${t}_ruler.json.part" > "$R/${t}_ruler.log" 2>&1 \
      && mv "$R/${t}_ruler.json.part" "$R/${t}_ruler.json" || log "$t RULER FAILED"; }
  heat_break; [ -f "$R/${t}_longdoc.json" ] || { log "$t long-document loss"; uv run python evaluation/long_doc_loss.py \
      --checkpoint "$ck" --data "$D/longdocs_lite_gpt2" --lengths 4096,16384,32768 --rows 8 --notes_off --open_off \
      --gist_off --out "$R/${t}_longdoc.json.part" > "$R/${t}_longdoc.log" 2>&1 \
      && mv "$R/${t}_longdoc.json.part" "$R/${t}_longdoc.json" || log "$t long-document loss FAILED"; }
}
sft_ours() {  # ARM INIT LR TOKENS OUTDIR RULE
  local arm=$1 init=$2 lr=$3 tok=$4 out=$5 rule=$6 cmd ck rc
  [ -n "$(ours_ck "$out/sft")" ] && return 0
  cmd=$(uv run python scripts/run_e33n.py cmd --arm "$arm" --size m100 --phase sft --data "$D" --out "$out" \
        --init "$init" --sft_manifest "$MAN" --lr "$lr" --tokens "$tok" --micro "${E37_SFT_MICRO:-1}" --gpus 1 \
        --wandb) || { log "sft command FAILED"; return 1; }
  echo "$cmd" > "$out/sft.cmd"
  while true; do   # bursts as infra/aws/jobs/e33n_run.sh: resume from the newest complete checkpoint
    ck=$(ls -d "$out"/sft/*/checkpoint-* 2>/dev/null | awk -F'checkpoint-' '{print $NF" "$0}' | sort -rn | cut -d' ' -f2 \
         | while read -r d; do [ -f "$d/trainer_state.json" ] && { echo "$d"; break; }; done)
    ( export RUN_ID_SUFFIX="${arm}_m100_sft"
      [ "$rule" = polonez ] && B=" --max_train_hours 5.5" || B=""
      eval "$cmd$B${ck:+ --resume_from_checkpoint $ck}" ) >> "$out/sft.log" 2>&1
    rc=$?
    [ "$rc" = 0 ] && { [ "$rule" = polonez ] && { log "sft done; heat rest 20 min"; sleep 1200; LAST_BREAK=$(date +%s); }; return 0; }
    awk '/STARTING TRAINING/{f=0} /Wall-clock budget of/{f=1} END{exit !f}' "$out/sft.log" || [ "$rc" = 3 ] || return "$rc"
    log "sft burst ended; cooldown 20 min"; sleep 1200; LAST_BREAK=$(date +%s)
  done
}

# ------------------------------------------------------------------------------------------------ the original LSA
E() { PYTHONPATH="$OV${PYTHONPATH:+:$PYTHONPATH}" "$VENV/bin/python" evaluation/lsa_orig_eval.py "$@"; }
ev_orig() {  # CK TAG [base]
  local ck=$1 t=$2 base=${3:-}
  heat_break; [ -f "$R/${t}_val.json" ] || { log "$t val"; E val --checkpoint "$ck" --name "$t" --data "$D" --sft_manifest "$MAN" \
      --out "$R/${t}_val.json" > "$R/${t}_val.log" 2>&1 || log "$t val FAILED"; }
  heat_break; [ -f "$R/${t}_extra.json" ] || { log "$t extra"; E extra --checkpoint "$ck" --data "$D" --limit 100 \
      --extra_lengths 2048,4096,8192,16384 --out "$R/${t}_extra.json.part" > "$R/${t}_extra.log" 2>&1 \
      && mv "$R/${t}_extra.json.part" "$R/${t}_extra.json" || log "$t extra FAILED"; }
  [ -n "$base" ] && return 0
  heat_break; [ -f "$R/${t}_core.json" ] || { log "$t core"; E core --checkpoint "$ck" --data "$D" --limit 0 \
      --out "$R/${t}_core.json.part" > "$R/${t}_core.log" 2>&1 && mv "$R/${t}_core.json.part" "$R/${t}_core.json" \
      || log "$t core FAILED"; }
  heat_break; [ -f "$R/${t}_ruler.json" ] || { log "$t RULER 4k + 16k"; E ruler --checkpoint "$ck" --data "$D" --lengths 4096,16384 \
      --limit 100 --batch_tokens 65536 --out "$R/${t}_ruler.json.part" > "$R/${t}_ruler.log" 2>&1 \
      && mv "$R/${t}_ruler.json.part" "$R/${t}_ruler.json" || log "$t RULER FAILED"; }
  heat_break; [ -f "$R/${t}_longdoc.json" ] || { log "$t long-document loss"; E longdoc --checkpoint "$ck" --data "$D" \
      --lengths 4096,16384,32768 --rows 8 --open_off --out "$R/${t}_longdoc.json.part" > "$R/${t}_longdoc.log" 2>&1 \
      && mv "$R/${t}_longdoc.json.part" "$R/${t}_longdoc.json" || log "$t long-document loss FAILED"; }
}
sft_orig() {  # INIT LR TOKENS OUTDIR
  local init=$1 lr=$2 tok=$3 out=$4
  [ -f "$out/sft/DONE" ] && return 0
  "$VENV/bin/torchrun" --standalone --nproc_per_node=1 scripts/lsa_orig.py train --phase sft --data "$D" --out "$out" \
    --init "$init" --sft_manifest "$MAN" --lr "$lr" --tokens "$tok" --micro "${E37_SFT_MICRO:-1}" >> "$out/sft.log" 2>&1 \
    && touch "$out/sft/DONE"
}
orig_final() { ls -d "$1"/sft/checkpoint/step-* 2>/dev/null | sort -V | tail -1; }

preflight_one() {  # LR: the LSA port, E37_SFT_PREFLIGHT_TOKENS of the mix at this peak LR, then its held-out answer loss
  local lrv=$1 pr="$W/preflight/lr$1"
  mkdir -p "$pr"
  [ -f "$pr/val.json" ] && return 0
  log "pre-flight LR $lrv ($PF_TOKENS tokens)"
  sft_ours lsa "$W/base/lsa" "$lrv" "$PF_TOKENS" "$pr" "$RULE" || { log "pre-flight $lrv FAILED (see $pr/sft.log)"; return 1; }
  uv run python evaluation/val_loss.py --checkpoint "$(ours_ck "$pr/sft")" --name "pf_$lrv" --data "$D" \
    --sft_manifest "$MAN" --sft_only --out "$pr/val.json" > "$pr/val.log" 2>&1 || { log "pre-flight $lrv val FAILED"; return 1; }
  log "pre-flight $lrv done: $(tr -d '\n ' < "$pr/val.json" | grep -o '"sft_test_16k":{[^}]*}' | cut -c1-200)"
}

# ------------------------------------------------------------------------------------------------------------ modes
case "$MODE" in
  lane)
    ARM=${1:?arm}; GPU=${2:?gpu}; RULE=${3:?polonez|odra}; LANE="$ARM@gpu$GPU"
    R="$W/runs/$ARM"; mkdir -p "$R"; export CUDA_VISIBLE_DEVICES="$GPU"
    log "lane start ($RULE); waiting for GPU $GPU"
    wait_gpu "$GPU" "$RULE"
    build_mix || exit 1
    if [ -n "${E37_SFT_PREFLIGHT_FIRST:-}" ]; then   # the LR pre-flights, back to back on this card, then the pick
      for LRV in ${E37_SFT_PREFLIGHT_LRS:-5e-5 1.5e-4 4.5e-4}; do preflight_one "$LRV"; heat_break; done
      bash "$0" pick | tee -a "$W/events.log"
    fi
    if [ "$ARM" = lsa_orig ]; then BASE="$W/base/lsa_orig/decay/checkpoint/step-763"; ev_orig "$BASE" base base
    else BASE="$W/base/$ARM"; ev_ours "$BASE" base base; fi
    LR="${E37_SFT_LR:-}"
    if [ -z "$LR" ]; then
      rm -f "$CLAIMS/gpu$GPU"; log "waiting for the learning rate ($LR_FILE); GPU $GPU released meanwhile"
      while [ ! -s "$LR_FILE" ]; do sleep 300; done
      LR=$(cat "$LR_FILE"); wait_gpu "$GPU" "$RULE"
    fi
    log "sft start (LR $LR, $SFT_TOKENS tokens)"
    if [ "$ARM" = lsa_orig ]; then
      sft_orig "$BASE" "$LR" "$SFT_TOKENS" "$R" || { log "sft FAILED (see $R/sft.log)"; exit 1; }
      ev_orig "$(orig_final "$R")" sft
    else
      sft_ours "$ARM" "$BASE" "$LR" "$SFT_TOKENS" "$R" "$RULE" || { log "sft FAILED (see $R/sft.log)"; exit 1; }
      ev_ours "$(ours_ck "$R/sft")" sft
    fi
    log "lane done" ;;
  preflight)   # one pre-flight on its own card (the lane mode runs all three with E37_SFT_PREFLIGHT_FIRST=1)
    LRV=${1:?lr}; GPU=${2:?gpu}; RULE=${3:?polonez|odra}; LANE="preflight_$LRV@gpu$GPU"
    export CUDA_VISIBLE_DEVICES="$GPU"
    log "pre-flight start; waiting for GPU $GPU"
    wait_gpu "$GPU" "$RULE"
    build_mix || exit 1
    preflight_one "$LRV"
    rm -f "$CLAIMS/gpu$GPU"
    bash "$0" pick ;;
  pick)
    [ -s "$LR_FILE" ] && { echo "already picked: $(cat "$LR_FILE")"; exit 0; }
    uv run python - "$W/preflight" "$LR_FILE" "${E37_SFT_PREFLIGHT_N:-3}" <<'EOF'
import json, sys
from pathlib import Path
root, out, need = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
res = {}
for f in root.glob("lr*/val.json"):
    rows = [x for x in json.loads(f.read_text()).values() if isinstance(x, dict) and "sft_test_16k" in x]
    v = rows[0]["sft_test_16k"].get("token_mean") if rows else None
    if v is not None:
        res[f.parent.name[2:]] = v
print("pre-flight held-out answer loss:", res)
if len(res) >= need:
    best = min(res, key=res.get)
    out.write_text(best + "\n")
    (out.parent / "lr_pick.json").write_text(json.dumps({"loss": res, "pick": best}, indent=2))
    print("picked", best)
EOF
    ;;
  *) echo "mode: lane | preflight | pick"; exit 2 ;;
esac
