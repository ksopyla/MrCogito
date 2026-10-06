#!/usr/bin/env python
"""Plan and launch the text capability checks (draft `text-v0`) on Polonez / Odra, or locally.

Spec: docs/engineering_specs/text_capability_checks.md · skill `text-checks` · definitions
`evaluation/text_checks.py` · scorer `evaluation/text_checks_eval.py` · scorecard
`analysis/text_checks_scorecard.py`.

Phases (each writes job scripts + a byobu starter; run them in this order):

  data      build the tokenizer, the training mix and the frozen eval sets (once per data version)
  tune      the recipe search: a step-size grid per model, short 1-GPU runs in parallel (§7)
  select    pick each model's step size by DEVELOPMENT loss, write it into its recipe card (commit it)
  train     one from-scratch run per model on all GPUs of the host, one after another
  eval      score each model (all lengths, controls, notebook off, learning-curve checkpoints), 1 GPU each
  status    what is done / running / over budget

  # on the server (scripts are generated where they run; Cache/ is gitignored)
  uv run python scripts/run_text_checks.py plan --phase data  --host polonez --data $TOK/text_checks_v0
  uv run python scripts/run_text_checks.py plan --phase tune  --tier screen --arches dense local e31c \\
      --host polonez --gpus 0 1 2 3 --data $TOK/text_checks_v0 --out Cache/text_checks/r1_screen
  bash Cache/text_checks/r1_screen/launch/tune_start.sh
  uv run python scripts/run_text_checks.py select --out Cache/text_checks/r1_screen --tier screen
  uv run python scripts/run_text_checks.py plan --phase train --tier screen ... (same args)
  bash Cache/text_checks/r1_screen/launch/train_start.sh            # trains, then evals on all GPUs
  uv run python analysis/text_checks_scorecard.py --in_dir Cache/text_checks/r1_screen

  # local plumbing check (minutes, MPS/CPU): the smoke tier, jobs run here one by one
  uv run python scripts/run_text_checks.py plan --phase train --tier smoke --mode local \\
      --data Cache/text_checks/smoke --out Cache/text_checks/smoke_runner

Every job is resumable: re-running a starter skips jobs with a DONE marker, and a training job
resumes from its newest checkpoint (Polonez shuts down for heat). Active training time is counted
across resumes; when it reaches the tier's GPU-hour cap the run is stopped and scored at its newest
checkpoint, labeled over-budget.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.text_checks import (  # noqa: E402
    ARCHES,
    CURVE_FRACTIONS,
    ENV_OF,
    TEXT_CHECKS_VERSION,
    TIERS,
    fit_params,
    grad_accum,
    in_band,
    load_recipe,
    steps_for,
)

ROOT = Path(__file__).resolve().parent.parent
SESSION = "textchk"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _q(v) -> str:
    return shlex.quote(str(v))


def _meta(data: Path) -> dict:
    path = data / "text_checks_meta.json"
    if not path.exists():
        raise SystemExit(f"no {path}: run the data phase first")
    return json.loads(path.read_text())


# ------------------------------------------------------------------------------------- training job
def train_env(args_model: dict, recipe: dict, tier: str, *, lr: float, steps: int, n_gpus: int, gpu_ids: str,
              data: Path, out_dir: Path, seed: int, exp_id: str, save_steps: int, extra: dict) -> dict:
    t = TIERS[tier]
    env = {ENV_OF[k]: v for k, v in args_model.items() if k in ENV_OF}
    env.update({
        "NUM_GPUS": n_gpus, "GPU_IDS": gpu_ids, "TRAIN_OUTPUT_DIR": out_dir,
        "EXPERIMENT_ID": exp_id, "TOKENIZER_NAME": data / "tokenizer",
        "PRETOKENIZED_MANIFEST": data / "manifest.json", "MAX_SEQ_LENGTH": t.seq_len,
        "BATCH_PACKING_MODE": "length_group", "PER_DEVICE_BATCH_SIZE": t.per_device_batch,
        "EVAL_BATCH_SIZE": t.per_device_batch, "GRADIENT_ACCUMULATION_STEPS": grad_accum(tier, n_gpus),
        "MAX_STEPS": steps, "WARMUP_STEPS": max(1, int(recipe["warmup_frac"] * steps)),
        "LEARNING_RATE": lr, "LR_SCHEDULER_TYPE": recipe["scheduler"], "OPTIMIZER": recipe["optimizer"],
        "WEIGHT_DECAY": recipe["weight_decay"], "MAX_GRAD_NORM": recipe["max_grad_norm"],
        "LOGGING_STEPS": max(1, steps // 200), "EVAL_STEPS": save_steps, "SAVE_STEPS": save_steps,
        "SAVE_TOTAL_LIMIT": 40, "LOAD_BEST_MODEL_AT_END": "False", "MAX_EVAL_SAMPLES": 256,
        "SEED": seed, "DATALOADER_NUM_WORKERS": 2, "CHUNKED_CE_BLOCK_SIZE": 2048,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    env.update(extra)
    return {k: str(v) for k, v in env.items()}


def local_train_cmd(args_model: dict, recipe: dict, tier: str, *, lr: float, steps: int, data: Path,
                    out_dir: Path, seed: int, save_steps: int) -> list[str]:
    """The same run through the trainer directly (no accelerate / nvidia-smi): MPS or CPU."""
    t = TIERS[tier]
    cmd = ["uv", "run", "python", "training/train_concept_pretraining.py"]
    for k, v in args_model.items():
        cmd += [f"--{k}", str(v)]
    cmd += [
        "--attn_backend", "sdpa", "--attn_pad_multiple", "1", "--use_liger", "False",
        "--pretokenized_manifest", str(data / "manifest.json"), "--tokenizer_name", str(data / "tokenizer"),
        "--max_seq_length", str(t.seq_len), "--batch_packing_mode", "none",
        "--per_device_train_batch_size", str(t.per_device_batch), "--per_device_eval_batch_size", str(t.per_device_batch),
        "--gradient_accumulation_steps", str(grad_accum(tier, 1)), "--learning_rate", str(lr),
        "--warmup_steps", str(max(1, int(recipe["warmup_frac"] * steps))), "--max_steps", str(steps),
        "--lr_scheduler_type", recipe["scheduler"], "--weight_decay", str(recipe["weight_decay"]),
        "--max_grad_norm", str(recipe["max_grad_norm"]), "--optim", "adamw_torch",
        "--logging_steps", str(max(1, steps // 20)), "--eval_strategy", "steps", "--eval_steps", str(save_steps),
        "--save_strategy", "steps", "--save_steps", str(save_steps), "--save_total_limit", "40",
        "--max_eval_samples", "16", "--output_dir", str(out_dir), "--seed", str(seed), "--report_to", "none",
        "--overwrite_output_dir", "True", "--disable_tqdm", "True", "--dataloader_num_workers", "0",
        "--prediction_loss_only", "True",
    ]
    return cmd


TRAIN_SH = r"""#!/usr/bin/env bash
# text checks {version} · {job} · generated {when}
set -uo pipefail
cd {root}
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=. WANDB_MODE="${{WANDB_MODE:-online}}"
JOB={jobdir}
[ -f "$JOB/DONE" ] && {{ echo "skip {job} (DONE)"; exit 0; }}
mkdir -p "$JOB/train"
CAP_S={cap_s}
ACTIVE=$(cat "$JOB/active_seconds" 2>/dev/null || echo 0)
if [ "$ACTIVE" -ge "$CAP_S" ]; then echo over_budget > "$JOB/status"; fi
RESUME=""
CK=$(ls -d "$JOB"/train/*/checkpoint-* 2>/dev/null | awk -F- '{{print $NF" "$0}}' | sort -n | tail -1 | cut -d' ' -f2- || true)
if [ -n "$CK" ] && [ ! -f "$JOB/status" ]; then RESUME="$CK"; echo "resume from $CK"; fi
if [ ! -f "$JOB/status" ]; then
{env}
    export RESUME_FROM_CHECKPOINT="$RESUME"
    echo running > "$JOB/status.run"
    {setsid}{cmd} > "$JOB/train.log" 2>&1 &
    PID=$!
    BURST_S={burst_s}   # 0 = no burst limit; else pause at the first checkpoint after this many seconds
    RUN_S=0; NCK_AT=""; PAUSED=0
    nck() {{ ls -d "$JOB"/train/*/checkpoint-*/trainer_state.json 2>/dev/null | wc -l; }}
    while kill -0 "$PID" 2>/dev/null; do
        sleep 30
        ACTIVE=$((ACTIVE + 30)); RUN_S=$((RUN_S + 30)); echo "$ACTIVE" > "$JOB/active_seconds"
        if [ "$ACTIVE" -ge "$CAP_S" ]; then
            echo "compute cap reached ($CAP_S s active): stopping"; kill -TERM -- -"$PID" 2>/dev/null
            sleep 60; kill -KILL -- -"$PID" 2>/dev/null; echo over_budget > "$JOB/status"; break
        fi
        if [ "$BURST_S" -gt 0 ] && [ "$RUN_S" -ge "$BURST_S" ]; then
            [ -z "$NCK_AT" ] && NCK_AT=$(nck) && echo "burst time reached: pausing at the next checkpoint"
            if [ "$(nck)" -gt "$NCK_AT" ]; then
                sleep 45   # let the checkpoint finish writing
                echo "paused after a checkpoint ($RUN_S s this burst)"; kill -TERM -- -"$PID" 2>/dev/null
                sleep 30; kill -KILL -- -"$PID" 2>/dev/null; PAUSED=1; break
            fi
        fi
    done
    wait "$PID"; RC=$?
    rm -f "$JOB/status.run"
    if [ "$PAUSED" = 1 ] && [ ! -f "$JOB/status" ] && [ -z "$(ls -d "$JOB"/train/*/final 2>/dev/null)" ]; then
        echo "EXIT {job} 75 (paused for cooldown)"; exit 75
    fi
fi
FINAL=$(ls -td "$JOB"/train/*/final 2>/dev/null | head -1 || true)
if [ -n "$FINAL" ]; then
    echo "$FINAL" > "$JOB/model_path"; echo finished > "$JOB/status"
elif [ "$(cat "$JOB/status" 2>/dev/null)" = "over_budget" ]; then
    CK=$(ls -d "$JOB"/train/*/checkpoint-* 2>/dev/null | awk -F- '{{print $NF" "$0}}' | sort -n | tail -1 | cut -d' ' -f2- || true)
    [ -n "$CK" ] && echo "$CK" > "$JOB/model_path"
else
    echo "EXIT {job} ${{RC:-1}} (no final model; re-run this script to resume)"; exit "${{RC:-1}}"
fi
python3 - "$JOB" <<'PY'
import json, sys, glob, os
job = sys.argv[1]
st = sorted(glob.glob(os.path.join(job, "train", "*", "*", "trainer_state.json")) +
            glob.glob(os.path.join(job, "train", "*", "trainer_state.json")), key=os.path.getmtime)
hist = json.load(open(st[-1]))["log_history"] if st else []
ev = [h for h in hist if "eval_loss" in h]
res = {{"status": open(os.path.join(job, "status")).read().strip(),
        "active_seconds": int(open(os.path.join(job, "active_seconds")).read() or 0) if os.path.exists(os.path.join(job, "active_seconds")) else None,
        "model_path": open(os.path.join(job, "model_path")).read().strip() if os.path.exists(os.path.join(job, "model_path")) else None,
        "eval_loss": [(h.get("step"), h["eval_loss"]) for h in ev],
        "train_loss": [(h.get("step"), h["loss"]) for h in hist if "loss" in h],
        "tokens_per_second": [h["perf/real_tokens_per_second"] for h in hist if "perf/real_tokens_per_second" in h]}}
json.dump(res, open(os.path.join(job, "result.json"), "w"), indent=1)
PY
touch "$JOB/DONE"; echo "EXIT {job} 0 ($(cat "$JOB/status"))"
"""

EVAL_SH = r"""#!/usr/bin/env bash
# text checks {version} · {job} · generated {when}
set -uo pipefail
cd {root}
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=.
JOB={jobdir}
[ -f "$JOB/DONE" ] && {{ echo "skip {job} (DONE)"; exit 0; }}
[ -f {model_file} ] || {{ echo "EXIT {job} 3 (training not finished: {model_file})"; exit 3; }}
MODEL=$(cat {model_file})
{model_sel}
mkdir -p "$JOB"
{cmds}
touch "$JOB/DONE"; echo "EXIT {job} 0"
"""


def _env_block(env: dict) -> str:
    return "\n".join(f"    export {k}={_q(v)}" for k, v in sorted(env.items()))


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def plan_train_job(name: str, arch: str, tier: str, *, lr: float, tokens: float, n_gpus: int, gpu_ids: str,
                   data: Path, out: Path, seed: int, mode: str, extra: dict, cap_gpu_hours: float,
                   burst_s: int = 0) -> dict:
    t = TIERS[tier]
    meta = _meta(data)
    mean_row = float(meta["mix"]["mean_row_tokens"])
    steps = steps_for(tokens, t.global_rows, mean_row)
    save_steps = max(1, steps // 20)
    args_model, n_params = fit_params(arch, tier)
    recipe = load_recipe(arch)
    jobdir = out / "jobs" / name
    exp_id = f"TEXTCHK-{out.name}-{name}"
    if mode == "local":
        cmd = " ".join(_q(c) for c in local_train_cmd(args_model, recipe, tier, lr=lr, steps=steps, data=data,
                                                      out_dir=jobdir / "train", seed=seed, save_steps=save_steps))
        env = {"PYTORCH_ENABLE_MPS_FALLBACK": "1", "WANDB_MODE": "disabled", "RESUME_FROM_CHECKPOINT": ""}
        cmd += ' ${RESUME:+--resume_from_checkpoint "$RESUME"}'
    else:
        env = train_env(args_model, recipe, tier, lr=lr, steps=steps, n_gpus=n_gpus, gpu_ids=gpu_ids, data=data,
                        out_dir=jobdir / "train", seed=seed, exp_id=exp_id, save_steps=save_steps, extra=extra)
        cmd = "bash scripts/train_concept_pretraining_multigpu.sh"
    cap_s = int(cap_gpu_hours * 3600 / n_gpus)
    _write(jobdir / "job.sh", TRAIN_SH.format(version=TEXT_CHECKS_VERSION, job=name, when=datetime.now().isoformat(timespec="seconds"),
                                              root=_q(ROOT), jobdir=_q(jobdir), cap_s=cap_s, env=_env_block(env), cmd=cmd,
                                              setsid="" if mode == "local" else "setsid ", burst_s=burst_s))
    card = {
        "job": name, "kind": "train", "arch": arch, "role": ARCHES[arch].role, "tier": tier, "version": TEXT_CHECKS_VERSION,
        "params": n_params, "params_in_band": in_band(n_params, tier), "model_args": args_model, "recipe": recipe,
        "lr": lr, "tokens": tokens, "steps": steps, "global_rows": t.global_rows, "mean_row_tokens": mean_row,
        "save_steps": save_steps, "n_gpus": n_gpus, "gpu_ids": gpu_ids, "cap_gpu_hours": cap_gpu_hours, "cap_seconds": cap_s,
        "seed": seed, "burst_seconds": burst_s, "data": str(data), "data_version": meta.get("version"), "tokenizer_sha256": meta.get("tokenizer_sha256"),
        "eval_sha256": meta.get("eval_sha256"), "git_commit": _git_commit(), "mode": mode, "extra_env": extra,
        "planned": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (jobdir / "job.json").write_text(json.dumps(card, indent=2))
    return card


def plan_eval_job(name: str, arch: str, tier: str, train_job: str, *, data: Path, out: Path, gpu: str, mode: str,
                  steps: int) -> dict:
    t = TIERS[tier]
    jobdir = out / "jobs" / name
    model_file = out / "jobs" / train_job / "model_path"
    dev = "" if mode == "local" else f"CUDA_VISIBLE_DEVICES={gpu} "
    device = "--device cpu" if mode == "local" else ""
    lens = " ".join(str(x) for x in t.eval_lengths)
    extra_lens = " ".join(str(x) for x in t.extra_lengths)
    cap = "--max_items_per_cell 3" if tier == "smoke" else ""
    ev = f"uv run python evaluation/text_checks_eval.py --tokenizer {_q(data / 'tokenizer')} {device} {cap}"
    # (output file, command): each final evaluation is skipped when its file exists (resumable)
    finals = [
        ("final_id.json", f'{dev}{ev} --checkpoint "$MODEL" --items {_q(data / "eval" / "id.jsonl")} --lengths {lens} '
                          f'--story_eval {_q(data / "stories" / "eval")} --out "$JOB/final_id.json"'),
        ("final_extra.json", f'{dev}{ev} --checkpoint "$MODEL" --items {_q(data / "eval" / "harder.jsonl")} '
                             f'{_q(data / "eval" / "paraphrase.jsonl")} --lengths {extra_lens} --out "$JOB/final_extra.json"'),
    ]
    if ARCHES[arch].has_notebook:
        finals.append(("final_id_notebook_off.json",
                       f'{dev}{ev} --checkpoint "$MODEL" --items {_q(data / "eval" / "id.jsonl")} --lengths {t.seq_len} '
                       f'{max(t.eval_lengths)} --message_override none --no_removed --out "$JOB/final_id_notebook_off.json"'))
    cmds = [f'[ -f "$JOB/{f}" ] || {c} || {{ echo "EXIT {name} 1 ({f})"; exit 1; }}' for f, c in finals]
    # learning curve: the checkpoints nearest to 10/25/50/75 % of the steps, scored at the training length
    curve = " ".join(str(max(1, round(f * steps))) for f in CURVE_FRACTIONS[:-1])
    model_sel = (f'TRAIN_ROOT=$(dirname "$(dirname "$MODEL")")\n'
                 f'CURVE=""\nfor s in {curve}; do\n'
                 f'  CK=$(ls -d "$TRAIN_ROOT"/*/checkpoint-* 2>/dev/null | awk -F- -v s=$s \'{{d=$NF-s; if(d<0)d=-d; print d" "$0}}\' | sort -n | head -1 | cut -d" " -f2-)\n'
                 f'  [ -n "$CK" ] && CURVE="$CURVE $CK"\ndone')
    cmds.append(f'for CK in $CURVE; do n=$(basename "$CK"); [ -f "$JOB/curve_$n.json" ] || {dev}{ev} --checkpoint "$CK" '
                f'--items {_q(data / "eval" / "id.jsonl")} --lengths {t.seq_len} --no_removed --out "$JOB/curve_$n.json"; done')
    # optimizer states are only needed to resume training, which is finished: free the disk
    cmds.append('find "$TRAIN_ROOT" -name "optimizer.pt" -delete 2>/dev/null || true')
    _write(jobdir / "job.sh", EVAL_SH.format(version=TEXT_CHECKS_VERSION, job=name, when=datetime.now().isoformat(timespec="seconds"),
                                             root=_q(ROOT), jobdir=_q(jobdir), model_file=_q(model_file),
                                             model_sel=model_sel, cmds="\n".join(cmds)))
    card = {"job": name, "kind": "eval", "arch": arch, "tier": tier, "train_job": train_job, "gpu": gpu,
            "version": TEXT_CHECKS_VERSION, "lengths": list(t.eval_lengths), "git_commit": _git_commit()}
    (jobdir / "job.json").write_text(json.dumps(card, indent=2))
    return card


def plan_data_job(data: Path, out: Path, tokens: float, num_proc: int, profile: str) -> dict:
    """profile `smoke`: the small 1k-token data of the smoke tier (from the story validation files);
    `full`: the 4k-token protocol data."""
    jobdir = out / "jobs" / "data"
    if profile == "smoke":
        cmd = (f"uv run python scripts/build_text_checks_data.py --stories smoke --out_dir {_q(data)} --seq_len 1024 "
               f"--target_tokens {int(tokens)} --num_proc 4 --eval_lengths 512 1024 2048 --extra_lengths 1024 "
               f"--eval_items 6 --tokenizer_sample 20000")
    else:
        cmd = (f"uv run python scripts/build_text_checks_data.py --stories full --out_dir {_q(data)} --seq_len 4096 "
               f"--target_tokens {int(tokens)} --num_proc {num_proc}")
    text = (f"#!/usr/bin/env bash\nset -uo pipefail\ncd {_q(ROOT)}\nexport PATH=\"$HOME/.local/bin:$PATH\" PYTHONPATH=.\n"
            f"JOB={_q(jobdir)}\n[ -f {_q(data / 'text_checks_meta.json')} ] && {{ echo 'skip data (built)'; touch \"$JOB/DONE\"; exit 0; }}\n"
            f"{cmd} 2>&1 | tee \"$JOB/build.log\"\nRC=${{PIPESTATUS[0]}}\n[ $RC = 0 ] && touch \"$JOB/DONE\"\necho \"EXIT data $RC\"\nexit $RC\n")
    _write(jobdir / "job.sh", text)
    card = {"job": "data", "kind": "data", "data": str(data), "tokens": tokens, "git_commit": _git_commit()}
    (jobdir / "job.json").write_text(json.dumps(card, indent=2))
    return card


# ------------------------------------------------------------------------------------- starters
def write_starter(out: Path, phase: str, queues: dict[str, list[str]], host: str, then: dict | None = None,
                  mode: str = "scripts", cooldown_s: int = 0):
    """queues: window name → job names run one after another. `then`: queues started after these finish."""
    launch = out / "launch"

    def queue_script(qname, jobs):
        log = _q(launch / (qname + ".log"))
        lines = ["#!/usr/bin/env bash", f"cd {_q(ROOT)}", f"COOLDOWN_S={cooldown_s}",
                 # Polonez heat rule: start or resume only when every GPU is below 70 C
                 'cool() { command -v nvidia-smi >/dev/null || return 0; '
                 'while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1)" -ge 70 ]; '
                 'do echo "GPU above 70 C, waiting $(date +%T)"; sleep 120; done; }']
        for i, j in enumerate(jobs):
            sh = _q(out / "jobs" / j / "job.sh")
            lines.append(f'echo "JOB {j}" | tee -a {log}; cool')
            lines.append(f'while true; do bash {sh} 2>&1 | tee -a {log}; rc=${{PIPESTATUS[0]}}; '
                         f'[ "$rc" = 75 ] || break; echo "COOLDOWN $COOLDOWN_S s $(date +%T)" | tee -a {log}; '
                         f'sleep "$COOLDOWN_S"; cool; done')
            if cooldown_s and i < len(jobs) - 1 and (j.startswith("train_") or j.startswith("tune_")):
                act = _q(out / "jobs" / j / "active_seconds")
                lines.append(f'[ "$(cat {act} 2>/dev/null || echo 0)" -ge 7200 ] '
                             f'&& {{ echo "COOLDOWN $COOLDOWN_S s after {j} $(date +%T)" | tee -a {log}; sleep "$COOLDOWN_S"; }}')
        path = launch / f"{phase}_{qname}.sh"
        _write(path, "\n".join(lines) + "\n")
        return path

    if mode == "local":
        allq = [queue_script(q, js) for q, js in queues.items()] + [queue_script(q, js) for q, js in (then or {}).items()]
        _write(launch / f"{phase}_start.sh", "#!/usr/bin/env bash\nset -e\n" + "\n".join(f"bash {_q(p)}" for p in allq) + "\n")
        return launch / f"{phase}_start.sh"
    first = [queue_script(q, js) for q, js in queues.items()]
    later = [queue_script(q, js) for q, js in (then or {}).items()]
    body = ["#!/usr/bin/env bash", f"# {phase} on {host}: one byobu window per queue in session '{SESSION}'",
            f"byobu has-session -t {SESSION} 2>/dev/null || byobu new-session -d -s {SESSION} -n home"]
    if later:
        chain = (" ; ".join(f"bash {_q(p)}" for p in first) +
                 " ; " + " ; ".join(f"byobu new-window -t {SESSION} -n {p.stem} {_q('bash ' + str(p))}" for p in later))
        body.append(f"byobu new-window -t {SESSION} -n {phase} {_q(chain)}")
    else:
        for p in first:
            body.append(f"byobu new-window -t {SESSION} -n {p.stem} {_q('bash ' + str(p))}")
    body.append(f'echo "started; attach: byobu attach -t {SESSION}"')
    _write(launch / f"{phase}_start.sh", "\n".join(body) + "\n")
    return launch / f"{phase}_start.sh"


# ------------------------------------------------------------------------------------- commands
def cmd_plan(a):
    out = Path(a.out)
    data = Path(a.data).resolve() if a.mode == "scripts" else Path(a.data)
    gpus = [str(g) for g in a.gpus]
    extra = dict(kv.split("=", 1) for kv in a.extra_env)
    t = TIERS[a.tier]
    # Polonez heat rule: training pauses at the first checkpoint after --burst_hours, cools down, resumes
    burst_s = int(a.burst_hours * 3600) if a.mode == "scripts" else 0
    cooldown_s = int(a.cooldown_min * 60) if a.mode == "scripts" else 0
    jobs = []
    if a.phase == "data":
        profile = "smoke" if a.tier == "smoke" else "full"
        tokens = a.data_tokens or (600_000 if profile == "smoke" else TIERS["main"].tokens * 1.05)
        jobs.append(plan_data_job(data, out, tokens, a.num_proc, profile))
        start = write_starter(out, "data", {"data": ["data"]}, a.host, mode=a.mode)
    elif a.phase == "tune":
        queues = {f"gpu{g}": [] for g in gpus}
        k = 0
        for arch in a.arches:
            prior = load_recipe(arch)["lr"][a.tier]
            for mult in (0.5, 1.0, 2.0, 4.0)[: t.tune_runs]:
                name = f"tune_{arch}_lr{prior * mult:.2e}"
                jobs.append(plan_train_job(name, arch, a.tier, lr=prior * mult, tokens=t.tune_tokens, n_gpus=1,
                                           gpu_ids=gpus[k % len(gpus)], data=data, out=out, seed=a.seed, mode=a.mode,
                                           extra=extra, cap_gpu_hours=t.cap_gpu_hours, burst_s=burst_s))
                queues[f"gpu{gpus[k % len(gpus)]}"].append(name)
                k += 1
        start = write_starter(out, "tune", {q: j for q, j in queues.items() if j}, a.host, mode=a.mode,
                              cooldown_s=cooldown_s)
    elif a.phase in ("train", "eval"):
        trains, evals = [], {f"gpu{g}": [] for g in gpus}
        for i, arch in enumerate(a.arches):
            lr = load_recipe(arch)["lr"][a.tier]
            name = f"train_{arch}"
            card = plan_train_job(name, arch, a.tier, lr=lr, tokens=t.tokens, n_gpus=len(gpus), gpu_ids=",".join(gpus),
                                  data=data, out=out, seed=a.seed, mode=a.mode, extra=extra, cap_gpu_hours=t.cap_gpu_hours,
                                  burst_s=burst_s)
            jobs.append(card)
            trains.append(name)
            g = gpus[i % len(gpus)]
            jobs.append(plan_eval_job(f"eval_{arch}", arch, a.tier, name, data=data, out=out, gpu=g, mode=a.mode,
                                      steps=card["steps"]))
            evals[f"gpu{g}"].append(f"eval_{arch}")
        evals = {q: j for q, j in evals.items() if j}
        if a.phase == "train":
            start = write_starter(out, "train", {"train": trains}, a.host, then=evals, mode=a.mode,
                                  cooldown_s=cooldown_s)
        else:
            start = write_starter(out, "eval", evals, a.host, mode=a.mode)
    else:
        raise SystemExit(a.phase)
    plan = {"version": TEXT_CHECKS_VERSION, "phase": a.phase, "tier": a.tier, "arches": a.arches, "host": a.host,
            "gpus": gpus, "data": str(data), "mode": a.mode, "planned": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "jobs": [j["job"] for j in jobs]}
    (out / f"plan_{a.phase}.json").write_text(json.dumps(plan, indent=2))
    for j in jobs:
        if j["kind"] == "train":
            band = "in band" if j["params_in_band"] else "OUT OF BAND"
            print(f"{j['job']:<34} {j['params'] / 1e6:6.1f}M ({band})  lr {j['lr']:.2e}  {j['steps']:,} steps "
                  f"× {j['global_rows']} rows ≈ {j['tokens'] / 1e9:.3g}B tokens  cap {j['cap_gpu_hours']} GPU-h")
        else:
            print(f"{j['job']:<34} {j['kind']}")
    print(f"start: bash {start}")


def cmd_select(a):
    """Pick each model's step size by development loss (never by exam scores) → recipe cards."""
    out = Path(a.out)
    best: dict[str, tuple] = {}
    for jj in sorted((out / "jobs").glob("tune_*/job.json")):
        card = json.loads(jj.read_text())
        res = jj.parent / "result.json"
        if card["tier"] != a.tier or not res.exists():
            continue
        ev = json.loads(res.read_text()).get("eval_loss") or []
        if not ev:
            continue
        loss = ev[-1][1]
        print(f"{card['arch']:<10} lr {card['lr']:.2e}  dev loss {loss:.4f}  ({json.loads(res.read_text())['status']})")
        if card["arch"] not in best or loss < best[card["arch"]][1]:
            best[card["arch"]] = (card["lr"], loss, card["job"])
    from evaluation.text_checks import RECIPE_DIR

    for arch, (lr, loss, job) in best.items():
        path = RECIPE_DIR / f"{arch}.json"
        card = json.loads(path.read_text()) if path.exists() else {"arch": arch}
        card.setdefault("lr", {})[a.tier] = lr
        card.setdefault("tuning", {})[a.tier] = {"chosen_lr": lr, "dev_loss": loss, "job": job, "run": str(out),
                                                 "selected": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        card["status"] = f"tuned at {a.tier}"
        if a.write:
            path.write_text(json.dumps(card, indent=2) + "\n")
        print(f"→ {arch}: lr {lr:.2e} (dev loss {loss:.4f}){'  written to ' + str(path) if a.write else '  (dry run: add --write)'}")


def cmd_status(a):
    out = Path(a.out)
    for jj in sorted((out / "jobs").glob("*/job.json")):
        d = jj.parent
        st = (d / "status").read_text().strip() if (d / "status").exists() else ""
        state = "DONE" if (d / "DONE").exists() else ("running" if (d / "status.run").exists() else "pending")
        act = (d / "active_seconds").read_text().strip() if (d / "active_seconds").exists() else ""
        print(f"{d.name:<34} {state:<8} {st:<12} {('active ' + act + ' s') if act else ''}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--phase", choices=("data", "tune", "train", "eval"), required=True)
    pl.add_argument("--tier", choices=tuple(TIERS), default="screen")
    pl.add_argument("--arches", nargs="+", default=["dense", "local", "e31c"], choices=tuple(ARCHES))
    pl.add_argument("--host", default="polonez")
    pl.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    pl.add_argument("--data", required=True, help="text-checks data dir (manifest, tokenizer, eval/)")
    pl.add_argument("--out", default=None, help="run folder (default: Cache/text_checks/<tier>_<date>)")
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--mode", choices=("scripts", "local"), default="scripts")
    pl.add_argument("--extra_env", nargs="*", default=[], help="KEY=VALUE launcher overrides (recorded in job.json)")
    pl.add_argument("--data_tokens", type=float, default=0, help="data phase: training-mix size in tokens")
    pl.add_argument("--num_proc", type=int, default=16, help="data phase: generator processes")
    pl.add_argument("--burst_hours", type=float, default=6.0,
                    help="pause a training job at its first checkpoint after this many hours (0 = never)")
    pl.add_argument("--cooldown_min", type=float, default=20.0, help="cooldown after a burst or a long job")
    se = sub.add_parser("select")
    se.add_argument("--out", required=True)
    se.add_argument("--tier", choices=tuple(TIERS), default="screen")
    se.add_argument("--write", action="store_true", help="write the chosen step sizes into the recipe cards")
    st = sub.add_parser("status")
    st.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "plan":
        a.out = a.out or f"Cache/text_checks/{a.tier}_{datetime.now():%Y%m%d}"
        Path(a.out).mkdir(parents=True, exist_ok=True)
        cmd_plan(a)
    elif a.cmd == "select":
        cmd_select(a)
    else:
        cmd_status(a)


if __name__ == "__main__":
    main()
