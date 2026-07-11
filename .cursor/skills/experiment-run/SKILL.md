---
name: experiment-run
description: The single source of truth for running and monitoring Concept Encoder training and evaluation on the Odra and Polonez GPU servers. Use when launching remote training, attaching to Byobu, syncing code via Git, setting environment variables and paths, locating the uv training/evaluation scripts, pretokenizing a data mix, reading logs, debugging failed runs, running concept analysis, running evaluation, or collecting artifacts before experiment-track. Owns the full run workflow and the env-var/path conventions; server hardware facts live in remote-servers.
---

# Experiment Run

Execute and babysit ONE approved run on a GPU server. This is the only doc needed to
run training + evaluation and understand the remote run environment end to end.

**Boundary:** this skill *runs and monitors* training and owns the **env-var and path
conventions** the launchers use. It does **not**:
- own server hardware/disk/network facts → `remote-servers`;
- pick experiments → `experiment-design`;
- interpret/record results → `experiment-track`;
- define the evaluation pipeline or run benchmark sweeps on checkpoints →
  `experiment-evaluate` (the single source of truth for *how to evaluate*).

## Hard rules
- No training without explicit user permission; **ONE experiment per server at a time**.
- Sync source by **Git only** — never `scp`/`rsync` code. (`rsync` is only for `Cache/` artifacts.)
- Local macOS (`uv`) = smoke tests only. Real runs are remote on Ubuntu.
- Don't migrate a working remote env mid-run. Don't relaunch a failed run before reading the first traceback.
- Read first: `docs/experiments_specs/<ID>.md` (spec) + `docs/experiments_specs/<ID>_plan.md` (launcher + env knobs).
- Before launch, verify W&B identity (`group`, `job_type`, tags, config `experiment_id`) matches the
  active spec/agenda entry. Do not start a long run with a generic or stale label.

## Paths and environment variables (canonical — both servers)

`scripts/remote_paths.sh` is the **single source of truth** for these variables.
Every training/eval launcher sources it; every Python script consumes the env vars
it exports. Do **not** re-derive HF cache paths inline in launchers or scripts.

```bash
PROJECT_ROOT="/home/ksopyla/dev/MrCogito"

# Hugging Face cache tree (sibling of repo, NOT under Cache/). NVMe.
export HF_HOME="${PROJECT_ROOT}/../hf_home"            # → /home/ksopyla/dev/hf_home
export HF_DATASETS_CACHE="${HF_HOME}/datasets"         # raw load_dataset() cache (HF-managed)
export DATASETS_TOK_DIR="${HF_HOME}/datasets_tok"      # pre-tokenized corpora (pretokenize_mix.py)
export DATASETS_RAW_DIR="${HF_HOME}/datasets_raw"      # transient raw parquet/zst downloads

# Training artifacts (under repo Cache/). NVMe.
OUTPUT_DIR="${PROJECT_ROOT}/Cache/Training"
LOGGING_DIR="${PROJECT_ROOT}/Cache/logs"
SHELL_LOG="${LOGGING_DIR}/shell_<family>_$(date +%Y%m%d_%H%M%S).log"
```

| Variable | Role |
|---|---|
| `HF_HOME` | Root of the HF cache tree; also holds `hub/` (model weights), `metrics/`, `modules/`. |
| `HF_DATASETS_CACHE` (`$HF_HOME/datasets`) | **Raw `load_dataset()` cache**, HF-managed (e.g. `JeanKaddour___minipile`, `HuggingFaceFW___fineweb-edu`). Passed to training via `--dataset_cache_dir`. |
| `DATASETS_TOK_DIR` (`$HF_HOME/datasets_tok`) | **Pre-tokenized corpora** written by `scripts/pretokenize_mix.py`, read by training via the manifest (`load_from_disk`). Passed to pretokenize via `--cache_dir`. |
| `DATASETS_RAW_DIR` (`$HF_HOME/datasets_raw`) | **Transient** raw parquet/zst shards during pretokenize; cleared after each source. Passed to pretokenize via `--raw_dir`. |
| `OUTPUT_DIR` (`Cache/Training`) | Active checkpoints, one subdir per `run_id`. |
| `LOGGING_DIR` (`Cache/logs`) | Shell + training logs. `SHELL_LOG` is each launcher's own log filename. |
| `Cache/Evaluation_reports` | Eval CSV/JSON. |

Other env set by launchers: `TOKENIZERS_PARALLELISM=false`,
`CUDA_VISIBLE_DEVICES` = all GPUs, `PYTORCH_CUDA_ALLOC_CONF`,
`OMP_NUM_THREADS`, `NCCL_DEBUG=WARN`.

**Tokenizer-specific tokenized tree:** a tokenizer switch (e.g. E10's Gemma) gets its
own tree by overriding `DATASETS_TOK_DIR` in that experiment's launcher (e.g.
`datasets_tok_gemma`) — never by adding a new env var name.

**run_id:** `<family>_H..L..C..D.._<date_time>`, reused as the W&B id/name.

## Locating the scripts

All run/eval entrypoints live under `scripts/`:

| Script | Role |
|---|---|
| `scripts/remote_paths.sh` | Sourced by every launcher; sets all path/env vars above + `mkdir`s them. |
| `scripts/train_concept_pretraining_multigpu.sh` | **The generic training launcher.** Auto-detects GPUs, runs `accelerate launch --multi_gpu --mixed_precision=bf16`, owns ALL training defaults + the gated pretokenize phase. Override any knob via env vars (never fork). |
| `scripts/train_perceiver_denoise_multigpu.sh` | Temporary compatibility wrapper for the old generic-launcher path; never add behavior here. |
| `scripts/launch_e05.sh` | E05 wrapper — pins E05 protocol (causal_ar, K=128, seq 2K, mix) and delegates to the generic launcher. |
| `scripts/launch_e10.sh` | E10 wrapper — pins Gemma-3-1B backbone + LoRA + Gemma-tokenized mix, delegates to the generic launcher. |
| `scripts/launch_e10_pipeline.sh` | E10 orchestration pipeline — waits for prerequisites, runs the Stage-0 gate and pretokenization, then invokes `launch_e10.sh`. |
| `scripts/pretokenize_mix.py` | Parallel download + tokenize a mix into `DATASETS_TOK_DIR`, write a manifest training consumes via `--pretokenized_manifest`. |
| `scripts/evaluate_concept_encoder_glue.sh` | GLUE eval (`all\|all-glue\|mrpc\|stsb\|qqp\|mnli-matched\|...`). |
| `scripts/evaluate_concept_encoder_sick.sh` | SICK eval (`sick_relatedness\|sick_entailment\|sick_all`). |
| `scripts/evaluate_concept_encoder_paws.sh` | PAWS eval. |
| `scripts/sync_evaluation_reports.sh` | Pull eval reports to local (`SSH_HOST=odra` to target odra; `--upload`/`--two-way`/`--dry-run`). |
| `scripts/clean_tee.py` | Tee filter the launchers pipe through → `SHELL_LOG` (strips progress spam). |

Parked families (set-aside; revive, don't modify casually): `parked/scripts/train_recursive_mlm.sh`,
`parked/scripts/train_diffusion_multigpu.sh`, `parked/scripts/train_prefix_diffusion_multigpu.sh`.

Python entrypoints: `training/train_concept_pretraining.py` (main;
`training/train_perceiver_denoise.py` is a temporary compatibility wrapper),
`analysis/run_concept_analysis.py`, `analysis/check_model_health.py`,
`evaluation/evaluate_on_benchmark.py`, `evaluation/evaluate_model_on_glue.py`.

## Server configuration and resource use

For hardware specs (CPU cores, RAM, GPU count, disk free) see `remote-servers`. The
**budget guidance below is run-specific** and lives here:

- **Python env:** prefer `uv` (`uv sync` + `uv run …`). On a legacy Poetry-only
  server, run `uv sync` to provision; do not migrate a working remote env mid-run
  without user OK.
- **Polonez CPU budget:** use the machine. Large dataset tokenization/preprocessing
  should use **32–48 workers** on Polonez when it is the only experiment running. A
  hardcoded/default `num_proc=8` underuses the 64 hardware threads and stretches
  preprocessing/barrier time. Keep headroom for DDP ranks, dataloader workers, OS, SSH.
- **Odra CPU budget:** much smaller (8C/16T); keep preprocessing/dataloader worker
  counts modest.
- **GPU memory budget / batch sizing:** do not accept a clearly underfilled GPU for a
  long/full run. Before a costly run, do a short calibration on the target
  model+sequence length: start from the planned batch, increase
  `PER_DEVICE_BATCH_SIZE` until near-OOM, then back off to leave ~1–2GB VRAM headroom
  on RTX 3090s. Use effective batch size as the invariant:
  `per_device_batch_size × num_gpus × gradient_accumulation_steps`. Judge by
  throughput (`samples/sec`), GPU util, and stable memory — not memory usage alone.
- For full FineWeb-Edu / other large corpora, expect large one-time cache writes —
  check disk before launch (see `remote-servers` for the commands).

## Workflow

**1. Connect + preflight** (one Byobu session per run, survives disconnect):
```bash
ssh <server>
cd /home/ksopyla/dev/MrCogito && byobu new-session -s <ID>   # attach: byobu attach -t <ID>; detach: F6
git fetch origin && git checkout <branch> && git pull --ff-only && git log -1 --oneline
nvidia-smi; df -h .; command -v uv poetry; byobu list-sessions
```

**2. Environment** — `uv sync` + `uv run …` (project standard). The eval bash
launchers call `uv run python` (falling back to `python3`); the training launcher uses
`accelerate launch` from the active env.

**3. W&B identity preflight** — confirm the run will be discoverable before any
training starts. Set `EXPERIMENT_ID=<ID>` (or `WANDB_EXPERIMENT_ID=<ID>`) for every
spec-backed run, especially matched controls that otherwise look like a prior baseline.

Dry-check the identity from the exact env vars before launching:
```bash
uv run python - <<'PY'
import os
from training.utils_training import build_perceiver_wandb_identity

def env(name, default):
    return os.environ.get(name, default)

identity = build_perceiver_wandb_identity(
    decoder_type=env("DECODER_TYPE", "perceiver_posonly"),
    objective_variant=env("OBJECTIVE_VARIANT", "reconstruction"),
    hidden_size=int(env("HIDDEN_SIZE", "512")),
    num_hidden_layers=int(env("NUM_LAYERS", "6")),
    concept_num=int(env("CONCEPT_NUM", "128")),
    decoder_num_layers=int(env("DECODER_NUM_LAYERS", "3")),
    checkpoint_family="concept_ar" if env("DECODER_TYPE", "") == "causal_ar" else "perceiver_denoise",
    pretraining_objective="ar_denoising_reconstruction"
        if env("DECODER_TYPE", "") == "causal_ar" else "denoising_full_reconstruction",
    use_bixt=True,
    anchor_loss=env("ANCHOR_LOSS", "false").lower() == "true",
    experiment_id=os.environ.get("WANDB_EXPERIMENT_ID") or os.environ.get("EXPERIMENT_ID"),
)
print("group:", identity.group)
print("job_type:", identity.job_type)
print("tags:", ", ".join(identity.tags))
print("config:", identity.to_config())
PY
```
If the group/job_type/tags are wrong, fix env vars or code first.

**4. (Optional) Pretokenize the data mix** — for long-context / multi-source mixes,
download + tokenize offline first so training loads via `load_from_disk` (instant) and
the DDP ranks don't block on a preprocessing barrier. The generic launcher runs this
phase automatically when `PRETOKENIZE_MIX` is set; you can also run it standalone:
```bash
uv run python scripts/pretokenize_mix.py \
    --mix <mix_id_or_recipe> \
    --tokenizer "$TOKENIZER_NAME" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --cache_dir "$DATASETS_TOK_DIR" \
    --raw_dir "$DATASETS_RAW_DIR" \
    --raw_archive_dir /nas/ml_data/mrcogito/hf_datasets/raw \
    --manifest "$DATASETS_TOK_DIR/<mix>_manifest.json" \
    --objective "$OBJECTIVE_VARIANT" \
    --train_num_proc 32 --test_num_proc 8 --jobs 1
```
Re-run training only (cache warm): `SKIP_PRETOKENIZE=1`.

**5. Launch training** — one shared launcher, override via env vars (never fork a script):
```bash
EXPERIMENT_ID=E03 DECODER_TYPE=causal_ar HIDDEN_SIZE=768 NUM_LAYERS=8 CONCEPT_NUM=128 \
DATASET_NAME=HuggingFaceFW/fineweb-edu DATASET_SUBSET=sample-10BT \
TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M PER_DEVICE_BATCH_SIZE=8 NUM_EPOCHS=1 \
bash scripts/train_concept_pretraining_multigpu.sh
```
For a new model size, run a **tiny calibration** before the full launch: cached/small
dataset or very short step/epoch budget, sweep `PER_DEVICE_BATCH_SIZE` upward, watch
for OOM, then choose the largest stable value with headroom.

Important env knobs: `EXPERIMENT_ID`/`WANDB_EXPERIMENT_ID` plus the launcher knobs
(`HIDDEN_SIZE`, `TOKEN_EMBEDDING_DIM`, `NUM_LAYERS`, `CONCEPT_NUM`, `DECODER_*`,
`HIDDEN_ACT`, `NORM_TYPE`, `DATASET_*`, `PRETOKENIZE_MIX`/`PRETOKENIZED_MANIFEST`,
`*_BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`, `*_STEPS`, `DATALOADER_NUM_WORKERS`,
`TRAIN_NUM_PROC`/`TEST_NUM_PROC`, `DDP_TIMEOUT`, `SAVE_TOTAL_LIMIT`,
`SAVE_SAFETENSORS`, `RESUME_FROM_CHECKPOINT`, `OPTIMIZER`/`MUON_*`, `WEIGHT_DECAY`,
`MAX_GRAD_NORM`, `LR_SCHEDULER_TYPE`).

Always set a finite checkpoint retention limit for long/full-corpus runs
(`SAVE_TOTAL_LIMIT=3–5`) so periodic checkpoints cannot fill `/home`; keep the final
saved model separately.

**6. Monitor** (short checks, not blind polling):
```bash
LOG=$(ls -t Cache/logs/shell_*.log | head -1)
rg -n "W&B group|W&B job_type|W&B run:|Train dataset size|loss|eval_loss|Saving model|Traceback|CUDA out of memory|NCCL|nan" "$LOG"
nvidia-smi; ls -lt Cache/Training | head
```
Healthy: GPUs busy, dataset sizes + W&B run logged, loss at `LOGGING_STEPS`,
checkpoints at `SAVE_STEPS`, old checkpoint dirs rotating per `SAVE_TOTAL_LIMIT`.
Also check VRAM headroom and throughput early — if VRAM is far below capacity and the
run is stable, stop after the smoke/calibration window and relaunch with a larger
`PER_DEVICE_BATCH_SIZE` rather than spending a full run underfilled.

**7. Debug** — read around the FIRST error, classify, then fix:
- OOM → lower `PER_DEVICE_BATCH_SIZE` / `MAX_SEQ_LENGTH`.
- NCCL stall → check earlier per-rank failure, preprocessing barrier, or a competing GPU process.
- Import error → `uv sync` / `poetry install`.
- Dataset/cache → check `HF_HOME`, `HF_DATASETS_CACHE`, `DATASETS_TOK_DIR`, name/subset, disk, HF auth.
- Shape/config → fix code locally, test, push, rerun from new commit.

**8. Concept analysis** (fast, run automatically after a checkpoint exists):
```bash
RUN=$(ls -t Cache/Training | head -1)
uv run python analysis/run_concept_analysis.py --model_path "Cache/Training/$RUN" --model_type <family>
```
`--model_type` ∈ `perceiver_denoise|concept_ar|weighted_mlm`. Health probe: `analysis/check_model_health.py`.

**9. Evaluation** (only when spec/plan/user asks; full eval pipeline owned by
`experiment-evaluate`). Set `MODEL_PATH_OVERRIDE` + `MODEL_TYPE_OVERRIDE`:
```bash
MODEL_PATH_OVERRIDE="Cache/Training/$RUN" MODEL_TYPE_OVERRIDE=perceiver_denoise \
  bash scripts/evaluate_concept_encoder_glue.sh stsb       # GLUE: all|all-glue|mrpc|stsb|qqp|mnli-matched
# zero-shot first: uv run python evaluation/evaluate_on_benchmark.py --benchmark stsb_zero_shot|sick_relatedness|paws|all
```
Recommended order for a fresh checkpoint: concept analysis → `stsb_zero_shot` → GLUE
(MRPC/STS-B/QQP/MNLI). Reports land in `Cache/Evaluation_reports/`.

**10. Sync + handoff** — pull artifacts to local, then hand to `experiment-track`:
```bash
bash scripts/sync_evaluation_reports.sh            # SSH_HOST=odra to target odra; --upload / --two-way / --dry-run
```
Report: server · Byobu session · branch/commit · launch command · `run_id` + W&B URL ·
shell log path · checkpoint path · concept-analysis + eval report paths · status
(healthy/stalled/failed/complete) · on failure, the first traceback lines and whether retry is safe.

## Artifact archive workflow (NVMe → NAS)

Keep NVMe lean: tokenize + train on NVMe canonical paths → rsync **best checkpoint
only** to `/nas/ml_data/mrcogito/checkpoints/<run_id>/` → delete intermediate
checkpoints and stale runs → Polonez `goodwrite_ml` on sdb; Odra/server cold data
under `/nas/ml_data/archive_<host>/` (mirror source paths). **`/nas/ml_data/mrcogito/`
is MrCogito project data only** — not retired users or other projects.

Full cleanup checklist: `docs/engineering_specs/remote_storage_layout_and_cleanup_plan.md`.
Helper: `scripts/finish_goodwrite_symlink.sh` (after goodwrite rsync to sdb completes).
