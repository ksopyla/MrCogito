---
name: experiment-run
description: Run, monitor, and debug Concept Encoder training/evaluation on the Odra and Polonez GPU servers. Use when launching remote training, attaching to Byobu, syncing code via Git, reading logs, debugging failed runs, running concept analysis, or collecting artifacts before experiment-track. The single source for the remote environment, folders, env vars, and run/eval commands.
---

# Experiment Run

Execute and babysit ONE approved run on a GPU server. This is the only doc needed to
run training + evaluation and understand the remote environment.

**Boundary:** this skill *runs and monitors* training. It does not pick experiments
(`experiment-design`), interpret/record results (`experiment-track`), or define the
evaluation pipeline / run benchmark sweeps on checkpoints (`experiment-evaluate` — the
single source of truth for *how to evaluate*, with `uv` commands).

## Hard rules
- No training without explicit user permission; ONE experiment per server at a time.
- Sync source by **Git only** — never `scp`/`rsync` code. (`rsync` is only for `Cache/` artifacts.)
- Local macOS (`uv`) = smoke tests only. Real runs are remote on Ubuntu.
- Don't migrate a working remote env mid-run. Don't relaunch a failed run before reading the first traceback.
- Read first: `docs/experiments/<ID>.md` (spec) + `docs/experiments/<ID>_plan.md` (launcher + env knobs).

## Servers (add a new one by appending a row + an `~/.ssh/config` alias)
| alias | CPU / RAM / disk | GPUs | project root | notes |
|---|---|---|---|---|
| `polonez` | AMD Threadripper 3970X, **32 cores / 64 threads**, **256GB RAM**, `/home` NVMe ~1.9TB (check free with `df -h /home`) | 4× RTX 3090 24GB | `/home/ksopyla/dev/MrCogito` | port 2205, Ubuntu 22.04 |
| `odra` | AMD Threadripper 1900X, **8 cores / 16 threads**, **96GB RAM** (check disk with `df -h /home`) | 3× RTX 3090 24GB | `/home/ksopyla/dev/MrCogito` | port 2203, Ubuntu 22.04 |

Env (set by launchers): `HF_HOME=<root>/../hf_home`, `HF_DATASETS_CACHE=$HF_HOME/datasets`,
`TOKENIZERS_PARALLELISM=false`, `CUDA_VISIBLE_DEVICES` = all GPUs.
Artifacts under repo root: `Cache/Training/<run_id>/` (checkpoints), `Cache/logs/` (shell logs),
`Cache/Evaluation_reports/*.csv` (eval), `wandb/`. `run_id = <family>_H..L..C..D.._<date_time>`,
reused as the W&B id/name. Shell logs: `Cache/logs/shell_<family>_<date_time>.log`.

## Server configuration and resource use
- Remote Python env may be `uv` on newer setups or Poetry on older ones. Prefer `uv` where available;
  on Polonez the working env may still be Poetry (`/home/ksopyla/.local/bin/poetry run ...`). Do not
  migrate a working remote env mid-run without user OK.
- **Polonez CPU budget:** use the machine. Large dataset tokenization/preprocessing should use
  **32-48 workers** on Polonez when it is the only experiment running. A hardcoded/default `num_proc=8`
  underuses the 64 hardware threads and stretches preprocessing/barrier time. Keep some headroom for
  DDP ranks, dataloader workers, OS, and SSH/monitoring.
- **GPU memory budget / batch sizing:** do not accept a clearly underfilled GPU for a long/full run.
  Before launching a costly run, do a short calibration on the target model+sequence length:
  start from the planned batch, increase `PER_DEVICE_BATCH_SIZE` until near-OOM, then back off to leave
  ~1-2GB VRAM headroom on RTX 3090s. Use effective batch size as the invariant:
  `per_device_batch_size × num_gpus × gradient_accumulation_steps`. If per-device batch increases,
  lower `GRADIENT_ACCUMULATION_STEPS` when needed to keep optimization comparable. Judge by
  throughput (`samples/sec` / tokens/sec), GPU utilization, and stable memory, not memory usage alone.
  Typical symptoms: ~13GB/24GB with 99% compute may be acceptable but likely leaves batch-size
  throughput on the table; low GPU util with low memory means dataloader/preprocessing is the bottleneck.
- **Odra CPU budget:** much smaller (8C/16T); keep preprocessing/dataloader worker counts modest.
- For full FineWeb-Edu / other large corpora, expect large one-time cache writes. Check:
  `df -h /home`, `du -sh /home/ksopyla/dev/hf_home`, `du -sh /home/ksopyla/dev/MrCogito/Cache`.



## Workflow
**1. Connect + preflight** (one Byobu session per run, survives disconnect):
```bash
ssh <server>
cd /home/ksopyla/dev/MrCogito && byobu new-session -s <ID>   # attach: byobu attach -t <ID>; detach: F6
git fetch origin && git checkout <branch> && git pull --ff-only && git log -1 --oneline
nvidia-smi; df -h .; command -v uv poetry; byobu list-sessions
```
**2. Environment** — use `uv sync` + `uv run …` (the project standard). The eval bash
launchers call `uv run python` (falling back to `python3`); the training launcher uses
`accelerate launch` from the active env. On a legacy Poetry-only server, run `uv sync` to
provision the env; don't migrate a working remote env mid-run without user OK.

**3. Launch training** — one shared launcher, override via env vars (never fork a script):
```bash
DECODER_TYPE=causal_ar HIDDEN_SIZE=768 NUM_LAYERS=8 CONCEPT_NUM=128 \
DATASET_NAME=HuggingFaceFW/fineweb-edu DATASET_SUBSET=sample-10BT \
TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M PER_DEVICE_BATCH_SIZE=8 NUM_EPOCHS=1 \
bash scripts/train_perceiver_denoise_multigpu.sh
```
For new model sizes, first run a **tiny calibration** before the full launch: use a cached/small dataset
or a very short step/epoch budget, sweep `PER_DEVICE_BATCH_SIZE` upward, watch for OOM, then choose the
largest stable value with headroom. Re-run the real command only after the batch/dataloader settings are
known for that model family.
Knobs live at the top of the launcher (`HIDDEN_SIZE`, `TOKEN_EMBEDDING_DIM`, `NUM_LAYERS`,
`CONCEPT_NUM`, `DECODER_*`, `HIDDEN_ACT`, `NORM_TYPE`, `DATASET_*`, `*_BATCH_SIZE`,
`LEARNING_RATE`, `NUM_EPOCHS`, `*_STEPS`, `DATALOADER_NUM_WORKERS`, `DDP_TIMEOUT`,
`SAVE_TOTAL_LIMIT`, `SAVE_SAFETENSORS`, `RESUME_FROM_CHECKPOINT`). For dataset preprocessing, prefer exposing launcher/env knobs for
`train_num_proc`/`test_num_proc` rather than accepting hardcoded `8` on Polonez; target 32-48
preprocess workers for large one-off tokenization there. It auto-detects GPUs, runs `accelerate launch
--multi_gpu --mixed_precision=bf16`, and tees through `scripts/clean_tee.py` to `Cache/logs/`.
Always set a finite checkpoint retention limit for long/full-corpus runs (`SAVE_TOTAL_LIMIT=3–5` is
usually enough) so periodic checkpoints cannot fill `/home`; keep the final saved model separately.

**4. Monitor** (short checks, not blind polling):
```bash
LOG=$(ls -t Cache/logs/shell_*.log | head -1)
rg -n "W&B run:|Train dataset size|loss|eval_loss|Saving model|Traceback|CUDA out of memory|NCCL|nan" "$LOG"
nvidia-smi; ls -lt Cache/Training | head
```
Healthy: GPUs busy, dataset sizes + W&B run logged, loss at `LOGGING_STEPS`, checkpoints at `SAVE_STEPS`,
and old checkpoint directories rotate according to `SAVE_TOTAL_LIMIT`.
Also check memory headroom and throughput early. If VRAM is far below capacity and the run is stable,
consider stopping after the first smoke/calibration window and relaunching with a larger
`PER_DEVICE_BATCH_SIZE` rather than spending a full run underfilled.

**5. Debug** — read around the FIRST error, classify, then fix:
OOM → lower `PER_DEVICE_BATCH_SIZE`/`MAX_SEQ_LENGTH`; NCCL stall → check earlier per-rank failure or
preprocessing barrier or competing GPU process; import error → `uv sync`/`poetry install`; dataset/cache
→ check `HF_HOME`, name/subset, disk, auth; shape/config → fix code locally, test, push, rerun from new commit.

**6. Concept analysis** (fast, run automatically after a checkpoint exists):
```bash
RUN=$(ls -t Cache/Training | head -1)
uv run python analysis/run_concept_analysis.py --model_path "Cache/Training/$RUN" --model_type <family>
```
`--model_type` ∈ `perceiver_denoise|concept_ar|weighted_mlm`. Health probe: `analysis/check_model_health.py`.

**7. Evaluation** (only when spec/plan/user asks). Set `MODEL_PATH_OVERRIDE` + `MODEL_TYPE_OVERRIDE`:
```bash
MODEL_PATH_OVERRIDE="Cache/Training/$RUN" MODEL_TYPE_OVERRIDE=perceiver_denoise \
  bash scripts/evaluate_concept_encoder_glue.sh stsb       # GLUE: all|all-glue|mrpc|stsb|qqp|mnli-matched
# zero-shot first: evaluation/evaluate_on_benchmark.py --benchmark stsb_zero_shot|sick_relatedness|paws|all
```
Recommended order for a fresh checkpoint: concept analysis → `stsb_zero_shot` → GLUE (MRPC/STS-B/QQP/MNLI).
Reports land in `Cache/Evaluation_reports/`.

**8. Sync + handoff** — pull artifacts to local, then hand to `experiment-track`:
```bash
bash scripts/sync_evaluation_reports.sh            # SSH_HOST=odra to target odra; --upload / --two-way / --dry-run
```
Report: server · Byobu session · branch/commit · launch command · `run_id` + W&B URL · shell log path ·
checkpoint path · concept-analysis + eval report paths · status (healthy/stalled/failed/complete) ·
on failure, the first traceback lines and whether retry is safe.
