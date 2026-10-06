---
name: text-checks
description: The text capability checks — the next step after the DNA capability checks. One from-scratch language model per architecture (30M screen, 100M main), trained on the same simple-language corpus (TinyStories/SimpleStories + generated "world documents" with facts and questions) with equal data, compute and parameter budgets and each architecture's own tuned recipe, then examined on T1 Quote → T2 Lookup → T3 Keyed → T4 Latest → T5 Compose (T6 Count, T7 Deduce frontier) at 1k → 128k after training at 4k, with exact-answer scoring, guessing floors, evidence-removed and notebook-off controls, learning curves (tokens to pass) and a scorecard. Use when the user wants to test or compare architectures on text-like data, run or calibrate the text checks, build their data, tune a model's recipe, launch them on Polonez/Odra, check their status, or read/record their results. Not for the DNA/Glyph capability checks (capability-checks), pretrained-checkpoint benchmarks (experiment-evaluate), or designing an architecture (experiment-design).
---

# Text capability checks

Trained from scratch on the same simple-language data, with the same compute and parameter budget, can
this architecture learn language **and** learn to look up, track and combine facts written in it — how
fast, and does it still work on texts far longer than it was trained on?

- **Protocol spec (first draft, being calibrated):** `docs/engineering_specs/text_capability_checks.md`
- **Definitions** (tiers, models, budgets, pass rules): `evaluation/text_checks.py`
- **Recipe cards** (each model's own init / optimizer / step size): `evaluation/text_checks_recipes/<arch>.json`
- **Generator:** `data/text_world.py` · **data builder:** `scripts/build_text_checks_data.py`
- **Runner:** `scripts/run_text_checks.py` · **scorer:** `evaluation/text_checks_eval.py` ·
  **scorecard:** `analysis/text_checks_scorecard.py`
- **Tests:** `tests/test_text_world.py`, `tests/test_text_checks_runner.py`, `tests/test_text_memory.py`
- **Round plan:** `scripts/study_plans/text_r1.py` (experiments X0–X7, their runs, gates, cost)
- **Board (the standard visual summary):** `docs/3_Evaluations_and_Baselines/text_capability_board.html`, drawn by
  `analysis/text_board.py` from the committed ledgers `docs/2_Experiments_Registry/results/capability/text/`;
  publish it as an Artifact after a refresh, always to https://claude.ai/artifact/AUUgBTgGQwBDjvKiLzgXo9 (pass it as `url`)
- **Server checkout:** Polonez `~/dev/MrCogito-text` (its own worktree; other sessions own the others)

**Boundary.** An architecture enters these checks only after passing the DNA **screen** package
(`capability-checks`). Server facts → `remote-servers`; recording → `experiment-track`; how to tell the
author → `research-comms`.

## The checks in one screen

| level | question (end of a story-world document) | answer | floor |
|---|---|---|---|
| T0 Language | held-out story loss (no question) | — | — |
| T1 Quote | "What did the sign of Lumo say?" (4 signs) | 4–6 words | 1/4 |
| T2 Lookup | "Where does Lumo live?" (one home stated) | invented place | ~0 |
| T3 Keyed | same, 16 homes, look-alike names | invented place | 1/16 |
| T4 Latest | "Where does Lumo live now?" after 4 moves | invented place | 1/5 |
| T5 Compose | "Where does the sister of the sister of Lumo live?" | invented place | 1/8 |
| T6 Count (frontier) | "How many times did Lumo visit the market?" | number word | 1/7 |
| T7 Deduce (frontier) | made-up category rules, "Is Pim shiny?" | yes / no | 1/2 |

| tier | params (±5 %) | tokens | train length | global batch | cap (active GPU-h) | use |
|---|---|---|---|---|---|---|
| smoke | 2M | 0.25M | 1k | 8 rows | 0.5 | plumbing only, never a result |
| screen | 30M | 0.6B | 4k | 96 rows | 24 | does it learn; tuning |
| main | 100M | 2B | 4k | 96 rows | 120 | the comparison |

Round-1 models: `dense` (ceiling at 4k), `local` (E31c trained without its notebook — the no-long-memory
control), `e31c` (E31 notebook, text read), `e31c_loop` (+ E33a reread loop). FFN width is fitted per
model so every model lands in the band. Evaluation lengths 1k … 128k; harder and paraphrase splits at 4k
and 16k.

**Pass** = exact answer ≥ 75 % **and** the evidence-removed twin at its floor. A task gates only where
`dense` passes it at 4k (else `uncalibrated`). Frontier = highest of T1–T5 passing at 4k; reach = longest
passing length; tokens to pass = first learning-curve checkpoint that passes; language no-harm = story loss
within 2 % of `dense`.

## The process

### 0 · Preconditions (local, minutes)
- The model is registered in `ARCHES` (`evaluation/text_checks.py`) as trainer arguments on the shared
  trainer (no fork) — every argument must have an env var in `ENV_OF` and in the launcher.
- It has a recipe card; it passed the DNA screen.
- `uv run pytest -q tests/test_text_world.py tests/test_text_checks_runner.py tests/test_text_memory.py`
- Local plumbing check (MPS/CPU, ~5 min) — the whole loop on 2M models:
  ```bash
  uv run python scripts/run_text_checks.py plan --phase data --tier smoke --mode local \
      --data Cache/text_checks/smoke --out Cache/text_checks/smoke_runner
  bash Cache/text_checks/smoke_runner/launch/data_start.sh
  uv run python scripts/run_text_checks.py plan --phase train --tier smoke --mode local \
      --arches dense local e31c e31c_loop --data Cache/text_checks/smoke --out Cache/text_checks/smoke_runner
  bash Cache/text_checks/smoke_runner/launch/train_start.sh
  uv run python analysis/text_checks_scorecard.py --in_dir Cache/text_checks/smoke_runner
  ```
  Expect every job `EXIT … 0` and all scores 0 (2M models learn the language, not the tasks).

### 1 · Sync and check the server (git only; ask before using a busy server)
Push the branch; on the server `git fetch && git checkout <branch> && git pull --ff-only`. Ask the
`server-checker` agent which GPUs are free. Odra/Polonez may be held by other studies: **ask the author
before starting**, and wake Polonez only with the author's go-ahead. Polonez runs in bursts under 10 h;
every job here resumes.

### 2 · Build the data once (CPU, ~1 h with 16 processes)
`TOK` = the server's tokenized-data folder (`../hf_home/datasets_tok`, see `remote-servers`).
```bash
ssh polonez 'cd ~/dev/MrCogito && export PATH="$HOME/.local/bin:$PATH" && \
  uv run python scripts/run_text_checks.py plan --phase data --tier main --host polonez \
    --data $TOK/text_checks_v0 --out Cache/text_checks/data_v0 --num_proc 16 && \
  bash Cache/text_checks/data_v0/launch/data_start.sh'
```
One build (sized for the main tier, ~2.1B tokens) serves every tier: smaller tiers read the first part of
the same row stream. It writes the tokenizer, the two-source manifest, the frozen `eval/*.jsonl` and
`text_checks_meta.json` (hashes). Never edit it; a change is a new data version and a new folder.

### 3 · First GPU run: the server smoke (1 GPU, minutes)
Checks the real launcher, flex attention and the notebook kernels on CUDA before spending a day:
build `--tier smoke` data into its own folder (step 2 with `--tier smoke`), then
`plan --phase train --tier smoke --gpus 0 …` and run it. Any crash here is a bug to fix first.

### 4 · Tune each model's recipe (screen: 4 × 1-GPU runs per model in parallel)
```bash
uv run python scripts/run_text_checks.py plan --phase tune --tier screen --arches dense local e31c \
    --host polonez --gpus 0 1 2 3 --data $TOK/text_checks_v0 --out Cache/text_checks/r1_screen
bash Cache/text_checks/r1_screen/launch/tune_start.sh
uv run python scripts/run_text_checks.py select --out Cache/text_checks/r1_screen --tier screen --write
```
`select` picks each model's step size by **development loss only** (never by exam scores) and writes it
into the recipe card with the tuning record. **Commit the cards** (git), pull them on the server, then
train. Every model gets the same number of tuning runs and tokens. Changing a model's initialization is
allowed only through its card, before its scored run.

### 5 · Train and evaluate
```bash
uv run python scripts/run_text_checks.py plan --phase train --tier screen --arches dense local e31c \
    --host polonez --gpus 0 1 2 3 --data $TOK/text_checks_v0 --out Cache/text_checks/r1_screen
bash Cache/text_checks/r1_screen/launch/train_start.sh     # trains one by one on all GPUs, then evals 1 GPU each
uv run python scripts/run_text_checks.py status --out Cache/text_checks/r1_screen
```
- Byobu session `textchk`: window `train`, then `train_gpuN` evaluation windows.
- Re-running a starter skips DONE jobs; a training job resumes from its newest checkpoint.
- Active time is counted across resumes. At the cap the run stops and is scored at its newest
  checkpoint, labeled `over_budget` — slowness is a result, not a reason to extend.
- Odra: `--gpus 0 1 2` (same global batch through accumulation; the cap is in GPU-hours).
- Evaluation scores the final model at all lengths with controls, the notebook-off twin and the
  learning-curve checkpoints (10/25/50/75 %), then deletes optimizer states.
- Gate to the main tier: T2 passes at 4k and story loss is within 5 % of `dense`.

### 6 · Pull, draw, record, report
```bash
bash scripts/pull_text_checks_results.sh polonez r1_smoke r1_screen   # ledgers + NAS archive + board redraw
```
- Pull after every phase, at least daily during a long run (Polonez can shut down for heat), and before
  any summary. Pulling is idempotent; a live run shows its state, loss and throughput on the board.
- Commit the ledgers and the board, then publish the board as an Artifact (same URL).
- The NAS archive skips intermediate checkpoints (`/nas/ml_data/mrcogito/results/text_checks/<run>.<host>/`).
- A single run's scorecard without the board: `uv run python analysis/text_checks_scorecard.py --in_dir <run>`.
- Hand the result to `experiment-track`.
- Tell the author (`research-comms`):
  - which levels each model passes at 4k and how far each reaches;
  - tokens to pass;
  - language loss vs `dense`;
  - any shortcut or over-budget flag;
  - what the calibration changed.

## Calibration (the protocol is a first draft)
- **First runs are calibration.** Expect to adjust the token budgets and caps, the mix share, task dials
  and tier shapes after the dense model's screen run.
- **Every change goes into the spec**, the definitions and a new data version, recorded in the spec's
  calibration notes.
- **Results before and after a change are compared only with that change stated.**
- **Open calibration questions:**
  - whether dense passes T1–T4 at 30M / 0.6B tokens;
  - whether E31's 16-token local window (as tested on DNA) is right for language;
  - real throughput vs the caps;
  - eval time at 128k.

## Pitfalls
- **Tuning on exam scores**: never. Development loss only; the exams are the test set.
- **Different data per model**: every model must read the same build in the same order (same data folder,
  seed, global batch). Steps are computed from the build's mean row length.
- **Over-band model**: fix it in `ARCHES` / the tier shape, not by hand-editing a job script.
- **Hand-edited job scripts** make a different run. Re-plan instead, and record overrides with `--extra_env`.
- **An expired Hugging Face login** makes public datasets look missing. The builder reads them
  anonymously; do not "fix" it by logging in.
- **Disk**: 20 checkpoints per run. Evaluation deletes optimizer states. A main-tier run still keeps
  about 20 × 0.4 GB of weights.
- **Uncalibrated is not failed**: if dense misses a task at 4k, the task gates nothing.
