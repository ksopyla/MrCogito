---
name: capability-checks
description: The one process for testing whether our own architectures can LEARN to carry, address, discriminate, hold, compose and reason over information — one leveled series of synthetic tasks (C0 Carry → C1 Address → C2 Discriminate → C3 Hold many → C4 Compose → C5 Reason → C6 Aggregate → C7 Language-like), always trained from scratch, with length (read to 128k) and model size (5M–50M) as separate dials, first-letter scoring, seeds 0–2, the dense ceiling, flawed tasks flagged, the no-harm comparison against the champion (E31 latent memory), the committed results ledger + NAS archive, and the capability dashboard (the standard visual summary). Use when the user wants to test, exam, screen, benchmark or compare a new or changed architecture or variant, asks for "full capability checks", whether it keeps past capabilities or is worth scaling, asks how the checks are organized or what a task measures, asks where past capability / suite / ladder / E30 / E31 / E31b / E33 results are, wants them synced or re-scored, or asks for a visual summary / comparison of capability results. Not for evaluating trained text checkpoints on STS-B/GLUE/lm-eval (experiment-evaluate), generic training runs (experiment-run), or designing the architecture (experiment-design).
---

# Capability checks

Can **our own architecture learn** to pick, hold, compose and reason over information? Every variant
takes the same leveled tasks, trained from scratch; the results land in the same committed ledger and
are compared on the same dashboard.

- **Process spec (structure, rules, migration, storage):** `docs/engineering_specs/capability_checks.md`
- **Task definitions (source of truth):** `evaluation/capability_tasks.py` — levels, every task with its
  recipe, length, written curriculum, chance, what it measures, status, and the map from old results
- **Engine:** `data/bapo_ladder.py` (task generators) + `verification/bapo_capability_probe.py` (train +
  score one task) + `verification/length_ladder.py` (read a trained model at longer lengths)

**Boundary.** Server hardware → `remote-servers`; generic launches, env vars, Byobu → `experiment-run`;
trained text checkpoints → `experiment-evaluate`; what a result means and the registry →
`experiment-track`; how to tell the author → `research-comms`.

## The structure in one screen

```
 LEVEL                                         × LENGTH (train, then read to 128k)  × SIZE (10M screen, 30M compare, 50M trend)
 C0 Carry         copy a span
 C1 Address       look up one fact
 C2 Discriminate  the fact among 1 → 8 decoys
 C3 Hold many     recall 1 of 8 → 16 facts
 C4 Compose       in-order chain, 4 → 8 hops
 C5 Reason        parallel chains, 2 → 4 hops   ← research frontier
 C6 Aggregate     the fact that appears once, count
 C7 Language-like the same skills in Markov "text" filler (C8 real text later)
 Flawed, never evidence: shuffled chains without decoy chains · triple-match · majority
```

| package | levels | size · seeds | ladder | use |
|---|---|---|---|---|
| screen | C0–C2 | 10M · 0 | no | does it learn at all |
| **core** | C0–C5 | 30M · 0, 1, 2 | to 128k | the full check every variant gets; no-harm uses it |
| frontier | C5–C6 | 30M · 0, 1, 2 | yes | the current research question |
| language | C7 | 30M · 0, 1, 2 | no | plausible filler |

## Rules (author's decisions, 2026-10-04) — never bend them

1. **From scratch only.** Random init, our own architecture. Never start a task from another run's
   checkpoint, an earlier experiment's weights or a pretrained model.
2. **In-run curriculum only when written** in the task's `curriculum` field, from random init, identical
   for every architecture and for the dense ceiling. No ad-hoc warm starts.
3. **Score on the first answer letter**; pass = median over seeds 0, 1, 2 ≥ 75 %; chance 25 % (DNA),
   12.5 % (Glyph). The dense model trains alongside as the ceiling; a miss dense shares is "uncalibrated".
4. **Flawed tasks** (`status="flawed"`) are never run as evidence, never gate, never cited as a
   capability; the runner and scorecard label them, the dashboard crosses them out.
5. **`calibrating` tasks** have no frozen from-scratch recipe yet: report them, never gate on them.

## The process

### 1 · Plan the checks in the experiment spec (before code)
A new variant gets the **core package** (no past capability may be lost) and the no-harm comparison
against the champion `e31_li_m1`: lost = more than 5 points below the champion on any task and length
where the champion passes. Write the tasks, seeds and expected cost into the spec. Use **one variant id
everywhere** (suite arch name = battery `{tag}_{arm}`, e.g. `e33a_loop`).

**Until the v4 runner lands** (spec "Migration", step 4), run with the v3 runners and label the results
as legacy protocol: the suite (`suite.md`: C0–C2, C4.chain4-1k, C6.unique-256, C7 — from scratch) and the
length battery (`battery.md`: C1.lookup-16k curriculum + ladders; its chains and hard exams start from the
run's own lookup weights, so they map as `curriculum-differs`).

### 2 · Register
- Arch in `evaluation/bapo_models.py` (`ARCHES` + `build_model`), config-selectable on the shared
  foundation; arch-only flags in `ARCH_FLAGS` (`evaluation/capability_suite.py`).
- Battery: one entry in `BATTERY_VARIANTS` (`scripts/study_plans/e30_vs_e31.py`); never copy a job list.
- `uv run pytest -q tests/test_capability_tasks.py tests/test_capability_suite.py tests/test_bapo_probe_tiers.py` passes.

### 3 · Smoke, sync, launch
Local smoke only if it finishes in under a minute; anything longer runs on a server. Code goes by git
only. Check the GPUs are free and respect the Polonez heat rule (long queues on Odra; Polonez < 10 h
bursts, and a 10–20 min cooldown after every 5–6 h of training). Commands: `suite.md` steps 2–5,
`battery.md` "Running it". Several bursts on Polonez: generate each run's launch folder (`--mode scripts`),
then chain them with `scripts/run_bursts.sh --host polonez --cooldown_min 20 <launch dir>...` in byobu.

**Length ladder on suite runs.** Add `--extra "--save_ckpt @out"` (the final weights of a from-scratch
run, saved only to be read longer; never a starting point). Then write read-only GPU scripts with
`bash scripts/ladder_suite_run.sh --host polonez --gpus "0 1 2 3" --launch Cache/capability/<ladders> <run dir>...`
(dense capped at 32k) and run them like a burst. The ledger picks up each job's `ladder.json`; a seed
re-run with weights replaces the older run of that seed on the board.

### 4 · Monitor
Suite: `DONE` count and `EXIT` lines (`suite.md` step 6). Battery: `analysis/study_table.py --prefix
<tag>_ --first` on the server. Hand long watches to the `server-checker` agent.

### 5 · Pull into the ledger and archive to the NAS — after every phase, daily on long studies
```bash
bash scripts/pull_capability_results.sh odra ~/dev/MrCogito/Cache/capability/<run>
bash scripts/pull_capability_results.sh odra ~/dev/MrCogito/Cache/study/e30_vs_e31
```
- Writes `docs/2_Experiments_Registry/results/capability/{suite,study}/<name>.<host>.json` — compact text
  only, refused above 1 MB. **Commit it**: the ledger is the source of truth. No checkpoints, logs or raw
  JSON ever enter the repo.
- Archives the raw folder to `/nas/ml_data/mrcogito/results/{capability,study}/<name>.<host>/` (additive).
- Other report folders on a server: `bash scripts/archive_reports_to_nas.sh <host>`.
- Not on W&B. List the ledger: `uv run python analysis/capability_ledger.py summary`.

### 6 · Score and compare
- Suite verdict for one run from the ledger: `uv run python analysis/capability_scorecard.py --in_dir
  <ledger files…> --out_dir Cache/capability/<run>_scored` (reading it: `suite.md` step 7).
- Map any past result onto v4: `legacy_suite(cell)` / `legacy_battery(variant, slot)` in
  `capability_tasks.py` give the task and the match label (`same`, `settings-differ`,
  `curriculum-differs`, `not-from-scratch`, `flawed`). Only `same` counts as v4 evidence; the rest is
  shown and marked.

### 7 · The dashboard — the standard visual summary
```bash
uv run python analysis/capability_board.py      # → docs/3_Evaluations_and_Baselines/capability_board.html
```
It also rewrites the re-run list `docs/3_Evaluations_and_Baselines/capability_reruns.md` (what must run, from
scratch, before the table is complete; flawed tasks never). The page: the no-harm verdict per model (`same`
evidence only), the task table by level with a status per row (active, partial, missing, calibrating,
flawed) and a protocol badge per legacy cell, the length charts by level, the task guide and the re-run list. **When the author asks for a visual
summary**, regenerate from a fresh pull and publish to the existing artifact
https://claude.ai/artifact/FNTv1zKLc6fh9QRTrJmVrR (pass it as `url` from a new session). Narrative pages
may sit on top, but their numbers come from the ledger.

### 8 · Record
Hand to `experiment-track`: the no-harm verdict and lost list, per-level results with match labels,
the length headline, and the ledger file names. Past reports that cite capability numbers get a **dated
re-scoring note** above the original numbers (never overwrite them). Report to the author per
`research-comms`: outcome first, plain words, dashboard link.

## Maintaining the checks
- A task changes (recipe, length, budget, curriculum, rows) → a new task id or a new `VERSION`.
- A new task runs first on dense (and the champion) to calibrate, then becomes `active` in a level.
- A shortcut is found → `status="flawed"` with the reason; never delete the task.
- Legacy runners: suite v3 (`SUITE_VERSION`) and battery v1 (`BATTERY_VERSION`) stay frozen as history.
- A new champion → `--champion` in `analysis/capability_board.py` and the spec line.

## Pitfalls
- **A warm start from another run** is not a capability result — label it `not-from-scratch`.
- **Mean over letters** overstates multi-candidate tasks; judge on first letter.
- **Same name, different protocol** — always read the match label before comparing two numbers. A suite
  cell re-run at another step size (`--lr_scale`, `--lr_pair`) is `settings-differ`, and two models are
  compared only at the same step size (2026-10-04: e33a lookup-2k at 1e-4 vs E31 at 5e-5 looked like a loss).
- **Results only on a server** — pull after every phase; Polonez can shut down for heat mid-study.
- **A copied job list** drifts from the protocol; register the variant instead.
- **Hand-edited commands** produce a different exam; use `ARCH_FLAGS`, `BATTERY_VARIANTS` or a written
  curriculum.
- **Uncalibrated ≠ failed**, **one seed is a screen**, **step-size cliffs** — see `suite.md`.
