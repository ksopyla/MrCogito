---
name: experiment-track
description: Record and interpret completed Concept Encoder training and evaluation results. Use after a run or benchmark finishes to update `docs/2_Experiments_Registry/master_experiment_log.md`, the experiment's spec in `docs/experiments_specs/ahead/<ID>.md` (Status + Result), the "what we've explored" learnings in `docs/1_Strategy_and_Plans/agenda.md`, and short reports in `docs/2_Experiments_Registry/run_reports/`. Judge results against the experiment's own success/kill criteria and the current focus, using fair baseline comparisons that consider model size, objective difficulty, data regime, checkpoint maturity, and compute. Not for remote execution, literature review, or choosing the next experiment family.
---

# Experiment Track

## Mission
Use this skill after results already exist. It is for:
- recording what happened,
- interpreting evaluation results in the context of project goals,
- updating the experiment registry, the experiment spec status, and the agenda ledger,
- writing a short run report when the run matters.

This skill focuses on evaluation results. Training details matter only insofar as they explain the evaluation outcome.

**Running the evaluations and the script pipeline are owned by `experiment-evaluate`** — the
single source of truth for *how to evaluate* (tiered health → concept geometry + AR
concept-ablation ΔCE + samples → generation vibe-check metrics → zero-shot STS-B →
supervised SICK/PAWS/GLUE), which scripts cover which aspect, and the exact `uv` commands.
This skill consumes the evidence `experiment-evaluate` produces and records *what it means*.
When you need to (re)run any metric on a checkpoint, switch to `experiment-evaluate`.

Do not use this skill for generic refactors, architecture cleanup, or standalone `CHANGELOG.md` updates. Use `engineering-change-tracking` for code-change traceability.
Do not use this skill to decide which hypothesis to test next. Use `research-synthesis` to choose experiments based on external evidence and project invariants.

## Use This Skill When
- a training run or evaluation sweep finished and needs documentation
- `WandB`, shell logs, `Cache/Evaluation_reports`, or checkpoint metadata need to be turned into a concise summary
- `docs/2_Experiments_Registry/master_experiment_log.md` needs a new row or row update
- the experiment spec `docs/experiments_specs/ahead/<ID>.md` needs its `Status` and `Result` filled before moving the spec/plan pair to its terminal lifecycle folder
- the `docs/1_Strategy_and_Plans/agenda.md` "what we've explored" learnings need a one-line update
- a short report should be added to `docs/2_Experiments_Registry/run_reports/`

## Do Not Use This Skill For
- remote training, SSH, Byobu sessions, log debugging, or report syncing. Use `experiment-run`.
- running evaluations / benchmark sweeps on checkpoints, or the question "which eval script covers which aspect, and how do I run it with uv?". Use `experiment-evaluate` (the single source of truth for the evaluation pipeline).
- deep literature research, architecture scouting, or paper-backed root-cause analysis. Use the `research-synthesis` skill (which spawns the `research-scout` agent for source fetching).
- deciding the next experiment family, redefining gates, or roadmap-level prioritization before a concrete run exists. Use `research-synthesis`.
- code-change traceability or `CHANGELOG.md` updates. Use `engineering-change-tracking`.
- pruning, deduplicating, or archiving stale TODOs/reports once a track is closed. Flag the drift and hand off to `docs-hygiene`.

## Read This Context First
Skip `docs/5_Archive/` and any `> **OBSOLETE — ...**` section — that content is historical, not current truth (see `project-overview.mdc` → Docs Hygiene). Read:
1. `docs/1_Strategy_and_Plans/agenda.md` (current focus + "what we've explored" learnings)
2. the experiment's spec `docs/experiments_specs/ahead/<ID>.md` (its hypothesis + success/kill criteria)
3. `docs/2_Experiments_Registry/master_experiment_log.md`
4. the relevant existing run report, if one exists
5. available evidence:
   - `WandB`
   - raw shell log
   - `Cache/Evaluation_reports/*.json`
   - `Cache/Evaluation_reports/*.csv`
   - checkpoint `config.json`

## Core Workflow
1. Reconstruct the run facts:
   - run id
   - model family and size
   - checkpoint path
   - dataset
   - machine
   - objective and important training protocol details
   - epochs or global steps
   - evaluation route and benchmark coverage
   - compute: `compute/gpu_hours`, `compute/energy_kwh`, `compute/max_tokens`, `compute/loss_tokens_est` + `compute/audit_state`/`compute/flag` (from the run's W&B summary, written by the compute audit — see `experiment-evaluate` run-level preamble). **Hard precondition:** if the summary lacks `compute/audit_state`, the audit was never run — and the absence is silent (no `compute/*` keys at all, not even `failed`). Stop and run `uv run python analysis/run_compute_audit.py --run-id <run_id>` (macOS-runnable, seconds) before recording; do not hand-estimate compute and proceed.

2. Extract the evidence:
   - concept geometry metrics
   - generation vibe-check metrics (`distinct_1` / `rep_3` at length cutoffs, one sample
     snippet; from `*_generation_quality.json` when present)
   - zero-shot semantic metrics
   - supervised benchmark results
   - training stability signals
   - missing or partial measurements

3. Compare fairly:
   - same objective family before cross-family claims
   - same or similar parameter regime before size claims
   - same data regime before dataset claims
   - same evaluation route before task-score claims
   - same checkpoint maturity before convergence claims

4. Interpret in agenda + spec context:
   - judge against the experiment spec's own success/kill criteria and the current `agenda.md` topic
   - decide whether the run is `promising`, `mixed`, `regression`, or `inconclusive`
   - explain why, not just what the metric was
   - recommend only the immediate next action justified by the evidence

5. Update the docs (in this order):
   - `docs/2_Experiments_Registry/master_experiment_log.md`:
     1. **Experiment Index** — upsert the ID row (Recent closed / Ahead / Canceled) with spec + report links
     2. **Training Runs** — append one lean row (or Evaluation Experiments for zero-train evals)
     3. **Focus note** — only when the result changes current priority
     4. **Run reports** list — add the new report near the top
   - the experiment spec `docs/experiments_specs/ahead/<ID>.md` (`Status` → done/killed, fill `Result`, then move the pair)
   - the `docs/1_Strategy_and_Plans/agenda.md` "what we've explored" learnings (one line)
   - a short `docs/2_Experiments_Registry/run_reports/<run_name>.md` when needed

6. Assign the terminal lifecycle from the experiment's own registered question:
   - `done_success/` — the decisive success criterion passed, or a control produced the
     pre-registered decisive answer it was designed to provide
   - `done_failed/` — the run completed or hit its kill gate without establishing the proposed
     mechanism; mixed partial positives stay here when the decisive criterion failed
   - `canceled/` is not a result verdict; use it only for an explicitly rejected, superseded, or
     abandoned spec that did not produce a completed experiment
   Draft cancellation is normally owned by `docs-hygiene`; a run that hits a kill gate belongs
   in `done_failed/`, never `canceled/`.
   Move the spec and `_plan.md` together, then repair links:
   - search the repository for `experiments_specs.*<ID>` and links to both filenames
   - update `agenda.md`, run reports, CHANGELOG entries being touched, cross-spec links, code
     comments, and catalog metadata
   - fix outbound `../` depth inside the moved files
   - verify all non-placeholder Markdown links resolve

## How To Judge Results Like A Project-Aware Researcher
Judge in this order:
1. concept geometry and collapse signals
2. zero-shot semantic signal
3. supervised pair-task results
4. training protocol, compute budget, and checkpoint maturity

For the compute-budget factor, prefer the audited `compute/*` scalars (GPU-hours,
energy kWh, max tokens) over hand-estimates, and note `compute/audit_state`: a
`flagged` or `running-partial` number is approximate (read `compute/flag`). Raw
GPU-hours/energy are meaningful within a `wandb_group` (matched setup); across
regimes use the derived ratios (`compute/tokens_per_gpu_hour`,
`compute/energy_per_billion_tokens`).

Use the experiment spec's success/kill criteria as anchors, not as blind pass/fail switches.

Do not reject a run only because one headline number looks average. Consider:
- model size and depth
- objective difficulty
- dataset difficulty and cleanliness
- warm-start vs random init
- number of steps or epochs
- partial vs full evaluation coverage
- training stability
- whether the result improves the nearest fair baseline

Useful interpretation rules:
- Better geometry with near-random STS-B usually means diversity without semantics.
- Strong zero-shot STS-B can justify one targeted follow-up even when supervised fine-tuning is still weak.
- Good supervised scores with collapsed geometry can still be useful, but they do not clear SG1 concept-quality goals.
- A harder objective or smaller model can make a mediocre absolute score informative rather than negative.
- Repeated same-family runs with rank stuck below `~10 / 128` and no semantic lift are evidence that the track is stalling.
- Partial spot-checks should produce partial conclusions, not broad track verdicts.

## Documentation Rules
- `master_experiment_log.md` is an **index**, not a lab notebook. Specs hold intent/criteria; run reports hold deep metrics. Never paste multi-paragraph analysis into log cells.
- Always keep **Experiment Index** and **Training Runs** in sync for the same ID (index = one row per ID; training = one row per run).
- Every E-numbered row must link to its lifecycle spec path (`../experiments_specs/<lifecycle>/<ID>_….md`). Search all lifecycle folders — never assume `ahead/`.
- `docs/experiments_specs/<lifecycle>/<ID>.md`: flip `Status` to `done`/`killed` and fill the `Result` block (run id, WandB, run-report link, one-line verdict). Do not paste full results back into the spec. Then move the spec and plan together to `done_success/` or `done_failed/`.
- `agenda.md`: add/update a one-line "what we've explored" entry (neutral, evidence-based — not a verdict) and move the experiment off Current focus.
- `run_reports/`: keep reports short unless the user explicitly wants a deeper note.
- Do not open a full research note by default. Use `research-synthesis` (which can spawn `research-scout` for source fetching) or a separate note only when the result requires deeper literature-backed analysis or a new cross-run diagnosis.

## Dates And Linkage Discipline
- Use exact dates from the evidence, not the current day unless the action really happened today.
- Use `YYYY-MM-DD` inside documents.
- Keep the full timestamped `run_id` exactly as recorded.
- Copy `git_commit`, `git_tag`, `checkpoint_path`, and `WandB` URLs from the run metadata, `config.json`, shell log, or `WandB`. Do not guess them from the current repo state.
- If sources disagree, say so explicitly in the note or report instead of silently normalizing them.

Use these date rules:
- `master_experiment_log.md` Training Runs `Date`: training start date, usually from the `run_id` or the training `WandB` run start.
- `master_experiment_log.md` Evaluation Experiments `Date`: evaluation date, not checkpoint training date.
- `master_experiment_log.md` Experiment Index has no date column; order Recent closed newest-first and rely on linked reports for dates.
- `run_reports/<name>_YYYYMMDD.md` filename date: the decisive evaluation or write-up date used for that report.
- Run report `**Date:**`: the same decisive result date used for the report, not necessarily the run start date.
- `docs/experiments_specs/<lifecycle>/<ID>.md` and `agenda.md`: keep separate dates for
  separate events such as training-done vs evaluation-done; record them with explicit `YYYY-MM-DD`.

Prefer source order:
1. checkpoint `config.json` and saved metadata
2. `WandB` run config and run URL
3. raw shell log
4. existing linked report or experiment log row

## File-Specific Format Rules

### `docs/2_Experiments_Registry/master_experiment_log.md`

File layout (do not invent new top-level sections without user ask):

1. Purpose + “Need → Where” routing table
2. Focus note (blockquote) — only when priority changes
3. Protocol notes (blockquote) — rare, durable caveats
4. `## Experiment Index` — primary human/agent navigation
5. `## Training Runs` — append-only chronological ledger
6. `## Evaluation Experiments (Zero Training Cost)`
7. `## Architecture notes (pointers only)` — short links, no essays
8. `## Run reports` — newest-first link list

#### Experiment Index (required on every track)
- One row per experiment **ID** (not per run). Multiple arms of one ID share one index row; point at the decisive report.
- Subsections: `### Recent closed (…)`, `### Ahead / open`, `### Canceled (no run)`.
- Newest closed IDs at the **top** of Recent closed.
- Columns: `ID | What | Lifecycle | Key result | Spec · report` (Ahead/Canceled may drop Lifecycle / Key result).
- `What` = short title (no hypothesis essay). `Key result` = one metric phrase + outcome, not a paragraph.
- Spec link must use the **current** lifecycle folder after any move (`done_success/` / `done_failed/` / `canceled/` / `ahead/`).
- When closing an ID: move it from Ahead → Recent closed; update lifecycle in the link; add/refresh the report link.
- Keep the one-line **Genealogy** pointer in sync when the series advances.

#### Training Runs (append-only)
- Columns (fixed schema): `Date | Exp | Run ID | Setup | Key metrics | Verdict | Links`
- `Date` = training start date (from `run_id` / W&B start).
- `Exp` = experiment ID (`E16b`) or `—` for pre-ID historical runs.
- `Run ID` in backticks; one row per training run (append at the **bottom**).
- `Setup` = family · size · objective · data/budget in ~8–12 words.
- `Key metrics` = 2–4 headline numbers only (rank/RankMe, decisive Δ, STS-B, loss).
- `Verdict` = **one short sentence** (status word + why). No multi-sentence essays.
- `Links` = `spec` (if E-numbered) · `report` (if exists) · `W&B`, joined with ` · `.
- Never start a row with `||` or `|||` — every data row starts with exactly one `|`.
- Do **not** reintroduce the old 14-column schema (Epochs / Concept Losses / Task Loss / Eff. Rank / GLUE / Speed / Git Tag as separate columns). Those details belong in the run report.

#### Evaluation Experiments
- Columns: `Date | Eval | Source | Key scores | Verdict | Links`
- `Date` = evaluation date, not training start.
- One row per eval sweep; link spec/report when applicable.

#### Style
- Backticks for run ids and checkpoint names.
- Bold only for decisive metrics or status words.
- Prefer ` · ` separators over `<br>` in cells (keep cells single-line when possible).
- Use `—` for not-applicable Exp / missing links.

### `docs/experiments_specs/<lifecycle>/<ID>.md` (the spec) and `docs/1_Strategy_and_Plans/agenda.md`
- Do NOT rewrite the frozen spec body (hypothesis, builds-on, single change, criteria). Only set `Status` and fill the `Result` block.
- `Result` block: run id, WandB link, run-report path, and a one-line verdict (`promising`/`mixed`/`regression`/`killed`) judged against that spec's own success/kill criteria.
- Keep experiment IDs stable; never reuse an ID.
- In `agenda.md`, move the experiment from `Current focus` to the "what we've explored so far" list as one neutral line: id + title + the decisive metric + what we learned + pointer. Avoid "best"/"killed"; keep it tentative.
- If training completed on one day and evaluation finished later, record both dates separately instead of collapsing them into one.

### `docs/2_Experiments_Registry/run_reports/`
- Keep filenames descriptive and date-stamped: `<descriptive_slug>_YYYYMMDD.md`.
- Preserve the current header block order whenever possible:
  - title with human-readable experiment name and run id
  - `**Date:**`
  - `**Machine:**`
  - `**Run ID:**`
  - `**WandB:**` or `**WandB (training):**`
  - `**Raw log:**` or `**Raw shell log:**`
  - `**Best checkpoint:**`
  - `**Git commit:**`
  - `**Git tag:**`
  - `**Related TODO:**`
- If several benchmark `WandB` runs exist, keep the training `WandB` link in the metadata block and list benchmark-specific `WandB` links in `## Evaluation`.
- Preserve the current section order unless the run truly does not need one:
  - `## Goal`
  - `## Configuration`
  - `## Training Outcome`
  - `## Concept Health`
  - `## Evaluation`
  - `## Interpretation`
  - `## Decision`
  - optional `## Notes`
- End with a short related-files line such as `*Related: ...*`.

## Run Report Template
Use this structure for non-trivial runs and keep it close to the current reports:

```markdown
# <human-readable title> — `<run_id>`

**Date:** <YYYY-MM-DD>
**Machine:** <machine>
**Run ID:** `<run_id>`
**WandB:** <link> or **WandB (training):** <link>
**Raw log:** `<raw_log_path>` or **Raw shell log:** `<raw_log_path>`
**Best checkpoint:** `<checkpoint_path>`
**Git commit:** `<git_commit>`
**Git tag:** `<git_tag>` or `—`
**Related TODO:** `<todo id>`

---

## Goal
<1 short paragraph>

## Configuration
| Item | Value |
|---|---|
| Family | ... |
| Encoder | ... |
| Decoder | ... |
| Dataset | ... |
| Objective | ... |
| Epochs | ... |
| Effective batch | ... |
| Throughput | ... |
| Compute | GPU-h, kWh, max-tokens (`compute/*` from the compute audit; note `compute/audit_state`) |

## Training Outcome
<short paragraph>

## Concept Health
<table or concise bullets>

## Evaluation
<zero-shot and supervised subsections when applicable>

## Interpretation
<1 short paragraph grounded in the experiment's success/kill criteria and fair baselines>

## Decision
<1 short paragraph with the immediate next action>

*Related: `master_experiment_log.md`, `docs/experiments_specs/<lifecycle>/<ID>.md`, `agenda.md`*
```

## Output Expectations
When using this skill, produce:
- a concise evidence-based summary
- a fair comparison to the nearest relevant baseline
- a project-aware verdict
- precise doc updates, not vague notes

If the user asks for a deep external research follow-up, hand off to `research-synthesis` (which spawns `research-scout` for source material).
