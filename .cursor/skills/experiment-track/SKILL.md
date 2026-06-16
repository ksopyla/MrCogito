---
name: experiment-track
description: Record and interpret completed Concept Encoder training and evaluation results. Use after a run or benchmark finishes to update `docs/2_Experiments_Registry/master_experiment_log.md`, the experiment's spec in `docs/experiments_specs/<ID>.md` (Status + Result), the "what we've explored" learnings in `docs/1_Strategy_and_Plans/agenda.md`, and short reports in `docs/2_Experiments_Registry/run_reports/`. Judge results against the experiment's own success/kill criteria and the current focus, using fair baseline comparisons that consider model size, objective difficulty, data regime, checkpoint maturity, and compute. Not for remote execution, literature review, or choosing the next experiment family.
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
concept-ablation ΔCE + samples → zero-shot STS-B → supervised SICK/PAWS/GLUE), which scripts
cover which aspect, and the exact `uv` commands. This skill consumes the evidence
`experiment-evaluate` produces and records *what it means*. When you need to (re)run any
metric on a checkpoint, switch to `experiment-evaluate`.

Do not use this skill for generic refactors, architecture cleanup, or standalone `CHANGELOG.md` updates. Use `engineering-change-tracking` for code-change traceability.
Do not use this skill to decide which hypothesis to test next. Use `research-synthesis` to choose experiments based on external evidence and project invariants.

## Use This Skill When
- a training run or evaluation sweep finished and needs documentation
- `WandB`, shell logs, `Cache/Evaluation_reports`, or checkpoint metadata need to be turned into a concise summary
- `docs/2_Experiments_Registry/master_experiment_log.md` needs a new row or row update
- the experiment spec `docs/experiments_specs/<ID>.md` needs its `Status` flipped to `done`/`killed` and its `Result` link filled
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
2. the experiment's spec `docs/experiments_specs/<ID>.md` (its hypothesis + success/kill criteria)
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

2. Extract the evidence:
   - concept geometry metrics
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

5. Update the docs:
   - `docs/2_Experiments_Registry/master_experiment_log.md` (the row)
   - the experiment spec `docs/experiments_specs/<ID>.md` (`Status` → done/killed, fill `Result`)
   - the `docs/1_Strategy_and_Plans/agenda.md` "what we've explored" learnings (one line)
   - a short `docs/2_Experiments_Registry/run_reports/<run_name>.md` when needed

## How To Judge Results Like A Project-Aware Researcher
Judge in this order:
1. concept geometry and collapse signals
2. zero-shot semantic signal
3. supervised pair-task results
4. training protocol, compute budget, and checkpoint maturity

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
- `master_experiment_log.md`: keep the row dense and factual. Include the key metric, the fair comparison point, and a one-sentence takeaway.
- `docs/experiments_specs/<ID>.md`: flip `Status` to `done`/`killed` and fill the `Result` block (run id, WandB, run-report link, one-line verdict). Do not paste full results back into the spec.
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
- `master_experiment_log.md` training row `Date`: training start date, usually from the `run_id` or the training `WandB` run start.
- `master_experiment_log.md` evaluation row `Date`: evaluation date, not checkpoint training date.
- `run_reports/<name>_YYYYMMDD.md` filename date: the decisive evaluation or write-up date used for that report.
- Run report `**Date:**`: the same decisive result date used for the report, not necessarily the run start date.
- `docs/experiments_specs/<ID>.md` and `agenda.md`: keep separate dates for separate events such as training-done vs evaluation-done; record them with explicit `YYYY-MM-DD`.

Prefer source order:
1. checkpoint `config.json` and saved metadata
2. `WandB` run config and run URL
3. raw shell log
4. existing linked report or experiment log row

## File-Specific Format Rules

### `docs/2_Experiments_Registry/master_experiment_log.md`
- Do not change the table schema unless the user explicitly asks.
- Add training checkpoints to `## Training Runs`.
- Add standalone evaluation sweeps to `## Evaluation Experiments (Zero Training Cost)`.
- Keep one row per run or evaluation sweep.
- Keep the row compact. Use the existing columns and phrasing style.
- Use backticks for run ids, tags, and checkpoint names.
- Use bold only for the most important metrics or verdict phrases.
- Use `<br>` inside table cells when listing multiple scores or links.
- Use `[Link](...)` for a single `WandB` run URL; use named links only when several evaluation runs belong in the same cell.
- Use `--` for unavailable metric cells and `—` for absent git tags or truly not-applicable entries, matching the current file.
- `Conclusion / Takeaway` should be short, factual, and usually start with a bold status phrase such as `**TODO 10A — MIXED.**` when that context exists.

### `docs/experiments_specs/<ID>.md` (the spec) and `docs/1_Strategy_and_Plans/agenda.md`
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

*Related: `master_experiment_log.md`, `docs/experiments_specs/<ID>.md`, `agenda.md`*
```

## Output Expectations
When using this skill, produce:
- a concise evidence-based summary
- a fair comparison to the nearest relevant baseline
- a project-aware verdict
- precise doc updates, not vague notes

If the user asks for a deep external research follow-up, hand off to `research-synthesis` (which spawns `research-scout` for source material).
