---
name: experiment-tracking
description: Record and interpret completed Concept Encoder training and evaluation results. Use after a run or benchmark finishes to update `docs/2_Experiments_Registry/master_experiment_log.md`, `docs/1_Strategy_and_Plans/active_todos.md`, and short reports in `docs/2_Experiments_Registry/run_reports/`. Judge results against `docs/1_Strategy_and_Plans/roadmap.md` using fair baseline comparisons that consider model size, objective difficulty, data regime, checkpoint maturity, and compute. Not for remote execution, literature review, or choosing the next experiment family.
---

# Experiment Tracking

## Mission
Use this skill after results already exist. It is for:
- recording what happened,
- interpreting evaluation results in the context of project goals,
- updating the experiment registry and TODO state,
- writing a short run report when the run matters.

This skill focuses on evaluation results. Training details matter only insofar as they explain the evaluation outcome.

## Use This Skill When
- a training run or evaluation sweep finished and needs documentation
- `WandB`, shell logs, `Cache/Evaluation_reports`, or checkpoint metadata need to be turned into a concise summary
- `docs/2_Experiments_Registry/master_experiment_log.md` needs a new row or row update
- `docs/1_Strategy_and_Plans/active_todos.md` needs a status update based on evidence
- a short report should be added to `docs/2_Experiments_Registry/run_reports/`

## Do Not Use This Skill For
- running evaluations, SSH, remote concept analysis, or report syncing. Use `experiment-remote-evaluator`.
- deep literature research, architecture scouting, or paper-backed root-cause analysis. Use the `Researcher` agent.
- deciding the next experiment family, redefining gates, or roadmap-level prioritization before a concrete run exists. Use `research-methodology`.
- code-change traceability or `CHANGELOG.md` updates. Use `engineering-change-tracking`.

## Read This Context First
1. `docs/1_Strategy_and_Plans/roadmap.md`
2. `docs/1_Strategy_and_Plans/active_todos.md`
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

4. Interpret in roadmap context:
   - cite the relevant gate from `roadmap.md`
   - decide whether the run is `promising`, `mixed`, `regression`, or `inconclusive`
   - explain why, not just what the metric was
   - recommend only the immediate next action justified by the evidence

5. Update the docs:
   - `docs/2_Experiments_Registry/master_experiment_log.md`
   - `docs/1_Strategy_and_Plans/active_todos.md`
   - a short `docs/2_Experiments_Registry/run_reports/<run_name>.md` when needed

## How To Judge Results Like A Project-Aware Researcher
Judge in this order:
1. concept geometry and collapse signals
2. zero-shot semantic signal
3. supervised pair-task results
4. training protocol, compute budget, and checkpoint maturity

Use roadmap gates as anchors, not as blind pass/fail switches.

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
- `active_todos.md`: update status, evidence, and the immediate next action for the relevant TODO.
- `run_reports/`: keep reports short unless the user explicitly wants a deeper note.
- Do not open a full research note by default. Use `Researcher` or a separate note only when the result requires deeper literature-backed analysis or a new cross-run diagnosis.

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
- `active_todos.md`: keep separate dates for separate events such as `**Done date:**`, `**Result (YYYY-MM-DD):**`, or `**Evaluation result (YYYY-MM-DD):**`.

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

### `docs/1_Strategy_and_Plans/active_todos.md`
- Preserve the document header style: `**Created:**`, `**Updated:**`, `**Status:**`.
- Keep TODO ids stable. Do not renumber existing TODOs.
- In `## Currently Running`, keep the checkbox style:
  - `- [ ]` for active work
  - `- [x]` with `~~...~~` when the item is completed in that summary list
- For a dedicated TODO section, preserve the current label style:
  - `## TODO N: <title> — <STATUS>`
  - `**Done date:**`
  - `**Why:**`
  - `**Planned experiment:**` or `**Implemented in code:**`
  - `**Result (YYYY-MM-DD):**` or `**Evaluation result (YYYY-MM-DD):**`
  - `**Decision:**`
  - `**Status:**`
- If training completed on one day and evaluation finished later, record both dates separately instead of collapsing them into one.
- When a TODO is updated from evidence, include:
  - the run id
  - the key metrics
  - the decision gate reached
  - the immediate next action

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
<1 short paragraph grounded in roadmap gates and fair baselines>

## Decision
<1 short paragraph with the immediate next action>

*Related: `master_experiment_log.md`, `active_todos.md`*
```

## Output Expectations
When using this skill, produce:
- a concise evidence-based summary
- a fair comparison to the nearest relevant baseline
- a project-aware verdict
- precise doc updates, not vague notes

If the user asks for a deep external research follow-up, hand off to `Researcher`.
