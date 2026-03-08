---
name: experiment-tracking
description: Record and maintain Concept Encoder training and evaluation metadata. Use when preparing a run, creating train or eval git tags, monitoring training, syncing evaluation reports, updating docs/2_Experiments_Registry/master_experiment_log.md, writing run reports, or checking WandB and git linkage. Not for changelog updates or deciding experiment priorities.
---

# Experiment Tracking

## Scope
Use this skill for experiment metadata and results:
- training runs
- evaluation sweeps
- WandB linkage
- benchmark summaries
- run reports

Do not use this skill for generic refactors, architecture cleanup, or standalone `CHANGELOG.md` updates. Use `engineering-change-tracking` for code-change traceability.
Do not use this skill to decide which hypothesis to test next. Use `research-methodology` for experiment selection and interpretation.

## Required linkage
Each tracked run should capture:
- `run_id`
- `git_commit`, `git_tag`, `git_branch`
- script or config used
- checkpoint or output path
- WandB URL or run name
- key metrics and short conclusion

Training scripts should call `get_git_info()` from `training/utils_training.py` and pass the returned values into `wandb.init(config=...)`.

## Before training
1. Create a tag: `train/{run_id}_{YYYYMMDD}`.
2. Verify WandB config includes `git_commit`, `git_tag`, and `git_branch`.
3. Add a pending entry to `docs/2_Experiments_Registry/master_experiment_log.md`.
4. Record the hypothesis, architecture variant, dataset, and important hyperparameters.

## During training
1. Monitor logs and WandB.
2. Run `analysis/run_concept_analysis.py` on intermediate checkpoints when useful.
3. Capture notable failures, anomalies, or early signals while they are fresh.

## After training
1. Run final concept analysis.
2. If concept health is poor, document the failure before spending on a broad benchmark sweep.
3. Record final loss, hardware, checkpoint path, and a one-line outcome in `docs/2_Experiments_Registry/master_experiment_log.md`.
4. Write a report in `docs/2_Experiments_Registry/run_reports/` when the run is non-trivial or teaches something important.

## After evaluation
1. For standalone benchmark sweeps, tag them as `eval/{benchmark}_{YYYYMMDD}`.
2. Run the relevant evaluation scripts and sync results with `scripts/sync_evaluation_reports.ps1` when needed.
3. Update `docs/2_Experiments_Registry/master_experiment_log.md` with scores, report links, and WandB reference.
4. Add a short diagnosis note if the results contradict the hypothesis.
5. Update `docs/4_Research_Notes/` when the run reveals a new failure mode or research insight.

## Minimum summary
When summarizing a run, include:
- goal or hypothesis
- exact model or config variant
- main metrics
- concept health outcome
- next decision: continue, modify, or stop

## Related skills
- Use `engineering-change-tracking` when the task is about code refactors, architecture edits, `CHANGELOG.md`, or direction shifts.
- Use `huggingface-project` when a promising checkpoint is ready for upload.
