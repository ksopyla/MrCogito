---
name: experiment-tracking
description: Engineering traceability and experiment tracking discipline for the Concept Encoder project. Full checklists for architecture changes, training runs, git tagging, WandB integration, and CHANGELOG updates. Use after every architecture change, training run, or implementation session.
---

# Experiment Tracking & Engineering Traceability

## Three-Document Traceability Model

```
CHANGELOG.md  (code changes)
    ↕ git tag
master_experiment_log.md  (training runs)
    ↕ run_id
run_reports/  (detailed analysis)
    ↑
active_todos.md  (tasks → completion → CHANGELOG)
```

Every WandB run config must include `git_commit` and `git_tag` to link metrics to code.

## After Architectural Change — Checklist

1. Update `CHANGELOG.md` with dated entry: `## [YYYY-MM-DD] — <title>`, what changed (files, classes), why.
2. Create git tag: `arch/{feature}` (e.g. `arch/recursive-encoder`).
3. Update `active_todos.md` — mark tasks completed with date and commit hash.
4. Verify `get_git_info()` from `training/utils_training.py` is called in training script and passed to `wandb.init(config=...)`.

## Training & Evaluation Workflow

### 1. Before Training
1. Tag: `git tag train/{run_id}_{YYYYMMDD}` and push tags.
2. Confirm training script includes `git_commit` + `git_tag` in WandB config.
3. Add pending row to `docs/2_Experiments_Registry/master_experiment_log.md` with tag, architecture, hyperparameters.

### 2. During Training (on remote server)
1. Monitor training progress: check shell logs (`Cache/logs/shell`), WandB dashboard.
2. Run `analysis/run_concept_analysis.py` on intermediate checkpoints — fast, can run automatically `Cache/Training` directory.
3. Document observations in `docs/` as they emerge.

### 3. After Training Completes
1. Run `analysis/run_concept_analysis.py` on final model — check effective rank, pairwise similarity.
2. If concept health is poor (rank < 10/128), stop here and document the failure.

### 4. Evaluate on Remote Server
1. Run GLUE eval (ViaDecoder): MRPC, STS-B, QQP, MNLI via evaluation bash scripts.
2. Sync eval reports to local: `.\scripts\sync_evaluation_reports.ps1`
   - `-Upload` to push local reports, `-TwoWay` for both directions, `-DryRun` to preview.

### 5. Document Results
1. Update `master_experiment_log.md` — fill in: eval scores, WandB link, git_tag, hardware, final loss.
2. Write run report in `docs/2_Experiments_Registry/run_reports/` if non-trivial.
3. Update `active_todos.md` — close resolved tasks.
4. Update `CHANGELOG.md` if code changed.

### 6. Publish (if promising)
1. If evaluation results are good, upload model to HuggingFace (see huggingface-project skill).

## Git Commit Prefixes

| Prefix | Use for |
|--------|---------|
| `arch:` | New architecture or significant structural change |
| `train:` | Training script changes (hyperparams, data pipeline) |
| `eval:` | Evaluation script or benchmark changes |
| `fix:` | Bug fixes |
| `docs:` | Documentation updates |
| `feat:` | New feature (loss function, metric, utility) |
| `test:` | Unit test additions or fixes |

## Git Tag Conventions

| Pattern | When |
|---------|------|
| `arch/{feature}` | After merging new architecture |
| `train/{run_id}_{date}` | Before starting a training run |
| `eval/{benchmark}_{date}` | Before a major eval sweep |

Push explicitly: `git push origin --tags`

## CHANGELOG Format

```markdown
## [YYYY-MM-DD] — Short Title

**What changed:**
- `nn/file.py`: description of change

**Why:**
- Hypothesis or motivation

**Git tag:** `arch/feature-name`
**Related TODO:** `active_todos.md` → "Task name" (completed)
```

## WandB Integration

Training scripts must call `get_git_info()` and pass to config:
- Required keys: `git_commit`, `git_tag`, `git_branch`
- Project: `"MrCogito"`

## Training Scripts

| Script | Platform | Model |
|--------|----------|-------|
| `scripts/train_weighted_mlm.ps1` | Windows | Weighted MLM |
| `scripts/train_perceiver_mlm.ps1` | Windows | Perceiver MLM |
| `scripts/test_diffusion_local.ps1` | Windows | Diffusion (test) |
| `scripts/test_tsdae_local.ps1` | Windows | TSDAE (test) |
| `scripts/train_mlm_multigpu_perceiver.sh` | Linux | Perceiver MLM (multi-GPU) |
| `scripts/train_recursive_mlm.sh` | Linux | Recursive MLM (multi-GPU) |
| `scripts/train_diffusion_multigpu.sh` | Linux | Diffusion (multi-GPU) |

## Evaluation Scripts

| Script | Platform | Benchmarks |
|--------|----------|------------|
| `scripts/evaluate_concept_encoder_glue.ps1` | Windows | GLUE tasks |
| `scripts/evaluate_concept_encoder_glue.sh` | Linux | GLUE tasks |
| `scripts/evaluate_concept_encoder_sick.sh` | Linux | SICK-Relatedness |
| `scripts/evaluate_concept_encoder_paws.sh` | Linux | PAWS |

Core eval code: `evaluation/evaluate_model_on_glue.py`, `evaluation/evaluate_on_benchmark.py`
