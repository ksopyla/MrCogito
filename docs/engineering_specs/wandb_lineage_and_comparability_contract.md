# W&B Lineage + Comparability Contract — engineering implementation plan

- **Type:** engineering (tracking/instrumentation), not an `E0NN` experiment
- **Status:** planned (opened 2026-06-20)
- **Owner:** Krzysztof Sopyla
- **Serves:** reliable train-checkpoint-eval traceability and architecture/objective comparability in W&B

## Problem statement (evidence from latest 30 runs)

Snapshot analyzed: **latest 30 runs in `ksopyla/MrCogito`**.

Observed structure:
- **30 total**: 5 train, 19 eval/benchmark, 6 pretraining.
- **Group coverage**: 13/30 have `group`, **17/30 missing**; for eval specifically, **17/19 missing**.
- **Eval linkage quality**:
  - 13/19 can be linked only by parsing `config.model_path` (`Cache/Training/<train_run>/checkpoint-<step>`).
  - 6/19 have explicit training run lineage (`source_run_id` / `source_training_run_id`).
  - In current runs, linkage is usually possible but often not first-class in UI filters.
- **Inconsistent identity semantics**:
  - `source_run_id` means **checkpoint id** in some runs (e.g. `checkpoint-66000`) and **training run id** in others.
  - One train run has a metadata mismatch (`group` points to E03 while config fields indicate E01).
- **Retrieval friction in W&B UI**:
  - Evals are primarily discoverable by run name string parsing, not durable lineage keys.
  - Comparing AR vs reconstruction vs diffusion under same size/config is cumbersome due mixed tag formats.

## Grill-me decisions (resolved)

1. **Strict lineage contract** for new eval runs: fail fast when source lineage cannot be resolved.
2. **Eval group = source training group** (same `group` value).
3. **Kill criterion**: linkage coverage must be **100%** for new eval runs in CI/smoke gates.
4. For checkpoint identity, log both:
   - `source_checkpoint_step`
   - `source_checkpoint_epoch`
5. **External checkpoint policy**: strict remains default; unlinked eval only via explicit override flag.
6. **Comparability preference**:
   - compare by family + params,
   - tokenizer included in tags (not hard-blocked by default).
7. **Namespaced tags** should be introduced (with minimal legacy aliases).
8. **Backfill scope**: recent history (target: last ~60 days), not full archive.
9. **Rollout**: 3 stages (tests/validator -> train+eval smoke -> default-on).

## Target contract (new metadata standard)

### 1) Training run identity (already partially present, to harden)

Required on every train run:
- `experiment_id`
- `model_family`
- `objective_family`
- `architecture_id`
- `wandb_group`
- `wandb_job_type`
- `model/num_parameters` (or `total_params`) normalized for filtering
- tokenizer identity (`tokenizer_name`)

Rule: `group`, config `wandb_group`, and experiment tag must be consistent or run init fails.

### 2) Eval lineage identity (new strict contract)

Required on every eval run:
- `lineage_schema_version` (start at `1`)
- `source_training_run_id` (canonical W&B parent id/name)
- `source_training_group`
- `source_training_experiment_id` (if available)
- `source_checkpoint_path`
- `source_checkpoint_step` (int)
- `source_checkpoint_epoch` (float, when derivable)
- `lineage_status` (`linked` or `unlinked`)

Rules:
- Default behavior: require `lineage_status=linked`; otherwise fail.
- Override behavior: explicit `--allow_unlinked_eval` permits run with `lineage_status=unlinked`.
- `group` for eval run must equal `source_training_group` when linked.

### 3) Namespaced tags (new canonical filter surface)

Canonical tags:
- `exp:E0NN` (when known)
- `family:<model_family>`
- `objective:<objective_family>`
- `job:<train|eval>`
- `benchmark:<name>` (eval only)
- `size:<XM>`
- `tokenizer:<tokenizer_id_short>`
- `ckpt_step:<N>`
- `ckpt_epoch:<E>`
- `lineage:<linked|unlinked>`

Keep small backward-compat aliases temporarily (e.g. `beyond-glue`, `concept_ar`) during migration.

## Implementation plan (repo-rooted)

### Step 1 — shared lineage/identity helpers

Files:
- `training/utils_training.py`
- new `evaluation/wandb_identity.py` (or equivalent helper module)

Implement:
- parsing helpers for checkpoint path -> `{training_run_id, checkpoint_step}`
- optional W&B API resolver for parent run metadata
- strict validator for required lineage fields
- namespaced tag builder shared by benchmark + GLUE eval scripts

### Step 2 — enforce strict eval lineage in benchmark evaluation

File:
- `evaluation/evaluate_on_benchmark.py`

Changes:
- replace ambiguous `source_run_id` usage with canonical lineage fields
- resolve parent training run and inherit `group`
- log `source_checkpoint_step` + `source_checkpoint_epoch`
- add `--allow_unlinked_eval` explicit override
- keep backward-compatible legacy keys for one migration window

### Step 3 — align GLUE evaluation with the same contract

File:
- `evaluation/evaluate_model_on_glue.py`

Changes:
- reuse shared lineage helper (not ad-hoc parsing)
- switch to namespaced tags and canonical lineage field names
- set eval `group` to resolved source training group
- preserve existing useful metadata (`architecture_tag`, urls) but map to canonical fields

### Step 4 — harden training identity consistency

Files:
- `training/train_concept_pretraining.py`
- `training/train_perceiver_denoise.py` (temporary compatibility wrapper)
- `training/utils_training.py`
- `tests/test_wandb_identity.py`

Changes:
- add consistency guard at init time:
  - `group == wandb_group config == experiment tag prefix`
- fail fast when overridden env/args produce contradictory identity fields
- keep current identity derivation as single source of truth

### Step 5 — comparability projection fields

Files:
- `training/utils_training.py`
- `evaluation/evaluate_on_benchmark.py`
- `evaluation/evaluate_model_on_glue.py`

Add normalized fields for cohort filtering/comparison:
- `compare_family`
- `compare_params_m`
- `compare_objective`
- `compare_tokenizer`
- `compare_architecture` (optional, for stricter slices)

### Step 6 — recent-history backfill (60-day window)

Files:
- new `scripts/backfill_wandb_lineage.py`

Behavior:
- scan recent runs
- infer and write missing canonical lineage fields/tags where confidence is high
- mark uncertain rows `lineage_status=unlinked`
- emit CSV report of patched vs unresolved runs

Safety:
- backfill is additive (no destructive rewrite of old metrics)
- include dry-run mode before update mode

### Step 7 — staged rollout gates

Gate A (local/unit):
- unit tests for identity builders and lineage parser
- validator tests for required fields

Gate B (smoke):
- one train run + multi-benchmark eval on one checkpoint
- verify 100% linked lineage and same-group inheritance

Gate C (default-on):
- strict mode enabled by default in both eval scripts
- override remains available for external checkpoints

## Validation and falsification

Primary success gate (falsifiable):
- On new eval runs in CI/smoke, **linked lineage coverage must be 100%**.

Secondary acceptance:
- For the latest-N dashboard view, train->checkpoint->eval retrieval should be possible using only:
  - `group`
  - namespaced tags
  - canonical lineage fields
without manual run-name parsing.

Failure conditions (kill/rework):
- any new eval run missing canonical lineage fields in strict mode
- conflicting train identity fields (`group` vs config vs tags)
- inability to filter same family+size cohorts cleanly via tags/fields

## Non-goals

- No model architecture/training-objective change.
- No benchmark metric definition changes.
- No deletion/rewrite of historical experiment records; only additive metadata backfill in scoped window.
