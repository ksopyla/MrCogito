# Training pipeline modularization

- **Type:** engineering refactor, not an `E0NN` experiment.
- **Status:** Stage 0 contract baseline complete (2026-07-11); Stage 1 extraction is next.
- **Serves:** every maintained and revivable training family.
- **Primary goal:** make training code easier to maintain and safer for AI-assisted changes without changing model behavior, data, logging, artifact layout, or historical identity.

## Why this is needed

`training/train_perceiver_denoise.py` is the maintained training hub, but it now owns multiple model families and objectives:

- parallel Perceiver reconstruction,
- concept-conditioned AR reconstruction,
- prefix-to-suffix AR generation,
- the E03 anchor objective,
- E05 windowed decoding,
- E10 pretrained-backbone causal LM.

The filename no longer describes the full responsibility, and the entrypoint also contains argument definitions, validation, model construction, objective-specific Trainer behavior, data routing, collator selection, W&B identity, and final orchestration. The parked recursive and diffusion scripts repeat parts of the same pipeline but are intentionally outside the maintained foundation.

The refactor must therefore proceed in small compatibility-preserving stages. A shorter file is not sufficient if the change silently alters training inputs, logging, checkpoint layout, or W&B lineage.

## Immutable behavior contracts

These contracts are the acceptance criteria for every stage.

### 1. Logging

Every training family keeps the shared logging sequence:

1. initialize distributed execution,
2. configure console and timestamped file logging on the main process,
3. log system, data, loss, model, and training configuration,
4. initialize one W&B run on the main process,
5. finish the W&B run after training.

The shared W&B contract remains:

- project: `MrCogito`,
- stable `group` for experiment/architecture comparison,
- unique timestamped `run_identifier` as W&B id/name and workspace run directory,
- existing `experiment_id`, `model_family`, `objective_family`, `architecture_id`, `checkpoint_family`, and `pretraining_objective` fields,
- dataset, tokenizer, parameter count, git revision, hostname, optimizer, and training arguments remain logged,
- tags remain bounded to W&B's 64-character limit.

Refactoring may move helper implementations, but it must not silently rename historical groups, run ids, family tags, job types, or checkpoint metadata.

### 2. Data processing

The maintained entrypoint keeps all current data sources and their selection priority:

1. `pretokenized_manifest` → load existing Hugging Face `Dataset.save_to_disk()` artifacts without downloading or tokenizing,
2. `dataset_mix_recipe` → resolve and tokenize the configured recipe,
3. `dataset_mix` → resolve and tokenize the registered mix,
4. `dataset_name` / `dataset_name_subset` → load a dataset directly from the Hugging Face Hub.

The following behavior is preserved:

- direct Hub loading passes the configured `dataset_cache_dir` to `datasets.load_dataset`,
- built-in validation/test splits are preferred; otherwise a deterministic seeded holdout is created,
- objective-specific EOS, padding, truncation, and collator behavior is unchanged,
- pretokenized source weights, seed, train interleave, and concatenated evaluation split remain manifest-driven,
- recipe and runtime weight overrides remain supported,
- no data route is made dependent on a specific model family.

### 3. Hugging Face cache roles

Launchers continue to consume the canonical environment variables from `scripts/remote_paths.sh`:

- `HF_HOME` — Hugging Face root,
- `HF_DATASETS_CACHE=$HF_HOME/datasets` — `load_dataset()` cache,
- `DATASETS_TOK_DIR=$HF_HOME/datasets_tok` — pretokenized corpora,
- `DATASETS_RAW_DIR=$HF_HOME/datasets_raw` — transient raw downloads.

Explicit environment overrides continue to win, including tokenizer-specific trees such as E10's Gemma pretokenized cache. Local development continues to use the project `.env` and project-local `HF_HOME` convention; no user-specific absolute local path is added to tracked code.

### 4. Workspace `Cache/` layout

The current repository-relative artifact structure remains valid for local development, verification, analysis, and playground code:

- `Cache/Training/<run_identifier>/` — training runs and checkpoints,
- `Cache/logs/` — shell and Python training logs,
- `Cache/Evaluation_reports/` — evaluation outputs,
- `Cache/hf_home/` — the recommended local Hugging Face root when configured in `.env`.

The refactor does not move or rename these directories. Final-checkpoint layout differences are characterized before any normalization; historical checkpoint paths remain loadable.

### 5. Checkpoints and evaluation

Existing values of `checkpoint_family`, `pretraining_objective`, `evaluation_contract_version`, canonical evaluation modes, and legacy model type strings remain supported. In particular:

- checkpoint family `concept_ar` remains shared by AR reconstruction and prefix-to-suffix AR checkpoints,
- W&B family `concept_ar_prefix` remains the discoverability identity for prefix-to-suffix runs,
- parked diffusion checkpoint families remain loadable by evaluation code,
- removing a training entrypoint must not remove historical checkpoint evaluation support.

### 6. Parked code

Diffusion remains a coherent parked snapshot until an approved experiment revives it. It is not imported into the live training foundation and is not opportunistically modernized during this refactor.

The old recursive MLM implementation is a removal candidate because the current research direction supersedes that specific training path. Its disposition is a later stage and requires a compact historical tombstone; Stage 0 does not delete it.

`weighted_mlm` remains unchanged until an explicit decision determines whether it stays as a reproducible training baseline or becomes checkpoint-evaluation support only.

## Staged implementation

### Stage 0 — characterize current contracts

- Add tests for shared console/file and W&B logging.
- Add tests for run-directory and cache-path conventions.
- Add tests for direct Hugging Face dataset loading and pretokenized-manifest loading.
- Pin dataset-source priority and checkpoint/W&B identity contracts.
- Make no production behavior, path, family, or entrypoint changes.

### Stage 1 — extract neutral modules

- Move objective/argument validation out of the entrypoint.
- Move the custom Trainer implementation to a dedicated module.
- Extract model, collator, data-route, and identity factories.
- Re-export currently imported symbols during migration.
- Keep the existing entrypoint and launcher commands operational.

### Stage 2 — correct active naming

- Introduce `training/train_concept_pretraining.py` as the canonical multi-family entrypoint.
- Keep `training/train_perceiver_denoise.py` as a compatibility wrapper for one migration window.
- Update live scripts, tests, skills, and implementation-path documentation.
- Preserve all checkpoint and W&B identities.

### Stage 3 — clarify launcher roles

- Keep one generic runner and thin `launch_eNN.sh` protocol wrappers.
- Document generic runner versus experiment wrapper versus orchestration pipeline.
- Rename or reorganize launchers only with compatibility shims and reference checks.
- Remove obsolete operational scripts only after confirming they have no users.

### Stage 4 — prune retired training paths

- Resolve the `weighted_mlm` training decision.
- Remove the superseded recursive MLM implementation with a tombstone and preserved git history.
- Keep diffusion parked and revivable.
- Reconcile live docs and skills without rewriting the append-only experiment ledger or historical run reports.

## Verification gates

Each stage must pass:

1. focused unit and characterization tests,
2. the existing training, W&B identity, evaluation routing, and checkpoint-lineage tests,
3. lint diagnostics for touched files,
4. a local no-network smoke where practical,
5. before a remote run, a CUDA launcher smoke that confirms console/file logs, one W&B run, expected data route, and unchanged `Cache/Training` output.

## Non-goals

- No model architecture or loss change.
- No dataset recipe or preprocessing-policy change.
- No checkpoint migration.
- No W&B historical backfill or run renaming.
- No `Cache/` relocation.
- No unparking of diffusion.
- No broad documentation cleanup inside this engineering change; only directly affected live references are updated.
