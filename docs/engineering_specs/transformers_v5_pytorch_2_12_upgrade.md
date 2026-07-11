# Transformers v5 + PyTorch 2.12 upgrade — engineering spec + implementation plan

- **Type:** engineering (dependency + interface migration). **Not** an `E0NN` experiment — no model-architecture, objective, or benchmark-metric change. Pure dependency bump + the code fixes the bumps force.
- **Status:** spec open (2026-06-29), **not started**. Implementation is deliberately deferred until the **E05 attempt-3 training run** finishes; do not start while a training run is in flight on Odra/Polonez.
- **Owner:** Krzysztof Sopyla
- **Serves:** keeping the foundation on supported library versions. transformers v5 is the first major in five years and the project is already pinned `>=4.47,<5`; torch 2.10 is two minors behind the current 2.12. Both pin ceilings and several removed APIs will block us within weeks (transformers weekly minors, torch cu128 deprecation).

## What we upgrade (target versions)

| Dependency | Current pin | Target pin | Notes |
|---|---|---|---|
| `transformers` | `>=4.47.1,<5` | `>=5.10,<6` (cap at next major; weekly minors) | v5.12.1 is current; 5.10.x is what vLLM syncs to, a useful stability anchor. |
| `huggingface-hub` | `>=0.24.0` | `>=1.0.0` | Hard requirement of transformers v5. |
| `torch` | `>=2.10.0,<3` | `>=2.12,<3` | 2.12.1 is current; minor bump, low-risk. |
| `torchvision` | `>=0.24.0` | `>=0.27` (match torch 2.12) | Move in lockstep with torch. |
| `accelerate` | `>=1.2.1,<2` | keep, but verify v5-compat (≥1.3 expected) | transformers v5 Trainer changes; accelerate must match. |
| `liger-kernel` | `>=0.5.0; sys_platform == 'linux'` | bump to latest 0.5.x after a smoke test | Re-validates against torch 2.12 / transformers v5. |
| `hf-transfer` | `>=0.1.9` | **replace with `hf-xet>=1.4`** | transformers v5 + hub 1.0 drop `hf_transfer` / `HF_HUB_ENABLE_HF_TRANSFER`. |

CUDA index: keep `pytorch-cu128` for now (2.12 still publishes cu128 wheels, just deprecated). A follow-up note moves to `cu130` — **out of scope** for this change to avoid coupling the dep bump to a driver/CUDA upgrade on both servers.

## Why now (and why deferred)

- transformers v5 ships a **minor every week**. Staying on `<5` means every weekly release is a release we cannot consume, including bug fixes for models we touch (ModernBERT, SmolLM2/3, Qwen2/3, Llama 3.x).
- Several v5 removals **directly hit our launchers and training glue** (see inventory below). The longer we stay on 4.x, the more code we write against an API that is already gone upstream.
- torch 2.12 is low-risk and brings a free ~100× win for `analysis/run_concept_analysis.py` (`torch.linalg.eigh` on CUDA → cuSolver backend). No reason to defer it independently.
- **Deferred until E05 attempt-3 finishes** because the upgrade touches `scripts/train_concept_pretraining_multigpu.sh` and `training/utils_training.py` — the exact files an in-flight run depends on. Mid-run edits there are a real source of "why did this checkpoint sequence break" confusion.

## What breaks today (concrete inventory against this codebase)

This is the result of a codebase-wide scan against the v5 migration guide and the torch 2.12 release notes. **Every item below is confirmed present in the repo** (no speculative bullets).

### Hard breaks — v5 (launch / training glue)

1. **`TrainingArguments.overwrite_output_dir` removed (no deprecation).** Passed as `--overwrite_output_dir True` in:
   - `scripts/train_concept_pretraining_multigpu.sh` (the main launcher, used by E05)
   - `parked/scripts/train_diffusion_multigpu.sh:205`
   - `parked/scripts/train_prefix_diffusion_multigpu.sh:202`
   - `verification/verify_prefix_diffusion_wikitext_smoke.py:79`
2. **`TrainingArguments.logging_dir` removed (no deprecation; use `TENSORBOARD_LOGGING_DIR` env var).** Passed as `--logging_dir` in:
   - `scripts/train_concept_pretraining_multigpu.sh`
   - `parked/scripts/train_diffusion_multigpu.sh:192`
   - `parked/scripts/train_prefix_diffusion_multigpu.sh:189`
   - `verification/verify_prefix_diffusion_wikitext_smoke.py:31, 71`
   - `evaluation/evaluate_model_on_glue.py:1134` (constructed in Python, not via CLI)
   - `training/utils_training.py:445–451` (mutates `training_args.logging_dir` post-parse — the attribute will be gone)
3. **`TrainingArguments.warmup_ratio` folded into `warmup_step`** (which now also accepts a float). We pass `--warmup_steps` everywhere (integer), so we are compatible — but verify the alias still accepts an int.

### Hard breaks — v5 (GLUE eval script)

4. **`TrainingArguments.no_cuda` removed → use `use_cpu`.** `evaluation/evaluate_model_on_glue.py:252, 919, 1685` parses a `--no_cuda` arg and threads it through. Rename to `--use_cpu` and flip polarity, or drop the custom flag and read `TrainingArguments.use_cpu` directly.

### Hard breaks — v5 (tokenization / API surface)

5. **`additional_special_tokens` → `extra_special_tokens`.** `tokenization/train_tokenizer_custom.py:385` passes `additional_special_tokens=[...]` to `PreTrainedTokenizerFast`. Auto-converted with a deprecation warning today; rename to silence and future-proof.
6. **`tokenizer.encode_plus(...)` deprecated → `tokenizer(...)`.** `tests/test_data_collators.py:137`. Mechanical rewrite.
7. **`apply_chat_template` now returns `BatchEncoding`, not bare `input_ids`.** Used heavily in `playground/Qwen/*` and `playground/load_llama32.py`. All current call sites pass `tokenize=False` (so they get text back) **except** `playground/Qwen/explore_qwen_omni.py:327`, which passes the result into `processor(...)`. Audit each call site; playground breakage is low-priority but should not silently regress.

### Hard breaks — v5 (custom `PreTrainedModel` subclasses)

8. **Default `_init_weights` now auto-applied to subclasses.** Our custom models subclass `PreTrainedModel` and may rely on bespoke init:
   - `nn/concept_encoder.py`, `nn/concept_encoder_perceiver.py`, `nn/concept_encoder_weighted.py`
   - `parked/nn/concept_encoder_diffusion.py`
   
   Per the migration guide, v5 will silently re-initialise any `nn.Parameter` / `nn.Linear` / `nn.Embedding` it finds unless the subclass overrides `_init_weights`. **Action:** add an explicit `_init_weights` to each (even a no-op `pass` if the `__init__` does its own init), so the v5 default does not clobber trained-from-scratch weights.

### Hard breaks — v5 (hub / transfer)

9. **`hf_transfer` / `HF_HUB_ENABLE_HF_TRANSFER` dropped in favour of `hf_xet`.**
   - `pyproject.toml:32` declares `hf-transfer>=0.1.9` — replace with `hf-xet>=1.4`.
   - `.env.example:11` sets `HF_HUB_ENABLE_HF_TRANSFER=1` — remove.
   - `.env` (local, gitignored) likely mirrors `.env.example` — remove there too.
10. **`huggingface-hub` pinned to `>=1.0.0`** by transformers v5. Our pin is `>=0.24.0`; bump.
11. **50 GB default shard size** (was 5 GB). Behavioural, not a break, but worth knowing: `save_pretrained` will write fewer, larger shards. Affects `Cache/Training/` layout and any script that counts shards.

### Hard breaks — torch 2.12 (minor)

12. **`pytorch-cu128` index deprecated.** Still publishes wheels in 2.12; will stop eventually. Out of scope for this change (deferred to a follow-up CUDA 13 driver-upgrade task). Document only.
13. **`torch.distributed.nn.functional.*` raises under `torch.compile`.** We do not compile over distributed ops — no action, but note for any future `torch.compile(model, fullgraph=True)` work.
14. **`torchrun` default port is now OS-assigned, not 29500.** We use `accelerate launch`, not raw `torchrun`; `scripts/bench_seq_parallel.py:14` uses `torchrun --standalone` (unaffected — standalone pins the port). No action.

## What does NOT break (already compliant)

Confirmed by scan — these are the v5 removals that would have bitten a typical project but do not bite us:

- `Trainer(tokenizer=...)` → already using `processing_class=tokenizer` everywhere (`training/train_concept_pretraining.py`, `parked/training/*`, `parked/scripts/*`).
- `use_auth_token` — not used anywhere.
- `load_in_4bit` / `load_in_8bit` — not used (we use `quantization_config` patterns or none).
- `AutoModelWithLMHead`, `AutoModelForVision2Seq` — not imported.
- TF / Flax model classes — none imported.
- `as_target_tokenizer`, `prepare_seq2seq_batch`, `special_tokens_map_extended`, `create_token_type_ids_from_sequences`, `sanitize_special_tokens` — none used.
- `torchscript`, `torch.fx` — not used in any model code.
- `from_xxx_config` config helpers — not used.
- `warmup_ratio`, `per_gpu_*`, `use_mps_device`, `fp16_backend`, `half_precision_backend`, `include_inputs_for_metrics`, `include_tokens_for_second`, `use_legacy_prediction_loop`, `tpu_num_cores`, `tpu_metrics_debug`, `push_to_hub_token`, `fsdp_min_num_params`, `fsdp_transformer_layer_cls_to_wrap`, `jit_mode_eval`, `past_index`, `ray_scope`, `mp_parameters` — none passed to `TrainingArguments` anywhere.

This is a **shorter break list than a typical v5 migration**. The work is mostly launcher glue + the `_init_weights` audit + the env/dep swap.

## Target design

### A. Dependency changes (`pyproject.toml`)

```toml
# runtime deps — target block
"datasets>=4.0.0",
"huggingface-hub>=1.0.0",
"torch>=2.12,<3",
"torchvision>=0.27",
"transformers>=5.10,<6",
"accelerate>=1.3.1,<2",
"hf-xet>=1.4",                              # replaces hf-transfer
# remove: "hf-transfer>=0.1.9",
"liger-kernel>=0.5.0; sys_platform == 'linux'",
# keep [tool.uv.sources] pytorch-cu128 index for now (deferred cu130 move)
```

Rationale for `transformers>=5.10` (not `>=5.0`): 5.10.x is the version vLLM syncs to, and 5.12.1 is current — pinning to `>=5.10` lets us consume the stable mid-line plus patches without forcing every weekly minor.

### B. Launcher / glue changes

**`scripts/train_concept_pretraining_multigpu.sh`** (and the three `parked/scripts/*` mirrors):
- Remove `--overwrite_output_dir True` (v5 removed; `resume_from_checkpoint` covers the resume case).
- Remove `--logging_dir "$LOGGING_DIR"` and add `export TENSORBOARD_LOGGING_DIR="$LOGGING_DIR"` near the top of the script (before the `accelerate launch` call).
- Keep `--warmup_steps` (int form, still accepted).

**`training/utils_training.py`** (`_configure_training_output_paths` or equivalent, lines ~438–451):
- Stop mutating `training_args.logging_dir`. Either:
  - read `TENSORBOARD_LOGGING_DIR` from env and join with `run_identifier` to compute the path the function currently builds, then `os.environ["TENSORBOARD_LOGGING_DIR"] = computed` before Trainer construction, **or**
  - drop the post-parse logging-dir logic entirely and rely on the launcher to set the env var.
- Preference: keep the run-identifier suffixing logic (it namespaces TB logs per run) but write it to the env var, not the removed attribute.

**`verification/verify_prefix_diffusion_wikitext_smoke.py`:**
- Remove `--overwrite_output_dir True` and the `--logging_dir` arg; set `TENSORBOARD_LOGGING_DIR` in-process via `os.environ` before constructing `TrainingArguments`.

**`evaluation/evaluate_model_on_glue.py`:**
- Line 1134: replace `logging_dir=...` kwarg with `os.environ["TENSORBOARD_LOGGING_DIR"] = ...` set before `TrainingArguments(...)`.
- Lines 252, 919, 1685: rename `--no_cuda` to `--use_cpu` (flip polarity: `args.use_cpu` instead of `not args.no_cuda`), or drop the custom flag and read `training_args.use_cpu` directly (cleaner — fewer custom args).

### C. Tokenizer / test changes

**`tokenization/train_tokenizer_custom.py:385`:**
```python
# v4
additional_special_tokens=[t for t in special_tokens if t not in ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]]
# v5
extra_special_tokens=[t for t in special_tokens if t not in ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]]
```

**`tests/test_data_collators.py:137`:**
```python
# v4
encoding = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
# v5
encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
```

### D. Custom-model `_init_weights` audit

For each `PreTrainedModel` subclass under `nn/` and `parked/nn/`:
1. Inspect `__init__` to see whether it does its own parameter init (custom `nn.Parameter`, bespoke scaling, etc.).
2. If it does, add an explicit override that preserves that behaviour:
   ```python
   @torch.no_grad()
   def _init_weights(self, module):
       # v5 would otherwise auto-apply its default scheme here; we keep our bespoke init.
       pass  # or: the actual init logic, moved out of __init__ if it was inline
   ```
3. If it relies on the parent default, leave as-is (v5's default is reasonable and the migration guide shows it).
4. The audit must confirm init is **identical before/after** for a fresh `model = ConceptEncoder(config)` — verified by comparing `state_dict()` hashes on a fixed seed in the test (see validation).

### E. Env / config changes

- `.env.example`: remove the `HF_HUB_ENABLE_HF_TRANSFER=1` line and the comment that references `hf-transfer`. Optionally add `# hf_xet is now the default transfer backend (transformers v5+); no env var needed.`
- `.env` (local): same removal.
- No code change needed for the 50 GB shard size — it just changes how many files land under `Cache/Training/`.

### F. Playground (low priority)

`playground/Qwen/explore_qwen_omni.py:327` and the other `apply_chat_template` call sites in `playground/`:
- Audit each call. For `tokenize=False` sites: no change (returns text). For tokenized sites: unwrap `BatchEncoding.input_ids` / `.attention_mask` where the old code assumed a bare tensor.
- Playgrounds are exploratory; breakage here is acceptable but should be fixed in the same change to avoid leaving broken examples around.

## Implementation plan (repo-rooted, ordered)

Sequenced so that each step leaves the repo in a state where `uv run pytest tests/ -v` either passes or fails loudly (no silent regressions). Each step is one commit.

### Step 0 — Branch + freeze (no code)
- Confirm E05 attempt-3 is **finished** (not just paused) on Odra. Check `run.state` in W&B and the latest run report under `docs/2_Experiments_Registry/run_reports/`.
- `git checkout dev && git pull && git checkout -b feat/transformers-v5-torch-2-12`.
- Announce in the agenda "Current focus" that the v5/torch-2.12 migration is in flight so nobody launches a run against the branch.

### Step 1 — Dependency bump + lock refresh
- Edit `pyproject.toml` per section A.
- `uv lock` (refresh `uv.lock`).
- `uv sync` locally on macOS (CPU/MPS wheels — confirms the dep graph resolves).
- Commit: `feat: bump transformers to v5, torch to 2.12, replace hf-transfer with hf-xet`.
- **Do not run training yet.** The launchers are still broken.

### Step 2 — Launcher + training glue (section B)
- Edit `scripts/train_concept_pretraining_multigpu.sh` (drop `--overwrite_output_dir`, drop `--logging_dir`, add `export TENSORBOARD_LOGGING_DIR`).
- Mirror the same edits in `parked/scripts/train_diffusion_multigpu.sh` and `parked/scripts/train_prefix_diffusion_multigpu.sh`.
- Edit `training/utils_training.py` to write the computed logging path to `TENSORBOARD_LOGGING_DIR` instead of the removed attribute.
- Edit `verification/verify_prefix_diffusion_wikitext_smoke.py`.
- Edit `evaluation/evaluate_model_on_glue.py` (both `logging_dir` and `--no_cuda`).
- Commit: `fix: adapt training launchers and glue to transformers v5 TrainingArguments`.

### Step 3 — Tokenizer + tests (section C)
- Edit `tokenization/train_tokenizer_custom.py` (`additional_special_tokens` → `extra_special_tokens`).
- Edit `tests/test_data_collators.py` (`encode_plus` → `__call__`).
- Commit: `refactor: rename additional_special_tokens → extra_special_tokens, drop encode_plus`.

### Step 4 — Custom-model `_init_weights` audit (section D)
- For each of `nn/concept_encoder.py`, `nn/concept_encoder_perceiver.py`,
  `nn/concept_encoder_weighted.py`, and `parked/nn/concept_encoder_diffusion.py`:
  - read `__init__`, decide bespoke-vs-default init,
  - add `_init_weights` override if bespoke.
- Add a regression test `tests/test_model_init_v5.py` that builds each model on a fixed seed and asserts `state_dict()` matches a pre-recorded hash (generated on the **pre-upgrade** `dev` branch). This is the falsification anchor for "v5 silently re-init my weights".
- Commit: `test: pin model init against transformers v5 default _init_weights`.

### Step 5 — Env + config (section E)
- Edit `.env.example` (remove `HF_HUB_ENABLE_HF_TRANSFER`).
- Edit local `.env` (same).
- Commit: `chore: drop HF_HUB_ENABLE_HF_TRANSFER (replaced by hf_xet in transformers v5)`.

### Step 6 — Playground sweep (section F)
- Audit and fix `playground/Qwen/*` and `playground/load_llama32.py` `apply_chat_template` call sites.
- Commit: `fix: unwrap BatchEncoding from apply_chat_template in playground scripts`.

### Step 7 — Local verification
- `uv sync` on macOS.
- `uv run pytest tests/ -v` — full suite green.
- `uv run python verification/torch_test.py` — torch + MPS sanity.
- Smoke-run `training/train_concept_pretraining.py` on a tiny dataset (e.g. 100 samples, 2 steps) on MPS to confirm the Trainer path works end-to-end under v5.
- Commit: `test: v5/torch-2.12 local smoke passes` (only if any test additions were needed; otherwise skip).

### Step 8 — Remote verification (Odra)
- `git push -u origin HEAD` and `ssh` to Odra.
- `git pull` on the branch, `uv sync` (pulls the cu128 torch 2.12 wheels + liger-kernel update).
- Run `scripts/train_concept_pretraining_multigpu.sh` with a tiny config (1 step, batch 2, 1 GPU) to confirm the launcher + DDP + Trainer path works on CUDA under v5 + torch 2.12.
- Confirm W&B logging still works (transformers v5 + hub 1.0 changed some internal logging paths).
- Confirm checkpoint save/load round-trips (50 GB shards — verify the file lands and `from_pretrained` reads it).

### Step 9 — Traceability
- `CHANGELOG.md` entry under `## [YYYY-MM-DD] - Transformers v5 + PyTorch 2.12 upgrade`, prefix `feat:` (capability bump) with `train:` + `eval:` impact. Follow the `engineering-change-tracking` template (Why / Impact / What changed / Git tag / Related).
- `docs/1_Strategy_and_Plans/agenda.md`: under "Engineering (parallel)", add a one-line entry that the v5/torch-2.12 migration is done and the foundation is now on supported versions.
- Optional architecture tag `arch/transformers-v5` if the `_init_weights` audit revealed anything structural; otherwise a plain commit tag.
- Commit message on the merge: `feat: migrate to transformers v5 + torch 2.12 (launchers, glue, init audit, hf_xet)`.

## Validation and falsification

Primary gates (falsifiable — block merge if any fail):

1. **Full test suite green on macOS:** `uv run pytest tests/ -v` — every existing test passes unchanged under v5 + torch 2.12. Any failure is a real interface break we missed.
2. **Model-init regression:** `tests/test_model_init_v5.py` asserts each model's `state_dict()` matches the pre-upgrade hash on a fixed seed. If this fails, v5’s default `_init_weights` is silently re-initialising our weights — **block, investigate, do not merge** until either the override is correct or the change is intentional and the hash is re-recorded with a dated note.
3. **Remote DDP smoke on Odra:** 2-step training run completes, W&B logs appear, checkpoint saves and loads. If the launcher fails at `HfArgumentParser` or `Trainer(...)`, the glue adaptation is incomplete.

Secondary acceptance (plausibility — investigate if it trips, do not necessarily block):

4. **Tokenizer round-trip unchanged:** `tests/test_data_collators.py` and `tests/test_tsdae_collator.py` still produce identical `input_ids` for the same input text (the v5 tokenizer backend rewrite could in principle change tokenisation for edge cases).
5. **`linalg.eigh` speed-up observed:** `analysis/run_concept_analysis.py` on an existing checkpoint runs materially faster on Odra (the torch 2.12 cuSolver backend). If it does not, we are not actually on torch 2.12 — check `uv.lock`.
6. **Checkpoint size sanity:** a saved checkpoint’s total bytes is within ~5% of the pre-upgrade save for the same model (the 50 GB shard change should not balloon total size; it just changes the number of files).

Failure / kill conditions:

- Any **primary gate** fails ⇒ block merge, root-cause before proceeding. Do not paper over with a `try/except`.
- `liger-kernel` fails to import on torch 2.12 ⇒ pin to the last-working 0.5.x patch and note it; do not block the whole migration on a single fused-kernel incompatibility (training still works without liger, just slower).
- `accelerate` version mismatch with transformers v5 ⇒ bump accelerate to whatever version transformers v5 requires in its install metadata; this is non-negotiable for the Trainer path.

## Risks / open spikes

- **`accelerate` ↔ transformers v5 compatibility:** the Trainer changes in v5 are non-trivial and accelerate must match. Spike at Step 1: after `uv lock`, confirm the resolved accelerate version is the one transformers v5 declares as compatible. If `uv` picks an accelerate that predates the v5 Trainer changes, force-bump.
- **`liger-kernel` on torch 2.12:** liger depends on Triton + CUDA internals that can break across torch minors. Low-probability but possible. Mitigation: the `sys_platform == 'linux'` marker means macOS dev is unaffected; if it breaks on Odra, pin and move on.
- **`_init_weights` silent re-init:** the highest-consequence risk. The Step-4 regression test is the falsification anchor — without it, a silent re-init would only surface as "the new run trains worse than the old one" weeks later. **Do not skip Step 4.**
- **Tokenizer backend rewrite changing edge-case tokenisation:** the v5 `TokenizersBackend` is a different code path from the v4 fast tokenizer. Mitigation: the existing collator tests (`tests/test_data_collators.py`, `tests/test_tsdae_collator.py`) cover the common paths; if they pass, the practical impact is bounded.
- **50 GB shard size on cache disk:** if a future large checkpoint is saved as a single 50 GB shard, NAS / cache-disk free space assumptions may break. Out of scope for this change but worth a note in the storage-cleanup spec (`remote_storage_layout_and_cleanup_plan.md`) follow-up.
- **CUDA 13 / `cu130` index move (deferred):** the `cu128` index is deprecated in torch 2.12 and will eventually stop publishing. This is a separate driver-upgrade task on both servers and is **explicitly out of scope** here to keep the dep bump un-coupled from infra work.

## Non-goals

- **No CUDA 13 / `cu130` index migration.** Deferred to a separate infra task (touches both servers' driver stacks).
- **No model architecture change.** The `_init_weights` audit is about *preserving* current init behaviour, not changing it.
- **No benchmark or eval-protocol change.** Numbers from a post-upgrade run are directly comparable to pre-upgrade runs *only if* the init regression test passes (gate 2); otherwise the comparison is invalid and must be re-baselined.
- **No `safe_serialization=False` re-introduction.** v5 removes it; we never used it; do not work around it.
- **No re-training of past experiments.** Existing checkpoints (E01–E05) load fine under v5 (the weight format is unchanged); only fresh training uses the new stack.
- **No `master_experiment_log.md` schema change.** This is an engineering dependency bump, not an experiment.
