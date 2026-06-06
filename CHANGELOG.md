# Changelog

All notable engineering and architecture changes are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

**Relationship to other docs:**
- This file: *What* changed in code and *when* (engineering log)
- `docs/2_Experiments_Registry/master_experiment_log.md`: *What* training runs produced which results (science log)
- `docs/1_Strategy_and_Plans/agenda.md`: *What* to do next (slim living agenda) + `docs/experiments/<ID>.md` specs

The `git_tag` column in the master experiment log links each training run to the
exact code version. Tag format: `arch/{feature}` for architecture changes,
`train/{run_id}` before launching a training run.

---

## [Unreleased]

## [2026-06-06] — Complete the research pipeline: add `implementation-plan`, remove duplicate spec index

**Why:**
- The skill set had a gap between *framing* an experiment (`experiment-design`) and *writing code* (`research-implementation`): nothing produced a detailed, repo-rooted implementation plan (which modules to reuse, forward pass with shapes, data, loss, config, snippets) — the research analog of a PRD. Also, the per-experiment `docs/experiments/README.md` carried a manual Index/Status table that duplicated the canonical results ledger (`master_experiment_log.md`) and would drift.

**Impact:**
- The pipeline is now explicit and complete: `research-scout` → `research-explain` → `research-synthesis` → `experiment-design` → `implementation-plan` → `research-implementation` → run → `experiment-tracking`. Canonical homes are unambiguous: intent → specs/plans, results → `master_experiment_log.md`, live memory → `agenda.md`.

**What changed:**
- [added] `.cursor/skills/implementation-plan/SKILL.md` — the bridge skill; writes `docs/experiments/<ID>_plan.md` (reuse map, forward pass with shapes, inputs/data, loss/objective, config + launch, tests, risks, optional code sketches), rooted in real repo classes.
- [added] `docs/experiments/PLAN_TEMPLATE.md` — template for `<ID>_plan.md`.
- [updated] `experiment-design`, `research-implementation` (now reads spec **and** `<ID>_plan.md`), `experiment-discipline.mdc` (Roles = full pipeline order), `research-synthesis` (handoff to design/plan), `project-overview.mdc` (compact Research Pipeline map + canonical-homes note), `docs/experiments/README.md` (two-file model, self-indexing, where-things-live), `docs/experiments/TEMPLATE.md` (link to plan).
- [removed] the manual `## Index` table in `docs/experiments/README.md` — `master_experiment_log.md` stays the single results ledger; the experiments folder is self-indexing.

## [2026-06-05] — Experiment-system consolidation: slim agenda, scoped specs, foundation audit

**Why:**
- This is a process change, not a direction change. Experiments had blended together: a 700-line `roadmap.md` read as a committed schedule, an 835-line `active_todos.md` result-diary, 8 tracks (A–H), and a forked `train_*.py` / `nn/concept_encoder_*.py` per idea — which made runs hard to interpret. The goal is smaller, well-defined increments built on the existing foundation. The long-term Vision is unchanged; the direction within it stays open and exploratory.

**Impact:**
- A slim living agenda + per-experiment frozen specs replace the monolith. Experiments become small increments expressed as args/configs over the shared foundation rather than new forks (encouraged by a rule + the `experiment-design` skill). Dead code removed; recursive + diffusion families parked but revivable.

**What changed:**
- [added] `docs/1_Strategy_and_Plans/agenda.md` — slim living agenda (the process, current focus, candidate directions, neutral "what we've explored" learnings). New daily driver.
- [added] `docs/experiments/` — `TEMPLATE.md` (frozen spec format) + `README.md` (lifecycle, ID scheme `E0NN_slug`).
- [added] `.cursor/skills/experiment-design/SKILL.md` — front-half skill: hypothesis → one minimal spec before code.
- [added] `.cursor/rules/experiment-discipline.mdc` — always-applied guardrail: spec-before-code, configs-over-forks, one variable at a time.
- [archived] `roadmap.md` → `docs/5_Archive/roadmap_v5_20260301.md`, `active_todos.md` → `docs/5_Archive/active_todos_v3_20260314.md` (OBSOLETE banners).
- [updated] `project-overview.mdc`, `experiment-tracking`, `engineering-change-tracking`, `docs-hygiene`, `research-synthesis`, `remote-experiment-evaluator`, `CHANGELOG.md`, `training_eval_matrix.md`, `vision_and_goals.md` — repointed from `roadmap.md`/`active_todos.md` to `agenda.md` + `docs/experiments/`.
- [renamed] `.cursor/skills/pytorch-architecture/` → `.cursor/skills/research-implementation/` — rewritten from generic PyTorch guidance into a codebase-grounded implementation skill (module map, encode→reason→decode patterns, configs-over-forks, training entrypoint + bash-launcher mechanics, and a hard reproducibility rule: never delete old code/checkpoints — park instead). It is the implementation half of the spec→code workflow.
- [removed] `nn/concept_encoder_methods.py` (dead stub, `forward`=`pass`), `nn/concept_encoder_sim_matrix.py` (orphan), `training/concept_enc_dec.py` (standalone ModernBERT→GPT-2 summarizer, never used the concept encoder), `training/model_sft.py` (orphan). Cleaned `train_mlm.py` registry (`sim_matrix_mlm`, `concept_mlm`), `run_concept_analysis.py` MODEL_CLASSES, GLUE eval `model_type` choices, and the broken stub test classes in `tests/test_concept_encoder_layer.py`.
- [parked] `parked/` — recursive family (`concept_encoder_recursive*`, `train_recursive_mlm`) and diffusion family (`concept_encoder_diffusion`, `train_diffusion`, `train_prefix_diffusion`) + their tests/scripts; excluded from foundation, registries, and `testpaths`. See `parked/README.md`.

**Git tag:** `pre-consolidation-20260605` (snapshot before this change)
**Related:** `docs/1_Strategy_and_Plans/agenda.md`

## [2026-04-20] — Local Dev Migration: Windows + Poetry → macOS + uv

**Why:**
- Primary dev machine moved from a Windows laptop with RTX 3080 to an Apple Silicon MacBook. Poetry was already swapped for `uv` upstream; the local toolchain, docs, and helper scripts had not caught up.
- Tests refused to collect on a fresh checkout because the project is application-style (no installable package) and pytest had no `pythonpath` configured.

**Impact:**
- Fresh-checkout setup on macOS is now `uv sync` → `uv run pytest tests/ -v` with no manual PYTHONPATH or per-shell tweaks.
- README, the local-environment Cursor rule, and `verification/torch_test.py` reflect the macOS / MPS reality. The remote-servers rule no longer claims poetry is the runtime.
- `scripts/sync_evaluation_reports.sh` gives macOS/Linux a first-class equivalent of the existing PowerShell sync (rsync-based, supports download / upload / two-way / dry-run).
- `~/.ssh/config` now defines `polonez` and `odra` host aliases that downstream scripts already assume; user still needs to push the public key with `ssh-copy-id`.

**What changed:**
- [updated] `pyproject.toml` — added `[tool.pytest.ini_options]` with `pythonpath = ["."]` and `testpaths = ["tests"]` so tests collect without an editable install.
- [updated] `verification/torch_test.py` — also detects Apple MPS and CPU fallback; reports the device and runs a sanity matmul.
- [updated] `README.md` — Setup section uses `uv sync` / `uv run pytest`; project tree comment notes uv instead of poetry.
- [updated] `.cursor/rules/local-environment.mdc` — rewritten for macOS / zsh / uv. Documents `uv add`, `uv add --group dev`, MPS fallback, and that liger-kernel is Linux-only.
- [updated] `.cursor/rules/remote-servers.mdc` — dropped the stale "servers use poetry" runtime block; minor CPU label cleanup.
- [updated] `.cursor/agents/experiment_remote_evaluator.md` — `model: inherit`.
- [added] `.env.example` — template for `HF_TOKEN`, `WANDB_API_KEY`, `HF_HOME`, `PYTORCH_ENABLE_MPS_FALLBACK`.
- [added] `scripts/sync_evaluation_reports.sh` — bash/rsync twin of the existing `.ps1`.
- [added] `.python-version`, `uv.lock` — pin Python 3.12 and lock dependencies for reproducible installs across macOS, Linux, Windows.

## [2026-03-08] — Perceiver V2 Denoising Reset

**Why:**
- The maintained perceiver path was still split across retired MLM-era assumptions: path-based evaluation routing, shallow decoder reuse, legacy `perceiver_mlm` / `perceiver_posonly_mlm` naming, and training wrappers that still advertised abandoned `combined` / `kendall_gal` defaults.
- The research direction is now denoising-first and semantic-first: BiXT encoder, position-only stacked decoder, checkpoint-declared evaluation routes, separate sentence-pair evaluation, and zero-shot STS-B before full fine-tuning sweeps.

**Impact:**
- Perceiver checkpoints now train and evaluate through one canonical denoising stack with a shared decoder between pretraining and `ViaDecoder` downstream evaluation.
- Pair-task evaluation is now contract-driven and separate-encoding by default for new perceiver checkpoints, and benchmark evaluation includes zero-shot STS-B as a first-class gate.
- The old interface names `perceiver_mlm`, `perceiver_posonly_mlm`, `perceiver_decoder_cls`, and the unused `training/train_tsdae.py` path are retired from the maintained training/evaluation surface.

**What changed:**
- [added] `training/train_perceiver_denoise.py` - canonical perceiver denoising entrypoint with `reconstruction` and `reconstruction+contrastive` objective variants.
- [removed] `training/train_tsdae.py` - deleted because it was never trained/evaluated as an independent path; it only duplicated the new denoising entrypoint name and added confusion.
- [changed] `training/train_mlm.py`, [added] `training/train_recursive_mlm.py` - generic MLM training now excludes recursive experiments, which live on their own isolated script.
- [changed] `nn/concept_encoder.py`, `nn/concept_encoder_perceiver.py` - added decoder-depth config, shared stacked position-only decoder, canonical denoising perceiver model, decoder reuse in `ViaDecoder`, and mean pooling for concept-space classifiers.
- [changed] `evaluation/concept_eval_routing.py`, `evaluation/evaluate_model_on_glue.py`, `evaluation/evaluate_on_benchmark.py`, [added] `evaluation/concept_checkpoint_loader.py` - metadata-driven perceiver routing, shared checkpoint loading, and zero-shot STS-B benchmark support.
- [changed] `analysis/run_concept_analysis.py`, `analysis/check_model_health.py` - aligned analysis defaults and recommendations with the new perceiver family and semantic-first gating.
- [removed] `scripts/train_perceiver_mlm.ps1`, `scripts/train_mlm_multigpu_perceiver.sh`, `scripts/test_tsdae_local.ps1` - deleted to avoid leaving misleading legacy wrappers that no longer matched the maintained code path.
- [added] `scripts/train_perceiver_denoise.ps1`, `scripts/train_perceiver_denoise_multigpu.sh`, `scripts/test_perceiver_denoise_local.ps1` - explicit denoising training/test entrypoints with non-legacy names.
- [added] `tests/test_perceiver_denoise.py`, [changed] `tests/test_evaluation_routing.py` - coverage for the denoising config contract, shared decoder stack, sentence-pair cosine path, and perceiver metadata routing.

**Retired on:** `2026-03-08`
**Git tag:** `arch/perceiver-denoise-reset`
**Reason for retirement:** the old perceiver MLM family encoded historical ablation names and evaluator aliases into the active interface, which was causing confusion about what is still a supported research path versus what is only a historical checkpoint/result.
**Related TODO:** internal implementation plan → `Perceiver V2 Training And Eval Reset`

## [2026-03-07] — Prefix Diffusion Evaluation Hardening & V2 Training Path

**Why:**
- Diffusion-family checkpoints were still easy to evaluate incorrectly: pair tasks were concatenated, diffusion/prefix checkpoints could silently fall back to encoder-only weighted pooling, and the checkpoint itself did not declare the only valid downstream route.
- The first prefix diffusion baseline also used the weakest training path: no BiXT requirement, no reduced token embedding default, and random token cuts on short/low-information samples.

**Impact:**
- Diffusion and prefix checkpoints now carry an explicit evaluation contract in `config.json`; pair tasks are routed automatically to separate sentence encoding, and diffusion-family checkpoints without the new metadata now fail loudly instead of being evaluated through a silent legacy fallback.
- Prefix diffusion training now defaults to the stronger v2 path: BiXT-only, reduced token embedding width, sentence-boundary prefix splits, and short-example filtering before training.

**What changed:**
- [added] `evaluation/concept_eval_routing.py` - shared routing contract for concept checkpoint families.
- [changed] `evaluation/evaluate_model_on_glue.py`, `evaluation/evaluate_on_benchmark.py` - automatic metadata-based routing, separate-encoding pair evaluation for diffusion-family checkpoints, and removal of silent legacy diffusion fallback.
- [changed] `nn/concept_encoder.py`, `training/train_diffusion.py`, `training/train_prefix_diffusion.py` - checkpoint metadata contract for evaluation, BiXT-only prefix training, reduced token embedding default, and richer training metadata logging.
- [changed] `data/data_collators.py` - added `sentence_boundary` split strategy for prefix/suffix generation.
- [changed] `scripts/train_prefix_diffusion_multigpu.sh` - aligned the Polonez/Odra launcher with the hardened prefix v2 defaults.
- [added] `tests/test_evaluation_routing.py`
- [changed] `tests/test_data_collators.py` - added coverage for sentence-boundary splitting and invalid split-strategy rejection.

**Git tag:** `arch/prefix-diffusion-v2-hardening`
**Related TODO:** `active_todos.md` → `TODO 13b: Prefix diffusion evaluation hardening + v2 training path` (completed)

## [2026-03-07] — Training Script Logging Unification & Rename

**Motivation:** Each training script carried ~100-200 lines of copy-pasted boilerplate
for directory setup, config logging, and WandB init. Divergent formats made cross-run
comparison unreliable, and every new script required re-copying the same block.

### Refactored — `training/utils_training.py`

Extracted 6 shared functions (`setup_file_logging`, `log_data_config`, `log_loss_config`,
`log_training_config`, `setup_run_dirs`, `init_wandb`) that all training scripts now
call. Each accepts an `extra_fields`/`extra_config` dict for script-specific values.

### Refactored — all training scripts

- `train_mlm.py`, `train_diffusion.py`, `train_prefix_diffusion.py`, `train_tsdae.py`
  replaced inline boilerplate with shared function calls.
- Renamed `mlm_training.py` → `train_mlm.py`; updated all references across 11 files.

### Impact

- **-270 net lines** (314 added in utils, 584 removed from scripts).
- Identical log format and WandB config structure across all runs.
- New training scripts need ~5 function calls instead of ~150 lines of copy-paste.

---

## [2026-03-04] — SODA-style Prefix Generation (Encode Prefix → Generate Suffix)

**Motivation:** Deep analysis ([diffusion_elbo_deep_analysis_20260301.md](docs/4_Research_Notes/diffusion_elbo_deep_analysis_20260301.md))
proved that **self-reconstruction through a concept bottleneck teaches a positional hash
function, not semantic representations.** All self-reconstruction variants collapsed:
MLM (rank 5/128), diffusion L2 (10.1/128), diffusion L6+ELBO (5.74/128), diffusion+VICReg
(5.09/128). Kendall-Gal forced rank to 95% but STS-B crashed (0.341). The SODA principle
(Hudson, CVPR 2024) shows bottleneck diffusion learns semantics only when the decoder
generates DIFFERENT content than the encoder saw. Prefix generation is the text equivalent:
encode the first 30-50% of a document, generate the remaining 50-70% via masked diffusion.
Concepts MUST carry semantic gist because surface tokens don't transfer across segments.

### Added — `nn/concept_encoder_diffusion.py`

- **`PrefixDiffusionDecoder`**: Diffusion decoder with sinusoidal (fixed, non-learnable)
  position embeddings so suffix positions always start at index 0 and generalise to any
  suffix length without retraining. Same `DiffusionDecoderLayer` (AdaLN-Zero +
  cross-attention only) as `ConceptDiffusionDecoder`.
- **`ConceptEncoderForPrefixDiffusion`**: Full model for prefix generation.
  - `forward()`: (1) encode clean prefix → concepts, (2) sample noise t ~ U(t_min, 1),
    (3) mask suffix tokens, (4) decode via concept cross-attention, (5) sparse ELBO-weighted
    CE loss at masked suffix positions.
  - `generate()`: iterative denoising from all-[MASK] suffix, confidence-based unmasking.
  - Supports `--model_name_or_path` for warm-starting encoder from MLM checkpoints.
  - Supports BiXT encoder via `use_bixt=True`.

### Added — `data/data_collators.py`

- **`DataCollatorForPrefixGeneration`**: Splits documents into prefix/suffix:
  - Strips [CLS]/[SEP], splits content at random ratio (default 30-50% prefix),
    wraps prefix as `[CLS] content [SEP]`, suffix as `content [SEP]`.
  - Enforces minimum content lengths per side (default: 5 prefix, 10 suffix).
  - Dynamic padding per batch (prefix and suffix independently).
  - Output: `prefix_input_ids`, `prefix_attention_mask`, `suffix_input_ids`,
    `suffix_attention_mask`, `labels` (with -100 at pad positions).

### Added — `training/train_prefix_diffusion.py`

- Full training script with HF Trainer integration, WandB logging, warm-start support.
- `PrefixDiffusionTrainer` subclass extracts loss from `DiffusionOutput`.
- CLI args: `--prefix_ratio_min`, `--prefix_ratio_max`, `--use_bixt`, `--token_embedding_dim`,
  `--model_name_or_path` (warm-start), concept loss args.

### Added — `scripts/train_prefix_diffusion_multigpu.sh`

- Multi-GPU launch script for Polonez/Odra via accelerate.
- Default config: H512 L6 C128 D2, ELBO=True, t_min=0.3, LR 3e-4, cosine schedule,
  batch 64, grad_accum 2, 20 epochs, no concept losses (clean baseline).

### Added — `tests/test_prefix_diffusion.py`

- 27 tests: forward shapes (9), gradient flow (8), ELBO weighting (3),
  sinusoidal positions (3), end-of-sequence (2), generation (2).

### Added — `tests/test_data_collators.py` (prefix generation tests)

- 12 tests for `DataCollatorForPrefixGeneration`: output keys/shapes, split ratios,
  no information leak, special tokens, labels padding, dynamic padding, minimum lengths.

### Verification

```
pytest tests/test_prefix_diffusion.py -v       → 27 passed
pytest tests/test_data_collators.py -k Prefix -v → 12 passed
```

**Git tag:** `arch/prefix-diffusion-20260304`

---

## [2026-03-02] — Paper-Faithful BiXT Cross-Attention (Shared Similarity Matrix)

**Motivation:** The previous `BiConceptEncoderLayer` used two separate
`nn.MultiheadAttention` modules, computing the similarity matrix twice with 6
projection matrices. The BiXT paper (Hiller et al., NeurIPS 2024, Eq. 2-3)
computes the similarity once and transposes it, using 4 projections (R+V per
side) — saving ~1/3 of cross-attention params and halving the dominant O(C*N)
matmul. At the project's 1M-token target (C=8192, N=1M), this eliminates
terabytes of redundant compute per forward pass.

The rewrite also enables true Dimension Inversion in BiXT mode: tokens stay at
`token_embedding_dim` (e.g. 32) throughout all layers instead of being projected
to `hidden_size` (e.g. 512) before the first layer. The BiXT attention handles
the dimension bridging internally via its R/V projections, yielding a 16x
reduction in persistent token memory for long sequences.

### Changed — `nn/concept_encoder.py`

- **Added `BiXTCrossAttention`**: Custom cross-attention module implementing
  Eq. 2-3 from the BiXT paper. Single similarity matrix `S = R_lat @ R_tok^T`,
  transposed for the reverse direction. Supports `dim_lat != dim_tok` for
  Dimension Inversion. Proper `key_padding_mask` handling via masked_fill.
- **Rewrote `BiConceptEncoderLayer`**: Now uses `BiXTCrossAttention` instead of
  two `nn.MultiheadAttention` modules. Both sides are updated simultaneously
  from pre-update representations (matching paper). Layer ordering: BiXCA →
  optional token FFN → concept self-attention → concept FFN.
- **Added optional token FFN** (`bixt_token_ffn` config flag, default True):
  Gated FFN on the token side after cross-attention, matching the reference
  implementation's `CABlock`. Very cheap at small `dim_tok` (e.g. 32 → 128
  intermediate).
- **`ConceptEncoder` skips `token_projection`** when `use_bixt=True`: Tokens
  stay at `token_embedding_dim` through all layers.
- **Added `bixt_token_ffn` to `ConceptEncoderConfig`**.

### Changed — `training/train_tsdae.py`

- Added `bixt_token_ffn` argument to `ModelArguments`, passed to config and
  logged to wandb.

### Changed — `training/utils_training.py`

- `count_model_params` now counts `bixt_cross_attn` params under
  `cross_attention` component.

### Changed — `analysis/concept_analysis.py`

- Attention hook registration handles both `bixt_cross_attn` (new BiXT layers)
  and `concept_token_attn` (standard layers).

### Breaking — State dict incompatibility

Old BiXT checkpoints (using `concept_token_attn` / `token_concept_attn` param
names) will NOT load into the new `BiConceptEncoderLayer`. No existing trained
BiXT models need migration — all prior experiments used standard
`ConceptEncoderLayer`.

---

## [2026-02-27] — VICReg + t_regs_mst Concept Regularization with Warmup

**Motivation:** The L6 diffusion baseline (step 20k) showed severe concept collapse
(effective rank 5.45/128 = 4.3%). Previous regularization attempts with `combined`
loss failed: Kendall-Gal weighting (Feb 19) destroyed MLM quality, fixed weight 0.1
(Feb 21) failed at both goals. Root cause analysis identified two issues: (1) `combined`
operates across-batch and cannot prevent intra-sample concept collapse, (2) Kendall-Gal
is wrong for secondary constraints. Solution: VICReg (cross-batch dimensional health) +
t_regs_mst (within-sample concept diversity) with fixed small weights and warmup.

### Changed — `nn/loss_manager.py`

- **Added `warmup_steps` to `LossConfig`**: Linear warmup for concept loss weights
  (0 = no warmup). Only applies to fixed weighting; learnable strategies adapt on
  their own.
- **Added warmup to `FixedWeighting`**: Concept loss weights are linearly ramped
  from 0 to their configured value over `warmup_steps`. Task loss weight is always
  full strength. Implementation: `weight *= min(1.0, step / warmup_steps)`.
- **Added `_current_step` tracking to `LossManager`**: Fallback step counter used
  when `step` is not passed to `forward()`. Set by `ConceptLossStepCallback`.
- **Added `ConceptLossStepCallback`**: Duck-typed TrainerCallback that sets
  `model.loss_manager._current_step = state.global_step` on every training step.
  Required for warmup since HF Trainer doesn't pass step through `model.forward()`.
- **Added `t_regs_mst` to `ConceptLossType` literal** (was missing from type hint).

### Changed — Training scripts

- **`training/train_diffusion.py`**: Added `--concept_loss_warmup_steps` CLI arg,
  registered `ConceptLossStepCallback` when warmup > 0, added to WandB config.
- **`training/train_mlm.py`**: Same CLI arg and callback registration.
- **`training/train_tsdae.py`**: Same CLI arg and callback registration.

### Changed — Shell scripts

- **`scripts/train_diffusion_multigpu.sh`**: Default changed from
  `CONCEPT_LOSSES="none"` to `"vicreg t_regs_mst"`, `LOSS_WEIGHT=0.02`,
  `CONCEPT_LOSS_WARMUP_STEPS=2000`. Added `--concept_loss_warmup_steps` to
  accelerate launch command.
- **`scripts/test_diffusion_local.ps1`**: Updated to test with `vicreg t_regs_mst`,
  weight 0.02, and warmup 10 steps.

### Added — Tests

- **`tests/test_loss_manager.py`**: 26 tests covering FixedWeighting warmup
  (8 tests), LossConfig warmup fields (5 tests), LossManager integration with
  VICReg + t_regs_mst (5 tests), VICReg component behavior (2 tests), t_regs_mst
  component behavior (3 tests), ConceptLossStepCallback (3 tests).

### Verification

```
pytest tests/test_loss_manager.py -v  → 26 passed
pytest tests/test_diffusion.py -v     → 10 passed (no regression)
```

---

## [2026-02-26] — ELBO Loss Weighting + L6 Diffusion Config

**Motivation:** Root cause analysis of L2 diffusion run (STS-B 0.138, near-random) identified
missing ELBO loss weighting and low t_min as two of five root causes. MDLM (Sahoo, NeurIPS 2024)
derives that the proper ELBO for masked diffusion is a weighted average of MLM losses with weight
proportional to 1/t. LLaDA (Nie, 2025) uses `loss / p_mask`. The diagnosis proposed a batch-level
approximation (`loss / t.mean()`), but this is mathematically incorrect — it's just a global scaling
that doesn't reweight across noise levels. Implemented correct per-token 1/t weighting instead.
Full analysis: `docs/4_Research_Notes/diffusion_diagnosis_20260226.md`.

### Changed — `nn/concept_encoder_diffusion.py`

- **Added ELBO per-token 1/t loss weighting** in `ConceptEncoderForMaskedDiffusion.forward()`:
  - Computes `F.cross_entropy(reduction='none')` to get per-token losses
  - Maps each masked token to its sample's `t` value via `sample_indices`
  - Weights each token's loss by `1 / t.clamp(min=0.1)`
  - Takes weighted mean: `(per_token_loss * weights).sum() / weights.sum()`
  - Controlled by `elbo_weight: bool` parameter (default `True`)
  - Old unweighted path preserved when `elbo_weight=False` for backward compat
- **Changed `t_min` default** from 0.1 to **0.3**: avoids the near-MLM regime (t<0.2)
  where local context suffices and concepts are unnecessary

### Changed — `training/train_diffusion.py`

- Added `elbo_weight: bool` field to `ModelArguments` (default `True`)
- Changed `t_min` default from 0.1 to **0.3**
- Passes `elbo_weight` to model constructor
- Logs `elbo_weight` to WandB config

### Changed — `scripts/train_diffusion_multigpu.sh`

| Parameter | Previous | New | Reason |
|---|---|---|---|
| `NUM_ENCODER_LAYERS` | 2 | **6** | L2 proven too shallow for compositional semantics |
| `INTERMEDIATE_SIZE` | 1024 | **2048** | Match L6 perceiver_mlm baseline FFN dim |
| `T_MIN` | 0.1 | **0.3** | Avoids near-MLM regime where concepts unnecessary |
| `ELBO_WEIGHT` | (N/A) | **True** | ELBO per-token 1/t weighting |

### Changed — `scripts/test_diffusion_local.ps1`

- Added `--t_min 0.3` and `--elbo_weight True` arguments

### Added — `tests/test_diffusion.py`

- 10 tests covering forward pass, backward pass, ELBO vs unweighted comparison,
  gradient magnitude stability across t values, t_min behavior, and generation.

### Verified

- All 10 tests pass (`poetry run pytest tests/test_diffusion.py -v`)
- Local training sanity test: 50 steps on wikitext-2, loss 9.5→2.3, grad_norm stable,
  eval_loss 2.78. No gradient explosion.

**Git tag:** `arch/elbo-loss-weighting-20260226`

---

## [2026-02-23] — Diffusion Decoder Architectural Redesign + Training Fixes

**Motivation:** Post-mortem of the first diffusion run (`diffusion_H512L2C128D2_20260221_195554`)
revealed three critical problems: (1) the decoder contained full O(N²) token self-attention —
directly contradicting the project's core O(C·N) efficiency goal; (2) AdaLN timestep conditioning
was unbounded and multiplicative, causing a catastrophic gradient explosion at epoch 12 when
the model had memorized Minipile (eval_loss → 0.009) but the LR was still 2e-4; (3) the lm_head
was applied to all L=512 positions instead of only the M masked positions (~6.6x wasted compute).
Full diagnosis: `agent_memory/cleaned_log.txt` (see conversation [Diffusion training log analysis](b3e92e31-4e4f-4e41-89ab-85e7bde3acb8)).

**Research basis:** Muse (Chang et al., 2023) — masked generation conditioned on latent embeddings
via cross-attention; DiT (Peebles & Xie, 2023) — AdaLN-Zero for stable timestep conditioning;
Perceiver IO (Jaegle et al., 2021) — cross-attention-only decoding.

### Changed — `nn/concept_encoder_diffusion.py` (complete rewrite)

**`DiffusionDecoderLayer`** — removed token self-attention, redesigned around concept cross-attention:
- **Removed:** `self.norm_self`, `self.self_attn` (full O(N²) self-attention between all token positions)
- **Kept:** `self.cross_attn` — tokens attend to C=128 concept keys/values: O(N·C)
- **Replaced AdaLN with AdaLN-Zero** (Peebles & Xie, DiT 2023):
  - Single `adaLN` linear maps timestep to 6 modulation vectors: `[scale_ca, shift_ca, gate_ca, scale_ff, shift_ff, gate_ff]`
  - `nn.init.zeros_()` on both weight and bias — layer starts as identity, gates start at zero
  - Modulates both cross-attention and FFN independently
  - **Eliminates multiplicative runaway** that caused grad_norm → 947 in the previous run

**`ConceptDiffusionDecoder`** — returns hidden states, NOT logits:
- Removed `self.lm_head` from decoder; lm_head now lives in the model class
- Enables sparse logit computation: lm_head applied only to M masked positions

**`ConceptEncoderForMaskedDiffusion`** — sparse loss, label smoothing, padding-safe noise:
- Added `self.lm_head` at model level; applied sparsely to masked positions only (matching MLM perceiver's sparse decoding pattern)
- Added `label_smoothing` parameter (default 0.1): prevents overconfident predictions and near-zero eval_loss that signals memorization
- Fixed `_apply_noise()`: now respects `attention_mask` — padding positions are never masked
- Changed `t_min` default: 0.05 → 0.1 (minimum ~51 masked tokens/sample vs ~25, reducing gradient variance)
- In `generate()`: full lm_head over all positions is acceptable (inference, no sparsity constraint)

**Complexity comparison per decoder layer:**

| Sequence length | Previous (self + cross) | New (cross-attention only) | Speedup |
|---|---|---|---|
| 512 | O(N²) + O(N·C) = 269K | O(N·C) = 65K | 4× |
| 4,096 | O(N²) + O(N·C) = 17.3M | O(N·C) = 524K | 33× |
| 2,000,000 | O(N²) + O(N·C) ≈ 4T | O(N·C) = 256M | **15,000×** |

### Changed — `scripts/train_diffusion_multigpu.sh`

| Parameter | Previous | New | Reason |
|---|---|---|---|
| `LEARNING_RATE` | 5e-4 | **3e-4** | Matches stable MLM perceiver L6; 5e-4 caused explosion post-overfit |
| `lr_scheduler_type` | linear | **cosine** | At 60% progress: cosine→3e-5 vs linear→2e-4; faster mid-training decay |
| `GRADIENT_ACCUMULATION_STEPS` | 1 | **2** | Effective batch 512 (matching MLM perceiver); halves step count 78K→39K |
| `T_MIN` | 0.05 | **0.1** | Reduces gradient variance; still covers full range to t=1.0 |
| `LABEL_SMOOTHING` | (none) | **0.1** | Prevents memorization and overconfident logits |
| `DECODER_LAYERS` description | "Diffusion decoder layers" | "Cross-attention layers (no self-attention)" | Clarified |

### Changed — `training/train_diffusion.py`

- Added `label_smoothing: float` field to `ModelArguments` (default 0.1)
- Passes `label_smoothing` to `ConceptEncoderForMaskedDiffusion.__init__()`
- Logs `label_smoothing` to WandB config for experiment traceability
- Changed `t_min` default from 0.05 to 0.1

### Changed — `scripts/test_diffusion_local.ps1`

- Updated `lr_scheduler_type` from `"linear"` to `"cosine"`
- Added `--label_smoothing 0.1` argument

### Verified

- Forward pass: loss computed, `masked_logits` shape `[M, V]` (sparse), `logits=None` during training
- Backward pass: gradients flowing through cross-attention and AdaLN-Zero modulation
- Generate: iterative denoising produces `[1, 64]` output
- No linter errors

**Git tag:** `arch/diffusion-xattn-only-20260223`

**Expected impact on next run:**
- No gradient explosion (AdaLN-Zero zero-initialization + cosine decay + label smoothing)
- ~3–4× faster per step (no self-attention + sparse lm_head)
- ~2× fewer steps (grad_accum=2)
- Scales to sequences of any length (O(N·C) decoder is the long-context foundation)

---

## [2026-02-21] — TSDAE Architecture Overhaul

**Motivation:** Five structural misalignments in MLM+Perceiver pipeline identified.
See `docs/4_Research_Notes/mlm_perceiver_diagnosis_20260221.md`.

### Added
- `training/data_collators.py`: `DataCollatorForTSDAE` — token deletion (60%),
  dense labels at all non-pad positions, attention_mask zeroing for deleted tokens.
- `nn/concept_encoder.py`: `BiConceptEncoderLayer` — BiXT-style bidirectional
  cross-attention (tokens update from concepts at each layer). O(C*N) preserved.
  Enabled via `use_bixt=True` in `ConceptEncoderConfig`.
- `nn/concept_encoder_perceiver.py`: `ConceptEncoderForSentencePairClassification` —
  separate sentence encoding, weighted concept pooling, InferSent-style feature
  engineering `[z_a; z_b; |z_a-z_b|; z_a*z_b]`, `cosine_only` mode for zero-shot STS-B.
- `training/train_tsdae.py`: Full TSDAE training script with BiXT support,
  WandB logging, warm-start from MLM checkpoints.
- `scripts/test_tsdae_local.ps1`: Local smoke test (standard + BiXT modes).
- `tests/test_tsdae_collator.py`: 10 tests for `DataCollatorForTSDAE`.
- `training/utils_training.py`: `get_git_info()` — returns current commit hash
  and git tags for WandB traceability.

### Changed
- `nn/concept_encoder_perceiver.py`:
  - `ConceptEncoderForMaskedLMPerceiverPosOnly`: rewrote `forward()` to use
    **dense reconstruction loss** (CE at all non-pad positions, `ignore_index=-100`).
    Removes sparse MLM loss path. TSDAE-compatible.
  - `ConceptEncoderForSequenceClassificationPerceiver`: replaced single CLS query
    cross-attention with **weighted concept pooling** (`concept_scorer` Linear +
    softmax + weighted sum). Removes `cls_query`, `cls_cross_attn`, `cls_ffn`.
  - `ConceptEncoderForSequenceClassificationViaDecoder`: added `decoder_posonly`
    config flag (default False for backward compat). When True, decoder queries
    use position embeddings only (matching PosOnly pretraining).
- `training/evaluate_model_on_glue.py`:
  - Added `preprocess_function_separate()` for per-sentence tokenization.
  - Added `perceiver_pair_cls` model type.
  - Added `ConceptEncoderForSentencePairClassification` to model registry.

### Verified
- All 10 TSDAE collator tests pass.
- TSDAE training: standard mode (exit 0, loss decreasing).
- TSDAE training: BiXT mode (exit 0, loss 10.84→10.72, gradients flowing).

**Git tag:** `arch/tsdae-overhaul-20260221`

---

## [2026-02-19] — Concept Loss Experiments + Beyond-GLUE Eval

### Added
- `nn/loss_manager.py`: `TREGSMSTLoss` — MST-based uniformity regularization.
- `nn/concept_encoder_recursive.py`: `RecursiveConceptEncoder` (1 shared layer, K iterations).
- `nn/concept_encoder_recursive_mlm.py`: `RecursiveConceptEncoderForMaskedLM`.
- `evaluation/evaluate_on_benchmark.py`: SICK + PAWS beyond-GLUE evaluation.
- `evaluation/evaluate_model_on_glue.py`: moved from `training/`, added `perceiver_decoder_cls`.

### Changed
- `scripts/train_mlm_multigpu_perceiver.sh`: enabled `combined` concept losses +
  `kendall_gal` weighting by default.
- `training/evaluate_model_on_glue.py`: STS-B bug fixed (predictions squeezed to 1D).
- `training/train_mlm.py`: added `torch_compile_dynamic` flag (fixes step 8000
  gradient explosion).

**Key result:** Combined+kendall_gal fixed concept rank (5/128 → 122/128) but muted
MLM gradient (loss 2.54 → 4.31). GLUE regressed. See `run_reports/concept_losses_20260219.md`.

---

## [2026-02-08] — L6 Scaling Baseline

### Added
- Training runs: `perceiver_mlm_H512L6C128`, `perceiver_posonly_mlm_H512L6C128`,
  `weighted_mlm_H512L6C128` — all 40 epochs on Minipile.
- Sparse MLM decoding fix: avoids OOM from accelerate fp32 conversion on [B,L,V] tensor.

**Key result:** `perceiver_mlm` L6 best model: MRPC 81.3%, MNLI 59.1%, STS-B 0.627.
Concept effective rank: 5/128 (severe collapse). See `master_experiment_log.md`.

---

## [2026-01-17] — L2 Baseline + ModernBERT Tokenizer

### Added
- Training runs: `weighted_mlm_H512L2C128`, `perceiver_mlm_H512L2C128`,
  `perceiver_posonly_mlm_H512L2C128` — 20 epochs on Minipile.
- ModernBERT-base tokenizer (50k vocab, 8192 max length).
- `nn/concept_encoder_perceiver.py`: initial `ConceptEncoderForMaskedLMPerceiver`,
  `ConceptEncoderForMaskedLMPerceiverPosOnly`.

**Key result:** `weighted_mlm` best MRPC F1 82.2%. All models hit architectural ceiling
on CoLA (MCC ~0.13). Average GLUE gap to BERT-Base: -23.7pts.
