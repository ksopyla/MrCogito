# Changelog

All notable engineering and architecture changes are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

**Relationship to other docs:**
- This file: *What* changed in code and *when* (engineering log)
- `docs/2_Experiments_Registry/master_experiment_log.md`: *What* training runs produced which results (science log)
- `docs/1_Strategy_and_Plans/agenda.md`: *What* to do next (slim living agenda) +
  `docs/experiments_specs/<lifecycle>/<ID>.md` specs

The `git_tag` column in the master experiment log links each training run to the
exact code version. Tag format: `arch/{feature}` for architecture changes,
`train/{run_id}` before launching a training run.

---

## [2026-08-14] - E17c depth-private gated working memory

**Why:**
- E17/E17b's private state tensors remained coupled to Gemma QKV and one tied additive
  writer, while ordinary causal LM loss rewarded the explicit local carry instead of
  durable concept content.

**Impact:**
- The shared backbone-concept family can train strictly block-causal, depth-private
  working-memory cells and directly test their necessity under deterministic carryless
  evaluation without changing E17b defaults or checkpoint keys.

**What changed:**
- [added] config-selectable dedicated concept reads, untied BiXT writers, content-gated
  retain/replace dynamics, per-example causal carry dropout, and weighted early-token CE
- [added] all-bank geometry, per-bank permutation, carryless first-64 diagnostics, dynamic
  update/state telemetry, and a thin E17c launcher over the existing Gemma training path
- [fixed] the E16b wrapper now honors explicit 4K cache/manifest overrides for isolated
  launcher verification; its default immutable 4K path remains unchanged
- [fixed] gated replacement casts autocast BF16 candidates/gates back to the FP32 recurrent
  state dtype before interpolation, preventing the first-step mixed-precision crash
- [fixed] eval/ablation forwards no longer overwrite the last observed training carry-drop
  fraction, so W&B reports the intervention rate instead of a misleading zero
- [preserved] the default E17b read/write path, model family, collator, data manifest,
  optimizer/evaluation plumbing, and W&B identity contract
- [coverage] legacy construction, private-cell gradients, causal pressure masking, weighted
  CE, causality, per-bank ablations, checkpoint round-trip, and launcher parameter flow

**Related:** [E17c](docs/experiments_specs/ahead/E17c_depth_private_working_memory.md)

---

## [2026-07-19] - Remove low-quality deductive-stories prototype

**Why:**
- In-repo synthetic narrative pipeline produced quality far below the intended hard
  long-context deduction dataset; better as a separate repository.

**Impact:**
- MrCogito no longer ships deductive-stories generation code, eng spec, or Azure
  dataset-generation env knobs.

**What changed:**
- [removed] `data/deductive_stories/`, `scripts/build_deductive_stories.py`, tests,
  mix snippet, eng spec; dropped `openai` dependency added for that prototype

**Related:** `docs/1_Strategy_and_Plans/agenda.md`

---

## [2026-07-15] - E16b long-context Muon 1B protocol

**Why:** Short 2K cautious pilots (E10–E16a) kept concepts geometrically healthy
but never unlocked persistent causal use. The next bet needs longer documents,
more tokens, and Muon together.

**Impact:** the shared depth-recurrent architecture can train at seq 4096 on a
long-document Gemma mix for a 1B-token Muon run without a model fork.

**What changed:**
- [added] `e16b_long_4k_v1` long-document mix (FinePDFs + PG19 + Wikipedia +
  DCLM/FineWeb/Stack); superseded the short-lived FinePDFs-heavy 4K draft.
- [added] thin `scripts/launch_e16b.sh` pinning Muon + 4K + 1B over `launch_e10.sh`.
- [fixed] `launch_e10.sh` now allows `MAX_SEQ_LENGTH` override (default remains 2048).
- [fixed] E16b 4K corpora isolated under `datasets_tok_gemma_4k` (avoids 2K fingerprint clash).
- [tested] mix weight/policy contract and E16b launcher parameter flow.

**Related:** [E16b](docs/experiments_specs/done_success/E16b_longctx_muon_1b.md)

---

## [2026-07-14] - E16a optimizer A/B launch protocol

**Why:** E16 remained stable but causally unused at 50M tokens; E16a tests whether
optimizer efficiency changes that trajectory before context length and data are scaled.

**Impact:** the shared depth-recurrent architecture can run a reproducible, unattended
100M-token Adam-versus-Muon comparison with matched model, data, initialization, and
evaluation cadence.

**What changed:**
- [added] a thin E16a protocol wrapper that pins the implemented shared-depth architecture
  and selects either calibrated differential AdamW or the stabilized Muon recipe.
- [added] a fail-fast sequential Adam→Muon pipeline for exclusive use of Odra's GPUs.
- [tested] optimizer-specific LR/weight-decay propagation, shared E16 invariants, invalid
  optimizer rejection, pipeline ordering, and existing Muon/backbone optimizer contracts.

**Related:** [E16a](docs/experiments_specs/done_failed/E16a_muon_optimizer_ab.md)

---

## [2026-07-13] - Shared depth-recurrent concept workspace

**Why:** E10e updated one shared concept state only after a full 26-layer block, so every
concept-reading depth saw the same stale state and persistent causal use remained negligible.

**Impact:** the backbone-concept family can now test repeated depth-wise refinement of one
coherent state without changing the backbone, read interface, objective, data, or legacy E10
checkpoint behavior.

**What changed:**
- [added] config-selectable `shared_depth_recurrent` execution that discovers Gemma's global
  layers, applies one tied BiXT write after each concept read, and gives each depth its own
  scalar gate.
- [preserved] the default `global_kv` monolithic path, legacy scalar write gate and checkpoint
  keys, `concept_num=0` control path, all recurrent-state ablations, and existing eval APIs.
- [added] checkpoint-safe explicit Gemma layer execution: recurrent state is passed to reads
  explicitly and updated only in the parent block loop after checkpointed layers return.
- [extended] per-depth gate telemetry and differential optimizer coverage for depth gates.
- [tested] native-loop equivalence with concepts disabled, interleaved write ordering and state
  chaining, all ablations, finite gradients, checkpointed gradient equivalence, checkpoint
  round-trip, control compatibility, CLI/launcher propagation, and optimizer routing.

**Related:** [E16](docs/experiments_specs/done_failed/E16_shared_depth_recurrent_concepts.md)

---

## [2026-07-13] - Forced delayed-recall memory diagnostic

**Why:** E10 through E10e retained diverse concept states but never developed measurable
persistent-memory dependence under natural plain-token CE, leaving the memory mechanism
confounded with a weak training signal.

**Impact:** the unchanged E10e architecture can now be tested on a counterfactual task where
only block-1 memory identifies a block-4 answer. Sparse labels and deterministic donor
interventions provide direct CE, accuracy, and paired-bootstrap gates before any E12/E13 pivot.

**What changed:**
- [added] deterministic Gemma-tokenized delayed-recall data and manifest generation with
  balanced single-token values, train/eval-disjoint counterfactual twins, and block-2/3
  memory-age views.
- [added] opt-in preservation of precomputed causal-LM label masks; defaults preserve all
  existing E10 behavior.
- [added] sparse per-position CE/top-1 model instrumentation and explicit batch concept-state
  permutation for conflicting-donor evaluation without changing the real recurrent path.
- [added] paired delayed-recall checkpoint evaluation, E14's thin protocol wrapper, and
  parser/launcher/model/data regression coverage.
- [verified] Gemma tokenizer artifact build, end-to-end tiny-checkpoint evaluator smoke, and
  full local suite: 333 passed, 9 skipped.

**Related:** [E14](docs/experiments_specs/done_failed/E14_forced_delayed_recall_memory.md)

---

## [2026-07-12] - E10 concept-path calibration controls

**Why:** E10's 100M pilot kept diverse concepts but showed <0.001-nat recurrent-state
dependence, with low-scale concept reads and concept-path updates far below LoRA.

**Impact:** the same pretrained-Gemma architecture can now isolate three cumulative recovery
tests without changing data or objective: normalized concept reads, small-live memory gates,
and a higher AdamW LR for newly initialized concept-memory parameters. Defaults reproduce E10.

**What changed:**
- [added] optional per-global-layer RMSNorm before concept-read K/V projections.
- [exposed] backward-compatible read/write gate initialization through model args, factory,
  launcher, checkpoints, and W&B config.
- [added] exhaustive AdamW parameter grouping for an optional concept-memory LR; LoRA retains
  the base LR, scalar/norm parameters remain no-decay, unknown trainables fail closed, and Muon
  rejects the incompatible option.
- [tested] zero-init equivalence and old defaults, normalized-read checkpoint round-trip,
  first-step gradients with 0.01 gates, optimizer partitioning, parser/factory plumbing, and
  shell env-to-CLI propagation.

**Related:** [E10b](docs/experiments_specs/done_failed/E10b_normalized_concept_read.md) →
[E10c](docs/experiments_specs/done_failed/E10c_nonzero_memory_gates.md) →
[E10d](docs/experiments_specs/done_failed/E10d_differential_concept_lr.md)

---

## [2026-07-12] - Portable remote path contract

**Why:** the training refactor contract suite passed locally but its isolated cache-path test
resolved Odra's real checkout when run on Odra, instead of the test workspace.

**Impact:** launchers keep the same canonical Odra/Polonez path by default, while tests and
explicit tooling can safely override `PROJECT_ROOT`.

**What changed:**
- [fixed] made `scripts/remote_paths.sh` preserve an explicit `PROJECT_ROOT`.
- [fixed] made the runtime-contract harness set its isolated project root explicitly.

---

## [2026-07-11] - Training refactor Stage 4 retired paths

**Why:** the active training tree still exposed sparse weighted-MLM as a maintained command, while
the parked tree carried an unrun recursive-MLM fork that no longer represented the project's
recurrent-memory direction.

**Impact:** new research has one maintained concept-pretraining foundation. Historical weighted-MLM
checkpoints keep their live model and evaluation routes, with the trainer retained under `parked/`
for exact reproduction. Diffusion and prefix diffusion remain parked and revivable. The old
recursive-MLM implementation is recoverable from git but no longer appears as a viable current
training family.

**What changed:**
- [parked] moved `training/train_mlm.py` to the accurately named
  `parked/training/train_weighted_mlm.py`; corrected its repository import path and retained its
  logging, data, W&B, and CLI behavior.
- [preserved] kept `nn/concept_encoder_weighted.py`, weighted checkpoint routing, benchmark
  choices, and 18 historical W&B runs intact.
- [retired] removed the isolated recursive encoder/model/trainer/tests/launcher after confirming
  there is no recursive W&B or experiment-ledger run; added a recovery tombstone pointing to git
  history and `pre-consolidation-20260605`.
- [preserved] made no code or protocol changes to masked diffusion or prefix diffusion.
- [reconciled] current matrix, README, engineering plan, tests, rules, and skills while preserving
  frozen results, run reports, dated diagnoses, archived plans, and historical changelog entries.
- [verified] parked weighted-MLM CLI help, historical evaluation routing, and checkpoint round-trip
  succeed; full suite: 319 passed, 9 skipped.

---

## [2026-07-11] - Training refactor Stage 3 launcher roles

**Why:** the generic multi-family launcher still carried the original Perceiver-denoising name,
while experiment wrappers and orchestration pipelines had distinct responsibilities that were
encoded only in comments and operational knowledge.

**Impact:** new ad hoc maintained-family runs use
`scripts/train_concept_pretraining_multigpu.sh`. Existing commands using
`train_perceiver_denoise_multigpu.sh` continue to generate the same training arguments through a
thin compatibility wrapper. E05/E10 protocols, preprocessing, logging paths, W&B identities,
checkpoint layout, and parked launchers are unchanged.

**What changed:**
- [renamed] the generic runner to `scripts/train_concept_pretraining_multigpu.sh`; retained the old
  path as an executable compatibility wrapper.
- [clarified] `launch_e05.sh` and `launch_e10.sh` are protocol wrappers, while
  `launch_e10_pipeline.sh` is a prerequisite/gate/pretokenization orchestration pipeline.
- [documented] launcher roles and change policy in `scripts/README.md`, README, workspace guidance,
  and operational skills.
- [preserved] frozen experiment specs, append-only ledgers, historical reports, and parked
  recursive/diffusion launchers retain their recorded paths and semantics.
- [tested] canonical and compatibility runners produce identical captured arguments; E05 and E10
  wrappers preserve their architecture, objective, tokenizer, data, and backbone pins; the E10
  pipeline remains chained through its protocol wrapper.
- [verified] full suite: 315 passed, 9 skipped; all live training launchers pass Bash syntax checks.

---

## [2026-07-11] - Training pipeline contract test hardening

**Why:** module-level tests covered the extracted pieces, but the highest-risk boundaries remained
under-tested: Bash environment variables becoming Python arguments, the canonical `main()`
assembling the runtime, and parsed values reaching Trainer, W&B, data routes, and final saves.

**Impact:** training refactors now fail locally if they drop or misroute representative E05/E10
arguments, cache paths, direct/pretokenized/mix data selection, logging/W&B metadata, resume state,
optimizer settings, DDP timeout, or final artifacts. No model, data, checkpoint, or W&B identity
changed.

**What changed:**
- [tested] representative E05 and E10 CLI profiles through the canonical parser, the complete
  family/objective validation matrix, config mapping, and model-family routing.
- [tested] direct-Hub and pretokenized `main()` orchestration through logging, factories, W&B,
  Trainer construction, resume, and final saves; added one real CPU Trainer optimizer step with
  loss logging.
- [tested] the generic Bash launcher end-to-end with command capture, including Hugging Face cache
  propagation, manifest precedence, exact-token guards/calculation, optional experiment arguments,
  Muon settings, and the first-process-group DDP timeout.
- [tested] registered-mix routing, deterministic eval caps, effective-dataset priority, and a local
  two-source split/tokenize/interleave/eval integration.
- [fixed] empty optional argument arrays now expand safely under macOS Bash 3 with `set -u`; Linux
  launcher arguments are unchanged.
- [verified] 30 new tests; full suite: 310 passed, 9 skipped; launcher syntax passes.

---

## [2026-07-11] - Training refactor Stage 2 canonical entrypoint

**Why:** the historical `train_perceiver_denoise.py` name described only the original parallel
reconstruction model, but the maintained command now trains Perceiver reconstruction, concept AR,
prefix-to-suffix generation, windowed decoding, anchors, and pretrained-backbone concept memory.

**Impact:** new commands and live documentation use the accurate
`training/train_concept_pretraining.py` path. The old path remains executable and re-exports its
public symbols for one migration window. Logging, data processing, Hugging Face caches, workspace
artifacts, CLI flags, checkpoint metadata, and W&B identities are unchanged.

**What changed:**
- [renamed] the multi-family orchestration implementation to
  `training/train_concept_pretraining.py`.
- [compatibility] retained `training/train_perceiver_denoise.py` as a thin executable/import
  wrapper.
- [updated] the generic launcher and local smoke command, maintained tests, README, training/eval
  matrix, workspace rule, and training skills to use the canonical path.
- [preserved] frozen experiment specs, append-only ledgers, historical reports, and older changelog
  entries keep the entrypoint path that was true when recorded; launcher filenames are deferred to
  Stage 3.
- [tested] both canonical and compatibility CLIs expose the same model/data arguments; the generic
  launcher targets the canonical path.
- [verified] full suite: 280 passed, 9 skipped; live launcher syntax and linter diagnostics pass.

---

## [2026-07-11] - Training refactor Stage 1 neutral extraction

**Why:** `training/train_perceiver_denoise.py` combined CLI schemas, compatibility validation,
custom Trainer behavior, data routing, collator selection, model construction, and W&B identity
with runtime orchestration. That coupling made small training changes difficult to review and
increased the chance of unrelated behavior drift.

**Impact:** the historical entrypoint and launcher commands remain operational, and its public
imports remain compatible. Training behavior, console/file/W&B logging, dataset priority and
processing, Hugging Face caches, workspace `Cache/` paths, checkpoint metadata, and W&B identities
are unchanged.

**What changed:**
- [refactored] import-light objective constants and EOS policy into
  `training/concept_pretraining_objectives.py`; preprocessing no longer imports the full training
  entrypoint for this shared policy.
- [refactored] argument dataclasses and family compatibility validation into
  `training/concept_pretraining_args.py`.
- [refactored] custom loss, anchor, Muon, deterministic-eval, concept-ablation, and geometry Trainer
  behavior into `training/concept_pretraining_trainer.py`.
- [refactored] direct/pretokenized/mix data routing plus model, collator, special-token, and W&B
  identity construction into `training/concept_pretraining_factories.py`.
- [compatibility] `training/train_perceiver_denoise.py` re-exports its previously imported public
  symbols and remains the canonical command during this migration stage.
- [tested] added extraction-boundary tests for re-exports, validation, data-source priority, cache
  propagation, objective-specific collators, deterministic eval seeds, E10 W&B grouping, and
  distributed arm-specific run ids.
- [verified] full suite: 278 passed, 9 skipped; preprocessing CLI import smoke passed; no linter or
  whitespace diagnostics.

---

## [2026-07-11] - Training refactor Stage 0 contract baseline

**Why:** the maintained training entrypoint now serves several model/objective families, while
parked training families duplicate parts of the runtime pipeline. Modularizing that code without
first pinning its external behavior risks changing datasets, W&B lineage, logs, cache locations,
or checkpoint compatibility under the guise of a structural refactor.

**Impact:** no production training behavior changed. The next extraction stage now has executable
compatibility gates for console/file/W&B logging, direct and pretokenized Hugging Face data,
canonical cache roles, and the workspace `Cache/` artifact layout.

**What changed:**
- [docs] added the staged
  [training pipeline modularization spec](docs/engineering_specs/training_pipeline_modularization.md),
  including immutable logging, data, cache, checkpoint, W&B, and parked-code contracts.
- [tested] every active/parked training entrypoint still uses the shared logging helpers; W&B
  project/group/run identity, effective dataset metadata, and bounded tags are characterized.
- [tested] direct Hub loading retains the configured `HF_DATASETS_CACHE`, while pretokenized
  manifests load `save_to_disk()` datasets without Hub access.
- [tested] `scripts/remote_paths.sh` retains the `datasets` / `datasets_tok` / `datasets_raw`
  roles, honors explicit overrides, and keeps `Cache/Training` plus `Cache/logs`.
- [verified] 40 focused training/data/W&B/evaluation-lineage tests pass; touched tests have no
  linter diagnostics.

---

## [2026-07-11] - E10 pre-launch protocol and observability audit

**Why:** the E10 Stage-0 architecture was sound, but the launch wrapper had drifted from the
frozen effective batch and the live diagnostics did not measure the pre-registered long-range
and collapse gates. Launching a ~2B-token arm in that state would make the 50%-budget kill
decision unreliable.

**Impact:** E10 now launches with the specified optimizer-update batch, reports the decisive
≥1024 concept-content signal and within-sample collapse metric during training, exposes gate
opening directly, and can be analyzed from saved checkpoints through the canonical Tier-1 runner.

**What changed:**
- [fixed] E10 launcher default from effective batch 24 to 72. Odra calibration selected
  `8 × 3 GPUs × accum 3` (19.97 GiB peak; batch 10 OOM); the matched Polonez control uses
  `6 × 4 × accum 3`.
- [fixed] recurrent concept ablation now separates the explicit-carry region `[K,2K)` from
  true beyond-carry positions `≥2K`; E10's Δshuffle gate therefore matches the spec's ≥1024
  region instead of being diluted/inflated by locally reachable context.
- [added] live within-sample raw/centered RankMe and tanh read/write-gate telemetry at each
  evaluation; added `backbone_concept` support to the held-out Tier-1 geometry/ablation runner.
- [tested] production gradient-checkpointing concept/write gradient path and E10 Tier-1
  ablation-contract normalization; 33 targeted tests pass.
- [fixed] Gemma tokenizer/model vocabulary mismatch: the tokenizer-only multimodal
  `<image_soft_token>` id 262144 exceeds the text backbone's 262144-entry embedding table.
  Pretokenization now splits literal special-token strings as ordinary text and the causal
  collator validates against the model vocabulary, preventing a delayed CUDA index assert.
- [fixed] frozen Gemma LM-head CE backward no longer allocates/computes a ~1.1 GiB fp32
  weight gradient; resume now restores Trainer/optimizer/scheduler/RNG state; final saves use
  `<run>/final` instead of duplicating `<run>/<run>`.
- [fixed] E10 defaults backbone gradient checkpointing to the non-reentrant PyTorch path.
  Reentrant checkpoint hooks marked block-reused LoRA parameters ready multiple times under DDP
  and aborted the first production launch before step 1.
- [fixed] the training banner now resolves the effective pretokenized manifest instead of printing
  the unused default MiniPile argument; causal-LM eval collation is labeled deterministic/no
  corruption. Gemma Trainer-facing special-token configs are explicitly aligned to canonical
  tokenizer IDs (PAD=0, BOS=2, EOS=1), eliminating one benign warning per DDP rank.
- [added] exact manifest token counting/checksum and a 2B non-padding-token target schedule
  (deterministic rounding within one optimizer batch); retained ~10%-budget checkpoints support
  matched 50%/100% A/B comparisons, while live eval
  uses a deterministic 2,048-row subset.
- [optimized] the one-time token count now uses an 8-worker reducing `Dataset.map`, reports
  progress/ETA, selects only `input_ids`, and atomically caches the result beside the manifest.
- [optimized] weighted `all_exhausted` dataset interleaving now vectorizes Hugging Face's exact
  1,000-choice RNG protocol instead of appending one row index per Python iteration. This removes
  the single-core multi-hour startup cost for both token counting and Trainer dataset loading.
- [added] recurrence-specific static and previous-block-only ablations plus
  `run_e10_comparison.py` for paired concept/control 2K/8K CE, local regression, RankMe, and
  bootstrap CIs. Stage 0 can now use one frozen held-out long-doc manifest paired across lengths.
- [hardened] pretoken caches require complete train+eval artifacts and persist a tokenizer /
  revision / sequence / objective-source fingerprint; eval-only manifests support immutable 8K
  protocol data. DDP loss is globally token-weighted across unequal rank padding.
- [docs/data] declared `causal_lm` compatibility on the reused 2K mix and reconciled the E10
  spec, implementation plan, and agenda.

**Related:** `docs/experiments_specs/done_failed/E10_gemma_backbone_concept_memory.md`

---

## [2026-07-08] - E10 foundation: pretrained-backbone concept memory (Gemma-3 graft, Design C)

**Why:** the platform pivot (spec
`docs/experiments_specs/done_failed/E10_gemma_backbone_concept_memory.md`): stop paying the
from-scratch language-acquisition cost per run — graft the concept read/write
machinery onto a frozen pretrained decoder (`google/gemma-3-1b-pt`, whose
5-sliding:1-global layer pattern is a ready-made socket) and make the concepts a
gated recurrent memory written per 512-token block (the E09 write design,
executed on the backbone). Recurrent encode == recurrent decode: no separate
encoder, unbounded input length at fixed memory, O(N·(K+C)).

**What changed:**
- [added] `nn/backbone_concept_lm.py` — `BackboneConceptLM` + `BackboneConceptConfig`
  (new config-selectable model family, `concept_io_mode="global_kv"`; E11
  `mem_tokens` / E12 `kv_prefix` land here later). Frozen backbone + peft LoRA
  (`inject_adapter_in_model`); mask-dict surgery windows ALL token↔token
  attention; the 4 global layers get a zero-init tanh-gated concept read using
  their own (LoRA-adapted) q/k/v/o projections, no RoPE on the read (position-free
  memory ⇒ length-extrapolation-safe); `ConceptWriteHead` reuses
  `BiXTCrossAttention(update_tokens=False)` with zero-init α + sandwich RMSNorm;
  block-recurrent forward with one-block carry, per-block position reset, and
  `ChunkedLMHeadCE` (the 262K-vocab [B,S,V] logits are never materialized);
  `per_position_ce` (blockwise / single_windowed / full_attention scorers),
  `concept_ablation_ce` + `encode_concepts` matching the trainer's eval-hook
  contract. Zero-init property: gates at 0 ⇒ block loop == plain window-masked
  Gemma for the first two blocks, harder history truncation beyond (that truncated
  context is exactly the concepts' channel); `concept_num=0` = the control arm.
- [added] `data/data_collators.py:DataCollatorForCausalLM` — plain next-token-LM
  collator (pad to batch max, labels −100 at pad).
- [changed] `training/train_perceiver_denoise.py` — new `objective_variant`
  `causal_lm` + `--backbone_model`/`--concept_block`/`--concept_io_mode`/`--lora_*`
  args; backbone branch for model build, collator, W&B identity
  (`backbone_concept` family, concept-arm/control-arm tags); default path
  byte-identical when `backbone_model` unset.
- [changed] `scripts/train_perceiver_denoise_multigpu.sh` — `BACKBONE_ARGS`
  (gated on `BACKBONE_MODEL`), `GRADIENT_CHECKPOINTING` knob (was hardcoded
  False), `DATASETS_TOK_DIR` overridable (tokenizer-switch cache isolation).
- [added] `scripts/launch_e10.sh` — E10 protocol wrapper (concept arm default,
  `CONCEPT_NUM=0` control arm; Gemma-tokenized `smollm3_inspired_2k_e05` mix into
  its own `datasets_tok_gemma` tree).
- [added] `analysis/run_e10_stage0.py` — Stage-0 go/no-go: full-attention vs
  blockwise-truncated CE gap G by position bucket at seq 2048/8192.
- [changed] `scripts/pretokenize_mix.py` — `--objective causal_lm` choice.
- [added] `tests/test_backbone_concept_lm.py` — 14 tests on a tiny random
  Gemma3 config (no hub): zero-init equivalence (blocks 0–1 exact vs
  single windowed forward + deliberate divergence beyond), padding/ragged blocks,
  loss==per-position mean, gradient reach (gate/LoRA at init; z0/write with open
  gates), block causality (no future leak), read-effect, ablation contract,
  encode shape, control-arm guards, save/load roundtrip, collator. Full suite
  242 passed / 9 skipped.
- [added] dependency `peft` 0.19.1 (`pyproject.toml`, `uv.lock`).
- [docs] specs E10 (+plan), E11/E12 (design-only, queued); agenda Current focus
  updated (E10 is the next run; E08 composes on top later; E09 folded into E10).

**Known deviations (recorded in the spec):** Gemma tokenizer (262K vocab)
replaces SmolLM2 ⇒ CE not comparable with E01–E09; backbone-native 1152-dim
token embeddings (the tiny-token-embedding asymmetry applies to the write-op
economics, not the frozen embedding table).

---

## [2026-07-07] - Tier-1 eval data-protocol upgrade: held-out, 2K, length-stratified, seeded

**Why:** a review of the concept-health measures found the metric *design* sound
(geometry / ΔCE-usage / faithfulness are deliberately orthogonal) but the *data
protocol* feeding them flawed: `run_concept_analysis.py` streamed the first ~320
docs of the **train** split (train-contaminated — training's holdout is a seeded
split of the same data), truncated everything to a single 512-token length (so
seq-2048 windowed E05 checkpoints were measured at a quarter of their trained
length and the L3 compression curve collapsed to one bucket), used an unseeded
`randperm` shuffle (with ~1 identity fixed point per batch, diluting Δshuffle),
judged hard Δ gates on point estimates, and silently fell back to the ModernBERT
tokenizer on load failure.

**What changed:**
- [changed] `analysis/run_concept_analysis.py` — new `--eval_source
  {holdout,pretokenized,stream}` (default `holdout` reproduces the training
  eval split via `_select_train_eval_splits`; `pretokenized` consumes a
  pretokenize-mix manifest's eval split; legacy `stream` kept behind a loud
  contamination warning); default seq **2048** with length-stratified batches
  (`--length_buckets`, per-bucket geometry + ΔCE in report/JSON); `--seed`
  seeds everything; ablation deltas reported ± per-batch std; recommendation
  verdict re-keyed on within-sample RankMe with C-scaled thresholds (was: the
  demoted slot-mean rank); tokenizer fallback removed (fails loudly,
  `--tokenizer_name` to override); JSON carries `data_protocol_version`.
- [changed] `nn/concept_encoder_perceiver.py` — `concept_ablation_ce` shuffle is
  now a random cyclic shift (guaranteed derangement, consistent with the
  specificity eval's `roll(1)`).
- [changed] `analysis/concept_analysis.py` — within-sample RankMe now also
  reports a **centered** variant to disambiguate shared-offset anisotropy
  (raw low / centered high) from genuine collapse (both low).
- [added] `tests/test_run_concept_analysis_protocol.py` — bucket parsing,
  std/per-bucket aggregation, derangement guarantee, centered-vs-raw RankMe.
- [docs] `docs/engineering_specs/concept_information_eval_upgrade.md` (dated
  section), `.cursor/skills/experiment-evaluate/SKILL.md` (Tier-1 commands +
  fixed stale doc pointer), dated "not comparable" notes in
  `master_experiment_log.md` and the E02-long / E05-attempt-3 / E05-Muon run
  reports.

**Impact:** pre-2026-07-07 Tier-1 numbers (RankMe, recon-contract Δzero/Δshuffle,
round-trip, compression curve) are internally comparable but NOT comparable with
post-upgrade numbers; recompute planned for E02-long and the E05 variants.
Training-time W&B `concept_ablation/*` (real holdout) and STS-B/SICK/PAWS/GLUE
are unaffected. Note: the derangement change also shifts training-time Δshuffle
logging slightly (removes the fixed-point dilution).

---

## [2026-06-30] - Eval-script fixes: wandb tag truncation + SmolLM2 pad_token

**Why:** the E05 attempt 3 eval pipeline (`eval-runner`) aborted at `wandb.init()`
with a 74-char tag built from the checkpoint path. Two latent bugs surfaced
together; both are pre-existing and would have blocked any SmolLM2-based
checkpoint eval, not just E05.

**Changes:**
- `evaluation/wandb_identity.py` — `_safe_tag_value` truncated the *value* to
  64 chars but callers prepended `prefix:`, so `tokenizer:<value>` could be
  up to 75 chars (wandb rejects any tag > 64). New `_safe_prefixed_tag(prefix,
  value)` reserves room for the prefix+colon and truncates the value so the
  total is always ≤ 64. New `resolve_tokenizer_name_for_tag` prefers
  `model.config.tokenizer_name` (the canonical HF id stored at training time)
  when the CLI arg is a local path — so tags read
  `tokenizer:huggingfacetb-smollm2-135m` instead of a truncated checkpoint dir.
- `evaluation/evaluate_on_benchmark.py`, `evaluation/evaluate_model_on_glue.py`
  — SmolLM2's tokenizer ships without a `pad_token`; `AutoTokenizer.from_pretrained`
  doesn't honor the model config's `pad_token_id`. All batch-encoding eval
  entrypoints now set `pad_token = eos_token` right after load. STS-B / SICK /
  PAWS / GLUE would otherwise crash with "Asking to pad but the tokenizer does
  not have a padding token".
- `tests/test_wandb_lineage.py` — 7 regression tests covering long-path
  truncation, hub-id-vs-path disambiguation, and the fallback chain.

**Impact:** eval pipeline runs end-to-end on SmolLM2 checkpoints. Commits
`730e607` (tag fix) + `70e1fd2` (pad_token fix).

---

## [2026-06-28] - Post-hoc compute audit (GPU-h, energy, tokens) + W&B compute panel

**Why:** Comparing runs on compute spent — especially within a `wandb_group`
(same experiment, varying data mix / optimization / hyperparameters) — needed
GPU-hours, total energy, and training-token numbers that W&B does not log as
scalars and HF Trainer does not provide (`train/total_flos` is 0 for the custom
`ConceptEncoder`; `include_num_input_tokens_seen` is off and would add a
per-micro-step DDP sync + startup dataloader enumeration). The data already
lives in W&B system metrics (`system/gpu.{i}.powerWatts` ~7.5 s cadence,
`_runtime`, config knobs), so a post-hoc audit covers past and future runs with
no training-loop change and no throughput tax.

**Impact:**
- New `analysis/run_compute_audit.py` — reads W&B via `run.history(stream='system',
  samples=1e6)` (full-res), integrates per-GPU power trapezoidally with real `dt`
  (gap-splitting >60 s) into `compute/energy_kwh`; computes `compute/gpu_hours`
  (`runtime × world_size / 3600`), `compute/max_tokens` (`global_step × grad_accum
  × pbs × world_size × max_seq_length`), a flagged `compute/loss_tokens_est`
  (per-family loss fraction; run-name fallback for older runs without
  `model_family`), and four derived ratios. Writes `compute/*` back into each
  finished run's W&B summary (deferred for running runs) and emits a CSV +
  matplotlib comparison chart + per-run JSON to `Cache/Evaluation_reports/compute_audit/`.
  Structural gates hard-fail (gpu-count mismatch / missing config →
  `compute/audit_state=failed`, no scalars written); plausibility gates
  write-with-flag (avg-power bounds, trapezoid-vs-avg, summary-vs-ts-span).
  `compute/group_for_panel` = `wandb_group` or architecture-prefix, so older runs
  without `wandb_group` still group in the panel.
- New `tests/test_compute_audit.py` — synthetic integrator falsification anchor
  (constant-power and linear-ramp exact-energy, gap-splitting), gate logic, token
  math, per-family loss fraction, running-run write-back deferral, offline FakeRun
  (no network). 24 tests.
- `experiment-evaluate` skill — run-level "Compute audit" preamble before Tier 0
  (W&B-only, no GPU, once per training run) so the audit fires automatically when a
  run is evaluated; script-inventory row + outputs/handoff updated.
- `experiment-track` skill — compute scalars added to "reconstruct run facts",
  the compute-budget judgment factor (prefer audited scalars; note
  `compute/audit_state`), and a `Compute` row in the run-report template.
- `docs/engineering_specs/compute_audit_wandb_panel.md` — frozen engineering spec
  (grill-me decisions, target design, W&B panel spec the user builds once, validation).

**Verified:** Dry-run + live audit on 5 runs (`concept_ar_prefix_*` E02/E05,
`concept_ar_*` E01, `perceiver_denoise_*`); `compute/*` scalars persisted to the 4
finished runs' W&B summaries (read back confirmed); numbers cross-checked (run2
290.7 GPU-h, 61.4 kWh ≈ 4×210 W×261,630 s/3.6e6; 24.5 B max-tokens). Added
`compute/max_tokens_b` (tokens in billions) so the grouped "compute profile"
panel can show GPU-hours / energy / tokens on one linear axis (raw tokens ~1e9
dominate); these are **stable absolute values** (comparable across past and future
runs — cohort-relative `%` was rejected because a future heavier run would rescale
everything), with a `_profile.png` chart mirroring the panel. Full pytest suite:
198 passed, 9 skipped. The running E05 run is `running-partial` (write-back
deferred — re-run after it finishes).

**After pull:** the W&B compute panel is built once manually in the UI from the
spec (bar charts on `compute/gpu_hours` / `compute/energy_kwh` / `compute/max_tokens`
+ a table panel, grouped by `compute/group_for_panel`); thereafter it auto-populates
for every audited run. No training-loop or model changes.

**Related:** `docs/engineering_specs/compute_audit_wandb_panel.md`,
`docs/1_Strategy_and_Plans/agenda.md` -> instrumentation / compute comparability.

## [2026-06-27] - Robust pretokenize for huge docs; pretokenized-as-standard data path

**Why:** The live mix loader (`load_dataset`) cannot cap huge recursive sources — DCLM
(27,838 `.jsonl.zst`) downloaded for ~190h, and a gigantic DCLM web doc killed a `num_proc`
tokenize worker (opaque "subprocess abruptly died"). Tokenization must be separated from
training so the long-context mix (DCLM, FinePDFs) is tractable and reusable across the E05
1-ep/5-ep arms and future phases (SFT, SFT+reasoning).

**Impact:**
- `data/dataset_preprocess.py` — `_make_tokenize_fn` gains an opt-in `max_chars` param that
  pre-truncates raw text BEFORE the Fast tokenizer scans it (lossless for the kept
  `max_seq_length` tokens; default `None` = backward-compatible).
- `scripts/pretokenize_mix.py` — passes `PRETOKENIZE_MAX_CHARS` (default 100k) and adds a
  `num_proc=1` fallback so a dead worker surfaces the real error instead of the generic
  multiprocessing message.
- `docs/experiments_specs/done_failed/E05_windowed_decoder_concept_memory.md` — switches the launch
  workflow to pretokenize → manifest → train (`PRETOKENIZED_MANIFEST`); live
  `DATASET_MIX_RECIPE` kept only as a small-dataset fallback. Documents the same spine as the
  standard data path for future phases (objective-agnostic manifest + one tokenize mode + one
  collator per phase). Records the staged 1ep/5ep proving plan, calibrated batch (8×2,
  effective 48), LR 1e-4 / warmup 1500.

**Verified:** DCLM tokenized cleanly (1.495M docs, ~11.5 min, no crash); the full e05 mix
pretokenized (7 sources, 4.93M train rows); E05 1-epoch training launched from the manifest
and passed the divergence kill-gate (step ~200: loss 14.28, grad_norm 3.4 — vs the 2026-06-26
divergence at grad_norm 500k).

**After pull:** no action for existing runs (defaults unchanged). For mix training, prefer
`scripts/pretokenize_mix.py` → `PRETOKENIZED_MANIFEST` over live `DATASET_MIX_RECIPE` for any
source with `file_glob`/`recursive`/`max_shards` (DCLM, FinePDFs, …).

## [2026-06-20] - Fix W&B MCP auth (hosted bridge)

**Why:** Local `wandb-mcp-server` pulls `wandb>=0.27.1`, whose Public API path goes through
`wandb-core` and returned `relogin required` for the project API key (while `wandb` 0.23.x /
0.27.0 still worked).

**Impact:**
- `.cursor/scripts/wandb-mcp.sh` — bridge Cursor stdio to hosted `mcp.withwandb.com` via
  `mcp-remote` + Bearer token from `.env`.
- `.cursor/mcp.json` — `bash` launcher + `envFile: ${workspaceFolder}/.env`.
- `verification/test_wandb_mcp.py` — smoke test for hosted MCP auth.
- `.cursor/skills/wandb-review/SKILL.md`, `.env.example` — updated troubleshooting.

**After pull:** run `uv run python verification/test_wandb_mcp.py`, then restart Cursor
(Tools and MCP → reload wandb server).


**Why:**
- E05 keeps the AR decoder (a good generator) but restricts its self-attention to the last K tokens,
  making the 128 concepts the ONLY carrier of cross-window context — the "scalpel" test of
  concepts-as-long-range-memory that a plain decoder swap (E04) cannot give. Scoped to **seq-len 2K**
  on a **dataset mix** per the 2026-06-17/18 discussion. Foundation only — not yet launched (gated behind E04).

**Impact:**
- **`nn/concept_encoder.py`** — new backward-compatible config `decoder_context_window: Optional[int] = None`
  (None = full causal, all prior checkpoints unchanged).
- **`nn/concept_encoder_perceiver.py`** — `build_sliding_window_causal_mask()` + `ConceptCausalDecoderStack`
  builds/caches the `[T,T]` window-causal mask and passes it to `ConceptCausalDecoderLayer._self_attention`,
  which uses `is_causal=(mask is None)` (keeps the flash path when no window). `concept_ablation_ce(...,
  window_k=K)` adds beyond/within-window deltas (`delta_{zero,shuffle}_beyond_window`) — the E05 long-range
  memory gate. **Note:** stacked window layers reach ≈ `L·(K−1)` back (depth grows the receptive field).
- **`data/dataset_preprocess.py`** — `load_and_preprocess_dataset_mix()` + `DATASET_MIXES["long_2k_base_v1"]`
  (FinePDFs-100BT 0.50 / FineWeb-Edu sample-10BT 0.30 / FineMath-3+ 0.20), weighted `interleave_datasets`;
  tokenize fn refactored to module-level `_make_tokenize_fn` (shared with the single-dataset path). No packing.
- **`training/train_perceiver_denoise.py`** — `--decoder_context_window`, `--dataset_mix`; trainer logs
  beyond-window ablation for the windowed arm. **`analysis/run_concept_analysis.py`** — `--ablation_window_k`
  for the offline windowed-vs-control A/B + a beyond-window report block. **launcher** — `DECODER_CONTEXT_WINDOW`,
  `DATASET_MIX` knobs (passed only when set; default behaviour unchanged).
- **Tests:** `tests/test_e05_windowed_decoder.py` (7, green); end-to-end MPS smoke green; full suite 140 passed
  (4 pre-existing `test_wandb_identity` failures from the E04 job_type rename — unrelated).

## [2026-06-18] - E04 parallel decoder: linear Perceiver-IO + data-contract fix + W&B clarity

**Why:**
- E04 swaps the causal-AR decoder for the parallel `perceiver_posonly` (Perceiver-IO) decoder, matched
  to the E03 anchor-OFF control. Tracing the path surfaced two blockers and a usability gap.

**Impact:**
- **`nn/concept_encoder_perceiver.py` — `PerceiverDecoderLayer` is now linear Perceiver-IO.** Removed the
  O(N²) self-attention over the N output position queries outright (no compat flag): it violated the
  project's O(C·N) bottleneck invariant and the long-context vision. Layers now cross-attend the C
  concepts + FFN only; output positions are conditionally independent given concepts (standard
  non-autoregressive decode). Also makes `AnchorDistillHead` consistent with its own "no self-attn" docstring.
- **`training/train_perceiver_denoise.py` — data-contract fix (`resolve_append_eos_token_id`).** The
  perceiver reconstruction path now appends EOS and stays variable-length (padding=False), like causal_ar.
  Previously it took the `padding="max_length"` path, and `DataCollatorForTSDAE` (which rebuilds the mask
  from row length) marked all pad positions real → the encoder attended the eos/pad tail and the decoder
  was trained to predict `<eos>` on hundreds of pad positions (a concept-free shortcut), on a *different*
  data contract than the E03 baseline. Now byte-identical to the control.
- **`training/utils_training.py` — W&B clarity.** Runs carry legible `decoder:parallel|autoregressive` and
  `task:reconstruction|generation` tags and scannable `job_type`s (`train_parallel_reconstruction`,
  `train_ar_generation_prefix_suffix`, …); parallel-recon defaults to experiment `E04`. The
  `checkpoint_family` eval-routing key is unchanged.
- **Tests:** `tests/test_perceiver_denoise.py` adds linear-decoder, EOS-append, and W&B-tag guards;
  existing collator/anchor/AR suites still green (45 tests). NOTE: dropping output self-attn changes the
  perceiver decoder state dict — old `perceiver_denoise` checkpoints with `self_attn` weights are no longer
  load-compatible (intentional; the parallel family is being retrained fresh for E04).

## [2026-06-16] - L1/L3 generation & compression faithfulness eval

**Why:**
- The eval protocol's L1 (generation faithfulness) and L3 (compression) tiers had no implementation:
  we measured whether the decoder *uses* concepts (ΔCE) but not whether the input can be *recovered
  from* the concepts, nor how recovery degrades with compression ratio.

**Impact (research eval-foundation; read-only on checkpoints, no training changes):**
- New `analysis/concept_generation_eval.py`: round-trip token recovery (teacher-forced accuracy +
  free-running exact-match / token-F1), reconstruction-vs-compression-ratio curve (bucketed by
  `⌈seq_len/C⌉`), and latent specificity (matched vs row-shuffled concepts → acc-drop + symmetric-KL).
  Reuses the model's exact teacher-forcing convention (`encode_concepts`/`_shift_right`/`decode_logits`).
- Wired into `analysis/run_concept_analysis.py` behind `--generation_eval` (default on for `concept_ar`)
  + `--free_running_examples`; prints an "L1/L3 — Generation & compression faithfulness" section and
  adds `generation_faithfulness` to the JSON report.
- Tests: `tests/test_concept_generation_eval.py` (token_f1, recovery ranges, compression bucketing,
  specificity).
- Docs: implementation pass-2 section in `engineering_specs/concept_information_eval_upgrade.md`;
  tier statuses updated in `3_Evaluations_and_Baselines/evaluation_protocol.md`. L2 SentEval/MTEB
  explicitly deferred to a remote slice (heavy deps, needs GPU).

## [2026-06-15] - Concept-information eval upgrade (probe + baselines + rank hygiene)

**Why:**
- The eval suite could not answer its own headline question ("do the 128 concepts store
  meaningful information?"): every semantic probe mean-pools the C concepts to one vector before
  scoring, so it is mathematically blind to the de-collapse E03+ chases; STS-B had no floor/ceiling
  anchor; and three different "rank" numbers (batch-avg slot rank, cross-sample RankMe, and a
  non-existent within-sample rank) were used interchangeably. Full-finetune GLUE re-routes around
  the bottleneck and was being read as concept-content evidence.

**Impact (eval-foundation only; read-only on checkpoints, no training changes):**
- **Within-sample concept-set RankMe** added (`analysis/concept_analysis.py`
  `compute_within_sample_concept_rank`) as the PRIMARY de-collapse metric; runner relabels the
  slot-mean rank as secondary and the cross-sample manifold RankMe as embedding-diversity.
- **Trivial-floor STS-B baselines** added to `evaluation/evaluate_on_benchmark.py`
  (`--baseline token_embed_mean|teacher_hidden_mean`) so STS-B numbers are interpretable.
- **Attention-pool readout** added to `ConceptEncoderForSentencePairClassification`
  (`pool_mode=mean|attention`, `AttentionPool`) + `--pool_mode` on both eval CLIs — the
  frozen-encoder probe that makes distributed-across-concepts information visible.
- `experiment-evaluate` skill rewritten: rank disambiguation, Tier-2 floors, new Tier 2.5 probe,
  GLUE full-finetune demoted from concept-content evidence.
- Tests: `tests/test_concept_manifold_metrics.py` (within-sample rank), `tests/test_sentence_pair_pool_modes.py`.
- Spec/plan: `docs/3_Evaluations_and_Baselines/concept_information_eval_upgrade.md`.
- Backward-compatible: absent flags reproduce prior numbers; `pool_mode='mean'` is byte-identical.

## [2026-06-14] - Make E03 anchor runs W&B-identifiable

**Why:**
- The live E03 anchor-ON warmup on Odra was launched with `anchor_loss=true` but inherited the E01 W&B group/tag identity, making it hard to separate from the AR reconstruction baseline.

**Impact:**
- Future anchor-enabled concept-AR runs self-label as E03 anchor runs in W&B, and the run checklist now requires verifying group, job type, tags, and `experiment_id` before training.

**What changed:**
- [fixed] `training/utils_training.py`, `training/train_perceiver_denoise.py` - make W&B identity anchor-aware (`E03`, `train_concept_ar_anchor_reconstruction`, `anchor`/`anchor-on` tags) and log anchor config explicitly.
- [updated] `.cursor/skills/experiment-run/SKILL.md`, `docs/experiments_specs/done_success/E03_concept_anchor_decollapse.md`, `docs/experiments_specs/done_success/E03_concept_anchor_decollapse_plan.md` - add W&B preflight guidance and `EXPERIMENT_ID=E03` to E03 launch recipes, including the matched control.
- [added] `tests/test_wandb_identity.py` - regression coverage for E03 anchor W&B identity.

## [2026-06-13] - Standardize W&B identity for shared perceiver/AR training

**Why:**
- E01 (`concept_ar`) and E02 (`concept_ar_prefix`) run through the same shared training entrypoint, but W&B still logged them with the legacy `perceiver_denoise` group and job type. That made model families and objectives hard to separate in the run table.

**Impact:**
- New W&B training runs expose experiment/model/objective identity directly in `group`, `job_type`, tags, and config. E01/E02 retries now group separately by experiment plus architecture while preserving timestamped run ids.

**What changed:**
- [added] `training/utils_training.py` - `WandbRunIdentity` and `build_perceiver_wandb_identity()` derive stable W&B group/job_type/tags/config facets for the shared perceiver/AR entrypoint.
- [updated] `training/train_perceiver_denoise.py` - uses the identity helper for W&B metadata; logs `experiment_id`, `model_family`, `objective_family`, `architecture_id`, `checkpoint_family`, and `pretraining_objective`; removes misleading `perceiver-denoise` tags from AR runs.
- [added] `tests/test_wandb_identity.py` - regression tests for legacy perceiver denoise, E01, E02, and explicit experiment-id overrides.
- [updated] `.cursor/skills/wandb-review/SKILL.md`, `docs/1_Strategy_and_Plans/training_eval_matrix.md` - document the W&B grouping contract and maintained `concept_ar` family.

## [2026-06-13] — E02 warm-up follow-ups: deterministic data split, single DDP run_id, early-suffix ablation + live effective-rank, linear-probe eval

**Why:**
- Full evaluation of the E02 0.3-epoch warm-up (`concept_ar_prefix_H768L6C128D4_20260612_094555`) was promising (zero-shot STS-B **0.683**, beats prior 0.607) but surfaced several issues before scaling to a full run:
  - The no-`seed` `train_test_split` reshuffled the holdout every launch, so the tokenization `.map()` cache never hit (the warm-up re-tokenized all 9.57M FineWeb-Edu rows, ~1.5–2 h) and each DDP rank built a *different* train/eval split.
  - `run_identifier` was computed from `datetime.now()` inside `main()`, so DDP ranks straddling a second boundary forked into two output dirs (`…094555`/`…094600`) with duplicated checkpoints and "Could not locate the best model" warnings.
  - The averaged suffix-CE concept-ablation Δ is diluted by teacher-forced AR self-context (late positions predictable without concepts), under-measuring concept usage; effective rank (collapse gate) was only computed offline.
  - Supervised sentence-pair eval (`concept_ar`) was near chance across SICK/PAWS/MRPC despite strong zero-shot STS-B — it full-fine-tunes the lightly-pretrained encoder at LR 1e-5 on tiny datasets, destroying the pretrained geometry.
  - Eval run names labelled the encoder-only sub-model as e.g. `73M`, implying the checkpoint is that small (full model is 161.6M).

**Impact:**
- Full-corpus runs reuse the tokenization cache (no re-tokenize per launch) and all DDP ranks share one split and one `run_id`/output dir. The full E02 run logs sharper concept-usage signals (early-suffix Δ) and live concept geometry (effective rank) each eval. A robust linear-probe path is available for trustworthy supervised concept-quality measurement.

**What changed:**
- [fixed] `data/dataset_preprocess.py` — `_select_train_eval_splits`/`load_and_preprocess_text_dataset` take a `split_seed` (default 42) passed to `train_test_split(seed=...)`; `training/train_perceiver_denoise.py` passes `training_args.seed`. Deterministic split → reusable tokenization cache and rank-consistent splits.
- [fixed] `training/utils_training.py` — added `broadcast_object()` (rank-0 broadcast via `broadcast_object_list`, no-op when not distributed); `train_perceiver_denoise.py` broadcasts `run_identifier` so all ranks share one output dir/W&B run.
- [added] `nn/concept_encoder_perceiver.py` — `concept_ablation_ce()` now also returns early-position metrics (`ce_intact_early`, `delta_zero_early`, `delta_shuffle_early`, default first `early_k=16` suffix tokens) via new `_teacher_forced_ce_early`.
- [added] `training/train_perceiver_denoise.py` — `PerceiverDenoiseTrainer` logs `concept_geometry/effective_rank(_normalized)` each eval (SVD nuclear/spectral norm of the mean concept matrix, matching `analysis/concept_analysis`).
- [added] `evaluation/evaluate_on_benchmark.py`, `evaluation/evaluate_model_on_glue.py`, `scripts/evaluate_concept_encoder_glue.sh` — `--freeze_encoder` (linear probe; `FREEZE_ENCODER=1` for the GLUE launcher) freezes the encoder and trains only the task head; eval run/report names mark encoder-only models with a `-enc` suffix.

## [2026-06-11] — Fix double-shift in AR teacher-forced CE (skip-one objective bug)

**Why:**
- `ConceptEncoderForConditionalLM` builds decoder inputs with `_shift_right` (`[bos, x0..x_{N-2}]`, T5 convention: `logits[t]` predicts `x_t`) **and** the loss helper shifted again (GPT convention: `logits[:, :-1]` vs `labels[:, 1:]`). Net effect: every target `x_t` was predicted from context ending at `x_{t-2}` — the decoder never saw the immediately preceding token of its target. This trained a harder "skip-one" objective, inflated all CE numbers, explains the at-chance no-concept floor in the E01 warm-up (a pure next-next-token LM is far weaker), and made the greedy generation loop (single-shift convention) inconsistent with training. The E02 plan even listed this exact risk ("loss shift is off by one for suffix generation").

**Impact:**
- All `concept_ar` losses (training CE, eval CE, concept-ablation CE, `ce_intact_wd`) now measure true next-token teacher forcing; generation and training conventions agree. CE values are not comparable with the buggy warm-up / first relaunch numbers. E01 was relaunched from scratch on the fixed code; E02 inherits the fix before its first run.

**What changed:**
- [fixed] `nn/concept_encoder_perceiver.py` — `_next_token_ce` → `_teacher_forced_ce`: plain `CE(logits, labels)` with `ignore_index=-100`, no second shift (decoder inputs are already shift-right-ed). All call sites updated (forward loss + concept ablation, reconstruction and prefix/suffix paths).
- [added] `tests/test_concept_ar_decoder.py::test_loss_is_single_shift_teacher_forcing` — contract regression test (reconstruction + prefix/suffix), fails on the old double-shift code.

## [2026-06-11] — E01 eval-protocol fixes: matched word-dropout CE, deterministic eval corruption, pad=eos analysis labels

**Why:**
- The E01 warm-up run (`concept_ar_H768L6C128D4_20260607_172931`) showed eval CE *rising* (6.82 → 9.0) while train CE fell (10.2 → 3.1) on a single data pass — impossible as overfitting, so a train/eval protocol mismatch. Diagnosis: training applies decoder word-dropout p=0.4 while eval scores with clean decoder inputs; the decoder specializes to the blanked-input distribution and the clean condition becomes out-of-distribution (supported by ce_zero ≈ ln(vocab): the decoder learned no pure-LM use of its left context). Two further eval-trust issues from code review: TSDAE deletion is resampled every eval call (noisy `eval_loss`, lucky best-checkpoint selection), and `run_concept_analysis.py` masked labels by token id — with SmolLM2 pad=eos that silently dropped every real eos target.

**Impact:**
- E01/E02 eval numbers become trustworthy: eval CE is now also measured under the train-matched word-dropout condition (`ce_intact_wd`, `gap_clean_vs_wd` — a large gap flags the OOD mismatch directly), held-out corruption is deterministic so `eval_loss` is comparable across evaluations and runs, and offline concept-analysis CE agrees with the training-eval label contract for pad=eos tokenizers.

**What changed:**
- [updated] `nn/concept_encoder_perceiver.py` — `ConceptCausalDecoderStack.embed()` honors an explicitly passed `word_dropout_p` regardless of train/eval mode (callers still gate the training default); `concept_ablation_ce()` additionally returns `ce_intact_wd` (intact concepts, train-matched word-dropout) and `gap_clean_vs_wd` when the config has `decoder_word_dropout > 0`.
- [updated] `data/data_collators.py` — `DataCollatorForTSDAE` and `DataCollatorForPrefixGeneration` accept `seed`; when set, deletion masks / prefix-suffix split points are a pure function of (seed, batch content) for reproducible eval corruption.
- [updated] `training/train_perceiver_denoise.py` — trainer takes a separate seeded `eval_data_collator` (swapped in via `get_eval_dataloader`); concept-ablation aggregation passes through whatever metric keys the model returns.
- [fixed] `analysis/run_concept_analysis.py` — ablation labels now mask padding positionally via `attention_mask` instead of by `pad_token_id` (pad=eos safe); prints the matched-word-dropout CE and the clean-vs-wd gap; documents that offline ablation encodes the full clean sequence (absolute CE not comparable with training eval).
- [added] tests: seeded-collator determinism, unseeded resampling, pad=eos TSDAE label/visibility contract (`tests/test_tsdae_collator.py`); eval-mode forced word-dropout, `ce_intact_wd` reporting (`tests/test_concept_ar_decoder.py`).

**Related:** `docs/experiments_specs/done_failed/E01_concept_ar_decoder.md` (rerun uses these fixes), E01 warm-up review (eval CE divergence diagnosis)

## [2026-06-14] - Make E03 anchor runs W&B-identifiable

**Why:**
- The live E03 anchor-ON warmup on Odra was launched with `anchor_loss=true` but inherited the E01 W&B group/tag identity, making it hard to separate from the AR reconstruction baseline.

**Impact:**
- Future anchor-enabled concept-AR runs self-label as E03 anchor runs in W&B, and the run checklist now requires verifying group, job type, tags, and `experiment_id` before training.

**What changed:**
- [fixed] `training/utils_training.py`, `training/train_perceiver_denoise.py` - make W&B identity anchor-aware (`E03`, `train_concept_ar_anchor_reconstruction`, `anchor`/`anchor-on` tags) and log anchor config explicitly.
- [updated] `.cursor/skills/experiment-run/SKILL.md`, `docs/experiments/E03_concept_anchor_decollapse.md`, `docs/experiments/E03_concept_anchor_decollapse_plan.md` - add W&B preflight guidance and `EXPERIMENT_ID=E03` to E03 launch recipes, including the matched control.
- [added] `tests/test_wandb_identity.py` - regression coverage for E03 anchor W&B identity.

## [2026-06-13] - Standardize W&B identity for shared perceiver/AR training

**Why:**
- E01 (`concept_ar`) and E02 (`concept_ar_prefix`) run through the same shared training entrypoint, but W&B still logged them with the legacy `perceiver_denoise` group and job type. That made model families and objectives hard to separate in the run table.

**Impact:**
- New W&B training runs expose experiment/model/objective identity directly in `group`, `job_type`, tags, and config. E01/E02 retries now group separately by experiment plus architecture while preserving timestamped run ids.

**What changed:**
- [added] `training/utils_training.py` - `WandbRunIdentity` and `build_perceiver_wandb_identity()` derive stable W&B group/job_type/tags/config facets for the shared perceiver/AR entrypoint.
- [updated] `training/train_perceiver_denoise.py` - uses the identity helper for W&B metadata; logs `experiment_id`, `model_family`, `objective_family`, `architecture_id`, `checkpoint_family`, and `pretraining_objective`; removes misleading `perceiver-denoise` tags from AR runs.
- [added] `tests/test_wandb_identity.py` - regression tests for legacy perceiver denoise, E01, E02, and explicit experiment-id overrides.
- [updated] `.cursor/skills/wandb-review/SKILL.md`, `docs/1_Strategy_and_Plans/training_eval_matrix.md` - document the W&B grouping contract and maintained `concept_ar` family.

## [2026-06-13] — E02 warm-up follow-ups: deterministic data split, single DDP run_id, early-suffix ablation + live effective-rank, linear-probe eval

**Why:**
- Full evaluation of the E02 0.3-epoch warm-up (`concept_ar_prefix_H768L6C128D4_20260612_094555`) was promising (zero-shot STS-B **0.683**, beats prior 0.607) but surfaced several issues before scaling to a full run:
  - The no-`seed` `train_test_split` reshuffled the holdout every launch, so the tokenization `.map()` cache never hit (the warm-up re-tokenized all 9.57M FineWeb-Edu rows, ~1.5–2 h) and each DDP rank built a *different* train/eval split.
  - `run_identifier` was computed from `datetime.now()` inside `main()`, so DDP ranks straddling a second boundary forked into two output dirs (`…094555`/`…094600`) with duplicated checkpoints and "Could not locate the best model" warnings.
  - The averaged suffix-CE concept-ablation Δ is diluted by teacher-forced AR self-context (late positions predictable without concepts), under-measuring concept usage; effective rank (collapse gate) was only computed offline.
  - Supervised sentence-pair eval (`concept_ar`) was near chance across SICK/PAWS/MRPC despite strong zero-shot STS-B — it full-fine-tunes the lightly-pretrained encoder at LR 1e-5 on tiny datasets, destroying the pretrained geometry.
  - Eval run names labelled the encoder-only sub-model as e.g. `73M`, implying the checkpoint is that small (full model is 161.6M).

**Impact:**
- Full-corpus runs reuse the tokenization cache (no re-tokenize per launch) and all DDP ranks share one split and one `run_id`/output dir. The full E02 run logs sharper concept-usage signals (early-suffix Δ) and live concept geometry (effective rank) each eval. A robust linear-probe path is available for trustworthy supervised concept-quality measurement.

**What changed:**
- [fixed] `data/dataset_preprocess.py` — `_select_train_eval_splits`/`load_and_preprocess_text_dataset` take a `split_seed` (default 42) passed to `train_test_split(seed=...)`; `training/train_perceiver_denoise.py` passes `training_args.seed`. Deterministic split → reusable tokenization cache and rank-consistent splits.
- [fixed] `training/utils_training.py` — added `broadcast_object()` (rank-0 broadcast via `broadcast_object_list`, no-op when not distributed); `train_perceiver_denoise.py` broadcasts `run_identifier` so all ranks share one output dir/W&B run.
- [added] `nn/concept_encoder_perceiver.py` — `concept_ablation_ce()` now also returns early-position metrics (`ce_intact_early`, `delta_zero_early`, `delta_shuffle_early`, default first `early_k=16` suffix tokens) via new `_teacher_forced_ce_early`.
- [added] `training/train_perceiver_denoise.py` — `PerceiverDenoiseTrainer` logs `concept_geometry/effective_rank(_normalized)` each eval (SVD nuclear/spectral norm of the mean concept matrix, matching `analysis/concept_analysis`).
- [added] `evaluation/evaluate_on_benchmark.py`, `evaluation/evaluate_model_on_glue.py`, `scripts/evaluate_concept_encoder_glue.sh` — `--freeze_encoder` (linear probe; `FREEZE_ENCODER=1` for the GLUE launcher) freezes the encoder and trains only the task head; eval run/report names mark encoder-only models with a `-enc` suffix.

## [2026-06-11] — Fix double-shift in AR teacher-forced CE (skip-one objective bug)

**Why:**
- `ConceptEncoderForConditionalLM` builds decoder inputs with `_shift_right` (`[bos, x0..x_{N-2}]`, T5 convention: `logits[t]` predicts `x_t`) **and** the loss helper shifted again (GPT convention: `logits[:, :-1]` vs `labels[:, 1:]`). Net effect: every target `x_t` was predicted from context ending at `x_{t-2}` — the decoder never saw the immediately preceding token of its target. This trained a harder "skip-one" objective, inflated all CE numbers, explains the at-chance no-concept floor in the E01 warm-up (a pure next-next-token LM is far weaker), and made the greedy generation loop (single-shift convention) inconsistent with training. The E02 plan even listed this exact risk ("loss shift is off by one for suffix generation").

**Impact:**
- All `concept_ar` losses (training CE, eval CE, concept-ablation CE, `ce_intact_wd`) now measure true next-token teacher forcing; generation and training conventions agree. CE values are not comparable with the buggy warm-up / first relaunch numbers. E01 was relaunched from scratch on the fixed code; E02 inherits the fix before its first run.

**What changed:**
- [fixed] `nn/concept_encoder_perceiver.py` — `_next_token_ce` → `_teacher_forced_ce`: plain `CE(logits, labels)` with `ignore_index=-100`, no second shift (decoder inputs are already shift-right-ed). All call sites updated (forward loss + concept ablation, reconstruction and prefix/suffix paths).
- [added] `tests/test_concept_ar_decoder.py::test_loss_is_single_shift_teacher_forcing` — contract regression test (reconstruction + prefix/suffix), fails on the old double-shift code.

## [2026-06-11] — E01 eval-protocol fixes: matched word-dropout CE, deterministic eval corruption, pad=eos analysis labels

**Why:**
- The E01 warm-up run (`concept_ar_H768L6C128D4_20260607_172931`) showed eval CE *rising* (6.82 → 9.0) while train CE fell (10.2 → 3.1) on a single data pass — impossible as overfitting, so a train/eval protocol mismatch. Diagnosis: training applies decoder word-dropout p=0.4 while eval scores with clean decoder inputs; the decoder specializes to the blanked-input distribution and the clean condition becomes out-of-distribution (supported by ce_zero ≈ ln(vocab): the decoder learned no pure-LM use of its left context). Two further eval-trust issues from code review: TSDAE deletion is resampled every eval call (noisy `eval_loss`, lucky best-checkpoint selection), and `run_concept_analysis.py` masked labels by token id — with SmolLM2 pad=eos that silently dropped every real eos target.

**Impact:**
- E01/E02 eval numbers become trustworthy: eval CE is now also measured under the train-matched word-dropout condition (`ce_intact_wd`, `gap_clean_vs_wd` — a large gap flags the OOD mismatch directly), held-out corruption is deterministic so `eval_loss` is comparable across evaluations and runs, and offline concept-analysis CE agrees with the training-eval label contract for pad=eos tokenizers.

**What changed:**
- [updated] `nn/concept_encoder_perceiver.py` — `ConceptCausalDecoderStack.embed()` honors an explicitly passed `word_dropout_p` regardless of train/eval mode (callers still gate the training default); `concept_ablation_ce()` additionally returns `ce_intact_wd` (intact concepts, train-matched word-dropout) and `gap_clean_vs_wd` when the config has `decoder_word_dropout > 0`.
- [updated] `data/data_collators.py` — `DataCollatorForTSDAE` and `DataCollatorForPrefixGeneration` accept `seed`; when set, deletion masks / prefix-suffix split points are a pure function of (seed, batch content) for reproducible eval corruption.
- [updated] `training/train_perceiver_denoise.py` — trainer takes a separate seeded `eval_data_collator` (swapped in via `get_eval_dataloader`); concept-ablation aggregation passes through whatever metric keys the model returns.
- [fixed] `analysis/run_concept_analysis.py` — ablation labels now mask padding positionally via `attention_mask` instead of by `pad_token_id` (pad=eos safe); prints the matched-word-dropout CE and the clean-vs-wd gap; documents that offline ablation encodes the full clean sequence (absolute CE not comparable with training eval).
- [added] tests: seeded-collator determinism, unseeded resampling, pad=eos TSDAE label/visibility contract (`tests/test_tsdae_collator.py`); eval-mode forced word-dropout, `ce_intact_wd` reporting (`tests/test_concept_ar_decoder.py`).

**Related:** `docs/experiments/E01_concept_ar_decoder.md` (rerun uses these fixes), E01 warm-up review (eval CE divergence diagnosis)

## [2026-06-06] — Complete the research pipeline: add `implementation-plan`, remove duplicate spec index

**Why:**
- The skill set had a gap between *framing* an experiment (`experiment-design`) and *writing code* (`research-implement`): nothing produced a detailed, repo-rooted implementation plan (which modules to reuse, forward pass with shapes, data, loss, config, snippets) — the research analog of a PRD. Also, the per-experiment `docs/experiments_specs/README.md` carried a manual Index/Status table that duplicated the canonical results ledger (`master_experiment_log.md`) and would drift.

**Impact:**
- The pipeline is now explicit and complete: `research-scout` → `research-explain` → `research-synthesis` → `experiment-design` → `implementation-plan` → `research-implement` → run → `experiment-track`. Canonical homes are unambiguous: intent → specs/plans, results → `master_experiment_log.md`, live memory → `agenda.md`.

**What changed:**
- [added] `.cursor/skills/implementation-plan/SKILL.md` — the bridge skill; writes `docs/experiments_specs/<ID>_plan.md` (reuse map, forward pass with shapes, inputs/data, loss/objective, config + launch, tests, risks, optional code sketches), rooted in real repo classes.
- [added] `docs/experiments_specs/PLAN_TEMPLATE.md` — template for `<ID>_plan.md`.
- [updated] `experiment-design`, `research-implement` (now reads spec **and** `<ID>_plan.md`), `experiment-discipline.mdc` (Roles = full pipeline order), `research-synthesis` (handoff to design/plan), `project-overview.mdc` (compact Research Pipeline map + canonical-homes note), `docs/experiments_specs/README.md` (two-file model, self-indexing, where-things-live), `docs/experiments_specs/TEMPLATE.md` (link to plan).
- [removed] the manual `## Index` table in `docs/experiments_specs/README.md` — `master_experiment_log.md` stays the single results ledger; the experiments folder is self-indexing.

## [2026-06-05] — Experiment-system consolidation: slim agenda, scoped specs, foundation audit

**Why:**
- This is a process change, not a direction change. Experiments had blended together: a 700-line `roadmap.md` read as a committed schedule, an 835-line `active_todos.md` result-diary, 8 tracks (A–H), and a forked `train_*.py` / `nn/concept_encoder_*.py` per idea — which made runs hard to interpret. The goal is smaller, well-defined increments built on the existing foundation. The long-term Vision is unchanged; the direction within it stays open and exploratory.

**Impact:**
- A slim living agenda + per-experiment frozen specs replace the monolith. Experiments become small increments expressed as args/configs over the shared foundation rather than new forks (encouraged by a rule + the `experiment-design` skill). Dead code removed; recursive + diffusion families parked but revivable.

**What changed:**
- [added] `docs/1_Strategy_and_Plans/agenda.md` — slim living agenda (the process, current focus, candidate directions, neutral "what we've explored" learnings). New daily driver.
- [added] `docs/experiments_specs/` — `TEMPLATE.md` (frozen spec format) + `README.md` (lifecycle, ID scheme `E0NN_slug`).
- [added] `.cursor/skills/experiment-design/SKILL.md` — front-half skill: hypothesis → one minimal spec before code.
- [added] `.cursor/rules/experiment-discipline.mdc` — always-applied guardrail: spec-before-code, configs-over-forks, one variable at a time.
- [archived] `roadmap.md` → `docs/5_Archive/roadmap_v5_20260301.md`, `active_todos.md` → `docs/5_Archive/active_todos_v3_20260314.md` (OBSOLETE banners).
- [updated] `project-overview.mdc`, `experiment-track`, `engineering-change-tracking`, `docs-hygiene`, `research-synthesis`, `remote-experiment-evaluator`, `CHANGELOG.md`, `training_eval_matrix.md`, `vision_and_goals.md` — repointed from `roadmap.md`/`active_todos.md` to `agenda.md` + `docs/experiments_specs/`.
- [renamed] `.cursor/skills/pytorch-architecture/` → `.cursor/skills/research-implement/` — rewritten from generic PyTorch guidance into a codebase-grounded implementation skill (module map, encode→reason→decode patterns, configs-over-forks, training entrypoint + bash-launcher mechanics, and a hard reproducibility rule: never delete old code/checkpoints — park instead). It is the implementation half of the spec→code workflow.
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
