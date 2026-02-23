# Changelog

All notable engineering and architecture changes are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

**Relationship to other docs:**
- This file: *What* changed in code and *when* (engineering log)
- `docs/2_Experiments_Registry/master_experiment_log.md`: *What* training runs produced which results (science log)
- `docs/1_Strategy_and_Plans/active_todos.md`: *What* to do next (planning log)

The `git_tag` column in the master experiment log links each training run to the
exact code version. Tag format: `arch/{feature}` for architecture changes,
`train/{run_id}` before launching a training run.

---

## [Unreleased]

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
- `training/mlm_training.py`: added `torch_compile_dynamic` flag (fixes step 8000
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
