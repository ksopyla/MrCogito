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
