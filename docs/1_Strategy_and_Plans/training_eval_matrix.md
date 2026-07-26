# Training and Evaluation Matrix

**Last updated:** 2026-07-11

This table is the current source of truth for which training paths are actively maintained, which are parked, and which interfaces are retired.

## Current Matrix

| Family / path | Status | Training entrypoint | Objective | Single-input evaluation | Pair-input evaluation | Zero-shot STS-B | Notes |
|---|---|---|---|---|---|---|---|
| `perceiver_denoise` | Maintained | `training/train_concept_pretraining.py` | TSDAE-style token deletion + full-sequence reconstruction, optional `reconstruction+contrastive` | `ViaDecoder` | `ConceptEncoderForSentencePairClassification` with separate encoding | Yes | Canonical perceiver research path. Uses BiXT + stacked position-only decoder. |
| `concept_ar` / `concept_ar_prefix` | Maintained current focus | `training/train_concept_pretraining.py --decoder_type causal_ar` | E01: AR denoising reconstruction; E02: prefix-to-suffix AR generation (`--objective_variant prefix_suffix`) | Weighted concept pooling over encoder concepts | Sentence-pair route with separate encoding | Yes | Shared encoder->AR-decoder foundation. New W&B runs log distinct groups/job types such as `E01_concept_ar_H...` and `E02_concept_ar_prefix_H...`. |
| `weighted_mlm` | Historical baseline; trainer parked | `parked/training/train_weighted_mlm.py` (reproduction only) | Sparse MLM | Weighted concept pooling classifier | Legacy sentence-pair route from `evaluate_model_on_glue.py` | No dedicated canonical route | No new training is planned. `nn/concept_encoder_weighted.py` and evaluation routing remain live so the 18 historical W&B runs/checkpoints stay loadable. |
| `diffusion_mlm` | **Parked** (`parked/`) | `parked/training/train_diffusion.py` | Masked diffusion self-reconstruction | — | — | — | Explored; concept rank stayed low so far. Set aside for now, likely to revisit. Existing checkpoints still evaluate via encoder-only routing into perceiver heads. See `parked/README.md`. |
| `prefix_diffusion` | **Parked** (`parked/`) | `parked/training/train_prefix_diffusion.py` | Prefix-to-suffix diffusion generation | — | — | — | Explored (random init) with low concept rank so far. Set aside, to revisit — especially with warm-start. See `parked/README.md`. |

## Retired Interfaces

| Interface | Retired on | Reason |
|---|---|---|
| `perceiver_mlm` | 2026-03-08 | Retired as an active training family. Historical checkpoints/results remain valid, but the maintained perceiver path is now `perceiver_denoise`. |
| `perceiver_posonly_mlm` | 2026-03-08 | Folded into the denoising-first perceiver path. The old name described decoder mechanics, not the maintained research objective. |
| `perceiver_decoder_cls` | 2026-03-08 | Replaced by checkpoint-declared canonical evaluation routing instead of manual evaluator aliases. |
| `training/train_tsdae.py` | 2026-03-08 | Removed because it was never a distinct trained/evaluated path after the reset; it only duplicated the denoising entrypoint under an obsolete name. |
| `training/train_mlm.py` | 2026-07-11 | New sparse-MLM runs are no longer planned. The trainer moved to `parked/training/train_weighted_mlm.py` for reproduction; weighted checkpoint evaluation remains supported. |
| `recursive_mlm` implementation | 2026-07-11 | The isolated TRM-style sparse-MLM fork had no recorded W&B/ledger run and was superseded by config-selectable recurrent-memory work on the maintained AR foundation (E09). Code remains recoverable from git history and `pre-consolidation-20260605`. |
| `scripts/train_perceiver_mlm.ps1` | 2026-03-08 | Removed to avoid implying that perceiver MLM is still a maintained launcher. |
| `scripts/train_mlm_multigpu_perceiver.sh` | 2026-03-08 | Removed for the same reason; its maintained successor is now `scripts/train_concept_pretraining_multigpu.sh`. |
| `scripts/test_tsdae_local.ps1` | 2026-03-08 | Replaced by `scripts/test_perceiver_denoise_local.ps1`. |

## Evaluation Rules

| Checkpoint family | Canonical single-input route | Canonical pair-input route | Notes |
|---|---|---|---|
| `perceiver_denoise` | `ViaDecoder` | Separate sentence encoding | Stored in checkpoint metadata and enforced by routing code. |
| `concept_ar` | Weighted concept pooling | Separate sentence encoding | AR checkpoints have no position-only decoder; concept-quality probes use encoder concepts directly. |
| `diffusion_mlm` / `prefix_diffusion` | Metadata-driven | Separate sentence encoding | Missing metadata should fail loudly instead of silently falling back. |
| `weighted_mlm` | Weighted concept pooling | Legacy pair route | Baseline path, not the semantic-first contract. |

## Practical Summary

- If you want the maintained perceiver path, use `perceiver_denoise`.
- If you want the maintained concept-pretraining path, use `training/train_concept_pretraining.py`; select `concept_ar` (`reconstruction`) or `concept_ar_prefix` (`prefix_suffix`) with its decoder/objective flags. The old `training/train_perceiver_denoise.py` path is a temporary compatibility wrapper.
- For the simple MLM baseline, evaluate the existing `weighted_mlm` checkpoints; its trainer is
  parked for reproduction, not new experiments.
- Diffusion and prefix diffusion are **parked and revivable**. The old recursive-MLM fork is
  retired; recurrent research continues through E09-style components on the maintained foundation.
- Current work is tracked as small, well-defined increments in `agenda.md` + `docs/experiments_specs/`; the direction is exploratory.
- If you need historical `perceiver_mlm` comparisons, use old checkpoints/results as archived baselines, not as current launch interfaces.
