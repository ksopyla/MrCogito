# Training and Evaluation Matrix

**Last updated:** 2026-06-13

This table is the current source of truth for which training paths are actively maintained, which are parked, and which interfaces are retired.

## Current Matrix

| Family / path | Status | Training entrypoint | Objective | Single-input evaluation | Pair-input evaluation | Zero-shot STS-B | Notes |
|---|---|---|---|---|---|---|---|
| `perceiver_denoise` | Maintained | `training/train_perceiver_denoise.py` | TSDAE-style token deletion + full-sequence reconstruction, optional `reconstruction+contrastive` | `ViaDecoder` | `ConceptEncoderForSentencePairClassification` with separate encoding | Yes | Canonical perceiver research path. Uses BiXT + stacked position-only decoder. |
| `concept_ar` / `concept_ar_prefix` | Maintained current focus | `training/train_perceiver_denoise.py --decoder_type causal_ar` | E01: AR denoising reconstruction; E02: prefix-to-suffix AR generation (`--objective_variant prefix_suffix`) | Weighted concept pooling over encoder concepts | Sentence-pair route with separate encoding | Yes | Shared encoder->AR-decoder foundation. New W&B runs log distinct groups/job types such as `E01_concept_ar_H...` and `E02_concept_ar_prefix_H...`. |
| `weighted_mlm` | Maintained baseline | `training/train_mlm.py --model_type weighted_mlm` | Sparse MLM | Weighted concept pooling classifier | Legacy sentence-pair route from `evaluate_model_on_glue.py` | No dedicated canonical route | Kept as the simple MLM baseline for comparison. |
| `diffusion_mlm` | **Parked** (`parked/`) | `parked/training/train_diffusion.py` | Masked diffusion self-reconstruction | — | — | — | Explored; concept rank stayed low so far. Set aside for now, likely to revisit. Existing checkpoints still evaluate via encoder-only routing into perceiver heads. See `parked/README.md`. |
| `prefix_diffusion` | **Parked** (`parked/`) | `parked/training/train_prefix_diffusion.py` | Prefix-to-suffix diffusion generation | — | — | — | Explored (random init) with low concept rank so far. Set aside, to revisit — especially with warm-start. See `parked/README.md`. |
| `recursive_mlm` | **Parked** (`parked/`) | `parked/training/train_recursive_mlm.py` | Recursive sparse MLM with weight-tied encoder | — | — | — | Not the current focus; recursive / latent reasoning stays part of the long-term Vision. See `parked/README.md`. |

## Retired Interfaces

| Interface | Retired on | Reason |
|---|---|---|
| `perceiver_mlm` | 2026-03-08 | Retired as an active training family. Historical checkpoints/results remain valid, but the maintained perceiver path is now `perceiver_denoise`. |
| `perceiver_posonly_mlm` | 2026-03-08 | Folded into the denoising-first perceiver path. The old name described decoder mechanics, not the maintained research objective. |
| `perceiver_decoder_cls` | 2026-03-08 | Replaced by checkpoint-declared canonical evaluation routing instead of manual evaluator aliases. |
| `training/train_tsdae.py` | 2026-03-08 | Removed because it was never a distinct trained/evaluated path after the reset; it only duplicated the denoising entrypoint under an obsolete name. |
| `scripts/train_perceiver_mlm.ps1` | 2026-03-08 | Removed to avoid implying that perceiver MLM is still a maintained launcher. |
| `scripts/train_mlm_multigpu_perceiver.sh` | 2026-03-08 | Removed for the same reason; replaced by `scripts/train_perceiver_denoise_multigpu.sh`. |
| `scripts/test_tsdae_local.ps1` | 2026-03-08 | Replaced by `scripts/test_perceiver_denoise_local.ps1`. |

## Evaluation Rules

| Checkpoint family | Canonical single-input route | Canonical pair-input route | Notes |
|---|---|---|---|
| `perceiver_denoise` | `ViaDecoder` | Separate sentence encoding | Stored in checkpoint metadata and enforced by routing code. |
| `concept_ar` | Weighted concept pooling | Separate sentence encoding | AR checkpoints have no position-only decoder; concept-quality probes use encoder concepts directly. |
| `diffusion_mlm` / `prefix_diffusion` | Metadata-driven | Separate sentence encoding | Missing metadata should fail loudly instead of silently falling back. |
| `weighted_mlm` | Weighted concept pooling | Legacy pair route | Baseline path, not the semantic-first contract. |
| `recursive_mlm` | Manual | Manual | Not yet a canonical benchmark family. |

## Practical Summary

- If you want the maintained perceiver path, use `perceiver_denoise`.
- If you want the current encoder->AR decoder path, use `concept_ar` (`reconstruction`) or `concept_ar_prefix` (`prefix_suffix`) through `training/train_perceiver_denoise.py`.
- If you want the simple MLM baseline, use `weighted_mlm`.
- Recursion and diffusion are **parked** in `parked/` (revivable) — see `parked/README.md`.
- Current work is tracked as small, well-defined increments in `agenda.md` + `docs/experiments/`; the direction is exploratory.
- If you need historical `perceiver_mlm` comparisons, use old checkpoints/results as archived baselines, not as current launch interfaces.
