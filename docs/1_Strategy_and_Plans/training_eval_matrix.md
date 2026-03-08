# Training and Evaluation Matrix

**Last updated:** 2026-03-08

This table is the current source of truth for which training paths are actively maintained, which are isolated experiments, and which interfaces are retired.

## Current Matrix

| Family / path | Status | Training entrypoint | Objective | Single-input evaluation | Pair-input evaluation | Zero-shot STS-B | Notes |
|---|---|---|---|---|---|---|---|
| `perceiver_denoise` | Maintained | `training/train_perceiver_denoise.py` | TSDAE-style token deletion + full-sequence reconstruction, optional `reconstruction+contrastive` | `ViaDecoder` | `ConceptEncoderForSentencePairClassification` with separate encoding | Yes | Canonical perceiver research path. Uses BiXT + stacked position-only decoder. |
| `weighted_mlm` | Maintained baseline | `training/train_mlm.py --model_type weighted_mlm` | Sparse MLM | Weighted concept pooling classifier | Legacy sentence-pair route from `evaluate_model_on_glue.py` | No dedicated canonical route | Kept as the simple MLM baseline for comparison. |
| `diffusion_mlm` | Maintained research track | `training/train_diffusion.py` | Masked diffusion self-reconstruction | Metadata-driven canonical route | Metadata-driven canonical route | Yes | Evaluation contract is stored in checkpoint config. |
| `prefix_diffusion` | Maintained research track | `training/train_prefix_diffusion.py` | Prefix-to-suffix diffusion generation | Metadata-driven canonical route | Metadata-driven canonical route | Yes | Primary generation-oriented text track. |
| `recursive_mlm` | Isolated experiment | `training/train_recursive_mlm.py` | Recursive sparse MLM with weight-tied encoder | Manual / experiment-specific | Manual / experiment-specific | Not wired as a standard path | Intentionally removed from generic `train_mlm.py` until recursion strategy is better defined. |

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
| `diffusion_mlm` / `prefix_diffusion` | Metadata-driven | Separate sentence encoding | Missing metadata should fail loudly instead of silently falling back. |
| `weighted_mlm` | Weighted concept pooling | Legacy pair route | Baseline path, not the semantic-first contract. |
| `recursive_mlm` | Manual | Manual | Not yet a canonical benchmark family. |

## Practical Summary

- If you want the maintained perceiver path, use `perceiver_denoise`.
- If you want the simple MLM baseline, use `weighted_mlm`.
- If you want recursion, use `training/train_recursive_mlm.py` and treat it as a separate experiment family.
- If you need historical `perceiver_mlm` comparisons, use old checkpoints/results as archived baselines, not as current launch interfaces.
