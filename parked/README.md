# Parked code

Coherent experiment families that are **not part of the active foundation** but
may be revived later. They are deliberately excluded from the maintained tree:
not importable by `nn/`/`training/`/`evaluation/`, not in any config registry,
and not collected by the test suite (`pyproject.toml` sets `testpaths = ["tests"]`).

Pre-parking snapshot tag: `pre-consolidation-20260605`.

## Contents
| Family | Why parked | Files |
|---|---|---|
| **Recursive** (concept refinement / TRM-style weight-tied encoder) | Not the current focus. Recursive / latent reasoning stays part of the long-term Vision and will likely be explored later, possibly with a different approach. | `nn/concept_encoder_recursive.py`, `nn/concept_encoder_recursive_mlm.py`, `training/train_recursive_mlm.py`, `tests/test_recursive_*.py`, `scripts/train_recursive_mlm.sh` |
| **Diffusion** (masked-diffusion + prefix-diffusion decoders) | Explored on MiniPile / WikiText-103; concept effective rank stayed low so far. Set aside for now and likely to be revisited (e.g. with warm-start, or as an alternative decoder). | `nn/concept_encoder_diffusion.py`, `training/train_diffusion.py`, `training/train_prefix_diffusion.py`, `tests/test_diffusion.py`, `tests/test_prefix_diffusion.py`, `scripts/train_diffusion_multigpu.sh`, `scripts/train_prefix_diffusion_multigpu.sh` |

## Reviving a family
1. Open a spec in `docs/experiments_specs/` with a materially new ingredient (e.g. warm-start) — see the "what we've explored" learnings in `agenda.md` first.
2. `git mv` the needed files back into `nn/`/`training/`/`tests/` and fix imports.
3. Re-register entries removed during parking (e.g. `analysis/run_concept_analysis.py` `MODEL_CLASSES`, eval `model_type` choices).

Full procedure (move → re-wire → align with current foundation → test → update docs): see `research-implement` skill → **Unparking**.
