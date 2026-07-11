# Parked code

Coherent experiment families that are **not part of the active foundation** but
may be revived later. They are deliberately excluded from the maintained tree:
not importable by `nn/`/`training/`/`evaluation/`, not in any config registry,
and not collected by the test suite (`pyproject.toml` sets `testpaths = ["tests"]`).

Pre-parking snapshot tag: `pre-consolidation-20260605`.

## Contents
| Family | Why parked | Files |
|---|---|---|
| **Weighted MLM trainer** | Historical comparison baseline with 18 W&B training runs (latest 2026-02-07), but no current experiment builds on sparse MLM training. The model/evaluation classes remain live so checkpoints still load and benchmark. | `training/train_weighted_mlm.py` |
| **Diffusion** (masked-diffusion + prefix-diffusion decoders) | Explored on MiniPile / WikiText-103; concept effective rank stayed low so far. Set aside for now and likely to be revisited (e.g. with warm-start, or as an alternative decoder). | `nn/concept_encoder_diffusion.py`, `training/train_diffusion.py`, `training/train_prefix_diffusion.py`, `tests/test_diffusion.py`, `tests/test_prefix_diffusion.py`, `scripts/train_diffusion_multigpu.sh`, `scripts/train_prefix_diffusion_multigpu.sh` |

## Retired snapshots

- **Recursive MLM / TRM-style weight-tied encoder — retired 2026-07-11.** No recursive training
  run was recorded in W&B or the experiment ledger, and the old sparse-MLM fork no longer matches
  the recurrent-memory direction in E09. Its six implementation/training/test/launcher files were
  removed from `parked/`; git history and snapshot tag `pre-consolidation-20260605` preserve them.
  Future recurrence is implemented as a config-selectable component over the maintained
  concept-pretraining foundation, not by reviving this fork.

## Reviving a family
1. Open a spec in `docs/experiments_specs/` with a materially new ingredient (e.g. warm-start) — see the "what we've explored" learnings in `agenda.md` first.
2. `git mv` the needed files back into `nn/`/`training/`/`tests/` and fix imports.
3. Re-register entries removed during parking (e.g. `analysis/run_concept_analysis.py` `MODEL_CLASSES`, eval `model_type` choices).

Full procedure (move → re-wire → align with current foundation → test → update docs): see `research-implement` skill → **Unparking**.
