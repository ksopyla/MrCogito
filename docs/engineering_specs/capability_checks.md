# Capability checks — one process for testing, storing and comparing what an architecture can do

- **Status:** active (2026-10-03). Replaces the scattered rules in the `capability-suite` skill, the
  E31b spec and the E33a "Capability checks" section.
- **Skill:** `capability-checks` (`.cursor/skills/capability-checks/SKILL.md`) runs it end to end.
- **Parts:** [capability suite](capability_suite.md) (graded exams) · length battery
  (`battery_jobs` in `scripts/study_plans/e30_vs_e31.py`, the E31b protocol) · results ledger
  (`docs/2_Experiments_Registry/results/capability/`) · capability board
  (`docs/3_Evaluations_and_Baselines/capability_board.html`).

## Why

By 2026-10-03 the capability results of E30, E31 and E31b lived only in `Cache/` folders on the
servers. Nothing was logged to W&B (the suite and study runners never pass `--wandb`), nothing was
on the NAS, and the E31 reference numbers existed only as constants typed into hand-written HTML
reports. When Polonez shut down for heat, half the E31 results (the `li_full_30m` suite, seed 0 of
the study, the dense ceilings on the hard exams) became unreachable. Each new variant also copied
the E31b job list by hand (`e33a_e31b_jobs`), and each visual summary was drawn from scratch.

## Which instrument answers which question

| question | instrument | skill |
|---|---|---|
| Can this architecture learn to pick, carry and reason over information? (from scratch, synthetic) | capability suite + length battery | `capability-checks` |
| Does a change keep everything the champion can do? | the no-harm check on the capability board | `capability-checks` |
| How good is a trained text checkpoint? (concept geometry, generation, STS-B, lm-eval, long-context probes) | the tiered checkpoint pipeline | `experiment-evaluate` |
| What does a finished result mean, and where is it recorded? | master log, spec Result, run report, agenda | `experiment-track` |

## The full capability check for a new variant

Every new architecture or variant built on the current line gets all three parts, planned in its
spec from the start (the author's rule since 2026-10-03: no past capability may be lost).

1. **Capability suite, full tier, 30M, 3 seeds**, with the champion's platform flags (for E31:
   `--message_raw_window 256`, as in `li_full_30m`). A screen at 5M/10M first is optional.
   Dense controls need not be rerun when the champion's suite folder holds them on the same seeds
   and suite version: the data is deterministic per seed.
2. **Length battery** (the E31b protocol), seeds 1 and 2: lookup trained at 2k with 8k and 16k
   stages, the in-order 4-hop chain at 2k with an 8k stage, the hard exams at 1k (recall among 8
   and 16, a fact among 8 decoys, unique item, triple match, 8-hop chain) and the 8k stage for the
   two recall exams. Every trained model is laddered to 128k, 128 rows per length, scored on the
   first answer letter. Register the variant in `BATTERY_VARIANTS` and run `--phase battery_<tag>`.
3. **Comparison against the champion** on the capability board, with the no-harm rule below.

**Variant naming.** The variant id is the same everywhere: the suite arch name equals the battery's
`{tag}_{arm}` (E33a: suite arch `e33a_loop`, battery tag `e33a`, arm `loop`). The board joins the
two parts on this id.

**Champion.** `e31_li_m1` (E31 latent memory, one reader entry per latent), chosen 2026-10-01 in
E31b. When a variant takes over, change `--champion` in `analysis/capability_board.py` and this line.

## Scoring and comparison rules

- **First letter is the honest score.** On exams whose book holds several candidate answers
  (lookalike, chain, shuffled, unique, story, decoys) the later answer letters are copied once the
  first identifies the candidate, so the mean over letters overstates the capability. Ladders and
  the no-harm rule use first-letter accuracy; the suite's own pass rule (mean ≥ 75 %) is kept for
  its scale-up verdict.
- **No-harm rule.** A capability is lost when the variant scores more than 5 points below the
  champion on any battery exam and length, or suite cell, where the champion passes (≥ 75 %),
  comparing medians over seeds. "Not run" (champion passes, variant has no result) is not a pass.
- **Like with like.** Same suite version, same size, same seeds where possible. Gaps smaller than
  about two standard errors are ties.

## Where results live (the storage contract)
The same rule covers every result in the project
([`results/README.md`](../2_Experiments_Registry/results/README.md)): `Cache/` is scratch, cited
results are committed under `docs/2_Experiments_Registry/results/`, raw folders go to the NAS.

| layer | what | where | who writes it |
|---|---|---|---|
| raw | every log, rung JSON, ladder JSON, checkpoint | server `Cache/capability/<run>` or `Cache/study/<study>` | the runners |
| archive | the full raw folder | NAS `/nas/ml_data/mrcogito/results/{capability,study}/<name>.<host>/` | `scripts/pull_capability_results.sh` |
| **ledger** (source of truth for comparisons) | compact JSON per (folder, host): per job and arch accuracy ± SE, first letter, per-letter accuracy, bits, speed, every ladder length | `docs/2_Experiments_Registry/results/capability/{suite,study}/<name>.<host>.json`, committed | `scripts/pull_capability_results.sh` (collector `analysis/capability_ledger.py`) |
| scorecard | suite verdict and frontier for one run | `analysis/capability_scorecard.py --in_dir <ledger files or folders>` | on demand |
| **board** (the visual summary) | all variants: no-harm, length battery, suite grid | `docs/3_Evaluations_and_Baselines/capability_board.html`, published as an Artifact | `analysis/capability_board.py` |
| registry | what it means | master log row, spec Result, run report, agenda line | `experiment-track` |
| references | suite-cell bits of past architectures for the scorecard | `REFERENCES` in `evaluation/capability_suite.py` | `capability-checks` step "record" |

**When to pull.** After each phase finishes, at least once a day during a multi-day study (Polonez
can shut down for heat at any time), and always before a write-up or a visual summary. Pulling is
idempotent: the ledger file is rewritten from the folder, the NAS copy only grows.

**W&B is not used for these results.** The probes log only a final summary, run offline in their
hundreds, and their numbers are small enough to version in git, where every agent can read them
without network access. W&B stays the home of text-training runs and their checkpoint evaluations.

**Hand-written HTML reports** (architecture explainers, study narratives) may still be written,
but their numbers must come from the ledger, and the standard visual comparison is the board.

## Maintaining the checks

- **Suite changes** follow [capability_suite.md](capability_suite.md): bump `SUITE_VERSION`.
- **Battery changes** (an exam, a stage, a ladder length, the row count) change `battery_jobs`;
  bump `BATTERY_VERSION` in the plan module and note it in the CHANGELOG. Results across battery
  versions are not compared.
- **A new exam family** first runs on the champion (and dense, for a ceiling), then joins the
  standard battery or the suite.
- **After each result:** pull → commit the ledger → regenerate the board → `experiment-track`.
- **Gaps to close** (2026-10-03): the Polonez folders (`li_full_30m`, seed 0 of `e30_vs_e31`, the
  dense hard-exam ceilings, any E30 standard suite) are not in the ledger yet — pull them when
  Polonez is back. `REFERENCES` has no E31 suite cells yet.
