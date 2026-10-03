---
name: capability-checks
description: The one process for testing what an architecture can do and keeping the results — the graded capability suite (L0 learns at all → L6 BAPO-hard, 5M–50M, from scratch), the length battery (the E31b protocol — lookup, chain, recall and decoy exams laddered to 128k, first-letter scoring), the no-harm comparison against the champion (E31 latent memory), the committed results ledger + NAS archive, and the capability board (the standard visual summary). Use when the user wants to test, exam, screen, benchmark or compare a new or changed architecture or variant, asks for "full capability checks", whether it keeps past capabilities or is worth scaling, asks where past capability / suite / ladder / E30 / E31 / E31b results are, wants them synced, or asks for a visual summary / comparison of capability results. Not for evaluating trained text checkpoints on STS-B/GLUE/lm-eval (experiment-evaluate), generic training runs (experiment-run), or designing the architecture (experiment-design).
---

# Capability checks

Every architecture variant takes the same exams, its results land in the same committed ledger,
and every comparison is drawn on the same board. Process spec (why, storage contract, rules):
`docs/engineering_specs/capability_checks.md`.

| part | answers | reference |
|---|---|---|
| **Capability suite** | can it learn to copy, look up, ignore a lookalike, reach far, chain facts? (19 cells, L0–L6, 5M–50M) | `suite.md` |
| **Length battery** | does it hold up from 1k to 128k tokens, on lookup, chain, recall and decoy exams? | `battery.md` |
| **No-harm check** | does it keep everything the champion (`e31_li_m1`) can do? | board, below |
| **Ledger + NAS** | where every number lives, readable with the servers off | below |
| **Board** | the one visual summary | below |

**Boundary.** Server hardware → `remote-servers`; generic launches, env vars, Byobu →
`experiment-run`; trained text checkpoints → `experiment-evaluate`; what a result means and the
registry → `experiment-track`; how to tell the author → `research-comms`.

## The process

### 1 · Plan the checks in the spec (before code)
A new variant gets the **full capability check** (author's rule, 2026-10-03: no past capability
may be lost):
1. suite, full tier, 30M, 3 seeds, with the champion's platform flags (E31: `--message_raw_window 256`);
2. the length battery, seeds 1 and 2;
3. the comparison with the champion and the no-harm rule: lost = more than 5 points below the
   champion on any exam, length or cell where the champion passes (≥ 75 %, first letter, median over seeds).

Write all three into the spec with their expected cost (≈ 1–2 nights for the suite on both
servers, 2–3 GPU-days per battery seed). Pick one **variant id** and use it everywhere: suite arch
name = battery `{tag}_{arm}` (E33a: `e33a_loop`).

### 2 · Register
- Suite: arch in `evaluation/bapo_models.py` (`ARCHES` + `build_model`), arch-only flags in
  `ARCH_FLAGS` (`evaluation/capability_suite.py`). Details and preconditions: `suite.md` step 0.
- Battery: one entry in `BATTERY_VARIANTS` (`scripts/study_plans/e30_vs_e31.py`). Never copy the job
  list into a new function. Details: `battery.md`.
- `uv run pytest -q tests/test_capability_suite.py tests/test_bapo_probe_tiers.py` passes.

### 3 · Smoke, sync, launch
Local smoke only if it finishes in under a minute; anything longer runs on a server. Sync code by
git only (commit → push → `git pull --ff-only` on the server). Check the GPUs are free and respect
the Polonez heat rule (long queues on Odra; Polonez only for < 10 h bursts). Commands: `suite.md`
steps 2–5, `battery.md` "Running it".

### 4 · Monitor
Suite: `DONE` count and `EXIT` lines (`suite.md` step 6). Battery: `analysis/study_table.py
--prefix <tag>_ --first` on the server. Hand long watches to the `server-checker` agent.

### 5 · Pull into the ledger and archive to the NAS — after every phase, and daily on long studies
```bash
bash scripts/pull_capability_results.sh odra ~/dev/MrCogito/Cache/capability/<run>
bash scripts/pull_capability_results.sh odra ~/dev/MrCogito/Cache/study/e30_vs_e31
```
- Writes `docs/2_Experiments_Registry/results/capability/{suite,study}/<name>.<host>.json` (compact:
  per job and arch accuracy ± SE, first letter, per-letter accuracy, bits, speed, every ladder
  length; a few hundred KB, refused above 1 MB). **Commit it** — the ledger is the source of truth
  for every comparison. Nothing else from the server enters the repo: no checkpoints, logs or raw JSON.
- Copies the whole raw folder (logs, ladders, checkpoints) to
  `/nas/ml_data/mrcogito/results/{capability,study}/<name>.<host>/` on the server (additive).
- The collector is streamed over ssh, so it works on any checkout (e.g. `~/dev/MrCogito-e31`).
- Results are not logged to W&B (final-only, offline, small); do not look for them there.
- List what the ledger holds: `uv run python analysis/capability_ledger.py summary`.
- Everything else report-like on a server (evaluation reports, eval suites, old probe folders, logs;
  no checkpoints) goes to the NAS with `bash scripts/archive_reports_to_nas.sh <host>`
  → `/nas/ml_data/mrcogito/results/reports/<host>/<checkout>/` (additive, safe to re-run).

### 6 · Score
- Suite verdict + frontier for one run, straight from the ledger:
  `uv run python analysis/capability_scorecard.py --in_dir <ledger files…> --out_dir Cache/capability/<run>_scored`
  (merge halves from two hosts by passing both files). Reading the verdict: `suite.md` step 7.
- First-letter re-read of multi-candidate suite cells: `analysis/suite_first_letter.py`.

### 7 · Compare and draw — the capability board
```bash
uv run python analysis/capability_board.py                                   # default variants
uv run python analysis/capability_board.py --variants e31_li_m1 e33a_loop e31_li --champion e31_li_m1
```
Writes `docs/3_Evaluations_and_Baselines/capability_board.html`: no-harm verdict per variant
(kept / lost / not run, with the lost list), one chart per battery exam and stage (first letter vs
length, champion first), and the suite grid (honest score per cell, pass outline). The console
prints the no-harm counts.

**When the author asks for a visual summary**, regenerate the board from a fresh pull and publish
it with the Artifact tool to the existing board, https://claude.ai/artifact/FNTv1zKLc6fh9QRTrJmVrR
(pass it as `url` from a new session so the link stays the same). A one-off narrative page is fine on top, but its numbers come from
the ledger, never retyped from memory or from old HTML.

### 8 · Record
Hand to `experiment-track`: the no-harm verdict and lost list, the suite verdict and frontier per
size, the battery headline (lengths where the variant beats or falls below the champion), and the
ledger file names. Suite runs also append their median bits to `REFERENCES`
(`suite.md` step 8). Report to the author per `research-comms`: outcome first, plain words,
board link.

## Maintaining the checks
- Changing a suite cell, size, budget or scoring rule → bump `SUITE_VERSION` (`suite.md`).
- Changing a battery exam, stage, ladder or row count → bump `BATTERY_VERSION` and note it in the
  CHANGELOG; versions are not compared.
- A new exam runs first on the champion (and dense) to calibrate, then joins a part.
- A new champion → change `--champion` in `analysis/capability_board.py` and the spec line.
- Gaps open on 2026-10-03: Polonez folders (`li_full_30m`, seed 0 of `e30_vs_e31`, dense hard
  ceilings) are not in the ledger yet — pull them when Polonez is back; `REFERENCES` lacks E31.

## Pitfalls
- **Results only on a server** — pull after every phase; Polonez can shut down for heat mid-study.
- **Mean over letters** overstates multi-candidate exams; judge on first letter.
- **A copied job list** drifts from the protocol; register the variant instead.
- **Mismatched variant ids** — the board cannot join suite and battery; keep `tag_arm` = suite arch.
- **Hand-edited commands** produce a different exam; use `ARCH_FLAGS`, `BATTERY_VARIANTS` or a
  recorded `--extra`.
- **Uncalibrated ≠ failed**, **one seed is a screen**, **step-size cliffs** — see `suite.md`.
