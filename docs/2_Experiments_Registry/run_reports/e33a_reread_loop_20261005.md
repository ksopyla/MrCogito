# E33a read–think–reread loop — capability checks and reasoning arms (Oct 3–5)

**Date:** 2026-10-05
**Machine:** Odra 3×3090 (reasoning arms, lookup curriculum, suite seeds 1–2, lookup-2k at 5e-5) · Polonez 4×3090 (suite seed 0)
**Run ID:** study `e30_vs_e31` jobs `e33a_*` · suites `e33a_full_30m`, `e33a_lookup2k_lr5e-5` (no W&B: capability probes)
**WandB:** — (capability probes log a final summary only)
**Raw log:** NAS `/nas/ml_data/mrcogito/results/study/e30_vs_e31.odra/`, `/nas/ml_data/mrcogito/results/capability/e33a_full_30m.{odra,polonez}/`, `/nas/ml_data/mrcogito/results/capability/e33a_lookup2k_lr5e-5.odra/`
**Best checkpoint:** — (per-job `.pt` in the NAS archive)
**Git commit:** `d72fe68` (loop), `a838aef` (answer exits for capability checks)
**Git tag:** —
**Related TODO:** —

---

## Goal
Test whether looping "global read + one local layer" four times with tied weights, before the answer layer, lets the E31
memory (`e31_li_m1`) follow parallel chains (spec gate: pchain3 46 % → ≥ 75 % first letter), and check that no past
capability is lost (no-harm rule: > 5 points below `e31_li_m1` where it passes, median of seeds).

## Configuration
| Item | Value |
|---|---|
| Family | E31 latent memory (page writer, 1 reader entry per latent, length-invariant address, windowed read 256) |
| Loop | prelude = layer 0 once; core = global read + local layer 2, tied ×4 (zero-init loop marker); answer = layer 3 + head |
| Supervision | loss = final CE + 0.3·Σ exits through the answer layer; exits trained on node r ("progress") or the answer |
| Size | 30M (H 960, 4 layers, 15 q-heads × 64, 1 KV head) |
| Arms (reasoning) | loop from scratch (2 seeds), fine-tuned from E31 lookup weights, R = 1 control, answer exits, frozen writer, prelude re-injection, wide core |
| Capability checks | suite v3 full tier, 17 non-flawed cells, seeds 0–2 (= v4 `same`); lookup 2k → 8k → 16k curriculum (C1.lookup-16k, seeds 1–2); lookup-2k at 5e-5 (3 seeds) |
| Compute | ≈ 210 GPU-h (phase 1 ≈ 70, lookup curriculum ≈ 40, suite ≈ 75, lookup-2k ≈ 20); 4 loops cost ≈ 3× a single read per step (16 s/step at 8k, 30 s/step at 16k) |

## Training Outcome
Lookup from scratch with the loop is seed-dependent like E31: seed 2 learns it (98 % at 2k), seed 1 half-learns it (70 %).
The variant with the head right after the loop (wide core) did not learn lookup (50 %). Stopped on 2026-10-04 when the
capability checks were rebuilt (v4): the in-order chain, the hard exams and their 8k stages (started from the run's own
lookup weights → `curriculum-differs`, tasks still `calibrating`; triple-match is `flawed`), and the remaining
answer-exit chain seeds.

## Concept Health
Not measured (synthetic capability probes). Per-exit first-letter traces were flat on every chain arm (exit 1 ≈ exit 4).

## Evaluation

**Reasoning (parallel chains, 4 chains, 8-letter nodes, 1k) — untested.** Every arm, the single read and dense
sat at the exam's guessing floor (answering with a random link target: 44 / 41 / 39 % for 2 / 3 / 4 hops). Loop arms
37–44 / 41–47 / 35–36 %; R = 1 control 41 / 46 / 35 %. Diagnosis:
[e33a_loop_diagnosis_20261004.md](../../4_Research_Notes/e33a_loop_diagnosis_20261004.md). The v1 parallel-chain tasks are
now `flawed` (`X.pchain*-1k-v1`).

**No harm vs `e31_li_m1` (first letter, median over seeds 0–2; v4 match label `same`):**

| task | E33a loop | `e31_li_m1` | dense |
|---|---|---|---|
| C0 copy 128 / 256 / 512 / 1k | 100 / 100 / 100 / 99 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| C1 lookup 128 / 256 / 512 / 1k | 100 / 100 / 100 / 96 | 100 / 100 / 100 / 99 | 100 / 100 / 100 / 98 |
| C1 lookup 2k (5e-5) | 99 (99 / 93 / 99) | 98 (30 / 98 / 99) | 91 |
| C2 lookalike 128 / 1k | 100 / 100 | 98 / 99 | 100 / 98 |
| C4 chain4-1k (in order) | 67 | 70 | 100 |
| C6 unique-256 | 100 | 100 | 98 |
| C7 fact-512 / story-512 / chain-512 / fact-1k | 100 / 100 / 100 / 99 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |

Seed 1 failed lookalike-1k (29) and lookup-1k (38) for E33a; `e31_li_m1` seed 1 fails the same two (45, 41): an
initialisation effect shared by both, not the loop. Lookup-2k at 1e-4 (the v3 cell step) fails for dense too and is not
comparable.

**C1.lookup-16k curriculum (2k → 8k → 16k), ladder to 128k:** E33a seed 2 **98 / 99 / 100 / 100 / 99 / 99 / 93** at
2k–128k (E31 `e31_li_m1` seed 1: 98 / 95 / 98 / 100 / 97 / 98 / 94); E33a seed 1 47–70 % at every length (its 2k start
was 70 %). E31's battery seed 0 also failed (31–40 %), but its seed 2 (tracking, `study/lookup16k.polonez.json`)
reaches 100 % at 16k and 128k. On medians that is E33a ≈ 80 % (2 seeds) vs `e31_li_m1` 98 % (3 seeds): the board counts
**5 lost cells**, all on C1.lookup-16k (training length and the longer lengths). With one bad seed in two the median is
seed-sensitive; the third E33a seed (seed 0, phase `e33a_lookup16k_s0`, Odra, started 2026-10-05) decides it.

Ledger: `results/capability/suite/e33a_full_30m.{polonez,odra}.json`, `suite/e33a_lookup2k_lr5e-5.odra.json`,
`study/e30_vs_e31.odra.json` (`e33a_*`); E31 references `suite/li_full_30m.polonez.json`,
`suite/li_full_30m_s3_odra.odra.json`, `li_seed2_30m.polonez.json` (tracking branch).

## Interpretation
On every non-flawed suite task the loop is within 3 points of `e31_li_m1` (median of three seeds), and its good lookup
seed carries to 128k like E31's. The one open loss is the long-lookup curriculum, where the loop learned lookup on one
seed of two and E31 on two of three; the third seed settles whether that is the loop or seed luck. The reasoning question is unanswered: the parallel-chain
exam could not distinguish a working loop from a broken one (every model including dense sat at its guessing floor), so
K1 is formally met but carries no information. The read never learned to address the start node, so the loop had no
first hop to refine.

## Decision
Close E33a as **inconclusive** (suite: no harm; C1.lookup-16k: lost on 2 seeds, third seed running; reasoning
untested). The reasoning question moves to the E33 reasoning-fixes track: the fixed parallel chain (C5 with overhang,
16-letter keys, candidate scoring) taught 1 → 2 → 3 hops, comparing the loop, a single read and dense.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E33a_reread_loop.md`, `agenda.md`*
