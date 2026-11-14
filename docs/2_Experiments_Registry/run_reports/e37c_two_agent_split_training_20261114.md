# E37c two-agent split training — the cold start is partly learned, the pass marks are missed — `perceiver_ar_perceiver_H896L3g1s7N128_20261114_064152_e37a_m100_split`

**Date:** 2026-11-14
**Machine:** AWS g6e.xlarge spot (1 × L40S), us-east-1b, node `e37c-split` (i-0b28a1250f3a5f112), 06:30 → 11:45 UTC, 5.24 h, $9.06
**Run ID:** `perceiver_ar_perceiver_H896L3g1s7N128_20261114_064152_e37a_m100_split`
**WandB:** offline run, kept in `s3://mrcogito-train-<account>/runs/e37c/e37c_m100_e37a_seed1/wandb/` (not synced; no compute audit, compute below is from the job's own timestamps)
**Raw log:** `s3://mrcogito-train-<account>/runs/e37c/e37c_m100_e37a_seed1/split.log`, `events.log`
**Best checkpoint:** `s3://mrcogito-train-<account>/runs/e37c/e37c_m100_e37a_seed1/split/perceiver_ar_perceiver_H896L3g1s7N128_20261114_064152_e37a_m100_split/final`
**Git commit:** `6379932` (branch `claude/agent-latent-communication-0479aa`)
**Git tag:** —
**Related TODO:** spec [E37c](../../experiments_specs/done_failed/E37c_two_agent_split_training.md)

---

## Goal
Test whether agent B's cold start is learnable: fine-tune the concept-read 100M model on long-document rows cut at
random points (after a cut the local layers see only their own side; the reading layer still reads the whole
document), then examine it as two agents with the eval-only probe of 2026-11-13.

## Configuration
| Item | Value |
|---|---|
| Family | `perceiver_ar`, E37a concept read, 162.4M parameters (100M without word tables) |
| Init | the 16k stage's final weights (`runs/e37/e33n_m100_e37a_tok3.0e9_seed42/decay/…/final`), fresh optimizer |
| Dataset | LongDocs-lite, rows of 16,384, one document per row, manifest seed 3042 |
| Objective | next token; every row cut, gaps log-uniform in [32, 4,096] tokens (`par_cut_gap_min/max`) |
| Steps | 381 (0.2B tokens), 32 rows a step (micro 1 × accumulation 32), 6.4e-5 → 6.4e-6, 1−sqrt, seed 1 |
| Throughput | 14.3k tokens/s on one L40S |
| Compute | training 3.93 GPU-h (06:41:45 → 10:37:57 UTC), exams and pre-flight 1.2 GPU-h; no audited `compute/*` (W&B offline) |

## Training Outcome
Training loss on cut rows stayed near 2.27 (first ten logs 2.270, last ten 2.270; noise of a 32-row step ± 0.05).
The stop rule K2 (mean of steps 41–60 above 2.39) read 2.234. The trainer's held-out loss without cuts went
2.2655 → 2.2574 → 2.2530 at steps 100 / 200 / 300 (the 16k stage ended at 2.2526 on the same rows).

## Concept Health
Not measured (no change to the concept read).

## Evaluation
All files: `docs/2_Experiments_Registry/results/evaluations/E37c_split_ft/` (`untuned_20261113/` holds the probe
files of the untuned model).

**Cut loss** (40 held-out rows of 16,384, a cut every 1,024 tokens; extra loss against the same positions without
cuts, ± standard error):

| positions after a cut | untuned | tuned |
|---|---|---|
| 0–7 | +1.334 ± 0.031 | +0.908 ± 0.025 |
| 8–31 | +0.355 ± 0.009 | +0.170 ± 0.007 |
| 32–127 | +0.081 ± 0.002 | +0.021 ± 0.001 |
| 128+ | +0.037 ± 0.001 | 0.000 ± 0.000 |
| whole row, cut as in training | 2.2575 (plain 2.1873) | 2.1927 (plain 2.1761) |

**Two-agent probe**, number needle in the filler haystack (`niah_single_1`), % found, untuned (13 Nov) → tuned.
The two runs drew different documents (see Notes), so the comparison is between two samples.

| length | docs | one agent | B, memory only | pass mark | + A's last 16 tokens | no memory / other document |
|---|---|---|---|---|---|---|
| 4k | 30 | 100 → 83 | 63 → 63 | ≥ 93 | 100 → 83 | 0 → 0 |
| 16k | 30 | 97 → 80 | 43 → 70 | ≥ 86 | 93 → 83 | 0 → 0 |
| 32k | 20 | 90 → 85 | 35 → 60 | ≥ 79 | 60 → 85 | 0 → 0 |
| 64k | 20 | 100 → 90 | 40 → 60 | — | 58 → 80 (last 32 / 64: 95 → 85, 100 → 85) | 0 → 0 |

- Pooled over 16k–64k, memory only: 28 of 70 → 45 of 70 (Fisher p ≈ 0.007). At 4k: 19 of 30 both times.
- Pooled over all four lengths, one agent: 97 of 100 → 84 of 100 (p ≈ 0.003).
- Every one of the tuned model's 16 one-agent misses starts its answer with a colon and a blank line and then
  quotes the filler or the question; the untuned model never started that way in 100 documents. The tuned B's
  memory-only misses are the same habit (32 of 36 start with the blank line), followed by document-like text
  (LaTeX, headings).
- Memory on 8 bits / 4 bits (15 documents a cell, B starting on its preamble): 60 / 47 % at 4k, 47 / 20 % at 16k,
  60 / 20 % at 32k.

Essay haystack (`niah_single_2`), untuned → tuned: at 4k one agent 63 → 90, memory only 10 → 53, with A's last 16
tokens 43 → 90; at 16k and 32k every condition stays at 0–17 %.

**Long-document loss without cuts** (8 rows per source, the same rows): 2.4120 → 2.4009 at 4k, 2.1873 → 2.1761 at
16k, 2.1912 → 2.1967 at 64k.

## Interpretation
Against the spec: S1 (memory only ≥ 93 / 86 / 79 %) is missed at all three lengths; K1 (under 61 % at 16k) is not
hit; S2 (controls ≤ 5 %) holds; S3 (no harm) holds for the loss on plain long text but fails for the needle exam
(−17 points as one agent at 4k and 16k, limit 7), which also trips K3 (more than twice the margin). What the
fine-tune trained did move: the cost of a cut on next-token prediction fell by about three quarters at 32–127
tokens and vanished past 128, and B with memory only went from 40 % to 64 % of needles at 16k–64k. The exam is
limited by a new answer habit of the tuned model (a blank line after the colon), which lowers both the one-agent
and the memory-only score. This run cannot say whether the habit comes from the cuts or from 0.2B more tokens of
long documents: the same fine-tune without cuts was not run.

## Decision
The tuned checkpoint is not a demo model (K3). For a demo today the untuned model with A's last 32–64 tokens' local
caches in the hand-over stays the working route (95–100 % at 64k, no training). The one follow-up the evidence
justifies is the control the spec left out: the same fine-tune with cuts off, and all three checkpoints examined on
the same seeded documents (`scripts/a2a_split_probe.py --seed`).

## Notes
- The spec said "same documents as 2026-11-13". They were not: lm-eval's needle generator draws from the global
  random state and the probe did not seed it. Found on 2026-11-14 when the references of the two runs did not
  match; `--seed` added the same day. The untuned and tuned columns are therefore two independent samples.
- The stop rule K2's reference was corrected at 06:41 UTC, before the first training loss (spec).
- One seed.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E37c_two_agent_split_training.md`, `agenda.md`*
