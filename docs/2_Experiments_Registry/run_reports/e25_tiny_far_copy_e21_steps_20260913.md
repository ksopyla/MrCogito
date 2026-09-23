# E25 tiny far_copy extra steps — E21 complete-block (rung 1d)

**Date:** 2026-09-13
**Machine:** Cursor Cloud VM (CPU)
**Run ID:** `e25_tiny_far_copy_e21_steps`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_tiny_far_copy_e21_steps/`
**Raw log:** `/opt/cursor/artifacts/e25_tiny_far_copy_e21_steps.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `94be1c0` (docs tip when this probe launched)
**Git tag:** —
**Related:** [`e25_tiny_far_copy_20260913.md`](e25_tiny_far_copy_20260913.md)

---

## Goal

After remainder pooling and QUERY-align both failed S1 at the 2400-step budget, one budget
change: train seq=128 complete-block E21 to 8000 steps (`--no-dense_first --arch e21`).
Dense/E18 ceilings stay the experiment-1 replica (99.4% / 63.3 bits and 99.2% / 63.0 bits).

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar` e21 only (remainder **off**, seq=128, QUERY at 92) |
| Width | H=128 · 0.603M |
| Steps | 8000 (no early stop; best acc 86.6% @7700) |
| Device | CPU, AdamW 3e-3 |

```
uv run python verification/bapo_capability_probe.py \
  --scale tiny --recipe far_copy --arch e21 \
  --no-dense_first --steps 8000 \
  --out /opt/cursor/artifacts/e25_tiny_far_copy_e21_steps
```

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (exp 1) | **99.4%** | 2400 | 63.31 | 0.989 |
| e18 (exp 1) | **99.2%** | 1400 | 63.05 | 0.985 |
| **e21 @2400** | 51.5% | 2400 | 17.70 | 0.277 |
| **e21 @8000** | **84.6%** | 8000 | **47.01** | **0.734** |
| e21 best | **86.6%** | 7700 | (final metric is @8000) | — |
| e18_local (exp 1) | 24.7% | 2400 | 0 | 0 |

S1 vs 0.75× E18: need **47.29 bits** / flow **0.739**. Final **47.01 / 0.734** — on the line,
miss by 0.28 bits. Acc 84.6% (best 86.6%) is a live copy channel, not the 2400-step half-span
signature. Still climbing through 8k (no plateau).

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS** (exp 1 dense, same recipe). |
| **S1** | **NEAR-PASS.** 47.01 vs 47.29 bits; flow 0.734 vs 0.739. |
| **S2** plots | **PASS.** |
| **K2** | **PASS** (exp 1). |
| **K3** | not triggered |

## Interpretation

The 2400-step S1 fail was **budget**, not a dead slot read and not an unpooled remainder.
Remainder pooling and QUERY-align at 2400 made bits *worse*. Given 8k steps, mean-pool r=16
slots copy INDEX to the 0.75× E18 line. Do not treat compressor geometry as the next knob.

## Decision

Treat tiny packed `far_copy` as a slow but live E21 copy channel. Next ONE harder task:
tiny `recall_single` (E18's 0-bit content wall), dense S0 in the same JSON. Not seq=512
(needs GPU) and not Glyph.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
