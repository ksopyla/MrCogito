# E25 tiny far_copy remainder pooling — E21 vs E18 vs dense (rung 1b)

**Date:** 2026-09-13
**Machine:** Cursor Cloud VM (CPU)
**Run ID:** `e25_tiny_far_copy_remainder`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_tiny_far_copy_remainder/`
**Raw log:** `/opt/cursor/artifacts/e25_tiny_far_copy_remainder.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `8b76ba5` (remainder flag that produced these numbers)
**Git tag:** —
**Related:** complete-block-only first rung [`e25_tiny_far_copy_20260913.md`](e25_tiny_far_copy_20260913.md)

---

## Goal

One architecture change after the mixed first rung: pool the incomplete last sender block
(`message_pool_remainder=True`) and re-score **only** packed tiny `far_copy`. Same S1
(≥ 0.75 × E18 bits). Experiment 1 stays reproducible (flag default remains off).

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · ~0.595M (e21 0.603M) |
| E21 | `query` boundary id 10, r=16, **`message_pool_remainder=True`** |
| Data | DNA `far_copy`, packed span 32 / **64 bits**, seq=128, gap≥17, window 16 |
| Steps | advertised 800, K1=4×; dense/e18 early-stop at 99% |
| Device | CPU, AdamW 3e-3 |

```
uv run python verification/bapo_capability_probe.py \
  --scale tiny --recipe far_copy --arch dense e18 e21 e18_local \
  --message_pool_remainder \
  --out /opt/cursor/artifacts/e25_tiny_far_copy_remainder
```

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **99.4%** | 2400 | 63.31 | 0.989 |
| e18 | **99.2%** | 1400 | 63.05 | 0.985 |
| **e21 remainder** | **47.6%** | 2400 | **12.71** | **0.199** |
| e18_local | 24.7% | 2400 | 0 | 0 |

Dense and E18 replica-match experiment 1 (same seed). E21 is **worse** than complete-block-only
(51.5% / 17.7 bits / flow 0.277), not better. Still climbing (0.414 @2000 → 0.476 @2400).

## Concept Health

Not a language-model run. Effective `a`: dense/E18 ~63 bits, E21 remainder **12.7**, local 0.

## Evaluation

Plots: `/opt/cursor/artifacts/e25_tiny_far_copy_remainder_*.png`.

### Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.4% / 63 bits. |
| **S1** | **FAIL.** E21 12.7 bits vs E18 63.0 (flow 0.199 vs 0.985). Need ≥47 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** |
| **K3** | not triggered (47.6% vs chance 25%; flow 0.199) |

## Interpretation

Remainder pooling did **not** close the INDEX gap. The ~half-accuracy / ~18-bit signature of
experiment 1 was not "the span sitting in an unpooled remainder." Adding that slot slightly
hurt at the same 2400-step budget (12.7 vs 17.7 bits). The slot channel is still live and
lossy. Stop stacking compressor tricks. Do not score recall or 512.

## Decision

Record this as a wall on remainder-pooling at tiny `far_copy`. Next ONE change is **not**
another compressor knob: land QUERY on an r-aligned complete block (`--seq_len 132` puts
QUERY at 96 = 6×16; remainder off) and re-run dense S0 at that length before scoring E21.
Extra steps are a follow-up if alignment also fails while still climbing.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
