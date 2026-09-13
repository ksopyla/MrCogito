# E25 tiny far_copy QUERY-aligned — E21 vs E18 vs dense (rung 1c)

**Date:** 2026-09-13
**Machine:** Cursor Cloud VM (CPU)
**Run ID:** `e25_tiny_far_copy_query_align`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_tiny_far_copy_query_align/`
**Raw log:** `/opt/cursor/artifacts/e25_tiny_far_copy_query_align.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `7a3fd8b` (docs tip when this probe launched; remainder flag default off)
**Git tag:** —
**Related:** [`e25_tiny_far_copy_20260913.md`](e25_tiny_far_copy_20260913.md) · [`e25_tiny_far_copy_remainder_20260913.md`](e25_tiny_far_copy_remainder_20260913.md)

---

## Goal

After remainder pooling failed S1, one data-placement change: land QUERY on an r-aligned
complete block so the last sender block is not dropped. `--seq_len 132` puts QUERY at 96
(= 6×16). Remainder off. Recalibrate dense S0 at this length before scoring E21.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · ~0.595M (e21 0.603M) |
| E21 | `query` id 10, r=16, remainder **off**, QUERY at **96** |
| Data | DNA `far_copy`, packed span 32 / **64 bits**, seq=**132**, gap≥17, window 16 |
| Steps | advertised 800, K1=4×; dense early-stop 2950; e21 trained to 2950 |
| Device | CPU, AdamW 3e-3 |

```
uv run python verification/bapo_capability_probe.py \
  --scale tiny --recipe far_copy --arch dense e18 e21 e18_local \
  --seq_len 132 \
  --out /opt/cursor/artifacts/e25_tiny_far_copy_query_align
```

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **99.1%** | 2950 | 62.11 | 0.970 |
| e18 | **99.2%** | 1650 | 62.92 | 0.983 |
| **e21** | **40.0%** | 2950 | **8.70** | **0.136** |
| e18_local | 23.7% | 2950 | 0 | 0 |

S0 holds at the new length (dense 99.1% ≥ 75%). E21 is **worse** than seq=128 complete-block-only
(17.7 bits) and remainder (12.7 bits). Still climbing slowly (0.355 @2000 → 0.400 @2950).

## Concept Health

Not a language-model run. Effective `a`: dense/E18 ~62–63 bits, E21 **8.7**, local 0.

## Evaluation

Plots: `/opt/cursor/artifacts/e25_tiny_far_copy_query_align_*.png`.

### Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.1% / 62 bits at seq=132. |
| **S1** | **FAIL.** E21 8.7 bits vs E18 62.9 (flow 0.136 vs 0.983). Need ≥47 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** |
| **K3** | not triggered (40% vs chance 25%; flow 0.136) |

## Interpretation

r-aligned QUERY means every sender token sits in a complete homogeneous slot. E21 still
recovers only ~9 bits. The experiment-1 story ("span sits in an unpooled remainder") is
falsified. Mean-pool r=16 slots are a **lossy INDEX channel** at this width/budget, not a
missing-block accident. Stop stacking compressor geometry. Do not score recall or 512.

## Decision

Tiny packed `far_copy` S1 is a wall under complete-block, remainder, and QUERY-align. Keep
the spec in `ahead/` (K3 not hit; still climbing). Next ONE cheap test is extra steps on the
**best** E21 so far (seq=128 complete-block, 51.5% @2400), not a new architecture. If that
also plateaus below 47 bits, tiny INDEX is the measured E21 limit and the next ladder rung
is a different task only after that is written down.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
