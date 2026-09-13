# E25 tiny chain_ordered — E21 vs E18 vs dense (rung 4)

**Date:** 2026-09-13
**Machine:** Cursor Cloud VM (CPU)
**Run ID:** `e25_tiny_chain_ordered`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_tiny_chain_ordered/`
**Raw log:** `/opt/cursor/artifacts/e25_tiny_chain_ordered.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `475c1f3` (docs tip when this probe launched)
**Git tag:** —
**Related:** [`e25_tiny_far_copy_20260913.md`](e25_tiny_far_copy_20260913.md)

---

## Goal

Last E24-calibrated tiny DNA rung: in-order DFA hops (`chain_ordered`, 26-bit prize).
E24 E18 was 97.5% vs dense 93%. Dense S0 recalibrated here. Remainder **off**.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · 0.595M (e21 0.603M) |
| E21 | query boundary id 10, r=16, remainder **off** |
| Data | on-the-fly DNA `chain_ordered`, packed answer 13 / **26-bit** prize, seq=128 |
| Steps | advertised 800, K1=4×; no 99% early-stop (dense finished 3200 at 93.1%) |
| Device | CPU, AdamW 3e-3 |

```
uv run python verification/bapo_capability_probe.py \
  --scale tiny --recipe chain_ordered --arch dense e18 e21 e18_local \
  --out /opt/cursor/artifacts/e25_tiny_chain_ordered
```

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **93.1%** | 3200 | 23.29 | 0.896 |
| e18 | **97.5%** | 3200 | 24.58 | 0.945 |
| **e21** | **49.3%** | 3200 | **8.22** | **0.316** |
| e18_local | 26.0% | 3200 | 0 | 0 |

Dense replica-matches E24 (~93%). E18 replica-matches 97.5% / ~25 bits. E21 is a live
partial hop channel (best = final 49.3%; still rising 35% @800 → 49% @3200). 0.75× E18 =
**18.43 bits / flow 0.709**; E21 has 8.22 / 0.316.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 93.1% ≥ 75%. |
| **S1 vs 0.75× E18** | **FAIL.** 8.22 vs 18.43 bits; flow 0.316 vs 0.709. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 26% / 0 bits. |
| **K3** | not triggered (flow 0.316; still climbing) |

## Interpretation

Same signature as tiny INDEX at 2400: about half accuracy, ~⅓ of E18 bits, still climbing.
Not the MATCH ~10-bit plateau and not the select type-cue near-chance wall. Extra budget is
the next ONE change (same as INDEX), not compressor stacking.

## Decision

Keep the spec in `ahead/`. Immediate next: 8000 steps on complete-block E21
(`--no-dense_first --arch e21`). Dense/E18 ceilings stay this JSON. Not Glyph. Not seq=512.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
