# E25 tiny chain_ordered extra steps — E21 complete-block (rung 4b)

**Date:** 2026-09-13
**Machine:** Cursor Cloud VM (CPU)
**Run ID:** `e25_tiny_chain_ordered_e21_steps`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_tiny_chain_ordered_e21_steps/`
**Raw log:** `/opt/cursor/artifacts/e25_tiny_chain_ordered_e21_steps.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `d826637` (docs tip when this probe launched)
**Git tag:** —
**Related:** [`e25_tiny_chain_ordered_20260913.md`](e25_tiny_chain_ordered_20260913.md)

---

## Goal

Tiny `chain_ordered` at 3200 steps recovered **8.22 bits** at 49.3% acc, still climbing.
Same extra-budget test that turned INDEX into a near-pass: 8000 steps, complete-block E21.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar` e21 only (remainder **off**) |
| Width | H=128 · 0.603M |
| Steps | 8000 (best acc 51.7% @5450) |
| Device | CPU, AdamW 3e-3 |

```
uv run python verification/bapo_capability_probe.py \
  --scale tiny --recipe chain_ordered --arch e21 \
  --no-dense_first --steps 8000 \
  --out /opt/cursor/artifacts/e25_tiny_chain_ordered_e21_steps
```

## Training Outcome

| | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (rung 4) | **93.1%** | 3200 | 23.29 | 0.896 |
| e18 (rung 4) | **97.5%** | 3200 | 24.58 | 0.945 |
| **e21 @3200** | 49.3% | 3200 | 8.22 | 0.316 |
| **e21 @8000** | **49.4%** | 8000 | **10.02** | **0.386** |
| e21 best | **51.7%** | 5450 | (final metric is @8000) | — |
| e18_local (rung 4) | 26.0% | 3200 | 0 | 0 |

S1 vs 0.75× E18: need **18.43 bits / flow 0.709**. Final **10.02 / 0.386**. Accuracy did
**not** rise with budget (49.3% → 49.4%; peak 51.7% mid-run). CE fell 0.95 → 0.85, which
adds a couple of bits without a copy-channel takeoff.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS** (rung 4 dense). |
| **S1 vs 0.75× E18** | **FAIL.** 10.02 vs 18.43 bits. |
| **S2** plots | **PASS.** |
| **K2** | **PASS** (rung 4). |
| **K3** | not triggered |

## Interpretation

Unlike INDEX (51% @2400 → 85% @8k), ordered hops **plateau at half accuracy**. Extra
steps are not a substitute. Tiny DNA limits for complete-block r=16 E21:

- INDEX `far_copy`: slow near-pass at 8k
- MATCH `recall_single`: ~10 bits / ~45% at 8k
- type-cue `select_1decoy`: ~1.5 bits / ~33% at 1900
- hops `chain_ordered`: ~10 bits / ~49% at 8k

## Decision

Stop extra-stepping chain. Next ONE experiment needs GPU: seq=512 packed `far_copy` with
dense S0 first (E18 already 100% at 512). Not Glyph. Not 16k. Not remainder pooling.
Keep spec in `ahead/` until a 512 rung is scored.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
