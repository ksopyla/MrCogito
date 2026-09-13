# E25 tiny select_1decoy — E21 vs E18 vs dense (rung 3)

**Date:** 2026-09-13
**Machine:** Cursor Cloud VM (CPU)
**Run ID:** `e25_tiny_select_1decoy`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_tiny_select_1decoy/`
**Raw log:** `/opt/cursor/artifacts/e25_tiny_select_1decoy.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `f977fab` (docs tip when this probe launched)
**Git tag:** —
**Related:** [`e25_tiny_recall_single_20260913.md`](e25_tiny_recall_single_20260913.md)

---

## Goal

After tiny INDEX near-pass and MATCH wall (~10 bits @8k), the next ONE calibrated DNA rung:
`select_1decoy` (keymark vs one decoy — a type cue, not MATCH2). E24 E18 was 87% / 26 bits.
Dense S0 recalibrated in this JSON. Remainder **off**.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · 0.595M (e21 0.603M) |
| E21 | query boundary id 10, r=16, remainder **off** |
| Data | on-the-fly DNA `select_1decoy`, packed answer 16 / **32-bit** prize, seq=128 |
| Steps | advertised 800, K1=4×; dense early-stop 99% @1900 sets later-arch budget |
| Device | CPU, AdamW 3e-3 |

```
uv run python verification/bapo_capability_probe.py \
  --scale tiny --recipe select_1decoy --arch dense e18 e21 e18_local \
  --out /opt/cursor/artifacts/e25_tiny_select_1decoy
```

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **99.2%** | 1900 | 31.31 | 0.978 |
| e18 | **82.3%** | 1900 | 23.17 | 0.724 |
| **e21** | **33.0%** | 1900 | **1.47** | **0.046** |
| e18_local | 25.3% | 1900 | 0 | 0 |

E18 replica is a bit under E24's 87% / 26 bits (82% / 23 bits at the dense-stop budget) but
still a live type-cue channel. E21 is barely above chance (best 34.6% @1750; CE 1.32 vs
floor 1.386). 0.75× E18 = **17.38 bits / flow 0.543**; E21 has 1.47 / 0.046.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.2% / 31 bits. |
| **S1 vs 0.75× E18** | **FAIL.** 1.47 vs 17.38 bits; flow 0.046 vs 0.543. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.3% / 0 bits. |
| **K3** | **borderline.** flow 0.046 < 0.05; acc 33% vs chance 25%. |

## Interpretation

Mean-pool r=16 slots destroy the `keymark` vs `decoy` type cue that E18's raw keys use.
This is not the INDEX slow-copy signature and not the MATCH ~10-bit leak. Do not extra-step
this rung as a substitute: dense and E18 both solved it inside 1900 steps.

## Decision

Record a **type-cue wall**. Keep the spec in `ahead/` (INDEX still near-pass; MATCH still
weakly live). Next ONE DNA rung: tiny `chain_ordered` (E24 E18 97.5% in-order hops). Dense
S0 in the same JSON. Not Glyph. Not seq=512.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
