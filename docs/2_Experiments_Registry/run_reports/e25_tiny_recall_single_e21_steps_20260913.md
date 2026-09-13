# E25 tiny recall_single extra steps — E21 complete-block (rung 2b)

**Date:** 2026-09-13
**Machine:** Cursor Cloud VM (CPU)
**Run ID:** `e25_tiny_recall_single_e21_steps`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_tiny_recall_single_e21_steps/`
**Raw log:** `/opt/cursor/artifacts/e25_tiny_recall_single_e21_steps.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `697122e` (docs tip when this probe launched)
**Git tag:** —
**Related:** [`e25_tiny_recall_single_20260913.md`](e25_tiny_recall_single_20260913.md)

---

## Goal

Tiny `recall_single` at 2850 steps recovered **3.19 bits** (not chance, not dense). Same
budget trick that turned INDEX from 17.7 bits into a near-pass: train complete-block E21
to 8000 steps. Dense/E18 ceilings stay the 2850 JSON.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar` e21 only (remainder **off**, seq=128) |
| Width | H=128 · 0.603M |
| Steps | 8000 (no early stop; best acc 48.5% @7700) |
| Device | CPU, AdamW 3e-3 |

```
uv run python verification/bapo_capability_probe.py \
  --scale tiny --recipe recall_single --arch e21 \
  --no-dense_first --steps 8000 \
  --out /opt/cursor/artifacts/e25_tiny_recall_single_e21_steps
```

## Training Outcome

| | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (rung 2) | **99.2%** | 2850 | 31.37 | 0.980 |
| e18 (rung 2) | 25.3% | 2850 | **0** | 0 |
| **e21 @2850** | 37.9% | 2850 | 3.19 | 0.100 |
| **e21 @8000** | **45.1%** | 8000 | **9.81** | **0.307** |
| e21 best | **48.5%** | 7700 | (final metric is @8000) | — |
| e18_local (rung 2) | 23.5% | 2850 | 0 | 0 |

S1 vs 0.75× dense: need **23.53 bits / flow 0.735**. Final **9.81 / 0.307**. Extra budget
tripled bits (3.19 → 9.81) but acc plateaus in the mid-40s (CE ~0.95 vs dense 0.03). Still
beats E18's 0-bit MATCH wall.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS** (rung 2 dense). |
| **S1 vs E18** | E21 **9.81 bits** vs E18 **0**. |
| **S1 vs 0.75× dense** | **FAIL.** 9.81 vs 23.53 bits; flow 0.307 vs 0.735. |
| **S2** plots | **PASS.** |
| **K2** | **PASS** (rung 2). |
| **K3** | not triggered |

## Interpretation

INDEX at 8k was a slow copy channel that reached the 0.75× E18 line. MATCH at the same
budget is a **content wall**: slots leak ~10 of 32 prize bits and do not keep climbing
through 75%. Do not spend a 16k budget as a substitute for a mechanism change. Do not
stack remainder / QUERY-align (those hurt INDEX).

## Decision

Record tiny MATCH as: live vs E18 (0 bits), dead vs dense (31 bits). Next ONE experiment:
tiny `select_1decoy` (E24's calibrated marker-cue rung; E18 was 87%). Dense S0 in the same
JSON. Not Glyph. Seq=512 INDEX needs GPU (none here).

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
