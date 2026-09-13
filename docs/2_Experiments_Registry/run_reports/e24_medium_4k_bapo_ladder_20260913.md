# E24 medium 4k INDEX + 2k bridge + Glyph tiny S0

**Date:** 2026-09-13
**Machine:** Polonez / Odra (DNA GPU) and Cloud CPU (Glyph)
**Run ID:** `right4k_warm` / `right4k_warm_s1` / `b2k_h512_warm` / `b2k_ssmax` / `e18_1k_ssmax` / `e18_1k_h512` / `bapo_glyph_tiny_s0`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e24_bridge512/` · `/opt/cursor/artifacts/e24_glyph_tiny_s0/` · `/opt/cursor/artifacts/e24_glyph_tiny_e18/`
**Best checkpoint:** none (on-the-fly rows; models discarded after each rung)
**Git commit:** this report's commit on `cursor/e18-bapo-capability-ladder-df1a`
**Related:** [tiny](e24_tiny_bapo_ladder_20260913.md) · [bridge 512/1024](e24_bridge512_bapo_ladder_20260913.md)

The original goal is **not** done. 4k INDEX S0 is closed and E18 is scored there. 16k–128k and the rest of the medium user-core (`select`, `chain`) are not.

---

## Goal (kept intact)

Find E18 capability limits vs matched dense and `e18_local` on calibrated synthetic BAPO DNA, models <100M, tiny → medium (4k–16k) → large (16k–128k). Dense must hit ≥75% before E18 is scored.

## Configuration (one INDEX recipe)

Right-align (`row.gap == min_gap+1`), `local_window < min_gap`, `--warm_residuals`, `--amp auto`.

| scale | seq | gap / window | width | params |
|---|---|---|---|---|
| 1024 dilution hunts | 1024 | 64 / 16 | H=256 KV=8 or H=512 MHA | 2.73M or 10.8M |
| 2k | 2048 | 64 / 16 | H=512 MHA, 2 layers, warm | 10.8M |
| 4k medium | 4096 | 1024 / 256 | H=512 MHA, 4 layers, warm | 16.1M |

Packed copy prize is 64 bits. `e18_local` cannot see the evidence (K2).

---

## Locked GPU numbers

### 1024 INDEX is a width/dilution wall, not a hard one-read limit

The earlier `e18_1k_fc2` 0-bit fail (H=256, `logit_scale=none`, 4000 steps) stays as a measurement. It is **not** “one global read cannot INDEX 1024 keys.”

| hunt | dense | E18 | e18_local | calibrated? |
|---|---|---|---|---|
| H=256, SSMax **none** (`e18_1k_fc2`) | **99.9%** @1200 / 63.6 bits | **26.7% / 0 bits** @4000 | chance | dense yes; E18 S1 fail at this width |
| H=256, SSMax **log** (`e18_1k_ssmax`) | **100%** @650 / 63.95 bits | **100%** @1400 / **63.96 bits** | 26.7% / 0 bits | **yes** — S1 pass |
| H=512 MHA (`e18_1k_h512`) | **100%** @1450 / 63.90 bits | **100%** @2100 / **63.93 bits** | chance | **yes** — S1 pass |

### 2k INDEX (honest rung between 1024 and 4k)

H=256 @2048 was already K1 (23% @5000). Next recipe = the 1024-winning H=512 MHA, plus warm residuals.

| hunt | dense | E18 | e18_local | calibrated? |
|---|---|---|---|---|
| H=512 warm, SSMax none | **100%** @300 / 63.94 bits | **25.0% / 0 bits** @6000 | 27.0% / 0 bits | dense yes; **E18 S1 fail** |
| H=512 warm, SSMax log | **99.5%** @300 / 62.68 bits | **25.0% / 0 bits** @6000 | 27.0% / 0 bits | dense yes; **E18 S1 fail** |

SSMax unsticks 1024 at 2.73M and does **not** unstick 2048 at 10.8M.

### 4k INDEX S0 closed; E18 scored on the exact recipe

The prior 74% @2000 seed (zero-init residuals, missed 75% by 1 pt) is **not** a pass. An 8000-step zero-init replica stayed at chance (25.7%). Width-only H=768 ~48M was 24.6% K1.

`--warm_residuals` is the INDEX recipe that actually calibrates 4k:

| seed | dense | E18 | calibrated? |
|---|---|---|---|
| 0 (`right4k_warm`) | **99.3%** @350 / **63.09 bits** / flow 0.986 | **23.7% / 0 bits** @8000 | dense **yes** (two-seed S0); **E18 S1 fail** |
| 1 (`right4k_warm_s1`) | **99.5%** @300 / 62.63 bits / flow 0.979 | dense-only confirmation | **yes** |

Seed-0 `e18_local` is still running as of this report (JSON after that arm). Window 256 vs gap 1025 cannot see the span; local traces at step 50–100 are already at chance. Do not treat missing local JSON as a K2 miss.

Zero-init 16M @4k is a miss (74% then 26% replica). Warm 16M @4k is a two-seed pass. Do not score E18 on the 74% seed.

### 4k user-core besides INDEX

| recipe | dense | E18 | calibrated? |
|---|---|---|---|
| `recall_single` 4k, same 16M warm recipe | **23.8%** @8000 / 0 bits | skipped | **no (K1)** |

Do not score E18 on 4k recall. `select` / `chain` at 4k were not launched.

H=256 @4k and advertised **spread** 4k remain K1 (previous report).

---

## Glyph tiny (Cloud CPU, independent)

Dense-only S0, then E18 only on rungs that hit 75%. Width-32 typed vocab; Markov haystack; chance 12.5% (`n_symbols=8` answer class, floor ln(8)).

| recipe | dense | E18 | e18_local | calibrated? |
|---|---|---|---|---|
| `copy_span` | **99.0%** @1550 / 94.71 bits / flow 0.987 | **99.3%** @1850 / **94.68 bits** | 12.3% / 0 bits | **yes** |
| `fact_markov_single` | **99.2%** @2200 / 47.48 bits / flow 0.989 | **99.1%** @1750 / **47.37 bits** | 13.5% / 0 bits | **yes** |
| `reverse` | 36.1% / 26.9 bits | skipped | — | **no (K1)** |
| `every_k` | 15.8% / 0.36 bits | skipped | — | **no (K1)** |
| `filter_mod` | 68.3% / 14.7 bits | skipped | — | **no (K1)** — missed 75% |

Glyph `copy_span` is spread (row gaps 18/33/60 at seq=128), not the GPU right-align INDEX recipe. Tiny spread still trains.

Glyph `fact_markov_single` is **not** a DNA S2 kill. DNA `recall_single` at seq=128 and 512 is still E18 0 bits. Glyph keyed fact in a Markov haystack is a different instrument (typed 8-way answer class, closed markers). It shows the DNA keymark/iid packing is doing work that “one global read cannot bind a key” does not by itself explain. K3 remains a DNA-tiny clause (`recall` + `select` + shuffled `chain`).

---

## What this changes

1. **Medium 4k INDEX exists as an S0.** Dense 16M + warm residuals + right-align hits 99% in ~350 steps, twice. The 74% seed was zero-init and did not replicate.
2. **E18 does not ride that 4k S0.** On the exact 16M recipe, E18 recovers **0 bits** in 8000 steps after dense finished in 350. Same pattern at 2048 H=512 (with and without SSMax).
3. **1024 E18 0 bits was dilution at 2.7M.** SSMax `log` or H=512 restores 64 bits. That unstick **stops between 1024 and 2048**.
4. **4k recall is not a user-core number yet.** Dense 24% @8000. K1.
5. **Glyph tiny `copy_span` and `fact_markov_single` calibrate**, and E18 matches dense there with a clean local arm. `reverse` / `every_k` / `filter_mod` stay K1.

## Gates vs this evidence

| gate | result |
|---|---|
| **S0** | 4k `far_copy` **pass** (two seeds). 2k `far_copy` **pass**. 4k `recall_single` K1. Glyph `copy_span` + `fact_markov_single` pass; Glyph reverse/every_k/filter_mod K1. |
| **S1** positional | Pass at 512, at 1024 with SSMax or H=512, and on Glyph `copy_span`. **Fail at 2048 and at 4k** on the calibrated dense recipes. |
| **S2** content | DNA 512 `recall_single` still 0 bits (previous report). Glyph `fact_markov_single` is **not** that wall. 4k recall uncalibrated. |
| **K1** | Honoured on 4k recall, Glyph reverse/every_k/filter_mod, spread 4k, H=256 @2048. |
| **K2** | `e18_local` chance on every scored retrieval rung that has finished. 4k copy local JSON still writing. |

## In flight (do not treat as results)

- 4k E18 `e18_local` on `right4k_warm` (K2 JSON)
- 4k E18 with SSMax `log` (`right4k_ssmax`) — dense already 99.2% @1350; E18 still at chance ~3k steps

Do not kill leftover E22 byobu sessions. Leave these E24 jobs.

## Immediate next action (original 4k–16k / 16k–128k goal)

1. Finish 4k SSMax E18 (in flight) — same recipe + the 1024 anti-dilution scale. If it is also 0 bits, 4k INDEX is an E18 scale wall at 16M, not a missing SSMax.
2. **16k INDEX S0** with the 16M warm right-align recipe (`--scale medium_16k`), dense ≥75% and a second seed, then E18. Batch/OOM unknown; do not launch a second training family.
3. 4k `recall` / `select` / `chain` stay hunts until a dense recipe hits 75%. Do not score E18 there.
4. Glyph: do not score E18 on K1 permutation/filter rungs. Width/pack hunts are later, not this program.

---

## Plots

- INDEX ladder: `/opt/cursor/artifacts/e24_gpu_index_scale_ladder.png`
- 1024 dilution vs hard-limit: `/opt/cursor/artifacts/e24_gpu_1024_dilution_vs_hard_limit.png`
- 1024 SSMax / H=512 official probe plots: `/opt/cursor/artifacts/e24_bridge512/plots_1024_ssmax/` · `plots_1024_h512/`
- Glyph dense S0 heatmap: `/opt/cursor/artifacts/e24_gpu_glyph_tiny_s0_heatmap.png`
- Glyph E18 on calibrated rungs: `/opt/cursor/artifacts/e24_gpu_glyph_tiny_dense_e18.png` · `/opt/cursor/artifacts/e24_gpu_glyph_e18_learning_curves.png`
