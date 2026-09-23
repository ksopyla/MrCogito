# E25 bridge 512 far_copy — E21 vs E18 vs dense (rung 5)

**Date:** 2026-09-13
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_fc` (`Cache/bapo_s0/e25_512_fc`)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_probe.log` (Odra `Cache/logs/e24_s0_e25_512_fc_20260913_235055.log`)
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `6dcb0c0` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** E24 512 INDEX [`e24_bridge512_bapo_ladder_20260913.md`](e24_bridge512_bapo_ladder_20260913.md) · tiny E21 INDEX [`e25_tiny_far_copy_e21_steps_20260913.md`](e25_tiny_far_copy_e21_steps_20260913.md)

---

## Goal

Next ONE harder DNA task after tiny limits: packed `far_copy` at seq=512 on the **same
right-align recipe E24 used** (E18 100% / 64 bits). Dense S0 in this JSON. Remainder **off**.
Do not relabel E18's 64 bits as an E21 score.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · 0.595M (e21 0.603M) |
| E21 | query boundary id 10, r=16, remainder **off**, ~32 slots |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed span 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` (`zero_resid=False`) |
| Device | CUDA bf16 (`--amp auto`), 0.04–0.05 s/step |
| Steps | advertised 800, K1=4×; dense early-stop 99% @300 sets later-arch floor via `max(800, 300)` |

```
bash scripts/e24_bapo_hunt.sh e25_512_fc 0 \
  --scale bridge --recipe far_copy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --steps 800 --k1_mult 4
```

Byobu session `E25` on Odra. Main E22 checkout left on `cursor/perceiver-revisit-synthesis-00f2`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **99.5%** | 300 | 61.22 | 0.957 |
| e18 | **99.4%** | 450 | 63.22 | 0.988 |
| **e21** | **25.1%** | 800 | **0.00** | **0.000** |
| e18_local | 25.1% | 800 | 0 | 0 |

Dense S0 replica of E24 (99.7% @1950 / 63.5 bits) — this seed ignited faster (300 steps).
E18 replica of E24's 100% / 63.9 bits (99.4% / 63.2; early-stop at 99%). E21 stayed at
chance for all 16 evals (acc 0.24–0.27, CE stuck at ln(4)). Tiny E21 at the same 800-step
mark was already 31.6% / CE 1.358.

S1 vs 0.75× E18: need **47.42 bits / flow 0.741**. E21 has **0**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.5% ≥ 75%. |
| **S1 vs 0.75× E18** | **FAIL.** 0 vs 47.4 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.1% / 0 bits. |
| **K3** | **triggered in spirit.** flow 0.000 < 0.05; not climbing. Probe budget for E21 was 800 (`max(800, dense_steps)`), short of 4× dense finish (1200). Tiny INDEX was already moving by step 500; this curve is a floor. Extra 8k **not** run (only allowed if climbing). |

Do **not** relabel E18's 63 bits as E21.

## Interpretation

r=16 exclusive slots copy INDEX slowly at seq=128 and **do not copy it at seq=512** on the
recipe whose raw one-read (E18) is a 64-bit machine. Slot count is 32 vs tiny's ~8; the
needle is the same 32-token packed span. Remainder pooling stays off (tiny INDEX got worse
with it). Do not score 1024.

## Decision

Keep the spec in `ahead/`. Stop after this 512 recording. Next ONE experiment (not run
here): **match tiny's slot cardinality** — `--message_ratio 64` at seq=512 so the prefix
is ~8 slots, the width that near-passed tiny INDEX. Not remainder. Not Glyph. Not 1024.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
