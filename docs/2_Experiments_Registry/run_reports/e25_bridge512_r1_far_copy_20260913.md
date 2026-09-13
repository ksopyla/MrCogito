# E25 bridge 512 far_copy r=1 — E21 vs E18 vs dense (rung 5c)

**Date:** 2026-09-13
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_r1`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_r1/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_r1_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `e417dd0` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** r=16 wall [`e25_bridge512_far_copy_20260913.md`](e25_bridge512_far_copy_20260913.md) · r=64 wall [`e25_bridge512_r64_far_copy_20260913.md`](e25_bridge512_r64_far_copy_20260913.md)

---

## Goal

One change vs the r=16 / r=64 seq=512 walls (E21 **0 bits**): `--message_ratio 1` so every
prefix token is its own slot (identity K/V; the uncompressed arm U). Exclusive QUERY cut
still on. Remainder **off**. Dense S0 in this JSON. Splits compression (pooling) from the
exclusive mask.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · 0.595M (e21 0.603M) |
| E21 | query boundary id 10, **r=1** (~512 slots), remainder **off** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed span 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), ~1046 MiB peak, 0.04–0.05 s/step |

```
bash scripts/e24_bapo_hunt.sh e25_512_r1 0 \
  --scale bridge --recipe far_copy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 1 --steps 800 --k1_mult 4
```

Byobu session `E25_r1`. Log line: `msg_boundary=10  msg_r=1  msg_remainder=False`.
`--message_ratio 1` is valid (`message_compress_ratio >= 1`; identity slots). Dense first
(underscore `--no-dense_first` unused).

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **100.0%** | 400 | 63.95 | 0.999 |
| e18 | **100.0%** | 300 | 63.94 | 0.999 |
| **e21 r=1** | **25.1%** | 800 | **0.00** | **0.000** |
| e18_local | 25.1% | 800 | 0 | 0 |

Locked prior rungs (do not re-run): dense 63.28 / E18 63.92 / E21 r=16 **0** / E21 r=64 **0.01**.
This JSON recalibrates S0: dense **63.95**, E18 **63.94**.

E21 CE stuck at ln(4) for all 16 evals (acc 0.24–0.27, bits 0.0008). Tiny E21 at 800 steps
was already 31.6% / CE 1.358. Identity slots did not move the 512 floor. No OOM (peak ~1.0 GiB
on the 24 GB card; batch 32 kept).

S1 vs 0.75× E18: need **47.95 bits / flow 0.749**. E21 has **0.00**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.95 bits. |
| **S1 vs 0.75× E18** | **FAIL.** 0.00 vs 47.95 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.1% / 0 bits. |
| **K3** | **triggered.** flow 0.000; chance; not climbing. Extra 8k not run. |

Do **not** relabel E18's 64 bits as E21. Do not extra-step. Do not 1024. Do not Glyph.

## Interpretation

r=1 **stays at chance.** Compression (mean-pool / slot cardinality) is **not** the 512
killer: identity token KV behind the exclusive QUERY cut still scores 0 bits, while E18's
in-place raw read copies 64 bits in 300 steps. The exclusive mask / document-start cut is
the killer even with full token KV.

Tiny INDEX still near-passed at seq=128 / 8k, so the cut is not universally dead — it is
dead on this 512 / 800-step recipe.

## Decision

Keep the spec in `ahead/`. Next ONE experiment: **`message_override=raw`** on the same 512
`far_copy` recipe (local layers stay severed; global read sees uncompressed prefix K/V
across QUERY). If that PASSES, exclusive *slot routing* (concatenated extra KV) is the
killer. If it stays at chance, QUERY-as-document-start (local RoPE / n-gram reset) is the
killer. Needs a probe `--message_override` flag. Not another ratio. Not remainder. Not
Glyph. Not 1024.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
