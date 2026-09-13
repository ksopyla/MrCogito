# E25 bridge 512 far_copy r=1 in-place raw KV — E21 vs E18 vs dense (rung 5f)

**Date:** 2026-09-13
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_rawkv`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_rawkv/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_rawkv_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `b3b0ffe` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** in-place wall [`e25_bridge512_ip_r1_far_copy_20260913.md`](e25_bridge512_ip_r1_far_copy_20260913.md) · raw PASS [`e25_bridge512_raw_far_copy_20260913.md`](e25_bridge512_raw_far_copy_20260913.md)

---

## Goal

One change vs in-place compressor r=1 (E21 **0 bits**) with the exclusive mask held
fixed: `--message_inplace_raw_kv` copies **token K/V** into existing `replace`
positions (skip compressor values). `KV_LEN` stays S. `msg_override=real`, **r=1**,
QUERY local-doc cut on. Receivers still cannot see uncompressed remainder (`~replace`).
Dense S0 in this JSON.

Splits compressor values from the exclusive leak mask.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · 0.595M (e21 0.603M) |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed span 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_rawkv 0 \
  --scale bridge --recipe far_copy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 1 --message_slots_inplace --message_inplace_raw_kv \
  --steps 800 --k1_mult 4
```

Byobu session `E25_ip_rawkv`. Log line:
`msg_boundary=10  msg_r=1  msg_remainder=False  msg_override=real  msg_inplace=True  msg_rawkv=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON:
`message_inplace_raw_kv: true`, `message_slots_inplace: true`, `message_override: real`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **100.0%** | 350 | 63.95 | 0.999 |
| e18 | **100.0%** | 300 | 63.90 | 0.998 |
| **e21 inplace raw KV** | **99.5%** | **450** | **63.29** | **0.989** |
| e18_local | 25.1% | 800 | 0 | 0 |

Locked prior rungs (do not re-run): concat r=16/64/1 **0 bits**; raw override **63.96
bits** @450; inplace compressor r=1 **0 bits**. This JSON recalibrates S0: dense
**63.95**, E18 **63.90**.

E21 still chance through step 350, **81.9% at 400**, **99.5% at 450** — the same
climb as raw override (chance through 350, 77.8% @400, 100% @450). Do not relabel
this 63-bit score as default E21 compressor slots.

S1 vs 0.75× E18: need **47.92 bits / flow 0.749**. E21 has **63.29 / 0.989**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.95 bits. |
| **S1 vs 0.75× E18** | **PASS.** 63.29 ≥ 47.92 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.1% / 0 bits. Local cut stays on. |
| **K3** | not triggered (climbing, then solved). |

Do **not** extra-step. Do **not** stack r=16. Do not 1024. Do not Glyph. Do not
another override.

## Interpretation

**Compressor values (even claimed r=1 identity) are the remaining 512 killer under
length-S geometry.** Exclusive `~replace` is **not** the killer vs `raw_cross`: the
same leak mask that scored 0 bits with compressor slots copies 63 bits when those
positions hold token K/V. Concat extra KV and compressor scatter/pool are both
dead; in-stream token K/V with the exclusive mask is live.

## Decision

Keep the spec in `ahead/`. Next ONE experiment: make **r=1 compressor a hard
identity** (bypass `u`/`delta` so arm U actually copies token K/V) and re-score
inplace r=1 **without** `--message_inplace_raw_kv` on this recipe. If that PASSES,
scatter/pool was the bug and r=16 pooling on the trusted inplace mask is the
compression question. If it stays at chance, training still wrecks identity. Not
r=16 yet. Not 1024. Not Glyph.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
