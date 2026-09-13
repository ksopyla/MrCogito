# E25 bridge 512 far_copy r=1 in-place hard identity — E21 vs E18 vs dense (rung 5g)

**Date:** 2026-09-13
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_id`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_id/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_id_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `ecd58b2` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** inplace compressor wall [`e25_bridge512_ip_r1_far_copy_20260913.md`](e25_bridge512_ip_r1_far_copy_20260913.md) · inplace raw KV PASS [`e25_bridge512_ip_rawkv_far_copy_20260913.md`](e25_bridge512_ip_rawkv_far_copy_20260913.md)

---

## Goal

One change vs in-place learned compressor r=1 (E21 **0 bits**): `--message_identity_slots`
bypasses `KVCompressor.u`/`delta` so r=1 is a **hard copy** of `k_norm(k_raw)`, `v`.
Still goes through the **inplace scatter** path. **No** `--message_inplace_raw_kv`.
`msg_override=real`, QUERY local-doc cut on. Dense S0 in this JSON.

Splits learned pool/delta from scatter.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · 0.595M (e21 0.603M) |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed span 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_id 0 \
  --scale bridge --recipe far_copy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
```

Byobu session `E25_ip_id`. Log line:
`msg_boundary=10  msg_r=1  msg_remainder=False  msg_override=real  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON:
`message_identity_slots: true`, `message_inplace_raw_kv: false`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **100.0%** | 350 | 63.92 | 0.999 |
| e18 | **100.0%** | 300 | 63.94 | 0.999 |
| **e21 inplace identity** | **99.8%** | **750** | **62.64** | **0.979** |
| e18_local | 25.1% | 800 | 0 | 0 |

Locked prior rungs (do not re-run): concat r=16/64/1 **0 bits**; raw override **63.96
bits** @450; inplace compressor r=1 **0 bits**; inplace raw KV **63.29 bits** @450.
This JSON recalibrates S0: dense **63.92**, E18 **63.94**.

E21 still chance through step 650, **56.6% at 700**, **99.8% at 750**. Slower than
skipping the compressor (raw KV solved at 450) but it copies. At r=1, softmax over one
token ignores `u`; learned `delta` is what can leave identity.

S1 vs 0.75× E18: need **47.96 bits / flow 0.749**. E21 has **62.64 / 0.979**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.92 bits. |
| **S1 vs 0.75× E18** | **PASS.** 62.64 ≥ 47.96 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.1% / 0 bits. Local cut stays on. |
| **K3** | not triggered (climbing, then solved). |

Do **not** extra-step. Do **not** run r=16 in this turn. Do not 1024. Do not Glyph.

## Interpretation

**Scatter is not the 512 killer.** A hard token-KV copy through the compressor module
and inplace scatter recovers 62.6 bits. Learned `u`/`delta` (at r=1: `delta`, since
one-token softmax ignores `u`) wrecked the claimed identity. Training does **not**
wreck a frozen copy. The remaining question is **compression**: do 16-token means on
this trusted inplace mask carry the 64-bit span?

## Decision

Keep the spec in `ahead/`. Next ONE experiment: **inplace r=16 frozen mean-pool**
(`--message_slots_inplace --message_identity_slots --message_ratio 16`, no raw_kv,
no learned `u`/`delta`) on this recipe. If that PASSES, frozen 16-token means are an
INDEX channel. If chance, need a better pooler than mean (or span/block alignment).
Not learned `u`/`delta` yet. Not 1024. Not Glyph.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
