# E25 bridge 512 far_copy r=16 in-place learned pool — E21 vs E18 vs dense (rung 5i)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_r16_learned`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_r16_learned/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_r16_learned_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `e4fe6bd` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** frozen mean NEAR-PASS [`e25_bridge512_ip_r16_mean_far_copy_20260914.md`](e25_bridge512_ip_r16_mean_far_copy_20260914.md)

---

## Goal

One change vs inplace r=16 frozen mean (E21 **47.36 bits @8k**): drop
`--message_identity_slots` so `KVCompressor` `u`/`delta` can learn. Same trusted
inplace mask, r=16, no raw_kv. Question: does learning help INDEX, or wreck the
channel the way `delta` wrecked r=1?

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · 0.595M (e21 0.603M) |
| E21 | query boundary id 10, **r=16**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=False`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed span 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_r16_learned 0 \
  --scale bridge --recipe far_copy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 16 --message_slots_inplace \
  --steps 800 --k1_mult 4
```

Byobu session `E25_ip_r16_learn`. Log line:
`msg_boundary=10  msg_r=16  msg_remainder=False  msg_override=real  msg_inplace=True  msg_rawkv=False  msg_idslots=False`.

Dense first. Hunt JSON: `message_identity_slots: false`, `message_inplace_raw_kv: false`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **100.0%** | 350 | 63.94 | 0.999 |
| e18 | **100.0%** | 300 | 63.94 | 0.999 |
| **e21 learned r=16** | 25.1% | 800 | **0.00** | **0.000** |
| e18_local | 25.1% | 800 | 0 | 0 |

This JSON recalibrates S0: dense **63.94**, E18 **63.94**.

E21 stayed at chance every eval (best 26.7% @50, CE stuck at the floor ~1.386). Frozen
mean on the same mask was **42.4% / 8.18 bits and climbing at 800**. Learned `u`/`delta`
erased that signal.

S1 vs 0.75× E18: need **47.96 bits**. E21 has **0.00**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.94 bits. |
| **S1 vs 0.75× E18** | **FAIL.** 0.00 vs 47.96 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.1% / 0 bits. |
| **K3** | chance/floor at 800 (flow 0). Do not extra-step. |

Do **not** extra-step (floor, not climbing). Do not 1024 this turn. Do not Glyph.
Do not remainder. Do not flex-vs-sdpa.

## Interpretation

**Learning wrecked r=16.** Unfreezing `u`/`delta` on the trusted inplace mask returns
the 0-bit floor that frozen mean had already left. Same signature as r=1 learned
compressor vs r=1 hard identity. Keep **frozen mean / identity** as the working
recipe. Do not treat learned pool as a default compressor on this ladder.

Do **not** relabel E18's 63.94 bits as E21.

## Decision

Keep the spec in `ahead/`. Working recipe: inplace + `--message_identity_slots`.
Next ONE experiment: **seq=1024 packed `far_copy`**, inplace r=16 **frozen mean**
(`--message_identity_slots`, no learned pool, no raw_kv). One scale change. Dense
S0 required (E24 E18 is 0 bits at 1024). Not Glyph. Not remainder.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
