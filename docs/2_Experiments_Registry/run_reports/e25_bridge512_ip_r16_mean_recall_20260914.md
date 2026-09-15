# E25 bridge 512 recall_single r=16 in-place frozen mean — E21 vs E18 vs dense (rung 5k)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_r16_mean_recall`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_r16_mean_recall/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_r16_mean_recall_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `56a1ab2` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 512 INDEX mean [`e25_bridge512_ip_r16_mean_far_copy_20260914.md`](e25_bridge512_ip_r16_mean_far_copy_20260914.md) · tiny MATCH [`e25_tiny_recall_single_e21_steps_20260913.md`](e25_tiny_recall_single_e21_steps_20260913.md)

---

## Goal

One **task** change vs 512/1024 packed `far_copy` frozen mean (INDEX live): same inplace
r=16 `--message_identity_slots` recipe at **H=128** (512 INDEX width, not 1024 SSMax)
on packed `recall_single`. Do frozen 16-token means bind a **key**, or only INDEX?

E24 GPU 512 `recall_single` had E18 **0 bits**. Train E18 anyway. If E18 is ~0, do
**not** call S1 PASS from 0.75×0; score vs 0.75× dense. Report both.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · 0.595M (e21 0.603M) · kv=1 · **`logit_scale=none`** |
| E21 | query boundary id 10, **r=16**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed answer 24 / **48-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_r16_mean_recall 0 \
  --scale bridge --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 16 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
```

Byobu `E25_512_recall`. Log:
`seq=512  logit_scale=none  msg_r=16  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON:
`message_identity_slots: true`, `message_inplace_raw_kv: false`, `hidden: 128`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **99.0%** | 350 | 46.97 | 0.979 |
| e18 | **100.0%** | 400 | **47.94** | 0.999 |
| **e21 frozen mean r=16** | 25.5% | 800 | **0.00** | **0.000** |
| e18_local | 25.3% | 800 | 0 | 0 |

E18 is **live** in this JSON (not the E24 0-bit 512-recall control). Do not score S1
against 0.75×0. E21 stayed at chance every eval (best 25.5% @450, CE stuck at the
floor ~1.387). INDEX frozen mean on the same mask was climbing at 800.

S1 vs 0.75× E18: need **35.95 bits**. E21 has **0.00**.
Content-gate vs 0.75× dense: need **35.23 bits**. E21 has **0.00**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.0% / 46.97 bits. |
| **S1 vs 0.75× E18** | **FAIL.** 0.00 vs 35.95 bits. |
| **content vs 0.75× dense** | **FAIL.** 0.00 vs 35.23 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.3% / 0 bits. |
| **K3** | chance/floor at 800 (flow 0). Do not extra-step. |

Do **not** extra-step (floor, not climbing). Do not 4k. Do not Glyph. Do not remainder.
Do not unfreeze `u`/`delta`. Do not 1024 recall this turn.

## Interpretation

**Frozen 16-token means carry INDEX, not MATCH, at 512.** Same inplace identity
recipe that recovered 47.36 bits on `far_copy` recovers **0 bits** on `recall_single`
while dense and E18 copy ~47–48 bits. Tiny recall still had a weak E21 MATCH leak
(9.81 bits @8k vs dense 31, E18 0). At 512 the compressed exclusive channel is a
chance floor.

Do **not** relabel E18's 47.94 bits as E21. E24's 0-bit 512-recall E18 does not apply
to this JSON — this control solved the packed 48-bit prize in 400 steps.

## Decision

Keep the spec in `ahead/`. Working INDEX recipe stays frozen mean. Next ONE
experiment: seq=512 packed **`recall_single`**, inplace **r=1 hard identity** (does
exclusive uncompressed KV bind a key, or only INDEX?). Not 4k. Not Glyph. Not
learned pool. Not 1024 recall.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
