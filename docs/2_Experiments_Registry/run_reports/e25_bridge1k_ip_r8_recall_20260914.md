# E25 bridge_1k 1024 recall_single r=8 remainder-off frozen mean — E21 vs E18 vs dense (rung 5ab)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1k_ip_r8_recall` (800) · `e25_1k_ip_r8_recall_s8k` (8000)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_ip_r8_recall/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_ip_r8_recall_probe.log` (8k; also `_800.log`)
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `67211c5` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 512 MATCH r=8 rem-off PASS [`e25_bridge512_ip_r8_mean_recall_20260914.md`](e25_bridge512_ip_r8_mean_recall_20260914.md) · 1024 INDEX r=16 H=256 log PASS [`e25_bridge1k_ip_r16_mean_far_copy_20260914.md`](e25_bridge1k_ip_r16_mean_far_copy_20260914.md)

---

## Goal

SELECT remainder sweep is done. Default r=8 rem-off covers MATCH (47.16) and
SELECT (45.53) at 512. INDEX already copies at 1024 with frozen mean r=16
(53.82 bits, H=256 SSMax log). One change: **scale MATCH** with the 512
MATCH+SELECT default recipe — seq=1024 packed `recall_single`, inplace frozen
mean **r=8 remainder-off**, start **H=128 / `logit_scale=none`**.

Score vs 0.75× E18; if E18 is ~0, score vs 0.75× dense. Climbing at 800 →
extra-step 8k. Dense K1 at H=128 would bump once to H=256 SSMax log; dense
passed, so no width bump this turn. Do **not** pass `--message_pool_remainder`.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · 0.595M (e21 0.603M) · kv=1 · **`logit_scale=none`** (512 MATCH width, not 1024 INDEX H=256 log) |
| E21 | query boundary id 10, **r=8**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge_1k` seq=1024, gap=64, window=16, packed answer 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.5–1.9GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1k_ip_r8_recall 0 \
  --scale bridge_1k --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 8 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# climbing: same flags --steps 8000 --k1_mult 1
# no --message_pool_remainder
```

Byobu `E25_1k_r8_recall` / `E25_1k_r8_rec8k`. Log:
`seq=1024  logit_scale=none  msg_r=8  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `message_ratio: 8`,
`message_pool_remainder: false`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`, `hidden: 128`, `global_logit_scale: none`.
Recalibrated dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (8k JSON) | **100.0%** | 6850 | **63.73** | 0.996 |
| e18 (8k JSON) | **99.5%** | 500 | **62.64** | 0.979 |
| **e21 @800** | 43.1% | 800 | 13.33 | 0.208 |
| **e21 @8000** | **26.8%** | **8000** | **0.01** | **0.000** |
| e18_local (8k JSON) | 24.2% | 8000 | 0 | 0 |

800 JSON: dense **63.80** @500; E18 **0 bits** @800 (chance); E21 **13.33**.
8k JSON: dense **63.73** @6850; E18 **62.64** @500 (live — not ~0). Extra-step
is a fresh run (not a resume). 800 climb did **not** replicate: 8k E21 stayed
at chance every eval (CE at ln(4)).

S1 vs 0.75× E18 in the 8k JSON: need **46.98 bits**. E21 has **0.01**.
Content vs 0.75× dense: need **47.79 bits**. E21 has **0.01**. Do not pass S1
via 0.75×0; E18 is live in the scored JSON.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.73 bits (8k JSON). Not K1. |
| **S1 vs 0.75× E18** | **FAIL** at 8000. 0.01 ≪ 46.98 bits. 800 climb (13.33) did not hold. |
| **content vs 0.75× dense** | **FAIL.** 0.01 ≪ 47.79 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered (dense S0 PASS at H=128). No H=256 bump this turn. |
| **K2** | **PASS.** `e18_local` 24.2% / 0 bits. |
| **K3** | 8k is a floor (flow 0). 800 was climbing; 8k extra-step is the score. |

Do **not** extra-step to 16k (8k floor). Do not another SELECT r. Do not hops.
Do not 4k. Do not Glyph. Do not unfreeze `u`/`delta`. Do not invent a third width.

## Interpretation

**MATCH does not survive 1024 on the default 512 r=8 rem-off H=128 recipe.**
Dense and E18 copy the 64-bit prize at this width; exclusive 8-token means do
not. INDEX at 1024 used **H=256 SSMax log** and r=16 (53.82 bits). This rung
did not bump width because dense was not K1.

Do **not** relabel E18's 62.64 bits as E21. E21 recovered **0.01 bits**.

## Decision

Keep the spec in `ahead/`. Next ONE: seq=1024 packed `recall_single` inplace
frozen mean **r=8 remainder-off**, **H=256 `--global_logit_scale log`** (the
INDEX-passing width). One change: width/SSMax. Same compressor. Not hops. Not
another SELECT r. Not 4k. Not Glyph. Not 16k. Default remainder stays off.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
