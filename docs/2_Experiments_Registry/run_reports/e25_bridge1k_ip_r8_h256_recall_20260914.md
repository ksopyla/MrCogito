# E25 bridge_1k 1024 recall_single r=8 remainder-off H=256 SSMax log — E21 vs E18 vs dense (rung 5ac)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1k_ip_r8_h256_recall` (800, dense/e21 ≤3200) · `e25_1k_ip_r8_h256_recall_s8k` (8000)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_ip_r8_h256_recall/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_ip_r8_h256_recall_probe.log` (8k; also `_800.log`)
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `b0f19cd` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 1024 MATCH H=128 FAIL [`e25_bridge1k_ip_r8_recall_20260914.md`](e25_bridge1k_ip_r8_recall_20260914.md) · 1024 INDEX H=256 log PASS [`e25_bridge1k_ip_r16_mean_far_copy_20260914.md`](e25_bridge1k_ip_r16_mean_far_copy_20260914.md)

---

## Goal

Locked: 1024 MATCH r=8 rem-off **H=128 / logit_scale=none** is S1 FAIL (E21
**0.01 bits @8000**) while dense 63.73 and E18 **62.64**. INDEX at 1024 already
copies with H=256 SSMax log (53.82 bits). One change: **width/SSMax**
(`--hidden 256 --global_logit_scale log`), same r=8 rem-off MATCH recipe.

Score vs 0.75× E18; if E18 were ~0, score vs 0.75× dense. Climbing at 800 →
extra-step 8k. Do **not** pass `--message_pool_remainder`. Do not 16k the H=128
fail. Do not remainder-on or r=1 identity this turn.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** (1024 INDEX width) |
| E21 | query boundary id 10, **r=8**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge_1k` seq=1024, gap=64, window=16, packed answer 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~3GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1k_ip_r8_h256_recall 0 \
  --scale bridge_1k --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 8 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# climbing: same flags --steps 8000 --k1_mult 1
# no --message_pool_remainder
```

Byobu `E25_1k_r8_h256` / `E25_1k_r8_h256s8k`. Log:
`seq=1024  logit_scale=log  msg_r=8  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `hidden: 256`,
`global_logit_scale: log`, `message_ratio: 8`, `message_pool_remainder: false`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`. Recalibrated
dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (8k JSON) | **99.4%** | 500 | **62.90** | 0.983 |
| e18 (8k JSON) | **100%** | 300 | **62.61** | 0.978 |
| **e21 @800** (first hunt) | 38.1% | 800 | climbing | — |
| **e21 @3200** (first hunt) | 49.0% | 3200 | **24.35** | 0.380 |
| **e21 @8000** | **96.5%** (best **97.6%** @7500) | **8000** | **60.35** | **0.943** |
| e18_local (8k JSON) | 23.6% | 8000 | 0 | 0 |

First hunt: dense **87.7% / 56.09 bits** @3200 (**S0 PASS**, no early-stop so
e21 trained 3200). E18 **100% / 63.96** @300 (live). E21 chance through ~700,
38.1% @800, 49.0% / 24.35 @3200 — climbing, extra-step. 8k JSON: dense **62.90**
@500; E18 **62.61** @300 (live — not ~0).

S1 vs 0.75× E18 in the 8k JSON: need **46.96 bits**. E21 has **60.35**.
Content vs 0.75× dense: need **47.18 bits**. E21 has **60.35**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.4% / 62.90 bits (8k JSON). First hunt 87.7% / 56.09 also ≥75%. |
| **S1 vs 0.75× E18** | **PASS** at 8000. 60.35 ≥ 46.96 bits. FAIL at 3200 (24.35). |
| **content vs 0.75× dense** | **PASS.** 60.35 ≥ 47.18 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 23.6% / 0 bits. |
| **K3** | not triggered (climbing at 800/3200; 8k flow 0.94). |

Do **not** extra-step to 16k (S1 already PASS). Do not 16k the H=128 fail. Do
not remainder-on. Do not r=1 identity. Do not hops. Do not 4k. Do not Glyph.
Do not unfreeze `u`/`delta`.

## Interpretation

**MATCH at 1024 needs the INDEX-passing width, not a new pooler.** Same r=8
rem-off frozen mean that failed at H=128 (0.01 bits) copies **60.35 bits** at
H=256 SSMax log — above E18's 0.75× bar and close to dense/E18 (~63 bits).
H=128 was the MATCH@1024 wall, not the compressor.

Do **not** relabel E18's 62.61 bits as E21. E21 recovered **60.35 bits**.

## Decision

Keep the spec in `ahead/`. Next ONE: seq=1024 packed **`select_1decoy` r=8
remainder-off H=256 `--global_logit_scale log`**. One change: MATCH → SELECT
at the width that just passed. Not hops. Not remainder-on. Not r=1 identity.
Not 4k. Not Glyph. Not 16k. Default remainder stays off.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
