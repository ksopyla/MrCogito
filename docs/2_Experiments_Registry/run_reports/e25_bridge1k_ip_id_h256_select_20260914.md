# E25 bridge_1k 1024 select_1decoy r=1 identity H=256 SSMax log — E21 vs E18 vs dense (rung 5ae)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1k_ip_id_h256_select` (800; floor — 8k JSON is this hunt)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_h256_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_h256_select_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `bd1547e` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 1024 SELECT r=8 H=256 log chance [`e25_bridge1k_ip_r8_h256_select_20260914.md`](e25_bridge1k_ip_r8_h256_select_20260914.md) · 1024 MATCH H=256 log PASS [`e25_bridge1k_ip_r8_h256_recall_20260914.md`](e25_bridge1k_ip_r8_h256_recall_20260914.md) · 512 SELECT identity PASS [`e25_bridge512_ip_id_select_20260914.md`](e25_bridge512_ip_id_select_20260914.md)

---

## Goal

Locked: 1024 MATCH r=8 rem-off H=256 SSMax log **S1 PASS (60.35 bits)**. 1024
SELECT on that **same** recipe is **chance 0 bits @800**. At 512, when MATCH
r=16 mean was chance, **r=1 identity** copied MATCH (43 bits) — pooling vs
exclusive channel. This rung is that split for 1024 SELECT, not remainder-on
and not another mean ratio.

Score vs 0.75× E18; if E18 were ~0, score vs 0.75× dense. Climbing at 800 →
extra-step 8k; **floor → do not extra-step**. Do **not** pass
`--message_pool_remainder`. Do not extra width. Do not hops. Do not 4k /
Glyph.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge_1k` seq=1024, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · batch 32 kept, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1k_ip_id_h256_select 0 \
  --scale bridge_1k --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# no --message_pool_remainder
# floor at 800: do not extra-step
```

Byobu `E25_1k_id_h256sel`. Log:
`seq=1024  logit_scale=log  msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `hidden: 256`,
`global_logit_scale: log`, `message_ratio: 1`, `message_pool_remainder: false`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`. Recalibrated
dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **100%** | 300 | **63.96** | 0.999 |
| e18 | **100%** | 350 | **63.80** | 0.997 |
| **e21 @800** | **25.7%** | 800 | **0.00** | **0.000** |
| e18_local | 24.9% | 800 | 0 | 0 |

E21 chance every eval (CE at ln(4) ≈ 1.386). Best acc 25.7% = chance. Not
climbing.

S1 vs 0.75× E18: need **47.85 bits**. E21 has **0**. E18 is live (do not pass
S1 via 0.75×0). Content vs 0.75× dense: need **47.97 bits**. E21 has **0**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.96 bits. |
| **S1 vs 0.75× E18** | **FAIL.** 0 ≪ 47.85 bits. Chance floor. |
| **content vs 0.75× dense** | **FAIL.** 0 ≪ 47.97 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.9% / 0 bits. |
| **K3** | floor at 800 (flow 0). |

Do **not** extra-step to 8k (floor). Do not 16k. Do not remainder-on. Do not
extra width. Do not hops. Do not 4k. Do not Glyph. Do not unfreeze `u`/`delta`.

## Interpretation

**Exclusive SELECT is dead at 1024 even with identity slots.** The 512 split
(MATCH r=16 mean chance → r=1 identity PASS 43 bits) does **not** hold for
1024 SELECT. r=8 means are not uniquely the killer; uncompressed identity KV
also recovers **0 bits**. Stop 1024 SELECT.

Do **not** relabel E18's 63.80 bits as E21. E21 recovered **0 bits**.

## Decision

Keep the spec in `ahead/`. Next ONE: **stop 1024 SELECT** (no remainder-on,
no extra width, no 8k a chance floor). Remaining measured walls: 512 chain
**K1**, 256 hops **FAIL**. Do not hops. Do not 4k. Do not Glyph. Default
remainder stays off.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
