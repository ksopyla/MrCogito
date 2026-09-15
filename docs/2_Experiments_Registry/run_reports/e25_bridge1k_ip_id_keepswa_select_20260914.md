# E25 bridge_1k 1024 select_1decoy r=1 identity SWA-unsever H=256 SSMax log — E21 vs E18 vs dense (rung 5ah)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1k_ip_id_keepswa_select` (800; floor — do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_keepswa_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_keepswa_select_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `8fa25ef` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 1024 SELECT identity floor [`e25_bridge1k_ip_id_h256_select_20260914.md`](e25_bridge1k_ip_id_h256_select_20260914.md) · 1024 SELECT r=8 chance [`e25_bridge1k_ip_r8_h256_select_20260914.md`](e25_bridge1k_ip_r8_h256_select_20260914.md)

---

## Goal

Locked: 1024 packed `select_1decoy` inplace r=1 identity remainder-off H=256 SSMax
log **S1 FAIL (0 bits @800)** while E18 copies **63.80 bits**. Inspection: r=1
inplace identity is **not** a type-cue coverage hole (every sender token is a
`replace` slot on the exclusive global read). One architectural knob: keep
exclusive identity slots for the global read, but **do not sever SWA**
(`--message_keep_local_swa`, default off).

Score vs 0.75× E18; if E18 were ~0, score vs 0.75× dense. Climbing at 800 →
extra-step 8k; **floor → do not extra-step**. Do **not** restore raw global KV.
Do not remainder-on. Do not H=512. Do not hops. Do not Glyph. Do not unfreeze
`u`/`delta`.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=True`** |
| Data | `--scale bridge_1k` seq=1024, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~2.7GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1k_ip_id_keepswa_select 0 \
  --scale bridge_1k --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --message_keep_local_swa --steps 800 --k1_mult 4
# no --message_pool_remainder
# floor at 800: do not extra-step
```

Byobu `E25_1k_id_keepswa`. Log:
`seq=1024  logit_scale=log  msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True  msg_keepswa=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `hidden: 256`,
`global_logit_scale: log`, `message_ratio: 1`, `message_pool_remainder: false`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: true`. Recalibrated dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **100%** | 200 | **63.03** | 0.985 |
| e18 | **100%** | 400 | **63.94** | 0.999 |
| **e21 @800** | **25.8%** | 800 | **0.01** | **0.000** |
| e18_local | 24.9% | 800 | 0 | 0 |

E21 chance every eval (CE at ln(4) ≈ 1.386). Best acc 25.8% = chance. Not
climbing.

S1 vs 0.75× E18: need **47.95 bits**. E21 has **0.01**. E18 is live (do not pass
S1 via 0.75×0). Content vs 0.75× dense: need **47.27 bits**. E21 has **0.01**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.03 bits @200. |
| **S1 vs 0.75× E18** | **FAIL.** 0.01 ≪ 47.95 bits. Chance floor. |
| **content vs 0.75× dense** | **FAIL.** 0.01 ≪ 47.27 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.9% / 0 bits. |
| **K3** | floor at 800 (flow 0). |

Do **not** extra-step to 8k (floor). Do not 16k. Do not remainder-on. Do not
restore raw global KV. Do not extra width. Do not hops. Do not Glyph. Do not
unfreeze `u`/`delta`.

## Interpretation

**Unsevering SWA does not rescue 1024 SELECT.** Exclusive identity slots still
recover **0.01 bits** (chance) while uncompressed E18 copies **63.94 bits** on
the same rows. The local window is 16 and the right-align gap is 100, so SWA
cannot reach the type-cue evidence even when QUERY is not a document start. The
exclusive global identity path remains the E21-specific wall. Do **not**
relabel E18's 63.94 bits as E21.

## Decision

Keep the spec in `ahead/`. Next ONE: **stop 1024 SELECT knobs** (no remainder-on,
no extra width, no 8k on a chance floor, do not restore raw global KV — that is
E18). Remaining measured DNA walls: 512 chain **K1**, 256 hops **FAIL**, 2048+
INDEX shared with E18. Not Glyph. Do not unfreeze `u`/`delta`. Default remainder
stays off. Default `--message_keep_local_swa` stays off.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
