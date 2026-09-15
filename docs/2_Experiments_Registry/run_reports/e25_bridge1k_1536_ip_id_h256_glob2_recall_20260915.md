# E25 bridge_1k seq=1536 recall_single r=1 identity `--global_layers 2` H=256 SSMax log — two exclusive global layers at the identity MATCH wall (rung 5bp)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1536_ip_id_h256_glob2_recall` (800; floor — do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_1536_ip_id_h256_glob2_recall/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_1536_ip_id_h256_glob2_recall/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `252dc6b` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 1536 MATCH identity glob=1 FAIL [`e25_bridge1k_1536_ip_id_h256_recall_20260914.md`](e25_bridge1k_1536_ip_id_h256_recall_20260914.md) · 1280 MATCH identity S1 PASS [`e25_bridge1k_1280_ip_id_h256_recall_20260914.md`](e25_bridge1k_1280_ip_id_h256_recall_20260914.md) · 1024 SELECT glob=2 chance [`e25_bridge1k_ip_id_glob2_select_20260914.md`](e25_bridge1k_ip_id_glob2_select_20260914.md)

---

## Goal

MATCH length extra-steps are stopped. Identity MATCH wall with glob=1 is
**(1280 S1 PASS 63.94, 1536 FAIL 1.50]** while uncompressed glob=1 E18 is
**100% / 63.95** at 1536. SWA sever is not the 1536 MATCH killer (1280
identity PASSES with severed SWA; SELECT 696 keepswa still 0.01). Hops
needed `--global_layers 2` for 2-hop.

Hypothesis: one exclusive attend cannot bind a key among ~1536 identity
slots; glob=2 mixing recovers MATCH. If chance, exclusive identity MATCH
dies at 1536 even with two layers (SELECT glob=2 class). Default
`--global_layers` stays **1** after the hunt.

Same 1536 identity compressor plus `--global_layers 2`. Remainder **off**.
`keep_local_swa` off. No extra hops / `update_slot_kv` / anchors /
`pack_stride`. Window 16 < gap 64. Recalibrate dense S0 (do **not** reuse
the glob=1 1536 JSON). Score vs 0.75× **live E18**; if this-JSON E18 ≈ 0,
do **not** pass S1 via 0.75×0 — score vs 0.75× **dense**. Climbing short of
S1 → extra-step 8k only; **chance at 800 → do not extra-step**. Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.802M (e21 2.835M) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`glob_layers=2`**, **`msg_anchors=none`**, **`msg_packstride=0`** |
| Data | `--scale bridge_1k --seq_len 1536`, gap=64, window=16, packed answer 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65; leftover **0** at r=1) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~5.5GB, 0.17 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1536_ip_id_h256_glob2_recall 0 \
  --scale bridge_1k --seq_len 1536 --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --global_layers 2 --steps 800 --k1_mult 4
# --seq_len 1536, --message_ratio 1, --global_layers 2 required
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# no --message_global_anchors (default none)
# no --message_pack_stride
# no --local_window (default 16 < gap 64)
# dense first (underscore --no-dense_first unused)
# 8k extra-step NOT run (chance floor @800)
```

Byobu `E25_1536_id_glob2`. Log:
`seq=1536  gap=64  window=16  prize=64.00 bits  answer_len=32
row_gap[min/med/max]=65/65/65  logit_scale=log
msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False
msg_idslots=True  msg_keepswa=False  glob_layers=2  msg_extrahops=0
msg_updatekv=False  msg_anchors=none  msg_packstride=0`.
e21 `patterns=[('swa', 16), ('full', 0), ('full', 0), ('swa', 16), ('swa', 16)]`.

Hunt JSON: `seq_len: 1536`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 2`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`message_pack_stride: 0`, `message_override: real`, `prize_bits: 64.0`,
`calibrated: true`, `dense_steps_used: 250`. Hunt exit **0**. Params <100M.
Dense S0 recalibrated in this JSON (not the glob=1 1536 bundle). Code
default `--global_layers` stays **1**.

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense | **100%** | 250 | **63.98** | 1.000 | 0.005207 |
| e18 | 23.5% | 800 | **0.00** | 0.000 | 0 |
| **e21 @800** | **24.7%** | 800 | **0.00** | **0.000** | 0 |
| e18_local | 24.7% | 800 | 0.00 | 0.000 | 0 |

Params: dense/e18 ~2.802M, e21 2.835M (<100M). Prize 64 bits. Chance ~25%.
Dense left chance through 150, **62.7% @200**, early-stop **100% / 63.98
bits** @250 (**S0 PASS**). This-JSON E18 chance every eval (acc 23.5–27.5%,
CE at ln(4); best 27.5%). E21 CE glued to ln(4)≈1.386 every eval (min 1.3856
@600, max 1.395; acc never > 27.5%; best 27.5% @250/600/650). Last eval
**24.7% / 0 bits** @800 (CE 1.388 vs floor 1.386; flow **0**). Not climbing.
Not a 1280-style late click (that was 47.9% @650). glob=1 1536 last-eval
wiggle was 29.2% / 1.50 bits; this is cleaner chance.

This-JSON E18 is **~0 bits**. Do **not** pass S1 via 0.75×0. S1 bar is
**0.75× dense = 47.99 bits**. E21 has **0**. Prior glob=1 1536 E18 was live
**63.95 bits** @400 — that is a different JSON (`glob_layers=1`); do **not**
relabel it as this hunt's E18 or as E21. **8k not run** (E21 chance floor,
not climbing short of S1).

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.98 bits @250. Recalibrated. |
| **S1 vs 0.75× live E18** | **FAIL.** This-JSON E18 is 0 bits. Do not pass via 0.75×0. Scored vs 0.75× dense **47.99**. E21 0 ≪ 47.99. Chance floor. |
| **content vs 0.75× dense** | **FAIL.** 0 ≪ 47.99 bits. |
| **S2** plots | **PASS.** traces + `learning_curves` / `recovered_bits` / `information_flow` in hunt artifacts. |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.7% / 0.00 bits. |
| **K3** | floor at 800 (flow 0). |

Do **not** extra-step to 8k (floor). Do not 16k. Do not remainder-on. Do
not `keep_local_swa` / extra hops / anchors / pack_stride. Do not glob=3.
Do not MATCH 1408. Do not hops 320. Do not SELECT 694. Do not Glyph. Do
not unfreeze `u`/`delta`. Do not relabel glob=1 live E18 63.95 as E21.

## Interpretation

**Two exclusive global layers do not recover identity MATCH at 1536.**
On the same H=256 SSMax log, remainder off, r=1 identity compressor that
copied **63.94 bits** at 1280 glob=1 and was **1.50 bits** at 1536 glob=1,
`--global_layers 2` recovers **0 bits / flow 0** after 800 steps. Dense
still copies 64 bits in 250 steps (S0). This-JSON E18 is also chance at
800 (glob=2 delayed the uncompressed click that glob=1 scored **63.95
@400**); that does not license an 8k extra-step, because E21 is a chance
floor, not climbing short of S1.

This is SELECT glob=2 class: two full exclusive attend+FFN Blocks over
slots (extra hops 0, not `stack_layers=2`) still leave content addressing
dead at the measured wall. Hops needed glob=2 for 2-hop; MATCH identity
at 1536 does not. Identity MATCH wall **(1280 PASS, 1536 FAIL]** stands
at glob=2. r=8 exclusive MATCH wall stays **(1024 PASS, 1280 FAIL]**.
Do **not** relabel glob=1 E18's 63.95 bits as E21.

## Decision

Keep the spec in `ahead/`. **8k not run** (chance floor). Do **not** 16k.
Default `--global_layers` stays **1**. Default remainder stays off.
Default `keep_local_swa` stays false. Default extra hops stay 0. Next ONE
(do not run): **STOP MATCH glob=2 stacks at 1536**; identity wall stands.
If this had PASSed, parent may later map identity MATCH length with glob=2
(not 1408 glob=1). Do not extra hops / keepswa / anchors in a MATCH glob=2
follow-up. Do not hops 320. Do not SELECT 694.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
