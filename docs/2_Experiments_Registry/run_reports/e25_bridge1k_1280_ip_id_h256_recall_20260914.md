# E25 bridge_1k seq=1280 recall_single r=1 identity H=256 SSMax log — pooling wall vs identity capacity (rung 5bi)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1280_ip_id_h256_recall` (800; S1 PASS — do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_1280_ip_id_h256_recall/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_1280_ip_id_h256_recall/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `32167a0` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 1280 MATCH r=8 exclusive FAIL [`e25_bridge1k_1280_ip_r8_h256_recall_20260914.md`](e25_bridge1k_1280_ip_r8_h256_recall_20260914.md) · 512 MATCH identity PASS [`e25_bridge512_ip_id_recall_20260914.md`](e25_bridge512_ip_id_recall_20260914.md) · 1024 MATCH H=256 log PASS [`e25_bridge1k_ip_r8_h256_recall_20260914.md`](e25_bridge1k_ip_r8_h256_recall_20260914.md)

---

## Goal

Locked: 1280 packed MATCH (`recall_single`) inplace r=8 rem-off H=256 SSMax
log **S1 FAIL (0 bits @800)** while live E18 **63.95 bits** @400. MATCH
length extra-steps are stopped. This rung is the **architecture change at the
measured wall**: same length, **r=1 identity** (the 512 MATCH-winning
identity recipe on the 1024-passing H=256 log width). Hypothesis: if E21
S1 PASS, 1280 FAIL is r=8 pooling (identity still binds a key). If chance
floor, exclusive MATCH dies at 1280 even with token-KV identity slots
(do not then stack SELECT-style attend knobs).

`--scale bridge_1k --seq_len 1280`. Remainder **off**. `--global_layers`
stays **1**. Window default 16 (must stay < gap 64). `--message_ratio 1`
(identity slots, not r=8 mean). No Glyph, no unfreeze `u`/`delta`, no raw
prefix, no hops, no SELECT, no INDEX extra-steps, no MATCH 1152/1408/2048,
no MATCH 1280 r=8 8k. Recalibrate dense S0. Score vs 0.75× **live E18**;
if E18 were ~0, score vs 0.75× dense. Climbing short of S1 → extra-step 8k
only; **floor or S1 PASS → do not extra-step**. Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, pack_stride 0, **`glob_layers=1`** |
| Data | `--scale bridge_1k --seq_len 1280`, gap=64, window=16, packed answer 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65; QUERY 1242 leftover **0** at r=1) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · **batch 32** (~3.9GB, no OOM) |

```
bash scripts/e24_bapo_hunt.sh e25_1280_ip_id_h256_recall 0 \
  --scale bridge_1k --seq_len 1280 --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# --seq_len 1280 and --message_ratio 1 required
# no --message_pool_remainder
# no --global_layers (default 1)
# dense first (underscore --no-dense_first unused)
# S1 PASS at 800: do not extra-step
```

Byobu `E25_1280_ip_id_h256_recall`. Log:
`seq=1280  gap=64  window=16  prize=64.00 bits  answer_len=32  row_gap[min/med/max]=65/65/65`
`logit_scale=log  msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True  glob_layers=1`.

Dense first. Hunt JSON: `hidden: 256`, `global_logit_scale: log`,
`global_layers: 1`, `message_ratio: 1`, `message_pool_remainder: false`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`batch: 32`, `seq_len: 1280`, `scale: bridge_1k`, `calibrated: true`.
Hunt exit **0**. Params <100M. **No new scale enum.**

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense | **100%** | 250 | **63.97** | 0.999 | 0.006247 |
| e18 | **100%** | 400 | **63.95** | 0.999 | 0.006245 |
| **e21 @800** | **100%** | 800 | **63.94** | **0.999** | 0.006244 |
| e18_local | 25.0% | 800 | 0.00 | 0.000 | 0 |

Dense left chance through 150, **55.4% @200**, early-stop **100% @250**.
E18 chance through 300, click **98.9% @350**, early-stop **100% / 63.95
bits** @400 (**live — not ~0**; same as the r=8 1280 control). E21 chance
every eval through **600**, then **47.9% @650**, **49.1% @700**, **73.5%
@750**, early-stop **100% / 63.94 bits** @800. Late click, not a floor.

E18 is **live**. S1 bar is **0.75× E18 = 47.96 bits** (also vs **0.75×
dense = 47.97 bits**). E21 has **63.94**. **8k not run.**

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.97 bits @250. |
| **S1 vs 0.75× live E18** | **PASS.** 63.94 ≥ 47.96 bits. Do not pass via 0.75×0 — E18 is live, and E21 itself is live. |
| **content vs 0.75× dense** | **PASS.** 63.94 ≥ 47.97 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.0% / 0 bits. |
| **K3** | not triggered (flow 0.999). |

Do **not** extra-step to 8k (S1 already PASS). Do not 16k. Do not remainder-on.
Do not stack `keep_local_swa` / extra hops / glob=2 / anchors. Do not Glyph.
Do not unfreeze `u`/`delta`. Do not MATCH 2048. Do not 1152/1408. Do not hops
268. Do not relabel live E18 as E21.

## Interpretation

**1280 exclusive MATCH FAIL was r=8 pooling, not identity-capacity.** On
the same seq=1280 packed MATCH rows, H=256 SSMax log, remainder off,
glob=1, and the same QUERY 1242, r=8 frozen mean was **0 bits / chance
@800** while r=1 identity **copies 63.94 bits** (full prize). Dense still
copies 64 bits in 250 steps (S0). E18 recovers **63.95 bits** at 400 —
do **not** relabel that as E21. E21's own 63.94 bits is the exclusive
token-KV identity channel, slower to click (600→800) than E18's 350
but the same 64-bit copy once it binds.

This is the same split as 512 MATCH: identity binds a key; mean pooling
kills MATCH. It is **not** the 1024 SELECT identity class (chance even
with r=1 slots). r=8 exclusive MATCH wall stays **(1024 PASS, 1280
FAIL]**; identity MATCH is **live at 1280**.

## Decision

Keep the spec in `ahead/`. Next ONE (do not run it): parent may later
**map identity MATCH length** (not 1152 r=8). Do not stack
`keep_local_swa` / extra hops / glob=2 / anchors here. MATCH r=8 length
extra-steps stay stopped. INDEX extra-steps stay stopped. Hops extra-steps
stay stopped. SELECT extra-steps stay stopped. Not Glyph. Do not unfreeze
`u`/`delta`. Do not 8k. Do not 16k. Do not MATCH 2048.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
