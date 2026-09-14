# E25 bridge_1k seq=1536 recall_single r=1 identity H=256 SSMax log — identity MATCH length wall (rung 5bj)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1536_ip_id_h256_recall` (800; floor — do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_1536_ip_id_h256_recall/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_1536_ip_id_h256_recall/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `0d0e9be` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 1280 MATCH r=1 identity S1 PASS [`e25_bridge1k_1280_ip_id_h256_recall_20260914.md`](e25_bridge1k_1280_ip_id_h256_recall_20260914.md) · 1536 MATCH r=8 exclusive FAIL [`e25_bridge1k_1536_ip_r8_h256_recall_20260914.md`](e25_bridge1k_1536_ip_r8_h256_recall_20260914.md) · 512 MATCH identity PASS [`e25_bridge512_ip_id_recall_20260914.md`](e25_bridge512_ip_id_recall_20260914.md)

---

## Goal

Locked: identity MATCH is live at 1280 (`recall_single` r=1 H=256 SSMax
log **S1 PASS 63.94 bits @800**, late click). r=8 MATCH wall stays
**(1024 PASS, 1280 FAIL]**. At 1536, r=8 E21 was chance while E18 was
live (52.58). This rung maps **identity MATCH length** past the pooling
wall: same compressor as 1280 identity PASS, `--scale bridge_1k
--seq_len 1536` (probe override; **do not reuse the r=8 1536 JSON**).
Hypothesis: if E21 S1 PASS, identity still binds a key at 1536. If
chance floor, identity MATCH dies in **(1280 PASS, 1536 FAIL]**.

`--scale bridge_1k --seq_len 1536`. Remainder **off**. `--global_layers`
stays **1**. Window default 16 (must stay < gap 64). `--message_ratio 1`
(identity slots, not r=8 mean). No Glyph, no unfreeze `u`/`delta`, no raw
prefix, no hops, no SELECT, no INDEX extra-steps, no MATCH 2048, no
1152/1408 r=8, no `keep_local_swa` / extra hops / glob=2 / anchors.
Recalibrate dense S0. Score vs 0.75× **live E18**; if E18 were ~0, score
vs 0.75× dense. Climbing short of S1 → extra-step 8k only; **floor → do
not extra-step**. Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, pack_stride 0, **`glob_layers=1`** |
| Data | `--scale bridge_1k --seq_len 1536`, gap=64, window=16, packed answer 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65; QUERY 1498 leftover **0** at r=1; r=8 residue was 2) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · **batch 32** (~3.7–4.7GB, no OOM) |

```
bash scripts/e24_bapo_hunt.sh e25_1536_ip_id_h256_recall 0 \
  --scale bridge_1k --seq_len 1536 --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# --seq_len 1536 and --message_ratio 1 required
# no --message_pool_remainder
# no --global_layers (default 1)
# dense first (underscore --no-dense_first unused)
# floor at 800: do not extra-step
```

Byobu `E25_1536_ip_id_h256_recall`. Log:
`seq=1536  gap=64  window=16  prize=64.00 bits  answer_len=32  row_gap[min/med/max]=65/65/65`
`logit_scale=log  msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True  glob_layers=1`.

Dense first. Hunt JSON: `hidden: 256`, `global_logit_scale: log`,
`global_layers: 1`, `message_ratio: 1`, `message_pool_remainder: false`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`batch: 32`, `seq_len: 1536`, `scale: bridge_1k`, `calibrated: true`.
Hunt exit **0**. Params <100M. **No new scale enum.** Recalibrated dense
in this JSON (not the r=8 1536 bundle).

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense | **100%** | 200 | **63.95** | 0.999 | 0.005204 |
| e18 | **100%** | 400 | **63.95** | 0.999 | 0.005204 |
| **e21 @800** | **29.2%** | 800 | **1.50** | **0.023** | 0.000122 |
| e18_local | 24.7% | 800 | 0.00 | 0.000 | 0 |

Dense left chance through 100, **49.3% @150**, early-stop **100% @200**.
E18 chance through 300, click **97.7% @350**, early-stop **100% / 63.95
bits** @400 (**live — not ~0**; stronger than this length's r=8 E18
52.58). E21 chance every eval through **750** (acc 23.1–27.7%, CE at
ln(4)). Last eval **29.2% / 1.50 bits** @800 (CE 1.354 vs floor 1.386;
flow **0.023 < 0.05**). Last-eval wiggle, not a 1280-style click (that
was 47.9% @650 with CE collapsing). Best acc is the last step.

E18 is **live**. S1 bar is **0.75× E18 = 47.96 bits** (also vs **0.75×
dense = 47.96 bits**). E21 has **1.50**. **8k not run** (chance floor, not
climbing short of S1).

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.95 bits @200. |
| **S1 vs 0.75× live E18** | **FAIL.** 1.50 ≪ 47.96 bits. Chance floor. Do not pass via 0.75×0 — E18 is live. |
| **content vs 0.75× dense** | **FAIL.** 1.50 ≪ 47.96 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.7% / 0 bits. |
| **K3** | floor at 800 (flow 0.023 < 0.05). |

Do **not** extra-step to 8k (floor). Do not 16k. Do not remainder-on. Do
not stack `keep_local_swa` / extra hops / glob=2 / anchors. Do not Glyph.
Do not unfreeze `u`/`delta`. Do not MATCH 2048. Do not 1408. Do not hops
268. Do not 1152 r=8. Do not relabel live E18 as E21.

## Interpretation

**Identity MATCH that passed at 1280 is chance at 1536.** On the same
H=256 SSMax log, remainder off, glob=1, r=1 identity compressor that
copied **63.94 bits** at 1280, packed MATCH at seq=1536 is **1.50 bits
/ flow 0.023** after 800 steps. Dense still copies 64 bits in 200 steps
(S0). E18 recovers **63.95 bits** at 400 — do **not** relabel that as
E21. The last-eval 29.2% / 1.50 bits is not the 1280 late click (chance
through 600, 47.9% @650, 100% @800). Here at 650 E21 is still 27.5% at
floor CE.

r=8 exclusive MATCH wall stays **(1024 PASS, 1280 FAIL]**. Identity
MATCH wall is **(1280 PASS, 1536 FAIL]**. Exclusive token-KV identity
binds a key at 1280 and does not at 1536. This is not a pooling residue
(leftover 0 at r=1). It is not the shared INDEX length wall (E18 ~0 at
1536 INDEX; here E18 fully copies).

## Decision

Keep the spec in `ahead/`. Next ONE (do not run it): **STOP identity
MATCH extra-steps** at this 256-token resolution (do not 1408). Do not
2048 identity MATCH. Do not stack attend knobs. Do not re-open r=8
length. MATCH r=8 length extra-steps stay stopped. INDEX extra-steps stay
stopped. Hops extra-steps stay stopped. SELECT extra-steps stay stopped.
Not Glyph. Do not unfreeze `u`/`delta`. Do not 8k. Do not 16k.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
