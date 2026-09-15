# E25 bridge_1k seq=1280 recall_single r=4 remainder-on H=256 SSMax log — leftover rescue at the r=4 wall (rung 5bl)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1280_ip_r4_rem_h256_recall` (800; dense-first, dense used 1550) · `e25_1280_ip_r4_rem_h256_recall_s8k` (8000 extra-step; `--no-dense_first --k1_mult 1`; chance floor)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_1280_ip_r4_rem_h256_recall/` · `/opt/cursor/artifacts/e25_bridge1k_1280_ip_r4_rem_h256_recall_s8k/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_1280_ip_r4_rem_h256_recall/probe.log` · `/opt/cursor/artifacts/e25_bridge1k_1280_ip_r4_rem_h256_recall_s8k/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `5b07039` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 1280 MATCH r=4 rem-off FAIL [`e25_bridge1k_1280_ip_r4_h256_recall_20260914.md`](e25_bridge1k_1280_ip_r4_h256_recall_20260914.md) · 1280 MATCH r=1 identity S1 PASS [`e25_bridge1k_1280_ip_id_h256_recall_20260914.md`](e25_bridge1k_1280_ip_id_h256_recall_20260914.md) · 512 MATCH r=10 rem-on leftover-2 rescue [`e25_bridge512_ip_r10_rem_recall_20260914.md`](e25_bridge512_ip_r10_rem_recall_20260914.md)

---

## Goal

Locked: pooling A/Bs (r=2/r=6) are stopped. MATCH length extra-steps are stopped. At seq=1280, r=1 identity S1 PASS 63.94; r=4 rem-off S1 FAIL (24.91 @800 then 0 @8k); r=8 rem-off exclusive FAIL 0. QUERY leftover **2** for both r=4 and r=8 — the same leftover-2 class that remainder-on rescued at 512 MATCH (r=10 rem-off chance → rem-on S1 PASS).

This rung is leftover rescue at the measured r=4 wall, not a new pooling ratio. Same compressor as the r=4 rem-off FAIL (`--message_ratio 4 --message_slots_inplace --message_identity_slots`, `u`/`delta` frozen), plus `--message_pool_remainder`. Default remainder stays **off** in code after this hunt.

Hypothesis: if E21 S1 PASS, the 1280 r=4 FAIL was leftover-2 drop, not 4-token means. If chance/fail, 4-token means die at 1280 even with leftover kept — pooling-ratio wall **(r=1 PASS, r=4 FAIL]** stands.

`--scale bridge_1k --seq_len 1280`. `--global_layers` stays **1**. Window default 16 (must stay < gap 64). No Glyph, no unfreeze `u`/`delta`, no raw prefix, no hops, no SELECT, no INDEX extra-steps, no MATCH 1408/2048, no 1152 r=8, no r=2/r=6, no attend-stack knobs. Recalibrate dense S0. Score vs 0.75× **live E18**; if E18 were ~0, score vs 0.75× dense. Also report vs 0.75× dense. Climbing short of S1 → extra-step 8k only; **do not 16k**.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=4**, remainder **on**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, pack_stride 0, **`glob_layers=1`** |
| Data | `--scale bridge_1k --seq_len 1280`, gap=64, window=16, packed answer 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65; QUERY **1242** leftover **2** at r=4; same residue as r=8) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · **batch 32** (~3.2–3.9GB, no OOM) |

```
bash scripts/e24_bapo_hunt.sh e25_1280_ip_r4_rem_h256_recall 0 \
  --scale bridge_1k --seq_len 1280 --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 4 --message_slots_inplace --message_identity_slots \
  --message_pool_remainder --steps 800 --k1_mult 4
# --seq_len 1280 and --message_ratio 4 required
# --message_pool_remainder ON (leftover pooling; code default stays off)
# no --global_layers (default 1)
# dense first (underscore --no-dense_first unused on the 800 hunt)
# @800 E21 climbing short of S1 (~6.80 bits) → 8k extra-step:
bash scripts/e24_bapo_hunt.sh e25_1280_ip_r4_rem_h256_recall_s8k 0 \
  --scale bridge_1k --seq_len 1280 --recipe recall_single --arch e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 4 --message_slots_inplace --message_identity_slots \
  --message_pool_remainder --steps 8000 --k1_mult 1 --no-dense_first
# dense already S0 PASS 99.7% / 62.39 in the 800 JSON; underscore --no-dense_first
# do not 16k
```

Byobu `E25_1280_ip_r4_rem_h256_recall` then `E25_1280_ip_r4_rem_h256_recall_s8k`. Log:
`seq=1280  gap=64  window=16  prize=64.00 bits  answer_len=32  row_gap[min/med/max]=65/65/65`
`logit_scale=log  msg_r=4  msg_remainder=True  msg_inplace=True  msg_rawkv=False  msg_idslots=True  glob_layers=1`.

QUERY leftover confirmed: `tail_len=38`, QUERY at **1242**, `1242 % 4 = 2` (same as r=8). Remainder-on pools that incomplete last sender block.

800 hunt JSON: `hidden: 256`, `global_logit_scale: log`, `global_layers: 1`,
`message_ratio: 4`, `message_pool_remainder: true`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`, `batch: 32`, `seq_len: 1280`, `scale: bridge_1k`,
`calibrated: true`. Hunt exit **0**. 8k hunt JSON: same remainder **true**; exit **0**.
Params <100M. **No new scale enum.** Code default `message_pool_remainder=False` unchanged.

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense (800 JSON) | **99.7%** | 1550 | **62.39** | 0.975 | 0.006092 |
| e18 (800 JSON) | **75.7%** | 1550 | **39.90** | 0.623 | 0.003896 |
| **e21 @800** | **39.0%** | 800 | **~6.80** | climbing | — |
| **e21 @1550** | **72.0%** (best **73.7%** @1450) | 1550 | **45.31** | **0.708** | 0.004425 |
| e18_local (800 JSON) | 25.0% | 1550 | 0.00 | 0.000 | 0 |
| e18 (8k JSON) | **100%** | 400 | **63.96** | 0.999 | 0.006246 |
| **e21 @8000** | **22.8%** (best **26.7%**) | 8000 | **0.00** | **0.000** | **0** |
| e18_local (8k JSON) | 25.7% | 8000 | 0.00 | 0.000 | 0 |

Dense left chance through 300, plateaued ~87–88% from 450–1450, then **93.3% @1500**, early-stop **99.7% / 62.39 bits** @1550. E18 (800) clicked ~65% @400, plateaued ~75% from 600–1550 (**live — not ~0**; last-eval CE spike 0.52). E21 chance every eval through **700**, then **30.3% @750**, **39.0% / ~6.80 bits** @800, **43.8% @850**, **51.7% @1100**, **70.1% @1300**, **73.7% @1450**, **72.0% / 45.31 bits** @1550. Climbing at 800, short of S1. Not a floor.

8k extra-step skipped dense. E18 early-stop **100% / 63.96 bits** @400 (**live**). E21 chance every eval (CE at ln(4); **0 bits**; best 26.7%). The 1550 climb **did not replicate**.

E18 is **live** in both JSONs. S1 bar at 800/1550 is **0.75× E18 = 29.92 bits** (also vs **0.75× dense = 46.79 bits**). E21 has **~6.80 @800** (short of S1) and **45.31 @1550** (over this-JSON E18 bar, short of the dense bar). S1 bar at 8k is **0.75× live E18 = 47.97 bits** (also vs **0.75× dense = 46.79 bits**). E21 has **0**. **8k ran. Do not 16k.**

Do **not** pass S1 via the 1550 45.31 vs a weak E18 39.90 — at the 800-step floor E21 was short, so extra-step was required; 8k E18 is the live 64-bit copy. Do **not** relabel live E18 as E21.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.7% / 62.39 bits @1550. |
| **S1 vs 0.75× live E18** | **FAIL.** ~6.80 < 29.92 @800 (climbing); 0 ≪ 47.97 @8k (chance). Do not pass via 0.75×0 — E18 is live in both JSONs (39.90 / 63.96). |
| **content vs 0.75× dense** | **FAIL.** 45.31 < 46.79 @1550 (1.48-bit miss); 0 ≪ 46.79 @8k. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.0% / 0 bits @1550; 25.7% / 0 bits @8k. |
| **K3** | 1550 climb flow 0.708; 8k floor flow 0. |

Do **not** extra-step to 16k. Do not r=8 rem-on. Do not r=2/r=6 A/B. Do not MATCH 1408. Do not stack `keep_local_swa` / extra hops / glob=2 / anchors. Do not Glyph. Do not unfreeze `u`/`delta`. Do not relabel live E18 as E21. Default remainder stays **off**.

## Interpretation

**Leftover rescue does not hold at 1280 r=4.** QUERY leftover is still **2** (`1242 % 4 = 2`, same residue as r=8 FAIL). Remainder-on is the same leftover-2 class that rescued 512 MATCH r=10 (rem-off chance → rem-on S1 PASS 44.21). Here rem-on climbed to **45.31 bits @1550** on the dense-stretched 800 hunt, then **returned to chance at 8k** — the same extra-step pattern as rem-off r=4 (24.91 @800 then 0 @8k). Dense still copies 62 bits (S0). E18 recovers **63.96 bits** at 400 on the 8k JSON — do **not** relabel that as E21.

4-token frozen means die at 1280 even with leftover kept. Pooling-ratio wall at 1280 stays **(r=1 PASS, r=4 FAIL]**. r=8 exclusive MATCH wall stays **(1024 PASS, 1280 FAIL]**. Identity MATCH wall stays **(1280 PASS, 1536 FAIL]**. Remainder-on at 512 MATCH through r=16 does not transfer to this length.

## Decision

Keep the spec in `ahead/`. Next ONE (do not run it): **STOP remainder stacks at 1280 r=4.** Pooling-ratio wall stands. Do not r=8 rem-on at 1280. Do not r=2/r=6. Do not MATCH 1408. Default remainder stays off. MATCH length extra-steps stay stopped. INDEX extra-steps stay stopped. Hops extra-steps stay stopped. SELECT extra-steps stay stopped. Not Glyph. Do not unfreeze `u`/`delta`. Do not 16k. Do not hops 268.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
