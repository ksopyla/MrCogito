# E25 bridge_1k seq=1536 far_copy r=16 remainder-off H=256 SSMax log — E21 vs E18 vs dense (rung 5ay)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1536_ip_r16_mean` (800; floor — do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_1536_ip_r16_mean/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_1536_ip_r16_mean/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `4880a7a` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 1024 INDEX H=256 log PASS [`e25_bridge1k_ip_r16_mean_far_copy_20260914.md`](e25_bridge1k_ip_r16_mean_far_copy_20260914.md) · 2048 INDEX chance [`e25_bridge1k_2k_ip_r16_mean_far_copy_20260914.md`](e25_bridge1k_2k_ip_r16_mean_far_copy_20260914.md) · 4k INDEX chance [`e25_medium_4k_ip_r16_mean_far_copy_20260914.md`](e25_medium_4k_ip_r16_mean_far_copy_20260914.md)

---

## Goal

Locked: 1024 packed INDEX (`far_copy`) inplace r=16 frozen mean **S1 PASS
(53.82 bits @8k)**; 2048 and 4096 packed INDEX **S1 FAIL (0 bits @800)** vs
0.75× dense. This rung **brackets** the INDEX length wall: same 1024-passing
compressor, `--scale bridge_1k --seq_len 1536` (probe override; no new
scale). Window/gap stay `16 < 64` (right-align gap 65). QUERY leftover is
**12** at 1024 / 1536 / 2048 (same r=16 residue). Packed span 32 / 64-bit
prize.

E18 may be ~0 (E24 2k/4k; E25 2k/4k). Do **not** pass S1 via 0.75×0; score vs
0.75× dense. Recalibrate dense in this JSON. Climbing at 800 → extra-step
8k; **floor → do not extra-step**. Do **not** pass `--message_pool_remainder`.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=16**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, pack_stride 0 |
| Data | `--scale bridge_1k --seq_len 1536`, gap=64, window=16, packed span 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · **batch 32** (~4.3–4.7GB, no OOM) |

```
bash scripts/e24_bapo_hunt.sh e25_1536_ip_r16_mean 0 \
  --scale bridge_1k --seq_len 1536 --recipe far_copy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 16 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# no --message_pool_remainder
# dense first (underscore --no-dense_first unused)
# floor at 800: do not extra-step
```

Byobu `E25_1536_ip_mean`. Log:
`seq=1536  gap=64  window=16  prize=64.00 bits  answer_len=32  row_gap[min/med/max]=65/65/65`
`logit_scale=log  msg_r=16  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first. Hunt JSON: `hidden: 256`, `global_logit_scale: log`,
`message_ratio: 16`, `message_pool_remainder: false`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`batch: 32`, `seq_len: 1536`, `scale: bridge_1k`, `calibrated: true`.
Hunt exit **0**. Params <100M. **No new scale enum.**

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense | **99.1%** | 300 | **62.41** | 0.975 | 0.005079 |
| e18 | **25.8%** | 800 | **0.00** | 0.000 | 0 |
| **e21 @800** | **24.8%** | 800 | **0.00** | **0.000** | **0** |
| e18_local | 26.0% | 800 | 0.02 | 0.000 | 1.92e-6 |

Dense left chance through 200 (24.8%), **97.0% @250**, early-stop **99.1% @300**.
E18 chance every eval (CE at ln(4); **0 bits**; best 27.1% @600). E21 chance
every eval (CE at ln(4); **0 bits**; best 26.0% @50). Not climbing.

E18 is **about 0**. Do **not** pass S1 via 0.75×0 (0 bits). Score vs
**0.75× dense = 46.81 bits**. E21 has **0**. **8k not run.**

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.1% / 62.41 bits @300. |
| **S1 vs 0.75× E18** | **not the bar.** E18 ~0 bits; 0.75×0 is not a pass. |
| **S1 vs 0.75× dense** | **FAIL.** 0 ≪ 46.81 bits. Chance floor. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 26.0% / 0.02 bits. |
| **K3** | floor at 800 (flow 0). |

Do **not** extra-step to 8k (floor). Do not 16k. Do not remainder-on. Do
not extra width (dense already S0 at H=256). Do not hops. Do not Glyph.
Do not unfreeze `u`/`delta`. Do not seq=1280 this turn.

## Interpretation

**Exclusive compressed INDEX that passed at 1024 is chance at 1536** on the
same H=256 SSMax-log identity-mean compressor **and the same gap=64 / window=16
/ leftover 12**. Dense still copies 62 bits in 300 steps, so the instrument is
live (not K1). E18 is also 0 bits at 800 — this is the one-global-read length
wall, **not an E21-only pooling failure**. Do **not** relabel E18's 0 bits as
an E21 pass via 0.75×0.

The INDEX length wall for this recipe sits in **(1024 PASS, 1536 FAIL]**. 2048
and 4096 were already chance; 1536 with the 1024 gap already kills both E18
and E21. Changing the E21 compressor here would chase E18's length wall.

## Decision

Keep the spec in `ahead/`. Next ONE: **stop INDEX length extra-steps** (do not
seq=1280). Shared wall is **(1024 PASS, 1536 FAIL]**. Do not remainder-on. Do
not 1536 H=512 (dense S0 already passed). Do not reopen SELECT **(692 PASS,
696 FAIL]**. Remaining measured DNA walls: 512 chain **K1**, 256 hops **FAIL**.
Not Glyph. Do not unfreeze `u`/`delta`.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
