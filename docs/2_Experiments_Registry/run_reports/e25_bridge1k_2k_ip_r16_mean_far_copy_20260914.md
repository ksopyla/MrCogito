# E25 bridge_1k seq=2048 far_copy r=16 remainder-off H=256 SSMax log — E21 vs E18 vs dense (rung 5ag)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_2k_ip_r16_mean` (800; floor — do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_2k_ip_r16_mean/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_2k_ip_r16_mean_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `9a5ff15` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 1024 INDEX H=256 log PASS [`e25_bridge1k_ip_r16_mean_far_copy_20260914.md`](e25_bridge1k_ip_r16_mean_far_copy_20260914.md) · 4k INDEX chance [`e25_medium_4k_ip_r16_mean_far_copy_20260914.md`](e25_medium_4k_ip_r16_mean_far_copy_20260914.md)

---

## Goal

Locked: 1024 packed INDEX (`far_copy`) inplace r=16 frozen mean **S1 PASS
(53.82 bits @8k)**; 4096 packed INDEX on `--scale medium` **S1 FAIL (0 bits
@800)** vs 0.75× dense. This rung **brackets** the INDEX length wall: same
1024-passing compressor, `--scale bridge_1k --seq_len 2048` (probe override;
no new scale). Window/gap stay `16 < 64` (right-align gap 65).

E18 may be ~0 (E24 2k/4k; E25 4k). Do **not** pass S1 via 0.75×0; score vs
0.75× dense. Recalibrate dense in this JSON. Climbing at 800 → extra-step
8k; **floor → do not extra-step**. Do **not** pass `--message_pool_remainder`.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=16**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge_1k --seq_len 2048`, gap=64, window=16, packed span 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · **batch 32** (1024 default; ~5.7GB, no OOM) |

```
bash scripts/e24_bapo_hunt.sh e25_2k_ip_r16_mean 0 \
  --scale bridge_1k --seq_len 2048 --recipe far_copy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 16 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# no --message_pool_remainder
# floor at 800: do not extra-step
```

Byobu `E25_2k_ip_mean`. Log:
`seq=2048  gap=64  window=16  logit_scale=log  msg_r=16  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `hidden: 256`,
`global_logit_scale: log`, `message_ratio: 16`, `message_pool_remainder: false`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`, `batch: 32`,
`seq_len: 2048`, `scale: bridge_1k`. Recalibrated dense S0 in the same JSON.
**No new scale enum** — probe `--seq_len` override plus a unit test that 2048
keeps `local_window < min_gap`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **100%** | 250 | **63.79** | 0.997 |
| e18 | **23.9%** | 800 | **0.00** | 0.000 |
| **e21 @800** | **25.0%** | 800 | **0.00** | **0.000** |
| e18_local | 26.0% | 800 | 0.01 | 0.000 |

Dense left chance at ~150 (23.9%), **37.8% @200**, early-stop **100% @250**.
E18 chance every eval (CE at ln(4); 0 bits; best 26.0%). E21 chance every eval
(CE at ln(4); **0 bits**; best 26.0%). Not climbing.

E18 is **about 0**. Do **not** pass S1 via 0.75×0 (0 bits). Score vs
**0.75× dense = 47.84 bits**. E21 has **0**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.79 bits @250. |
| **S1 vs 0.75× E18** | **not the bar.** E18 ~0 bits; 0.75×0 is not a pass. |
| **S1 vs 0.75× dense** | **FAIL.** 0 ≪ 47.84 bits. Chance floor. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 26.0% / 0.01 bits. |
| **K3** | floor at 800 (flow 0). |

Do **not** extra-step to 8k (floor). Do not 16k. Do not remainder-on. Do
not extra width (dense already S0 at H=256). Do not hops. Do not Glyph.
Do not unfreeze `u`/`delta`.

## Interpretation

**Exclusive compressed INDEX that passed at 1024 is chance at 2048** on the
same H=256 SSMax-log identity-mean compressor **and the same gap=64 / window=16**.
Dense still copies 64 bits in 250 steps, so the instrument is live (not K1). E18
is also 0 bits at 800 — this is the one-global-read length wall, not an E21-only
pooling failure. Do **not** relabel E18's 0 bits as an E21 pass via 0.75×0.

The INDEX length wall for this recipe sits in **(1024 PASS, 2048 FAIL]**. 4096
on `--scale medium` (gap 1024) was also chance; 2048 with the 1024 gap already
kills both E18 and E21.

## Decision

Keep the spec in `ahead/`. Next ONE: **stop INDEX length extra-steps** (do not
seq=1536; that is a seq A/B). Do not remainder-on. Do not 2048 H=512 (dense S0
already passed). Remaining measured DNA walls: 1024 SELECT dead, 512 chain K1,
256 hops FAIL. Not Glyph. Do not unfreeze `u`/`delta`.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
