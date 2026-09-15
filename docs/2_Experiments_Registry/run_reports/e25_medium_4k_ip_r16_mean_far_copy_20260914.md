# E25 medium 4k far_copy r=16 remainder-off H=256 SSMax log — E21 vs E18 vs dense (rung 5af)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_4k_ip_r16_mean` (800; floor — 8k JSON is this hunt)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_medium_4k_ip_r16_mean/`
**Raw log:** `/opt/cursor/artifacts/e25_medium_4k_ip_r16_mean_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `8bd3dc4` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 1024 INDEX H=256 log PASS [`e25_bridge1k_ip_r16_mean_far_copy_20260914.md`](e25_bridge1k_ip_r16_mean_far_copy_20260914.md) · 1024 SELECT identity chance [`e25_bridge1k_ip_id_h256_select_20260914.md`](e25_bridge1k_ip_id_h256_select_20260914.md)

---

## Goal

Locked: 1024 packed INDEX (`far_copy`) inplace r=16 frozen mean **S1 PASS
(53.82 bits @8k)** on H=256 SSMax log. Exclusive SELECT is dead at 1024.
This rung is the next harder **INDEX scale**: seq=4096 packed `far_copy` on
the existing `--scale medium` (no new scale enum), same compressor
(`--message_ratio 16 --message_slots_inplace --message_identity_slots`,
remainder **off**).

E24's 4k INDEX dense recipe was H=512 MHA / `stack_layers=4` / batch 8 /
right-align / `--warm_residuals`. Start here at **H=256 SSMax log** (the
1024-passing width). Jump to H=512 only if dense misses S0. Dense passed
S0 at H=256, so width stayed 256.

E18 is expected ~0 at 4k (E24). Do **not** pass S1 via 0.75×0; score vs
0.75× dense. Recalibrate dense in this JSON. Climbing at 800 → extra-step
8k; **floor → do not extra-step**. Do **not** pass `--message_pool_remainder`.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=16**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale medium` seq=4096, gap=1024, window=256, packed span 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 1025/1025/1025) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · **batch 8** (E24 4k batch; no OOM, ~4.4GB) |

```
bash scripts/e24_bapo_hunt.sh e25_4k_ip_r16_mean 0 \
  --scale medium --recipe far_copy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 16 --message_slots_inplace --message_identity_slots \
  --batch 8 --eval_rows 32 --steps 800 --k1_mult 4
# no --message_pool_remainder
# floor at 800: do not extra-step
```

Byobu `E25_4k_ip_mean`. Log:
`seq=4096  logit_scale=log  msg_r=16  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `hidden: 256`,
`global_logit_scale: log`, `message_ratio: 16`, `message_pool_remainder: false`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`, `batch: 8`,
`seq_len: 4096`. Recalibrated dense S0 in the same JSON. **No 4k scale code
change** — `medium` already exists.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **100%** | 400 | **63.94** | 0.999 |
| e18 | **25.7%** | 800 | **0.00** | 0.000 |
| **e21 @800** | **25.7%** | 800 | **0.00** | **0.000** |
| e18_local | 25.7% | 800 | 0.01 | 0.000 |

Dense left chance at ~300 (29.6%), **76.1% @350**, early-stop **100% @400**.
E18 chance every eval (CE at ln(4); 0.004 bits). E21 chance every eval (CE
at ln(4); **0 bits**). Best E21 acc 25.8% = chance. Not climbing.

E18 is **about 0**. Do **not** pass S1 via 0.75×0 (0.003 bits). Score vs
**0.75× dense = 47.96 bits**. E21 has **0**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.94 bits @400. H=512 unused. |
| **S1 vs 0.75× E18** | **not the bar.** E18 ~0 bits; 0.75×0 is not a pass. |
| **S1 vs 0.75× dense** | **FAIL.** 0 ≪ 47.96 bits. Chance floor. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.7% / 0.01 bits. |
| **K3** | floor at 800 (flow 0). |

Do **not** extra-step to 8k (floor). Do not 16k. Do not remainder-on. Do
not extra width (dense already S0 at H=256). Do not hops. Do not Glyph.
Do not unfreeze `u`/`delta`.

## Interpretation

**Exclusive compressed INDEX that passed at 1024 is chance at 4k** on the
same H=256 SSMax-log identity-mean compressor. Dense still copies 64 bits
in 400 steps, so the instrument is live (not K1). E18 is also 0 bits at
800 — this is not an E21-only wall. Do **not** relabel E18's 0 bits as an
E21 pass via 0.75×0.

Do **not** stack remainder-on or H=512 as the same rung. Dense S0 did not
need the E24 16M recipe.

## Decision

Keep the spec in `ahead/`. Next ONE: seq=**2048** packed `far_copy` with
the same r=16 identity inplace rem-off H=256 SSMax log compressor, to
localize the INDEX scale wall between 1024 PASS and 4k chance. Use
`--seq_len 2048` (probe override; no new fork). Dense-first. If E18 ~0,
score vs 0.75× dense. Do not remainder-on. Do not 4k H=512. Do not Glyph.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
