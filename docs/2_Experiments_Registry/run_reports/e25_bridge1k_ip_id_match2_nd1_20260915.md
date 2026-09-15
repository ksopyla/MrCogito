# E25 bridge_1k seq=1024 `--recipe recall` MATCH2 n_dist=1 H=128 identity — E21 S1 PASS vs 0.75× dense (rung 5bu)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1024_ip_id_match2_nd1` (800 advertised; dense-matched 3200; e18 / e21 / `e18_local` scored; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_match2_nd1/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_match2_nd1/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `40af063` (Odra E24 worktree; E22 main checkout untouched; hunt code same as `3759020`)
**Git tag:** —
**Related:** 512 MATCH2 n_dist=1 S1 PASS [`e25_bridge512_ip_id_match2_nd1_20260915.md`](e25_bridge512_ip_id_match2_nd1_20260915.md) · 512 MATCH2 n_dist=2 dense K1 [`e25_bridge512_ip_id_match2_20260915.md`](e25_bridge512_ip_id_match2_20260915.md)

---

## Goal

Grow 2-item MATCH2 length at the 512-passing H=128 identity recipe.
`--scale bridge_1k --seq_len 1024 --recipe recall --n_distractors 1`.
Do **not** switch to H=256. Do **not** run 3-item MATCH2 (`n_distractors 2`).
Do not relabel `recall_single` 43.08 or 512 MATCH2 20.78 as this score.

`--n_distractors 1` is a valid packed MATCH2 at 1024: `pack_overrides`
keeps `value_len=32` / 64-bit prize; seed-0 row `meta.n_items=2`.
`bridge_1k` + `recipe recall` is valid (no fallback to `--scale bridge
--seq_len 1024`). Native 1024 packing is 32 tokens / 64 bits (512 was 24
/ 48). Dense S0 first. If dense misses 75% (K1), skip E18/E21. If E18 ≈
0, score vs 0.75× dense (do not pass via 0.75×0). 800-step floor: chance
→ no 8k; climbing short of S1 → extra-step 8k only then. Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · **0.595M** (e21 **0.603M**) · kv=1 · **`logit_scale=none`** · stack=2 · glob=1 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`** |
| Data | `--scale bridge_1k --seq_len 1024 --recipe recall --n_distractors 1`, packed answer 32 / **64-bit** prize, **2 items** |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/65/100 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.5–1.9GB, 0.05–0.06 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1024_ip_id_match2_nd1 0 \
  --scale bridge_1k --seq_len 1024 --recipe recall --n_distractors 1 \
  --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# named recipe recall + --n_distractors 1 = 2-item MATCH2 (not recall_single)
# no --global_logit_scale (none; same width as 512 MATCH2 n_dist=1)
# no --global_layers 2
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused)
# S1 PASS vs 0.75× dense at dense-matched 3200: do not extra-step; do not 16k
```

Byobu `E25_1024_match2_nd1`. Log:
`seq=1024  gap=64  window=16  prize=64.00 bits  answer_len=32
row_gap[min/med/max]=65/65/100`
dense `patterns=[('full', 0)×4]` · 0.595M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_boundary=10  msg_r=1  msg_remainder=False  msg_inplace=True
msg_rawkv=False  msg_idslots=True  msg_keepswa=False  glob_layers=1
msg_extrahops=0  msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 1024`, `min_gap: 64`, `local_window: 16`, `hidden: 128`,
`global_logit_scale: none`, `global_layers: 1`, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`n_distractors: 1`, `prize_bits: 64.0`, `calibrated: true`,
`dense_steps_used: 3200`. Hunt exit **0**. Params <100M. Dense S0
recalibrated at H=128 (0.595M). Code defaults unchanged.

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense @800 | **35.0%** | 800 | **3.32** | — | — |
| dense | **93.5%** | 3200 | **58.81** | **0.919** | **0.007179** |
| dense best acc | **94.0%** | 2400 | **58.76** | — | — |
| e18 @800 | **25.1%** | 800 | **0.01** | — | — |
| e18 | **25.3%** | 3200 | **0.00** | **0.000** | **0** |
| e21 @800 | **58.8%** | 800 | **24.86** | — | — |
| **e21** | **97.8%** | **3200** | **61.81** | **0.966** | **0.007545** |
| e21 best acc | **97.9%** | 3200 | (final metric is @3200) | — | — |
| e18_local | **25.3%** | 3200 | **0.00** | **0.000** | **0** |

Dense left chance ~750, **35.0% @800**, crossed 75% at 1750, plateaued
**93–94% / ~58.8 bits** through 3200 (no 99% early-stop; **S0 PASS**).
E18 stayed at chance every eval through 3200 (**0.00 bits**, ~0). E21
left chance ~750, **58.8% / 24.86 bits @800** (climbing), **92.5% /
57.17 @1600**, final **97.8% / 61.81 bits** @3200 (flow 0.966).
`e18_local` at chance (**K2 PASS**). JSON `calibrated: true`. Hunt exit 0.

S1 vs 0.75× live E18: E18 ≈ 0, so **do not pass via 0.75×0**. Score vs
0.75× dense: need **44.11 bits**. E21 has **61.81**. Content vs dense:
E21 **61.81 > 58.81**.

Do **not** relabel 512 MATCH2 n_dist=1 20.78 bits as this 1024 score.
Do not relabel `recall_single` 43.08 as MATCH2. Do not relabel E18 0 as
E21 — E21 recovered **61.81**.

**8k not run** (S1 already PASS vs 0.75× dense at dense-matched 3200;
E18 chance floor so no 8k; E21 left chance at 800 but cleared S1 by
3200). Do not 16k.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 93.5% / 58.81 bits @3200. |
| **S1 vs 0.75× live E18** | **not used.** E18 ≈ 0; do not pass via 0.75×0. |
| **S1 vs 0.75× dense** | **PASS.** 61.81 ≥ 44.11 bits. |
| **content vs dense** | E21 61.81 > dense 58.81. |
| **S2** plots | **PASS** (curves + bits/flow/bytes-per-token). |
| **K1** | **not triggered.** |
| **K2** | **PASS.** `e18_local` 25.3% / 0.00 bits. |
| **K3** | **not triggered** (flow 0.97). |

## Interpretation

**Packed 1024 2-item MATCH2 (`recall`, 1 distractor) is dense-solvable at
the identity MATCH-passing H=128 recipe, and exclusive identity recovers
the 64-bit prize while the raw E18 global read stays at chance.** Dense
solves 58.81 of 64 prize bits. E18 is **0 bits** at this length and
width (unlike 512, where E18 plateaued at 26.66). E21 recovers
**61.81 bits** (97.8%), clearing 0.75× dense and beating dense's own
score. Item-count wall at H=128 identity stays **(2-item S1 PASS, 3-item
dense K1]**; 2-item length at this recipe is live at 1024. Do not treat
E18's 0-bit floor as an E21 kill — the exclusive identity channel is
the live one here.

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not H=256.
Do not MATCH2 n_dist=2. Do not hops 320. Do not MATCH 1408/2048. Do not
SELECT 694. Do not INDEX extra-steps. Do not Glyph. Do not unfreeze
`u`/`delta`. Do not restore full raw prefix KV (that is E18). Do not
stack keepswa / extra hop / glob=2 on this first 1024 MATCH2 grow. Code
defaults unchanged.
Next ONE (do not run): **1280 MATCH2 `--n_distractors 1` H=128 identity**
(grow 2-item MATCH2 length at the 1024-passing recipe).

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
