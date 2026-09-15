# E25 bridge_1k seq=1280 `--recipe recall` MATCH2 n_dist=1 H=128 identity — E21 S1 FAIL vs 0.75× dense (rung 5bv)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1280_ip_id_match2_nd1` (800 advertised; dense-matched 3200; e18 / e21 / `e18_local` scored; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_1280_ip_id_match2_nd1/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_1280_ip_id_match2_nd1/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `99fa858` (Odra E24 worktree; E22 main checkout untouched; hunt code same as `40af063`)
**Git tag:** —
**Related:** 1024 MATCH2 n_dist=1 S1 PASS [`e25_bridge1k_ip_id_match2_nd1_20260915.md`](e25_bridge1k_ip_id_match2_nd1_20260915.md) · 512 MATCH2 n_dist=1 S1 PASS [`e25_bridge512_ip_id_match2_nd1_20260915.md`](e25_bridge512_ip_id_match2_nd1_20260915.md)

---

## Goal

Grow 2-item MATCH2 length at the 1024-passing H=128 identity recipe.
`--scale bridge_1k --seq_len 1280 --recipe recall --n_distractors 1`.
Do **not** switch to H=256. Do **not** run 3-item MATCH2 (`n_distractors 2`).
Do not confuse this with 2-key MATCH at 1280 (`recall_single` r=1 identity
H=256 log S1 PASS 63.94). Do not relabel 1024 MATCH2 61.81 as this score.

`--n_distractors 1` is a valid packed MATCH2 at 1280: `pack_overrides`
keeps `value_len=32` / 64-bit prize; seed-0 row `meta.n_items=2`;
`--evidence_align right` row_gap 65/100/100. Dense S0 first. If dense
misses 75% (K1), skip E18/E21. If E18 ≈ 0, score vs 0.75× dense (do not
pass via 0.75×0). 800-step floor: chance → no 8k; climbing short of S1 →
extra-step 8k only then. Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · **0.595M** (e21 **0.603M**) · kv=1 · **`logit_scale=none`** · stack=2 · glob=1 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`** |
| Data | `--scale bridge_1k --seq_len 1280 --recipe recall --n_distractors 1`, packed answer 32 / **64-bit** prize, **2 items** |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/100/100 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.7–2.4GB, 0.07–0.08 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1280_ip_id_match2_nd1 0 \
  --scale bridge_1k --seq_len 1280 --recipe recall --n_distractors 1 \
  --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# named recipe recall + --n_distractors 1 = 2-item MATCH2 (not recall_single)
# no --global_logit_scale (none; same width as 1024 MATCH2 n_dist=1)
# no --global_layers 2
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused)
# E21 chance floor at dense-matched 3200: do not extra-step; do not 16k
```

Byobu `E25_1280_match2_nd1`. Log:
`seq=1280  gap=64  window=16  prize=64.00 bits  answer_len=32
row_gap[min/med/max]=65/100/100`
dense `patterns=[('full', 0)×4]` · 0.595M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_boundary=10  msg_r=1  msg_remainder=False  msg_inplace=True
msg_rawkv=False  msg_idslots=True  msg_keepswa=False  glob_layers=1
msg_extrahops=0  msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 1280`, `min_gap: 64`, `local_window: 16`, `hidden: 128`,
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
| dense @800 | **78.9%** | 800 | **43.38** | — | — |
| dense | **98.9%** | 3200 | **62.94** | **0.983** | **0.006147** |
| dense best acc | **98.9%** | 2350 | **63.01** | — | — |
| e18 @800 | **25.0%** | 800 | **0.00** | — | — |
| e18 | **25.0%** | 3200 | **0.00** | **0.000** | **0** |
| e21 @800 | **25.0%** | 800 | **0.00** | — | — |
| **e21** | **26.1%** | **3200** | **0.01** | **0.000** | **0** |
| e21 best acc | **26.1%** | 250 | (chance; CE at ln(4)) | — | — |
| e18_local | **24.3%** | 3200 | **0.00** | **0.000** | **0** |

Dense left chance ~400, **78.9% @800** (crossed 75% at 800), plateaued
**98–99% / ~62.9 bits** through 3200 (no 99% early-stop; **S0 PASS**).
E18 stayed at chance every eval through 3200 (**0.00 bits**, ~0). E21
stayed at chance every eval through 3200 (**0.01 bits**, CE at ln(4);
**26.1%** is chance noise, not a climb). `e18_local` at chance
(**K2 PASS**). JSON `calibrated: true`. Hunt exit 0.

S1 vs 0.75× live E18: E18 ≈ 0, so **do not pass via 0.75×0**. Score vs
0.75× dense: need **47.21 bits**. E21 has **0.01**. Content vs dense:
E21 **0.01 << dense 62.94**.

Do **not** relabel 1024 MATCH2 n_dist=1 61.81 bits as this 1280 score.
Do not relabel 2-key MATCH identity at 1280 (H=256 log 63.94) as MATCH2.
Do not relabel E18 0 as an E21 pass.

**8k not run** (E21 chance floor at 800 and at dense-matched 3200; do
not extra-step a chance floor). Do not 16k.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 98.9% / 62.94 bits @3200. |
| **S1 vs 0.75× live E18** | **not used.** E18 ≈ 0; do not pass via 0.75×0. |
| **S1 vs 0.75× dense** | **FAIL.** 0.01 < 47.21 bits. |
| **content vs dense** | E21 0.01 << dense 62.94. |
| **S2** plots | **PASS** (curves + bits/flow/bytes-per-token). |
| **K1** | **not triggered.** |
| **K2** | **PASS.** `e18_local` 24.3% / 0.00 bits. |
| **K3** | **triggered.** E21 chance floor (flow 0.000). |
| **8k** | **not run** (chance floor). |

## Interpretation

**Packed 1280 2-item MATCH2 (`recall`, 1 distractor) is dense-solvable at
the identity MATCH2-passing H=128 recipe, but exclusive identity does not
recover the 64-bit prize.** Dense solves 62.94 of 64 prize bits. E18
is **0 bits** at this length and width (same as 1024 MATCH2, unlike
512 where E18 plateaued at 26.66). E21 is **0 bits** here — the
opposite of 1024 MATCH2, where exclusive identity recovered **61.81
bits** and beat dense. 2-item MATCH2 length wall at H=128 identity is
**(1024 S1 PASS, 1280 FAIL]**. Item-count wall at 512 stays **(2-item S1
PASS, 3-item dense K1]**. Do not treat 2-key MATCH identity at 1280
(H=256 log S1 PASS) as this 2-item MATCH2 result.

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not H=256
on this first 1280 MATCH2 grow (already mapped as FAIL at H=128). Do
not MATCH2 n_dist=2. Do not hops 320. Do not MATCH 1408/2048. Do not
SELECT 694. Do not INDEX extra-steps. Do not Glyph. Do not unfreeze
`u`/`delta`. Do not restore full raw prefix KV (that is E18). Do not
stack keepswa / extra hop / glob=2 on this first 1280 MATCH2 grow. Code
defaults unchanged.
Next ONE (do not run): **1152 MATCH2 `--n_distractors 1` H=128 identity**
(pin 2-item MATCH2 length wall between 1024 S1 PASS and 1280 FAIL).

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
