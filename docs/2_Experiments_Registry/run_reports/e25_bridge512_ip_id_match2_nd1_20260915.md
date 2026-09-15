# E25 bridge seq=512 `--recipe recall` MATCH2 n_dist=1 H=128 identity — E21 S1 PASS vs live E18 (rung 5bt)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_id_match2_nd1` (800 advertised; dense-matched 1650; e18 / e21 / `e18_local` scored; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_id_match2_nd1/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_id_match2_nd1/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `3759020` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 512 MATCH2 n_dist=2 dense K1 [`e25_bridge512_ip_id_match2_20260915.md`](e25_bridge512_ip_id_match2_20260915.md) · 512 `recall_single` identity S1 PASS [`e25_bridge512_ip_id_recall_20260914.md`](e25_bridge512_ip_id_recall_20260914.md)

---

## Goal

Calibrated shrink after 512 packed MATCH2 `--n_distractors 2` dense **K1**
(55.8% / 18.44 of 48 bits @3200; E18/E21 skipped). Same H=128 identity
recipe as 512 `recall_single` **S1 PASS 43.08 bits**, with generator
`--n_distractors 1` so MATCH2 is **2 items** (`n_items = n_distractors+1`).
Do **not** relabel `recall_single` (1-item) 43.08 bits as MATCH2. Do not
relabel n_dist=2 (3-item) as this score.

`--n_distractors 1` is a valid packed MATCH2: `pack_overrides` never drops
recall distractors below 1; packed `value_len=24` / 48-bit prize still
fits at `--scale bridge`; seed-0 row `meta.n_items=2`. Dense S0 first. If
dense misses 75% (K1), skip E18/E21. If E18 ≈ 0, score vs 0.75× dense (do
not pass via 0.75×0). 800-step floor: chance → no 8k; climbing short of
S1 → extra-step 8k only then. Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · **0.595M** (e21 **0.603M**) · kv=1 · **`logit_scale=none`** · stack=2 · glob=1 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`** |
| Data | `--scale bridge --recipe recall --n_distractors 1`, packed answer 24 / **48-bit** prize, **2 items** |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/65/92 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~0.9GB, 0.04 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_id_match2_nd1 0 \
  --scale bridge --recipe recall --n_distractors 1 --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# named recipe recall + --n_distractors 1 = 2-item MATCH2 (not recall_single)
# no --global_logit_scale (none; same width as 512 recall_single identity)
# no --global_layers 2
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused)
# S1 already PASS vs live E18: do not extra-step; do not 16k
```

Byobu `E25_512_match2_nd1`. Log:
`seq=512  gap=64  window=16  prize=48.00 bits  answer_len=24
row_gap[min/med/max]=65/65/92`
dense `patterns=[('full', 0)×4]` · 0.595M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_boundary=10  msg_r=1  msg_remainder=False  msg_inplace=True
msg_rawkv=False  msg_idslots=True  msg_keepswa=False  glob_layers=1
msg_extrahops=0  msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 512`, `min_gap: 64`, `local_window: 16`, `hidden: 128`,
`global_logit_scale: none`, `global_layers: 1`, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`n_distractors: 1`, `prize_bits: 48.0`, `calibrated: true`,
`dense_steps_used: 1650`. Hunt exit **0**. Params <100M. Dense S0
recalibrated at H=128 (0.595M). Code defaults unchanged.

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense @800 | **66.5%** | 800 | (climbing) | — | — |
| dense | **99.3%** | 1650 | **47.03** | **0.980** | **0.01148** |
| e18 @800 | **66.7%** | 800 | (plateau) | — | — |
| e18 | **68.3%** | 1650 | **26.66** | **0.555** | **0.006508** |
| e21 @800 | **35.2%** | 800 | (left chance) | — | — |
| **e21** | **55.9%** | **1650** | **20.78** | **0.433** | **0.005073** |
| e21 best acc | **58.4%** | 1300 | (final metric is @1650) | — | — |
| e18_local | **26.2%** | 1650 | **0.02** | **0.000** | **4.6e-6** |

Dense left chance ~450, sat ~66% through 1500, then clicked **99.3% /
47.03 bits** @1650 (early-stop; S0 PASS). E18 clicked ~400 to ~67% and
**plateaued 66–68%** through 1650 (**26.66 bits**, live, not ~0). E21
chance through ~750, **35.2% @800** (left chance; CE 1.360 vs floor 1.386),
then **58.4% @1300** and final **55.9% / 20.78 bits** @1650 (flow 0.433).
`e18_local` at chance (**K2 PASS**). JSON `calibrated: true`. Hunt exit 0.

S1 vs 0.75× live E18: need **19.99 bits**. E21 has **20.78**.
Content vs 0.75× dense: need **35.27 bits**. E21 has **20.78**.

Do **not** relabel 512 `recall_single` 43.08 bits as this MATCH2 score.
Do not relabel n_dist=2 18.44 dense bits as E21. Do not relabel E18
26.66 as E21 — E21 recovered **20.78**.

**8k not run** (S1 already PASS vs live E18 at dense-matched 1650; E21
left chance at 800 so this is not a floor skip, and it is not climbing
short of S1). Do not 16k. E18/E21 both plateaued well short of dense's
late click; extra-step is not the registered gate here.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.3% / 47.03 bits @1650. |
| **S1 vs 0.75× live E18** | **PASS.** 20.78 ≥ 19.99 bits (thin; E18 plateau 68% / 26.66 of 48). |
| **content vs 0.75× dense** | **FAIL.** 20.78 < 35.27 bits. |
| **S2** plots | **PASS** (curves + bits/flow/bytes-per-token). |
| **K1** | **not triggered.** |
| **K2** | **PASS.** `e18_local` 26.2% / 0.02 bits. |
| **K3** | **not triggered** (flow 0.43). |

## Interpretation

**Packed 512 2-item MATCH2 (`recall`, 1 distractor) is dense-solvable at
the identity MATCH-passing H=128 recipe, and exclusive identity carries
a live but incomplete contrast-set channel.** Dense solves 47 of 48 prize
bits. E18 binds the 2-item set only to **26.66 bits** (plateau ~68%). E21
recovers **20.78 bits** (55.9%), clearing 0.75× that live E18 bar and
missing 0.75× dense. Three-item MATCH2 at this width stays dense **K1**.
One-item `recall_single` stays E21 **43.08**. Item-count wall at H=128
identity is **(2-item S1 PASS, 3-item dense K1]**.

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not H=256
on 3-item MATCH2 (1-distractor S0 passed). Do not MATCH2 n_dist=2 again.
Do not hops 320. Do not MATCH 1408/2048. Do not SELECT 694. Do not INDEX
extra-steps. Do not Glyph. Do not unfreeze `u`/`delta`. Do not restore
full raw prefix KV (that is E18). Code defaults unchanged.
Next ONE (do not run): **1024 MATCH2 `--n_distractors 1` H=128 identity**
(grow 2-item MATCH2 length at the 512-passing recipe).

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
