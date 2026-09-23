# E25 bridge_1k seq=1280 `recall_single` r=8 rem-off H=256 log `--message_extra_slot_attends 1` — E21 S1 FAIL vs 0.75× live E18 (rung 5ce)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1280_ip_r8_h256_recall_extrahop` (800 advertised; dense-matched 3200; e18 early-stop 1200; e21 / `e18_local` scored to 3200; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_1280_ip_r8_h256_recall_extrahop/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_1280_ip_r8_h256_recall_extrahop/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `6bd9f1b` (Odra E24 worktree HEAD at launch; `nn/` unchanged vs hunt SHA `c5c13a2`)
**Git tag:** —
**Related:** 1280 MATCH r=8 rem-off severed S1 FAIL [`e25_bridge1k_1280_ip_r8_h256_recall_20260914.md`](e25_bridge1k_1280_ip_r8_h256_recall_20260914.md) · 1280 MATCH r=1 identity S1 PASS [`e25_bridge1k_1280_ip_id_h256_recall_20260914.md`](e25_bridge1k_1280_ip_id_h256_recall_20260914.md) · 1024 MATCH r=8 H=256 log S1 PASS [`e25_bridge1k_ip_r8_h256_recall_20260914.md`](e25_bridge1k_ip_r8_h256_recall_20260914.md) · 1280 MATCH2 extra hop S1 FAIL [`e25_bridge1k_1280_ip_id_match2_nd1_extrahop_20260915.md`](e25_bridge1k_1280_ip_id_match2_nd1_extrahop_20260915.md) · 256 shuffled n_dist=2 extra hop S1 PASS [`e25_bridge256_chain_shuf_k13_glob2_extrahop_20260915.md`](e25_bridge256_chain_shuf_k13_glob2_extrahop_20260915.md)

---

## Goal

Architectural bet at the packed 2-key MATCH pooling wall (`--recipe
recall_single` seq=1280 r=8 rem-off H=256 log), where severed E21 was
**0 bits** and uncompressed E18 was **live 63.95**. Extra hop failed
MATCH2 1280 identity (0 bits; E18 also chance) and **passed** shuffled
n_dist=2 (22.60). Same r=8 compressor as the exclusive FAIL cell; **one**
knob: `--message_extra_slot_attends 1`. Keepswa **off**. glob=**1**. Do
**not** `--n_distractors` (this is not MATCH2). Falsifiable claim: extra
exclusive hop composes 8-token frozen means at 1280 when E18 is live
(pooling wall), unlike MATCH2 1280 identity.

Do **not** relabel MATCH2 extra hop 0, MATCH identity 63.94, MATCH r=8
0, shuffled extra hop 22.60, or 1024 r=8 60.35 as this score. Score vs
0.75× **live E18**; if E18 ≈ 0, score vs 0.75× dense (do not pass via
0.75×0). Chance at 800 → no 8k. Climbing short of S1 → extra-step 8k
only then. S1 PASS → no 8k. Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / `e18_local` |
| Width | **H=256** · **2.261M** (e21 **2.277M**) · kv=1 · **`logit_scale=log`** · stack=2 · glob=1 |
| E21 | query boundary id 10, **r=8**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=1`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=1`** |
| Data | `--scale bridge_1k --seq_len 1280 --recipe recall_single`, packed answer 32 / **64-bit** prize, **2-key MATCH** (`n_distractors=0`) |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/65/65 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~3.2–4.0GB, 0.09–0.12 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1280_ip_r8_h256_recall_extrahop 0 \
  --scale bridge_1k --seq_len 1280 --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 8 --message_slots_inplace --message_identity_slots \
  --message_extra_slot_attends 1 --steps 800 --k1_mult 4
# named recipe recall_single = 2-key MATCH (n_distractors=0; not MATCH2)
# --message_extra_slot_attends 1 required (this bet); exclusive slots stay on the global read
# no --n_distractors
# no --global_layers (default 1)
# no --message_pool_remainder
# no --message_keep_local_swa (default false; SWA still severed; do not stack)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused)
# E21 chance floor at 800: do not extra-step; do not 16k
```

Byobu `E25_1280_r8_h256_recall_extrahop`. Log:
`seq=1280  gap=64  window=16  prize=64.00 bits  answer_len=32
row_gap[min/med/max]=65/65/65`
dense `patterns=[('full', 0)×4]` · 2.261M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_boundary=10  msg_r=8  msg_remainder=False  msg_inplace=True
msg_rawkv=False  msg_idslots=True  msg_keepswa=False  glob_layers=1
msg_extrahops=1  msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 1280`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 1`**, `message_ratio: 8`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
**`message_keep_local_swa: false`**, **`message_extra_slot_attends: 1`**,
`message_update_slot_kv: false`, `message_global_anchors: none`,
**`n_distractors: 0`**, `prize_bits: 64.0`, `calibrated: true`,
`dense_steps_used: 3200`. Hunt exit **0**. Params <100M. Dense S0
recalibrated at H=256 (2.261M). Code defaults unchanged
(`--message_extra_slot_attends` stays 0; `--message_keep_local_swa`
stays false; `--global_layers` stays 1). Packed 2-key MATCH
(`recall_single`).

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense @800 | **87.2%** | 800 | **55.78** | — | — |
| dense | **93.7%** | 3200 | **58.94** | **0.921** | **0.005756** |
| dense best acc | **93.7%** | 3200 | **58.94** | — | — |
| e18 @800 | **75.4%** | 800 | **47.83** | — | — |
| e18 | **100%** | 1200 | **63.99** | **1.000** | **0.006249** |
| e18 best acc | **100%** | 1200 | **63.99** | — | — |
| **e21 @800** | **22.8%** | **800** | **0.00** | — | — |
| **e21** | **25.0%** | **3200** | **0.01** | **0.000** | **0** |
| e21 best acc | **26.5%** | 200 | (chance; CE at ln(4)) | — | — |
| e18_local | **25.0%** | 3200 | **0.00** | **0** | **0** |

Dense left chance through 250, **78.0% @350** (S0 crossed), plateau
~87% from 400–3150, late climb **93.7% / 58.94 bits @3200** (**S0
PASS**; no 99% early-stop). JSON `calibrated: true`. Hunt exit 0.

This-JSON E18 crossed 75% at **400** (75.9%), plateaued ~75% through
800 (75.4% / 47.83 bits), then clicked **97.5% @1150** and early-stop
**100% / 63.99 bits** @1200 (**live — not ~0**). S1 bar is **0.75× E18
= 47.99 bits** (also vs **0.75× dense = 44.20 bits**).

E21 stayed at chance every eval through 3200 (**0.00 bits @800**, CE
1.3888 > ln(4); **0.01 bits** @3200; best acc **26.5%** @200 is chance
noise, not a climb; **S1 FAIL** vs 0.75× live E18 47.99). `e18_local`
at chance (**K2 PASS**).

Do **not** relabel MATCH2 extra hop **0**, MATCH identity **63.94**,
MATCH r=8 severed **0**, shuffled n_dist=2 extra hop **22.60**, or 1024
r=8 **60.35** as this score. Do not relabel this-JSON E18 **63.99** as
an E21 pass.

**8k not run** (E21 chance floor at 800; do not extra-step a chance
floor). Do not 16k.

One plot: learning curves — dense and live E18 solve; extra-hop E21
and `e18_local` stay on the chance line.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** (dense ≥ 75%) | **PASS.** Dense 93.7% / 58.94 bits @3200 (crossed 75% at 350). Recalibrated at H=256. |
| **S1 vs 0.75× live E18** | **FAIL.** 0.01 ≪ 47.99 bits. Chance floor. Do not pass via 0.75×0 — E18 is live. |
| **content vs 0.75× dense** | **FAIL.** 0.01 ≪ 44.20 bits. |
| **S2** plots | **PASS** (curves + bits/flow/bytes-per-token). |
| **K1** | not triggered. |
| **K2** | **PASS.** `e18_local` 25.0% / 0.00 bits @3200. |
| **K3** | **triggered.** E21 chance floor (flow 0.000). |
| **8k** | **not run.** Chance at 800. Do not 16k. |

## Interpretation

**An extra exclusive hop over frozen 8-token means does not rescue
packed 1280 MATCH.** Same compressor, one global Block, identity-mean
slots, 64-bit prize, keepswa **off**, glob=1: severed E21 was **0 @800**
vs live E18 **63.95** / dense **63.85**; extra-hop E21 is **0 @800 /
0.01 @3200** vs this-JSON live E18 **63.99** and vs 0.75× E18 **47.99**.
Extra hop at shuffled n_dist=2 **S1 PASS 22.60** does not transfer to
MATCH pooling at 1280. MATCH2 extra hop at 1280 identity was also 0,
but that cell had dead E18; here E18 is live and the exclusive r=8
channel still dies.

SWA stays severed (`msg_keepswa=False`). Exclusive slots stay on the
global read (`msg_rawkv=False`). Extra hop is in-attention over frozen
slot K/V (`msg_extrahops=1`, `msg_updatekv=False`). Window 16 still
cannot span gap 65. `e18_local` stays at chance (K2) — extra hop did not
leak a solvable local-only channel.

Exclusive MATCH length wall stays **(1024 S1 PASS, 1280 FAIL]** even
with one extra exclusive hop on the r=8 pooler. Distinct from 2-key
MATCH identity at 1280 (H=256 log S1 PASS 63.94). Pooling-ratio wall at
1280 **(r=1 PASS, r=4 FAIL]** still stands; extra hop does not move r=8.

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not r=2/r=6.
Do not r=8 rem-on. Do not keepswa. Do not glob=2. Do not MATCH 1408. Do
not MATCH2 1280 extra hop again. Do not hops 320. Do not SELECT 694. Do
not INDEX 1280. Do not shuffled n_dist=3. Do not stack keepswa+extra hop.
Not Glyph. Do not unfreeze `u`/`delta`. Do not restore full raw prefix KV
(that is E18). Code default `--message_extra_slot_attends` stays **0**.
Code default `--message_keep_local_swa` stays **false**. Code default
`--global_layers` stays **1**. Next ONE (do not run): **STOP extra-hop
transfer to MATCH pooling** (extra hop does not rescue r=8 at 1280).

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
