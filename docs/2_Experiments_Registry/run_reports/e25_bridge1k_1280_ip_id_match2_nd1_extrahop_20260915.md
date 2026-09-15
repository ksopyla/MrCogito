# E25 bridge_1k seq=1280 `--recipe recall` MATCH2 n_dist=1 H=128 identity `--message_extra_slot_attends 1` — E21 S1 FAIL vs 0.75× dense (rung 5cd)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1280_ip_id_match2_nd1_extrahop` (800 advertised; dense-matched 1150 early-stop; e18 / e21 / `e18_local` scored to 1150; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_1280_ip_id_match2_nd1_extrahop/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_1280_ip_id_match2_nd1_extrahop/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `c5c13a2` (Odra E24 worktree HEAD at launch; `nn/` unchanged)
**Git tag:** —
**Related:** 1280 MATCH2 n_dist=1 severed S1 FAIL [`e25_bridge1k_1280_ip_id_match2_nd1_20260915.md`](e25_bridge1k_1280_ip_id_match2_nd1_20260915.md) · 1024 MATCH2 n_dist=1 S1 PASS [`e25_bridge1k_ip_id_match2_nd1_20260915.md`](e25_bridge1k_ip_id_match2_nd1_20260915.md) · 256 shuffled n_dist=2 extra hop S1 PASS [`e25_bridge256_chain_shuf_k13_glob2_extrahop_20260915.md`](e25_bridge256_chain_shuf_k13_glob2_extrahop_20260915.md)

---

## Goal

Architectural bet at the packed 2-item MATCH2 length wall (`--recipe recall
--n_distractors 1` seq=1280), where severed E21 was **0.01 bits** and
uncompressed E18 was **0** vs dense **62.94**. Extra hop just rescued
shuffled n_dist=2 identity (E21 **22.60** vs E18 **0**). Same seq=1280
`--recipe recall --n_distractors 1` H=128 identity recipe; **one** knob:
`--message_extra_slot_attends 1`. Keepswa **off**. glob=**1**. Do **not**
stack keepswa + extra hop. Do **not** H=256. Do **not** MATCH2 n_dist=2.
Falsifiable claim: extra exclusive hop over frozen identity slots
composes 2-item MATCH2 at 1280 without restoring SWA.

Do **not** relabel severed 1280 MATCH2 E21 0.01, 1024 MATCH2 61.81, 2-key
MATCH identity at 1280 (H=256 log 63.94), shuffled extra hop 22.60, hops
272 extra hop 25.48, or hops 288 extra hop 0 as this score. If this-JSON
E18 ≈ 0, score vs 0.75× dense (do not pass via 0.75×0). Chance at 800 →
no 8k. Climbing short of S1 → extra-step 8k only then. S1 PASS → no 8k.
Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / `e18_local` |
| Width | **H=128** · **0.595M** (e21 **0.603M**) · kv=1 · **`logit_scale=none`** · stack=2 · glob=1 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=1`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=1`** |
| Data | `--scale bridge_1k --seq_len 1280 --recipe recall --n_distractors 1`, packed answer 32 / **64-bit** prize, **2 items** |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/100/100 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.7–2.5GB, 0.07–0.09 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1280_ip_id_match2_nd1_extrahop 0 \
  --scale bridge_1k --seq_len 1280 --recipe recall --n_distractors 1 \
  --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --message_extra_slot_attends 1 \
  --steps 800 --k1_mult 4
# named recipe recall + --n_distractors 1 = 2-item MATCH2 (not recall_single)
# --message_extra_slot_attends 1 required (this bet); exclusive slots stay on the global read
# no --global_logit_scale (none; same width as the failed 1280 MATCH2 cell)
# no --global_layers 2 (glob=1; hops-only exception unused)
# no --message_pool_remainder
# no --message_keep_local_swa (default false; SWA still severed; do not stack)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused)
# E21 chance floor at 800 and at dense-matched 1150: do not extra-step; do not 16k
```

Byobu `E25_1280_match2_extrahop`. Log:
`seq=1280  gap=64  window=16  prize=64.00 bits  answer_len=32
row_gap[min/med/max]=65/100/100`
dense `patterns=[('full', 0)×4]` · 0.595M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_boundary=10  msg_r=1  msg_remainder=False  msg_inplace=True
msg_rawkv=False  msg_idslots=True  msg_keepswa=False  glob_layers=1
msg_extrahops=1  msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 1280`, `min_gap: 64`, `local_window: 16`, `hidden: 128`,
`global_logit_scale: none`, **`global_layers: 1`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
**`message_keep_local_swa: false`**, **`message_extra_slot_attends: 1`**,
`message_update_slot_kv: false`, `message_global_anchors: none`,
**`n_distractors: 1`**, `prize_bits: 64.0`, `calibrated: true`,
`dense_steps_used: 1150`. Hunt exit **0**. Params <100M. Dense S0
recalibrated at H=128 (0.595M). Code defaults unchanged
(`--message_extra_slot_attends` stays 0; `--message_keep_local_swa`
stays false; `--global_layers` stays 1). Packed 2-item MATCH2
(`n_items = n_distractors+1 = 2`).

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense @800 | **75.1%** | 800 | **36.05** | — | — |
| dense | **99.5%** | 1150 | **63.20** | **0.987** | **0.006171** |
| dense best acc | **99.5%** | 1150 | **63.20** | — | — |
| e18 @800 | **25.0%** | 800 | **0.00** | — | — |
| e18 | **25.0%** | 1150 | **0.00** | **0.000** | **0** |
| e18 best acc | **27.1%** | 150 | — | — | — |
| **e21 @800** | **25.0%** | **800** | **0.00** | — | — |
| **e21** | **25.0%** | **1150** | **0.01** | **0.000** | **0** |
| e21 best acc | **26.7%** | 900 | (chance; CE at ln(4)) | — | — |
| e18_local | **26.1%** | 1150 | **0.01** | **0** | **0** |

Dense left chance ~400, **75.1% / 36.05 bits @800**, crossed 75% at
**800**, early-stop **99.5% / 63.20 bits** @1150 (**S0 PASS**). JSON
`calibrated: true`. Hunt exit 0.

This-JSON E18 stayed at chance every eval through 1150 (**0.00 bits**, CE
at ln(4); best acc 27.1% @150 is chance noise). Live **~0**. Do **not**
pass S1 via 0.75×0. Score vs **0.75× dense 47.40 bits**.

E21 stayed at chance every eval through 1150 (**0.00 bits @800**,
**0.01 bits** @1150; CE at ln(4); **26.7%** @900 is chance noise, not a
climb; **S1 FAIL** vs 0.75× dense 47.40). `e18_local` at chance
(**K2 PASS**).

Do **not** relabel severed 1280 MATCH2 E21 **0.01**, 1024 MATCH2
**61.81**, 2-key MATCH identity at 1280 **63.94**, shuffled n_dist=2
extra hop **22.60**, hops 272 extra hop **25.48**, or hops 288 extra hop
**0** as this score. Do not relabel this-JSON E18 0 as an E21 pass.

**8k not run** (E21 chance floor at 800 and at dense-matched 1150; do
not extra-step a chance floor). Do not 16k.

One plot: learning curves — dense solves; this-JSON E18, extra-hop E21,
and `e18_local` stay on the chance line.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** (dense ≥ 75%) | **PASS.** Dense 99.5% / 63.20 bits @1150. Recalibrated at H=128. |
| **S1 vs 0.75× live E18** | **not used.** E18 ≈ 0; do not pass via 0.75×0. |
| **S1 vs 0.75× dense** | **FAIL.** 0.01 < 47.40 bits. |
| **content vs dense** | E21 0.01 << dense 63.20. |
| **S2** plots | **PASS** (curves + bits/flow/bytes-per-token). |
| **K1** | not triggered. |
| **K2** | **PASS.** `e18_local` 26.1% / 0.01 bits @1150. |
| **K3** | **triggered.** E21 chance floor (flow 0.000). |
| **8k** | **not run.** Chance at 800. Do not 16k. |

## Interpretation

**An extra exclusive hop over frozen identity slots does not rescue
packed 1280 2-item MATCH2.** Same compressor, one global Block,
identity slots, 64-bit prize, n_dist=1 (`n_items=2`), keepswa **off**,
glob=1: severed E21 was **0 @800 / 0.01 @3200** vs E18 **0** / dense
**62.94**; extra-hop E21 is **0 @800 / 0.01 @1150** vs this-JSON E18
**0** and vs 0.75× dense **47.40**. Extra hop at shuffled n_dist=2
**S1 PASS 22.60** does not transfer to 2-item MATCH2 at 1280.

SWA stays severed (`msg_keepswa=False`). Exclusive slots stay on the
global read (`msg_rawkv=False`). Extra hop is in-attention over frozen
slot K/V (`msg_extrahops=1`, `msg_updatekv=False`). Window 16 still
cannot span gap 65–100. `e18_local` stays at chance (K2) — extra hop did
not leak a solvable local-only channel.

2-item MATCH2 length wall stays **(1024 S1 PASS, 1280 FAIL]** even with
one extra exclusive hop. Distinct from 2-key MATCH identity at 1280
(H=256 log S1 PASS 63.94). Item-count wall stays **(2-item S1 PASS,
3-item dense K1]**. Extra hop composes shuffled REACHABILITY, not this
MATCH2 length cell.

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not H=256
on MATCH2. Do not MATCH2 n_dist=2. Do not MATCH2 length extra-steps
(1100/1200/1408). Do not hops 320. Do not glob=3. Do not stack
`--message_keep_local_swa` on this extra-hop JSON. Do not n_dist=3. Do
not SELECT 694. Do not SELECT 696 extra hop. Do not INDEX 1280
extra-steps. Not Glyph. Do not unfreeze `u`/`delta`. Do not restore full
raw prefix KV (that is E18). Code default `--message_extra_slot_attends`
stays **0**. Code default `--message_keep_local_swa` stays **false**.
Code default `--global_layers` stays **1**. Next ONE (do not run):
**1280 MATCH (`recall_single`) r=8 rem-off H=256 log `--message_extra_slot_attends 1`**
(keepswa off, glob=1, same r=8 compressor as the exclusive MATCH wall;
E18 live 63.95). Do not relabel this MATCH2 extra-hop 0 as that score.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
