# E25 bridge_1k seq=1536 `recall_single` r=1 identity H=256 log `--message_extra_slot_attends 1` — E21 S1 FAIL vs 0.75× live E18 (rung 5cf)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1536_ip_id_h256_recall_extrahop` (800 advertised; dense early-stop 200; e18 early-stop 350; e21 / `e18_local` scored to 800; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_1536_ip_id_h256_recall_extrahop/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_1536_ip_id_h256_recall_extrahop/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `0d95e2c` (Odra E24 worktree HEAD at launch; `nn/` unchanged vs hunt SHA `c5c13a2`)
**Git tag:** —
**Related:** 1536 MATCH identity glob=1 S1 FAIL [`e25_bridge1k_1536_ip_id_h256_recall_20260914.md`](e25_bridge1k_1536_ip_id_h256_recall_20260914.md) · 1536 MATCH identity glob=2 S1 FAIL [`e25_bridge1k_1536_ip_id_h256_glob2_recall_20260915.md`](e25_bridge1k_1536_ip_id_h256_glob2_recall_20260915.md) · 1280 MATCH identity S1 PASS [`e25_bridge1k_1280_ip_id_h256_recall_20260914.md`](e25_bridge1k_1280_ip_id_h256_recall_20260914.md) · 1280 MATCH r=8 extra hop S1 FAIL [`e25_bridge1k_1280_ip_r8_h256_recall_extrahop_20260915.md`](e25_bridge1k_1280_ip_r8_h256_recall_extrahop_20260915.md) · 256 shuffled n_dist=2 extra hop S1 PASS [`e25_bridge256_chain_shuf_k13_glob2_extrahop_20260915.md`](e25_bridge256_chain_shuf_k13_glob2_extrahop_20260915.md)

---

## Goal

Architectural bet at the packed 2-key MATCH identity exclusive-capacity
wall (`--recipe recall_single` seq=1536 r=1 identity H=256 log), where
severed E21 was **1.50 bits** (last-eval wiggle) and uncompressed E18
was **live 63.95**. Extra hop **passed** shuffled n_dist=1/2 and ordered
hops 272, and **failed** MATCH pooling r=8 at 1280, MATCH2 1280, and
ordered hops 288. Same identity compressor as the exclusive FAIL cell;
**one** knob: `--message_extra_slot_attends 1`. Keepswa **off**. glob=**1**.
Do **not** `--n_distractors` (this is not MATCH2). Falsifiable claim:
extra exclusive hop composes identity keys at 1536 vs capacity (distinct
from pooling r=8 extra hop and from MATCH2 1280).

Do **not** relabel MATCH identity 1.50, MATCH identity glob=2 0, MATCH
identity 1280 63.94, MATCH r=8 extra hop 0.01, MATCH2 extra hop 0, or
shuffled extra hop 22.60 as this score. Score vs 0.75× **live E18**; if
E18 ≈ 0, score vs 0.75× dense (do not pass via 0.75×0). Chance at 800 →
no 8k. Climbing short of S1 → extra-step 8k only then. S1 PASS → no 8k.
Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / `e18_local` |
| Width | **H=256** · **2.261M** (e21 **2.277M**) · kv=1 · **`logit_scale=log`** · stack=2 · glob=1 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=1`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=1`** |
| Data | `--scale bridge_1k --seq_len 1536 --recipe recall_single`, packed answer 32 / **64-bit** prize, **2-key MATCH** (`n_distractors=0`) |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/65/65 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~4.3–4.9GB, 0.11–0.16 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1536_ip_id_h256_recall_extrahop 0 \
  --scale bridge_1k --seq_len 1536 --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --message_extra_slot_attends 1 --steps 800 --k1_mult 4
# named recipe recall_single = 2-key MATCH (n_distractors=0; not MATCH2)
# --seq_len 1536 and --message_ratio 1 required (identity capacity wall)
# --message_extra_slot_attends 1 required (this bet); exclusive slots stay on the global read
# no --n_distractors
# no --global_layers (default 1)
# no --message_pool_remainder
# no --message_keep_local_swa (default false; SWA still severed; do not stack)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused)
# E21 chance floor at 800: do not extra-step; do not 16k
```

Byobu `E25_1536_id_extrahop`. Log:
`seq=1536  gap=64  window=16  prize=64.00 bits  answer_len=32
row_gap[min/med/max]=65/65/65`
dense `patterns=[('full', 0)×4]` · 2.261M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_boundary=10  msg_r=1  msg_remainder=False  msg_inplace=True
msg_rawkv=False  msg_idslots=True  msg_keepswa=False  glob_layers=1
msg_extrahops=1  msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 1536`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 1`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
**`message_keep_local_swa: false`**, **`message_extra_slot_attends: 1`**,
`message_update_slot_kv: false`, `message_global_anchors: none`,
**`n_distractors: 0`**, `prize_bits: 64.0`, `calibrated: true`,
`dense_steps_used: 200`. Hunt exit **0**. Params <100M. Dense S0
recalibrated at H=256 (2.261M). Code defaults unchanged
(`--message_extra_slot_attends` stays 0; `--message_keep_local_swa`
stays false; `--global_layers` stays 1). Packed 2-key MATCH
(`recall_single`).

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense | **100%** | 200 | **63.92** | **0.999** | **0.005202** |
| e18 | **100%** | 350 | **63.94** | **0.999** | **0.005204** |
| **e21 @800** | **24.7%** | **800** | **0.00** | **0.000** | **0** |
| e21 best acc | **27.5%** | 100 / 600 | (chance; CE at ln(4)) | — | — |
| e18_local | **24.7%** | 800 | **0.00** | **0** | **0** |

Dense left chance through 100, **54.6% @150**, early-stop **100% /
63.92 bits** @200 (**S0 PASS**). JSON `calibrated: true`. Hunt exit 0.

This-JSON E18 chance through 250, click **53.9% @300**, early-stop
**100% / 63.94 bits** @350 (**live — not ~0**). S1 bar is **0.75× E18
= 47.96 bits** (also vs **0.75× dense = 47.94 bits**).

E21 stayed at chance every eval through 800 (**24.7% / 0.00 bits @800**,
CE 1.3872 > ln(4); flow **0**; best acc **27.5%** @100/@600 is chance
noise, not a climb; **S1 FAIL** vs 0.75× live E18 47.96). `e18_local`
at chance (**K2 PASS**).

Do **not** relabel MATCH identity **1.50**, MATCH identity glob=2 **0**,
MATCH identity 1280 **63.94**, MATCH r=8 extra hop **0.01**, MATCH2 extra
hop **0**, or shuffled extra hop **22.60** as this score. Do not relabel
this-JSON E18 **63.94** as an E21 pass.

**8k not run** (E21 chance floor at 800; do not extra-step a chance
floor). Do not 16k.

One plot: learning curves — dense and live E18 solve; extra-hop E21
and `e18_local` stay on the chance line.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** (dense ≥ 75%) | **PASS.** Dense 100% / 63.92 bits @200 (crossed 75% at 150). Recalibrated at H=256. |
| **S1 vs 0.75× live E18** | **FAIL.** 0.00 ≪ 47.96 bits. Chance floor. Do not pass via 0.75×0 — E18 is live. |
| **content vs 0.75× dense** | **FAIL.** 0.00 ≪ 47.94 bits. |
| **S2** plots | **PASS** (curves + bits/flow/bytes-per-token). |
| **K1** | not triggered. |
| **K2** | **PASS.** `e18_local` 24.7% / 0.00 bits @800. |
| **K3** | **triggered.** E21 chance floor (flow 0.000). |
| **8k** | **not run.** Chance at 800. Do not 16k. |

## Interpretation

**An extra exclusive hop over frozen identity slots does not rescue
packed 1536 MATCH.** Same compressor, one global Block, identity slots,
64-bit prize, keepswa **off**, glob=1: severed E21 was **1.50 @800** vs
live E18 **63.95** / dense **63.95**; extra-hop E21 is **0 @800** vs
this-JSON live E18 **63.94** and vs 0.75× E18 **47.96**. Extra hop at
shuffled n_dist=2 **S1 PASS 22.60** does not transfer to MATCH identity
capacity at 1536. MATCH pooling extra hop at 1280 r=8 was also 0 with
live E18; here the identity channel that passed at 1280 (63.94) still
dies at 1536 even with a second exclusive attend.

SWA stays severed (`msg_keepswa=False`). Exclusive slots stay on the
global read (`msg_rawkv=False`). Extra hop is in-attention over frozen
slot K/V (`msg_extrahops=1`, `msg_updatekv=False`). Window 16 still
cannot span gap 65. `e18_local` stays at chance (K2) — extra hop did not
leak a solvable local-only channel.

Identity MATCH length wall stays **(1280 S1 PASS, 1536 FAIL]** even
with one extra exclusive hop. Distinct from pooling r=8 extra hop at
1280 (also FAIL) and from MATCH2 1280 extra hop (FAIL with dead E18).
Glob=2 at this length stays 0.

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not 1408.
Do not glob=2. Do not keepswa. Do not MATCH pooling extra hop again.
Do not MATCH2 1280 extra hop again. Do not hops 320. Do not SELECT 694.
Do not INDEX 1280. Do not shuffled n_dist=3. Do not stack keepswa+extra hop.
Not Glyph. Do not unfreeze `u`/`delta`. Do not restore full raw prefix KV
(that is E18). Code default `--message_extra_slot_attends` stays **0**.
Code default `--message_keep_local_swa` stays **false**. Code default
`--global_layers` stays **1**. Next ONE (do not run): **STOP extra-hop
transfer to MATCH identity capacity** (extra hop does not rescue r=1
identity at 1536).

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
