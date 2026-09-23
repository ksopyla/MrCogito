# E25 bridge seq=256 `--recipe chain` shuffled REACHABILITY --key_len 13 `--global_layers 2` — E21 S1 FAIL vs 0.75× dense (rung 5bx)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_256_chain_shuf_k13_glob2` (800 advertised; dense-matched 3200; e18 / e21 / `e18_local` scored; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge256_chain_shuf_k13_glob2/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge256_chain_shuf_k13_glob2/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `527abd3` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 256 ordered hops glob=2 S1 PASS [`e25_bridge256_chain_k13_glob2_20260914.md`](e25_bridge256_chain_k13_glob2_20260914.md) · 1152 MATCH2 n_dist=1 dense K1 [`e25_bridge1k_1152_ip_id_match2_nd1_20260915.md`](e25_bridge1k_1152_ip_id_match2_nd1_20260915.md)

---

## Goal

Unmeasured USER_CORE DNA recipe after STOP MATCH2 2-item length extra-steps:
shuffled REACHABILITY (`--recipe chain`) at the geometry where ordered DFA hops
already **S1 PASS** (seq=256, `--key_len 13`, `--global_layers 2`, H=256 SSMax
log, inplace r=1 identity). Same hops=2, same packed `n_distractors=2`, 26-bit
prize. Only the generator shuffles hop-edges (`meta.shuffled=True`).

Do **not** relabel ordered hops 25.85 as this score. Do not confuse with 512
`chain_ordered` dense K1. If E18 ≈ 0, score vs 0.75× dense (do not pass via
0.75×0). Chance at 800 → no 8k. Climbing short of S1 → extra-step 8k only
then. Do not 16k. Do not hops 320. glob=2 is required for hops.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · **2.802M** (e21 **2.835M**) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 256 --recipe chain`, **`--key_len 13`**, hops=2, n_distractors=**2**, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/92/119 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.1GB, 0.04–0.05 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_256_chain_shuf_k13_glob2 0 \
  --scale bridge --seq_len 256 --recipe chain --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --steps 800 --k1_mult 4
# named recipe chain = shuffled REACHABILITY (not chain_ordered, not chain_shuffled)
# --global_layers 2 required (hops); default glob stays 1
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused)
# E21 chance floor at 800 and at dense-matched 3200: do not extra-step; do not 16k
```

Byobu `E25_256_chain_shuf_glob2`. Log:
`seq=256  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=65/92/119`
dense `patterns=[('full', 0)×5]  glob_layers=2` · 2.802M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_boundary=10  msg_r=1  msg_remainder=False  msg_inplace=True
msg_rawkv=False  msg_idslots=True  msg_keepswa=False  glob_layers=2
msg_extrahops=0  msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 256`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 2`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`n_distractors: 2`, `hops: 2`, `prize_bits: 26.0`, `calibrated: true`,
`dense_steps_used: 3200`. Hunt exit **0**. Params <100M. Dense S0
recalibrated at two full layers (2.802M). Seed-0 row `meta.shuffled=True`.
Code defaults unchanged (`--global_layers` stays 1 except hops hunts).

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense @800 | **73.4%** | 800 | **13.18** | **0.507** | — |
| dense | **93.1%** | 3200 | **22.89** | **0.880** | **0.01118** |
| dense best acc | **93.8%** | 3050 | — | — | — |
| e18 @800 | **24.3%** | 800 | **0.00** | — | — |
| e18 | **25.2%** | 3200 | **0.00** | **0.000** | **0** |
| **e21 @800** | **24.3%** | 800 | **0.00** | **0** | **0** |
| **e21** | **24.3%** | **3200** | **0.00** | **0.000** | **0** |
| e18_local | **24.9%** | 3200 | **0.00** | **0.000** | **0** |

Dense left chance ~150, **73.4% / 13.18 bits @800**, crossed 75% at **2600**
(75.1%), plateaued **93.1% / 22.89 bits** @3200 (best 93.8% @3050; no 99%
early-stop; **S0 PASS**). JSON `calibrated: true`. Hunt exit 0.

E18 stayed at chance every eval through 3200 (**0.00 bits**, CE at ln(4);
best acc 25.6% @300 is chance noise). Live **~0**. Do **not** pass S1 via
0.75×0. Score vs **0.75× dense 17.17 bits**.

E21 stayed at chance every eval through 800 and through dense-matched 3200
(**0.00 bits**, CE at ln(4); best 26.1% @350 is chance noise; **S1 FAIL** vs
0.75× dense 17.17; flow 0). `e18_local` at chance (**K2 PASS**).

Do **not** relabel 256 ordered hops glob=2 E21 **25.85 bits** as this shuffled
score. Do not relabel E18 0 as an E21 pass.

**8k not run** (E21 chance floor at 800 and at dense-matched 3200; do not
extra-step a chance floor). Do not 16k.

One plot: learning curves — dense solves shuffled hops; E18 / E21 / local
stay on the chance line.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 93.1% / 22.89 bits @3200. Recalibrated at `global_layers=2`. |
| **S1 vs 0.75× live E18** | **not used.** E18 ≈ 0; do not pass via 0.75×0. |
| **S1 vs 0.75× dense** | **FAIL.** 0.00 < 17.17 bits. |
| **content vs dense** | E21 0.00 << dense 22.89. |
| **S2** plots | **PASS** (curves + bits/flow/bytes-per-token). |
| **K1** | not triggered. |
| **K2** | **PASS.** `e18_local` 24.9% / 0 bits. |
| **K3** | **triggered.** E21 chance floor @800 and @3200. |
| **8k** | **not run** (chance floor). |

## Interpretation

**Packed 256 shuffled REACHABILITY is dense-solvable at the ordered-hops
geometry, but neither the raw one-read (E18) nor the exclusive slot read
(E21) composes shuffled hops.** Same compressor, two global Blocks, identity
slots, 26-bit prize, n_distractors=2: ordered DFA was E21 **25.85 bits S1
PASS** / E18 live **5.56**; shuffled is E21 **0** / E18 **0** / dense
**22.89**. This is not an E21-only compression failure — uncompressed E18 is
also chance. Shuffle of hop-edges (REACHABILITY-hard) is the wall, not
length and not exclusive QUERY sever.

Ordered hops walls stay as mapped (severed **(264 PASS, 272 FAIL]**; keepswa
**(272 PASS, 288 K1]**; extra hop **(272 PASS, 288 FAIL]**; dense **(288 S0,
320 K1]**). Do not relabel those as shuffled.

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not hops 320.
Do not glob=3 / keepswa / extra hop / update_slot_kv / window 32 on this
hunt. Do not MATCH2 length extra-steps. Do not SELECT 696 extra hop. Do not
Glyph. Do not unfreeze `u`/`delta`. Do not restore full raw prefix KV (that
is E18). Code default `--global_layers` stays **1**. Code default extra hops
/ keep_local_swa / remainder stay off.
Next ONE (do not run): **256 `--recipe chain_shuffled` (n_distractors=0)
`--key_len 13 --global_layers 2` H=256 log identity** — same geometry; tests
whether extra shuffled distractor edges are the REACHABILITY wall versus
shuffle of the two-edge chain itself.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
