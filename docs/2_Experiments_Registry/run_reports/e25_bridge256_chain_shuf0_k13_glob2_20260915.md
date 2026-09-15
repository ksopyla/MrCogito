# E25 bridge seq=256 `--recipe chain_shuffled` n_dist=0 --key_len 13 `--global_layers 2` — E21 S1 PASS vs live E18 (rung 5by)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_256_chain_shuf0_k13_glob2` (800 advertised; dense early-stop 700; e18 / e21 / `e18_local` scored to 800; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge256_chain_shuf0_k13_glob2/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge256_chain_shuf0_k13_glob2/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `b3d2f68` (Odra E24 worktree HEAD at launch; `nn/` same as `527abd3`; E22 main checkout untouched)
**Git tag:** —
**Related:** 256 n_dist=2 shuffled chain glob=2 S1 FAIL [`e25_bridge256_chain_shuf_k13_glob2_20260915.md`](e25_bridge256_chain_shuf_k13_glob2_20260915.md) · 256 ordered hops glob=2 S1 PASS [`e25_bridge256_chain_k13_glob2_20260914.md`](e25_bridge256_chain_k13_glob2_20260914.md)

---

## Goal

Unmeasured USER_CORE DNA recipe after n_dist=2 shuffled REACHABILITY S1 FAIL:
`--recipe chain_shuffled` (`n_distractors=0`) at the geometry where ordered DFA hops
already **S1 PASS** (seq=256, `--key_len 13`, `--global_layers 2`, H=256 SSMax
log, inplace r=1 identity). Same hops=2, 26-bit prize. Only the extra shuffled
distractor hop-edges are removed. Seed-0 `meta.shuffled=True`. Tests whether
those extra edges are the REACHABILITY wall versus shuffle of the two-edge
chain itself.

Do **not** relabel n_dist=2 E21 0 bits as this score. Do not relabel ordered
hops 25.85. If E18 ≈ 0, score vs 0.75× dense (do not pass via 0.75×0). Chance
at 800 → no 8k. Climbing short of S1 → extra-step 8k only then. S1 PASS → no
8k. Do not 16k. Do not hops 320. glob=2 is required for hops.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · **2.802M** (e21 **2.835M**) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 256 --recipe chain_shuffled`, **`--key_len 13`**, hops=2, n_distractors=**0**, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/65/65 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.1GB, 0.04–0.05 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_256_chain_shuf0_k13_glob2 0 \
  --scale bridge --seq_len 256 --recipe chain_shuffled --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --steps 800 --k1_mult 4
# named recipe chain_shuffled = shuffled REACHABILITY with n_distractors=0
# (not chain_ordered, not default chain n_dist=2)
# --global_layers 2 required (hops); default glob stays 1
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused)
# E21 S1 PASS vs live E18 at 800: do not extra-step; do not 16k
```

Byobu `E25_256_chain_shuf0_glob2`. Log:
`seq=256  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=65/65/65`
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
`n_distractors: 0`, `hops: 2`, `prize_bits: 26.0`, `calibrated: true`,
`dense_steps_used: 700`. Hunt exit **0**. Params <100M. Dense S0
recalibrated at two full layers (2.802M). Seed-0 row `meta.shuffled=True`.
Code defaults unchanged (`--global_layers` stays 1 except hops hunts).

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense @600 | **73.7%** | 600 | **12.61** | **0.485** | — |
| dense | **99.3%** | 700 | **25.66** | **0.987** | **0.01253** |
| e18 | **71.5%** | 800 | **16.37** | **0.630** | **0.00799** |
| e18 best acc | **72.5%** | 700 | — | — | — |
| **e21** | **71.9%** | **800** | **16.60** | **0.638** | **0.00810** |
| e18_local | **22.1%** | 800 | **0.00** | **0.000** | **0** |

Dense left chance ~150, **73.7% / 12.61 bits @600**, early-stopped
**99.3% / 25.66 bits** @700 (**S0 PASS**; no 800 eval). JSON
`calibrated: true`. Hunt exit 0.

E18 left chance ~300, clicked ~45% @400, **71.5% / 16.37 bits** @800
(best acc 72.5% @700; CE 0.513; live). Score S1 vs **0.75× live E18
12.28 bits**. Do **not** pass via 0.75×0 — E18 is live.

E21 left chance ~350, clicked ~43% @400, **71.9% / 16.60 bits** @800
(best acc at last eval; CE 0.501; **S1 PASS** vs 0.75× live E18 12.28;
flow 0.638). **FAIL** vs 0.75× dense 19.24 (content vs dense).
`e18_local` at chance (**K2 PASS**).

Do **not** relabel n_dist=2 shuffled E21 **0 bits** as this n_dist=0
score. Do not relabel ordered hops glob=2 E21 **25.85 bits**. Do not
relabel E18 16.37 as an E21 score.

**8k not run** (S1 PASS at 800 vs live E18; SOP: S1 PASS → no extra-step).
Do not 16k. Both global-read arms are still climbing at ~72% while dense
is 99.3%; that is a content gap vs dense, not an S1 miss vs E18.

One plot: learning curves — dense solves the two-edge shuffle; E18 and
E21 leave chance together and track each other; local stays on the
chance line.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** (dense ≥ 75%) | **PASS.** Dense 99.3% / 25.66 bits @700. Recalibrated at `global_layers=2`. |
| **S1** (compressed read ≥ 75% of the raw-read control) | **PASS.** 16.60 > 0.75× live E18 12.28 bits. |
| **S1 vs 0.75× dense** | **FAIL.** 16.60 < 19.24 bits (content vs dense; not the S1 bar — E18 is live). |
| **S2** plots | **PASS** (curves + bits/flow/bytes-per-token). |
| **K1** | not triggered. |
| **K2** | **PASS.** `e18_local` 22.1% / 0 bits. |
| **K3** | not triggered (not a chance floor). |
| **8k** | **not run** (S1 PASS). |

## Interpretation

**Packed 256 shuffled REACHABILITY without extra distractor edges is
dense-solvable, and both the raw one-read (E18) and the exclusive slot
read (E21) compose the two-edge shuffle at this geometry.** Same
compressor, two global Blocks, identity slots, 26-bit prize: n_dist=2
was E21 **0** / E18 **0** / dense **22.89**; n_dist=0 is E21 **16.60** /
E18 **16.37** / dense **25.66**. Shuffle of the two hop-edges is *not*
the wall that killed n_dist=2 — the extra shuffled distractor edges
are. E21 slightly beats same-JSON E18 (16.60 vs 16.37). Do not relabel
this as ordered hops 25.85 (those were 99% / full prize).

Shuffled distractor wall at this geometry is **(n_dist=0 S1 PASS,
n_dist=2 S1 FAIL]**. Ordered hops walls stay as mapped (severed
**(264 PASS, 272 FAIL]**; keepswa **(272 PASS, 288 K1]**; extra hop
**(272 PASS, 288 FAIL]**; dense **(288 S0, 320 K1]**).

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not hops
320. Do not glob=3 / keepswa / extra hop / update_slot_kv / window 32
on this hunt. Do not MATCH2 length extra-steps. Do not SELECT 696 extra
hop. Do not Glyph. Do not unfreeze `u`/`delta`. Do not restore full raw
prefix KV (that is E18). Code default `--global_layers` stays **1**.
Code default extra hops / keep_local_swa / remainder stay off.
Next ONE (do not run): **256 `--recipe chain --n_distractors 1 --key_len
13 --global_layers 2` H=256 log identity** — interpolate the shuffled
distractor wall **(n_dist=0 S1 PASS, n_dist=2 S1 FAIL]**.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
