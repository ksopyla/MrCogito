# E25 bridge seq=256 `--recipe chain --n_distractors 1` --key_len 13 `--global_layers 2` — E21 S1 FAIL vs live E18 (rung 5bz)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_256_chain_shuf1_k13_glob2` (800 advertised; dense-matched 3200; e18 early-stop 2450; e21 / `e18_local` scored to 3200; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge256_chain_shuf1_k13_glob2/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge256_chain_shuf1_k13_glob2/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `2c41fec` (Odra E24 worktree HEAD at launch; `nn/` same as `b3d2f68`)
**Git tag:** —
**Related:** 256 n_dist=0 shuffled chain glob=2 S1 PASS [`e25_bridge256_chain_shuf0_k13_glob2_20260915.md`](e25_bridge256_chain_shuf0_k13_glob2_20260915.md) · 256 n_dist=2 shuffled chain glob=2 S1 FAIL [`e25_bridge256_chain_shuf_k13_glob2_20260915.md`](e25_bridge256_chain_shuf_k13_glob2_20260915.md) · 256 ordered hops glob=2 S1 PASS [`e25_bridge256_chain_k13_glob2_20260914.md`](e25_bridge256_chain_k13_glob2_20260914.md)

---

## Goal

Unmeasured USER_CORE DNA interpolant after n_dist=0 shuffled REACHABILITY S1 PASS
16.60 and n_dist=2 S1 FAIL 0: `--recipe chain --n_distractors 1` (one extra
shuffled distractor hop-edge) at the geometry where ordered DFA hops already
**S1 PASS** (seq=256, `--key_len 13`, `--global_layers 2`, H=256 SSMax log,
inplace r=1 identity). Same hops=2, 26-bit prize. Packed `n_distractors=1`
confirmed before launch (`hops=2`, `meta.shuffled=True`, 3 hop-blocks). Do
**not** use `chain_shuffled` (that forces n_dist=0).

Do **not** relabel n_dist=0 E21 16.60, ordered hops 25.85, or n_dist=2 E21 0
as this score. If E18 ≈ 0, score vs 0.75× dense (do not pass via 0.75×0).
Chance at 800 → no 8k. Climbing short of S1 → extra-step 8k only then. S1
PASS → no 8k. Do not 16k. Do not hops 320. glob=2 is required for hops.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · **2.802M** (e21 **2.835M**) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 256 --recipe chain`, **`--n_distractors 1`**, **`--key_len 13`**, hops=2, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/65/92 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.1GB, 0.04 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_256_chain_shuf1_k13_glob2 0 \
  --scale bridge --seq_len 256 --recipe chain --n_distractors 1 --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --steps 800 --k1_mult 4
# named recipe chain + --n_distractors 1 = shuffled REACHABILITY with one extra edge
# (not chain_ordered, not chain_shuffled which forces n_dist=0)
# --global_layers 2 required (hops); default glob stays 1
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused)
# E21 chance at 800: do not extra-step; do not 16k
```

Byobu `E25_256_chain_shuf1_glob2`. Log:
`seq=256  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=65/65/92`
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
`n_distractors: 1`, `hops: 2`, `prize_bits: 26.0`, `calibrated: true`,
`dense_steps_used: 3200`. Hunt exit **0**. Params <100M. Dense S0
recalibrated at two full layers (2.802M). Seed-0 row `meta.shuffled=True`.
Code defaults unchanged (`--global_layers` stays 1 except hops hunts).

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense @800 | **60.2%** | 800 | **10.51** | **0.404** | — |
| dense | **86.4%** | 3200 | **21.54** | **0.829** | **0.01052** |
| dense best acc | **87.3%** | 2800 | — | — | — |
| e18 @800 | **58.1%** | 800 | **11.27** | **0.434** | — |
| e18 | **99.0%** | 2450 | **25.44** | **0.979** | **0.01242** |
| **e21 @800** | **26.1%** | **800** | **0.00** | **0** | **0** |
| **e21** | **46.0%** | **3200** | **6.28** | **0.242** | **0.00307** |
| e21 best acc | **47.4%** | 3150 | — | — | — |
| e18_local | **24.8%** | 3200 | **0.00** | **0.000** | **0** |

Dense left chance ~250, **60.2% / 10.51 bits @800**, crossed 75% at **1200**
(75.2%), plateaued **86.4% / 21.54 bits** @3200 (best 87.3% @2800; no 99%
early-stop; **S0 PASS**). JSON `calibrated: true`. Hunt exit 0.

E18 left chance ~500, **58.1% / 11.27 bits @800**, clicked 89.8% @1900,
early-stopped **99.0% / 25.44 bits** @2450 (CE 0.030; live). Score S1 vs
**0.75× live E18 19.08 bits**. Do **not** pass via 0.75×0 — E18 is live.

E21 stayed at chance every eval through 800 and through 1600 (**26.1% /
0.00 bits @800**, CE at ln(4)). Left chance ~1950, **46.0% / 6.28 bits**
@3200 (best 47.4% @3150; **S1 FAIL** vs 0.75× live E18 19.08; also **FAIL**
vs 0.75× dense 16.16; flow 0.242). `e18_local` at chance (**K2 PASS**).

Do **not** relabel n_dist=0 shuffled E21 **16.60 bits** as this n_dist=1
score. Do not relabel ordered hops glob=2 E21 **25.85 bits**. Do not
relabel n_dist=2 E21 **0 bits**. Do not relabel E18 25.44 as an E21 score.

**8k not run** (E21 chance floor at the 800 advertised budget; SOP: chance
at 800 → no extra-step). The late 6.28-bit climb at dense-matched 3200 is
still short of S1 and does not unlock 8k. Do not 16k.

One plot: learning curves — dense and E18 solve one extra shuffled edge;
E21 stays on the chance line through 800 and only later climbs to ~46%.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** (dense ≥ 75%) | **PASS.** Dense 86.4% / 21.54 bits @3200. Recalibrated at `global_layers=2`. |
| **S1** (compressed read ≥ 75% of the raw-read control) | **FAIL.** 6.28 < 0.75× live E18 19.08 bits. |
| **S1 vs 0.75× dense** | **FAIL.** 6.28 < 16.16 bits. |
| **S2** plots | **PASS** (curves + bits/flow/bytes-per-token). |
| **K1** | not triggered. |
| **K2** | **PASS.** `e18_local` 24.8% / 0 bits. |
| **K3** | **triggered at 800.** E21 chance floor @800 (late climb @3200 is not S1). |
| **8k** | **not run** (chance at 800). |

## Interpretation

**Packed 256 shuffled REACHABILITY with one extra distractor edge is
dense-solvable and raw-read solvable, but the exclusive slot read does not
compose it at the 800 floor.** Same compressor, two global Blocks, identity
slots, 26-bit prize: n_dist=0 was E21 **16.60 S1 PASS** / E18 live **16.37**
/ dense **25.66**; n_dist=1 is E21 **6.28** / E18 live **25.44** / dense
**21.54**; n_dist=2 was E21 **0** / E18 **0** / dense **22.89**. One extra
shuffled hop-edge is enough to kill exclusive composition at 800, even
though uncompressed E18 still solves the rung. This is an E21 compression
failure at n_dist=1, not a shared E18/E21 chance wall (that was n_dist=2).

Shuffled distractor wall at this geometry is now **(n_dist=0 S1 PASS,
n_dist=1 S1 FAIL]**. Ordered hops walls stay as mapped (severed
**(264 PASS, 272 FAIL]**; keepswa **(272 PASS, 288 K1]**; extra hop
**(272 PASS, 288 FAIL]**; dense **(288 S0, 320 K1]**).

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not hops
320. Do not glob=3 / keepswa / extra hop / update_slot_kv / window 32
on this hunt. Do not MATCH2 length extra-steps. Do not SELECT 694. Do not
INDEX 1280 extra-steps. Do not Glyph. Do not unfreeze `u`/`delta`. Do not
restore full raw prefix KV (that is E18). Code default `--global_layers`
stays **1**. Code default extra hops / keep_local_swa / remainder stay off.
Next ONE (do not run): **STOP shuffled-distractor extra-steps at this
geometry** — wall is now **(n_dist=0 S1 PASS, n_dist=1 S1 FAIL]**.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
