# E25 bridge seq=256 `--recipe chain --n_distractors 2` --key_len 13 `--global_layers 2 --message_extra_slot_attends 1` — E21 S1 PASS vs 0.75× dense (rung 5cc)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_256_chain_shuf_k13_glob2_extrahop` (800 advertised; dense-matched 3200; e18 / e21 / `e18_local` scored to 3200; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge256_chain_shuf_k13_glob2_extrahop/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge256_chain_shuf_k13_glob2_extrahop/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `4c145bf` (Odra E24 worktree HEAD at launch; `nn/` unchanged vs hunt SHA `ac94bda`)
**Git tag:** —
**Related:** 256 n_dist=2 shuffled chain glob=2 severed S1 FAIL [`e25_bridge256_chain_shuf_k13_glob2_20260915.md`](e25_bridge256_chain_shuf_k13_glob2_20260915.md) · 256 n_dist=1 extra hop S1 PASS [`e25_bridge256_chain_shuf1_k13_glob2_extrahop_20260915.md`](e25_bridge256_chain_shuf1_k13_glob2_extrahop_20260915.md) · 256 n_dist=1 keepswa S1 FAIL [`e25_bridge256_chain_shuf1_k13_glob2_keepswa_20260915.md`](e25_bridge256_chain_shuf1_k13_glob2_keepswa_20260915.md) · 256 n_dist=0 shuffled chain glob=2 S1 PASS [`e25_bridge256_chain_shuf0_k13_glob2_20260915.md`](e25_bridge256_chain_shuf0_k13_glob2_20260915.md)

---

## Goal

Architectural bet at the packed shuffled REACHABILITY wall (`--recipe chain`
default n_dist=2 hops=2), where severed E21 was **0** and uncompressed E18
was **0** vs dense **22.89**. Extra hop at n_dist=1 already **S1 PASS
24.08**. Same seq=256 `--recipe chain --n_distractors 2` `--key_len 13`
`--global_layers 2` H=256 identity recipe; **one** knob:
`--message_extra_slot_attends 1`. Keepswa **off**. Do **not** stack keepswa
+ extra hop. Do **not** glob=3. Falsifiable claim: extra exclusive hop
over frozen slots composes packed shuffled distractor edges without
restoring SWA.

Do **not** relabel severed n_dist=2 E21 0, n_dist=1 extra hop 24.08,
keepswa 14.77, n_dist=0 16.60, ordered hops 25.85, hops 272 extra hop
25.48, or hops 288 extra hop 0 as this score. If this-JSON E18 ≈ 0, score
vs 0.75× dense (do not pass via 0.75×0). Chance at 800 → no 8k. Climbing
short of S1 → extra-step 8k only then. S1 PASS → no 8k. Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / `e18_local` |
| Width | **H=256** · **2.802M** (e21 **2.835M**) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=1`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 256 --recipe chain`, **`--n_distractors 2`**, **`--key_len 13`**, hops=**2**, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/92/119 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.1–1.2GB, 0.04–0.05 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_256_chain_shuf_k13_glob2_extrahop 0 \
  --scale bridge --seq_len 256 --recipe chain --n_distractors 2 --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --message_extra_slot_attends 1 \
  --steps 800 --k1_mult 4
# named recipe chain default = shuffled REACHABILITY packed n_dist=2 hops=2
# --global_layers 2 required (hops); default glob stays 1
# --message_extra_slot_attends 1 required (this bet); exclusive slots stay on the global read
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_keep_local_swa (default false; SWA still severed; do not stack)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused)
# S1 PASS at 3200 → no 8k; do not 16k
```

Byobu `E25_256_chain_shuf_extrahop`. Log:
`seq=256  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=65/92/119`
dense `patterns=[('full', 0)×5]  glob_layers=2` · 2.802M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_boundary=10  msg_r=1  msg_remainder=False  msg_inplace=True
msg_rawkv=False  msg_idslots=True  msg_keepswa=False  glob_layers=2
msg_extrahops=1  msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 256`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 2`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
**`message_keep_local_swa: false`**, **`message_extra_slot_attends: 1`**,
`message_update_slot_kv: false`, `message_global_anchors: none`,
**`n_distractors: 2`**, **`hops: 2`**, `prize_bits: 26.0`, `calibrated: true`,
`dense_steps_used: 3200`. Hunt exit **0**. Params <100M. Dense S0
recalibrated at two full layers (2.802M). Seed-0 row `meta.shuffled=True`.
Code defaults unchanged (`--global_layers` stays 1 except hops hunts;
`--message_extra_slot_attends` stays 0; `--message_keep_local_swa` stays
false). Packed named recipe `chain` at this CLI is hops=2 n_dist=2.

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense @800 | **52.3%** | 800 | **4.68** | — | — |
| dense | **93.4%** | 3200 | **22.94** | **0.882** | **0.01120** |
| dense best acc | **93.4%** | 3200 | **22.94** | — | — |
| e18 @800 | **24.3%** | 800 | **0.00** | — | — |
| e18 | **25.2%** | 3200 | **0.00** | **0.000** | **0** |
| e18 best acc | **25.7%** | 200 | — | — | — |
| **e21 @800** | **43.4%** | **800** | **3.81** | — | — |
| **e21** | **91.5%** | **3200** | **22.60** | **0.869** | **0.01103** |
| e21 best acc | **93.1%** | 3000 | **23.00** | — | — |
| e18_local | **24.9%** | 3200 | **0.00** | **0** | **0** |

Dense left chance ~200, **52.3% / 4.68 bits @800**, crossed 75% at
**1800**, **93.4% / 22.94 bits** @3200 (best acc at last eval; **S0
PASS**). JSON `calibrated: true`. Hunt exit 0.

This-JSON E18 stayed at chance every eval through 3200 (**0.00 bits**, CE
at ln(4); best acc 25.7% @200 is chance noise). Live **~0**. Do **not**
pass S1 via 0.75×0. Score vs **0.75× dense 17.21 bits**.

E21 left chance ~600, **43.4% / 3.81 bits @800** (not chance floor),
crossed 75% at **1850**, **91.5% / 22.60 bits** @3200 (best acc 93.1% /
23.00 bits @3000; **S1 PASS** vs 0.75× dense 17.21). `e18_local` at
chance (**K2 PASS**).

Do **not** relabel severed n_dist=2 E21 **0**, n_dist=1 extra hop
**24.08**, keepswa **14.77**, n_dist=0 **16.60**, ordered hops glob=2
**25.85**, hops 272 extra hop **25.48**, or hops 288 extra hop **0** as
this score. Do not relabel this-JSON E18 0 as an E21 pass.

**8k not run** (S1 PASS at the 800 advertised / 3200 matched budget). Do
not 16k.

One plot: learning curves — dense and extra-hop E21 both solve; this-JSON
E18 and `e18_local` stay on the chance line.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** (dense ≥ 75%) | **PASS.** Dense 93.4% / 22.94 bits @3200. Recalibrated at `global_layers=2`. |
| **S1 vs 0.75× live E18** | **not used.** E18 ≈ 0; do not pass via 0.75×0. |
| **S1 vs 0.75× dense** | **PASS.** 22.60 > 17.21 bits. |
| **content vs dense** | E21 22.60 tracks dense 22.94. |
| **S2** plots | **PASS** (curves + bits/flow/bytes-per-token). |
| **K1** | not triggered. |
| **K2** | **PASS.** `e18_local` 24.9% / 0 bits @3200. |
| **K3** | not triggered at 800 (E21 43.4% / 3.81 bits, climbing). |
| **8k** | **not run.** S1 PASS. Do not 16k. |

## Interpretation

**An extra exclusive hop over frozen slots rescues packed shuffled
REACHABILITY without restoring SWA.** Same compressor, two global Blocks,
identity slots, 26-bit prize, n_dist=2 hops=2, keepswa **off**: severed
E21 was **0 @800 / 0 @3200** vs E18 **0** / dense **22.89**; extra-hop
E21 is **3.81 @800 / 22.60 @3200** vs this-JSON E18 **0** and vs 0.75×
dense **17.21**. Extra hop at n_dist=1 **S1 PASS 24.08** transfers to the
packed default. Keepswa at n_dist=1 **S1 FAIL 14.77** does not.

SWA stays severed (`msg_keepswa=False`). Exclusive slots stay on the
global read (`msg_rawkv=False`). Extra hop is in-attention over frozen
slot K/V (`msg_extrahops=1`, `msg_updatekv=False`). Window 16 still
cannot span gap 65–119. `e18_local` stays at chance (K2) — extra hop did
not leak a solvable local-only channel.

Severed shuffled wall stays **(n_dist=0 S1 PASS, n_dist=1 S1 FAIL]**.
Keepswa shuffled wall stays **(n_dist=0 S1 PASS, n_dist=1 S1 FAIL]**.
Extra-hop shuffled packed n_dist=2 is **S1 PASS**. Ordered hops walls
stay as mapped (severed **(264 PASS, 272 FAIL]**; keepswa **(272 PASS,
288 K1]**; extra hop **(272 PASS, 288 FAIL]**).

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not hops
320. Do not glob=3. Do not stack `--message_keep_local_swa` on this
extra-hop JSON. Do not n_dist=3. Do not MATCH2 length extra-steps. Do not
SELECT 694. Do not INDEX 1280 extra-steps. Not Glyph. Do not unfreeze
`u`/`delta`. Do not restore full raw prefix KV (that is E18). Code
default `--message_extra_slot_attends` stays **0**. Code default
`--message_keep_local_swa` stays **false**. Code default `--global_layers`
stays **1**. Next ONE (do not run): **STOP extra-hop shuffled-distractor
extra-steps** (do not n_dist=3). Live extra-hop shuffled S1 is packed
n_dist=2 **22.60** (n_dist=1 **24.08** still mapped; do not relabel).

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
