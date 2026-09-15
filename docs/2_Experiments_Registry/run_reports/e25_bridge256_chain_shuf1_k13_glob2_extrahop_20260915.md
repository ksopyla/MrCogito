# E25 bridge seq=256 `--recipe chain --n_distractors 1` --key_len 13 `--global_layers 2 --message_extra_slot_attends 1` — E21 S1 PASS (rung 5cb)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_256_chain_shuf1_k13_glob2_extrahop` (800 advertised; dense-matched 3200; e18 / e21 / `e18_local` scored to 3200; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge256_chain_shuf1_k13_glob2_extrahop/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge256_chain_shuf1_k13_glob2_extrahop/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `ac94bda` (Odra E24 worktree HEAD at launch; `nn/` unchanged)
**Git tag:** —
**Related:** 256 n_dist=1 shuffled chain glob=2 severed S1 FAIL [`e25_bridge256_chain_shuf1_k13_glob2_20260915.md`](e25_bridge256_chain_shuf1_k13_glob2_20260915.md) · 256 n_dist=1 keepswa S1 FAIL [`e25_bridge256_chain_shuf1_k13_glob2_keepswa_20260915.md`](e25_bridge256_chain_shuf1_k13_glob2_keepswa_20260915.md) · 256 n_dist=0 shuffled chain glob=2 S1 PASS [`e25_bridge256_chain_shuf0_k13_glob2_20260915.md`](e25_bridge256_chain_shuf0_k13_glob2_20260915.md) · 272 ordered hops extra hop S1 PASS [`e25_bridge272_chain_k13_glob2_extrahop_20260915.md`](e25_bridge272_chain_k13_glob2_extrahop_20260915.md)

---

## Goal

Architectural bet at the measured n_dist=1 shuffled exclusive wall, where
uncompressed E18 **solves** (25.44 bits) and severed E21 **fails** (0 @800 /
6.28 @3200). Keepswa at this geometry still **S1 FAIL 14.77 vs 0.75× live
E18 15.71**. Same seq=256 `--recipe chain --n_distractors 1` `--key_len 13`
`--global_layers 2` H=256 identity recipe; **one** knob:
`--message_extra_slot_attends 1`. Keepswa **off**. Do **not** stack keepswa
+ extra hop. Falsifiable claim: extra exclusive hop over frozen slots
composes the extra shuffled hop-edge without restoring SWA. Ordered hops
272 extra hop rescued 25.48 with SWA still severed.

Do **not** relabel severed n_dist=1 E21 6.28, keepswa 14.77, n_dist=0
16.60, ordered hops 25.85, or hops 272 extra hop 25.48 as this score. If
E18 ≈ 0, score vs 0.75× dense (do not pass via 0.75×0). Chance at 800 → no
8k. Climbing short of S1 → extra-step 8k only then. S1 PASS → no 8k. Do
not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / `e18_local` |
| Width | **H=256** · **2.802M** (e21 **2.835M**) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=1`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 256 --recipe chain`, **`--n_distractors 1`**, **`--key_len 13`**, hops=2, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/65/92 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.2GB, 0.04–0.05 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_256_chain_shuf1_k13_glob2_extrahop 0 \
  --scale bridge --seq_len 256 --recipe chain --n_distractors 1 --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --message_extra_slot_attends 1 \
  --steps 800 --k1_mult 4
# named recipe chain + --n_distractors 1 = shuffled REACHABILITY with one extra edge
# --global_layers 2 required (hops); default glob stays 1
# --message_extra_slot_attends 1 required (this bet); exclusive slots stay on the global read
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_keep_local_swa (default false; SWA still severed; do not stack)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused)
# S1 PASS at 3200 → no 8k; do not 16k
```

Byobu `E25_256_chain_shuf1_extrahop`. Log:
`seq=256  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=65/65/92`
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
`n_distractors: 1`, `hops: 2`, `prize_bits: 26.0`, `calibrated: true`,
`dense_steps_used: 3200`. Hunt exit **0**. Params <100M. Dense S0
recalibrated at two full layers (2.802M). Seed-0 row `meta.shuffled=True`.
Code defaults unchanged (`--global_layers` stays 1 except hops hunts;
`--message_extra_slot_attends` stays 0; `--message_keep_local_swa` stays
false).

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense @800 | **64.9%** | 800 | **11.06** | — | — |
| dense | **93.8%** | 3200 | **23.92** | **0.920** | **0.01168** |
| dense best acc | **93.8%** | 2900 | **23.59** | — | — |
| e18 @800 | **40.3%** | 800 | **5.33** | — | — |
| e18 | **47.1%** | 3200 | **9.75** | **0.375** | **0.00476** |
| e18 best acc | **51.1%** | 2600 | **9.41** | — | — |
| **e21 @800** | **60.7%** | **800** | **6.85** | — | — |
| **e21** | **93.6%** | **3200** | **24.08** | **0.926** | **0.01176** |
| e21 best acc | **95.4%** | 2300 | **23.85** | — | — |
| e21 best bits | **93.5%** | 3000 | **24.13** | — | — |
| e18_local | **24.6%** | 3200 | **0.00** | **0** | **0** |

Dense left chance ~200, **64.9% / 11.06 bits @800**, crossed 75% at
**1100**, **93.8% / 23.92 bits** @3200 (best acc 93.8% @2900; **S0
PASS**). JSON `calibrated: true`. Hunt exit 0.

This-JSON E18 left chance ~400, **40.3% / 5.33 bits @800**, plateaued
**47.1% / 9.75 bits** @3200 (best acc 51.1% @2600; live, not ≈ 0). Do
**not** treat 9.75 as the n_dist=1 uncompressed ceiling — the severed
hunt's E18 was **99.0% / 25.44 @2450**. Score S1 vs **0.75× this-JSON
E18 7.31**, vs **0.75× dense 17.94**, and vs **0.75× prior live E18
19.08**. Do **not** pass via 0.75×0 — E18 is live.

E21 left chance ~500, **60.7% / 6.85 bits @800** (not chance floor),
crossed 75% at **2200**, clicked **95.4% / 23.85 bits @2300**, **93.6% /
24.08 bits** @3200 (best bits 24.13 @3000; **S1 PASS** vs 0.75×
this-JSON E18 7.31, vs 0.75× dense 17.94, and vs 0.75× prior live E18
19.08). `e18_local` at chance (**K2 PASS**).

Do **not** relabel severed n_dist=1 E21 **6.28**, keepswa **14.77**,
n_dist=0 **16.60**, ordered hops glob=2 **25.85**, or hops 272 extra hop
**25.48** as this score. Do not relabel this-JSON E18 9.75 as an E21
score.

**8k not run** (S1 PASS at the 800 advertised / 3200 matched budget). Do
not 16k.

One plot: learning curves — dense and extra-hop E21 both solve; this-JSON
E18 plateaus ~47–51%; `e18_local` stays on the chance line.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** (dense ≥ 75%) | **PASS.** Dense 93.8% / 23.92 bits @3200. Recalibrated at `global_layers=2`. |
| **S1** (compressed read ≥ 75% of the raw-read control) | **PASS.** 24.08 > 0.75× this-JSON E18 7.31 bits. |
| **S1 vs 0.75× dense** | **PASS.** 24.08 > 17.94 bits. |
| **S1 vs 0.75× prior live E18** | **PASS.** 24.08 > 19.08 bits. |
| **S2** plots | **PASS** (curves + bits/flow/bytes-per-token). |
| **K1** | not triggered. |
| **K2** | **PASS.** `e18_local` 24.6% / 0 bits @3200. |
| **K3** | not triggered at 800 (E21 60.7% / 6.85 bits, climbing). |
| **8k** | **not run.** S1 PASS. Do not 16k. |

## Interpretation

**An extra exclusive hop over frozen slots composes the extra shuffled
hop-edge without restoring SWA.** Same compressor, two global Blocks,
identity slots, 26-bit prize, n_dist=1, keepswa **off**: severed E21 was
**0 @800 / 6.28 @3200** vs live E18 **25.44**; keepswa E21 was **14.77
@8000** vs live E18 **20.95** (S1 FAIL); extra-hop E21 is **6.85 @800 /
24.08 @3200** vs this-JSON E18 **9.75** and vs 0.75× dense **17.94**.
Hops 272 extra hop **S1 PASS 25.48** transfers to this shuffled n_dist=1
cell. Keepswa **S1 FAIL 14.77** does not.

SWA stays severed (`msg_keepswa=False`). Exclusive slots stay on the
global read (`msg_rawkv=False`). Extra hop is in-attention over frozen
slot K/V (`msg_extrahops=1`, `msg_updatekv=False`). Window 16 still
cannot span gap 65–92. `e18_local` stays at chance (K2) — extra hop did
not leak a solvable local-only channel.

Severed and keepswa shuffled walls stay **(n_dist=0 S1 PASS, n_dist=1 S1
FAIL]**. Extra-hop shuffled n_dist=1 is **S1 PASS**. Ordered hops walls
stay as mapped (severed **(264 PASS, 272 FAIL]**; keepswa **(272 PASS,
288 K1]**; extra hop **(272 PASS, 288 FAIL]**).

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not hops
320. Do not glob=3. Do not stack `--message_keep_local_swa` on this
extra-hop JSON. Do not n_dist=1.5 / same recipe again. Do not MATCH2
length extra-steps. Do not SELECT 694. Do not INDEX 1280 extra-steps.
Not Glyph. Do not unfreeze `u`/`delta`. Do not restore full raw prefix
KV (that is E18). Code default `--message_extra_slot_attends` stays
**0**. Code default `--message_keep_local_swa` stays **false**. Code
default `--global_layers` stays **1**. Next ONE (do not run): **256
`--recipe chain` n_dist=2 `--key_len 13 --global_layers 2
--message_extra_slot_attends 1` H=256 log identity** (keepswa off; do not
stack).

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
