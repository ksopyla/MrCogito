# E25 bridge seq=256 `--recipe chain --n_distractors 1` --key_len 13 `--global_layers 2 --message_keep_local_swa` — E21 S1 FAIL vs live E18 (rung 5ca)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_256_chain_shuf1_k13_glob2_keepswa` (800 advertised; dense early-stop 3000; e18 / e21 / `e18_local` scored to 3000) · `e25_256_chain_shuf1_k13_glob2_keepswa_s8k` (8k extra-step; `--no-dense_first --k1_mult 1`; S1 FAIL)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge256_chain_shuf1_k13_glob2_keepswa/` · `/opt/cursor/artifacts/e25_bridge256_chain_shuf1_k13_glob2_keepswa_s8k/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge256_chain_shuf1_k13_glob2_keepswa/probe.log` · `/opt/cursor/artifacts/e25_bridge256_chain_shuf1_k13_glob2_keepswa_s8k/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `90af928` (Odra E24 worktree HEAD at launch; `nn/` unchanged)
**Git tag:** —
**Related:** 256 n_dist=1 shuffled chain glob=2 severed S1 FAIL [`e25_bridge256_chain_shuf1_k13_glob2_20260915.md`](e25_bridge256_chain_shuf1_k13_glob2_20260915.md) · 256 n_dist=0 shuffled chain glob=2 S1 PASS [`e25_bridge256_chain_shuf0_k13_glob2_20260915.md`](e25_bridge256_chain_shuf0_k13_glob2_20260915.md) · 272 ordered hops keepswa S1 PASS [`e25_bridge272_chain_k13_glob2_keepswa_20260915.md`](e25_bridge272_chain_k13_glob2_keepswa_20260915.md) · 696 SELECT keepswa S1 FAIL [`e25_bridge1k_696_ip_id_h256_keepswa_select_20260915.md`](e25_bridge1k_696_ip_id_h256_keepswa_select_20260915.md)

---

## Goal

Architectural bet at the measured n_dist=1 shuffled exclusive wall, where
uncompressed E18 **solves** (25.44 bits) and severed E21 **fails** (0 @800 /
6.28 @3200). Same seq=256 `--recipe chain --n_distractors 1` `--key_len 13`
`--global_layers 2` H=256 identity recipe; **one** knob:
`--message_keep_local_swa`. Falsifiable claim: SWA sever vs exclusive slot
composition of the extra shuffled hop-edge. Ordered hops 272: keepswa rescued
~25.5 bits (S1 PASS). SELECT 696 keepswa did not. Do **not** stack
`--message_extra_slot_attends`. Do not glob=3. Do not n_dist extra-steps.

Do **not** relabel severed n_dist=1 E21 6.28, n_dist=0 16.60, ordered hops
25.85, or hops 272 keepswa 25.55 as this score. If E18 ≈ 0, score vs 0.75×
dense (do not pass via 0.75×0). Chance at 800 → no 8k. Climbing short of S1
→ extra-step 8k only then. S1 PASS → no 8k. Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / `e18_local` |
| Width | **H=256** · **2.802M** (e21 **2.835M**) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=True`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 256 --recipe chain`, **`--n_distractors 1`**, **`--key_len 13`**, hops=2, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/65/92 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.1GB, 0.04 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_256_chain_shuf1_k13_glob2_keepswa 0 \
  --scale bridge --seq_len 256 --recipe chain --n_distractors 1 --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --message_keep_local_swa \
  --steps 800 --k1_mult 4
# named recipe chain + --n_distractors 1 = shuffled REACHABILITY with one extra edge
# --global_layers 2 required (hops); default glob stays 1
# --message_keep_local_swa required (this bet); exclusive slots stay on the global read
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_extra_slot_attends (default 0; do not stack)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused on the 800 hunt)
# E21 climbing short of S1 @3000 vs 0.75× dense → 8k extra-step:
bash scripts/e24_bapo_hunt.sh e25_256_chain_shuf1_k13_glob2_keepswa_s8k 0 \
  --scale bridge --seq_len 256 --recipe chain --n_distractors 1 --arch e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --message_keep_local_swa \
  --steps 8000 --k1_mult 1 --no-dense_first
# dense already S0 PASS 99.2% / 25.56 in the 800 JSON; underscore --no-dense_first
# do not 16k (S1 FAIL after 8k)
```

Byobu `E25_256_chain_shuf1_keepswa` then `E25_256_chain_shuf1_keepswa_s8k`. Log:
`seq=256  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=65/65/92`
dense `patterns=[('full', 0)×5]  glob_layers=2` · 2.802M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_boundary=10  msg_r=1  msg_remainder=False  msg_inplace=True
msg_rawkv=False  msg_idslots=True  msg_keepswa=True  glob_layers=2
msg_extrahops=0  msg_updatekv=False  msg_anchors=none`.

800 hunt JSON: `seq_len: 256`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 2`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
**`message_keep_local_swa: true`**, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`n_distractors: 1`, `hops: 2`, `prize_bits: 26.0`, `calibrated: true`,
`dense_steps_used: 3000`. Hunt exit **0**. Params <100M. Dense S0
recalibrated at two full layers (2.802M). Seed-0 row `meta.shuffled=True`.
Code defaults unchanged (`--global_layers` stays 1 except hops hunts;
`--message_keep_local_swa` stays false).

## Training Outcome

800 JSON (dense-first; dense early-stop 99.2% @3000):

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense @800 | **84.3%** | 800 | **20.18** | — | — |
| dense | **99.2%** | 3000 | **25.56** | **0.983** | **0.01248** |
| e18 @800 | **26.1%** | 800 | **0.01** | ~0 | — |
| e18 | **48.0%** | 3000 | **9.57** | **0.368** | **0.00468** |
| e18 best acc | **50.5%** | — | — | — | — |
| **e21 @800** | **40.5%** | **800** | **1.67** | **0.064** | — |
| **e21** | **45.8%** | **3000** | **6.52** | **0.251** | **0.00318** |
| e21 best bits | **45.0%** | 2800 | **6.65** | — | — |
| e18_local | **23.2%** | 3000 | **0.00** | **0** | **0** |

8k extra-step (`--no-dense_first`; dense already S0):

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| e18 @800 | **38.9%** | 800 | **2.37** | — | — |
| e18 | **80.8%** | 8000 | **20.95** | **0.806** | **0.01023** |
| e18 best acc | **82.5%** | 7950 | **20.97** | — | — |
| **e21 @800** | **37.0%** | **800** | **4.81** | — | — |
| e21 @3000 | **42.9%** | 3000 | **8.13** | — | — |
| e21 @5450 | **41.2%** | 5450 | **8.15** | plateau | — |
| **e21** | **63.8%** | **8000** | **14.77** | **0.568** | **0.00721** |
| e21 best bits | **61.9%** | 7450 | **14.93** | — | — |
| e18_local | **26.1%** | 8000 | **0.00** | **0** | **0** |

Dense left chance ~200, **84.3% / 20.18 bits @800**, crossed 75% at **450**,
early-stopped **99.2% / 25.56 bits** @3000 (**S0 PASS**). JSON
`calibrated: true`. Hunt exit 0.

800-JSON E18 stayed near chance through 800 (**26.1% / 0.01 bits**), left
chance ~1150, **48.0% / 9.57 bits** @3000 (best acc 50.5%; live, climbing).
Do **not** treat this-JSON E18 9.57 as the n_dist=1 uncompressed ceiling —
the prior severed hunt's E18 was **99.0% / 25.44 @2450**. Score the 800 JSON
vs 0.75× dense **19.17** as well as vs 0.75× this-JSON E18 **7.18**.

800-JSON E21 left chance ~700, **40.5% / 1.67 bits @800** (not chance floor),
**45.8% / 6.52 bits** @3000 (best bits 6.65 @2800; **S1 FAIL** vs 0.75×
this-JSON E18 7.18 and vs 0.75× dense 19.17). SOP: climbing short of S1 →
extra-step 8k. `e18_local` at chance (**K2 PASS**).

8k E18 left chance ~700, crossed 75% at **2600**, **80.8% / 20.95 bits**
@8000 (best 82.5% / 20.97 @7950; live). Score S1 vs **0.75× live E18 15.71
bits**. Do **not** pass via 0.75×0 — E18 is live.

8k E21 left chance ~700, **37.0% / 4.81 bits @800**, plateaued ~8.1 bits
through 5450, then climbed from ~6000, **63.8% / 14.77 bits** @8000 (best
14.93 @7450; **S1 FAIL** vs 0.75× live E18 15.71 by **0.94 bits**; also
**FAIL** vs 0.75× dense 19.17). `e18_local` at chance (**K2 PASS**). Hunt
exit 0.

Do **not** relabel severed n_dist=1 E21 **6.28**, n_dist=0 **16.60**, ordered
hops glob=2 **25.85**, or hops 272 keepswa **25.55** as this score. Do not
relabel 8k E18 20.95 as an E21 score.

**8k ran** (E21 climbing short of S1 at the 800 advertised / 3000 matched
budget). Do not 16k.

Two plots: 800-JSON learning curves (dense solves; E18/E21 plateau ~45–50%)
and 8k curves (E18 climbs to ~81%; E21 plateau ~40% until ~6k then ~64%,
still short of E18).

## Gates vs this rung

| gate | result |
|---|---|
| **S0** (dense ≥ 75%) | **PASS.** Dense 99.2% / 25.56 bits @3000. Recalibrated at `global_layers=2`. |
| **S1** (compressed read ≥ 75% of the raw-read control) | **FAIL.** 14.77 < 0.75× live E18 15.71 bits (8k JSON). |
| **S1 vs 0.75× dense** | **FAIL.** 14.77 < 19.17 bits. |
| **S2** plots | **PASS** (curves + bits/flow/bytes-per-token). |
| **K1** | not triggered. |
| **K2** | **PASS.** `e18_local` 23.2% / 0 bits @3000 and 26.1% / 0 bits @8000. |
| **K3** | not triggered at 800 (E21 40.5% / 1.67 bits, climbing). |
| **8k** | **ran.** S1 still FAIL. Do not 16k. |

## Interpretation

**Unsevering SWA at QUERY does not make exclusive slots compose the extra
shuffled hop-edge.** Same compressor, two global Blocks, identity slots,
26-bit prize, n_dist=1: severed E21 was **0 @800 / 6.28 @3200** vs live E18
**25.44**; keepswa E21 is **1.67 @800 / 6.52 @3000 / 14.77 @8000** vs live
E18 **20.95**. Keepswa lifts the 800 floor (not chance) and adds ~8 bits by
8k versus the severed 6.28, but stays **0.94 bits short of S1**. Hops 272
keepswa **S1 PASS 25.55** does not transfer. SELECT 696 keepswa stays 0.01.

The n_dist=1 wall is **exclusive slot composition of the extra edge**, not
SWA sever. Local SWA still sees across QUERY (`msg_keepswa=True`). Exclusive
slots stay on the global read (`msg_rawkv=False`). Window 16 still cannot
span gap 65–92. `e18_local` stays at chance (K2) — keep_local_swa did not
leak a solvable local-only channel.

Shuffled distractor wall stays **(n_dist=0 S1 PASS, n_dist=1 S1 FAIL]** even
unsevered. Ordered hops walls stay as mapped (severed **(264 PASS, 272
FAIL]**; keepswa **(272 PASS, 288 K1]**; extra hop **(272 PASS, 288 FAIL]**).

## Decision

Keep the spec in `ahead/`. **8k ran.** Do **not** 16k. Do not hops 320. Do
not glob=3. Do not stack `--message_extra_slot_attends` on this keepswa
JSON. Do not n_dist=1.5 / n_dist=3 / same recipe again. Do not MATCH2 length
extra-steps. Do not SELECT 694. Do not INDEX 1280 extra-steps. Not Glyph. Do
not unfreeze `u`/`delta`. Do not restore full raw prefix KV (that is E18).
Code default `--message_keep_local_swa` stays **false**. Code default
`--global_layers` stays **1**. Next ONE (do not run): **256 `--recipe chain
--n_distractors 1` `--key_len 13 --global_layers 2 --message_extra_slot_attends
1` H=256 log identity** (keepswa off; the other knob at this exclusive wall).

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
