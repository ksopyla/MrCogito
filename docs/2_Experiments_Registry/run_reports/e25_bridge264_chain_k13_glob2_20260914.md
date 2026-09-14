# E25 bridge seq=264 chain_ordered --key_len 13 `--global_layers 2` — E21 hops S1 PASS @8k (rung 5bf)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_264_chain_k13_glob2` (800 advertised; dense-matched 3200; climbing short of S1 vs 0.75× dense) · `e25_264_chain_k13_glob2_s8k` (8k extra-step; `--no-dense_first --k1_mult 1`; S1 PASS)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge264_chain_k13_glob2/` · `/opt/cursor/artifacts/e25_bridge264_chain_k13_glob2_s8k/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge264_chain_k13_glob2/probe.log` · `/opt/cursor/artifacts/e25_bridge264_chain_k13_glob2_s8k/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `53f567e` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** seq=256 glob2 S1 PASS [`e25_bridge256_chain_k13_glob2_20260914.md`](e25_bridge256_chain_k13_glob2_20260914.md) · seq=272 glob2 E21 FAIL [`e25_bridge272_chain_k13_glob2_20260914.md`](e25_bridge272_chain_k13_glob2_20260914.md)

---

## Goal

Measured wall: seq=256 packed `chain_ordered --key_len 13` hops=2, H=256 SSMax
log, **two** global layers — E21 **S1 PASS 25.85**; live E18 **5.56 bits**. Same
recipe at seq=272: dense **S0 PASS 25.33**, E18 live **25.35 / 23.64**, E21
**8.87 @2850 then 0.02 @8k** (S1 FAIL). Hypothesis: packed 264 (midpoint of
**(256 S1 PASS, 272 FAIL]**) lets exclusive identity compose hops. Same DNA +
compressor as 256/272 glob2, **only** `--seq_len 264` (bridge default is 512).
Recalibrate dense S0 at two global Blocks in the 800 JSON. Score e18/e21 if
dense S0 PASSes.

800 floor: chance → no 8k; climbing short of S1 → 8k only then. Do not 16k.
`--hops 1` is illegal. Do not relabel E18 as E21. If E18 ~0, S1 vs 0.75× dense.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · **2.802M** (e21 **2.835M**) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 264`, **`--key_len 13`**, hops=2, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. Right-align gap 157/164/179 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.2GB, 0.04–0.05 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_264_chain_k13_glob2 0 \
  --scale bridge --seq_len 264 --recipe chain_ordered --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --steps 800 --k1_mult 4
# --seq_len 264 required (bridge default is 512)
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# no --message_global_anchors (default none)
# dense first (underscore --no-dense_first unused on the 800 hunt)
# climbing short of S1 @3200 vs 0.75× dense → 8k extra-step:
bash scripts/e24_bapo_hunt.sh e25_264_chain_k13_glob2_s8k 0 \
  --scale bridge --seq_len 264 --recipe chain_ordered --arch e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --steps 8000 --k1_mult 1 \
  --no-dense_first
# dense already S0 PASS 82.3% / 17.65 in the 800 JSON; underscore --no-dense_first
# do not 16k (S1 PASS)
```

Byobu `E25_264_chain_glob2` then `E25_264_chain_glob2_s8k`. Log:
`seq=264  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=157/164/179`
dense `patterns=[('full', 0)×5]  glob_layers=2` · 2.802M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False
msg_idslots=True  msg_keepswa=False  glob_layers=2  msg_extrahops=0
msg_updatekv=False  msg_anchors=none`.

800 hunt JSON: `seq_len: 264`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 2`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`hops: 2`, `prize_bits: 26.0`, `calibrated: true`, `dense_steps_used: 3200`.
Hunt exit **0**. 8k JSON same knobs, `steps: 8000`, `k1_mult: 1`, no dense arm,
`calibrated: true`, hunt exit **0**. Params <100M. Dense S0 recalibrated at two
full layers (2.802M).

## Training Outcome

### 800 advertised / dense-matched 3200 (`e25_264_chain_k13_glob2`, exit 0)

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **82.3%** | 3200 | **17.65** | **0.679** |
| e18 | **26.2%** | 3200 | **0.01** | **~0** |
| **e21 @800** | **25.1%** | 800 | **0.00** | **~0** |
| **e21 final** | **49.3%** | 3200 | **11.00** | **0.423** |
| e18_local | **25.1%** | 3200 | **0.00** | **0** |

Dense crossed 75% at 2900 (75.5%), final **82.3% / 17.65 bits** @3200
(best = final; no 99% early-stop). JSON `calibrated: true`. Hunt exit 0.
**S0 PASS.** Recalibrated at `global_layers=2`.

E18 is **~0** in this JSON: chance every eval (best 28.0% @1650; CE glued to
ln(4)). Do **not** pass S1 via 0.75×0. Score S1 vs **0.75× dense 13.24 bits**.

E21 sat at chance through the advertised 800 floor (25.1% / CE 1.387).
Dense-matched budget continued to 3200. Climb started ~1200 (28.6% / 0.68
bits), **47.0% / 9.72 bits** @2850, best **49.8% @3100**, final **49.3% /
11.00 bits** @3200 (flow 0.423). S1 vs 0.75× dense **13.24 bits**: **FAIL**
(11.00). Climbing, short of S1 → 8k extra-step.

### 8k extra-step (`e25_264_chain_k13_glob2_s8k`, exit 0)

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| e18 | **48.0%** | 8000 | **9.87** | **0.380** |
| **e21** | **99.0%** | 4050 | **25.61** | **0.985** |
| e18_local | **24.6%** | 8000 | **0.00** | **0** |

Dense skipped (`--no-dense_first`; prior S0 **82.3% / 17.65 bits**). E18 is
**live, not ~0** in this JSON: chance through ~2400, first 30% @2800, 44.6%
@4800, best **52.0% @6950**, final **48.0% / 9.87 bits** @8000. Do not relabel
E18 as E21. S1 vs **0.75× this-JSON live E18 7.40 bits**.

E21 sat at chance through 800 (25.1% / CE 1.387), then climbed: 48.0% / 8.91
bits @2000, first ≥75% **75.7% @2350**, 88.7% / 20.47 bits @2400, 95.8% /
24.16 bits @3200, early-stop **99.0% / 25.61 bits** @4050 (flow 0.985). S1 vs
0.75× live E18 **7.40 bits**: **PASS** (25.61). Content vs 0.75× 800-JSON dense
**13.24 bits**: **PASS** (25.61). Exclusive identity even exceeds the 800 dense
control (17.65). The 800-budget 11.00-bit climb **did replicate and finished**.

**8k ran.** Do **not** 16k (S1 already PASS). Do not relabel the live E18 arm as
E21.

## Gates vs this rung

| gate | 800 / 3200 | 8k |
|---|---|---|
| **S0** | **PASS.** Dense 82.3% / 17.65 bits @3200 ≥ 75%. Recalibrated at `global_layers=2`. | dense skipped; prior S0 stands. |
| **S1 vs 0.75× E18 / dense** | **FAIL.** E18 ~0 so vs 0.75× dense 13.24: 11.00 < 13.24 (climbing). | **PASS.** 25.61 ≥ 0.75× live E18 7.40. Also ≥ 0.75× dense 13.24. E18 is live (9.87), not ~0. |
| **S2** plots | **PASS** (learning-curves / heatmap / recovered-bits / flow / bytes-per-token). | **PASS** (same). |
| **K1** | not triggered. | not triggered. |
| **K2** | **PASS.** `e18_local` 25.1% / 0 bits. | **PASS.** `e18_local` 24.6% / 0 bits. |
| **K3** | not floor at 3200 (flow 0.423). | not triggered (flow 0.985). |

## Interpretation

**Packed 264 hops is dense-solvable at two global Blocks; exclusive identity
composes hops once the 8k replica is the score.** The glob=2 recipe that
composed hops for E21 at seq=256 (25.85 bits @1650) is slow at 264 — chance
through 800, 11.00 bits @3200 short of 0.75× dense — then early-stops at
**99.0% / 25.61 bits @4050**. Seq=272 on the same recipe was an unreplicated
2850 climb that died at 8k chance. This is an E21 exclusive-channel hops
**PASS** at 264 and a **FAIL** at 272, not a shared 288-style 0/0 floor.

800-JSON E18 was ~0; 8k-JSON E18 is live **9.87 bits**. Do not pass S1 via
0.75×0 on the 800 JSON; the 8k score uses live E18. Do not relabel E18's 9.87
as E21.

E21 hops wall at glob=2 is now **(264 S1 PASS, 272 FAIL]**. Dense hops wall
is unchanged **(288 S0 PASS, 320 K1]**. E18 glob=2 hops is live at 264 (8k)
and 272, dead at 288.

## Decision

Keep the spec in `ahead/`. **8k ran.** Do **not** 16k (S1 already PASS). Do
not hops seq shrink below 256. Do not Glyph. Do not unfreeze `u`/`delta`. Do
not restore full raw prefix KV (that is E18). Do not reopen SELECT
**(692, 696]**. Do not INDEX extra-steps. Code default `--global_layers`
stays 1. Next ONE (do not run): seq=**268** packed `chain_ordered --key_len
13` `--global_layers 2` (tighten the E21 hops wall **(264 S1 PASS, 272
FAIL]**). Recalibrate dense S0. `--hops 1` is illegal.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
