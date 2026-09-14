# E25 bridge seq=272 chain_ordered --key_len 13 `--global_layers 2` — E21 hops S1 FAIL, 8k chance (rung 5be)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_272_chain_k13_glob2` (800 advertised; dense-matched 2850; climbing short of S1) · `e25_272_chain_k13_glob2_s8k` (8k extra-step; `--no-dense_first --k1_mult 1`; chance floor)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2/` · `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2_s8k/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2/probe.log` · `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2_s8k/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `f1f85bf` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** seq=256 glob2 S1 PASS [`e25_bridge256_chain_k13_glob2_20260914.md`](e25_bridge256_chain_k13_glob2_20260914.md) · seq=288 glob2 E21 FAIL [`e25_bridge288_chain_k13_glob2_20260914.md`](e25_bridge288_chain_k13_glob2_20260914.md)

---

## Goal

Measured wall: seq=256 packed `chain_ordered --key_len 13` hops=2, H=256 SSMax
log, **two** global layers — E21 **S1 PASS 25.85**; live E18 **5.56 bits**. Same
recipe at seq=288: dense **S0 PASS 22.20**, E18 **0**, E21 **0** (chance @800
and @3200). Hypothesis: packed 272 (midpoint of **(256 S1 PASS, 288 FAIL]**)
lets exclusive identity compose hops. Same DNA + compressor as 256/288 glob2,
**only** `--seq_len 272` (bridge default is 512). Recalibrate dense S0 at two
global Blocks in the 800 JSON. Score e18/e21 if dense S0 PASSes.

800 floor: chance → no 8k; climbing short of S1 → 8k only then. Do not 16k.
`--hops 1` is illegal. Do not relabel E18 as E21.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · **2.802M** (e21 **2.835M**) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 272`, **`--key_len 13`**, hops=2, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. Right-align gap 163/176/187 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.2GB, 0.04–0.05 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_272_chain_k13_glob2 0 \
  --scale bridge --seq_len 272 --recipe chain_ordered --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --steps 800 --k1_mult 4
# --seq_len 272 required (bridge default is 512)
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# no --message_global_anchors (default none)
# dense first (underscore --no-dense_first unused on the 800 hunt)
# climbing short of S1 @2850 → 8k extra-step:
bash scripts/e24_bapo_hunt.sh e25_272_chain_k13_glob2_s8k 0 \
  --scale bridge --seq_len 272 --recipe chain_ordered --arch e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --steps 8000 --k1_mult 1 \
  --no-dense_first
# dense already S0 PASS 99.0% / 25.33 in the 800 JSON; underscore --no-dense_first
# do not 16k (8k chance floor)
```

Byobu `E25_272_chain_glob2` then `E25_272_chain_glob2_s8k`. Log:
`seq=272  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=163/176/187`
dense `patterns=[('full', 0)×5]  glob_layers=2` · 2.802M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False
msg_idslots=True  msg_keepswa=False  glob_layers=2  msg_extrahops=0
msg_updatekv=False  msg_anchors=none`.

800 hunt JSON: `seq_len: 272`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 2`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`hops: 2`, `prize_bits: 26.0`, `calibrated: true`, `dense_steps_used: 2850`.
Hunt exit **0**. 8k JSON same knobs, `steps: 8000`, `k1_mult: 1`, no dense arm,
`calibrated: true`, hunt exit **0**. Params <100M. Dense S0 recalibrated at two
full layers (2.802M).

## Training Outcome

### 800 advertised / dense-matched 2850 (`e25_272_chain_k13_glob2`, exit 0)

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **99.0%** | 2850 | **25.33** | **0.974** |
| e18 | **99.0%** | 1650 | **25.35** | **0.975** |
| **e21 @800** | **27.6%** | 800 | **0.00** | **~0** |
| **e21 final** | **46.2%** | 2850 | **8.87** | **0.341** |
| e18_local | **22.5%** | 2850 | **0.00** | **0** |

Dense crossed 75% at 2250 (78.6%), early-stop **99.0% / 25.33 bits** @2850
(best = final). JSON `calibrated: true`. Hunt exit 0.

E18 is **live, not ~0**: 27.3% @800, click 41.6% @1300, 76.4% @1550, early-stop
**99.0% / 25.35 bits** @1650. Do **not** pass S1 via 0.75×0. Uncompressed
one-read composes hops at 272 (it did not at 288).

E21 sat at chance through the advertised 800 floor (27.6% / CE 1.386).
Dense-matched budget continued to 2850. Climb started ~1450 (28.5%), best
**51.3% @2750**, final **46.2% / 8.87 bits** @2850 (flow 0.341). S1 vs 0.75×
live E18 **19.02 bits**: **FAIL** (8.87). Content vs 0.75× dense **19.00**:
**FAIL**. Climbing, short of S1 → 8k extra-step.

### 8k extra-step (`e25_272_chain_k13_glob2_s8k`, exit 0)

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| e18 | **95.2%** | 8000 | **23.64** | **0.909** |
| **e21** | **27.6%** | 8000 | **0.02** | **0.001** |
| e18_local | **22.5%** | 8000 | **0.00** | **0** |

Dense skipped (`--no-dense_first`; prior S0 **99.0% / 25.33 bits**). E18 live
in this JSON too: chance through 2500, first 40% @4300, 75.8% @5100, final
**95.2% / 23.64 bits** @8000 (best 95.4% @7900). Slower click than the 800
JSON (advertised budget is the LR horizon). Do not relabel E18 as E21.

E21 sat at chance **every** eval (CE glued to ln(4); best acc 27.6% = noise
around 25%; min CE 1.384 @350). Final **27.6% / 0.02 bits**. The 2850-budget
climb **did not replicate** (same class as 696 pack_stride 32: 800 climb, 8k
floor). S1 vs 0.75× this-JSON live E18 **17.73 bits**: **FAIL** (0.02). vs
0.75× 800-JSON dense **19.00**: **FAIL**.

**8k ran.** Do **not** 16k (chance floor). Do not relabel the live E18 arm as
E21.

## Gates vs this rung

| gate | 800 / 2850 | 8k |
|---|---|---|
| **S0** | **PASS.** Dense 99.0% / 25.33 bits @2850 ≥ 75%. Recalibrated at `global_layers=2`. | dense skipped; prior S0 stands. |
| **S1 vs 0.75× live E18** | **FAIL.** 8.87 < 19.02 bits (climbing). E18 is live (25.35), not ~0. | **FAIL.** 0.02 ≪ 17.73 bits (chance floor). E18 live 23.64. |
| **content vs 0.75× dense** | **FAIL.** 8.87 < 19.00 bits. | **FAIL.** 0.02 ≪ 19.00 (800-JSON dense). |
| **S2** plots | **PASS** (learning-curves / heatmap / recovered-bits / flow / bytes-per-token). | **PASS** (same). |
| **K1** | not triggered. | not triggered. |
| **K2** | **PASS.** `e18_local` 22.5% / 0 bits. | **PASS.** `e18_local` 22.5% / 0 bits. |
| **K3** | not floor at 2850 (flow 0.341). | **triggered** at 8k (flow 0.001). |

## Interpretation

**Packed 272 hops is dense-solvable and E18-solvable at two global Blocks;
exclusive identity is not, once the 8k replica is the score.** The glob=2
recipe that composed hops for E21 at seq=256 (25.85 bits) yields a
dense-matched climb to 8.87 bits at 272 that dies at chance on the 8k
re-run. Uncompressed E18 recovers **25.35 / 23.64 bits** in the two JSONs —
this is an E21 exclusive-channel hops miss, not a shared 288-style 0/0
floor. Do not pass S1 via 0.75×0; E18 is live.

E21 hops wall at glob=2 is now **(256 S1 PASS, 272 FAIL]**. Dense hops wall
is unchanged **(288 S0 PASS, 320 K1]**. E18 glob=2 hops is live at 272 and
dead at 288.

## Decision

Keep the spec in `ahead/`. **8k ran.** Do **not** 16k (chance floor). Do not
hops seq shrink below 256. Do not Glyph. Do not unfreeze `u`/`delta`. Do not
restore full raw prefix KV (that is E18). Do not reopen SELECT
**(692, 696]**. Do not INDEX extra-steps. Code default `--global_layers`
stays 1. Next ONE (do not run): seq=**264** packed `chain_ordered --key_len
13` `--global_layers 2` (tighten the E21 hops wall **(256 S1 PASS, 272
FAIL]**). Recalibrate dense S0. `--hops 1` is illegal.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
