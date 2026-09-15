# E25 bridge_1k seq=696 select_1decoy r=1 identity `--message_keep_local_swa` H=256 SSMax log — E21 vs E18 vs dense (rung 5bo)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_696_ip_id_h256_keepswa_select` (800; floor — do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_696_ip_id_h256_keepswa_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_696_ip_id_h256_keepswa_select/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `cbe3d39` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** seq=696 SELECT identity chance [`e25_bridge1k_696_ip_id_h256_select_20260914.md`](e25_bridge1k_696_ip_id_h256_select_20260914.md) · seq=692 SELECT identity PASS [`e25_bridge1k_692_ip_id_h256_select_20260914.md`](e25_bridge1k_692_ip_id_h256_select_20260914.md) · seq=272 hops keep_local_swa S1 PASS [`e25_bridge272_chain_k13_glob2_keepswa_20260915.md`](e25_bridge272_chain_k13_glob2_keepswa_20260915.md) · 1024 SELECT keep_local_swa chance [`e25_bridge1k_ip_id_keepswa_select_20260914.md`](e25_bridge1k_ip_id_keepswa_select_20260914.md)

---

## Goal

Hops length extra-steps are stopped. SELECT wall with severed SWA is
**(692 PASS, 696 FAIL]** (E21 0 @800; live E18 ~64). 1024 SELECT
`--message_keep_local_swa` was 0.01 bits — far past the wall. Hops seq=272
glob=2 keep_local_swa **S1 PASS 25.55** after severed-SWA FAIL — keepswa at
the measured wall rescued hops.

Hypothesis: the 4-token SELECT cliff is **SWA sever** (local type/QUERY
mixing), not exclusive global capacity. Test the same unsever **at the
SELECT cliff** (seq=696), not at 1024. If E21 S1 PASS, SELECT 696 is
hops-class SWA sever. If chance, exclusive SELECT still dies at 696 even
with local SWA (stronger than 1024 keepswa fail).

Same 696 identity recipe plus `--message_keep_local_swa`. Default
keep_local_swa stays **false** in code. Recalibrate dense S0. Score vs
0.75× live E18; if E18 were ~0, score vs 0.75× dense. Climbing short of
S1 → extra-step 8k only; **chance at 800 → do not extra-step**. Do not
16k. Do not seq=694. Do not pack_stride / spread / window 32. Remainder
off. `--global_layers` 1. Window default 16 < gap 64. Extra hops 0.
`update_slot_kv` off. Anchors none.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=True`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`glob_layers=1`**, **`msg_anchors=none`**, **`msg_packstride=0`** |
| Data | `--scale bridge_1k --seq_len 696`, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~2.1GB, 0.06 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_696_ip_id_h256_keepswa_select 0 \
  --scale bridge_1k --seq_len 696 --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --message_keep_local_swa --steps 800 --k1_mult 4
# --seq_len 696 and --message_keep_local_swa required
# no --global_layers (default 1)
# no --message_pool_remainder
# no --message_pack_stride
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# no --message_global_anchors (default none)
# no --local_window (default 16 < gap 64)
# dense first (underscore --no-dense_first unused)
# 8k extra-step NOT run (chance floor @800)
```

Byobu `E25_696_id_keepswa`. Log:
`seq=696  gap=64  window=16  prize=64.00 bits  answer_len=32
row_gap[min/med/max]=100/100/100  logit_scale=log
msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False
msg_idslots=True  msg_keepswa=True  glob_layers=1  msg_extrahops=0
msg_updatekv=False  msg_anchors=none  msg_packstride=0`.

Hunt JSON: `seq_len: 696`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, `global_layers: 1`, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
**`message_keep_local_swa: true`**, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`message_pack_stride: 0`, `message_override: real`, `prize_bits: 64.0`,
`calibrated: true`, `dense_steps_used: 200`. Hunt exit **0**. Params <100M.
Dense S0 recalibrated in the same JSON. Code default `--message_keep_local_swa`
stays **false**.

## Training Outcome

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **100%** | 200 | **63.92** | 0.999 | 0.01148 |
| e18 | **100%** | 350 | **63.91** | 0.999 | 0.01148 |
| **e21 @800** | **25.0%** | 800 | **0.01** | **0.000** | **1.67e-6** |
| e18_local | 24.7% | 800 | 0.00 | 0.000 | 0 |

Params: dense/e18 ~2.261M, e21 2.277M (<100M). Prize 64 bits. Chance ~25%.
E21 CE glued to ln(4)≈1.386 every eval (min 1.3854 @500, max 1.397; acc
never > 0.263; best 26.3% @750). Not climbing. E18 live **100% / 63.91
bits** @350 (do not pass S1 via 0.75×0). Dense early-stop 100% @200.
Hunt exit 0. **8k extra-step did not run.**

S1 vs 0.75× this-JSON live E18: need **47.93 bits**. E21 has **0.01**.
Content vs 0.75× dense: need **47.94 bits**. E21 has **0.01**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.92 bits @200. Recalibrated. |
| **S1 vs 0.75× this-JSON live E18** | **FAIL.** 0.01 ≪ 47.93 bits. Chance floor. |
| **content vs 0.75× dense** | **FAIL.** 0.01 ≪ 47.94 bits. |
| **S2** plots | **PASS.** traces + `learning_curves` / `recovered_bits` / `information_flow` in hunt artifacts. |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.7% / 0.00 bits. keep_local_swa did not leak a full raw prefix. |
| **K3** | floor at 800 (flow 0.000). |

Do **not** extra-step to 8k (floor). Do not 16k. Do not seq=694. Do not
pack_stride / spread / window 32. Do not remainder-on. Do not restore raw
global KV. Do not extra hops / glob=2 / anchors / update_slot_kv. Do not
relabel E18 as E21.

Geometry: seq=696, scale=`bridge_1k`, window=16, min_gap=64, right-align
row gap 100/100/100 (fact then decoy packed against hi; window still < gap).
Local SWA cannot span the type-cue evidence even when QUERY is not a
document start.

## Interpretation

**SELECT 696 is not hops-class SWA sever.** Same identity compressor that
scored E21 **0 bits** with QUERY as a SWA document start still recovers
**0.01 bits** (chance) when local SWA still sees across QUERY
(`msg_keepswa=True`). Uncompressed E18 copies **63.91 bits** on the same
rows. Exclusive identity slots stay on the global read (`msg_rawkv=False`).
`e18_local` stays at chance (K2) — keep_local_swa did not leak a solvable
local-only channel.

Hops 272 keep_local_swa **S1 PASS 25.55** does not transfer: chain uses
local hop glue near QUERY; SELECT type-then-value at the 4-token cliff
does not. 1024 SELECT keepswa was already 0.01 bits far past the wall;
this is the same floor **at the measured cliff**. Wall **(692 PASS, 696
FAIL]** stands even unsevered. Do **not** relabel E18's 63.91 bits as E21.

## Decision

Keep the spec in `ahead/`. **8k not run** (chance floor). Do **not** 16k.
Do not seq=694. Do not pack_stride / spread / window 32. Do not hops 320.
Do not MATCH remainder/pooling A/Bs. Do not Glyph. Do not unfreeze
`u`/`delta`. Do not restore full raw prefix KV (that is E18). Code default
`--message_keep_local_swa` stays **false**. Code default `--global_layers`
stays 1. Next ONE (do not run): **STOP SELECT keepswa stacks**; wall
**(692, 696]** stands even unsevered. If PASS had happened, parent may
later map SELECT length with SWA kept (not 694 without the flag).

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
