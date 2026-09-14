# E25 bridge_1k seq=696 select_1decoy r=1 identity `--local_window 32` H=256 SSMax log — E21 vs E18 vs dense (rung 5ax)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_696_w32_ip_id_h256_select` (800; floor — do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_696_w32_ip_id_h256_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_696_w32_ip_id_h256_select/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `c810a49` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 696 SELECT identity chance [`e25_bridge1k_696_ip_id_h256_select_20260914.md`](e25_bridge1k_696_ip_id_h256_select_20260914.md) · 692 SELECT identity PASS [`e25_bridge1k_692_ip_id_h256_select_20260914.md`](e25_bridge1k_692_ip_id_h256_select_20260914.md) · 696 leftover-drop pack_stride 32 [`e25_bridge1k_696_pack32_ip_id_h256_select_20260914.md`](e25_bridge1k_696_pack32_ip_id_h256_select_20260914.md) · 696 spread K1 [`e25_bridge1k_696_spread_ip_id_h256_select_20260914.md`](e25_bridge1k_696_spread_ip_id_h256_select_20260914.md)

---

## Goal

Locked: exclusive identity SELECT **692 S1 PASS 62.76 bits @500** vs **696 chance 0 bits @800** on right-align. Pack-stride 32 leftover-drop **31.67 @800 then 0 @8k**. Spread at 696 is **K1**. Cliff is not leftover count and not incomplete DNA pack. QUERY **654 vs 658** (654%16=**14**, 658%16=**2**). Hypothesis: the 4-token cliff interacts with **SWA-16** alignment at QUERY. This rung keeps seq=**696** packed `select_1decoy` and the **same** 1024-passing identity recipe (`--evidence_align right`, pack_stride **0**, remainder **off**), and changes **one existing flag**: `--local_window 32` (must stay **< min_gap 64**; not a new attend stack; not seq=694). DNA geometry is unchanged. Recalibrate dense in this JSON.

Score vs 0.75× live E18 (if E18 ~0, vs 0.75× dense). Dense first. 800-step floor: chance = no 8k; climbing short of S1 = 8k only then. Do not 16k.

## Banner / window vs gap

Hunt banner:

`seq=696 gap=64 window=32 prize=64.00 bits answer_len=32 … row_gap[min/med/max]=100/100/100`

**window 32 < min_gap 64** and **window 32 < row gap 100**. E18/E21 patterns `('swa', 32)` confirm the override hit the stack, not only the card. Hunt JSON: `local_window: 32`, `min_gap: 64`. Default `bridge_1k` window stays **16**; this hunt only overrides.

QUERY alignment (DNA unchanged vs right-align 696):

| seq | QUERY | QUERY%16 | QUERY%32 | leftover 32-pack |
|---|---|---|---|---|
| 692 PASS (window 16) | **654** | **14** | 14 | 14 |
| 696 FAIL (window 16) | **658** | **2** | 18 | 18 |
| 696 this hunt (window 32) | **658** | **2** | **18** | 18 |

Window 32 does **not** put 692 and 696 in the same residue class mod 32 (14 vs 18).

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_packstride=0`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`glob_layers=1`**, **`msg_anchors=none`** |
| Data | `--scale bridge_1k --seq_len 696`, gap=64, **window=32**, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · batch 32, ~0.06 s/step, ~2.1GB, no OOM |

**Knob:** `--local_window 32`. Pack_stride default **0**. Remainder **off**. Not seq=694. Not a new attend stack.

```
bash scripts/e24_bapo_hunt.sh e25_696_w32_ip_id_h256_select 0 \
  --scale bridge_1k --seq_len 696 --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --local_window 32 --steps 800 --k1_mult 4
# no --message_pack_stride (default 0)
# no --message_pool_remainder
# no --message_keep_local_swa
# extra hops 0, update_slot_kv off, global_layers 1, anchors none
# dense first (underscore --no-dense_first unused)
# 8k extra-step NOT run (chance floor @800)
# 16k NOT run
```

Byobu `E25_696_w32`. Hunt JSON: `seq_len: 696`, `min_gap: 64`, `local_window: 32`,
`hidden: 256`, `global_logit_scale: log`, `global_layers: 1`, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_pack_stride: 0`, `message_keep_local_swa: false`,
`message_extra_slot_attends: 0`, `message_update_slot_kv: false`,
`message_global_anchors: none`, `message_override: real`. Recalibrated dense
S0 in this JSON (`calibrated: true`). Hunt exit **0**. Params <100M.

## Training Outcome

### 800-step floor (`e25_696_w32_ip_id_h256_select`, exit 0)

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **99.9%** | 200 | **63.87** | 0.998 | 0.01147 |
| e18 | **99.9%** | 600 | **63.62** | 0.994 | 0.01143 |
| **e21 @800** | **25.0%** | 800 | **0.00** | **0.000** | **4.16e-7** |
| e18_local | 24.7% | 800 | 0.00 | 0.000 | 0 |

Params: dense/e18 ~2.261M, e21 2.277M (<100M). Prize 64 bits. Chance ~25%.
E21 CE glued to ln(4) every eval (min 1.3860, max 1.4020; acc 24.1–26.2%).
Best acc **26.2%** @600 = chance. Not climbing. E18 live **99.9% / 63.62
bits** @600 (do not pass S1 via 0.75×0). Dense early-stop 99.9% @200. Hunt
exit 0. **8k extra-step did not run.** GPU idle after 800; E22 Byobu left running.

S1 vs 0.75× this-JSON E18: need **47.71 bits**. E21 has **0.00**. Content vs
0.75× dense: need **47.90 bits**. E21 has **0.00**.

Do **not** relabel E18 as E21.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.9% / 63.87 bits @200. |
| **S1 vs 0.75× this-JSON E18** | **FAIL.** 0.00 ≪ 47.71 bits. Chance floor. |
| **content vs 0.75× dense** | **FAIL.** 0.00 ≪ 47.90 bits. |
| **S2** plots | **PASS.** traces + `learning_curves` / `recovered_bits` / `bytes_per_token`. |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.7% / 0.00 bits (window 32 still < gap 100). |
| **K3** | floor at 800 (flow 0.000). |

Do **not** extra-step to 8k (floor). Do not 16k. Do not seq=694 here. Do not
restore raw global KV. Do not extra hops / `update_slot_kv` / `keep_local_swa` /
`global_layers` 2 / anchors / H=512 / Glyph / unfreeze `u`/`delta`. Default
`--local_window` stays **16**. Default `--message_pack_stride` stays **0**.
Default remainder stays **off**. Default `--evidence_align` for SELECT length
rungs stays **right**. Do not relabel E18 as E21.

## Interpretation

**Widening SWA from 16 to 32 does not rescue exclusive identity SELECT at 696.**
Dense and uncompressed E18 still copy **~64 bits**. E21 recovers **0.00 bits**
(CE glued to ln(4); acc never above chance). The local stack really used
window 32 (`('swa', 32)`), K2 still holds (window < gap), and QUERY geometry
is the same 658 / leftover 18 as the window-16 fail. The 4-token cliff is
**not** SWA-16 QUERY alignment that a wider existing window can fix. Wall
remains **(692 PASS, 696 FAIL]** on **right-align** identity slots. Do **not**
relabel E18's 63.62 bits as E21.

## Decision

Keep the spec in `ahead/`. Defaults stay: remainder off, pack_stride **0**,
extra hops 0, `update_slot_kv` off, `keep_local_swa` off, `global_layers` 1,
anchors `none`, `--evidence_align right` on SELECT length rungs,
`--local_window` **16**. Stop stacking window on 696. Next ONE (do **not**
run): seq=**694** packed `select_1decoy` with the **same** 1024-passing
identity recipe and **`--evidence_align right`** (pack_stride default 0,
remainder off, **default window 16**). Existing `--seq_len`. Tightens
**(692, 696]** with a calibrated S0. Not another window. Not spread. Not
pack_stride 32. Not 8k a chance floor. Not Glyph. Do not unfreeze `u`/`delta`.
Do not restore raw global KV (that is E18).

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
