# E25 bridge_1k seq=696 select_1decoy r=1 identity `--evidence_align spread` H=256 SSMax log — dense K1 (rung 5aw)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_696_spread_ip_id_h256_select` (800 advertised / K1 3200 dense; uncalibrated — do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_696_spread_ip_id_h256_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_696_spread_ip_id_h256_select/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `fcd04eb` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 696 SELECT identity chance [`e25_bridge1k_696_ip_id_h256_select_20260914.md`](e25_bridge1k_696_ip_id_h256_select_20260914.md) · 696 leftover-drop pack_stride 32 [`e25_bridge1k_696_pack32_ip_id_h256_select_20260914.md`](e25_bridge1k_696_pack32_ip_id_h256_select_20260914.md) · 692 SELECT identity PASS [`e25_bridge1k_692_ip_id_h256_select_20260914.md`](e25_bridge1k_692_ip_id_h256_select_20260914.md)

---

## Goal

Locked: exclusive identity SELECT **692 S1 PASS 62.76 bits @500** vs **696 chance 0 bits @800** on right-align. Pack-stride 32 leftover-drop **31.67 @800 then 0 @8k** — leftover *count* is not the cliff. Geometry: 693–696 are **left-filler after BOS**, not an incomplete DNA pack. This rung keeps seq=**696** packed `select_1decoy` and the **same** 1024-passing identity recipe, and changes **one existing flag**: `--evidence_align spread` (pack_stride default **0**, remainder **off**). QUERY index stays 658 (leftover *size* 18 unchanged); spread redistributes slack so the extra 4 tokens are not locked as left-filler after BOS.

Score vs 0.75× live E18 (if E18 ~0, vs 0.75× dense). Dense first; recalibrate dense in this JSON. 800-step floor: chance = no 8k; climbing short of S1 = extra-step 8k only then. Do not 16k. Do not invent seq=694 here. Do not restore raw KV. Recalibrate dense in this JSON.

## Geometry (seed 0, seq=696, right vs spread)

`select_1decoy` on `--scale bridge_1k --seq_len 696`. Prize stays **32 tokens / 64 bits**. Each KV pack is **35 tokens**. **2 complete evidence packs**. QUERY **658** and leftover **18** on both aligns. Tail after QUERY **38**. Window 16 < gap 64.

| align | QUERY | sender | fact | decoy | left filler after BOS | mid filler | decoy→QUERY filler | 32-pack leftover | complete 32-packs | tail | row gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **right** (prior 696) | **658** | 658 | **[528, 563)** | **[563, 598)** | **527** | 0 | 60 | **18** | 20 | 38 | 100/100/100 |
| **spread seed-0** (this hunt) | **658** | 658 | **[202, 237)** | **[534, 569)** | **201** | 297 | 89 | **18** | 20 | 38 | 336/506/623 |

Spread vs right-align at seed 0: QUERY/leftover/tail unchanged. **fact0 528 → 202**. **decoy0 563 → 534**. **left-filler 527 → 201**. Slack moves into mid-filler (0 → 297) and a longer decoy→QUERY filler (60 → 89). Probe `row_gap[min/med/max]=336/506/623` vs right-align **100/100/100**. Evidence is no longer packed against `min_gap`. Hunt JSON does not store `evidence_align`; log args + row_gap prove spread. Geometry dump: `/opt/cursor/artifacts/e25_bridge1k_696_spread_ip_id_h256_select/geometry_seed0.json`.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense first; e18 / e21 / `e18_local` skip on K1 |
| Width | **H=256** · 2.261M · kv=1 · **`logit_scale=log`** |
| E21 (requested, not scored) | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_packstride=0`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`glob_layers=1`**, **`msg_anchors=none`** |
| Data | `--scale bridge_1k --seq_len 696`, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | **`--evidence_align spread`** (row gap 336/506/623) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · batch 32, ~0.05 s/step, ~1.9GB, no OOM |

**Knob:** `--evidence_align spread`. Pack_stride default **0**. Remainder **off**. Not pack_stride 32. Not seq=694. Not a new attend stack.

```
bash scripts/e24_bapo_hunt.sh e25_696_spread_ip_id_h256_select 0 \
  --scale bridge_1k --seq_len 696 --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align spread --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# no --message_pack_stride (default 0)
# no --message_pool_remainder
# no --message_keep_local_swa
# extra hops 0, update_slot_kv off, global_layers 1, anchors none
# dense first (underscore --no-dense_first unused)
# 8k extra-step NOT run (dense chance floor at K1)
# 16k NOT run
```

Byobu `E25_696_spread`. Log: `seq=696  gap=64  window=16  prize=64.00 bits  answer_len=32
logit_scale=log` · `row_gap[min/med/max]=336/506/623` · args include
`--evidence_align spread`. Hunt JSON: `seq_len: 696`, `min_gap: 64`,
`local_window: 16`, `hidden: 256`, `global_logit_scale: log`, `global_layers: 1`,
`message_ratio: 1`, `message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_pack_stride: 0`, `message_keep_local_swa: false`,
`message_extra_slot_attends: 0`, `message_update_slot_kv: false`,
`message_global_anchors: none`, `message_override: real`. Recalibrated dense
S0 in this JSON (`calibrated: false`). Hunt exit **2**.

## Training Outcome

### 800-step floor / K1 (`e25_696_spread_ip_id_h256_select`, exit 2)

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **24.7%** | 3200 | **0.00** | **0.000** | **0** |
| e18 | skipped (dense < 75% at K1) | — | — | — | — |
| **e21** | skipped (dense < 75% at K1) | — | — | — | — |
| e18_local | skipped (dense < 75% at K1) | — | — | — | — |

Dense chance every eval (64 evals; acc 23.8–26.6%; CE glued to ln(4), final 1.3868 vs floor 1.3863). Best acc **26.6%** = chance. `dense_steps_used: 3200` (`steps 800 × k1_mult 4`). Params 2.261M (<100M). Prize 64 bits. Chance ~25%. **8k not run.** GPU idle after K1; E22 Byobu left running.

Do **not** score S1 vs E18 or vs dense. There is no E21 number. Do not relabel E18 as E21.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **FAIL / K1.** Dense 24.7% / 0 bits @3200 < 75%. |
| **S1 vs 0.75× this-JSON E18** | **not scored** (E18 skipped). |
| **content vs 0.75× dense** | **not scored** (dense itself 0 bits). |
| **S2** plots | **PASS.** dense-only traces + `learning_curves` / `recovered_bits` / `bytes_per_token`. |
| **K1** | **triggered.** Stop. Do not score E21. |
| **K2** | **not scored** (`e18_local` skipped). |
| **K3** | **not scored.** |

Do **not** 8k (chance floor, not climbing). Do **not** 16k. Do not seq=694 here. Do not restore raw global KV. Do not extra hops / `update_slot_kv` / `keep_local_swa` / `global_layers` 2 / anchors / H=512 / Glyph / unfreeze `u`/`delta`. Default `--message_pack_stride` stays **0**. Default remainder stays **off**. Default SELECT length rungs stay `--evidence_align right` (spread at 696 is K1). Do not relabel E18 as E21.

## Interpretation

**Spreading slack off the right-align left-filler does not yield a calibrated 696 SELECT rung.** QUERY/leftover stay 658/18, so leftover *size* is unchanged. Seed-0 fact0 moves 528 → 202 and left-filler 527 → 201 as intended, but evidence is no longer a near-copy packed against `min_gap`. Dense, which solved right-align 696 at **100% / 63.95 bits @200**, stays at chance through the full K1 budget on spread. Same failure mode as advertised 4k spread INDEX (K1 until dense ≥ 75%). The 4-token cliff remains **(692 PASS, 696 FAIL]** on **right-align** identity slots. This rung does not isolate left-filler vs leftover vs length, because S0 failed. Do **not** treat skipped E18/E21 as scores.

## Decision

Keep the spec in `ahead/`. Defaults stay: remainder off, pack_stride **0**, extra hops 0, `update_slot_kv` off, `keep_local_swa` off, `global_layers` 1, anchors `none`, `--evidence_align right` on SELECT length rungs. Stop stacking spread on 696. Next ONE (do **not** run): seq=**694** packed `select_1decoy` with the **same** 1024-passing identity recipe and **`--evidence_align right`** (pack_stride default 0, remainder off). Existing `--seq_len`. Tightens **(692, 696]** with a calibrated S0. Not spread again. Not pack_stride 32. Not 8k a chance floor. Not Glyph. Do not unfreeze `u`/`delta`. Do not restore raw global KV (that is E18).

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
