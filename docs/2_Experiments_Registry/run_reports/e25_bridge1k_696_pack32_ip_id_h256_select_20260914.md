# E25 bridge_1k seq=696 select_1decoy r=1 identity `--message_pack_stride 32` H=256 SSMax log — E21 vs E18 vs dense (rung 5av)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_696_pack32_h256_select` (800 floor; climbing short of S1) · `e25_696_pack32_h256_select_s8k` (8k extra-step; chance floor)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_696_pack32_ip_id_h256_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_696_pack32_ip_id_h256_select/800/probe.log` · `/opt/cursor/artifacts/e25_bridge1k_696_pack32_ip_id_h256_select/s8k/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `ded8ecc` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 692 SELECT identity PASS [`e25_bridge1k_692_ip_id_h256_select_20260914.md`](e25_bridge1k_692_ip_id_h256_select_20260914.md) · 696 SELECT identity chance [`e25_bridge1k_696_ip_id_h256_select_20260914.md`](e25_bridge1k_696_ip_id_h256_select_20260914.md) · 688 SELECT identity PASS [`e25_bridge1k_688_ip_id_h256_select_20260914.md`](e25_bridge1k_688_ip_id_h256_select_20260914.md)

---

## Goal

Locked: exclusive identity SELECT **692 S1 PASS 62.76 bits @500** vs **696 chance 0 bits @800**. 4-token cliff. Not 32-token pack stride (692 ≡ 20 mod 32; 696 ≡ 24 mod 32). This rung inspects `select_1decoy` packing under `--evidence_align right`, then tries **one packing/geometry knob** at seq=696: `--message_pack_stride 32` remainder **off**. `--message_pool_remainder` at r=1 identity is a **no-op** (every sender token is already an identity slot). Do not invent seq=694. Do not restore raw KV. Recalibrate dense in this JSON.

Score vs 0.75× live E18. Climbing at 800 but short of S1 → extra-step 8k; chance/plateau at 8k → do not 16k.

## Geometry (Step 1 — 688 / 692 / 696, seed 0, right-align)

`select_1decoy` on `--scale bridge_1k --evidence_align right`. Prize stays **32 tokens / 64 bits** (`target_answer_len` uses scale seq 1024, not the override). Each KV pack is **35 tokens** (keymark/decoy + 2 key + 32 value). **2 complete evidence packs**; tokens **693–696 do not add an incomplete DNA block** — they are **left-filler after BOS**. Tail after QUERY is **38** on all three lengths. Gap filler decoy→QUERY stays **60**. Window 16 < gap 64. Row gap **100/100/100**.

| seq | QUERY | sender | fact | decoy | left filler after BOS | decoy→QUERY filler | 32-pack leftover | complete 32-packs | tail |
|---|---|---|---|---|---|---|---|---|---|
| 688 PASS | 650 | 650 | [520, 555) | [555, 590) | 519 | 60 | **10** | 20 | 38 |
| 692 PASS | 654 | 654 | [524, 559) | [559, 594) | 523 | 60 | **14** | 20 | 38 |
| 696 FAIL | 658 | 658 | [528, 563) | [563, 598) | 527 | 60 | **18** | 20 | 38 |

**692 → 696 delta = +4** on QUERY, sender, fact, decoy, and left filler. Gap filler stays 60. Evidence packs stay 2 complete.

r=1 remainder-on vs remainder-off: **same exclusive sender count** (658 at seq=696). `--message_pack_stride 32` remainder **off** drops leftover `[0, P % 32)` from exclusive replace → **640** exclusive sender slots at both 692 and 696. Remainder-on plus pack_stride **keeps** leftover as identity slots (back to 658 at 696).

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / `e18_local` |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_packstride=32`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`glob_layers=1`**, **`msg_anchors=none`** |
| Data | `--scale bridge_1k --seq_len 696`, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · batch 32, ~0.06 s/step, no OOM |

**Knob:** `--message_pack_stride 32` remainder **off**. Not remainder-on (no-op at r=1). Not a new attend stack.

```
bash scripts/e24_bapo_hunt.sh e25_696_pack32_h256_select 0 \
  --scale bridge_1k --seq_len 696 --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --message_pack_stride 32 --steps 800 --k1_mult 4
# 8k extra-step (climbing short of S1 @800):
bash scripts/e24_bapo_hunt.sh e25_696_pack32_h256_select_s8k 0 \
  ...same flags... --message_pack_stride 32 --steps 8000 --k1_mult 1
# no --message_pool_remainder
# dense first (underscore --no-dense_first unused)
# 16k NOT run (8k chance floor)
```

Byobu `E25_696_pack32` then `E25_696_pack32_s8k`. Hunt JSON: `message_pack_stride: 32`, remainder false, identity inplace, extra hops 0, update_slot_kv false, keep_local_swa false, global_layers 1, anchors none. Recalibrated dense S0 in each JSON.

## Training Outcome

### 800-step floor (`e25_696_pack32_h256_select`, exit 0)

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **100%** | 200 | **63.95** | 0.999 | 0.01148 |
| e18 | **99.8%** | 550 | **63.33** | 0.990 | 0.01137 |
| **e21 @800** | **49.7%** | 800 | **31.67** | **0.495** | **0.005688** |
| e18_local | 24.5% | 800 | 0.00 | 0.000 | 0 |

E21 chance through ~700 (CE ~ln(4)); click **50.2% @750**, **49.7% / 31.67 bits @800** (best acc 50.2%). S1 vs 0.75× this-JSON E18: need **47.50 bits**. E21 has **31.67** — climbing, short of S1 → 8k extra-step.

### 8k extra-step (`e25_696_pack32_h256_select_s8k`, exit 0)

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **100%** | 200 | **63.96** | 0.999 | 0.01149 |
| e18 | **100%** | 350 | **63.91** | 0.999 | 0.01148 |
| **e21 @8000** | **24.1%** | 8000 | **0.00** | **0.000** | **0** |
| e18_local | 24.1% | 8000 | 0.00 | 0.000 | 0 |

E21 acc never > 0.262 across 160 evals (CE glued to ln(4); min CE 1.3859 @900). Best acc 26.2% = chance. The 800-step click **did not replicate**. S1 vs 0.75× this-JSON E18: need **47.94 bits**. E21 has **0.00**. **Do not 16k.**

Params: dense/e18 ~2.261M, e21 2.277M (<100M). Prize 64 bits. Chance ~25%. E18 live in both JSONs (do not pass S1 via 0.75×0). Dense early-stop 100% @200. GPU idle after 8k; E22 Byobu left running.

## Gates vs this rung

| gate | 800 | 8k |
|---|---|---|
| **S0** | **PASS.** Dense 100% / 63.95 bits @200. | **PASS.** Dense 100% / 63.96 bits @200. |
| **S1 vs 0.75× this-JSON E18** | **FAIL.** 31.67 < 47.50 bits (climbing). | **FAIL.** 0.00 ≪ 47.94 bits (chance floor). |
| **content vs 0.75× dense** | **FAIL.** 31.67 < 47.96. | **FAIL.** 0.00 ≪ 47.97. |
| **S2** plots | **PASS.** traces + `learning_curves` / `recovered_bits` / `bytes_per_token`. | **PASS.** same. |
| **K1** | not triggered | not triggered |
| **K2** | **PASS.** `e18_local` 24.5% / 0 bits. | **PASS.** `e18_local` 24.1% / 0 bits. |
| **K3** | not floor at 800 (flow 0.495). | floor at 8k (flow 0.000). |

Do **not** 16k. Do not seq=694. Do not restore raw global KV. Do not extra hops / `update_slot_kv` / `keep_local_swa` / `global_layers` 2 / anchors / H=512 / Glyph / unfreeze `u`/`delta`. Default `--message_pack_stride` stays **0**. Default remainder stays **off**. Do not relabel E18 as E21.

## Interpretation

**Dropping QUERY-aligned leftover identity slots does not rescue exclusive SELECT at 696.** Remainder-on cannot change the 4-token leftover at r=1 (no-op). Pack-stride 32 equalizes exclusive sender count to **640** at both 692 and 696, so leftover *count* is not the MATCH-style incomplete-block wall. The 800-step snapshot climbed to ~50% / 31.67 bits (unlike default-696's chance floor) but stayed short of S1; the 8k replica stayed at chance every eval. Seed-sensitive late click, not a packing rescue. Wall remains **(692 PASS, 696 FAIL]** on identity slots. Do **not** relabel E18's ~64 bits as E21.

## Decision

Keep the spec in `ahead/`. Defaults stay: remainder off, pack_stride **0**, extra hops 0, `update_slot_kv` off, `keep_local_swa` off, `global_layers` 1, anchors `none`. Stop stacking leftover-drop on 696. Next ONE (do **not** run): seq=**696** packed `select_1decoy` with the **same** 1024-passing identity recipe and **`--evidence_align spread`** (pack_stride default 0, remainder off). QUERY index stays 658 (leftover *size* 18 unchanged); spread redistributes slack so the extra 4 tokens are not locked as left-filler after BOS (right-align fact0=528 vs spread seed-0 fact0=202). Existing flag. Not seq=694. Not a new attend stack. Not 8k a chance floor. Not Glyph. Do not unfreeze `u`/`delta`. Do not restore raw global KV (that is E18).

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
