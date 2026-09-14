# E25 bridge_1k seq=692 select_1decoy r=1 identity H=256 SSMax log — E21 vs E18 vs dense (rung 5au)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_692_ip_id_h256_select` (800 advertised; E21 early-stop 500 — S1 PASS)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_692_ip_id_h256_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_692_ip_id_h256_select/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `5611b67` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 688 SELECT identity PASS [`e25_bridge1k_688_ip_id_h256_select_20260914.md`](e25_bridge1k_688_ip_id_h256_select_20260914.md) · 696 SELECT identity chance [`e25_bridge1k_696_ip_id_h256_select_20260914.md`](e25_bridge1k_696_ip_id_h256_select_20260914.md) · 672 SELECT identity PASS [`e25_bridge1k_672_ip_id_h256_select_20260914.md`](e25_bridge1k_672_ip_id_h256_select_20260914.md)

---

## Goal

Locked: seq=512 packed `select_1decoy` r=1 identity **S1 PASS 47.98 bits**
(H=128). Seq=640 / 672 / 688 r=1 identity H=256 log **S1 PASS ~64 bits**
(688 needed extra-step 8k after a 31.57-bit climb @800). Seq=696 / 704 /
768 / 1024 identity chance. E18 still copies at 672/688/692/696/704/1024.
This rung **tightens SELECT length** at seq=**692** with the **same
1024-passing recipe** (`--hidden 256 --global_logit_scale log`) so width
is not the confound. Existing probe `--seq_len` on `--scale bridge_1k`
(not a new scale; `bridge` would pack 24 tokens / 48 bits). Window 16 <
gap 64. Recalibrate dense in this JSON.

Score vs 0.75× live E18; if E18 were ~0, score vs 0.75× dense. Climbing at
800 but short of S1 → extra-step 8k; **S1 PASS inside 800 → do not extra-step**.
Do not 16k. No new architecture flags. Remainder off. Extra hops 0.
`update_slot_kv` off. `keep_local_swa` off. `global_layers` 1. Anchors
**none**.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`glob_layers=1`**, **`msg_anchors=none`** |
| Data | `--scale bridge_1k --seq_len 692`, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · batch 32, ~0.06 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_692_ip_id_h256_select 0 \
  --scale bridge_1k --seq_len 692 --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# no --global_layers (default 1)
# anchors none (default)
# dense first (underscore --no-dense_first unused)
# 8k extra-step NOT run (S1 PASS @500)
```

Byobu `E25_692_id_select`. Log:
`seq=692  gap=64  window=16  prize=64.00 bits  answer_len=32  logit_scale=log
msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False
msg_idslots=True  msg_keepswa=False  glob_layers=1  msg_extrahops=0
msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 692`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, `global_layers: 1`, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`message_override: real`. Recalibrated dense S0 in the same JSON.
**No new scale enum** — probe `--seq_len` plus
`test_bridge_1k_seq_len_692_select_keeps_window_below_gap`.

Pack alignment: prize stays **32 tokens / 64 bits** because
`target_answer_len` uses the `bridge_1k` scale seq (1024), not the
override (`bridge` would pack 24 / 48). Seq=692 ≡ **20 (mod 32)** and
**4 (mod 16)** — not a packed-answer or SWA-window boundary. 688 PASS
≡ 16 (mod 32) / 0 (mod 16); 696 FAIL ≡ 24 (mod 32) / 8 (mod 16). The
wall is not sitting on the 32-token pack stride.

## Training Outcome

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **100%** | 200 | **63.97** | 1.000 | 0.01156 |
| e18 | **100%** | 350 | **63.89** | 0.998 | 0.01154 |
| **e21 @500** | **99.8%** | 500 | **62.76** | **0.981** | **0.01134** |
| e18_local | 23.0% | 800 | 0.00 | 0.000 | 0 |

Params: dense/e18 ~2.261M, e21 2.277M (<100M). Prize 64 bits. Chance ~25%.
E21 chance through 450 (CE at ln(4)≈1.386); **99.8% / CE 0.027 @500**
early-stop (best acc 99.8%). E18 live **100% / 63.89 bits** @350 (do not
pass S1 via 0.75×0). Dense early-stop 100% @200. Hunt exit 0. **8k
extra-step did not run.**

S1 vs 0.75× this-JSON E18: need **47.91 bits**. E21 has **62.76**. Content
vs 0.75× dense: need **47.98 bits**. E21 has **62.76**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.97 bits @200. |
| **S1 vs 0.75× this-JSON E18** | **PASS.** 62.76 ≥ 47.91 bits. |
| **content vs 0.75× dense** | **PASS.** 62.76 ≥ 47.98 bits. |
| **S2** plots | **PASS.** traces + `learning_curves` / `recovered_bits` in hunt artifacts. |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 23.0% / 0.00 bits. |
| **K3** | not triggered (flow 0.981, not chance). |

Do **not** extra-step to 8k (S1 already PASS at 500). Do not 16k. Do not
remainder-on. Do not restore raw global KV. Do not extra width. Do not hops
seq shrink. Do not Glyph. Do not unfreeze `u`/`delta`. Do not relabel E18
as E21.

Geometry: seq=692, scale=`bridge_1k`, window=16, min_gap=64, right-align
row gap 100/100/100 (fact then decoy packed against hi; window still < gap).

## Interpretation

**Exclusive identity SELECT that passed at 688 also passes at 692** on the
1024-passing H=256 SSMax-log recipe (**99.8% / 62.76 bits** @500 vs live
E18 **63.89**). Dense and uncompressed E18 still copy **~64 bits**. Unlike
688, the 800-step budget was enough (click between 450 and 500; no
mid-click dip). Unlike 696, this is not a chance floor. The SELECT length
wall is now **(692 PASS, 696 FAIL]**. Do **not** relabel E18's 63.89 bits
as E21.

## Decision

Keep the spec in `ahead/`. Defaults stay: remainder off, extra hops 0,
`update_slot_kv` off, `keep_local_swa` off, `global_layers` 1, anchors
`none`. Next ONE (do **not** run): seq=**694** packed `select_1decoy` with
the **same** 1024-passing recipe (`--scale bridge_1k --seq_len 694 --hidden
256 --global_logit_scale log`, r=1 identity inplace, remainder off, extra
hops 0, update_slot_kv off, keep_local_swa off, global_layers 1, anchors
none). Tightens the SELECT length wall in (692, 696]. Not architecture
knobs. Not 8k an S1 pass. Not Glyph. Do not unfreeze `u`/`delta`. Do
not restore raw global KV (that is E18).

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
