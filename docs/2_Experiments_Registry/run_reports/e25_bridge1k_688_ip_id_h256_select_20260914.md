# E25 bridge_1k seq=688 select_1decoy r=1 identity H=256 SSMax log — E21 vs E18 vs dense (rung 5as)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_688_ip_id_h256_select` (800 advertised; climb 72.4% / 31.57 bits, short of S1) · `e25_688_ip_id_h256_select_s8k` (8000 advertised; E21 early-stop 400 — S1 PASS)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_688_ip_id_h256_select/` · `/opt/cursor/artifacts/e25_bridge1k_688_ip_id_h256_select_s8k/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_688_ip_id_h256_select/probe.log` · `/opt/cursor/artifacts/e25_bridge1k_688_ip_id_h256_select_s8k/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `b047c04` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 672 SELECT identity PASS [`e25_bridge1k_672_ip_id_h256_select_20260914.md`](e25_bridge1k_672_ip_id_h256_select_20260914.md) · 704 SELECT identity chance [`e25_bridge1k_704_ip_id_h256_select_20260914.md`](e25_bridge1k_704_ip_id_h256_select_20260914.md) · 640 SELECT identity PASS [`e25_bridge1k_640_ip_id_h256_select_20260914.md`](e25_bridge1k_640_ip_id_h256_select_20260914.md)

---

## Goal

Locked: seq=512 packed `select_1decoy` r=1 identity **S1 PASS 47.98 bits**
(H=128). Seq=640 / 672 r=1 identity H=256 log **S1 PASS ~64 bits**. Seq=704
/ 768 / 1024 identity chance. E18 still copies at 672/688/704/1024. This
rung **tightens SELECT length** at seq=**688** with the **same 1024-passing
recipe** (`--hidden 256 --global_logit_scale log`) so width is not the
confound. Existing probe `--seq_len` on `--scale bridge_1k` (not a new
scale; `bridge` would pack 24 tokens / 48 bits). Window 16 < gap 64.
Recalibrate dense in this JSON.

Score vs 0.75× live E18; if E18 were ~0, score vs 0.75× dense. Climbing at
800 but short of S1 → extra-step 8k; chance at 800 → do not extra-step.
No new architecture flags. Remainder off. Extra hops 0. `update_slot_kv`
off. `keep_local_swa` off. `global_layers` 1. Anchors **none**.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`glob_layers=1`**, **`msg_anchors=none`** |
| Data | `--scale bridge_1k --seq_len 688`, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · batch 32, ~0.06 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_688_ip_id_h256_select 0 \
  --scale bridge_1k --seq_len 688 --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# climbing short of S1:
bash scripts/e24_bapo_hunt.sh e25_688_ip_id_h256_select_s8k 0 \
  --scale bridge_1k --seq_len 688 --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 8000 --k1_mult 1
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# no --global_layers (default 1)
# anchors none (default)
# dense first (underscore --no-dense_first unused)
```

Byobu `E25_688_id_select` / `E25_688_id_select_s8k`. Log:
`seq=688  gap=64  window=16  prize=64.00 bits  answer_len=32  logit_scale=log
msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False
msg_idslots=True  msg_keepswa=False  glob_layers=1  msg_extrahops=0
msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 688`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, `global_layers: 1`, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`message_override: real`. Recalibrated dense S0 in **both** JSONs.
**No new scale enum** — probe `--seq_len` plus
`test_bridge_1k_seq_len_688_select_keeps_window_below_gap`.

## Training Outcome

800-step hunt (climb, short of S1):

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **100%** | 250 | **63.96** | 0.999 | 0.01162 |
| e18 | **100%** | 350 | **63.73** | 0.996 | 0.01158 |
| **e21 @800** | **72.4%** | 800 | **31.57** | **0.493** | **0.00574** |
| e18_local | 25.3% | 800 | 0.00 | 0.000 | 0 |

E21 chance through 450 (CE at ln(4)≈1.386); **26.8% / CE 1.361 @500**;
**45.0% @550**; **88.9% / CE 0.192 @750** (best); **72.4% / 31.57 bits
@800** (dip, not a floor). S1 vs 0.75× this-JSON E18 needs **47.80 bits**.
E21 has **31.57**. Extra-step 8k.

8k extra-step (dense-first, same recipe, `--k1_mult 1`):

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **100%** | 200 | **63.96** | 0.999 | 0.01162 |
| e18 | **100%** | 350 | **63.96** | 0.999 | 0.01162 |
| **e21 @400** | **99.5%** | 400 | **62.68** | **0.979** | **0.01139** |
| e18_local | 26.1% | 8000 | 0.01 | 0.000 | ~0 |

Params: dense/e18 ~2.261M, e21 2.277M (<100M). Prize 64 bits. Chance ~25%.
E21 chance through 300; **30.1% @350**; **99.5% / 62.68 bits @400**
early-stop. E18 live **100% / 63.96 bits** @350 (do not pass S1 via
0.75×0). Dense early-stop 100% @200. Hunt exit 0.

S1 vs 0.75× this-JSON E18: need **47.97 bits**. E21 has **62.68**. Content
vs 0.75× dense: need **47.97 bits**. E21 has **62.68**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.96 bits @200 (8k JSON; 800 JSON 100% / 63.96 @250). |
| **S1 vs 0.75× this-JSON E18** | **PASS** at extra-step. 62.68 ≥ 47.97 bits. 800-step 31.57 < 47.80 (climb, not chance). |
| **content vs 0.75× dense** | **PASS.** 62.68 ≥ 47.97 bits. |
| **S2** plots | **PASS.** traces in both hunt JSONs. |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 26.1% / 0.01 bits @8000 (800 JSON 25.3% / 0). |
| **K3** | not triggered (flow 0.979, not chance). |

Do **not** 16k (S1 already PASS at 400). Do not remainder-on. Do not restore
raw global KV. Do not extra width. Do not hops seq shrink. Do not Glyph. Do
not unfreeze `u`/`delta`. Do not relabel E18 as E21.

Geometry: seq=688, scale=`bridge_1k`, window=16, min_gap=64, right-align
row gap 100/100/100 (fact then decoy packed against hi; window still < gap).

## Interpretation

**Exclusive identity SELECT that passed at 672 also passes at 688** on the
1024-passing H=256 SSMax-log recipe (**99.5% / 62.68 bits** @400 vs live
E18 **63.96**). The 800-step snapshot was a mid-click dip (88.9% @750 →
72.4% / 31.57 bits @800), not a chance floor, so extra-step 8k was required
and clicked fully at 400. Dense and uncompressed E18 still copy **~64
bits**. The exclusive compressed read that is chance at 704 is a full copy
at 688. The SELECT length wall is now **(688 PASS, 704 FAIL]**. Do **not**
relabel E18's 63.96 bits as E21.

## Decision

Keep the spec in `ahead/`. Defaults stay: remainder off, extra hops 0,
`update_slot_kv` off, `keep_local_swa` off, `global_layers` 1, anchors
`none`. Next ONE (do **not** run): seq=**696** packed `select_1decoy` with
the **same** 1024-passing recipe (`--scale bridge_1k --seq_len 696 --hidden
256 --global_logit_scale log`, r=1 identity inplace, remainder off, extra
hops 0, update_slot_kv off, keep_local_swa off, global_layers 1, anchors
none). Tightens the SELECT length wall in (688, 704]. Not architecture
knobs. Not 16k an S1 pass. Not Glyph. Do not unfreeze `u`/`delta`. Do
not restore raw global KV (that is E18).

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
