# E25 bridge_1k seq=640 select_1decoy r=1 identity H=256 SSMax log — E21 vs E18 vs dense (rung 5ap)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_640_ip_id_h256_select` (800 advertised; E21 early-stop 450 — S1 PASS, do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_640_ip_id_h256_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_640_ip_id_h256_select/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `dbc3b1a` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 512 SELECT identity PASS [`e25_bridge512_ip_id_select_20260914.md`](e25_bridge512_ip_id_select_20260914.md) · 768 SELECT identity chance [`e25_bridge1k_768_ip_id_h256_select_20260914.md`](e25_bridge1k_768_ip_id_h256_select_20260914.md) · 1024 SELECT identity chance [`e25_bridge1k_ip_id_h256_select_20260914.md`](e25_bridge1k_ip_id_h256_select_20260914.md)

---

## Goal

Locked: seq=512 packed `select_1decoy` r=1 identity **S1 PASS 47.98 bits**
(H=128). Seq=768 r=1 identity H=256 log **chance 0.01 bits**. Seq=1024
identity chance after every exclusive-slot architecture knob. E18 still
copies at 768 and 1024. This rung **tightens SELECT length** at seq=**640**
with the **same 1024-passing recipe** (`--hidden 256 --global_logit_scale
log`) so width is not the confound. Existing probe `--seq_len` on `--scale
bridge_1k` (not a new scale; `bridge` would pack 24 tokens / 48 bits).
Window 16 < gap 64. Recalibrate dense in this JSON.

Score vs 0.75× live E18; if E18 were ~0, score vs 0.75× dense. Climbing at
800 but short of S1 → extra-step 8k; **S1 already PASS → do not extra-step**.
No new architecture flags. Remainder off. Extra hops 0. `update_slot_kv`
off. `keep_local_swa` off. `global_layers` 1. Anchors **none**.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`glob_layers=1`**, **`msg_anchors=none`** |
| Data | `--scale bridge_1k --seq_len 640`, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · batch 32, ~0.05–0.06 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_640_ip_id_h256_select 0 \
  --scale bridge_1k --seq_len 640 --recipe select_1decoy --arch dense e18 e21 e18_local \
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
```

Byobu `E25_640_id_select`. Log:
`seq=640  gap=64  window=16  prize=64.00 bits  answer_len=32  logit_scale=log
msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False
msg_idslots=True  msg_keepswa=False  glob_layers=1  msg_extrahops=0
msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 640`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, `global_layers: 1`, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`message_override: real`. Recalibrated dense S0 in the same JSON.
**No new scale enum** — probe `--seq_len` plus
`test_bridge_1k_seq_len_640_select_keeps_window_below_gap`.

## Training Outcome

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **99.8%** | 200 | **63.76** | 0.996 | 0.01245 |
| e18 | **100%** | 300 | **63.91** | 0.999 | 0.01248 |
| **e21 @450** | **100%** | 450 | **63.95** | **0.999** | **0.01249** |
| e18_local | 25.9% | 800 | 0.00 | 0.000 | 0 |

Params: dense/e18 ~2.261M, e21 2.277M (<100M). Prize 64 bits. Chance ~25%.
E21 chance through 350 (CE at ln(4)≈1.386; acc 23.4–25.4%); **51.3% / CE
0.817 @400**; **100% / 63.95 bits @450** early-stop. Not a floor. E18 live
**100% / 63.91 bits** @300 (do not pass S1 via 0.75×0). Dense early-stop
99.8% @200. Hunt exit 0.

S1 vs 0.75× this-JSON E18: need **47.93 bits**. E21 has **63.95**. Content vs
0.75× dense: need **47.82 bits**. E21 has **63.95**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.8% / 63.76 bits @200. |
| **S1 vs 0.75× this-JSON E18** | **PASS.** 63.95 ≥ 47.93 bits. |
| **content vs 0.75× dense** | **PASS.** 63.95 ≥ 47.82 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.9% / 0 bits. |
| **K3** | not triggered (flow 0.999, not chance). |

Do **not** extra-step to 8k (S1 already PASS at 450). Do not 16k. Do not
remainder-on. Do not restore raw global KV. Do not extra width. Do not hops
seq shrink. Do not Glyph. Do not unfreeze `u`/`delta`. Do not relabel E18
as E21.

Geometry: seq=640, scale=`bridge_1k`, window=16, min_gap=64, right-align
row gap 100/100/100 (fact then decoy packed against hi; window still < gap).

## Interpretation

**Exclusive identity SELECT that passed at 512 also passes at 640** on the
1024-passing H=256 SSMax-log recipe (**100% / 63.95 bits** @450 vs live E18
**63.91**). Dense and uncompressed E18 still copy **~64 bits**. The exclusive
compressed read that is chance at 768 is a full copy at 640. The SELECT
length wall is now **(640 PASS, 768 FAIL]**. 1024 architecture knobs were
hunting a wall that already exists in that interval. Do **not** relabel
E18's 63.91 bits as E21.

## Decision

Keep the spec in `ahead/`. Defaults stay: remainder off, extra hops 0,
`update_slot_kv` off, `keep_local_swa` off, `global_layers` 1, anchors
`none`. Next ONE (do **not** run): seq=**704** packed `select_1decoy` with
the **same** 1024-passing recipe (`--scale bridge_1k --seq_len 704 --hidden
256 --global_logit_scale log`, r=1 identity inplace, remainder off, extra
hops 0, update_slot_kv off, keep_local_swa off, global_layers 1, anchors
none). Tightens the SELECT length wall in (640, 768]. Not architecture
knobs. Not 8k an S1 pass. Not Glyph. Do not unfreeze `u`/`delta`. Do
not restore raw global KV (that is E18).

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
