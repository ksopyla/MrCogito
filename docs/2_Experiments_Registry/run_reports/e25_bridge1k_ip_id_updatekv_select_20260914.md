# E25 bridge_1k 1024 select_1decoy r=1 identity extra hop + unfrozen slot K/V H=256 SSMax log — E21 vs E18 vs dense (rung 5al)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1k_ip_id_updatekv_select` (800; floor — do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_updatekv_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_updatekv_select_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `c54e08b` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** frozen extra hop floor [`e25_bridge1k_ip_id_extrahop_select_20260914.md`](e25_bridge1k_ip_id_extrahop_select_20260914.md) · type_marks 8k [`e25_bridge1k_ip_id_anchors_select_s8k_20260914.md`](e25_bridge1k_ip_id_anchors_select_s8k_20260914.md)

---

## Goal

Locked: 1024 packed `select_1decoy` inplace r=1 identity remainder-off H=256 SSMax
log stays at chance with extra exclusive hops over **frozen** slot K/V (0 bits
@800) while uncompressed E18 copies ~64 bits. Type_marks 8k extra-step also
failed S1 (6.48 vs live E18 63.96). One architectural knob that is **not** E18
raw prefix KV: rewrite exclusive slot K/V from the post-attend residual before
the next exclusive hop (`--message_extra_slot_attends 1 --message_update_slot_kv`,
default off).

Score vs 0.75× live E18; if E18 were ~0, score vs 0.75× dense. Climbing at 800
→ extra-step 8k; **floor → do not extra-step**. Do **not** restore raw global
KV. Do not remainder-on. Do not H=512. Do not hops seq shrink. Do not Glyph.
Do not unfreeze `u`/`delta`. Do not relabel E18 as E21.

## Inspection (authoritative)

`--message_extra_slot_attends` **is** two sequential attends with updated Q and
frozen K/V. Not a no-op bug.

- `_extra_exclusive_attends`: hop 2 Q is `wq(q_norm(x + wo(attend_1)))`; K/V
  stay the first-hop snapshot (`is` identity in tests).
- Extra=1 vs extra=0 logits differ at receiver positions
  (`test_extra_slot_attends_default_off_is_byte_identical_and_param_matched`).
- Q at QUERY changes between hops
  (`test_extra_slot_attends_updates_q_at_query_over_frozen_kv`).
- Hop-2 Q at QUERY is sensitive to a sender identity slot, so type **can**
  enter Q after hop 1 (`test_extra_hop_query_q_can_contain_sender_slot_content`).
- The measured frozen-hop wall is frozen **K/V**, not a dead second attend.

This hunt's knob (`message_update_slot_kv`): rewrite exclusive slot K/V from
that post-attend residual before hop 2. Non-slot token K/V stay the first-hop
snapshot. Exclusive `~replace` still hides uncompressed remainder. Still not
full raw prefix.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=1`**, **`msg_updatekv=True`**, **`msg_anchors=none`** |
| Data | `--scale bridge_1k` seq=1024, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~2.9GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1k_ip_id_updatekv_select 0 \
  --scale bridge_1k --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --message_extra_slot_attends 1 --message_update_slot_kv --steps 800 --k1_mult 4
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_global_anchors (default none)
# floor at 800: do not extra-step
```

Byobu `E25_1k_id_updatekv`. Log:
`seq=1024  logit_scale=log  msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True  msg_keepswa=False  msg_extrahops=1  msg_updatekv=True  msg_anchors=none`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON knobs: `hidden:
256`, `global_logit_scale: log`, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, **`message_extra_slot_attends: 1`**,
**`message_update_slot_kv: true`**, `message_global_anchors: none`,
`message_override: real`. Recalibrated dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **100%** | 250 | **63.95** | 0.999 | 0.007807 |
| e18 | **100%** | 450 | **63.80** | 0.997 | 0.007788 |
| **e21 @800** | **25.7%** | 800 | **0.00** | **0.000** | **0** |
| e18_local | 24.9% | 800 | 0 | 0 | 0 |

Params: dense/e18 ~2.261M, e21 2.277M (<100M). Prize 64 bits. Chance ~25%.
E21 CE at ln(4)≈1.386 every eval (min 1.3862). Best acc 25.7% = chance. Not
climbing.

S1 vs 0.75× E18: need **47.85 bits**. E21 has **0**. E18 is live (do not pass
S1 via 0.75×0). Content vs 0.75× dense: need **47.96 bits**. E21 has **0**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.95 bits @250. |
| **S1 vs 0.75× live E18** | **FAIL.** 0 ≪ 47.85 bits. Chance floor. |
| **content vs 0.75× dense** | **FAIL.** 0 ≪ 47.96 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.9% / 0 bits. |
| **K3** | floor at 800 (flow 0). |

Do **not** extra-step to 8k (floor). Do not 16k. Do not remainder-on. Do not
restore raw global KV. Do not extra width. Do not hops seq shrink. Do not
Glyph. Do not unfreeze `u`/`delta`.

## Interpretation

**Rewriting exclusive slot K/V between hops does not rescue 1024 SELECT.**
Extra hop + `--message_update_slot_kv` still recovers **0 bits** (chance)
while uncompressed E18 copies **63.80 bits** on the same rows. The extra hop
is a real second exclusive attend with updated Q (QUERY Q can contain type);
unfreezing slot K/V from the post-attend residual still leaves type-then-value
across two blocks dead at 1024. MATCH at 1024 still needs only one exclusive
read (60.35 bits). Do **not** relabel E18's 63.80 bits as E21.

## Decision

Keep the spec in `ahead/`. Default `--message_update_slot_kv` stays **off**.
Default `--message_extra_slot_attends` stays **0**. Default
`--message_keep_local_swa` stays **off**. Default `--message_global_anchors`
stays **`none`**. Next ONE (do not run): **stop 1024 SELECT architecture
hunts** (do not 8k). Remaining DNA walls: 512 chain **K1**, 256 hops **FAIL**,
2048+ INDEX shared with E18. Not remainder-on. Not H=512. Not hops seq shrink.
Not Glyph. Do not unfreeze `u`/`delta`. Do not restore raw global KV (that is
E18). If SELECT is hunted again, the unused next knob is a **second exclusive
global layer** (`global_layers=2`, extra hops 0, update_slot_kv off) — a full
block (norm+attn+MLP) over exclusive slots, not a second attend inside one
Attention.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
