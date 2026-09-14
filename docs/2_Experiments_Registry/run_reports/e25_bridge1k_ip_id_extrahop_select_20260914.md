# E25 bridge_1k 1024 select_1decoy r=1 identity extra exclusive hop H=256 SSMax log — E21 vs E18 vs dense (rung 5ai)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1k_ip_id_extrahop_select` (800; floor — do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_extrahop_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_extrahop_select_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `c9fd85c` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 1024 SELECT SWA-unsever floor [`e25_bridge1k_ip_id_keepswa_select_20260914.md`](e25_bridge1k_ip_id_keepswa_select_20260914.md) · identity floor [`e25_bridge1k_ip_id_h256_select_20260914.md`](e25_bridge1k_ip_id_h256_select_20260914.md) · r=8 chance [`e25_bridge1k_ip_r8_h256_select_20260914.md`](e25_bridge1k_ip_r8_h256_select_20260914.md) · MATCH 60.35 bits [`e25_bridge1k_ip_r8_h256_recall_20260914.md`](e25_bridge1k_ip_r8_h256_recall_20260914.md)

---

## Goal

Locked: 1024 packed `select_1decoy` inplace r=1 identity remainder-off H=256 SSMax
log **S1 FAIL (0 / 0.01 bits @800)** while E18 copies ~64 bits. Pooling and
SWA-sever are not the killer. Exclusive global read is. One architectural
knob that is **not** E18 raw prefix KV: a second exclusive hop over the **same
frozen exclusive slots** (`--message_extra_slot_attends 1`, default 0).

Score vs 0.75× E18; if E18 were ~0, score vs 0.75× dense. Climbing at 800 →
extra-step 8k; **floor → do not extra-step**. Do **not** restore raw global KV.
Do not remainder-on. Do not H=512. Do not hops seq shrink. Do not Glyph. Do
not unfreeze `u`/`delta`.

## Inspection (authoritative)

### Slot K/V source at r=1 inplace identity

**Post-pre-SWA mixed states, not frozen/unmixed token embeddings.**

Stack: embed → `[swa] × pre_layers` (default 1, window 16) → `[full] ×
global_layers` → `[swa] × stack_layers`. At the global layer,
`Attention.kv_raw(x)` projects `x = attn_norm(mix(residual after pre-SWA, x0,
skip))`. Identity slots (`KVCompressor` with `identity_slots`) copy
`k_norm(k_raw), v` of that residual and bypass `u`/`delta`. They are **not**
raw embeddings. They are also **not** post-stack (stack is after the global
read). Test: `test_identity_kv_are_post_pre_swa_not_frozen_embeddings`.

Knob **A** (“if currently snapshotting K/V *before* local SWA, copy from
post-local”) is **already true** for pre-SWA. `message_keep_local_swa` only
changes the local SWA/n-gram mask (QUERY as document start). It does **not**
change which K/V go into exclusive slots. Extra hops reuse that same frozen
snapshot (`_extra_exclusive_attends`); they do not recompute K/V.

### Type-cue coverage

`select_1decoy`: `n_distractors=0`, `n_decoys=1`.
- Fact: `[keymark, k0, k1, v…]`
- Decoy: `[decoy, k0′, k1′, v…]` (keys are distinct)
- Tail: `[query, *query_key, answer, *value, end, eos]`
- `--evidence_align right` packs blocks against `hi` (just before min_gap).
  This hunt: seq=1024, gap=64, window=16, **row_gap 100/100/100**.

At r=1 inplace identity every sender token is a `replace` slot, including
`keymark` and `decoy`. They sit **before QUERY**, so they **do land in exclusive
slots**. Test: `test_select_1decoy_type_cues_land_in_r1_identity_slots`.
Coverage is not the hole (already shown by keepswa). Extra hops still hide
uncompressed remainder (`~replace`; `test_extra_slot_attends_hides_uncompressed_remainder`).

### MATCH vs SELECT at 1024

- MATCH `recall_single` r=8 rem-off H=256 SSMax log: **S1 PASS 60.35 bits**
  (one key→value; key+value co-located in a ~7-token block, window 16 mixes
  them; one exclusive read suffices).
- SELECT identity / identity+SWA / r=8: **chance**. Two blocks (`keymark` vs
  `decoy`). E18 raw global copies ~64 bits. Exclusive identity still chance.
  Hypothesis tested here: SELECT may need **two reads in slot space** (type
  then value) while MATCH needs one.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=1`** |
| Data | `--scale bridge_1k` seq=1024, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1k_ip_id_extrahop_select 0 \
  --scale bridge_1k --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --message_extra_slot_attends 1 --steps 800 --k1_mult 4
# no --message_pool_remainder
# no --message_keep_local_swa
# floor at 800: do not extra-step
```

Byobu `E25_1k_id_xhop`. Log:
`seq=1024  logit_scale=log  msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True  msg_keepswa=False  msg_extrahops=1`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON knobs: `hidden:
256`, `global_logit_scale: log`, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, **`message_extra_slot_attends: 1`**,
`message_override: real`. Recalibrated dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **100%** | 200 | **63.95** | 0.999 | 0.007807 |
| e18 | **100%** | 350 | **63.92** | 0.999 | 0.007803 |
| **e21 @800** | **25.7%** | 800 | **0.00** | **0.000** | **0** |
| e18_local | 24.9% | 800 | 0 | 0 | 0 |

Params: dense/e18 ~2.261M, e21 2.277M. Prize 64 bits. Chance ~25%. E21 CE at
ln(4)≈1.386 every eval except a spike at step 400 (CE 4.61, acc 0). Best acc
25.7% = chance. Not climbing.

S1 vs 0.75× E18: need **47.94 bits**. E21 has **0**. E18 is live (do not pass
S1 via 0.75×0). Content vs 0.75× dense: need **47.97 bits**. E21 has **0**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.95 bits @200. |
| **S1 vs 0.75× E18** | **FAIL.** 0 ≪ 47.94 bits. Chance floor. |
| **content vs 0.75× dense** | **FAIL.** 0 ≪ 47.97 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.9% / 0 bits. |
| **K3** | floor at 800 (flow 0). |

Do **not** extra-step to 8k (floor). Do not 16k. Do not remainder-on. Do not
restore raw global KV. Do not extra width. Do not hops seq shrink. Do not
Glyph. Do not unfreeze `u`/`delta`.

## Interpretation

**A second exclusive hop over frozen slots does not rescue 1024 SELECT.** Extra
`message_extra_slot_attends=1` still recovers **0 bits** (chance) while
uncompressed E18 copies **63.92 bits** on the same rows. Slots already carry
post-pre-SWA K/V; type-cue tokens already land in r=1 identity slots; the
second hop re-reads those same exclusive K/V (not raw prefix). MATCH at 1024
still needs only one exclusive read (60.35 bits). SELECT’s type-then-value
hypothesis is not saved by one extra exclusive hop. Do **not** relabel E18's
63.92 bits as E21.

## Decision

Keep the spec in `ahead/`. Default `--message_extra_slot_attends` stays **0**.
Default `--message_keep_local_swa` stays **off**. Next ONE: **sparse
exclusive-plus-anchors** leak of a few non-slot tokens (QUERY neighborhood or
type markers only), still not full raw prefix; min config flag default OFF;
same 1024 SELECT hunt. Not remainder-on. Not H=512. Not 8k on a chance floor.
Not hops seq shrink. Not Glyph. Do not unfreeze `u`/`delta`. Do not restore
raw global KV (that is E18). Remaining DNA walls: 512 chain **K1**, 256 hops
**FAIL**, 2048+ INDEX shared with E18.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
