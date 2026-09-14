# E25 bridge_1k 1024 select_1decoy r=1 identity QUERY-side anchors (`query_side`) H=256 SSMax log — E21 vs E18 vs dense (rung 5an)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1k_ip_id_qside_select` (800; floor — do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_qside_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_qside_select_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `264d19f` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** second exclusive global layer floor [`e25_bridge1k_ip_id_glob2_select_20260914.md`](e25_bridge1k_ip_id_glob2_select_20260914.md) · type_marks no-op [`e25_bridge1k_ip_id_anchors_select_20260914.md`](e25_bridge1k_ip_id_anchors_select_20260914.md)

---

## Goal

Locked: 1024 packed `select_1decoy` inplace r=1 identity remainder-off H=256 SSMax
log stays at **0 bits after all attend-stack knobs** while MATCH at the same
scale S1-passes (key lives in the prefix) and uncompressed E18 copies ~64 bits.
Hypothesis: SELECT's **type request sits at/after QUERY**, so exclusive **prefix**
slots never contain the question type; E18's global read includes QUERY tokens.
Knob C `type_marks` leaked `keymark`+`decoy` which are **already r=1 replace
slots** (extra non-slot count 0). Existing `query_nbhd` is 4 sender tokens
*before* QUERY — also r=1 slots. This hunt is the other Knob C option:
`--message_global_anchors query_side` leaks QUERY plus a small window **after**
the message boundary, still not the full raw prefix.

Score vs 0.75× live E18; if E18 were ~0, score vs 0.75× dense. Climbing at 800
→ extra-step 8k; **floor → do not extra-step**. Do **not** restore raw global
KV. Do not remainder-on. Do not H=512. Do not hops seq shrink. Do not Glyph.
Do not unfreeze `u`/`delta`. Do not relabel E18 as E21.

## Inspection (authoritative)

`select_1decoy` tail is `[QUERY] [key₀] [key₁] [ANSWER] y…`. The asked key
(`query_len = key_len = 2`) lives **after** QUERY. Prefix `keymark`/`decoy` are
already identity slots at r=1.

`--message_global_anchors query_side` (default-off; window 4) marks QUERY and
the next 3 same-document receiver tokens. On 16 right-aligned `bridge_1k` rows
the leak is always:

| position | token | already r=1 prefix replace slot? |
|---|---|---|
| QUERY | control `query` (id 10) | no (side ≥ 1) |
| QUERY+1 | asked-key symbol | no |
| QUERY+2 | asked-key symbol | no |
| QUERY+3 | control `answer` | no |

**Count = 4 vs seq=1024 (4 ≪ 1024).** Extra non-slot count = 4. Existing
`query_nbhd` at the same r=1 identity recipe has extra count **0** (prefix
tokens before QUERY are already replace slots). Default `none` is unchanged
(E18-loadable; no new params). Concat exclusive mask still hides the raw
prefix (`concat[QUERY, :QUERY]` all False). Still not E18 raw prefix KV.

Tests: `test_query_side_anchors_leak_receiver_not_prefix_slots`,
`test_select_1decoy_query_side_anchors_are_not_r1_prefix_slots`.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`glob_layers=1`**, **`msg_anchors=query_side`** |
| Data | `--scale bridge_1k` seq=1024, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~2.9GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1k_ip_id_qside_select 0 \
  --scale bridge_1k --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --message_global_anchors query_side --steps 800 --k1_mult 4
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# no --global_layers (default 1)
# dense first (underscore --no-dense_first unused)
```

Byobu `E25_1k_id_qside`. Log: e21
`msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True
msg_keepswa=False  glob_layers=1  msg_extrahops=0  msg_updatekv=False
msg_anchors=query_side`.

Hunt JSON knobs: `hidden: 256`, `global_logit_scale: log`, `global_layers: 1`,
`stack_layers: 2`, `message_ratio: 1`, `message_pool_remainder: false`,
`message_slots_inplace: true`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`, `message_keep_local_swa: false`,
`message_extra_slot_attends: 0`, `message_update_slot_kv: false`,
**`message_global_anchors: query_side`**, `message_override: real`. Recalibrated
dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **100%** | 250 | **63.96** | 0.999 | 0.007807 |
| e18 | **75.1%** | 800 | **46.77** | 0.731 | 0.005709 |
| **e21 @800** | **25.7%** | 800 | **0.00** | **0.000** | **0** |
| e18_local | 24.9% | 800 | 0 | 0 | 0 |

Params: dense/e18 ~2.261M, e21 2.277M (<100M). Prize 64 bits. Chance ~25%.
E21 CE at ln(4)≈1.386 every eval (min 1.3864). Best acc 25.7% = chance. Not
climbing. E18 was chance through 750 and late-clicked at 800 (75.1% / 46.77
bits) — live, not 0; weaker than prior-hunt E18 ~64 bits.

S1 vs 0.75× this-JSON E18: need **35.07 bits**. E21 has **0**. E18 is live (do
not pass S1 via 0.75×0). Content vs 0.75× dense: need **47.97 bits**. E21 has
**0**. Vs 0.75× prior live E18 63.97: need **47.98 bits**. E21 has **0**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.96 bits @250. |
| **S1 vs 0.75× this-JSON E18** | **FAIL.** 0 ≪ 35.07 bits. Chance floor. |
| **content vs 0.75× dense** | **FAIL.** 0 ≪ 47.97 bits. |
| **S1 vs 0.75× prior live E18** | **FAIL.** 0 ≪ 47.98 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.9% / 0 bits. |
| **K3** | floor at 800 (flow 0). |

Do **not** extra-step to 8k (floor). Do not 16k. Do not remainder-on. Do not
restore raw global KV. Do not extra width. Do not hops seq shrink. Do not
Glyph. Do not unfreeze `u`/`delta`.

## Interpretation

**QUERY-side neighborhood leak does not rescue 1024 SELECT.** Extra non-slot
count is **4** (QUERY + 2 asked-key symbols + ANSWER), unlike `type_marks` /
`query_nbhd` whose extra count is 0 at r=1 identity. E21 still recovers
**0 bits** (chance every eval) while this-JSON E18 late-clicks to **46.77
bits** and dense copies **63.96**. The type request is now in the exclusive
anchor set and was already on the receiver raw path of the inplace read
(same-side tokens after QUERY). Putting those four tokens into exclusive
slot K/V does not bind type-then-value at 1024. MATCH at 1024 still needs
only prefix keys (60.35 bits). Do **not** relabel E18's bits as E21.

## Decision

Keep the spec in `ahead/`. Default `--message_global_anchors` stays **`none`**.
Default extra hops stay 0. Default `keep_local_swa` stays off. Default
`--global_layers` stays 1. Default `--message_update_slot_kv` stays off.
Next ONE (do not run): **stop 1024 SELECT architecture hunts** (do not 8k).
Knob C is measured: type_marks extra 0, query_side extra 4, both fail S1.
Remaining DNA walls: 512 chain **K1**, 256 hops **FAIL**, 2048+ INDEX shared
with E18. Not remainder-on. Not H=512. Not hops seq shrink. Not Glyph. Do
not unfreeze `u`/`delta`. Do not restore raw global KV (that is E18).

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
