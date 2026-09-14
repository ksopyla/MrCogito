# E25 bridge_1k 1024 select_1decoy r=1 identity type_marks anchors H=256 SSMax log — E21 vs E18 vs dense (rung 5aj)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1k_ip_id_anchors_select` (advertised 800; dense late-clicked so other arches ran to 1050). Replica `e25_1k_ip_id_anchors_e18` (e18-only, 800).
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_anchors_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_anchors_select_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `9be40ff` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** extra exclusive hop floor [`e25_bridge1k_ip_id_extrahop_select_20260914.md`](e25_bridge1k_ip_id_extrahop_select_20260914.md) · SWA-unsever [`e25_bridge1k_ip_id_keepswa_select_20260914.md`](e25_bridge1k_ip_id_keepswa_select_20260914.md) · identity floor [`e25_bridge1k_ip_id_h256_select_20260914.md`](e25_bridge1k_ip_id_h256_select_20260914.md)

---

## Goal

Locked: 1024 packed `select_1decoy` inplace r=1 identity remainder-off H=256 SSMax
log stays at **chance after r=8 mean, identity, keep_local_swa, and extra exclusive
hop**, while prior-hunt E18 copies ~64 bits. Knob **C**: leak a *few* non-slot
tokens into the exclusive global read (QUERY neighborhood and/or type markers),
still **not** the full raw prefix (that is E18).

One flag: `--message_global_anchors {none,query_nbhd,type_marks,query_nbhd+type}`
(default `none`). This hunt used **`type_marks`** — DNA `keymark` + `decoy`
(2 tokens vs seq=1024). Extra hop 0. `keep_local_swa` off. Remainder off.
Identity inplace r=1.

Score vs 0.75× E18; if E18 is ~0, score vs 0.75× dense. Do **not** pass S1 via
0.75×0. Advertised **800-step floor**; **no 8k if chance at 800**. Do not restore
raw global KV. Do not remainder-on. Do not H=512. Do not hops seq shrink. Do not
Glyph. Do not unfreeze `u`/`delta`. Do not relabel E18 as E21.

## Inspection (authoritative)

### What `type_marks` leaks

`--message_global_anchors type_marks` adds sender positions whose token id is in
`{keymark, decoy, spanmark, hop, mark}` to exclusive slot K/V as **raw** keys.
`select_1decoy` (`n_distractors=0`, `n_decoys=1`) emits **one `keymark` and one
`decoy`** in the sender prefix (spanmark/hop/mark count 0). Measured on 16
right-aligned `bridge_1k` rows: **anchor count = 2**, seq=1024, **2 << 1024**.

QUERY neighborhood (`query_nbhd`, window 4) would leak 4 tokens immediately
before QUERY; unused this hunt (smallest type-then-value leak is the two type
marks).

### r=1 identity: extra non-slot count is 0

At `--message_ratio 1 --message_slots_inplace --message_identity_slots`, every
sender token is already a `replace` slot, including `keymark` and `decoy`.
`exclusive_visible = replace | anchor` equals `replace`. The 2 type-mark tokens
are a sparse subset of seq, but they are **not extra non-slot positions** on this
rung. Coverage of type cues in identity slots was already shown by
`test_select_1decoy_type_cues_land_in_r1_identity_slots`. Anchors are a real extra
leak only when they sit in uncompressed remainder (proven at r>1 in
`test_type_mark_anchors_join_exclusive_kv_not_full_prefix_inplace`).

Slots remain post-pre-SWA mixed states, not frozen embeddings.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_anchors=type_marks`** |
| Data | `--scale bridge_1k` seq=1024, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1k_ip_id_anchors_select 0 \
  --scale bridge_1k --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --message_global_anchors type_marks --steps 800 --k1_mult 4
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# dense first (underscore --no-dense_first unused)
```

Byobu `E25_1k_id_anchors`. Log:
`seq=1024  logit_scale=log  msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True  msg_keepswa=False  msg_extrahops=0  msg_anchors=type_marks`.

Hunt JSON knobs: `hidden: 256`, `global_logit_scale: log`, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
**`message_global_anchors: type_marks`**, `message_override: real`,
`attn_backend: sdpa`, `warm_residuals: true`, `seq_len: 1024`, `min_gap: 64`,
`local_window: 16`, `kv_heads: 1`, `stack_layers: 2`. Recalibrated dense S0 in
the same JSON. Dense plateaued ~87–88% through 950 then hit 100% at 1050, so
`dense_steps_used=1050` and e18/e21/`e18_local` ran to 1050.

e18-only replica (Byobu `E25_1k_id_anchors_e18`):
`--no-dense_first --arch e18 --steps 800` same flags → still **0 bits**.

## Training Outcome

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **100%** | 1050 | **63.98** | 0.9997 | 0.007810 |
| e18 | 24.9% | 1050 | **0** | 0 | 0 |
| **e21 @800** | **25.7%** | 800 | **0.00** | **0.000** | **0** |
| **e21 @1000** | **50.5%** | 1000 | **29.21** | 0.456 | 0.003566 |
| **e21 @1050** | **48.1%** | 1050 | **31.90** | 0.4985 | 0.003895 |
| e18_local | 25.7% | 1050 | 0 | 0 | 0 |

Params: dense/e18 ~2.261M, e21 2.277M (<100M). Prize 64 bits. Chance ~25%.
E21 CE at ln(4)≈1.386 through step 950, then 0.754 @1000 / 0.695 @1050.
Best acc 50.5% @1000. This-JSON E18 chance the entire trace (and the 800-step
e18-only replica). Prior same-recipe E18 on `c9fd85c` was **63.92 bits @350** —
do not treat this replica's 0 as a new E18 ceiling.

S1 vs 0.75× this-JSON E18: **invalid** (E18=0; do not pass via 0.75×0).
S1 vs 0.75× dense: need **47.99 bits**. E21 has **31.90** at 1050 and **0** at 800.
S1 vs 0.75× prior-hunt E18 63.92: need **47.94 bits**. E21 has **31.90**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.98 bits @1050. |
| **S1 vs 0.75× this-JSON E18** | **invalid.** E18 0 bits. Do not pass via 0.75×0. |
| **S1 vs 0.75× dense** | **FAIL.** 31.90 < 47.99 at 1050; **0** at the 800 floor. |
| **S1 vs 0.75× prior live E18** | **FAIL.** 31.90 < 47.94. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.7% / 0 bits. |
| **K3** | chance at advertised **800** (flow 0). Late click after 950 is not a 800-floor pass. |

Do **not** extra-step to 8k on this turn (800-floor chance). Do not 16k. Do not
remainder-on. Do not restore raw global KV. Do not extra width. Do not hops seq
shrink. Do not Glyph. Do not unfreeze `u`/`delta`. Do not relabel E18 as E21.

## Interpretation

**Sparse type-mark anchors do not clear 1024 SELECT S1.** At the registered 800
floor E21 is still chance (0 bits), matching identity / keepswa / extra-hop.
Because dense late-clicked, the hunt continued to 1050 and E21 produced the
**first non-zero exclusive 1024 SELECT bits** (31.90, flow 0.50) after
chance-through-950. That is below 0.75× dense (47.99) and below 0.75× prior live
E18 (47.94). This-JSON E18 also missed (0 bits); score vs dense / prior E18, not
via 0.75×0.

At r=1 identity the type-mark leak adds **zero extra non-slot keys** — the click
is **not** evidence that a remainder leak rescued SELECT. It is a late-budget
signal on the same exclusive identity channel that was killed at 800 on earlier
replicas. Do **not** relabel E18's prior 63.92 bits as E21.

## Decision

Keep the spec in `ahead/`. Default `--message_global_anchors` stays **`none`**.
Default extra hops stay 0. Default `keep_local_swa` stays off. Next ONE:
**extra-step this late click to 8k** (`--arch e18 e21 e18_local --steps 8000
--no-dense_first --k1_mult 1`, same `type_marks` + identity flags) so S1 has a
live uncompressed control and we learn whether 31.90 climbs. Not remainder-on.
Not H=512. Not raw global KV. Not Glyph. Do not unfreeze `u`/`delta`. Remaining
DNA walls: 512 chain **K1**, 256 hops **FAIL**, 2048+ INDEX shared with E18.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
