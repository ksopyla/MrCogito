# E25 bridge_1k 1024 select_1decoy r=1 identity second exclusive global layer (`global_layers=2`) H=256 SSMax log — E21 vs E18 vs dense (rung 5am)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1k_ip_id_glob2_select` (800; floor — do not extra-step)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_glob2_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_glob2_select_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `046ac50` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** extra hop + unfrozen KV floor [`e25_bridge1k_ip_id_updatekv_select_20260914.md`](e25_bridge1k_ip_id_updatekv_select_20260914.md) · frozen extra hop [`e25_bridge1k_ip_id_extrahop_select_20260914.md`](e25_bridge1k_ip_id_extrahop_select_20260914.md)

---

## Goal

Locked: 1024 packed `select_1decoy` inplace r=1 identity remainder-off H=256 SSMax
log stays at chance with extra exclusive hops over frozen or rewritten slot K/V
(0 bits @800) while uncompressed E18 copies ~64 bits. Extra hops are two
sequential attends *inside one* Attention. One architectural knob that is **not**
E18 raw prefix KV: a **second exclusive global Attention+FFN Block** over slots
(`--global_layers 2`, extra hops 0, `update_slot_kv` off, anchors none). Distinct
from `--stack_layers` (already 2 on prior hunts; SWA local) and from
`--message_extra_slot_attends` (extra attends inside one Attention).

Score vs 0.75× live E18; if E18 were ~0, score vs 0.75× dense. Climbing at 800
→ extra-step 8k; **floor → do not extra-step**. Do **not** restore raw global
KV. Do not remainder-on. Do not H=512. Do not hops seq shrink. Do not Glyph.
Do not unfreeze `u`/`delta`. Do not relabel E18 as E21.

## Inspection (authoritative)

`--global_layers` already existed (default **1**, E18-loadable). No new config
field. `global_layers=2` is two sequential full Blocks after the pre-encoder:
`[swa] × pre` → `[full] × 2` → `[swa] × stack`. Each E21 full layer has its own
compressor and exclusive `~replace` mask. Tests:

- Default is one exclusive global
  (`test_default_global_layers_is_one_exclusive_block`).
- Two full exclusive Blocks, FFN between the two attends, extra hops stay 0,
  param count rises vs extra hops (which reuse one Attention)
  (`test_global_layers_two_are_two_exclusive_blocks_not_extra_hops`).
- Remainder still hidden
  (`test_global_layers_two_hides_uncompressed_remainder`).
- Factory: e21 two compressors; e18 two raw full layers; `e18_local` zero full
  (`test_e21_global_layers_two_is_wired_on_factory`).

Still not full prefix.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.802M (e21 2.835M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge_1k` seq=1024, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~3.5GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1k_ip_id_glob2_select 0 \
  --scale bridge_1k --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --global_layers 2 --steps 800 --k1_mult 4
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# no --message_global_anchors (default none)
# floor at 800: do not extra-step
```

Byobu `E25_1k_id_glob2`. Log: e21
`patterns=[('swa', 16), ('full', 0), ('full', 0), ('swa', 16), ('swa', 16)]`
`glob_layers=2  msg_extrahops=0  msg_updatekv=False  msg_anchors=none`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON knobs: `hidden:
256`, `global_logit_scale: log`, **`global_layers: 2`**, `stack_layers: 2`,
`message_ratio: 1`, `message_pool_remainder: false`,
`message_slots_inplace: true`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`, `message_keep_local_swa: false`,
`message_extra_slot_attends: 0`, `message_update_slot_kv: false`,
`message_global_anchors: none`, `message_override: real`. Recalibrated dense
S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense | **100%** | 200 | **63.95** | 0.999 | 0.007806 |
| e18 | **100%** | 750 | **63.97** | 0.999 | 0.007808 |
| **e21 @800** | **25.7%** | 800 | **0.00** | **0.000** | **0** |
| e18_local | 24.9% | 800 | 0 | 0 | 0 |

Params: dense/e18 ~2.802M, e21 2.835M (<100M). Prize 64 bits. Chance ~25%.
E21 CE at ln(4)≈1.386 every eval (min 1.3862). Best acc 25.7% = chance. Not
climbing.

S1 vs 0.75× E18: need **47.97 bits**. E21 has **0**. E18 is live (do not pass
S1 via 0.75×0). Content vs 0.75× dense: need **47.96 bits**. E21 has **0**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.95 bits @200. |
| **S1 vs 0.75× live E18** | **FAIL.** 0 ≪ 47.97 bits. Chance floor. |
| **content vs 0.75× dense** | **FAIL.** 0 ≪ 47.96 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.9% / 0 bits. |
| **K3** | floor at 800 (flow 0). |

Do **not** extra-step to 8k (floor). Do not 16k. Do not remainder-on. Do not
restore raw global KV. Do not extra width. Do not hops seq shrink. Do not
Glyph. Do not unfreeze `u`/`delta`.

## Interpretation

**A second exclusive global attend+FFN Block does not rescue 1024 SELECT.**
`--global_layers 2` (extra hops 0) still recovers **0 bits** (chance) while
uncompressed E18 copies **63.97 bits** on the same rows. This is not a re-run
of `stack_layers=2` (those stay SWA) and not extra hops inside one Attention.
Two full exclusive reads with an FFN between them still leave type-then-value
across two blocks dead at 1024. MATCH at 1024 still needs only one exclusive
read (60.35 bits). Do **not** relabel E18's 63.97 bits as E21.

## Decision

Keep the spec in `ahead/`. Default `--global_layers` stays **1**. Default
`--message_extra_slot_attends` stays **0**. Default `--message_update_slot_kv`
stays **off**. Default `--message_keep_local_swa` stays **off**. Default
`--message_global_anchors` stays **`none`**. Next ONE (do not run): **stop
1024 SELECT architecture hunts** (do not 8k). Remaining DNA walls: 512 chain
**K1**, 256 hops **FAIL**, 2048+ INDEX shared with E18. Not remainder-on. Not
H=512. Not hops seq shrink. Not Glyph. Do not unfreeze `u`/`delta`. Do not
restore raw global KV (that is E18).

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
