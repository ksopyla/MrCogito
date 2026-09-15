# E25 bridge_1k 1024 select_1decoy r=1 identity type_marks anchors 8k extra-step — live E18 control (rung 5ak)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1k_ip_id_anchors_select_s8k` (8000; `--arch e18 e21 e18_local --no-dense_first --k1_mult 1`)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_anchors_select_s8k/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_ip_id_anchors_select_s8k_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `bdc32ed` (Odra E24 worktree HEAD at launch; probe code still `9be40ff`)
**Git tag:** —
**Related:** 800/1050 type_marks hunt [`e25_bridge1k_ip_id_anchors_select_20260914.md`](e25_bridge1k_ip_id_anchors_select_20260914.md)

---

## Goal

Locked: 1024 packed `select_1decoy` inplace r=1 identity remainder-off H=256 SSMax
log `--message_global_anchors type_marks` late-clicked to **31.90 bits / ~50%**
when the dense-first hunt ran to 1050, after a **chance floor @800**. This-JSON
E18 in that hunt was **0**; prior live E18 on `c9fd85c` was **63.92**. Hypothesis:
the 800-floor was **too short**. Extra-step to **8000** with a **live E18
control** (`--no-dense_first`; dense already S0 PASS **63.98**). At r=1 identity,
type_marks leak is a **no-op** (keymark+decoy already replace slots). Do **not**
claim anchors rescued SELECT.

Score vs 0.75× **live** E18; if live E18 were ~0, vs 0.75× dense 63.98. Do **not**
pass S1 via 0.75×0. Do not restore raw global KV. Do not remainder-on. Do not
H=512. Do not hops seq shrink. Do not Glyph. Do not unfreeze `u`/`delta`. Do not
relabel E18 as E21.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: e18 / e21 / e18_local (dense skipped; prior S0 63.98) |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_anchors=type_marks`** |
| Data | `--scale bridge_1k` seq=1024, gap=64, window=16, packed answer 32 / **64-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 100/100/100) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~2.8GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1k_ip_id_anchors_select_s8k 0 \
  --scale bridge_1k --recipe select_1decoy --arch e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --message_global_anchors type_marks --steps 8000 --no-dense_first --k1_mult 1
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# dense already S0 PASS 63.98; underscore --no-dense_first used
```

Byobu `E25_1k_id_anchors_s8k`. Log:
`seq=1024  logit_scale=log  msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True  msg_keepswa=False  msg_extrahops=0  msg_anchors=type_marks`.

Hunt JSON knobs: `hidden: 256`, `global_logit_scale: log`, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
**`message_global_anchors: type_marks`**, `message_override: real`,
`attn_backend: sdpa`, `warm_residuals: true`, `seq_len: 1024`, `min_gap: 64`,
`local_window: 16`, `kv_heads: 1`, `stack_layers: 2`. Live E18 recalibrated in
this JSON. Dense not rerun.

## Training Outcome

| arch | acc | step | recovered bits | flow | bytes/input token |
|---|---|---|---|---|---|
| dense (prior hunt) | **100%** | 1050 | **63.98** | 0.9997 | 0.007810 |
| **e18** | **100%** | **3250** | **63.96** | **0.999** | **0.007808** |
| e21 @800 | 25.7% | 800 | **0.00** | 0 | 0 |
| e21 @1050 | 25.7% | 1050 | **0.00** | 0 | 0 |
| e21 climb | 30.7% | 4850 | ~3.21 | — | — |
| e21 best bits | 42.0% | 7100 | **14.43** | — | — |
| e21 best acc | **42.9%** | 7850 | (CE 1.074) | — | — |
| **e21 @8000** | **39.3%** | **8000** | **6.48** | **0.101** | **0.000791** |
| e18_local | 25.7% | 8000 | 0 | 0 | 0 |

Params: e18 2.261M, e21 2.277M, e18_local 2.261M (<100M). Prize 64 bits. Chance
~25%. E18 clicked to 85.9% @350, plateaued ~88% through 1400, dipped ~74%
@1450–1800, then **100% / 63.96 bits @3250** (early stop). This-JSON E18 is
**live** — the 1050-hunt 0-bit replica was short budget, not a new E18 ceiling.

E21 CE at ln(4)≈1.386 through step 4800 (chance at 800 **and** 1050; the prior
31.90 @1050 **did not replicate**). First climb @4850 (30.7% / ~3.21 bits).
Plateau ~42% / CE≈1.074 from 6050–7950 (best acc **42.9% @7850**; best recovered
**14.43 bits @7100**). Final eval CE spiked to 1.246 → **6.48 bits @8000**.
One train spike @350 (train 7.61) recovered to the floor.

S1 vs 0.75× **live** E18: need **47.97 bits**. E21 has **6.48** at 8000 and
**14.43** at best. Did **not** climb past the prior 31.90.

S1 vs 0.75× dense 63.98: need **47.99 bits**. E21 has **6.48**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS** (prior JSON). Dense 100% / 63.98 bits @1050. Not rerun. |
| **S1 vs 0.75× live E18** | **FAIL.** 6.48 (best 14.43) < 47.97. Live E18 **63.96**, not 0. |
| **S1 vs 0.75× dense** | **FAIL.** 6.48 < 47.99. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.7% / 0 bits. |
| **K3** | chance at 800 (flow 0). 8k is a **plateau**, not a slow INDEX climb. |

Do **not** extra-step to 16k (plateau 6050–8000). Do not remainder-on. Do not
restore raw global KV. Do not extra width. Do not hops seq shrink. Do not Glyph.
Do not unfreeze `u`/`delta`. Do not relabel E18 as E21.

## Interpretation

**8k did not turn the 31.90 late click into an S1 pass.** With a live
uncompressed control (**E18 63.96 bits @3250**), exclusive identity SELECT at
1024 plateaus around **42% / 14 bits**, then finishes at **6.48**. The prior
1050 replica's 31.90 did not repeat at the same step (this run still chance
through 1050; first climb @4850). Type-mark leak remains a no-op at r=1
identity. Budget was the question; the answer is **not S1**.

## Decision

Keep the spec in `ahead/`. Default `--message_global_anchors` stays **`none`**.
Default extra hops stay 0. Default `keep_local_swa` stays off. Next ONE:
**stop 1024 SELECT extra-steps** (do not 16k). Remaining DNA walls: 512 chain
**K1**, 256 hops **FAIL**, 2048+ INDEX shared with E18. Not remainder-on. Not
H=512. Not raw global KV. Not Glyph. Do not unfreeze `u`/`delta`.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
