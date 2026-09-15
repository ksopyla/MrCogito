# E25 bridge seq=272 chain_ordered --key_len 13 `--global_layers 2 --message_extra_slot_attends 1` — E21 hops S1 PASS @8k (rung 5bq)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_272_chain_k13_glob2_extrahop` (800 advertised; dense-matched 3200; climbing short of a stable S1 vs 0.75× dense) · `e25_272_chain_k13_glob2_extrahop_s8k` (8k extra-step; `--no-dense_first --k1_mult 1`; S1 PASS)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2_extrahop/` · `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2_extrahop_s8k/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2_extrahop/probe.log` · `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2_extrahop_s8k/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `2f4b4d8` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** seq=272 glob2 S1 FAIL without extra hop [`e25_bridge272_chain_k13_glob2_20260914.md`](e25_bridge272_chain_k13_glob2_20260914.md) · seq=272 glob2 keep_local_swa S1 PASS [`e25_bridge272_chain_k13_glob2_keepswa_20260915.md`](e25_bridge272_chain_k13_glob2_keepswa_20260915.md) · 1024 SELECT extra hop chance [`e25_bridge1k_ip_id_extrahop_select_20260914.md`](e25_bridge1k_ip_id_extrahop_select_20260914.md)

---

## Goal

Hops length extra-steps are stopped. MATCH glob=2 stacks stopped. SELECT
keepswa stopped. At seq=272 glob=2 identity: severed SWA E21 **S1 FAIL**
(8.87 @2850 then 0.02 @8k); `--message_keep_local_swa` **S1 PASS 25.55**.
glob=2 is already two exclusive attend+FFN layers.

Hypothesis: `--message_extra_slot_attends 1` (in-attention extra exclusive
hop over frozen slots, SWA still severed, keep_local_swa OFF) can substitute
for local SWA. If S1 PASS, extra exclusive hop replaces local token mixing.
If chance/fail, hops 272 specifically needs local SWA, not more global
attends over slots.

Default `--message_extra_slot_attends` stays **0**. Default keep_local_swa
stays **false**. Do **not** pass `--message_keep_local_swa`. Do **not**
`--message_update_slot_kv`. Do not glob=3. Do not window 32.

Recalibrate dense S0 at two global Blocks in the 800 JSON. Score e18/e21 if
dense S0 PASSes. If E18 ≈ 0, score S1 vs 0.75× dense (do not pass via 0.75×0).
800 floor: chance → no 8k; climbing short of S1 → 8k only. Do not 16k.
`--hops 1` is illegal. Do not relabel E18 as E21.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · **2.802M** (e21 **2.835M**) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=1`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 272`, **`--key_len 13`**, hops=2, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. Right-align gap 163/176/187 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.2–1.4GB, 0.04–0.05 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_272_chain_k13_glob2_extrahop 0 \
  --scale bridge --seq_len 272 --recipe chain_ordered --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --message_extra_slot_attends 1 \
  --steps 800 --k1_mult 4
# --seq_len 272, --global_layers 2, --message_extra_slot_attends 1 required
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_keep_local_swa (default false; SWA still severed)
# no --message_update_slot_kv
# no --message_global_anchors (default none)
# dense first (underscore --no-dense_first unused on the 800 hunt)
# climbing short of a stable S1 @3200 vs 0.75× dense → 8k extra-step:
bash scripts/e24_bapo_hunt.sh e25_272_chain_k13_glob2_extrahop_s8k 0 \
  --scale bridge --seq_len 272 --recipe chain_ordered --arch e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --message_extra_slot_attends 1 \
  --steps 8000 --k1_mult 1 --no-dense_first
# dense already S0 PASS 80.5% / 15.84 in the 800 JSON; underscore --no-dense_first
# do not 16k (S1 PASS)
```

Byobu `E25_272_chain_extrahop` then `E25_272_chain_extrahop_s8k`. Log:
`seq=272  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=163/176/187`
dense `patterns=[('full', 0)×5]  glob_layers=2` · 2.802M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False
msg_idslots=True  msg_keepswa=False  glob_layers=2  msg_extrahops=1
msg_updatekv=False  msg_anchors=none`.

800 hunt JSON: `seq_len: 272`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 2`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, **`message_extra_slot_attends: 1`**,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`hops: 2`, `prize_bits: 26.0`, `calibrated: true`, `dense_steps_used: 3200`.
Hunt exit **0**. 8k JSON same knobs, `steps: 8000`, `k1_mult: 1`, no dense arm,
`calibrated: true`, hunt exit **0**. Params <100M. Dense S0 recalibrated at two
full layers (2.802M). Code default `--message_extra_slot_attends` stays **0**.
Code default `--message_keep_local_swa` stays **false**.

## Training Outcome

### 800 advertised / dense-matched 3200 (`e25_272_chain_k13_glob2_extrahop`, exit 0)

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **80.5%** | 3200 | **15.84** | **0.609** |
| e18 | **45.4%** | 3200 | **10.17** | **0.391** |
| **e21 @800** | **28.0%** | 800 | **0.00** | **~0** |
| **e21 final** | **39.9%** | 3200 | **8.49** | **0.327** |
| e18_local | **27.6%** | 3200 | **0.02** | **~0** |

Dense crossed 75% at 2850 (75.7%), final **80.5% / 15.84 bits** @3200.
JSON `calibrated: true`. Hunt exit 0. **S0 PASS.** Recalibrated at
`global_layers=2`.

E18 is **live, not ~0** in this JSON: 45.4% / 10.17 bits @3200 (climbing;
CE off ln(4)). Do not pass S1 via 0.75×0. Score S1 vs **0.75× this-JSON
E18 7.63 bits**. Also report vs **0.75× dense 11.88 bits**.

E21 sat at chance through the advertised 800 floor (28.0% / CE 1.385).
Dense-matched budget continued to 3200. Climb started ~1750 (31.5%), first
≥40% @2700, best acc **46.0% @3000**, final **39.9% / 8.49 bits** @3200
(flow 0.327, CE 0.934). S1 vs 0.75× this-JSON E18 **7.63 bits**: thin
numerical pass (8.49) while both arms are still climbing. Content vs
0.75× dense **11.88 bits**: **FAIL** (8.49). Climbing, short of a stable
S1 → 8k extra-step. This 8.49-bit climb is the same shape as severed
glob2's unreplicated **8.87 @2850**; 8k distinguishes finish vs collapse.

### 8k extra-step (`e25_272_chain_k13_glob2_extrahop_s8k`, exit 0)

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| e18 | **99.0%** | 2550 | **25.49** | **0.980** |
| **e21** | **99.0%** | 5450 | **25.48** | **0.980** |
| e18_local | **22.5%** | 8000 | **0.00** | **0** |

Dense skipped (`--no-dense_first`; prior S0 **80.5% / 15.84 bits**). E18 is
**live, not ~0** in this JSON: chance through ~1400, first ≥40% @1450,
first ≥75% **86.2% @1700**, early-stop **99.0% / 25.49 bits** @2550. Do
not relabel E18 as E21. S1 vs **0.75× this-JSON live E18 19.12 bits**.

E21 sat at chance through 800 (27.6% / CE 1.386), then climbed: first ≥40%
@2050, first ≥75% **76.8% @3050**, 83.1% @3200, 90.3% @3750, early-stop
**99.0% / 25.48 bits** @5450 (flow 0.980). S1 vs 0.75× live E18 **19.12
bits**: **PASS** (25.48). Content vs 0.75× 800-JSON dense **11.88 bits**:
**PASS** (25.48). Exclusive identity with one extra frozen-slot hop matches
live E18 (25.49) and keepswa 272 (25.55). The 3200 8.49-bit climb
**replicated and finished**. Do not relabel the live E18 arm as E21.

**8k ran.** Do **not** 16k (S1 already PASS).

## Gates vs this rung

| gate | 800 / 3200 | 8k |
|---|---|---|
| **S0** | **PASS.** Dense 80.5% / 15.84 bits @3200 ≥ 75%. Recalibrated at `global_layers=2`. | dense skipped; prior S0 stands. |
| **S1 vs 0.75× live E18** | thin 8.49 ≥ 7.63 while E18 still climbing (45.4%). | **PASS.** 25.48 ≥ 19.12. E18 is live (25.49), not ~0. |
| **content vs 0.75× dense** | **FAIL.** 8.49 < 11.88 (climbing). | **PASS.** 25.48 ≥ 11.88. |
| **S2** plots | **PASS** (learning-curves / heatmap / recovered-bits / flow / bytes-per-token). | **PASS** (same). |
| **K1** | not triggered. | not triggered. |
| **K2** | **PASS.** `e18_local` 27.6% / 0.02 bits. Extra hop did not leak a full raw prefix. | **PASS.** `e18_local` 22.5% / 0 bits. |
| **K3** | not floor at 3200 (flow 0.327). | not triggered (flow 0.980). |

## Interpretation

**Packed 272 hops FAIL without keep_local_swa was not a hard requirement for
local SWA.** Same DNA + identity compressor + two global Blocks that scored
E21 **0.02 bits @8k** with QUERY treated as a SWA document start now
early-stop at **99.0% / 25.48 bits @5450** when an extra exclusive attend
over frozen slots is added (`msg_extrahops=1`) and local SWA stays severed
(`msg_keepswa=False`). Exclusive slots stay on the global read
(`msg_rawkv=False`, `msg_override=real`, `msg_updatekv=False`). Window 16
still cannot span gap 163–187. `e18_local` stays at chance (K2).

The extra hop is **not** a second exclusive Block (that is glob=2, already
on) and **not** unsevered SWA. It is an in-attention re-read of the same
frozen slot K/V with updated queries. At hops 272 that extra global attend
substitutes for the local token mixing keep_local_swa provided.

SELECT 1024 extra hop remains 0 bits. Extra exclusive hops are hops-specific
here (chain uses multi-hop composition). Do not reopen SELECT knobs. Do not
treat this as reopening the severed-SWA hops length map without the extra
hop: without extra hops or keep_local_swa, E21 hops wall at glob=2 still
stands at **(264 S1 PASS, 272 FAIL]**. With extra hops, 272 PASSES.

## Decision

Keep the spec in `ahead/`. **8k ran.** Do **not** 16k (S1 PASS). Do not hops
320. Do not MATCH glob=2. Do not SELECT 694. Do not glob=3 / window 32 /
`--message_update_slot_kv`. Do not Glyph. Do not unfreeze `u`/`delta`. Do
not restore full raw prefix KV (that is E18). Code default
`--message_extra_slot_attends` stays **0**. Code default
`--message_keep_local_swa` stays **false**. Code default `--global_layers`
stays 1. Next ONE (do not run): parent may later map hops length **with
extra hop** (not keepswa). If that map is not taken, extra hop remains the
272 rescue alongside keepswa.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
