# E25 bridge seq=272 chain_ordered --key_len 13 `--global_layers 2 --message_keep_local_swa` — E21 hops S1 PASS @8k (rung 5bm)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_272_chain_k13_glob2_keepswa` (800 advertised; dense-matched 3200; climbing short of S1 vs 0.75× dense) · `e25_272_chain_k13_glob2_keepswa_s8k` (8k extra-step; `--no-dense_first --k1_mult 1`; S1 PASS)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2_keepswa/` · `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2_keepswa_s8k/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2_keepswa/probe.log` · `/opt/cursor/artifacts/e25_bridge272_chain_k13_glob2_keepswa_s8k/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `85a2eaf` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** seq=272 glob2 S1 FAIL without keep_local_swa [`e25_bridge272_chain_k13_glob2_20260914.md`](e25_bridge272_chain_k13_glob2_20260914.md) · seq=264 glob2 S1 PASS [`e25_bridge264_chain_k13_glob2_20260914.md`](e25_bridge264_chain_k13_glob2_20260914.md) · 1024 SELECT keep_local_swa chance [`e25_bridge1k_ip_id_keepswa_select_20260914.md`](e25_bridge1k_ip_id_keepswa_select_20260914.md)

---

## Goal

Hops length extra-steps are stopped. E21 hops wall at glob=2 (severed SWA) is
**(264 S1 PASS, 272 FAIL]**. At 272 without `--message_keep_local_swa`: dense
S0 PASS, live E18 **25.35 @1650 / 23.64 @8k**, E21 **8.87 @2850 then 0.02 @8k**
unreplicated. At 256 glob=2, E21 beat live E18 without keep_local_swa.

Hypothesis: seq=272 FAIL is **SWA sever** starving local hop assembly, not
exclusive global capacity. Test `--message_keep_local_swa` at the same 272
glob=2 identity compressor. This is hops-specific (chain uses local
structure). SELECT 1024 keep_local_swa already failed — not a SELECT knob
reopen. Default `--message_keep_local_swa` stays **false**.

Recalibrate dense S0 at two global Blocks in the 800 JSON. Score e18/e21 if
dense S0 PASSes. If E18 ≈ 0, score S1 vs 0.75× dense (do not pass via 0.75×0).
800 floor: chance → no 8k; climbing short of S1 → 8k only. Do not 16k.
`--hops 1` is illegal. Do not relabel E18 as E21.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · **2.802M** (e21 **2.835M**) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=True`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 272`, **`--key_len 13`**, hops=2, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. Right-align gap 163/176/187 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.2GB, 0.04–0.05 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_272_chain_k13_glob2_keepswa 0 \
  --scale bridge --seq_len 272 --recipe chain_ordered --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --message_keep_local_swa \
  --steps 800 --k1_mult 4
# --seq_len 272, --global_layers 2, --message_keep_local_swa required
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# no --message_global_anchors (default none)
# dense first (underscore --no-dense_first unused on the 800 hunt)
# climbing short of S1 @3200 vs 0.75× dense → 8k extra-step:
bash scripts/e24_bapo_hunt.sh e25_272_chain_k13_glob2_keepswa_s8k 0 \
  --scale bridge --seq_len 272 --recipe chain_ordered --arch e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --message_keep_local_swa \
  --steps 8000 --k1_mult 1 --no-dense_first
# dense already S0 PASS 94.8% / 23.53 in the 800 JSON; underscore --no-dense_first
# do not 16k (S1 PASS)
```

Byobu `E25_272_chain_keepswa` then `E25_272_chain_keepswa_s8k`. Log:
`seq=272  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=163/176/187`
dense `patterns=[('full', 0)×5]  glob_layers=2` · 2.802M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False
msg_idslots=True  msg_keepswa=True  glob_layers=2  msg_extrahops=0
msg_updatekv=False  msg_anchors=none`.

800 hunt JSON: `seq_len: 272`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 2`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
**`message_keep_local_swa: true`**, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`hops: 2`, `prize_bits: 26.0`, `calibrated: true`, `dense_steps_used: 3200`.
Hunt exit **0**. 8k JSON same knobs, `steps: 8000`, `k1_mult: 1`, no dense arm,
`calibrated: true`, hunt exit **0**. Params <100M. Dense S0 recalibrated at two
full layers (2.802M). Code default `--message_keep_local_swa` stays **false**.

## Training Outcome

### 800 advertised / dense-matched 3200 (`e25_272_chain_k13_glob2_keepswa`, exit 0)

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **94.8%** | 3200 | **23.53** | **0.905** |
| e18 | **26.1%** | 3200 | **0.01** | **~0** |
| **e21 @800** | **27.6%** | 800 | **0.00** | **~0** |
| **e21 final** | **38.3%** | 3200 | **3.72** | **0.143** |
| e18_local | **27.6%** | 3200 | **0.02** | **~0** |

Dense crossed 75% at 2550 (77.2%), best **98.6% @3150**, final **94.8% /
23.53 bits** @3200. JSON `calibrated: true`. Hunt exit 0. **S0 PASS.**
Recalibrated at `global_layers=2`.

E18 is **~0** in this JSON: chance every eval (best 27.6%; CE glued to ln(4)).
Do **not** pass S1 via 0.75×0. Score S1 vs **0.75× dense 17.65 bits**. (Prior
272 without keep_local_swa had live E18 in the 800 JSON; this-JSON E18 miss
is LR-horizon / bf16 variance, not a keep_local_swa effect — the flag is
E21-only.)

E21 sat at chance through the advertised 800 floor (27.6% / CE 1.386).
Dense-matched budget continued to 3200. Climb started ~2400 (30.6%), best
**39.3% @2900**, final **38.3% / 3.72 bits** @3200 (flow 0.143, CE 1.188).
S1 vs 0.75× dense **17.65 bits**: **FAIL** (3.72). Climbing, short of S1 →
8k extra-step.

### 8k extra-step (`e25_272_chain_k13_glob2_keepswa_s8k`, exit 0)

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| e18 | **99.3%** | 7800 | **25.46** | **0.979** |
| **e21** | **99.2%** | 4650 | **25.55** | **0.983** |
| e18_local | **22.5%** | 8000 | **0.00** | **0** |

Dense skipped (`--no-dense_first`; prior S0 **94.8% / 23.53 bits**). E18 is
**live, not ~0** in this JSON: chance through ~5400, first ≥40% @5450, first
≥75% @7450, early-stop **99.3% / 25.46 bits** @7800. Do not relabel E18 as
E21. S1 vs **0.75× this-JSON live E18 19.10 bits**.

E21 sat at chance through 800 (27.0% / CE 1.386), then climbed: first ≥40%
@2200, first ≥75% **77.2% @2700**, 88.5% @3000, 98.0% @3500, early-stop
**99.2% / 25.55 bits** @4650 (flow 0.983). S1 vs 0.75× live E18 **19.10
bits**: **PASS** (25.55). Content vs 0.75× 800-JSON dense **17.65 bits**:
**PASS** (25.55). Exclusive identity matches live E18 (25.46) and exceeds
the 800 dense control (23.53). The 3200 3.72-bit climb **replicated and
finished**. Do not relabel the live E18 arm as E21.

**8k ran.** Do **not** 16k (S1 already PASS).

## Gates vs this rung

| gate | 800 / 3200 | 8k |
|---|---|---|
| **S0** | **PASS.** Dense 94.8% / 23.53 bits @3200 ≥ 75%. Recalibrated at `global_layers=2`. | dense skipped; prior S0 stands. |
| **S1 vs 0.75× live E18** | E18 ~0 — do **not** pass via 0.75×0. | **PASS.** 25.55 ≥ 19.10. E18 is live (25.46), not ~0. |
| **content vs 0.75× dense** | **FAIL.** 3.72 < 17.65 (climbing). | **PASS.** 25.55 ≥ 17.65. |
| **S2** plots | **PASS** (learning-curves / heatmap / recovered-bits / flow / bytes-per-token). | **PASS** (same). |
| **K1** | not triggered. | not triggered. |
| **K2** | **PASS.** `e18_local` 27.6% / 0.02 bits. keep_local_swa did not leak a full raw prefix. | **PASS.** `e18_local` 22.5% / 0 bits. |
| **K3** | not floor at 3200 (flow 0.143). | not triggered (flow 0.983). |

## Interpretation

**Packed 272 hops FAIL without keep_local_swa was SWA sever starving local hop
assembly, not exclusive global capacity.** Same DNA + identity compressor +
two global Blocks that scored E21 **0.02 bits @8k** with QUERY treated as a
SWA document start now early-stop at **99.2% / 25.55 bits @4650** when local
SWA still sees across QUERY (`msg_keepswa=True`). Exclusive slots stay on the
global read (`msg_rawkv=False`, `msg_override=real`). Window 16 still cannot
span gap 163–187, so this is local hop glue near QUERY, not a full raw-prefix
bypass. `e18_local` stays at chance (K2) — keep_local_swa did not leak a
solvable local-only channel.

SELECT 1024 keep_local_swa remains 0.01 bits. Unsevering SWA is hops-specific
here (chain uses local structure). Do not reopen SELECT knobs. Do not treat
this as reopening the severed-SWA hops length map: without the flag, E21
hops wall at glob=2 still stands at **(264 S1 PASS, 272 FAIL]**. With the
flag, 272 PASSES.

## Decision

Keep the spec in `ahead/`. **8k ran.** Do **not** 16k (S1 PASS). Do not hops
268 without the flag. Do not hops extra-steps / glob=3 / window 32. Do not
Glyph. Do not unfreeze `u`/`delta`. Do not restore full raw prefix KV (that is
E18). Do not reopen SELECT **(692, 696]**. Do not INDEX extra-steps. Do not
MATCH remainder/pooling A/Bs. Code default `--message_keep_local_swa` stays
**false**. Code default `--global_layers` stays 1. Next ONE (do not run):
parent may later map hops length **with SWA kept** (not 268 without the flag).
If that map is not taken, exclusive hops wall without the flag stands at 272.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
