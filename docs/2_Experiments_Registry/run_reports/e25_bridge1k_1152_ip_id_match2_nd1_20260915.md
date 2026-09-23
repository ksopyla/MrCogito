# E25 bridge_1k seq=1152 `--recipe recall` MATCH2 n_dist=1 H=128 identity — dense K1 (rung 5bw)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1152_ip_id_match2_nd1` (800 advertised; dense K1 budget 3200; e18 / e21 / `e18_local` skipped; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_1152_ip_id_match2_nd1/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_1152_ip_id_match2_nd1/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `00296e0` (Odra E24 worktree; E22 main checkout untouched; hunt code same as `99fa858` / `40af063`)
**Git tag:** —
**Related:** 1024 MATCH2 n_dist=1 S1 PASS [`e25_bridge1k_ip_id_match2_nd1_20260915.md`](e25_bridge1k_ip_id_match2_nd1_20260915.md) · 1280 MATCH2 n_dist=1 S1 FAIL [`e25_bridge1k_1280_ip_id_match2_nd1_20260915.md`](e25_bridge1k_1280_ip_id_match2_nd1_20260915.md)

---

## Goal

Pin the 2-item MATCH2 length wall between 1024 S1 PASS and 1280 S1 FAIL
at the same H=128 identity compressor. `--scale bridge_1k --seq_len 1152
--recipe recall --n_distractors 1`. Do **not** switch to H=256. Do **not**
run 3-item MATCH2. Do not also run 1100/1200/1408. Do not confuse this with
2-key MATCH at 1280 (`recall_single` r=1 identity H=256 log S1 PASS 63.94).
Do not relabel 1024 MATCH2 61.81 or 1280 MATCH2 0.01 as this score.

`--n_distractors 1` is a valid packed MATCH2 at 1152: `pack_overrides`
keeps `value_len=32` / 64-bit prize; seed-0 row `meta.n_items=2`;
`--evidence_align right` row_gap 65/82/100. Dense S0 first. If dense
misses 75% (K1), skip E18/E21. If E18 ≈ 0, score vs 0.75× dense (do not
pass via 0.75×0). 800-step floor: chance → no 8k; climbing short of S1 →
extra-step 8k only then. Do not 16k. Dense already using `k1_mult=4`
means no 8k on K1.

## Configuration

| Item | Value |
|---|---|
| Family | dense first; e18 / e21 / e18_local skip on K1 |
| Width | **H=128** · **0.595M** · kv=1 · **`logit_scale=none`** · stack=2 · glob=1 |
| E21 (not scored) | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`** |
| Data | `--scale bridge_1k --seq_len 1152 --recipe recall --n_distractors 1`, packed answer 32 / **64-bit** prize, **2 items** |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/82/100 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.6GB, 0.06 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1152_ip_id_match2_nd1 0 \
  --scale bridge_1k --seq_len 1152 --recipe recall --n_distractors 1 \
  --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# named recipe recall + --n_distractors 1 = 2-item MATCH2 (not recall_single)
# no --global_logit_scale (none; same width as 1024/1280 MATCH2 n_dist=1)
# no --global_layers 2
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused)
# dense K1: do not extra-step; do not score E21; do not 16k
```

Byobu `E25_1152_match2_nd1`. Log:
`seq=1152  gap=64  window=16  prize=64.00 bits  answer_len=32
row_gap[min/med/max]=65/82/100`
dense `patterns=[('full', 0)×4]` · 0.595M.

Hunt JSON: `seq_len: 1152`, `min_gap: 64`, `local_window: 16`, `hidden: 128`,
`global_logit_scale: none`, `global_layers: 1`, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`n_distractors: 1`, `prize_bits: 64.0`, `calibrated: false`,
`dense_steps_used: 3200`. Hunt exit **2**. Params <100M. Dense S0
recalibrated at H=128 (0.595M). Code defaults unchanged.

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense @800 | **24.9%** | 800 | **0.00** | — | — |
| dense | **24.1%** | 3200 | **0.00** | **0.000** | **0** |
| dense best acc | **26.3%** | 100 | (chance noise) | — | — |
| e18 | skipped (K1) | — | — | — | — |
| e21 | skipped (K1) | — | — | — | — |
| e18_local | skipped (K1) | — | — | — | — |

Dense stayed at chance every eval through 3200 (acc 23.5–26.3%; CE at
ln(4); **0.00 bits**). **24.9% @800**. Final **24.1% / 0.00 bits** @3200
(flow 0.000). Never left chance, never crossed 75%. JSON `calibrated:
false`. Hunt exit 2. Packed layout holds: 2 items, `value_len=32`, 64-bit
prize, seed-0 `answer_start=1118` / gap 65.

Same H=128 identity recipe as 1024 MATCH2 n_dist=1 (dense **93.5% /
58.81 bits** S0 PASS; E21 **61.81** S1 PASS) and 1280 MATCH2 n_dist=1
(dense **98.9% / 62.94 bits** S0 PASS; E21 **0.01** S1 FAIL). Length is
not monotone for dense S0 at this seed/budget: 1152 is chance, 1024 and
1280 are not.

Do **not** score S1 vs E18 or vs dense. There is no E21 number. Do not
relabel the skipped E18 arm as E21. Do not relabel 1024 MATCH2 61.81
or 1280 MATCH2 0.01 as this score. Do not treat this K1 as an E21 S1
FAIL.

**8k not run** (dense K1; extra-step 8k is for E21 climbing short of S1,
not for dense S0 — dense already used `k1_mult=4`). Do not 16k.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **FAIL / K1.** Dense 24.1% / 0.00 bits @3200 < 75%. Recalibrated at H=128. |
| **S1 vs 0.75× live E18** | **not scored.** |
| **S1 vs 0.75× dense** | **not scored.** |
| **content vs dense** | **not scored.** |
| **S2** plots | **PASS** (dense-only uncalibrated curves + bits/flow/bytes-per-token). |
| **K1** | **triggered.** Stop. Do not score E21. |
| **K2** | **not scored** (`e18_local` skipped). |
| **K3** | **not scored.** |
| **8k** | **not run** (dense already `k1_mult=4`). |

## Interpretation

**Packed 1152 2-item MATCH2 (`recall`, 1 distractor) is uncalibrated at
the identity MATCH2-passing H=128 recipe.** Dense recovered **0 of 64
prize bits** (chance floor, not the 512 3-item MATCH2 18-bit plateau).
Exclusive identity that recovered 61.81 bits at 1024 and failed at
1280 is **not scored** here. Protocol skip: E18/E21 not scored, so this
is **not** an E21 MATCH2 S1 FAIL and **not** an E21 MATCH2 S1 PASS.

2-item S1 length among S0-calibrated lengths stays **(1024 S1 PASS,
1280 FAIL]**. The 1152 pin is a **dense K1 hole** inside that interval,
not an S1 interior. Item-count wall at 512 stays **(2-item S1 PASS,
3-item dense K1]**. Do not treat 2-key MATCH identity at 1280 (H=256
log S1 PASS) as this 2-item MATCH2 result.

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not
1100/1200/1408. Do not H=256. Do not MATCH2 n_dist=2. Do not 3-item at
1024. Do not hops 320. Do not MATCH 1408/2048. Do not SELECT 694. Do not
INDEX extra-steps. Do not Glyph. Do not unfreeze `u`/`delta`. Do not
restore full raw prefix KV (that is E18). Do not stack keepswa / extra
hop / glob=2 on this pin. Code defaults unchanged.
Next ONE (do not run): **STOP MATCH2 2-item length extra-steps**.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
