# E25 bridge seq=512 `--recipe recall` MATCH2 n_dist=2 H=128 identity — dense K1 (rung 5bs)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_id_match2` (800 advertised; dense K1 budget 3200; e18 / e21 / `e18_local` skipped; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_id_match2/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_id_match2/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `3759020` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 512 `recall_single` identity S1 PASS [`e25_bridge512_ip_id_recall_20260914.md`](e25_bridge512_ip_id_recall_20260914.md) · tiny MATCH2 uncalibrated at H=128 [`e24_tiny_bapo_ladder_20260913.md`](e24_tiny_bapo_ladder_20260913.md) · 288 hops extra hop S1 FAIL [`e25_bridge288_chain_k13_glob2_extrahop_20260915.md`](e25_bridge288_chain_k13_glob2_extrahop_20260915.md)

---

## Goal

Unmeasured original DNA recipe after INDEX / `recall_single` / `select_1decoy` /
hops walls were mapped: packed **MATCH2** (`--recipe recall`, generator `n_distractors=2`,
3 items) at the seq=512 identity MATCH-passing geometry. Tiny packed `far_copy` and
512/1024 packed `far_copy` / `recall_single` are already scored — do not relabel those
as this rung. Sibling `perceiver_concept` Arm-A `far_copy` @128 is a different workstream.

One change vs 512 `recall_single` r=1 identity H=128 **S1 PASS 43.08 bits**: keep
inplace identity, remainder off, glob=1, extra hops 0, keep_local_swa off, and
score default MATCH2 (`n_distractors=2`, packed `value_len=24`, 48-bit prize)
instead of `n_distractors=0`. Dense S0 first. If dense misses 75% (K1), skip
E18/E21. Do not 8k dense (already `k1_mult=4`). Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | dense first; e18 / e21 / e18_local skip on K1 |
| Width | **H=128** · **0.595M** · kv=1 · **`logit_scale=none`** · stack=2 · glob=1 |
| E21 (not scored) | query boundary id 10, **r=1**, remainder **off**, **`msg_inplace=True`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`** |
| Data | `--scale bridge --recipe recall`, n_distractors **2**, packed answer 24 / **48-bit** prize |
| Geometry | min_gap **64** > local_window **16**. `--evidence_align right`; row_gap 65/92/119 (queried MATCH2 fact not a fixed right edge) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~0.9GB, 0.04 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_id_match2 0 \
  --scale bridge --recipe recall --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# named recipe recall = MATCH2 n_distractors=2 (not recall_single)
# no --n_distractors override
# no --global_logit_scale (none; same width as 512 recall_single identity)
# no --global_layers 2
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# dense first (underscore --no-dense_first unused)
# dense K1: do not extra-step; do not score E21; do not 16k
```

Byobu `E25_512_match2`. Log:
`seq=512  gap=64  window=16  prize=48.00 bits  answer_len=24
row_gap[min/med/max]=65/92/119`
dense `patterns=[('full', 0)×4]` · 0.595M.

Hunt JSON: `seq_len: 512`, `min_gap: 64`, `local_window: 16`, `hidden: 128`,
`global_logit_scale: none`, `global_layers: 1`, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`n_distractors: 2`, `prize_bits: 48.0`, `calibrated: false`,
`dense_steps_used: 3200`. Hunt exit **2**. Params <100M. Dense S0
recalibrated at H=128 (0.595M). Code defaults unchanged.

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense @800 | **51.8%** | 800 | (climbing) | — | — |
| dense | **55.8%** | 3200 | **18.44** | **0.384** | **0.004501** |
| e18 | skipped (K1) | — | — | — | — |
| e21 | skipped (K1) | — | — | — | — |
| e18_local | skipped (K1) | — | — | — | — |

Dense left chance by ~400 (39.7%), **51.8%** @800 (CE 0.998 vs floor 1.386),
then plateaued ~52–56%. Best **55.9% @2450**. Final **55.8% / 18.44 bits**
@3200 (CE 0.854, flow 0.384, 18.44 of 48 prize bits). Never crossed 75%.
JSON `calibrated: false`. Hunt exit 2.

Same geometry + H=128 identity as 512 `recall_single` (dense **99.9% /
47.69 bits** @350; E21 **43.08 bits**). Three-item MATCH2 is live for dense
(18 bits, not the chance floor) and still **not** S0. Tiny MATCH2 at H=128
was ~30% @3200; 512 packed MATCH2 is higher but still K1.

Do **not** score S1 vs E18 or vs dense. There is no E21 number. Do not
relabel the skipped E18 arm as E21. Do not relabel 512 `recall_single`
43.08 bits as this MATCH2 score.

**8k not run** (dense K1; extra-step 8k is for E21 climbing short of S1, not
for dense S0 — dense already used `k1_mult=4`). Do not 16k.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **FAIL / K1.** Dense 55.8% / 18.44 bits @3200 < 75%. Recalibrated at H=128. |
| **S1 vs 0.75× live E18** | **not scored.** |
| **content vs 0.75× dense** | **not scored.** |
| **S2** plots | **PASS** (dense-only uncalibrated curves + bits/flow/bytes-per-token). |
| **K1** | **triggered.** Stop. Do not score E21. |
| **K2** | **not scored** (`e18_local` skipped). |
| **K3** | **not scored.** |

## Interpretation

**Packed 512 MATCH2 (`recall`, 2 distractors) is uncalibrated at the
identity MATCH-passing H=128 recipe.** Dense recovered **18.44 of 48 prize
bits** (not chance), then plateaued ~56% through 3200. Exclusive identity
that binds a unique planted key at this length does not make 3-item
MATCH2 dense-solvable. Protocol skip: E18/E21 not scored, so this is
**not** an E21 MATCH2 S1 FAIL and **not** an E21 MATCH2 S1 PASS.

Tiny MATCH2 H=128 stays ~30% K1. 512 `recall_single` identity stays S1
PASS 43.08. Default named `recall` remains a hunt until a dense control
hits 75% at the same hidden size.

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not hops
length extra-steps. Do not MATCH 1408/2048. Do not SELECT 694. Do not
INDEX extra-steps. Do not Glyph. Do not unfreeze `u`/`delta`. Do not
restore full raw prefix KV (that is E18). Code defaults unchanged.
Next ONE (do not run): **512 MATCH2 `--n_distractors 1` H=128 identity**
(smallest 2-item MATCH2 at the `recall_single`-passing recipe). Do not
H=256 on 3-item MATCH2 unless 1-distractor also K1. `--hops 1` is
illegal.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
