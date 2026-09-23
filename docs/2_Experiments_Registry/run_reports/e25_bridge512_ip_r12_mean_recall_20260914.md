# E25 bridge 512 recall_single r=12 in-place frozen mean — E21 vs E18 vs dense (rung 5t)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_r12_mean_recall` (800) · `e25_512_ip_r12_mean_recall_s8k` (8000)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_r12_mean_recall/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_r12_mean_recall_probe.log` (8k; also `_800.log`)
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `28a1ab7` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** r=8 MATCH PASS [`e25_bridge512_ip_r8_mean_recall_20260914.md`](e25_bridge512_ip_r8_mean_recall_20260914.md) · r=16 MATCH wall [`e25_bridge512_ip_r16_mean_recall_20260914.md`](e25_bridge512_ip_r16_mean_recall_20260914.md)

---

## Goal

Pooling threshold on packed `recall_single` at seq=512. Locked: r=8 frozen mean
**47.16 bits @7100** (S1 PASS); r=16 frozen mean **0 bits**. One change: keep
inplace + `--message_identity_slots` (no learned `u`/`delta`, no raw_kv) and set
`--message_ratio 12`. Remainder **off** (512/12 is not integer: 42 complete
blocks + 8 leftover). Do 12-token frozen means still bind a key at the S1 bar?

H=128 (same 512 MATCH width). Score vs 0.75× E18; also report vs 0.75× dense.
If E18 were ~0, would not call S1 from 0.75×0. Climbing at 800 → extra-step 8k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · 0.595M (e21 0.603M) · kv=1 · **`logit_scale=none`** |
| E21 | query boundary id 10, **r=12**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed answer 24 / **48-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_r12_mean_recall 0 \
  --scale bridge --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 12 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# climbing: same flags --steps 8000 --k1_mult 1
```

Byobu `E25_512_r12_recall` / `E25_512_r12_recall_s8k`. Log:
`msg_r=12  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `message_ratio: 12`,
`message_pool_remainder: false`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (8k JSON) | **100.0%** | 550 | 47.91 | 0.998 |
| e18 (8k JSON) | **100.0%** | 350 | **47.96** | 0.999 |
| **e21 @800** | 39.8% | 800 | 5.79 | 0.121 |
| **e21 @8000** | **81.7%** | **8000** | **34.33** | **0.715** |
| e18_local (8k JSON) | 24.9% | 8000 | 0 | 0 |

800 JSON recalibrates S0: dense **47.92**, E18 **47.96**. 8k JSON: dense **47.91**,
E18 **47.96**. No early-stop on E21 (8k budget exhausted).

E21 at 800 was above chance and still climbing (chance through ~450, then 29.4%
@500 → 39.8% @800). CE falling 1.386 → 1.219. r=16 frozen mean on the same task
was **0 bits / floor at 800**. r=8 was **15.90 bits @800** then S1 at 7100.

S1 vs 0.75× E18 in the 8k JSON: need **35.97 bits**. E21 has **34.33**.
Content vs 0.75× dense: need **35.93 bits**. E21 has **34.33**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 47.91 bits (both JSONs). |
| **S1 vs 0.75× E18** | **FAIL** at 8000. 34.33 < 35.97 bits. Also FAIL at 800 (5.79). |
| **content vs 0.75× dense** | **FAIL.** 34.33 < 35.93 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.9% / 0 bits. |
| **K3** | not triggered (flow 0.12 at 800, 0.72 at 8k; climbing, not a floor). |

Do **not** extra-step to 16k (8k budget, same rule as r=8). Do not 4k. Do not
Glyph. Do not remainder. Do not unfreeze `u`/`delta`. Do not hops.

## Interpretation

**S1-passing MATCH dies between r=8 and r=12** at the 8k budget. r=12 is **not
chance**: 81.7% / 34.33 bits, still climbing after a long ~50% plateau then a
jump near 6k. It misses 0.75× E18 by **1.64 bits**. r=16 remains 0 bits. So:

- r=8: S1 **PASS** (47.16)
- r=12: live, S1 **FAIL** at 8k (34.33)
- r=16: chance **0 bits**

Do **not** relabel E18's 47.96 bits as E21. E21 recovered **34.33 bits**.

## Decision

Keep the spec in `ahead/`. Next ONE experiment: **inplace r=10 frozen mean**
(`--message_ratio 10 --message_slots_inplace --message_identity_slots`, remainder
off, no raw_kv) on this recipe. One change: pooling width. If that PASSES, the
S1 wall is 10–12. If it looks like r=12 (sub-S1 climb), the S1 wall is 8–10.
Do not 16k r=12.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
