# E25 bridge 512 recall_single r=10 remainder-on frozen mean — E21 vs E18 vs dense (rung 5v)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_r10_rem_recall` (800) · `e25_512_ip_r10_rem_recall_s8k` (8000)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_r10_rem_recall/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_r10_rem_recall_probe.log` (8k; also `_800.log`)
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `4321d66` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** r=10 remainder-off chance [`e25_bridge512_ip_r10_mean_recall_20260914.md`](e25_bridge512_ip_r10_mean_recall_20260914.md) · r=12 MATCH S1 fail [`e25_bridge512_ip_r12_mean_recall_20260914.md`](e25_bridge512_ip_r12_mean_recall_20260914.md) · r=8 MATCH PASS [`e25_bridge512_ip_r8_mean_recall_20260914.md`](e25_bridge512_ip_r8_mean_recall_20260914.md)

---

## Goal

Non-monotone MATCH pooling at seq=512: r=10 remainder-off was **0 bits chance
@800** while r=12 remainder-off was already climbing at 800. One architecture
flag on the same r=10 inplace frozen-mean identity recipe:
`--message_pool_remainder` (default stays off elsewhere). Remainder-off left 2
tokens unpooled (51×10 + 2). Was the r=10 floor a dropped incomplete block
(key in leftover 2), or the 10-token grid itself?

H=128. Score vs 0.75× E18; also report vs 0.75× dense. If E18 were ~0, would
not call S1 from 0.75×0. Climbing at 800 → extra-step 8k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · 0.595M (e21 0.603M) · kv=1 · **`logit_scale=none`** |
| E21 | query boundary id 10, **r=10**, remainder **on**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed answer 24 / **48-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_r10_rem_recall 0 \
  --scale bridge --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 10 --message_slots_inplace --message_identity_slots \
  --message_pool_remainder \
  --steps 800 --k1_mult 4
# climbing: same flags --steps 8000 --k1_mult 1
```

Byobu `E25_512_r10_rem_recall` / `E25_512_r10_rem_s8k`. Log:
`msg_r=10  msg_remainder=True  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `message_ratio: 10`,
`message_pool_remainder: true`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`. Recalibrated dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (8k JSON) | **99.9%** | 400 | **47.88** | 0.998 |
| e18 (8k JSON) | **100%** | 350 | **47.79** | 0.996 |
| **e21 @800** | 44.2% | 800 | 8.65 | 0.180 |
| **e21 @8000** | **95.1%** (best **96.4%** @7800) | **8000** | **44.21** | **0.921** |
| e18_local (8k JSON) | 24.9% | 8000 | 0 | 0 |

800 JSON recalibrates S0: dense **47.96**, E18 **46.52** (live). 8k JSON: dense
**47.88**, E18 **47.79**. No early-stop on E21 (8k budget exhausted).

E21 at 800 was above chance and climbing (chance through ~400, then 30.8% @450
→ 44.2% @800, CE 1.387 → 1.137). Remainder-off r=10 on the same task was
**0 bits / floor at 800**. Extra-step to 8k.

S1 vs 0.75× E18 in the 8k JSON: need **35.85 bits**. E21 has **44.21**.
Content vs 0.75× dense: need **35.91 bits**. E21 has **44.21**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.9% / 47.88 bits (8k JSON). |
| **S1 vs 0.75× E18** | **PASS** at 8000. 44.21 ≥ 35.85 bits. FAIL at 800 (8.65). |
| **content vs 0.75× dense** | **PASS.** 44.21 ≥ 35.91 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.9% / 0 bits. |
| **K3** | not triggered (flow 0.18 at 800, 0.92 at 8k; climbing, not a floor). |

Do **not** extra-step to 16k (S1 already PASS). Do not r=9. Do not r=11. Do not
hops. Do not 4k. Do not Glyph. Do not unfreeze `u`/`delta`. Do not 1024.

## Interpretation

**Leftover pooling rescued r=10.** Same inplace frozen-mean identity recipe that
was **chance 0 bits** with remainder off recovers **44.21 bits** with remainder
on. The r=10 floor was a dropped incomplete block (key in leftover 2 tokens),
not the 10-token grid/straddle itself.

Do **not** relabel E18's 47.79 bits as E21. E21 recovered **44.21 bits**.

## Decision

Keep the spec in `ahead/`. Next ONE experiment: seq=512 packed `recall_single`
inplace frozen mean **r=12 remainder-on** (same architecture flag on the live
S1-fail width; leftover 8). Not r=9/11. Not 16k. Not hops. Default remainder
stays off elsewhere.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
