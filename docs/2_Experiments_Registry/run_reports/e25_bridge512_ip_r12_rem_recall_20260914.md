# E25 bridge 512 recall_single r=12 remainder-on frozen mean — E21 vs E18 vs dense (rung 5w)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_r12_rem_recall` (800) · `e25_512_ip_r12_rem_recall_s8k` (8000)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_r12_rem_recall/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_r12_rem_recall_probe.log` (8k; also `_800.log`)
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `054cc99` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** r=12 remainder-off S1 fail [`e25_bridge512_ip_r12_mean_recall_20260914.md`](e25_bridge512_ip_r12_mean_recall_20260914.md) · r=10 remainder-on PASS [`e25_bridge512_ip_r10_rem_recall_20260914.md`](e25_bridge512_ip_r10_rem_recall_20260914.md)

---

## Goal

Same remainder flag that rescued r=10 (leftover 2) on the live S1-fail width.
Locked: r=12 remainder-off **34.33 bits @8000** (S1 FAIL vs 35.97, leftover 8);
r=10 remainder-on **44.21 bits @8000** (S1 PASS). One change: keep inplace
frozen-mean identity **r=12** and set `--message_pool_remainder`. Default stays
off elsewhere. Does leftover also explain the 1.64-bit miss?

H=128. Score vs 0.75× E18; also report vs 0.75× dense. If E18 were ~0, would
not call S1 from 0.75×0. Climbing at 800 → extra-step 8k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · 0.595M (e21 0.603M) · kv=1 · **`logit_scale=none`** |
| E21 | query boundary id 10, **r=12**, remainder **on**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed answer 24 / **48-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_r12_rem_recall 0 \
  --scale bridge --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 12 --message_slots_inplace --message_identity_slots \
  --message_pool_remainder \
  --steps 800 --k1_mult 4
# climbing: same flags --steps 8000 --k1_mult 1
```

Byobu `E25_512_r12_rem_recall` / `E25_512_r12_rem_s8k`. Log:
`msg_r=12  msg_remainder=True  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `message_ratio: 12`,
`message_pool_remainder: true`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`. Recalibrated dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (8k JSON) | **100%** | 350 | **47.87** | 0.997 |
| e18 (8k JSON) | **100%** | 400 | **47.96** | 0.999 |
| **e21 @800** | 41.0% | 800 | 6.14 | 0.128 |
| **e21 @8000** | **97.9%** (best **98.2%** @7850) | **8000** | **46.06** | **0.960** |
| e18_local (8k JSON) | 24.9% | 8000 | 0 | 0 |

800 JSON recalibrates S0: dense **47.72**, E18 **47.96**. 8k JSON: dense
**47.87**, E18 **47.96**. No early-stop on E21 (8k budget exhausted).

E21 at 800 was above chance and climbing (41.0% / 6.14 bits, CE 1.209).
Remainder-off r=12 at 800 was 39.8% / 5.79 bits. Extra-step to 8k.

S1 vs 0.75× E18 in the 8k JSON: need **35.97 bits**. E21 has **46.06**.
Content vs 0.75× dense: need **35.90 bits**. E21 has **46.06**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 47.87 bits (8k JSON). |
| **S1 vs 0.75× E18** | **PASS** at 8000. 46.06 ≥ 35.97 bits. FAIL at 800 (6.14). |
| **content vs 0.75× dense** | **PASS.** 46.06 ≥ 35.90 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.9% / 0 bits. |
| **K3** | not triggered (flow 0.13 at 800, 0.96 at 8k; climbing, not a floor). |

Do **not** extra-step to 16k (S1 already PASS). Do not r=9. Do not r=11. Do not
16k remainder-off. Do not hops. Do not 4k. Do not Glyph. Do not unfreeze
`u`/`delta`. Do not 1024.

## Interpretation

**Leftover also explains the r=12 1.64-bit miss.** Remainder-off r=12 was live
but S1 FAIL (34.33). Remainder-on copies **46.06 bits** (97.9%). The r=12
pooling width is not the S1 wall once the incomplete last sender block is
pooled. Same leftover story as r=10 (0 bits off → 44.21 on).

Do **not** relabel E18's 47.96 bits as E21. E21 recovered **46.06 bits**.

## Decision

Keep the spec in `ahead/`. Next ONE experiment: seq=512 packed `recall_single`
inplace frozen mean **r=16 remainder-on** (remainder-off r=16 MATCH was chance
0 bits; sender prefix may still drop an incomplete last block even though
512/16 divides the row). If it PASSES, leftover/alignment was the MATCH wall
through r=16. If still chance, 16-token means do not bind a key even with
complete coverage. Not r=9/11. Not 16k. Default remainder stays off elsewhere.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
