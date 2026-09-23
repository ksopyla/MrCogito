# E25 bridge 512 recall_single r=16 remainder-on frozen mean — E21 vs E18 vs dense (rung 5x)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_r16_rem_recall` (800) · `e25_512_ip_r16_rem_recall_s8k` (8000)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_r16_rem_recall/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_r16_rem_recall_probe.log` (8k; also `_800.log`)
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `480f4fc` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** r=16 remainder-off MATCH wall [`e25_bridge512_ip_r16_mean_recall_20260914.md`](e25_bridge512_ip_r16_mean_recall_20260914.md) · r=12 remainder-on PASS [`e25_bridge512_ip_r12_rem_recall_20260914.md`](e25_bridge512_ip_r12_rem_recall_20260914.md)

---

## Goal

Remainder-on rescued r=10 and r=12 MATCH. Remainder-off r=16 MATCH was **0 bits
chance**. 512/16 divides the row, but the sender prefix before QUERY may still
leave an incomplete last block. One change: keep inplace frozen-mean identity
**r=16** and set `--message_pool_remainder`. Default stays off elsewhere. Was
leftover/alignment the MATCH wall through r=16, or do 16-token means fail to
bind a key even with complete coverage?

H=128. Score vs 0.75× E18; also report vs 0.75× dense. If E18 were ~0, would
not call S1 from 0.75×0. Climbing at 800 → extra-step 8k. Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · 0.595M (e21 0.603M) · kv=1 · **`logit_scale=none`** |
| E21 | query boundary id 10, **r=16**, remainder **on**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed answer 24 / **48-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_r16_rem_recall 0 \
  --scale bridge --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 16 --message_slots_inplace --message_identity_slots \
  --message_pool_remainder \
  --steps 800 --k1_mult 4
# climbing: same flags --steps 8000 --k1_mult 1
```

Byobu `E25_512_r16_rem_recall` / `E25_512_r16_rem_s8k`. Log:
`msg_r=16  msg_remainder=True  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `message_ratio: 16`,
`message_pool_remainder: true`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`. Recalibrated dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (8k JSON) | **100%** | 350 | **47.95** | 0.999 |
| e18 (8k JSON) | **100%** | 350 | **47.86** | 0.997 |
| **e21 @800** | 36.4% | 800 | 4.72 | 0.098 |
| **e21 @8000** | **92.6%** (best **96.4%** @7750) | **8000** | **40.92** | **0.852** |
| e18_local (8k JSON) | 24.9% | 8000 | 0 | 0 |

800 JSON recalibrates S0: dense **47.77**, E18 **47.73**. 8k JSON: dense
**47.95**, E18 **47.86**. No early-stop on E21 (8k budget exhausted).

E21 at 800 was above chance and climbing (chance through ~450, then 30.3% @500
→ 36.4% @800, CE 1.387 → 1.250). Remainder-off r=16 on the same task was
**0 bits / floor at 800**. Extra-step to 8k. Final acc dipped from 96.4% @7750
to 92.6% @8000; recovered bits still 40.92.

S1 vs 0.75× E18 in the 8k JSON: need **35.89 bits**. E21 has **40.92**.
Content vs 0.75× dense: need **35.96 bits**. E21 has **40.92**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 47.95 bits (8k JSON). |
| **S1 vs 0.75× E18** | **PASS** at 8000. 40.92 ≥ 35.89 bits. FAIL at 800 (4.72). |
| **content vs 0.75× dense** | **PASS.** 40.92 ≥ 35.96 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.9% / 0 bits. |
| **K3** | not triggered (flow 0.10 at 800, 0.85 at 8k; climbing, not a floor). |

Do **not** extra-step to 16k (S1 already PASS). Do not r=9. Do not r=11. Do not
hops. Do not 4k. Do not Glyph. Do not unfreeze `u`/`delta`. Do not 1024.

## Interpretation

**Leftover/alignment was the MATCH wall through r=16.** Remainder-off r=16 was
chance 0 bits. Remainder-on copies **40.92 bits** (S1 PASS). The same frozen-mean
INDEX pooler can MATCH at r=16 once the incomplete last sender block is pooled.
The INDEX vs MATCH split at r=16 remainder-off does **not** survive remainder-on.

Do **not** relabel E18's 47.86 bits as E21. E21 recovered **40.92 bits**.

## Decision

Keep the spec in `ahead/`. MATCH recipe at 512 is inplace frozen mean +
remainder-on through r=16. Next ONE experiment: seq=512 packed
**`select_1decoy`**, inplace frozen mean **r=16 remainder-on** (does type-cue
survive the same pooler?). Not r=9/11. Not hops. Not 4k. Not Glyph. Not 1024.
Default remainder stays off elsewhere.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
