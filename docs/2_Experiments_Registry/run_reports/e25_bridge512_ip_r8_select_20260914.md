# E25 bridge 512 select_1decoy r=8 remainder-off frozen mean — E21 vs E18 vs dense (rung 5aa)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_r8_select` (800) · `e25_512_ip_r8_select_s8k` (8000)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_r8_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_r8_select_probe.log` (8k; also `_800.log`)
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `61367b0` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** MATCH r=8 rem-off PASS [`e25_bridge512_ip_r8_mean_recall_20260914.md`](e25_bridge512_ip_r8_mean_recall_20260914.md) · SELECT r=12 rem-on PASS [`e25_bridge512_ip_r12_rem_select_20260914.md`](e25_bridge512_ip_r12_rem_select_20260914.md)

---

## Goal

MATCH r=8 rem-off copies **47.16 bits** (S1 PASS; 512/8 divides the row).
SELECT r=12 rem-on copies **43.29 bits** (S1 PASS). SELECT r=16 rem-on was
live S1 FAIL (35.94 vs 35.95). One change: default remainder **off** on the
MATCH-passing **r=8** grid, packed `select_1decoy`. Does SELECT need leftover
pooling, or is the MATCH-passing default grid enough?

H=128. Score vs 0.75× E18; also report vs 0.75× dense. E18 is live (do not pass
S1 via 0.75×0). Climbing at 800 → extra-step 8k. Do not 16k. Do not 16k the
r=16 SELECT miss. Do **not** pass `--message_pool_remainder`.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · 0.595M (e21 0.603M) · kv=1 · **`logit_scale=none`** |
| E21 | query boundary id 10, **r=8**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed answer 24 / **48-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 92/92/92) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_r8_select 0 \
  --scale bridge --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 8 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# climbing: same flags --steps 8000 --k1_mult 1
# no --message_pool_remainder
```

Byobu `E25_512_r8_select` / `E25_512_r8_sel8k`. Log:
`msg_r=8  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `message_ratio: 8`,
`message_pool_remainder: false`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`. Recalibrated dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (8k JSON) | **100%** | 400 | **47.91** | 0.998 |
| e18 (8k JSON) | **100%** | 350 | **47.90** | 0.998 |
| **e21 @800** | 43.3% | 800 | 11.73 | 0.244 |
| **e21 @8000** | **96.4%** (best **97.9%** @7450) | **8000** | **45.53** | **0.948** |
| e18_local (8k JSON) | 24.6% | 8000 | 0 | 0 |

800 JSON recalibrates S0: dense **47.52**, E18 **47.94**. 8k JSON: dense
**47.91**, E18 **47.90**. No early-stop on E21 (8k budget exhausted).

E21 at 800 was above chance and climbing (chance through ~350, then 36.2% @400
→ 45.7% @650, 43.3% / 11.73 bits @800). Extra-step to 8k. 82.9% @3000, best
97.9% @7450, final 96.4% @8000. Recovered bits **45.53**.

S1 vs 0.75× E18 in the 8k JSON: need **35.93 bits**. E21 has **45.53**.
Content vs 0.75× dense: need **35.94 bits**. E21 has **45.53**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 47.91 bits (8k JSON). |
| **S1 vs 0.75× E18** | **PASS** at 8000. 45.53 ≥ 35.93 bits. FAIL at 800 (11.73). |
| **content vs 0.75× dense** | **PASS.** 45.53 ≥ 35.94 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.6% / 0 bits. |
| **K3** | not triggered (flow 0.24 at 800, 0.95 at 8k; climbing, not a floor). |

Do **not** extra-step to 16k (S1 already PASS). Do not 16k the r=16 SELECT miss.
Do not r=9. Do not r=11. Do not hops. Do not 4k. Do not Glyph. Do not unfreeze
`u`/`delta`. Do not 1024.

## Interpretation

**SELECT works on the default r=8 recipe (no remainder flag).** The
MATCH-passing default grid is enough for type-cue. Remainder-on is not
required for SELECT at r=8. Remainder-on remains the leftover/alignment
probe at r that leave an incomplete last sender block (MATCH r=10/12/16).

Do **not** relabel E18's 47.90 bits as E21. E21 recovered **45.53 bits**.

## Decision

Keep the spec in `ahead/`. SELECT remainder sweep can stop: default **r=8
rem-off** covers MATCH + SELECT at 512. Next ONE: **do not another SELECT r**.
Remaining measured walls are 512 chain **K1** and 256 hops **FAIL**. Do not
hops. Do not 4k. Do not Glyph. Do not 1024. Not r=9/11. Not 16k. Default
remainder stays off elsewhere.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
