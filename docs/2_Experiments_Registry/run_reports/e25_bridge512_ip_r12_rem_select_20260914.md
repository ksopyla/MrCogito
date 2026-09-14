# E25 bridge 512 select_1decoy r=12 remainder-on frozen mean — E21 vs E18 vs dense (rung 5z)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_r12_rem_select` (800) · `e25_512_ip_r12_rem_select_s8k` (8000)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_r12_rem_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_r12_rem_select_probe.log` (8k; also `_800.log`)
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `0840a21` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** r=12 remainder-on MATCH PASS [`e25_bridge512_ip_r12_rem_recall_20260914.md`](e25_bridge512_ip_r12_rem_recall_20260914.md) · r=16 remainder-on SELECT FAIL [`e25_bridge512_ip_r16_rem_select_20260914.md`](e25_bridge512_ip_r16_rem_select_20260914.md)

---

## Goal

MATCH rem-on r=12 copies **46.06 bits** (S1 PASS). SELECT rem-on r=16 was live
S1 FAIL (**35.94 vs 35.95**, miss 0.016). One change: same inplace frozen-mean
identity remainder-on recipe at **r=12** on packed `select_1decoy`. Does SELECT
clear 0.75× E18 at the next-finer rem-on r that already S1-passes MATCH?

H=128. Score vs 0.75× E18; also report vs 0.75× dense. E18 is live (do not pass
S1 via 0.75×0). Climbing at 800 → extra-step 8k. Do not 16k. Do not 16k the
r=16 SELECT miss.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · 0.595M (e21 0.603M) · kv=1 · **`logit_scale=none`** |
| E21 | query boundary id 10, **r=12**, remainder **on**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed answer 24 / **48-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 92/92/92) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_r12_rem_select 0 \
  --scale bridge --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 12 --message_slots_inplace --message_identity_slots \
  --message_pool_remainder \
  --steps 800 --k1_mult 4
# climbing: same flags --steps 8000 --k1_mult 1
```

Byobu `E25_512_r12_rem_select` / `E25_512_r12_rem_sel8k`. Log:
`msg_r=12  msg_remainder=True  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `message_ratio: 12`,
`message_pool_remainder: true`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`. Recalibrated dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (8k JSON) | **100%** | 350 | **47.93** | 0.999 |
| e18 (8k JSON) | **100%** | 350 | **47.90** | 0.998 |
| **e21 @800** | 45.1% | 800 | 8.92 | 0.186 |
| **e21 @8000** | **93.8%** (best **98.0%** @7900) | **8000** | **43.29** | **0.902** |
| e18_local (8k JSON) | 24.6% | 8000 | 0 | 0 |

800 JSON recalibrates S0: dense **47.92**, E18 **47.95**. 8k JSON: dense
**47.93**, E18 **47.90**. No early-stop on E21 (8k budget exhausted).

E21 at 800 was above chance and climbing (chance through ~350, then 38.1% @400
→ 45.1% / 8.92 bits @800). Extra-step to 8k. ~50% through 4k, then 86.7% @5000,
best 98.0% @7900, final 93.8% @8000. Recovered bits **43.29**.

S1 vs 0.75× E18 in the 8k JSON: need **35.92 bits**. E21 has **43.29**.
Content vs 0.75× dense: need **35.95 bits**. E21 has **43.29**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 47.93 bits (8k JSON). |
| **S1 vs 0.75× E18** | **PASS** at 8000. 43.29 ≥ 35.92 bits. FAIL at 800 (8.92). |
| **content vs 0.75× dense** | **PASS.** 43.29 ≥ 35.95 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.6% / 0 bits. |
| **K3** | not triggered (flow 0.19 at 800, 0.90 at 8k; climbing, not a floor). |

Do **not** extra-step to 16k (S1 already PASS). Do not 16k the r=16 SELECT miss.
Do not r=9. Do not r=11. Do not hops. Do not 4k. Do not Glyph. Do not unfreeze
`u`/`delta`. Do not 1024.

## Interpretation

**SELECT clears at r=12 rem-on; r=16 is the SELECT S1 edge.** Same frozen-mean
remainder-on pooler: MATCH r=12 **46.06 PASS**, MATCH r=16 **40.92 PASS**,
SELECT r=12 **43.29 PASS**, SELECT r=16 **35.94 live FAIL**. One recipe covers
INDEX + MATCH + SELECT at 512 for r=12 rem-on. Type-cue needs a slightly
finer grid than key-bind at r=16, not a different pooler family.

Do **not** relabel E18's 47.90 bits as E21. E21 recovered **43.29 bits**.

## Decision

Keep the spec in `ahead/`. Next ONE experiment: seq=512 packed
**`select_1decoy`**, inplace frozen mean **r=8 remainder-off** (MATCH r=8
rem-off already S1 PASS; default remainder off — does SELECT need leftover
pooling, or is the MATCH-passing default grid enough?). Not r=9/11. Not hops.
Not 4k. Not Glyph. Not 1024. Not 16k. Default remainder stays off elsewhere.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
