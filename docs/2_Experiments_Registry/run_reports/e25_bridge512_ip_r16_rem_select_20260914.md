# E25 bridge 512 select_1decoy r=16 remainder-on frozen mean — E21 vs E18 vs dense (rung 5y)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_r16_rem_select` (800) · `e25_512_ip_r16_rem_select_s8k` (8000)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_r16_rem_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_r16_rem_select_probe.log` (8k; also `_800.log`)
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `63fa8cb` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** r=16 remainder-on MATCH PASS [`e25_bridge512_ip_r16_rem_recall_20260914.md`](e25_bridge512_ip_r16_rem_recall_20260914.md) · identity SELECT PASS [`e25_bridge512_ip_id_select_20260914.md`](e25_bridge512_ip_id_select_20260914.md)

---

## Goal

Remainder-on r=16 MATCH copies **40.92 bits / 92.6% @8000** (S1 PASS). Identity
r=1 SELECT already **47.98 bits / 100% @700**. Tiny select was a type-cue wall
(1.47 bits). One change: same inplace frozen-mean identity **r=16 remainder-on**
recipe on packed `select_1decoy`. Does the INDEX/MATCH pooler also do type-cue
SELECT, or only MATCH/INDEX?

H=128. Score vs 0.75× E18; also report vs 0.75× dense. E18 is live (do not pass
S1 via 0.75×0). Climbing at 800 → extra-step 8k. Do not 16k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · 0.595M (e21 0.603M) · kv=1 · **`logit_scale=none`** |
| E21 | query boundary id 10, **r=16**, remainder **on**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed answer 24 / **48-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 92/92/92) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_r16_rem_select 0 \
  --scale bridge --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 16 --message_slots_inplace --message_identity_slots \
  --message_pool_remainder \
  --steps 800 --k1_mult 4
# climbing: same flags --steps 8000 --k1_mult 1
```

Byobu `E25_512_r16_rem_select` / `E25_512_r16_rem_sel8k`. Log:
`msg_r=16  msg_remainder=True  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `message_ratio: 16`,
`message_pool_remainder: true`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`. Recalibrated dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (8k JSON) | **99.8%** | 400 | **47.71** | 0.994 |
| e18 (8k JSON) | **100%** | 400 | **47.94** | 0.999 |
| **e21 @800** | 35.6% | 800 | 4.56 | 0.095 |
| **e21 @8000** | **84.5%** (best **91.4%** @7950) | **8000** | **35.94** | **0.749** |
| e18_local (8k JSON) | 24.6% | 8000 | 0 | 0 |

800 JSON recalibrates S0: dense **47.02**, E18 **47.76**. 8k JSON: dense
**47.71**, E18 **47.94**. No early-stop on E21 (8k budget exhausted).

E21 at 800 was above chance and climbing (chance through ~450, then 32.9% @500
→ 38.9% @750, 35.6% / 4.56 bits @800). Extra-step to 8k. Slow through 6k
(~49%), then 84.8% @7000, best 91.4% @7950, final 84.5% @8000. Recovered bits
**35.94**.

S1 vs 0.75× E18 in the 8k JSON: need **35.95 bits**. E21 has **35.94**
(miss 0.016). Content vs 0.75× dense: need **35.78 bits**. E21 has **35.94**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.8% / 47.71 bits (8k JSON). |
| **S1 vs 0.75× E18** | **FAIL** at 8000. 35.94 < 35.95 bits (0.016 miss). FAIL at 800 (4.56). |
| **content vs 0.75× dense** | **PASS.** 35.94 ≥ 35.78 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.6% / 0 bits. |
| **K3** | not triggered (flow 0.10 at 800, 0.75 at 8k; live, not a floor). |

Do **not** extra-step to 16k (live S1 FAIL at 8k; extra-step rule is 8k if
climbing at 800, not 16k). Do not r=9. Do not r=11. Do not hops. Do not 4k.
Do not Glyph. Do not unfreeze `u`/`delta`. Do not 1024.

## Interpretation

**Live S1 FAIL, not chance.** The same r=16 remainder-on frozen-mean pooler
that MATCH-passes (40.92 bits) is **type-cue live** on SELECT (84.5% / 35.94
bits / flow 0.749) but **0.016 bits short** of 0.75× E18 35.95. Identity r=1
SELECT was 47.98 bits; 16-token means almost bind the cue, not quite at the
E18-relative gate. SELECT is slightly harder than MATCH on this pooler. Do
not call this a one-recipe INDEX+MATCH+SELECT cover at r=16 rem-on.

Do **not** relabel E18's 47.94 bits as E21. E21 recovered **35.94 bits**.

## Decision

Keep the spec in `ahead/`. Next ONE experiment: seq=512 packed
**`select_1decoy`**, inplace frozen mean **r=12 remainder-on** (MATCH rem-on
r=12 was 46.06 S1 PASS — does SELECT clear 0.75× E18 at the next-finer
remainder-on r?). Not r=9/11. Not hops. Not 4k. Not Glyph. Not 1024. Not 16k.
Default remainder stays off elsewhere.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
