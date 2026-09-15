# E25 bridge 512 recall_single r=8 in-place frozen mean — E21 vs E18 vs dense (rung 5s)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_r8_mean_recall` (800) · `e25_512_ip_r8_mean_recall_s8k` (8000)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_r8_mean_recall/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_r8_mean_recall_probe.log` (8k; also `_800.log`)
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `faded5d` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** r=4 MATCH PASS [`e25_bridge512_ip_r4_mean_recall_20260914.md`](e25_bridge512_ip_r4_mean_recall_20260914.md) · r=16 MATCH wall [`e25_bridge512_ip_r16_mean_recall_20260914.md`](e25_bridge512_ip_r16_mean_recall_20260914.md)

---

## Goal

Pooling threshold on packed `recall_single` at seq=512. Locked: r=1 identity
**43.08 bits @800**; r=4 frozen mean **47.04 bits @1900**; r=16 frozen mean
**0 bits**. One change: keep inplace + `--message_identity_slots` (no learned
`u`/`delta`, no raw_kv) and set `--message_ratio 8`. Halfway to the MATCH wall.

H=128 (same 512 MATCH width). Score vs 0.75× E18; also report vs 0.75× dense.
If E18 were ~0, would not call S1 from 0.75×0. Climbing at 800 → extra-step 8k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · 0.595M (e21 0.603M) · kv=1 · **`logit_scale=none`** |
| E21 | query boundary id 10, **r=8**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed answer 24 / **48-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_r8_mean_recall 0 \
  --scale bridge --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 8 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# climbing: same flags --steps 8000 --k1_mult 1
```

Byobu `E25_512_r8_recall` / `E25_512_r8_recall_s8k`. Log:
`msg_r=8  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `message_ratio: 8`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (8k JSON) | **100.0%** | 400 | 47.95 | 0.999 |
| e18 (8k JSON) | **100.0%** | 400 | **47.96** | 0.999 |
| **e21 @800** | 56.5% | 800 | 15.90 | 0.331 |
| **e21 @7100** | **99.1%** | **7100** | **47.16** | **0.982** |
| e18_local (8k JSON) | 24.9% | 8000 | 0 | 0 |

800 JSON recalibrates S0: dense **47.97**, E18 **47.79**. 8k JSON: dense **47.95**,
E18 **47.96**. E21 early-stop 99.1% @7100 (8k budget).

E21 at 800 was above chance and still climbing (chance through ~400, then 35.0%
@450 → 56.5% @800). Slower than r=4 (78.3% / 29.76 @800). r=16 frozen mean on
the same task was **0 bits / floor at 800**.

S1 vs 0.75× E18 in the 8k JSON: need **35.97 bits**. E21 has **47.16**.
Content vs 0.75× dense: need **35.96 bits**. E21 has **47.16**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 47.95 bits (both JSONs). |
| **S1 vs 0.75× E18** | **PASS** at 7100. 47.16 ≥ 35.97 bits. FAIL at 800 (15.90). |
| **content vs 0.75× dense** | **PASS.** 47.16 ≥ 35.96 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.9% / 0 bits. |
| **K3** | not triggered (flow 0.33 at 800, 0.98 at 7100). |

Do **not** extra-step past 8k (S1 already PASS @7100). Do not 4k. Do not Glyph.
Do not remainder. Do not unfreeze `u`/`delta`. Do not hops.

## Interpretation

**MATCH survives 8-token frozen means.** Exclusive identity still binds a key at
`k_norm(mean of 8)` / `mean(v)`. The MATCH wall is **between r=8 (47.16 bits) and
r=16 (0 bits)**, not between 4 and 8. r=8 is slower than r=4 (~7k vs ~2k) but
recovers **47.16 / 48 prize bits**.

Do **not** relabel E18's 47.96 bits as E21. E21 recovered **47.16 bits**.

## Decision

Keep the spec in `ahead/`. Next ONE experiment: **inplace r=12 frozen mean**
(`--message_ratio 12 --message_slots_inplace --message_identity_slots`, no raw_kv)
on this recipe. One change: pooling width. If that PASSES, the wall is nearer
r=16. If chance, MATCH dies between 8 and 12.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
