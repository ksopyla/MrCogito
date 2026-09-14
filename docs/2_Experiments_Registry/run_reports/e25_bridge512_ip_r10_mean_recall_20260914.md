# E25 bridge 512 recall_single r=10 in-place frozen mean — E21 vs E18 vs dense (rung 5u)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_r10_mean_recall`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_r10_mean_recall/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_r10_mean_recall_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `c2a0276` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** r=8 MATCH PASS [`e25_bridge512_ip_r8_mean_recall_20260914.md`](e25_bridge512_ip_r8_mean_recall_20260914.md) · r=12 MATCH S1 fail [`e25_bridge512_ip_r12_mean_recall_20260914.md`](e25_bridge512_ip_r12_mean_recall_20260914.md) · r=16 MATCH wall [`e25_bridge512_ip_r16_mean_recall_20260914.md`](e25_bridge512_ip_r16_mean_recall_20260914.md)

---

## Goal

Pooling threshold on packed `recall_single` at seq=512. Locked: r=8 frozen mean
**47.16 bits @7100** (S1 PASS); r=12 frozen mean **34.33 bits @8000** (S1 FAIL,
live); r=16 frozen mean **0 bits**. One change: keep inplace +
`--message_identity_slots` (no learned `u`/`delta`, no raw_kv) and set
`--message_ratio 10`. Remainder **off** (512/10 is not integer: 51 complete
blocks + 2 leftover). Do 10-token frozen means still bind a key at the S1 bar?

H=128 (same 512 MATCH width). Score vs 0.75× E18; also report vs 0.75× dense.
If E18 were ~0, would not call S1 from 0.75×0. Floor at 800 → no extra-step.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · 0.595M (e21 0.603M) · kv=1 · **`logit_scale=none`** |
| E21 | query boundary id 10, **r=10**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed answer 24 / **48-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_r10_mean_recall 0 \
  --scale bridge --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 10 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
```

Byobu `E25_512_r10_recall`. Log:
`msg_r=10  msg_remainder=False  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `message_ratio: 10`,
`message_pool_remainder: false`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`. Recalibrated dense S0 in the same JSON.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **99.9%** | 350 | **47.89** | 0.998 |
| e18 | **100%** | 350 | **47.94** | 0.999 |
| **e21 frozen mean r=10** | 25.5% | 800 | **0.00** | **0** |
| e18_local | 25.2% | 800 | 0 | 0 |

E18 is **live** (47.94 bits). Do not score S1 against 0.75×0. E21 sat at chance
every eval (best 25.5%; CE stuck at ln(4) ~1.386–1.390). r=12 on the same task
was already live at 800 (39.8% / 5.79 bits). This curve is a **floor**, not a
climb. No extra-step to 8k.

S1 vs 0.75× E18: need **35.96 bits**. E21 has **0.00**.
Content vs 0.75× dense: need **35.92 bits**. E21 has **0.00**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.9% / 47.89 bits. |
| **S1 vs 0.75× E18** | **FAIL.** 0.00 vs 35.96 bits. |
| **content vs 0.75× dense** | **FAIL.** 0.00 vs 35.92 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.2% / 0 bits. |
| **K3** | chance/floor at 800 (flow 0). Do not extra-step. |

Do **not** extra-step (floor, not climbing). Do not r=9. Do not r=11. Do not
remainder-on. Do not 16k r=12. Do not hops. Do not 4k. Do not Glyph. Do not
unfreeze `u`/`delta`. Do not 1024.

## Interpretation

**Pooling width is not monotone.** Chance at r=10 is unexpected vs r=12 live:

- r=1 identity: S1 **PASS** (43.08)
- r=4: S1 **PASS** (47.04)
- r=8: S1 **PASS** (47.16)
- r=10: **chance 0 bits** @800 (this rung)
- r=12: live, S1 **FAIL** at 8k (34.33)
- r=16: chance **0 bits**

This is not a clean 8–10 or 10–12 S1 wall. r=10 died at the hunt budget while
r=12 was already climbing at 800. Remainder-off leftover size does not explain
it (r=10 leaves 2 tokens; r=12 leaves 8 and is live; r=16 divides 512 and is
chance). S1-passing frozen-mean MATCH at this recipe is r∈{1,4,8}.

Do **not** relabel E18's 47.94 bits as E21. E21 recovered **0 bits**.

## Decision

Keep the spec in `ahead/`. **Stop stacking MATCH r-sweeps.** Next ONE
experiment: stop MATCH r-hunts (not r=9, not r=11, not remainder-on, not 16k
r=12). Architecture change after this measured non-monotone wall, not another
pooling-width hunt.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
