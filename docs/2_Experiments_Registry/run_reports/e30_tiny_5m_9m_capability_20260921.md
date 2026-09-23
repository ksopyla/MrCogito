# E30 capability — fair tiny concat, then 5.16M / 9.04M

**Date:** 2026-09-21
**Machine:** Cursor Cloud VM (CPU, 4 threads, torch 2.14 CPU)
**Run IDs:** `e30_tiny_fair_concat` · `e30_tiny_wide` · `e30_5m_tiny` · `e30_5m_tiny_index_budget` · `e30_5m_tiny_index_lr1e3` · `e30_5m_tiny_match_lr1e3` · `e30_9m_tiny_index` · `e30_9m_tiny_match`
**WandB:** n/a (BAPO probe; no `compute/*` audit)
**Raw JSON/plots:** `/opt/cursor/artifacts/e30_tiny_fair_concat/` · `e30_tiny_wide/` · `e30_5m_tiny/` · `e30_9m_tiny_index/` · `e30_9m_tiny_match/`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git tag:** —
**Related:** [E30 spec](../../experiments_specs/ahead/E30_sliding_window_perceiver.md) · [small-model protocol](../../engineering_specs/small_model_capability_protocol.md)

---

## Goal

Fair write comparison of E30 overlapping Perceiver banks vs E21 frozen-mean concat
(`--message_identity_slots`, **no** inplace), after the first tiny smoke mixed inplace
e21 with concat e30. Then scale width-matched 4-layer models to ~5M and ~9M (under
the 10M cap) and check LR.

## Configuration

| knob | tiny 0.6M | 5.16M | 9.04M |
|---|---|---|---|
| hidden / head_dim / stack | 128 / 32 / 2 | 384 / 64 / 2 | 512 / 64 / 2 |
| dense / e21 / e30 params | 0.595 / 0.603 / 0.617M | 5.108 / 5.157 / 5.163M | 8.969 / 9.035 / 9.041M |
| SWP heads / qdim | 4 / 128 | 8 / 128 (pinned) | 8 / 128 |
| E21 control | identity concat, r=16, no inplace | same | same |
| device | CPU AdamW, packed CE | same | same |

H=384 `stack_layers=6` is 10.03M — over the cap. Width, not extra SWA depth.

## Training outcome

### Tiny seq=128, LR 3e-3, identity concat (fair)

INDEX `far_copy` 64-bit prize; MATCH `recall_single` 32-bit. Dense INDEX this replica
peaked 80.9% / 43.5 bits (S0 pass, still falling); E18 INDEX 63.05 @1400.

| arch | INDEX bits @step | MATCH bits @step |
|---|---|---|
| dense | 43.52 @3200 | 31.37 @2850 |
| e18 | 63.05 @1400 | **0** (E25 wall) |
| e18_local | 0 | 0 |
| e21 frozen-mean concat | **16.55 @3200** | **3.03 @2850** |
| e30 | 12.19 @3200 | **0.01** RankMe 1.05 |

E21 concat beats the previous identity+inplace 12.3 INDEX bits at this budget.
E30 INDEX is live (entropy/log W 0.82, `none` chance) but slower gist than frozen
mean. Tiny MATCH is not the claim: E18 is the 0-bit wall.

LR hunt on e30 INDEX @1600: **1e-3 chance**; **3e-3 0.34 acc**; **1e-2 0.40 acc /
9.5 bits**. Protocol 3e-3 is right at H=128.

### Tiny_wide seq=256, LR 1e-3, 0.6M

INDEX prize 48 bits. Dense 47.78 @3900 (K1 extend then 99%). Geometry K=16 W=128
n_windows=3.

| arch | INDEX bits @3900 |
|---|---|
| dense | 47.78 |
| e18 | 6.26 |
| e21 | 4.47 |
| **e30** | **7.75** (late takeoff ~3500–3900) |

E30 > E18 > E21 on this two-window analogue. E18 itself is weak vs dense. MATCH
dense 33% @3200 K1 — takeoff started, 0.2-nat last-third rule did not extend;
uncalibrated.

### 5.16M seq=128

At **LR 3e-3** dense INDEX early-stops 99% @1600 and starves writes. E30 RankMe
**1.01 / 0 bits**. E18 MATCH **breaks the 0-bit wall**: 25.35 bits / 85% @3200.
E21 MATCH 11.63 bits @6400 (k1 extend). Extra INDEX budget 3200/6400 at 3e-3:
e21 **37.10 bits / 68%**; e30 still **0 bits**.

At **LR 1e-3** E30 INDEX **11.95 bits / 36% @2400**, RankMe 3.59 (not collapsed).
E30 MATCH **3.76 bits / 38% @3200**. Width needs a lower LR than the tiny H=128 default.

### 9.04M seq=128, LR 1e-3

| arch | INDEX | MATCH |
|---|---|---|
| e21 frozen-mean concat | 13.64 bits / 39% @2400 | 4.84 bits / 40% @2400 |
| **e30** | **36.96 bits / 73% @4800** (k1 extend; already 63% @2400) | **12.30 bits / 45% @4800** |

INDEX: E30 beats frozen mean at matched 2400 (CE 0.76 vs 1.09) and after the
extension. entropy/log W **0.66**, RankMe 5.3, `none` chance — S2 pick + load-bearing.
MATCH: E30 12.30 vs e21 4.84; vs 5.16M live E18 25.35 the 0.75× bar is **19.0** —
**S1 miss at seq=128**. RankMe 1.87. Claim length remains seq=512 E21-mean wall.

## Gates vs E30 spec (seq=128 is still smoke / width hunt)

| gate | tiny 0.6M 3e-3 | 5.16M 1e-3 | 9.04M 1e-3 |
|---|---|---|---|
| S0 dense ≥75% | PASS (INDEX 81–99%; MATCH 99%) | PASS (INDEX 99%; MATCH 76%) | dense not re-run; 5.16M S0 holds |
| K2 e18_local | PASS | not re-run | not re-run |
| S1 MATCH ≥0.75× live E18 | n/a (E18 0 bits) | 3.76 vs 19.0 **miss** | 12.30 vs 19.0 **miss** |
| S2 entropy/log W <0.85 and `none` chance | INDEX yes / MATCH smear | INDEX 0.86 borderline | INDEX **0.66 yes** / MATCH 0.83 |
| S3 INDEX ≥0.75× live E18 | 12.2 vs 47.3 miss | 12.0 vs 36.8 miss (5M e18 49 bits @1600) | **37.0 vs ~36.8** thin pass vs 5M e18; 9M e18 not measured |
| K5 INDEX collapse vs e21 mean | slower, not dead | dead at 3e-3; live at 1e-3 | **e30 beats e21** |

Do not kill. The MATCH claim is still the E21-mean wall at seq=512 with `n_windows≥2`
and LR 3e-4.

## Protocol lessons

1. Fair E21 control is **concat frozen-mean**, not inplace.
2. LR is length **and** width: 3e-3 trains H=128; 3e-3 RankMe-collapses E30 at ≥5M.
3. Dense 99% early-stop at 1600 starved 5M writes — non-dense budget is now
   `max(steps × k1_mult, dense_steps_used)`.
4. Compare 4-layer width (0.6 / 5.16 / 9.04M), not extra SWA depth.
