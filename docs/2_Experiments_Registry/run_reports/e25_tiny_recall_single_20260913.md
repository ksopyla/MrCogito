# E25 tiny recall_single — E21 vs E18 vs dense (rung 2)

**Date:** 2026-09-13
**Machine:** Cursor Cloud VM (CPU)
**Run ID:** `e25_tiny_recall_single`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_tiny_recall_single/`
**Raw log:** `/opt/cursor/artifacts/e25_tiny_recall_single.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `c9f94bc` (docs tip when this probe launched)
**Git tag:** —
**Related:** [`e25_tiny_far_copy_20260913.md`](e25_tiny_far_copy_20260913.md) · [`e25_tiny_far_copy_e21_steps_20260913.md`](e25_tiny_far_copy_e21_steps_20260913.md)

---

## Goal

After tiny packed `far_copy` near-passed S1 at 8k (47.01 vs 47.29 bits), the next ONE harder
task: keyed `recall_single` (E18's 0-bit content wall). Dense S0 is recalibrated in this JSON.
Not seq=512 and not Glyph. Remainder pooling stays **off**.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · 0.595M (e21 0.603M with compressor) |
| E21 | `message_boundary_token_id=query` (id 10), `message_compress_ratio=16`, remainder **off** |
| Data | on-the-fly DNA `recall_single`, packed answer 16 / **32-bit** prize, seq=128, gap≥16, window 16 |
| Steps | advertised 800, K1=4×; dense early-stop 99% @2850 sets later-arch budget |
| Device | CPU, AdamW 3e-3 |

```
uv run python verification/bapo_capability_probe.py \
  --scale tiny --recipe recall_single --arch dense e18 e21 e18_local \
  --out /opt/cursor/artifacts/e25_tiny_recall_single
```

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **99.2%** | 2850 | 31.37 | 0.980 |
| e18 | 25.3% | 2850 | **0** | 0 |
| **e21** | **37.9%** | 2850 | **3.19** | **0.100** |
| e18_local | 23.5% | 2850 | 0 | 0 |

Dense replica-matches E24 tiny recall (99.2% / 31 bits). E18 replica-matches the 0-bit
content wall. E21 is **not** chance: acc 37.9% (best 38.0%) vs 25%, CE 1.25 vs floor 1.386,
3.19 recovered bits. Versus E18 that is a live slot MATCH signal. Versus dense it is a
wall: 0.75× dense = **23.53 bits / flow 0.735**; E21 has 3.19 / 0.100.

Learning is still rising (acc ~0.25 through 1k → 0.38 @2850; CE still falling). Extra
budget is the next ONE change, same as INDEX at 2400 vs 8k.

## Concept Health

Not a language-model run. Geometry / RankMe / STS-B are not in scope.

Nominal cache `a` (bf16 KV bytes / token): dense 512, E18 128, E21 8, `e18_local` 0.
Effective `a` is recovered bits: dense ~31, E18 **0**, E21 **3.19**, local 0.

## Evaluation

Plots: learning curves, accuracy heatmap, recovered bits, information flow (copied to
`/opt/cursor/artifacts/e25_tiny_recall_single_*.png`). CSV: `capability_table.csv`. JSON:
`tiny_recall_single.json` / `summary.json`.

### Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.2% / 31 bits (75% gate). |
| **S1 vs E18** | E21 **3.19 bits** vs E18 **0**. Slots beat the raw one-read MATCH wall. |
| **S1 vs 0.75× dense** | **FAIL.** Need 23.53 bits / flow 0.735; got 3.19 / 0.100. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 23.5% / 0 bits. |
| **K3** | not triggered (flow 0.100 > 0.05; still climbing) |

Do **not** relabel E18's 0 bits as an E21 score. Do not treat 3.19 bits as closing MATCH.

## Interpretation

Mean-pool r=16 slots carry a **weak keyed-content signal** that E18's raw global read does
not, at the same 2850-step budget. That is surprising relative to E18's 0-bit wall, and
still a content wall relative to dense. The 2850-step INDEX fail was budget; this MATCH
curve has the same shape (slow lift, no plateau). Do not stack compressor geometry.

## Decision

Keep the spec in `ahead/` (MATCH ceiling not measured). Immediate next experiment: **extra
steps** on the same tiny `recall_single`, complete-block E21, 8000 steps
(`--no-dense_first --arch e21`). Dense/E18 ceilings stay this JSON. Not seq=512 (no GPU
here) and not Glyph.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
