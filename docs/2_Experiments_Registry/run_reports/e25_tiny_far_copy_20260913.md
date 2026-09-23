# E25 tiny far_copy — E21 vs E18 vs dense (first rung)

**Date:** 2026-09-13
**Machine:** Cursor Cloud VM (CPU)
**Run ID:** `e25_tiny_far_copy`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_tiny_far_copy/`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `73f2a31` (probe that produced these numbers)
**Git tag:** —
**Related TODO:** E25 first calibrated rung; remainder-block pooling is the next architecture change

---

## Goal

First calibrated E21 exam: exclusive compressed read (DNA `query` severs SWA; prefix is r=16
slots) on packed `far_copy` at seq=128. Dense and E18 are the reused solvability / uncompressed
ceilings from E24, re-run in the same JSON. Complete-block-only pooling (faithful E21 port);
remainder sender tokens next to QUERY were **not** pooled.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · ~0.595M (e21 0.603M with compressor) |
| E21 | `message_boundary_token_id=query` (id 10), `message_compress_ratio=16`, remainder pooling **off** |
| Data | on-the-fly DNA `far_copy`, packed span 32 / **64 bits**, gap≥17, window 16 |
| Steps | advertised 800, K1=4×; dense/e18 early-stop at 99% |
| Device | CPU, AdamW 3e-3, 50-step warmup, packed CE on `y` only |

```
uv run python verification/bapo_capability_probe.py \
  --scale tiny --recipe far_copy --arch dense e18 e21 e18_local \
  --out /opt/cursor/artifacts/e25_tiny_far_copy
```

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **99.4%** | 2400 | 63.31 | 0.989 |
| e18 | **99.2%** | 1400 | 63.05 | 0.985 |
| **e21** | **51.5%** | 2400 | **17.70** | **0.277** |
| e18_local | 24.7% | 2400 | 0 | 0 |

S0 holds (dense replica matches E24). K2 holds (`e18_local` at chance). S1 fails: E21 flow
is 0.28 × E18, below the 0.75× gate (need ≥ 47 bits). K3 does **not** fire: E21 is not at
chance (51.5% vs 25%; 17.7 bits). The slot channel is live and lossy.

Learning still rising at 2400 (acc 0.500 @2000 → 0.515 @2400). Extra budget is a
follow-up, not this rung's architecture change.

## Concept Health

Not a language-model run. Geometry / RankMe / STS-B are not in scope.

Nominal cache `a` (bf16 KV bytes / token): dense 512, E18 128, E21 8 (global KV / r),
`e18_local` 0. Effective `a` is recovered bits: dense/E18 ~63, E21 **17.7**, local 0.

## Evaluation

Plots: learning curves, accuracy heatmap, recovered bits, information flow (copied to
`/opt/cursor/artifacts/e25_tiny_far_copy_*.png`). CSV: `capability_table.csv`. JSON:
`tiny_far_copy.json` / `summary.json`.

### Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.4% / 63 bits. |
| **S1** | **FAIL.** E21 17.7 bits vs E18 63.0 (flow 0.277 vs 0.985). Need 0.75× E18 ≈ 47 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 24.7% / 0 bits. |
| **K3** | not triggered (not chance; still climbing slowly) |

## Interpretation

E21 at r=16 on tiny INDEX is a **partial copy channel**, not a dead one and not E18. Complete
homogeneous blocks only drop the last incomplete sender block (up to 15 tokens next to
QUERY). A 32-token span often straddles that remainder, which matches ~half accuracy / ~18
bits rather than 0 or 64.

Do **not** score recall or 512 yet. Do not relabel these as E18 scores.

## Decision

Keep the spec in `ahead/` (only the first rung ran; K3 not hit). Immediate next experiment:
one architecture change — `message_pool_remainder` (default off, so this JSON stays
reproducible) and re-run **only** this tiny packed `far_copy`. Same S1. Extra step budget
is not a substitute.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
