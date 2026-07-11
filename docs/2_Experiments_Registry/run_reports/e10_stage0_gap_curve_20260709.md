# E10 Stage 0 — full-vs-blockwise CE gap G vs length (GO, clean extrapolation bet)

> **SUPERSEDED INTERPRETATION — 2026-07-11:** The numeric measurement remains recorded, but
> calling this split “held-out” was incorrect: it came from the same FineWeb-Edu source namespace
> used by training and selected different documents independently at each length. Do not use these
> G values for final recovery fractions. Before training, rerun Stage 0 on the frozen,
> train-disjoint 8K eval-only manifest with the same documents truncated to 2K/8K; append that
> report as the operative gate.

**Date:** 2026-07-09
**Machine:** macOS (Apple Silicon, MPS, fp32, `PYTORCH_ENABLE_MPS_FALLBACK=1`)
**Run ID:** — (no training; prerequisite measurement, zero GPU spend)
**WandB:** —
**Raw output:** `Cache/e10_stage0_full.json`
**Script:** `analysis/run_e10_stage0.py` (`--seq_lens 2048 4096 8192 16384 --num_docs 64 --batch_size 1`)
**Backbone:** `google/gemma-3-1b-pt` (untrained wrapper, `concept_num=0`, `lora_r=0`)
**Eval data:** `HuggingFaceFW/fineweb-edu` `sample-10BT`, streaming split, 64 docs per length (truncated to seq_len, kept only if ≥ seq_len tokens)
**Git commit:** working-tree edits to `analysis/run_e10_stage0.py` (BUCKETS extended through 32K; default `--seq_lens` → `[2048, 4096, 8192, 16384]`; added per-length checkpoint + post-loop decision-tree summary)
**Git tag:** —
**Related TODO:** E10 Stage-0 gate in `docs/experiments_specs/E10_gemma_backbone_concept_memory.md` (Success criteria → Stage 0)

---

## Goal

Pass E10's spec-mandated **Stage 0 go/no-go gate before any GPU training spend**: confirm that the CE gap `G = CE(windowed blockwise) − CE(intact full-attention)` averaged over positions ≥ 1024 is **≥ 0.05 nats at seq 2048**. If Gemma-3-1B doesn't actually lose long-range information when its 4 global layers are windowed, the concept mechanism has nothing to close and the experiment must be re-scoped (longer seq/eval) before training.

Sweeping four lengths (2048, 4096, 8192, 16384) instead of the spec's two-point default was added to also resolve the *shape* of the G curve — needed to decide between (a) keeping the spec as-written (train 2K, eval 8K), (b) bumping training seq to where the gap actually lives, or (c) adding a length curriculum (LongLoRA-style) if the gap grows steeply past the training horizon.

## Configuration

| Item | Value |
|---|---|
| Backbone | `google/gemma-3-1b-pt` (1B, 26 layers = 22 SWA + 4 global, H=1152, SWA window 512) |
| Wrapper config | `concept_num=0`, `concept_block=512`, `lora_r=0`, no training |
| Eval data | `HuggingFaceFW/fineweb-edu` `sample-10BT`, 64 docs per length |
| `--seq_lens` | 2048, 4096, 8192, 16384 |
| Scoring modes | `full_attention` (intact global layers, upper baseline) · `blockwise` (the E10 training protocol without concepts: 512-token blocks + one-block carry, every layer window-masked) |
| Bucket boundaries | (0,512) (512,1024) (1024,2048) (2048,4096) (4096,8192) (8192,16384) (16384,32768) |
| Device / dtype | MPS / fp32 |
| Runtime | ~25 min total (no OOM; full 16K forward fit on unified memory) |

## Result — the G curve

| seq | G (positions ≥ 1024) | vs 0.05 gate | per-length verdict |
|---|---|---|---|
| 2048 | **0.2840** | 5.7× above | GO |
| 4096 | 0.3136 | 6.3× above | GO |
| 8192 | 0.3176 | 6.4× above | GO |
| 16384 | 0.3645 | 7.3× above | GO |

`G(8K) / G(2K) = 1.12×` (gentle growth — well below the 3.0× "steep" threshold that would trigger a curriculum amendment).

## Per-bucket breakdown

| bucket | 2K full | 2K win | 2K gap | ··· | 16K full | 16K win | 16K gap |
|---|---|---|---|---|---|---|---|
| [0, 512) | 2.561 | 2.561 | **0.000** | | 2.511 | 2.511 | **0.000** |
| [512, 1024) | 2.437 | 2.605 | 0.169 | | 2.522 | 2.690 | 0.167 |
| [1024, 2048) | 2.451 | 2.735 | **0.284** | | 2.524 | 2.793 | **0.269** |
| [2048, 4096) | — | — | — | | 2.490 | 2.798 | 0.308 |
| [4096, 8192) | — | — | — | | 2.440 | 2.789 | 0.349 |
| [8192, 16384) | — | — | — | | 2.379 | 2.777 | **0.398** |

Two structural observations:
- **Gap is exactly zero in the local window** ([0, 512)) at every length — the mechanism is sound; windowing only starts hurting past window+carry reach.
- **Windowed CE saturates around 2.77–2.80 nats** at every length, regardless of how far past the window you go. Full-attention CE keeps dropping with position (2.56 → 2.38). That saturation ceiling is exactly the long-range target the concept state has to break through.

## Interpretation

1. **Stage 0 gate passed decisively.** G(2K) = 0.284 ≫ 0.05. The windowed-backbone protocol genuinely loses information past local reach, even at the shortest length E10 cares about. The hypothesis has something to bite on.
2. **Spec as-written is well-posed — no curriculum amendment needed.** G grows only 1.12× from 2K to 8K, so the 4× extrapolation bet (train 2K, eval 8K) is *not* asking the concept state to bridge a gap wildly larger at eval than at training. This is the `KEEP_SPEC` branch of the decision tree.
3. **Real headroom for the PRIMARY criterion.** Spec says the concept arm must close ≥ 40% of G at positions ≥ 1024. At 2K that is ≥ 0.114 nats of recovery — a meaningful, measurable target, not a noise floor.
4. **Length extrapolation criterion also well-posed.** At 8K, G = 0.318. Spec wants concept arm to close ≥ 20% of G₈K = 0.064 nats at 4× training horizon — achievable if the recurrence works at all, clearly diagnosable if it doesn't.

## Decision

**Stage 0 GO. Proceed to E10 training (concept arm + matched `CONCEPT_NUM=0` control arm) without re-scoping seq/eval and without adding a length curriculum.** Spec training seq 2048 / eval horizon 8K stands. Foundation is implemented (`nn/backbone_concept_lm.py`, `causal_lm` objective, `scripts/launch_e10.sh`, 14 unit tests green); the only pre-training prerequisite was this gate.

## Notes

- **n=64 docs per length.** Per-doc CE variance is non-trivial; the smoke at n=2 gave G(2K)=0.234, the n=64 run gives 0.284 — same ballpark, but the headline G numbers carry roughly ±0.03–0.05 nats of doc-sampling noise. The qualitative verdict (clean GO, gentle growth) is robust to that; the absolute ratios (G(8K)/G(2K), 40%-of-G target) should be read with that uncertainty in mind.
- **16K docs are rare in fineweb-edu `sample-10BT`.** All four lengths hit the n=64 target without the script's "only N docs found" warning firing, but the 16K row is closest to the scarcity edge. If future Stage-0 re-runs go longer than 16K, switch the eval-data source to a dedicated long-doc set (FinePDFs or PG-19) to avoid the warning.
- **MPS fp32 fit at 16K** without OOM — the 16K×16K attention matrices on the 4 global layers in `full_attention` mode were fine on this Mac's unified memory. A CUDA run on a 24 GB 3090 would use bf16 and be ~10–20× faster.

*Related: `docs/experiments_specs/E10_gemma_backbone_concept_memory.md` (Stage 0 gate), `master_experiment_log.md` (Evaluation Experiments section), `agenda.md` (Current focus).*
