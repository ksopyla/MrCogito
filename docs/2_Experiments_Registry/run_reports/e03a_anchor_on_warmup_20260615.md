# E03A — Anchor-ON Warmup (De-collapse via Frozen-Encoder Anchor) — `concept_ar_H768L6C128D4_20260614_164206`

**Date:** 2026-06-15
**Machine:** Odra (3× RTX 3090, 24 GB VRAM each)
**Run ID:** `concept_ar_H768L6C128D4_20260614_164206`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_H768L6C128D4_20260614_164206)
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260614_164148.log`
**Best checkpoint:** `Cache/Training/concept_ar_H768L6C128D4_20260614_164206/checkpoint-19000`
**Git commit:** `0c14061`
**Git tag:** —
**Related TODO:** E03 in `docs/experiments_specs/done_success/E03_concept_anchor_decollapse.md`

> **Note:** This run was launched without `EXPERIMENT_ID=E03` so W&B shows no experiment tag.
> The matched control arm (anchor-OFF) is queued as the next Odra run.

---

## Goal

Run the first arm of the matched E03 pair: **anchor-ON warmup (~0.3 epoch)** with a frozen
SmolLM2-135M teacher providing per-token hidden-state MSE targets. The 0.3-epoch budget is the
spec's 25%-of-budget kill-gate checkpoint: verify that anchor MSE decreases, AR eval CE does not
diverge, and concept rank moves above the control baseline before committing to a full 1-epoch run.

---

## Configuration

| Item | Value |
|---|---|
| Family | `concept_ar` |
| Encoder | H768, L6, C128, BiXT |
| Decoder | causal AR, D4, `decoder_word_dropout=0.2` |
| Token width | `token_embedding_dim=256` |
| Anchor | `anchor_loss=True`, SmolLM2-135M (frozen), λ=0.5, standardize=True |
| Dataset | `HuggingFaceFW/fineweb-edu`, `sample-10BT` (9.57M / 100k) |
| Objective | AR denoising reconstruction (`reconstruction`) |
| Corruption | token deletion `deletion_rate=0.6`, word-dropout 0.2 |
| Concept losses | None |
| Epochs | 0.30 (warmup gate budget) |
| Global steps | 19,942 |
| Effective batch | 24 × grad_accum 2 × 3 GPUs = 144 |
| Throughput | 68.3 samples/s · 0.474 train steps/s |
| Runtime | ~11.7 h (42,058 s) |
| Precision | bf16 |

---

## Training Outcome

Training completed cleanly to 0.30 epochs. `load_best_model_at_end=True` selected **checkpoint-19000**
(eval_loss 4.4457) as the best model; the final checkpoint-19942 also at eval_loss ~4.446.

| Metric | checkpoint-19000 (best) | checkpoint-19942 (last) |
|---|---|---|
| eval loss (AR CE) | **4.4457** | ~4.446 |
| anchor/mse_eval | 0.512 | 0.512 |
| train_loss (epoch avg) | — | 5.444 |

The eval loss (4.4457) is lower than E01's best (4.676) despite running 5× more steps — the anchor
auxiliary loss neither destabilizes training nor inflates eval AR CE. Anchor MSE
decreased over training (~0.65+ early → 0.512 at best eval), confirming the head is learning.

---

## Concept Health

All numbers from `analysis/run_concept_analysis.py` (2026-06-15), `--num_batches 20 --batch_size 16`.

| Metric | checkpoint-19000 (best) | E01 best (ck-4000) | E01 last (ck-37392) | E03 gate |
|---|---|---|---|---|
| Slot eff. rank (batch-avg SVD) | 10.34 / 128 (8.1%) | 14.64 / 128 | 4.64 / 128 | secondary |
| **Manifold RankMe** (per-sample) | **167.09** | — | — | ≥ control + 8 (PRIMARY) |
| Manifold anisotropy | 0.492 | — | — | — |
| Dims for 95% var (slot) | 32.95 | 43.15 | 4.0 | — |
| Active slot fraction | 1.0 (100%) | — | — | — |
| CE intact | 4.576 | 4.579 | 9.248 | — |
| CE zero (floor) | 5.523 | 6.054 | 9.706 | — |
| CE shuffle | 5.921 | 6.081 | 12.634 | — |
| **Δzero** | **0.947** nats | 1.476 nats | 0.457 | ≥ 0.5 ✓ |
| **Δshuffle** | **1.345** nats | 1.502 nats | 3.386 | ≥ 0.5 ✓ |
| **Δzero early-pos** | **2.320** nats | — | — | ≥ control (PRIMARY) ✓ |
| **Δshuffle early-pos** | **3.342** nats | — | — | ≥ control (PRIMARY) ✓ |
| CE intact (wd-matched) | 4.448 | — | — | — |
| Gap clean-vs-wd | 0.128 | — | — | ✓ < 0.2 |

Key observations:
- **Slot eff. rank (10.34)** is similar to E01-best (14.64) and E02-best (11.57). It did not spike higher — but crucially, it did NOT collapse to E01-last (4.64) even after 19k steps. The anchor is holding the geometry steady.
- **Manifold RankMe (167.09)** is the per-sample entropy-based rank (new metric from `0c14061`). No prior comparison baselines exist yet; the control arm will provide the matched reference.
- **Concept ablation is strong:** both all-position and early-position Δ pass their gates. Early-position Δshuffle of **3.342** is substantially higher than E01-best (no early split available there, but the W&B training eval at epoch 0.286 reported Δshuffle_early=1.32). This suggests the anchor is increasing concept information density at early positions (where AR bypass is hardest).
- Anchor MSE (eval) = **0.512** and decreasing throughout training. ✓

JSON artifacts: `Cache/Evaluation_reports/e03a_best_concept_analysis.json`

---

## Evaluation

### Tier 2 — Zero-shot STS-B

Evaluated on checkpoint-19000 (2026-06-15).
WandB eval: [bench-stsb_zero_shot-checkpoint-19000](https://wandb.ai/ksopyla/MrCogito/runs/84lc12ha)
CSV: `Cache/Evaluation_reports/bench-stsb_zero_shot-checkpoint-19000-73M-enc-20260615_2008-results.csv`

| Checkpoint | Pearson | Spearman | Gate (≥ 0.62) |
|---|---|---|---|
| checkpoint-19000 (best) | **0.556** | 0.572 | FAIL |
| E01 best (checkpoint-4000, ref) | 0.556 | 0.575 | FAIL |
| E02 best (checkpoint-78000, ref) | 0.702 | 0.701 | ✓ |
| Prior best (perceiver_denoise) | 0.607 | 0.622 | — |

STS-B (0.556) equals E01-best. This run used reconstruction (not prefix→suffix), same as E01 —
the semantic signal is on par despite 5× more steps and the anchor auxiliary loss, which
neither helps nor hurts zero-shot STS-B at the 0.3-epoch budget.

---

## Interpretation

This is the **anchor-ON arm of the matched E03 warmup pair**. It passes every continuation
gate defined in the spec:

1. **Anchor MSE decreasing ✓** — MSE goes from ~0.65+ to 0.512; the anchor head is learning.
2. **AR CE stable ✓** — eval_loss 4.446, lower than E01-best (4.676), no divergence.
3. **Concept ablation strong ✓** — Δshuffle 1.345 and Δshuffle_early 3.342, both ≥ 0.5 gate.
4. **Compute within cap ✓** — ~35 GPU-hours (well below 60-GPU-hour pair cap).

The spec's **kill gates are NOT triggered**. The anchor-ON arm is healthy and the pair should
proceed to the matched control arm.

The spec's success criteria cannot be evaluated yet because the **matched control arm (anchor-OFF)
has not been run**. Without the control, we cannot compute `rank(anchor) − rank(control)` or
`STS-B(anchor) − STS-B(control)`. The STS-B of 0.556 matches E01-best at a much earlier
step (19k vs 4k), but this is an apples-to-apples comparison only after the control warmup.

One notable signal: early-position Δshuffle (3.342) is much higher than what the W&B training
eval reported for prior runs at similar stages. This is a positive early sign that the anchor
is increasing information density at concept-dependent positions, but it requires the matched
control for quantification.

---

## Decision

**Verdict: inconclusive** — Anchor-ON arm complete. All continuation gates pass. The matched
control arm (anchor-OFF, `EXPERIMENT_ID=E03 ANCHOR_LOSS=false`) must be run next on Odra to
evaluate the spec's de-collapse criteria. The full E03 verdict is deferred until both arms are
compared. If the control arm also passes its kill gates, the next step is a matched full-epoch
pair (1 epoch each) for the definitive STS-B and rank comparison.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E03_concept_anchor_decollapse.md`, `agenda.md`*
