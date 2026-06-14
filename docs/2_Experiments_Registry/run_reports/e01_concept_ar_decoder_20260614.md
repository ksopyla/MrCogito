# E01 — Concept AR Decoder (Denoising Reconstruction) — `concept_ar_H768L6C128D4_20260613_185955`

**Date:** 2026-06-14
**Machine:** Polonez (4× RTX 3090, 24 GB VRAM each)
**Run ID:** `concept_ar_H768L6C128D4_20260613_185955`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_H768L6C128D4_20260613_185955)
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260613_184713.log`
**Best checkpoint:** `Cache/Training/concept_ar_H768L6C128D4_20260613_185955/checkpoint-4000`
**Git commit:** `6c48d12`
**Git tag:** `arch/perceiver-denoise-reset-51-g6c48`
**Related TODO:** E01 in `docs/experiments/E01_concept_ar_decoder.md`

---

## Goal

Establish the E01 baseline for the concept-conditioned AR decoder family. Verify that the BiXT encoder → 128 concepts → causal AR decoder (reconstruction) pipeline works, and measure concept ablation ΔCE to confirm the decoder actually uses the concept bottleneck. The main posterior-collapse defence is decoder-word-dropout (`p=0.2`) + lean L4 decoder.

---

## Configuration

| Item | Value |
|---|---|
| Family | `concept_ar` |
| Encoder | H768, L6, C128, BiXT |
| Decoder | causal AR, D4, `decoder_word_dropout=0.2` |
| Token width | `token_embedding_dim=256` |
| Dataset | `HuggingFaceFW/fineweb-edu`, `sample-10BT` (9.57M / 100k) |
| Objective | AR denoising reconstruction (`ar_denoising_reconstruction`) |
| Corruption | token deletion `deletion_rate=0.6`, word-dropout 0.2 |
| Concept losses | None |
| Epochs | 1 |
| Global steps | 37,392 |
| Effective batch | 32 × grad_accum 2 × 4 GPUs = 256 |
| Throughput | 242 samples/s · 0.945 train steps/s |
| Runtime | ~11.0 h (39,557 s) |
| Precision | bf16 |
| W&B group | `E01_concept_ar_H768L6C128D4` |

---

## Training Outcome

Training completed cleanly to 1 epoch. `load_best_model_at_end=True` selected **checkpoint-4000** (eval_loss 4.676) as the best model; the final checkpoint-37392 has eval_loss 6.509. The model overfit: eval CE rose monotonically after the first eval point (~step 4000), and W&B sparklines confirm increasing eval/loss throughout training. This is the defining finding of this run — the reconstruction objective provides strong early concept usage signal but cannot sustain it over a full epoch.

| Metric | checkpoint-4000 (best) | checkpoint-37392 (last) |
|---|---|---|
| eval loss | 4.676 | 6.509 |
| train loss (final step) | — | 2.702 |
| train_loss (epoch avg) | — | 3.376 |

---

## Concept Health

All numbers from `analysis/run_concept_analysis.py` run 2026-06-14.

| Metric | checkpoint-4000 (best) | checkpoint-37392 (last) | E01 gate |
|---|---|---|---|
| Effective rank | **14.64 / 128** (11.4%) | 4.64 / 128 (3.6%) | > 32/128 |
| Dimensions for 95% var | 43.15 | 4.0 | — |
| Participation ratio | 31.54 | 2.52 | — |
| ce_intact | 4.579 | 9.248 | — |
| ce_zero | 6.054 | 9.706 | — |
| ce_shuffle | 6.081 | 12.634 | — |
| **Δzero** | **1.476 nats** | 0.457 nats | ≥ 0.5 |
| **Δshuffle** | **1.502 nats** | 3.386 nats | ≥ 0.5 |

Notes:
- At checkpoint-4000 the decoder genuinely uses concepts (Δshuffle 1.502 ≥ 0.5 gate; Δzero 1.476 ≥ 0.5 gate).
- At checkpoint-37392 the geometry has completely collapsed to 4 dimensions (dimensions_for_95_variance = 4.0). Δzero barely misses the gate; Δshuffle appears large (3.386) but relative to a very high ce_intact (9.25), signalling that the CE has inflated rather than ablation genuinely mattering.
- Collapse is progressive: effective rank drops from 14.64 → 4.64 over training. The reconstruction objective with word-dropout `p=0.2` does not maintain the bottleneck under longer training.

JSON artifacts: `Cache/Evaluation_reports/concept_ar_H768L6C128D4_20260613_185955_best_concept_analysis.json` and `..._last_concept_analysis.json`.

---

## Evaluation

### Zero-shot STS-B

WandB eval (best, checkpoint-4000): `bench-stsb_zero_shot-checkpoint-4000-73M-enc-20260614_0940`
WandB eval (last, checkpoint-37392): [bench-stsb_zero_shot-checkpoint-37392-73M-enc-20260614_0940](https://wandb.ai/ksopyla/MrCogito/runs/8wlm9bq1)

| Checkpoint | Pearson | Spearman | Gate (≥ 0.62) |
|---|---|---|---|
| checkpoint-4000 (best) | **0.556** | 0.575 | FAIL |
| checkpoint-37392 (last) | 0.207 | 0.341 | FAIL |
| Prior best (perceiver_denoise) | 0.607 | 0.622 | — |

The best checkpoint (4000) achieves 0.556 — below the E01 gate (0.62) and below the prior perceiver_denoise baseline (0.607), but well above the collapsed last checkpoint (0.207). The semantic signal clearly peaks with the concept geometry.

CSV artifacts: `Cache/Evaluation_reports/bench-stsb_zero_shot-checkpoint-4000-73M-enc-20260614_0940-results.csv` and `...-checkpoint-37392-...`.

---

## Interpretation

E01 confirms AR plumbing works and that the decoder uses concepts at early training, but the AR denoising reconstruction objective cannot sustain concept quality over a full epoch:

- **Positive signals:** at step 4000, concepts carry real information (Δshuffle 1.50), effective rank (14.64) is above the collapsed 5–10 history, and STS-B (0.556) is better than collapsed runs. Generation is qualitatively coherent (CE 4.68 << random 10.82).
- **Negative signals:** effective rank (14.64 at best) is still less than half the 32/128 gate; STS-B (0.556) misses the 0.62 gate; over training the geometry collapses to 4.64 and eval CE rises — classic overfitting under reconstruction with word-dropout.
- **Root cause hypothesis:** AR reconstruction with 40% deletion + 20% word-dropout leaves enough left context for the decoder to operate without the bottleneck after the initial gradient warmup. The "plumbing" works, but the task pressure is insufficient to force sustained concept diversity. This is precisely why E02 (prefix→suffix) removes the decoder's access to source tokens.

Compared fairly: the prior best (`perceiver_denoise`, STS-B 0.607) was a 20-epoch run on MiniPile; E01 is a 1-epoch run on FineWeb-Edu with a harder AR objective. The lower STS-B is not surprising given the different objective and shorter training, but the rank trajectory is concerning regardless.

---

## Decision

**Verdict: mixed** — AR plumbing is confirmed and concepts are briefly used (meets E01 success #1 and #4 at checkpoint-4000), but de-collapse (#2) and semantics (#3) gates fail. The reconstruction objective is not a viable stand-alone semantic objective over full training. E02 (prefix→suffix) with STS-B 0.702 is the decisive follow-up.

Immediate next actions:
1. Record E02 as the stronger semantic objective result.
2. Proceed to E03 (frozen-encoder anchor de-collapse), which addresses the rank collapse that affects both E01 and E02.
3. Keep checkpoint-4000 for concept-ablation analysis and as the E01 plumbing reference.

*Related: `master_experiment_log.md`, `docs/experiments/E01_concept_ar_decoder.md`, `agenda.md`*
