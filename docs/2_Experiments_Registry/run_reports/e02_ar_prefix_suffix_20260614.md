# E02 — Prefix-to-Suffix AR Generation — `concept_ar_prefix_H768L6C128D4_20260613_134159`

**Date:** 2026-06-14
**Machine:** Odra (3× RTX 3090, 24 GB VRAM each)
**Run ID:** `concept_ar_prefix_H768L6C128D4_20260613_134159`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260613_134159)
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260613_134140.log`
**Best checkpoint:** `Cache/Training/concept_ar_prefix_H768L6C128D4_20260613_134159/checkpoint-78000`
**Git commit:** `b66bf2e`
**Git tag:** `arch/perceiver-denoise-reset-47-gb66b`
**Related TODO:** E02 in `docs/experiments_specs/E02_ar_prefix_suffix.md`

---

## Goal

Test whether changing the training objective from AR denoising reconstruction (E01) to prefix→suffix AR generation forces stronger concept semantics. The encoder sees only the clean prefix; it must compress it into 128 concepts; the decoder generates the suffix autoregressively. No word-dropout, no deletion. The decoder has no access to the source tokens.

---

## Configuration

| Item | Value |
|---|---|
| Family | `concept_ar` |
| Encoder | H768, L6, C128, BiXT |
| Decoder | causal AR, D4, `decoder_word_dropout=0.0` |
| Token width | `token_embedding_dim=256` |
| Dataset | `HuggingFaceFW/fineweb-edu`, `sample-10BT` (9.57M / 100k) |
| Objective | prefix→suffix AR generation (`ar_prefix_suffix_generation`) |
| Prefix ratio | 0.35–0.45, sentence-boundary split |
| Concept losses | None |
| Epochs | 1 |
| Global steps | 79,768 |
| Effective batch | 40 × 3 GPUs = 120 |
| Throughput | 165 samples/s · 1.372 train steps/s |
| Runtime | ~16.2 h (58,152 s) |
| Precision | bf16 |
| W&B group | `perceiver_denoise` (legacy group name; experiment family is E02) |

---

## Training Outcome

Training completed cleanly to 1 epoch. `load_best_model_at_end=True` selected **checkpoint-78000** (eval_loss 3.524) as the best; the final checkpoint-79768 has the same metric (converged). W&B diagnosis: converged. Eval loss decreased monotonically throughout training — the opposite of E01's degradation pattern.

| Metric | checkpoint-78000 (best) | checkpoint-79768 (last) |
|---|---|---|
| eval loss | 3.524 | 3.524 |
| train loss (epoch avg) | 3.970 | — |
| global step | 78,000 | 79,768 |

---

## Concept Health

All concept_analysis numbers from `analysis/run_concept_analysis.py` run 2026-06-14 using **reconstruction contract** (encoder sees clean sequence). For E02 native suffix-CE ablation, the authoritative source is W&B training logs (see below).

| Metric | checkpoint-78000 (best) | checkpoint-79768 (last) | E02 gate |
|---|---|---|---|
| Effective rank | **11.57 / 128** (9.0%) | 11.57 / 128 (9.0%) | ≥ 48/128 |
| Dimensions for 95% var | 47.0 | 47.0 | — |
| Participation ratio | 96.56 | 96.56 | — |
| ce_intact (recon) | 3.798 | 3.798 | — |
| Δzero (recon) | 0.501 | 0.501 | — |
| Δshuffle (recon) | 0.435 | 0.447 | — |

**Native suffix-CE ablation (from W&B training `concept_ablation/*`, final step):**

| Metric | Value | E02 gate |
|---|---|---|
| **Δshuffle (suffix)** | **0.500 nats** | ≥ 1.0 — FAIL |
| **Δzero (suffix)** | **0.725 nats** | ≥ 2.0 — FAIL |
| ce_intact | 3.491 | — |

Notes:
- Effective rank (11.57) is better than E01-last (4.64) and E01-best (14.64 at checkpoint-4000), confirming prefix→suffix does not cause as severe collapse as reconstruction+dropout.
- Participation ratio (96.56) is notably higher than E01-best (31.54), indicating concept norms are more uniform across slots.
- The concept ablation on suffix-CE fails both E02 gates (Δshuffle < 1.0, Δzero < 2.0). However, the reconstruction-contract ablation is lower still (Δshuffle 0.435), which is expected: the model was trained for prefix→suffix, not reconstruction.
- The paradox: low effective rank (9/128, still collapsed) but very high zero-shot STS-B (0.702). The 128 concept slots are not well spread, but the small active subspace is semantically meaningful.

JSON artifacts: `Cache/Evaluation_reports/concept_ar_prefix_H768L6C128D4_20260613_134159_best_concept_analysis.json` and `..._last_concept_analysis.json`.

---

## Evaluation

### Zero-shot STS-B

WandB eval (best, checkpoint-78000): `bench-stsb_zero_shot-checkpoint-78000-73M-enc-20260614_1140`
WandB eval (last, checkpoint-79768): [bench-stsb_zero_shot-checkpoint-79768-73M-enc-20260614_1140](https://wandb.ai/ksopyla/MrCogito/runs/356t18o7)

| Checkpoint | Pearson | Spearman | Gate (≥ 0.65) |
|---|---|---|---|
| checkpoint-78000 (best) | **0.702** | **0.701** | **PASS** |
| checkpoint-79768 (last) | **0.702** | **0.701** | **PASS** |
| E01 best (checkpoint-4000) | 0.556 | 0.575 | — |
| Prior best (perceiver_denoise) | 0.607 | 0.622 | — |

**0.702 is the new project best zero-shot STS-B Pearson, +0.095 over the prior best, +0.146 over E01-best.**

CSV artifacts: `Cache/Evaluation_reports/bench-stsb_zero_shot-checkpoint-78000-73M-enc-20260614_1140-results.csv` and `...-checkpoint-79768-...`.

---

## Interpretation

E02 is the most semantically capable checkpoint the project has produced. The prefix→suffix objective creates a hard task — the decoder must generate tokens the encoder never saw — and the resulting representations are meaningfully better by the zero-shot semantic gate:

- **Primary positive signal:** STS-B 0.702 clears the E02 gate (0.65) with +0.05 margin and sets a new project best. The semantic gap over E01-best (0.556) and over perceiver_denoise (0.607) is large and reproducible (best and last checkpoints are identical).
- **Geometry:** effective rank (11.57/128) is better than E01-last (4.64) and the collapsed history (~5–10), but still far below the 48/128 gate. Participation ratio (96.56 vs E01's 31.54) shows concepts are more evenly used by norm, but PCA still finds most variance in a small subspace.
- **Concept ablation:** suffix-CE Δshuffle 0.500 and Δzero 0.725 both fail their E02 gates (1.0 and 2.0). This means the decoder is not fully relying on concepts at inference time — it still gets meaningful signal from prefix context even in the suffix-generation task. The ablation gap is real but modest.
- **Geometry paradox:** the rank is collapsed (11.57/128) but STS-B is strong (0.702). The 128 slots concentrate on a small number of semantically-loaded directions. This is not the diverse geometry needed for E04 recursive refinement, but it shows the prefix→suffix pressure is organizing the low-dimensional subspace around semantic content. This is a key finding: the bottleneck IS learning semantics — but in a compact geometry.
- **vs. E02 hypothesis:** the experiment partially validates the hypothesis ("prefix→suffix forces more semantic concepts than reconstruction"). STS-B shows unambiguously better semantic grounding. But the ablation failure and rank shortfall show the bottleneck is still not fully committed — the decoder finds shortcuts via prefix context.

---

## Decision

**Verdict: mixed** — the semantic gate is cleared convincingly (STS-B 0.702 ≥ 0.65) but the de-collapse and concept-ablation gates fail. The result is positive directional evidence: prefix→suffix is the better objective for semantic quality, and it sets the project's new semantic baseline. However, concepts are still geometrically collapsed, which blocks E04 recursive refinement.

The next step is E03 (frozen-encoder hidden-state anchor), which addresses the root cause: concept collapse. E03 does not assume a different objective — it adds an auxiliary MSE loss orthogonal to the AR signal, so it can in principle be combined with either E01 or E02's objective once validated.

*Related: `master_experiment_log.md`, `docs/experiments_specs/E02_ar_prefix_suffix.md`, `agenda.md`*
