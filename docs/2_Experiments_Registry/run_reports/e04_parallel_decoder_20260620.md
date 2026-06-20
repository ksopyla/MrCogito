# E04 — Concept-only parallel decoder — `perceiver_denoise_H768L6C128D4_20260618_200645`

**Date:** 2026-06-20 (eval); training 2026-06-18 → 2026-06-19
**Machine:** Odra (3× RTX 3090, 24 GB each)
**Run ID:** `perceiver_denoise_H768L6C128D4_20260618_200645`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_denoise_H768L6C128D4_20260618_200645)
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260618_200626.log`
**Best checkpoint:** `Cache/Training/perceiver_denoise_H768L6C128D4_20260618_200645/checkpoint-66000`
**Git commit:** `224fdc1`
**Git tag:** —
**Related TODO:** E04 in `docs/experiments_specs/E04_concept_only_parallel_decoder.md`

---

## Goal

Test whether removing the **autoregressive decoder bypass** (swap causal AR → position-query
parallel Perceiver-IO decoder, no token self-attention) improves concept geometry on the
reconstruction objective, holding encoder/data/tokenizer fixed. Fair baseline: **E03 anchor-OFF
control** (causal AR reconstruction, 0.3 ep). Reference semantics: **E02** prefix→suffix AR.

---

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_denoise` (`checkpoint_family`) |
| Encoder | H768, L6, C128, BiXT |
| Decoder | **parallel Perceiver-IO** (`perceiver_posonly`), D4, position queries only |
| Token width | `token_embedding_dim=256` |
| Anchor | OFF |
| Dataset | `HuggingFaceFW/fineweb-edu`, `sample-10BT` (9.57M / 100k holdout) |
| Objective | TSDAE denoising reconstruction, `deletion_rate=0.6` |
| Epochs | **1.0** (spec planned 0.3-ep gate → extend; run went full epoch) |
| Global steps | 66,473 |
| Effective batch | 24 × grad_accum 2 × 3 GPUs = 144 |
| LR / sched / seed | 3e-4 / cosine / 42 |
| Throughput | 1.181 steps/s (~15.6 h) |
| Precision | bf16 |

Foundation shipped for E04 (2026-06-18): linear Perceiver-IO decoder (O(C·N) only), EOS/padding
data-contract fix aligned with E03 control, W&B `decoder:parallel` + `task:reconstruction` tags.

---

## Training Outcome

Training completed cleanly to 1.0 epoch. Eval loss fell smoothly from 7.12 (0.03 ep) to **2.747**
(best at step 66000); no NaNs or divergence. `load_best_model_at_end` selected **checkpoint-66000**.
Train loss 3.267. Parallel decoder learns the denoising task despite no token self-attention.

| Epoch | eval_loss (best in window) |
|---|---|
| 0.30 | 3.001 |
| 0.60 | 2.803 |
| 0.99 | **2.747** (ck-66000) |

Note: E03 control best eval CE is 4.792 — different decoder/loss scale; not directly comparable.

---

## Concept Health

From `analysis/run_concept_analysis.py` on **checkpoint-66000** (`--num_batches 20 --batch_size 16`).
JSON: `Cache/Evaluation_reports/perceiver_denoise_H768L6C128D4_20260618_200645_best_concept_analysis.json`.

| Metric | E04 (ck-66000) | E03 control (ck-10000) | E03A anchor (ck-19000) | E02 (ck-78000, 1 ep) |
|---|---|---|---|---|
| Slot eff. rank | **10.78 / 128** | 5.93 / 128 | 10.34 / 128 | 11.57 / 128 |
| **Within-sample RankMe** | **107.8** | — (not logged) | — | — |
| Cross-sample RankMe | **177.8** | 150.4 | 167.1 | — |
| Manifold anisotropy | 0.83 | 0.57 | 0.49 | — |
| Mean concept cosine | 0.209 | 0.436 | 0.240 | 0.213 |
| Dims for 95% var (slot) | 67.0 | 11.2 | 32.9 | — |
| Participation ratio | 136.3 | — | — | — |
| Early-Δzero / Δshuffle | **N/A** (parallel decoder) | 4.10 / 5.58 | 2.32 / 3.34 | — |

Key observations:
- **Within-sample RankMe 107.8** is very high — per-input concepts span many independent directions
  despite modest slot rank (~11). This is the upgraded primary de-collapse signal and was not
  available for the E03 eval runs.
- Cross-sample RankMe **177.8** beats E03 control (+27.4, clears spec +8 gate) and anchor-ON (+10.7).
- Slot rank (~10.8) is similar to E02 1-ep (11.6) and E03A (10.3), much better than collapsed E03
  control (5.9).
- No concept-ablation ΔCE — the parallel decoder has no AR suffix/reconstruction ablation path in
  the current tooling; the spec's early-Δ leg cannot be scored.

---

## Evaluation

### Tier 0 — Health (checkpoint-66000)

PASS — no NaN/Inf, forward pass stable.

### Tier 2 — Zero-shot STS-B (checkpoint-66000)

WandB: [bench-stsb_zero_shot-checkpoint-66000](https://wandb.ai/ksopyla/MrCogito/runs/g8rhx8zj)

| Checkpoint | Pearson | Spearman | vs E03 control gate (≥ control) | vs E02 (ref) |
|---|---|---|---|---|
| E04 ck-66000 (best) | **0.532** | 0.557 | ✓ (control 0.485) | ✗ (E02 0.702) |
| E03 control ck-10000 | 0.485 | 0.533 | — | — |
| E03A anchor ck-19000 | 0.556 | 0.572 | — | — |
| E02 ck-78000 | 0.702 | 0.701 | — | — |
| E02-long ck-296000 | 0.714 | 0.710 | — | — |

CSV: `Cache/Evaluation_reports/bench-stsb_zero_shot-checkpoint-66000-73M-enc-20260620_1533-results.csv`

**Not yet run:** Tier 2.5 attention-pool probe, SICK/PAWS, last-checkpoint eval, 0.3-ep matched
checkpoint for fair E03 budget comparison.

---

## Interpretation

E04 is a **parallel-vs-AR concept-formation A/B** on reconstruction. Against the designed E03
control baseline:

- **RankMe gate (+8):** PASS — cross-sample RankMe 177.8 vs 150.4 (+27.4).
- **STS-B ≥ control:** PASS — 0.532 vs 0.485 (+0.047).
- **Early-Δzero gate:** unscored — parallel decoder has no AR ablation probe.

Removing the AR bypass buys **richer per-sample concept geometry** (within-sample RankMe 108) and
modestly better semantics than the collapsed E03 control, while slot rank stays in the E02/E03A
ballpark. This is **directional support** for the bypass hypothesis, but confounded: (1) decoder
family changes bypass + info channel + prediction target together; (2) E04 ran **1 epoch vs 0.3**
for E03 control; (3) high within-sample RankMe does not translate to strong mean-pool STS-B.

Against **E02** (the semantic reference): E04 is far weaker (STS-B 0.532 vs 0.702). Reconstruction
+ parallel decode is a representation probe, not a semantic objective — prefix→suffix AR remains
the project's best semantic line (E02-long: STS-B 0.714, RankMe 246).

---

## Decision

**Verdict: mixed.** Geometry gates vs E03 control clear on RankMe and STS-B; early-Δ leg missing;
absolute semantics well below E02. The bypass removal correlates with healthier concept structure
but does not close the semantic gap. Next options justified by evidence: (1) Tier-2.5
attention-pool probe on E04 to test whether distributed geometry is hidden from mean-pool STS-B;
(2) prefix→suffix + parallel decoder or anchor (combine levers); (3) E05 windowed decoder now
unblocked per spec gate.

*Related: `master_experiment_log.md`, `docs/experiments_specs/E04_concept_only_parallel_decoder.md`, `agenda.md`*
