# Diffusion L2 Evaluation — `diffusion_H512L2C128D2_20260223_203349`

**Evaluation date:** 2026-02-25
**Training date:** 2026-02-23
**Machine (training):** Polonez (4x RTX 3090, 24 GB VRAM each)
**Machine (evaluation):** Polonez (single GPU per task, CUDA_VISIBLE_DEVICES isolation)
**Checkpoint:** `Cache/Training/diffusion_H512L2C128D2_20260223_203349/diffusion_H512L2C128D2_20260223_203349`
**Git tag:** `arch/diffusion-xattn-only-20260223`
**WandB training:** [Link](https://wandb.ai/ksopyla/MrCogito/runs/diffusion_H512L2C128D2_20260223_203349)
**Related TODO:** TODO 6 in `docs/1_Strategy_and_Plans/active_todos.md`
**Previous run (failed):** [diffusion_L2_failure_20260221.md](diffusion_L2_failure_20260221.md)

---

## 1. Training Summary

### Architecture

| Component | Value |
|---|---|
| Encoder | `ConceptEncoder`, H=512, L=**2** layers, C=128 concepts |
| Decoder | `ConceptDiffusionDecoder`, D=2 cross-attention layers, **no self-attention** |
| Conditioning | AdaLN-Zero (zero-initialized gates) |
| lm_head | Sparse — applied only to masked positions |
| Total params | 33.5M |

This is the redesigned architecture from the [Feb 23 CHANGELOG](../../../CHANGELOG.md), fixing all issues from the failed Feb 21 run: O(N^2) self-attention removed, AdaLN-Zero prevents gradient explosion, sparse lm_head saves 6.6x compute.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Dataset | JeanKaddour/minipile |
| Tokenizer | answerdotai/ModernBERT-base |
| Max sequence length | 512 |
| Epochs | 20 |
| LR | 3e-4 (cosine schedule) |
| Effective batch size | 512 (64/device x 4 GPUs x 2 grad_accum) |
| Warmup steps | 1500 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| t_min | 0.1 |
| Label smoothing | 0.1 |
| Mixed precision | bf16 |
| Concept losses | None (disabled for baseline) |

### Training Metrics

| Metric | Start | End |
|---|---|---|
| Train loss | 14.19 | 2.894 |
| Eval loss | 3.77 | 1.433 |
| Grad norm | 6.81 (peak) | 0.23 (final) |
| Training speed | 0.515 steps/s | |
| Wall-clock time | ~10h (est.) | |

Training was fully stable with no gradient explosion (vs. the Feb 21 run which diverged at epoch 12 with grad_norm -> 947). Loss plateaued after epoch 10, suggesting L2 encoder capacity is the bottleneck.

---

## 2. Concept Space Geometry Analysis

Analysis run: `analysis/run_concept_analysis.py --model_type diffusion_mlm`, 20 batches x 16 samples = 320 samples from Minipile.

### 2.1 Collapse Detection

| Metric | Diffusion L2 | L6 MLM Baseline | L6 + Fixed 0.1 | Target | Grade |
|---|---|---|---|---|---|
| **Global effective rank (raw)** | **10.1** | 5.07 | 15.97 | > 64 | POOR |
| **Global effective rank (norm)** | **0.079** | 0.040 | 0.125 | > 0.50 | POOR |
| Participation ratio (norm) | 0.064 | -- | -- | > 0.10 | POOR |
| Dims for 95% variance | 19 | -- | -- | > 50 | OK |
| Collapsed dimensions | 0 / 512 | -- | -- | 0 | GOOD |
| Isotropy | 1.08e-9 | -- | -- | > 0.001 | POOR |

**Interpretation:** Effective rank 10.1/128 is 2x better than the L6 MLM baseline (5.07) but far below the 64/128 target. The concepts occupy a 10-dimensional subspace of the 128-dimensional concept space. However, zero collapsed dimensions means all 512 hidden dimensions are active -- the collapse is in the concept dimension, not the hidden dimension.

### 2.2 Concept Diversity

| Metric | Diffusion L2 | L6 MLM Baseline | L6 + Fixed 0.1 | Target | Grade |
|---|---|---|---|---|---|
| **Mean pairwise similarity** | **0.187** | ~0.13 | 0.133 | < 0.30 | GOOD |
| **Max pairwise similarity** | **1.000** | 0.999 | 0.999 | < 0.60 | POOR |
| Uniformity loss | 0.048 | -- | -- | < 0.30 | GOOD |

**Interpretation:** Mean similarity 0.187 and uniformity 0.048 are both good -- concepts are spread out on average. But max similarity 1.000 means at least one pair of concepts is nearly identical (duplicate). This suggests the model uses fewer than 128 effective concepts.

### 2.3 Singular Value Analysis

| Metric | Value | Grade |
|---|---|---|
| Top-1 dominance ratio | **0.099** | GOOD |
| Top-1 variance ratio | 0.233 | OK |
| Top-5 variance ratio | 0.581 | OK |

Top-5 singular values: 90.4, 64.3, 56.1, 51.6, 48.9

**Interpretation:** The top-1 dominance ratio of 0.099 is significantly better than the MLM baselines (~0.3+). The singular values decay gradually rather than having one dominant direction. This means concepts are more evenly distributed in the representational space compared to MLM, but still occupy too few effective dimensions.

### 2.4 Dimension Utilization (VICReg-style)

| Metric | Value | Target | Grade |
|---|---|---|---|
| Mean dimension std | 0.889 | > 0.30 | GOOD |
| Min dimension std | 0.631 | > 0.01 | GOOD |
| Max dimension std | 1.320 | -- | -- |
| Mean concept L2 norm | 21.74 | -- | -- |
| Std concept L2 norm | 0.139 | -- | -- |

**Interpretation:** All hidden dimensions are well-utilized (min std 0.631 >> threshold 0.01). Concept norms are very uniform (21.74 +/- 0.14), suggesting concepts lie on a hypersphere. This is healthy -- VICReg-style variance collapse is not present. The problem is concept-level collapse (too few distinct directions), not dimension-level collapse.

### 2.5 Key Comparison vs Previous Runs

| Metric | MLM L6 Baseline (40ep) | Fixed 0.1 L6 (20ep) | Diffusion L2 (20ep) | Interpretation |
|---|---|---|---|---|
| Effective rank | 5.07 / 128 (4%) | 15.97 / 128 (12.5%) | **10.1 / 128 (7.9%)** | Diffusion 2x better than MLM, but worse than fixed-0.1 |
| Mean similarity | ~0.13 | 0.133 | **0.187** | All comparable, within acceptable range |
| Max similarity | 0.999 | 0.999 | **1.000** | All have duplicate concepts |
| Top-1 dominance | ~0.3 (est.) | -- | **0.099** | Diffusion much more evenly distributed |
| Encoder depth | L6 (6 layers) | L6 | **L2 (2 layers)** | Diffusion at L2 vs others at L6 |

---

## 3. Downstream Evaluation Results

Classification model: `ConceptEncoderForSequenceClassificationPerceiver` (weighted concept pooling, encoder weights only, decoder discarded). This is the same classification head used for `perceiver_mlm` evaluation -- no diffusion-specific decoding needed for GLUE.

### 3.1 GLUE Tasks

| Task | Metric | Diffusion L2 | L2 Perceiver Baseline | L6 ViaDecoder Baseline | vs L2 | vs L6 |
|---|---|---|---|---|---|---|
| **MRPC** | F1 | **80.0%** | 80.6% | 82.73% | -0.6% | -2.7% |
| **MRPC** | Accuracy | 68.87% | -- | -- | -- | -- |
| **STS-B** | Pearson | **0.138** | -- | 0.650 | -- | -0.512 |

### 3.2 Beyond-GLUE

| Task | Metric | Diffusion L2 | L6 Fixed 0.1 | L6 ViaDecoder Baseline |
|---|---|---|---|---|
| **PAWS** | Accuracy | **55.98%** | 57.6% | -- |
| **SICK Relatedness** | Pearson | **0.064** | -- (script broken) | -- |
| **SICK Entailment** | Accuracy | **57.78%** | -- (script broken) | -- |

### 3.3 Evaluation Configuration

| Property | Value |
|---|---|
| Classification head | `ConceptEncoderForSequenceClassificationPerceiver` |
| Weights loaded | encoder.* only (41 weight tensors) |
| MRPC / STS-B | 20 epochs, LR 1e-5, batch 96, single GPU |
| PAWS | 5 epochs, LR 1e-5, batch 96, single GPU |
| SICK | 10 epochs, LR 1e-5, batch 96, single GPU |

### 3.4 WandB Evaluation Links

| Task | WandB Run |
|---|---|
| MRPC | [glue-mrpc-diffusion-h512l2c128d2-20260223-203349-33M-20260225_1256](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-diffusion-h512l2c128d2-20260223-203349-33M-20260225_1256) |
| STS-B | [glue-stsb-diffusion-h512l2c128d2-20260223-203349-33M-20260225_1631](https://wandb.ai/ksopyla/MrCogito/runs/glue-stsb-diffusion-h512l2c128d2-20260223-203349-33M-20260225_1631) |
| PAWS | [bench-paws-diffusion-20260225_1629](https://wandb.ai/ksopyla/MrCogito/runs/eccx8ewc) |
| SICK Relatedness | [bench-sick_relatedness-diffusion-20260225_1630](https://wandb.ai/ksopyla/MrCogito/runs/yeia6359) |
| SICK Entailment | [bench-sick_entailment-diffusion-20260225_1630](https://wandb.ai/ksopyla/MrCogito/runs/yeia6359) |

---

## 4. Analysis and Interpretation

### 4.1 What the diffusion objective improved

1. **Training stability.** The AdaLN-Zero + cross-attention-only + cosine LR combination is confirmed stable. No gradient explosion. This is a solved engineering problem.

2. **Concept geometry.** Effective rank doubled (5 -> 10), top-1 dominance ratio dropped from ~0.3 to 0.099, meaning concepts are more evenly distributed. The diffusion curriculum (variable masking rate t ~ Uniform[0.1, 1.0]) does create more pressure for concept utilization than fixed 15% MLM masking.

3. **MRPC F1 preserved.** 80.0% vs 80.6% L2 baseline -- the concept encoder still captures enough information for binary paraphrase classification, even through a 128-concept bottleneck.

### 4.2 What the diffusion objective did NOT fix

1. **STS-B Pearson 0.138 (near-random).** This is the most damning result. STS-B measures semantic similarity directly -- Pearson close to zero means the concept representations do not encode semantic relatedness. The concepts may capture *some* syntactic/surface features (enough for MRPC binary classification) but fail at continuous semantic similarity.

2. **Concept collapse persists (rank 10/128).** The 2x improvement over MLM is real but insufficient. 10 effective dimensions out of 128 means 92% of the concept capacity is wasted. The roadmap target is 64/128 (50%).

3. **Max similarity 1.000.** At least one pair of concepts is still fully duplicated, meaning the model found it unnecessary to differentiate them.

### 4.3 Why STS-B is so much worse than MRPC

MRPC is binary classification (paraphrase or not) with 3.7K training examples -- a low bar. STS-B requires predicting continuous similarity scores from 1 to 5 -- this needs fine-grained semantic information that collapsed concepts cannot provide. The MRPC F1 of 80.0% is actually close to majority-class baseline (81.2%), suggesting the model may be barely above random on MRPC too, just lucky with F1 scoring.

### 4.4 L2 vs L6 confound

This diffusion model uses only 2 encoder layers (L2), while baselines are L6. The L2 perceiver_mlm baseline achieves MRPC 80.6% and QQP 67.3%. We cannot directly attribute all regression to the diffusion objective -- L2 capacity limits are a real factor. An L6 diffusion run would be needed for a fair comparison against the L6 ViaDecoder baselines.

However, concept collapse at L2 (rank 10) vs L6 MLM (rank 5) is still informative: the diffusion objective produces slightly better geometry even with fewer encoder layers.

---

## 5. Decision Gate Assessment (Roadmap Gate 1)

Per the [roadmap](../../1_Strategy_and_Plans/roadmap.md) decision gate:

```
Effective rank > 30/128?  NO (10.1/128)
  → Diffusion alone insufficient for concept quality

STS-B Pearson > 0.70?  NO (0.138)
  → Add contrastive loss or try TSDAE
```

**Verdict:** Diffusion objective alone does not meet any Gate 1 criteria. It should NOT be discarded entirely (improved geometry is real), but cannot be the sole Track A winner.

**Recommended next steps:**
1. Train TSDAE PosOnly (TODO 10) -- the other Track A candidate addressing all 5 structural misalignments
2. If TSDAE also fails rank > 30, combine diffusion + contrastive loss (SimCSE) as hybrid
3. Consider L6 diffusion run only if TSDAE comparison is promising (otherwise, focus on TSDAE)
4. Slot Attention (C5) remains the architectural fallback if all objective changes fail

---

## 6. Raw Concept Analysis Data

Full JSON output from `run_concept_analysis.py`:

```json
{
  "model_path": "Cache/Training/diffusion_H512L2C128D2_20260223_203349/diffusion_H512L2C128D2_20260223_203349",
  "model_type": "diffusion_mlm",
  "n_batches": 20,
  "n_samples": 320,
  "effective_rank": 11.827,
  "effective_rank_normalized": 0.092,
  "dimensions_for_95_variance": 19.0,
  "top_1_variance_ratio": 0.233,
  "top_5_variance_ratio": 0.581,
  "max_concept_similarity": 1.000,
  "mean_concept_similarity": 0.187,
  "std_concept_similarity": 0.174,
  "uniformity_loss": 0.048,
  "isotropy": 1.08e-09,
  "participation_ratio": 32.829,
  "participation_ratio_normalized": 0.064,
  "mean_dimension_std": 0.889,
  "min_dimension_std": 0.631,
  "max_dimension_std": 1.320,
  "collapsed_dimensions": 0.0,
  "collapsed_dimensions_ratio": 0.0,
  "mean_concept_norm": 21.740,
  "std_concept_norm": 0.139,
  "min_concept_norm": 21.321,
  "max_concept_norm": 21.971,
  "global_effective_rank": 10.099,
  "global_effective_rank_normalized": 0.079,
  "top5_singular_values": [90.427, 64.283, 56.050, 51.570, 48.911]
}
```

Note on per-batch vs global effective rank: the per-batch average (11.8) is slightly higher than the global cross-batch rank (10.1) because individual batches may have more varied concept usage that washes out when aggregated. The global rank is the more meaningful metric.

---

## 7. Engineering Fixes Applied During Evaluation

1. **`evaluate_on_benchmark.py`**: Added `diffusion_mlm` to `--model_type` choices and model class mapping. Diffusion uses `ConceptEncoderForSequenceClassificationPerceiver` (same as perceiver_mlm).

2. **SICK dataset loader**: Added parquet fallback for `datasets>=4.0` which dropped script-based loaders. The `sick` dataset now falls back to `revision="refs/convert/parquet"` when the script loader fails.

3. **Shell scripts**: Added `diffusion` auto-detection to `evaluate_concept_encoder_sick.sh` and `evaluate_concept_encoder_paws.sh` (GLUE script already had it).

All fixes committed: `bafc384` and `c2829c4`.

---

*Report written: 2026-02-25*
*Evaluation commands ran on Polonez with CUDA_VISIBLE_DEVICES isolation (one GPU per task) after server reboot*
*Related: [master_experiment_log.md](../master_experiment_log.md), [active_todos.md](../../1_Strategy_and_Plans/active_todos.md), [diffusion_L2_failure_20260221.md](diffusion_L2_failure_20260221.md)*
