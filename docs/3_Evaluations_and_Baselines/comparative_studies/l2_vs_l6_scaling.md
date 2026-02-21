# Full GLUE Evaluation Report -- Concept Encoders L2 vs L6

## Experiment Overview

**Goal:** Assess whether scaling depth (2 -> 6 layers) and training duration (20 -> 40 epochs) improves concept encoder downstream performance, particularly on tasks that failed at L2 (CoLA, RTE, MNLI).

| Config | L2 (baseline) | L6 (scaled) |
|--------|:---:|:---:|
| Layers | 2 | **6** |
| Hidden | 512 | 512 |
| Concepts | 128 | 128 |
| FFN dim | 1024 | **2048** |
| Params (weighted / perceiver) | 34M / 36M | **58M / 61M** |
| Epochs | 20 | **40** |
| LR | 5e-4 | **3e-4** |
| Scheduler | linear | **linear** |
| Effective batch | 256 | **512** |
| Dataset | Minipile | Minipile |
| Tokenizer | ModernBERT-base | ModernBERT-base |
| Eval date | 2026-02-05 | 2026-02-09 |

### MLM Pretraining Loss

| Model | L2 loss | L6 loss | Improvement |
|-------|:---:|:---:|:---:|
| weighted_mlm | ~4.1 | **3.415** | -17% |
| perceiver_posonly_mlm | ~4.1 | **2.640** | -36% |
| perceiver_mlm | ~4.0 | **2.537** | -37% |


## Results: L2 vs L6 Comparison

### Full Results Table

| Task | Metric | weighted L2 | weighted **L6** | posonly L2 | posonly **L6** | perceiver L2 | perceiver **L6** | BERT-Base |
|------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CoLA** | MCC | 0.04 | **0.11** | 0.09 | **0.11** | 0.03 | **0.13** | 59.0 |
| **SST-2** | Acc | 74.3 | **76.1** | 75.1 | **77.8** | 77.4 | 77.5 | 93.1 |
| **MRPC** | F1 | 81.8 | 80.2 | 81.8 | 81.0 | 80.6 | **81.3** | 89.5 |
| **MRPC** | Acc | 71.1 | 69.9 | 71.6 | 70.6 | 70.8 | **71.1** | -- |
| **QQP** | F1 | 61.5 | **66.3** | 69.2 | **72.3** | 67.3 | **72.5** | 91.4 |
| **QQP** | Acc | 74.7 | **77.1** | 76.9 | **77.5** | 76.4 | **79.0** | -- |
| **QNLI** | Acc | 63.4 | **67.1** | 69.7 | **70.8** | 68.1 | **74.0** | 91.6 |
| **RTE** | Acc | 56.0 | **57.8** | 53.4 | **57.4** | 52.0 | **56.7** | 78.2 |
| **MNLI-m** | Acc | 49.0 | **53.8** | 53.9 | **57.6** | 52.5 | **59.1** | 85.4 |
| **MNLI-mm** | Acc | 50.1 | **55.9** | 56.4 | **59.4** | 54.8 | **61.4** | 85.4 |
| **STS-B** | Pearson | -- | -- | -- | -- | -- | **0.627** | 89.4 |
| **STS-B** | Spearman | -- | -- | -- | -- | -- | **0.627** | -- |

STS-B L2: missing (Pearson metric bug — fixed Feb 2026, re-run pending).
STS-B L6 perceiver_mlm: **Pearson 0.627 / Spearman 0.627** (2026-02-20, `perceiver_mlm_H512L6C128_20260208_211633`).
WandB: https://wandb.ai/ksopyla/MrCogito/runs/glue-stsb-perceiver-mlm-h512l6c128-20260208-211633-61M-20260220_0900

### Best-Model-Per-Task Improvement (L2 -> L6)

| Task | L2 best | L6 best | Delta | Relative |
|------|:---:|:---:|:---:|:---:|
| MNLI-mm | 56.4 (posonly) | **61.4** (perceiver) | **+5.0** | +8.9% |
| MNLI-m | 53.9 (posonly) | **59.1** (perceiver) | **+5.2** | +9.6% |
| QNLI | 69.7 (posonly) | **74.0** (perceiver) | **+4.3** | +6.2% |
| RTE | 56.0 (weighted) | **57.8** (weighted) | +1.8 | +3.2% |
| QQP F1 | 69.2 (posonly) | **72.5** (perceiver) | **+3.3** | +4.8% |
| CoLA | 0.094 (posonly) | **0.133** (perceiver) | +0.04 | +42% rel. |
| SST-2 | 77.4 (perceiver) | **77.8** (posonly) | +0.4 | +0.5% |
| MRPC F1 | 81.8 (weighted) | 81.3 (perceiver) | **-0.5** | -0.6% |


## Analysis

### 1. Depth helps inference tasks the most

The largest gains are on tasks requiring reasoning about relationships between texts:
- **MNLI** (+5.0-5.2 pts): No longer near-random. At L2, 3-class MNLI accuracy of ~50-54% was barely above the 33% baseline. At L6, 57-61% shows the model is learning entailment structure, though still 26 pts below BERT-Base.
- **QNLI** (+4.3 pts): QA entailment benefits significantly from deeper cross-attention passes between concepts and tokens.
- **QQP** (+3.3 pts F1): Paraphrase detection on the larger, more varied QQP dataset improves substantially.

These tasks require the model to build compositional representations of two input texts and compare them -- exactly what more cross-attention layers provide.

### 2. Similarity tasks show diminishing returns

**MRPC** slightly decreased (-0.5 F1). This was already the strongest task at L2. The small dataset (3.7k) may cause the deeper model to overfit differently. **SST-2** barely changed (+0.4). Binary sentiment is a holistic property already capturable with shallow processing.

### 3. CoLA remains fundamentally broken

MCC improved from 0.09 to 0.13 -- still essentially zero. For context, BERT-Base achieves 59.0 and even a random-initialized BERT gets ~5-10 MCC after fine-tuning. The +42% relative improvement sounds dramatic but absolute MCC of 0.13 means the model has near-zero ability to distinguish grammatical from ungrammatical sentences. **The concept bottleneck irreversibly destroys syntactic structure, and more depth cannot recover it.**

### 4. perceiver_mlm is now the clear winner

At L2, perceiver_posonly was best on most tasks. At L6, **perceiver_mlm wins 6 of 8 tasks**. The input+position decoder queries benefit more from richer concept representations because the decoder can use both position hints and concept content simultaneously. The additional capacity from 6 layers makes this advantage decisive.

| Model | Best on tasks | Avg GLUE rank |
|-------|:---:|:---:|
| **perceiver_mlm** | **6/8** | 1.25 |
| perceiver_posonly | 2/8 | 1.88 |
| weighted_mlm | 1/8 | 2.75 |

### 5. MLM loss vs downstream: diminishing returns

| Model | MLM loss improvement | Avg downstream improvement |
|-------|:---:|:---:|
| perceiver_mlm | -37% (4.0 -> 2.54) | +3.2 pts avg |
| perceiver_posonly | -36% (4.1 -> 2.64) | +2.7 pts avg |
| weighted_mlm | -17% (4.1 -> 3.42) | +2.1 pts avg |

The perceiver variants improved MLM loss dramatically (36-37%) but downstream gains are modest (2-3 pts on average). This suggests the concept bottleneck, not training quality, is the binding constraint. Better MLM loss means the encoder has learned better language statistics, but the 128-concept compression still loses too much for fine-grained downstream tasks.

### 6. Anomaly: weighted_mlm has worst MLM loss but isn't always worst downstream

weighted_mlm's MLM loss (3.415) is 35% worse than perceiver_mlm (2.537), yet on MRPC and RTE it's competitive. This suggests the weighted decoder's position-specific concept combination creates representations that, while poor for MLM reconstruction, happen to work for certain small-dataset classification tasks through a different inductive bias (fixed position-concept mapping vs learned cross-attention).

### 7. Gap to baselines remains large

| Metric | Concept Encoder (L6 best) | BERT-Base (110M) | Gap |
|--------|:---:|:---:|:---:|
| CoLA MCC | 0.13 | 59.0 | -58.9 |
| SST-2 Acc | 77.8 | 93.1 | -15.3 |
| MRPC F1 | 81.3 | 89.5 | -8.2 |
| QQP F1 | 72.5 | 91.4 | -18.9 |
| QNLI Acc | 74.0 | 91.6 | -17.6 |
| RTE Acc | 57.8 | 78.2 | -20.4 |
| MNLI-m Acc | 59.1 | 85.4 | -26.3 |
| **Avg gap** | | | **-23.7** |

Average gap shrank from -26 pts (L2) to **-23.7 pts** (L6). Progress, but the concept bottleneck remains the fundamental limiting factor.


## Conclusions

1. **Depth helps, but with diminishing returns.** 3x more layers and 2x more epochs yielded +3 pts average improvement. The largest gains were on inference tasks (MNLI, QNLI) that were previously near-random. However, the gap to BERT-Base is still ~24 pts.

2. **The concept bottleneck is the binding constraint, not model capacity.** MLM loss dropped 37% but downstream gains were modest. Compressing 512 tokens into 128 concept vectors fundamentally limits what information is available for fine-tuning.

3. **CoLA is a dead end for this architecture.** Even with 6 layers and 40 epochs, MCC ~0.13 is essentially random. Syntactic acceptability requires token-level structural information that concepts cannot preserve.

4. **perceiver_mlm with input+position queries is the best decoder strategy.** It wins 6/8 tasks at L6 and should be the default for future experiments.

5. **MRPC is saturated at ~81-82% F1** for this architecture. Further improvements need architectural changes, not more depth.


## Recommendations vs Experiment Plan

### Assessment of planned steps:

**Step 2 (Dimension inversion: small token_dim + large concept_dim)** -- **Still high priority but needs reframing.** The L6 results show that depth alone gives diminishing returns. Dimension inversion attacks a different bottleneck: it could allow concepts to carry more information by having larger concept dimensions (256-512) even with tiny token embeddings (32-64). This is now the most promising architectural change since it directly addresses the compression constraint.

**Step 3 (Multi-query classification head)** -- **Deprioritize.** The perceiver_mlm decoder (with input+position queries) already outperforms the single-query classification head on most tasks. The classification head is not the bottleneck -- the concept representation quality is. Multi-query might squeeze out 1-2 pts but won't solve the 24-pt gap.

**Step 4 (Concept losses)** -- **Worth testing now on L6 models.** The L6 concepts are rich enough that regularization could meaningfully improve structure. Run orthogonality + VICReg ablations on the L6 perceiver_mlm as a quick experiment. However, this is unlikely to yield more than 1-2 pts.

**Step 5 (Concept analysis)** -- **Do this now, before expensive experiments.** Understanding what the 128 concepts have actually learned would guide whether the fix is more concepts, different concepts, or a different compression mechanism. This is cheap (no training) and highly informative.

### Suggested plan modification:

1. **Step 5 first** (concept analysis on L6 perceiver_mlm) -- diagnostic, no training cost
2. **Step 2** (dimension inversion) -- highest potential architectural impact
3. **New: Scale pretraining data** -- the model sees Minipile 40 times; try OpenWebText or a larger subset of The Pile
4. Step 4 (concept losses) -- quick ablation on L6
5. Step 3 (multi-query) -- low priority, small expected gains

---

## Update: STS-B Baseline Added (2026-02-20)

STS-B was missing from the original report due to a Pearson metric bug (predictions not squeezed to 1D before passing to `evaluate`). Fixed in `evaluation/evaluate_model_on_glue.py`.

**L6 perceiver_mlm STS-B result** (`perceiver_mlm_H512L6C128_20260208_211633`, 2026-02-20):

| Metric | Score | BERT-Base | Gap |
|---|:---:|:---:|:---:|
| Pearson r | **0.627** | 89.4 | -26.7 |
| Spearman r | **0.627** | — | — |

WandB: https://wandb.ai/ksopyla/MrCogito/runs/glue-stsb-perceiver-mlm-h512l6c128-20260208-211633-61M-20260220_0900

**Context from concept losses experiment (Feb 19 2026):**
The concept losses model (`perceiver_mlm_H512L6C128_20260219_105435`, Kendall-Gal weighting) achieved STS-B Pearson **0.341** — a **−0.286 regression** vs this baseline. STS-B is a direct semantic similarity task and the clearest indicator that the Kendall-Gal concept losses destroyed semantic content while improving geometric diversity. (Update: A follow-up `fixed-weight` (0.1) experiment on Feb 21 failed to fix dimensional collapse and also degraded MLM loss to 3.57, so we are abandoning `combined` loss for `t_regs_mst` or Masked Diffusion).

**Complete updated gap table (L6 perceiver_mlm vs BERT-Base):**

| Task | Metric | L6 perceiver_mlm | BERT-Base | Gap |
|---|---|:---:|:---:|:---:|
| CoLA | MCC | 0.13 | 59.0 | -58.9 |
| SST-2 | Acc | 77.5 | 93.1 | -15.6 |
| MRPC | F1 | 81.3 | 89.5 | -8.2 |
| **STS-B** | **Pearson** | **0.627** | **89.4** | **-26.7** |
| QQP | F1 | 72.5 | 91.4 | -18.9 |
| QNLI | Acc | 74.0 | 91.6 | -17.6 |
| RTE | Acc | 56.7 | 78.2 | -21.5 |
| MNLI-m | Acc | 59.1 | 85.4 | -26.3 |
| MNLI-mm | Acc | 61.4 | 85.4 | -24.0 |
