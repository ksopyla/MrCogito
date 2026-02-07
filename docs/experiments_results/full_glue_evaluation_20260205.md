# Full GLUE Evaluation Report -- Concept Encoders (2026-02-05)

## Experiment Setup

Three concept encoder checkpoints evaluated across 8 GLUE tasks (STS-B failed, MRPC perceiver variants from earlier runs). All models: H512, L2, C128, Minipile pretraining, ModernBERT tokenizer. Fine-tuning: LR 1e-5, batch 96, 20 epochs (small tasks) / 3-5 epochs (large tasks).

## Results

| Task | Type | Metric | weighted (34M) | posonly (36M) | perceiver (36M) | BERT-Base (110M) | ModernBERT (149M) |
|------|------|--------|:-:|:-:|:-:|:-:|:-:|
| **CoLA** | Linguistic acceptability | MCC | 0.04 | **0.09** | 0.03 | 59.0 | 65.1 |
| **SST-2** | Sentiment | Acc | 74.3 | 75.1 | **77.4** | 93.1 | 96.0 |
| **MRPC** | Paraphrase (F1/Acc) | F1 | **81.8**/71.1 | 81.8/71.6* | 80.6/70.8* | 89.5 | 92.2 |
| **QQP** | Question paraphrase (F1/Acc) | F1 | 61.5/74.7 | **69.2**/76.9 | 67.3/76.4 | 91.4 | 92.1 |
| **QNLI** | QA entailment | Acc | 63.4 | **69.7** | 68.1 | 91.6 | 93.9 |
| **RTE** | Textual entailment | Acc | **56.0** | 53.4 | 52.0 | 78.2 | 87.4 |
| **MNLI-m** | NLI (3-class) | Acc | 49.0 | **53.9** | 52.5 | 85.4 | 89.1 |
| **MNLI-mm** | NLI (3-class) | Acc | 50.1 | **56.4** | 54.8 | 85.4 | 89.1 |
| **STS-B** | Similarity (regression) | -- | -- | -- | -- | 89.4 | 91.8 |

\* MRPC perceiver variants from earlier evaluation runs (20260119, 20260204), not 20260205.

Bold = best concept encoder per task. BERT-Base and ModernBERT-Base from published papers (dev set).

## Extended Baselines (Published Dev Set Results)

| Task | ALBERT-Base (12M) | BERT-Base (110M) | RoBERTa-Base (125M) | DeBERTaV3-Base (183M) | ModernBERT-Base (149M) |
|------|:-:|:-:|:-:|:-:|:-:|
| CoLA | 55.8 | 59.0 | 63.6 | 69.2 | 65.1 |
| SST-2 | 92.8 | 93.1 | 94.8 | 95.6 | 96.0 |
| MRPC | 89.1 | 89.5 | 90.2 | 89.5 | 92.2 |
| QQP | 90.4 | 91.4 | 91.9 | 92.4 | 92.1 |
| QNLI | 90.1 | 91.6 | 92.8 | 94.0 | 93.9 |
| RTE | 72.2 | 78.2 | 78.7 | 83.8 | 87.4 |
| MNLI | 82.7 | 85.4 | 87.6 | 90.0 | 89.1 |
| STS-B | 89.8 | 89.4 | 91.2 | 91.6 | 91.8 |

## Analysis

### Performance by Task Category

**Best: Similarity/Paraphrase tasks (MRPC, QQP).**
MRPC F1 of 80-82% reaches ~91% of BERT-Base performance and ~89% of ModernBERT. The concept bottleneck preserves enough semantic overlap information for coarse paraphrase detection. QQP is weaker (61-69% F1 vs 91%+) because its larger and more varied dataset exposes the bottleneck's limits.

**Moderate: Single-sentence classification (SST-2).**
Accuracy of 74-77% vs 93-96% baseline. The encoder captures overall sentiment polarity but misses nuanced expressions. The ~19pt gap suggests concepts retain broad meaning but lose discriminative details.

**Poor: Inference/entailment (RTE, MNLI, QNLI).**
MNLI accuracy of 49-56% is barely above the 33% random baseline for 3-class classification. RTE at 52-56% barely beats the 50% coin flip. QNLI at 63-70% is the best inference task but still 22-30pt below baselines. These tasks require reasoning about logical relationships between premise and hypothesis -- information the concept bottleneck destroys.

**Failing: Linguistic acceptability (CoLA).**
Matthews correlation of 0.03-0.09 is essentially zero. CoLA requires fine-grained syntactic judgment. The concept bottleneck completely eliminates grammatical structure information.

### Model Comparison

| Rank | Best on tasks | Model | Strength |
|------|:---:|---|---|
| 1 | 6/8 | **perceiver_posonly** | Best on CoLA, QQP, QNLI, MNLI-m, MNLI-mm, MRPC(tied) |
| 2 | 2/8 | **perceiver_mlm** | Best on SST-2, competitive on QQP/QNLI |
| 3 | 2/8 | **weighted_mlm** | Best on RTE, MRPC; worst on inference tasks |

**Surprising finding**: perceiver_posonly (position-only queries, no input token hints) is the strongest across most tasks. This contradicts the MRPC-only evaluation where weighted_mlm led. The full GLUE picture reveals that the pure Perceiver IO approach, which forces concepts to encode all information without decoder shortcuts, generalizes better.

The weighted decoder wins only on MRPC and RTE (both small datasets), suggesting it overfits more easily to small fine-tuning sets.

### Gap to SoTA Baselines

| Gap metric | Concept Encoders (best) | vs BERT-Base | vs ALBERT-Base (12M) |
|---|:-:|:-:|:-:|
| Avg gap across tasks | -- | **-26 pt** | **-22 pt** |
| Best task (MRPC F1) | 81.8% | -7.7 pt | -7.3 pt |
| Worst task (CoLA MCC) | 0.09 | -58.9 pt | -55.7 pt |

Even ALBERT-Base with only 12M parameters (vs our 34-36M) outperforms concept encoders by 20+ points on average. The gap is not explained by parameter count.

## Conclusions

1. **The concept bottleneck architecture works for coarse semantic similarity** (MRPC) but fails catastrophically on tasks requiring syntactic knowledge (CoLA), logical reasoning (MNLI, RTE), or fine-grained discrimination (QNLI).

2. **The information bottleneck is too aggressive.** Compressing 512 tokens through 128 concept vectors of dim 512, with only 2 encoder layers, destroys too much task-relevant information. The architecture needs either more layers, more concepts, or a less lossy compression mechanism.

3. **Pretraining scale is insufficient.** Minipile (~1.5M samples) is orders of magnitude smaller than BERT (3.3B words) or ModernBERT (2T tokens). The concepts likely haven't learned rich enough representations.

4. **The perceiver_posonly variant is the most promising.** Forcing the decoder to reconstruct entirely from concepts (no input hints) leads to better concept representations that transfer more broadly.

5. **The approach is at proof-of-concept stage.** The MRPC results show the concept idea has merit for semantic tasks. The immediate priorities for improvement are: (a) scaling to 4+ encoder layers, (b) pretraining on larger data, (c) testing concept regularization losses, and (d) the dimension inversion experiment (small token embeddings + large concept dims).
