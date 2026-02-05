# Baseline models evaluation on GLUE

## GLUE MRPC evaluation

Evaluation of baseline encoder models on GLUE MRPC task using [evaluate_model_on_glue.py](../../training/evaluate_model_on_glue.py).

### Concept Encoder Fair Comparison (H512L2C128, Minipile + ModernBERT)

These 3 models form the **canonical fair comparison set**. All trained with:
- Same architecture config: H512, L2, C128, intermediate=1024
- Same dataset: JeanKaddour/minipile
- Same tokenizer: answerdotai/ModernBERT-base
- Same GLUE fine-tuning protocol: 20 epochs, LR 1e-5, batch_size 96
- All evaluated after major architecture bug fixes (LayerNorm, weight loading, classification head naming)

| Model Type | Checkpoint | Params | F1 | Accuracy | Loss | Wandb Run |
|------------|-----------|--------|-----|----------|------|-----------|
| **weighted_mlm** | `weighted_mlm_H512L2C128_20260117_153544` | 34M | **82.2%** | **71.8%** | 0.651 | [20260117_2156](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-weighted-mlm-h512l2c128-20260117-153544-34M-20260117_2156) |
| **perceiver_posonly_mlm** | `perceiver_posonly_mlm_H512L2C128_20260119_204015` | 36M | 81.8% | 71.6% | 0.611 | [20260204_1943](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-perceiver-posonly-mlm-h512l2c128-20260119-204015-36M-20260204_1943) |
| **perceiver_mlm** | `perceiver_mlm_H512L2C128_20260118_172328` | 36M | 80.6% | 70.8% | 0.591 | [20260119_2026](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-perceiver-mlm-h512l2c128-20260118-172328-36M-20260119_2026) |

**Note**: The previous best perceiver_mlm result (83.3% F1) was from checkpoint `20260111_210335` (dense MLM, different training run). This is NOT part of the fair comparison set. The canonical `20260118_172328` checkpoint (sparse MLM) achieves 80.6% F1, ranking third. Despite "identical" MLM training loss, the different training run leads to measurably different downstream GLUE results (~2.7% F1 gap).

Commands used for these evaluations:
```bash
P="/home/ksopyla/dev/MrCogito/Cache/Training"

# weighted_mlm (best F1: 82.2%)
python training/evaluate_model_on_glue.py --model_type weighted_mlm \
  --model_name_or_path "$P/weighted_mlm_H512L2C128_20260117_153544/weighted_mlm_H512L2C128_20260117_153544" \
  --tokenizer_name "$P/weighted_mlm_H512L2C128_20260117_153544/weighted_mlm_H512L2C128_20260117_153544" \
  --task mrpc --batch_size 96 --epochs 20 --learning_rate 1e-5 --visualize --save_model

# perceiver_posonly_mlm (F1: 81.8%)
python training/evaluate_model_on_glue.py --model_type perceiver_posonly_mlm \
  --model_name_or_path "$P/perceiver_posonly_mlm_H512L2C128_20260119_204015/perceiver_posonly_mlm_H512L2C128_20260119_204015" \
  --tokenizer_name "$P/perceiver_posonly_mlm_H512L2C128_20260119_204015/perceiver_posonly_mlm_H512L2C128_20260119_204015" \
  --task mrpc --batch_size 96 --epochs 20 --learning_rate 1e-5 --visualize --save_model

# perceiver_mlm (F1: 80.6%)
python training/evaluate_model_on_glue.py --model_type perceiver_mlm \
  --model_name_or_path "$P/perceiver_mlm_H512L2C128_20260118_172328/perceiver_mlm_H512L2C128_20260118_172328" \
  --tokenizer_name "$P/perceiver_mlm_H512L2C128_20260118_172328/perceiver_mlm_H512L2C128_20260118_172328" \
  --task mrpc --batch_size 96 --epochs 20 --learning_rate 1e-5 --visualize --save_model
```


### Transformer Baselines (MRPC)

Standard pretrained encoder models for reference. Evaluated with default HuggingFace fine-tuning.

| Date Time | Model Name | F1 Score | Accuracy | Eval Runtime | Wandb Run |
|-----------|------------|----------|----------|--------------|-----------|
| 2025-07-07 22:29 | **deberta-base (139M)** | **90.8%** | **87.5%** | ~23.9s | [i22gt7am](https://wandb.ai/ksopyla/MrCogito/runs/i22gt7am) |
| 2025-08-05 09:11 | **albert-base-v2 (12M)** | **90.6%** | **87.0%** | ~19.2s | [no2yyx5n](https://wandb.ai/ksopyla/MrCogito/runs/no2yyx5n) |
| 2025-07-07 21:57 | **roberta-base (125M)** | **90.2%** | **86.5%** | ~17.9s | [taf1duya](https://wandb.ai/ksopyla/MrCogito/runs/taf1duya) |
| 2025-07-07 21:01 | **xlnet-base-cased (117M)** | **84.8%** | **88.3%** | ~21.7s | [nn7az5wf](https://wandb.ai/ksopyla/MrCogito/runs/nn7az5wf) |
| 2025-07-07 21:15 | **bert-base-cased (108M)** | **86.4%** | **81.6%** | ~19.7s | [d9nqaaan](https://wandb.ai/ksopyla/MrCogito/runs/d9nqaaan) |
| 2025-07-06 15:03 | **distilbert-base-cased (66M)** | **83.5%** | **78.7%** | ~18.1s | [lmwzvmpm](https://wandb.ai/ksopyla/MrCogito/runs/lmwzvmpm) |


### Summary

- **Best Transformer**: DeBERTa-base (90.8% F1, 87.5% Acc, 139M params) - **LEADER**
- **Parameter Efficiency Champion**: ALBERT-base-v2 (90.6% F1 with only 12M params)
- **Best Concept Encoder** (fair comparison): `weighted_mlm_H512L2C128` achieved **82.2% F1** and **71.8% Accuracy** (34M params)
- **Concept Encoder Ranking** (fair comparison, same training protocol):
    | Rank | Model Type | F1 | Acc | Decoder Strategy |
    |------|------------|-----|-----|------------------|
    | 1 | weighted_mlm | 82.2% | 71.8% | Learned position-specific concept weights |
    | 2 | perceiver_posonly_mlm | 81.8% | 71.6% | Position-only cross-attention queries |
    | 3 | perceiver_mlm | 80.6% | 70.8% | Input+Position cross-attention queries |
- **Key Observations**:
    - All 3 models are within 1.6% F1 of each other -- decoder strategy matters less than expected
    - The simpler weighted decoder slightly outperforms both perceiver variants
    - All concept encoders show a large F1-Accuracy gap (~10pt) vs baselines (~5pt), suggesting the concept bottleneck biases toward predicting the positive class
    - Previous best perceiver_mlm result (83.3% F1 from older `20260111` checkpoint) is NOT part of this fair comparison -- it was a different training run


---

## Full GLUE Benchmark Results

TODO: Full GLUE evaluation in progress. Run with:
```bash
P="/home/ksopyla/dev/MrCogito/Cache/Training"
for model in \
  "perceiver_mlm_H512L2C128_20260118_172328" \
  "weighted_mlm_H512L2C128_20260117_153544" \
  "perceiver_posonly_mlm_H512L2C128_20260119_204015"; do
    bash scripts/evaluate_concept_encoder_glue.sh "$P/$model/$model" all
done
```

---

## Historical Results (Legacy - Pre-Bugfix / Different Protocol)

These results are from earlier development iterations and are **not directly comparable** to the fair comparison set above due to:
- Different tokenizer (bert-base-cased vs ModernBERT)
- Different dataset (Wikipedia vs Minipile)
- Architecture bugs (LayerNorm missing, weight loading errors, naming conflicts)
- Different model sizes (23M vs 34-36M)

| Date Time | Model Name | F1 Score | Accuracy | Notes | Wandb Run |
|-----------|------------|----------|----------|-------|-----------|
| 2026-01-18 12:01 | perceiver_mlm_H512L2C128 (36M) | 83.3% | 74.0% | Dense MLM checkpoint `20260111`, different training run | [20260118_1200](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-perceiver-mlm-h512l2c128-20260111-210335-36M-20260118_1200) |
| 2026-01-14 18:48 | perceiver_mlm_H512L2C128 (36M) | 76.3% | 65.2% | Initial ModernBERT eval, untuned hyperparams | [20260114_1848](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-perceiver-mlm-h512l2c128-20260111-210335-36M-20260114_1848) |
| 2025-11-30 16:22 | weighted_mlm_H512L2C256 (23M) | 79.9% | 68.6% | C256 variant, old tokenizer | [20251130_1622](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-weighted-mlm-h512l2c256-20251128-133227-23M-20251130_1622) |
| 2025-11-27 14:48 | weighted_mlm_H512L2C128 (23M) | 81.2% | 68.4% | 100 epochs, old tokenizer | [20251127_1448](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-weighted-mlm-h512l2c128-20251123-213949-23M-20251127_1448) |
| 2025-11-23 21:20 | weighted_mlm_H512L2C128 (23M) | 81.2% | 68.4% | 20 epochs, old tokenizer | [20251123_2120](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-weighted-mlm-h512l2c128-20251119-090233-23M-20251123_2120) |
| 2025-11-30 | perceiver_mlm_H512L2C128 (25M) | 74.2% | 64.0% | Architecture fix confirmed | - |
| 2025-11-07 21:29 | weighted_mlm_H256L2C128 (Undertrained) | 0.0% | 31.6% | Failed to converge | - |


### Historical Research Log

<details>
<summary>Click to expand historical research log entries</summary>

**2026-01-19**: Concept Encoder `perceiver_mlm_H512L2C128` evaluation - **Sparse MLM Checkpoint (Fair Comparison)**
- **Experiment ID**: `glue-mrpc-perceiver-mlm-h512l2c128-20260118-172328-36M-20260119_2026`
- **Training Run**: `perceiver_mlm_H512L2C128_20260118_172328` (sparse MLM decoding)
- **F1 Score**: 80.6%
- **Accuracy**: 70.8%
- **Loss**: 0.591
- **Model Size**: 36M parameters (35,581,954)
- **Training Time**: 57.3s fine-tuning
- **Eval Runtime**: ~0.43s
- **Analysis**: This is the canonical perceiver_mlm checkpoint for fair comparison. Scores lower than the previous `20260111` checkpoint (83.3% F1) despite reportedly identical MLM training loss. The ~2.7% F1 gap suggests that sparse MLM decoding during pretraining may subtly affect encoder representation quality, or that different training trajectory/random seed matters for downstream performance.

**2026-02-04**: Concept Encoder `perceiver_posonly_mlm_H512L2C128` evaluation - **Position-Only Queries**
- **Experiment ID**: `glue-mrpc-perceiver-posonly-mlm-h512l2c128-20260119-204015-36M-20260204_1943`
- **Training Run**: `perceiver_posonly_mlm_H512L2C128_20260119_204015`
- **Training Time**: 12h 44m (45,845s) on 4x RTX 3090
- **MLM Training Loss**: 4.089
- **F1 Score**: 81.8%
- **Accuracy**: 71.6%
- **Loss**: 0.611
- **Analysis**: Position-only decoder queries. Slightly lower than perceiver_mlm (83.3% F1). Forces concepts to encode all information without input token hints.

**2026-01-18**: Concept Encoder `perceiver_mlm_H512L2C128` evaluation - **Best Result with Optimized Hyperparameters**
- **Experiment ID**: `glue-mrpc-perceiver-mlm-h512l2c128-20260111-210335-36M-20260118_1200`
- **Training Run**: `perceiver_mlm_H512L2C128_20260111_210335` (trained 2026-01-11)
- **F1 Score**: 83.3% - **BEST CONCEPT ENCODER!**
- **Accuracy**: 74.0%
- **Loss**: 0.591
- **Model Size**: 36M parameters
- **Analysis**: Best concept encoder result to date. Achieved with optimized fine-tuning on MRPC (20 epochs, LR 1e-5). Approaches DistilBERT (83.5% F1) performance.
- **Note**: Sparse MLM retrained version `perceiver_mlm_H512L2C128_20260118_172328` has identical encoder and MLM loss. Used as canonical checkpoint going forward.

**2026-01-17**: Concept Encoder `weighted_mlm_H512L2C128` evaluation - **ModernBERT Tokenizer**
- **Experiment ID**: `glue-mrpc-weighted-mlm-h512l2c128-20260117-153544-34M-20260117_2156`
- **Training Run**: `weighted_mlm_H512L2C128_20260117_153544` (trained 2026-01-17)
- **F1 Score**: 82.2%
- **Accuracy**: 71.8%
- **Loss**: 0.651
- **Model Size**: 34M parameters
- **Analysis**: Weighted attention pooling with ModernBERT tokenizer. Solid performance, slightly below perceiver_mlm best.

**2026-01-14**: Concept Encoder `perceiver_mlm_H512L2C128` evaluation - **Initial ModernBERT + Minipile**
- **Experiment ID**: `glue-mrpc-perceiver-mlm-h512l2c128-20260111-210335-36M-20260114_1848`
- **F1 Score**: 76.3%
- **Accuracy**: 65.2%
- **Loss**: 1.388
- **Training**: 20 epochs on Minipile dataset with ModernBERT-base tokenizer
- **Model Size**: 36M parameters
- **Analysis**: Initial evaluation with default hyperparameters. Later improved significantly with hyperparameter tuning (see 2026-01-18 entry).

**2025-11-30**: Concept Encoder `weighted_mlm_H512L2C256` evaluation.
- **Experiment ID**: `glue-mrpc-weighted-mlm-h512l2c256-20251128-133227-23M-20251130_1622`
- **F1 Score**: 79.9%
- **Accuracy**: 68.6%
- **Analysis**: Increasing concept dimension to 256 (from 128) resulted in similar performance (F1 79.9% vs 81.2%, Acc 68.6% vs 68.4%). The larger concept space didn't immediately yield better generalization on MRPC.

**2025-11-30**: Concept Encoder `perceiver_mlm_H512L2C128` evaluation - **Architecture Fixed**
- **Experiment ID**: `glue-mrpc-perceiver-mlm-h512l2c128-20251129-174003-25M-20251130_1333`
- **Status**: **Stable** (Loss Normal)
- **Loss**: Started high (~15.3), dropped to 2.66. No explosion.
- **F1 Score**: 74.2%
- **Accuracy**: 64.0%
- **Analysis**: Training is now stable with the correct architecture (`ConceptEncoderForSequenceClassificationPerceiver`). The model is learning (loss decreased significantly), but performance is currently below the majority class baseline (81.2% F1). This is typical for early training or unoptimized hyperparameters. The "exploding loss" from the previous attempt is resolved.
- **Next Steps**: Optimize hyperparameters (Learning Rate, Epochs) to surpass the baseline.

**2025-11-27**: Concept Encoder `weighted_mlm_H512L2C128` (100 epochs) evaluation completed.
- **Experiment ID**: `glue-mrpc-weighted-mlm-h512l2c128-20251123-213949-23M-20251127_1448`
- **F1 Score**: 81.2% (Same as 20-epoch model)
- **Accuracy**: 68.4%
- **Insight**: Training for 5x more epochs (100 vs 20) yielded no downstream improvement on MRPC. This indicates the "concept bottleneck" capacity or the pretraining task itself is the limiting factor, not training duration. The model learns its optimal representation quickly. 
- **Architecture Fix**: This run confirmed the fix for the architecture mismatch (Weighted vs Mean pooling).

**2025-11-23**: Concept Encoder `weighted_mlm_H512L2C128` (20 epochs) evaluation completed - **Significant Progress!**
- **Experiment ID**: `glue-mrpc-weighted-mlm-h512l2c128-20251119-090233-23M-20251123_2120`
- **F1 Score**: 81.2% (Massive improvement from 0.0%)
- **Accuracy**: 68.4%
- **Model Size**: 23M parameters (Very small/efficient compared to 100M+ baselines)
- **Eval Runtime**: ~0.49s (Extremely fast inference)
- **Comparison**: Approaching DistilBERT (83.5% F1) performance while being significantly smaller (23M vs 66M params). 
- **Improvement**: Corrected training issues from previous `weighted_mlm_H256L2C128` run which failed to converge (Loss: inf).

**2025-08-05**: ALBERT-base-v2 evaluation completed - **PARAMETER EFFICIENCY BREAKTHROUGH!**
- **Experiment ID**: `no2yyx5n`
- **File Format**: `glue-mrpc-albert-base-v2-12M-20250805_0911-*.csv`
- **F1 Score**: 90.6% (Exceptional performance for 12M parameters!)
- **Accuracy**: 87.0% (Close to DeBERTa despite 91% fewer parameters)
- **Training Time**: 2.91 minutes (174 seconds) - highly efficient
- **Model Size**: Only 12M parameters (vs 139M for DeBERTa)
- **Evaluation Time**: 19.2s (faster than most larger models)
- **Critical Finding**: ALBERT achieves 99.8% of DeBERTa's F1 performance with only 8.6% of its parameters!
- **vs Original Paper**: Our 90.6% F1 slightly **exceeds** the original ALBERT paper's reported ~89.3% F1 on MRPC
- **Parameter Efficiency**: 7.55 F1 points per million parameters (vs 0.65 for DeBERTa)

**2025-07-07**: RoBERTa-base evaluation completed
- **Experiment ID**: `taf1duya`
- **New File Format**: `glue-mrpc-roberta-base-125M-20250707_2200-*.csv`
- **F1 Score**: 90.2% (NEW RECORD - surpasses XLNet!)
- **Accuracy**: 86.5% (excellent performance)
- **Training Time**: 2.78 minutes (167 seconds)
- **Model Size**: 125M parameters
- **Performance Analysis**: RoBERTa demonstrates superior paraphrase detection capabilities, achieving the highest F1 score in our baseline comparison

**2025-07-07**: BERT-base-cased evaluation completed with new naming convention - **Excellent Reproducibility!**
- **Experiment ID**: `d9nqaaan`
- **New File Format**: `glue-mrpc-bert-base-cased-108M-20250707_2115-*.csv`
- **F1 Score**: 86.4% (identical to previous run!)
- **Accuracy**: 81.6% (identical to previous run!)
- **Training Time**: 2.95 minutes (177 seconds)
- **Model Size**: 108M parameters (corrected count)
- **Perfect Consistency**: Results match previous evaluation exactly

**2025-07-07**: XLNet evaluation with new naming convention - **Excellent Reproducibility!**
- **Experiment ID**: `nn7az5wf`
- **New File Format**: `glue-mrpc-xlnet-base-cased-117M-20250707_2106-*.csv`
- **F1 Score**: 88.3% (identical to previous run!)
- **Accuracy**: 84.8% (identical to previous run!)
- **Training Time**: 3.55 minutes (213 seconds)
- **Model Size**: 117M parameters

**2025-07-06**: Updated DistilBERT evaluation with new naming convention and direct WandB link
- **Experiment ID**: `lmwzvmpm`
- **New File Format**: `glue-mrpc-distilbert-base-cased-66M-20250706_1503-*.csv`
- **Training Time**: 2.4 minutes (144 seconds)
- **Model Size**: 66M parameters

</details>

Wandb report: https://wandb.ai/ksopyla/MrCogito/reports/Baseline-Glue-MRPC-evaluation--VmlldzoxMzM5MjAyOA
