# Baseline models evaluation on GLUE

## GLUE MRPC evaluation

Evaluation of baseline encoder models on GLUE MRPC task using [evaluate_model_on_glue.py](../../training/evaluate_model_on_glue.py).

### Results for MRPC

|Date Time| Model Name | F1 Score | Accuracy | Eval Runtime | Wandb Run |
|---------|-------|----------|----------|--------------|--------------|
|2025-07-07 22:29:53| **deberta-base (139M)** | **90.8%** | **87.5%** | ~23.9s | [i22gt7am](https://wandb.ai/ksopyla/MrCogito/runs/i22gt7am) |
|2025-08-05 09:11:00| **albert-base-v2 (12M)** | **90.6%** | **87.0%** | ~19.2s | [no2yyx5n](https://wandb.ai/ksopyla/MrCogito/runs/no2yyx5n) |
|2025-07-07 21:57:00| **roberta-base (125M)** | **90.2%** | **86.5%** | ~17.9s | [taf1duya](https://wandb.ai/ksopyla/MrCogito/runs/taf1duya) |
|2025-07-07 21:01:00| **xlnet-base-cased (117M)** | **88.3%** | **84.8%** | ~21.7s | [nn7az5wf](https://wandb.ai/ksopyla/MrCogito/runs/nn7az5wf) |
|2025-07-07 21:15:12| **bert-base-cased (108M)** | **86.4%** | **81.6%** | ~19.7s | [d9nqaaan](https://wandb.ai/ksopyla/MrCogito/runs/d9nqaaan) |
|2025-07-06 15:03:00| **distilbert-base-cased (66M)** | **83.5%** | **78.7%** | ~18.1s | [lmwzvmpm](https://wandb.ai/ksopyla/MrCogito/runs/lmwzvmpm) |
| | modernbert-base (149M) | *In Progress* | *In Progress* | - | [run-20250629_152554](../wandb/run-20250629_152554-37641cew/) |

### Summary

- **Best Performance**: DeBERTa-base (90.8% F1, 87.5% Acc) - **NEW LEADER!** 🏆
- **Parameter Efficiency Champion**: ALBERT-base-v2 (90.6% F1 with only 12M params) - **Most Efficient!** ⚡
- **Strong Performance**: RoBERTa-base (90.2% F1, 86.5% Acc) 
- **Balanced Choice**: XLNet-base-cased (88.3% F1, solid and reliable)
- **Speed Champion**: DistilBERT-base-cased (18.1s eval time, good for resource constraints)
- **Classic Baseline**: BERT-base-cased (86.4% F1, standard reference)


### Research log updates for MRPC

**2025-08-05**: ALBERT-base-v2 evaluation completed - **PARAMETER EFFICIENCY BREAKTHROUGH!** 🚀
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

Wandb report: https://wandb.ai/ksopyla/MrCogito/reports/Baseline-Glue-MRPC-evaluation--VmlldzoxMzM5MjAyOA