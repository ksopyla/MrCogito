# Baseline models evaluation on GLUE

## GLUE MRPC evaluation

Evaluation of baseline encoder models on GLUE MRPC task using [evaluate_model_on_glue.py](../../training/evaluate_model_on_glue.py).

### Results for MRPC

|Date Time| Model Name | F1 Score | Accuracy | Eval Runtime | Wandb Run |
|---------|-------|----------|----------|--------------|--------------|
|2025-07-07 22:29:53| **deberta-base (139M)** | **90.8%** | **87.5%** | ~23.9s | [i22gt7am](https://wandb.ai/ksopyla/MrCogito/runs/i22gt7am) |
|2025-07-07 21:57:00| **roberta-base (125M)** | **90.2%** | **86.5%** | ~17.9s | [taf1duya](https://wandb.ai/ksopyla/MrCogito/runs/taf1duya) |
|2025-07-07 21:01:00| **xlnet-base-cased (117M)** | **88.3%** | **84.8%** | ~21.7s | [nn7az5wf](https://wandb.ai/ksopyla/MrCogito/runs/nn7az5wf) |
|2025-07-07 21:15:12| **bert-base-cased (108M)** | **86.4%** | **81.6%** | ~19.7s | [d9nqaaan](https://wandb.ai/ksopyla/MrCogito/runs/d9nqaaan) |
|2025-07-06 15:03:00| **distilbert-base-cased (66M)** | **83.5%** | **78.7%** | ~18.1s | [lmwzvmpm](https://wandb.ai/ksopyla/MrCogito/runs/lmwzvmpm) |
| | modernbert-base (149M) | *In Progress* | *In Progress* | - | [run-20250629_152554](../wandb/run-20250629_152554-37641cew/) |

### Summary

- **Best Performance**: RoBERTa-base (90.2% F1) - **NEW LEADER!** 🏆
- **Efficiency Champion**: DistilBERT-base-cased (18.1s eval time)
- **Balanced Performance**: XLNet-base-cased (88.3% F1, solid runner-up)
- **Reliable Baseline**: BERT-base-cased (86.4% F1)


### Reserch log updates for MRPC

**2025-07-07**: RoBERTa-base evaluation completed - **NEW PERFORMANCE LEADER!** 🚀
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