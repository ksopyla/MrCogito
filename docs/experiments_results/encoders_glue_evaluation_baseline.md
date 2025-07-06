



# Baseline models evaluation on GLUE



## 2025-06-29 15:00 GLUE MRPC evaluation

Evaluation of the baseline models on the GLUE benchmark. With use of the [evaluate_model_on_glue.py](../training/evaluate_model_on_glue.py) script.
Date: 2025-06-29 15:00

Wandb report: https://wandb.ai/ksopyla/MrCogito/reports/Baseline-Glue-MRPC-evaluation--VmlldzoxMzM5MjAyOA 


| Model | F1 Score (MRPC) | Accuracy (MRPC) | Eval Runtime |
|-------|----------------|-----------------|--------------|
| xlnet-base-cased | 88.3% | 84.8% | ~21.4s |
| bert-base-cased | 86.4% | 81.6% | ~29.7s |
| distilbert-base-cased | 83.5% | 78.7% | ~18.1s |