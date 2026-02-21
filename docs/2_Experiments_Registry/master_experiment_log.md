# Master Experiment Log

This is the central registry for all training runs and ablations. It is intended to be the single source of truth for tracking experiments, metrics, and key takeaways for future publications.

## Training Runs

| Date | Run ID / Model Type | Architecture | Pretraining Data | Machine | Epochs | Concept Losses | MLM Loss | Eff. Rank | Key GLUE Scores | Speed (Steps/s) | WandB Link | Conclusion / Takeaway |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-01-17 | `weighted_mlm_H512L2C128_20260117_153544` | H512 L2 C128 | Minipile (1x) | Odra | 20 | None | 4.089 | -- | **MRPC:** 82.2% <br> **QQP:** 61.5% | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/weighted_mlm_H512L6C128_20260207_174251) | Best F1 on MRPC at L2. ModernBERT tokenizer. |
| 2026-01-18 | `perceiver_mlm_H512L2C128_20260118_172328` | H512 L2 C128 | Minipile (1x) | Odra | 20 | None | 4.010 | -- | **MRPC:** 80.6% <br> **QQP:** 67.3% | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-perceiver-mlm-h512l2c128-20260118-172328-36M-20260119_2026) | Canonical L2 baseline. Sparse MLM decoding. |
| 2026-01-19 | `perceiver_posonly_mlm_H512L2C128_20260119_204015` | H512 L2 C128 | Minipile (1x) | Polonez | 20 | None | 4.089 | -- | **MRPC:** 81.8% <br> **QQP:** 69.2% | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-perceiver-posonly-mlm-h512l2c128-20260119-204015-36M-20260204_1943) | Position-only queries. |
| 2026-02-07 | `weighted_mlm_H512L6C128_20260207_174251` | H512 **L6** C128 | Minipile (1x) | Polonez | **40** | None | 3.415 | -- | **MRPC:** 80.2% <br> **QQP:** 66.3% | 2.14 | [Link](https://wandb.ai/ksopyla/MrCogito/runs/weighted_mlm_H512L6C128_20260207_174251) | L6 scaling. Worse MLM loss, decent on inference. |
| 2026-02-08 | `perceiver_posonly_mlm_H512L6C128_20260208_102656` | H512 **L6** C128 | Minipile (1x) | Polonez | **40** | None | 2.640 | -- | **MRPC:** 81.0% <br> **QQP:** 72.3% | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_posonly_mlm_H512L6C128_20260208_102656) | L6 scaling. |
| 2026-02-08 | `perceiver_mlm_H512L6C128_20260208_211633` | H512 **L6** C128 | Minipile (1x) | Polonez | **40** | None | 2.537 | **5 / 128** (4%) | **MRPC:** 81.3% <br> **QQP:** 72.5% <br> **MNLI-m:** 59.1% <br> **STS-B:** 0.627 | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_mlm_H512L6C128_20260208_211633) | **Best L6 canonical model.** Wins 6/8 GLUE tasks. However, concept analysis reveals **severe dimensional collapse**. |
| 2026-02-19 | `perceiver_mlm_H512L6C128_20260219_105435` | H512 L6 C128 | Minipile (1x) | Polonez | 20 | `combined` + `kendall_gal` | **4.31** | **122 / 128** (95.5%) | **MRPC:** 81.4% <br> **QQP:** 58.7% <br> **MNLI-m:** 48.9% <br> **STS-B:** 0.341 | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-perceiver-mlm-h512l6c128-20260219-105435-61M-20260219_2027) | **Collapse fixed, but GLUE crashed.** Kendall-Gal weighting muted the MLM loss. Diverse concepts but no semantic content. |
| 2026-02-21 | `perceiver_mlm_H512L6C128_20260220_184029` | H512 L6 C128 | Minipile (1x) | Polonez | 20 | `combined` + `fixed=0.1` | 3.57 | **15.9 / 128** (12.5%) | *Not evaluated* | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_mlm_H512L6C128_20260220_184029) | **Failed to fix collapse.** The 0.1 weight hurts MLM loss but still doesn't prevent collapse. Abandon `combined` loss for `t_regs_mst` or Diffusion. |

## Quick Links to Detailed Reports
- [L2 vs L6 Scaling Analysis](../3_Evaluations_and_Baselines/comparative_studies/l2_vs_l6_scaling.md)
- [Baseline Models on GLUE](../3_Evaluations_and_Baselines/canonical_baselines.md)
- [Concept Losses (Kendall-Gal vs Fixed)](run_reports/concept_losses_20260219.md)
