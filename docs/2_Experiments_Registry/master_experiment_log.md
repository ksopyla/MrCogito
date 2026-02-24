# Master Experiment Log

This is the central registry for all training runs and ablations. It is intended to be the single source of truth for tracking experiments, metrics, and key takeaways for future publications.

## Training Runs

| Date | Run ID / Model Type | Architecture | Pretraining Data | Machine | Epochs | Concept Losses | Task Loss | Eff. Rank | Key GLUE Scores | Speed (Steps/s) | WandB Link | Git Tag | Conclusion / Takeaway |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-01-17 | `weighted_mlm_H512L2C128_20260117_153544` | H512 L2 C128 | Minipile (1x) | Odra | 20 | None | MLM 4.089 | -- | **MRPC:** 82.2% <br> **QQP:** 61.5% | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/weighted_mlm_H512L6C128_20260207_174251) | — | Best F1 on MRPC at L2. ModernBERT tokenizer. |
| 2026-01-18 | `perceiver_mlm_H512L2C128_20260118_172328` | H512 L2 C128 | Minipile (1x) | Odra | 20 | None | MLM 4.010 | -- | **MRPC:** 80.6% <br> **QQP:** 67.3% | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-perceiver-mlm-h512l2c128-20260118-172328-36M-20260119_2026) | — | Canonical L2 baseline. Sparse MLM decoding. |
| 2026-01-19 | `perceiver_posonly_mlm_H512L2C128_20260119_204015` | H512 L2 C128 | Minipile (1x) | Polonez | 20 | None | MLM 4.089 | -- | **MRPC:** 81.8% <br> **QQP:** 69.2% | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-perceiver-posonly-mlm-h512l2c128-20260119-204015-36M-20260204_1943) | — | Position-only queries. |
| 2026-02-07 | `weighted_mlm_H512L6C128_20260207_174251` | H512 **L6** C128 | Minipile (1x) | Polonez | **40** | None | MLM 3.415 | -- | **MRPC:** 80.2% <br> **QQP:** 66.3% | 2.14 | [Link](https://wandb.ai/ksopyla/MrCogito/runs/weighted_mlm_H512L6C128_20260207_174251) | — | L6 scaling. Worse MLM loss, decent on inference. |
| 2026-02-08 | `perceiver_posonly_mlm_H512L6C128_20260208_102656` | H512 **L6** C128 | Minipile (1x) | Polonez | **40** | None | MLM 2.640 | -- | **MRPC:** 81.0% <br> **QQP:** 72.3% | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_posonly_mlm_H512L6C128_20260208_102656) | — | L6 scaling. |
| 2026-02-08 | `perceiver_mlm_H512L6C128_20260208_211633` | H512 **L6** C128 | Minipile (1x) | Polonez | **40** | None | MLM 2.537 | **5 / 128** (4%) | **MRPC:** 81.3% <br> **QQP:** 72.5% <br> **MNLI-m:** 59.1% <br> **STS-B:** 0.627 | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_mlm_H512L6C128_20260208_211633) | `54ee870` | **Best L6 canonical model.** Wins 6/8 GLUE tasks. Severe concept collapse. |
| 2026-02-19 | `perceiver_mlm_H512L6C128_20260219_105435` | H512 L6 C128 | Minipile (1x) | Polonez | 20 | `combined` + `kendall_gal` | MLM **4.31** | **122 / 128** (95.5%) | **MRPC:** 81.4% <br> **QQP:** 58.7% <br> **MNLI-m:** 48.9% <br> **STS-B:** 0.341 | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-perceiver-mlm-h512l6c128-20260219-105435-61M-20260219_2027) | — | **Collapse fixed, but GLUE crashed.** Kendall-Gal muted MLM loss. |
| 2026-02-21 | `perceiver_mlm_H512L6C128_20260220_184029` | H512 L6 C128 | Minipile (1x) | Polonez | 20 | `combined` + `fixed=0.1` | MLM 3.57 | **15.9 / 128** (12.5%) | **MRPC:** 80.7% <br> **QQP:** 64.9% <br> **MNLI-m:** 56.9% <br> **STS-B:** 0.507 <br> **PAWS:** 57.6% | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_mlm_H512L6C128_20260220_184029) | — | **Failed to fix collapse.** Abandon `combined` loss. |
| 2026-02-21 | `diffusion_H512L2C128D2_20260221_195554` | H512 **L2** C128 D2 | Minipile (1x) | Polonez | 20 | None | Diffusion CE (**0.009** at best → diverged to **5.0**) | Not evaluated (diverged) | Not evaluated | 0.81 step/s | [Link](https://wandb.ai/ksopyla/MrCogito/runs/diffusion_H512L2C128D2_20260221_195554) | `7768576` | **FAILED: gradient explosion at epoch 12.** Root causes: (1) O(N²) self-attention in decoder — architecture violated O(C·N) goal; (2) unbounded AdaLN scale caused grad_norm→947 once model memorised dataset; (3) linear LR schedule too slow to decay (LR still 2e-4 when eval_loss=0.009); (4) full lm_head over all positions (6.6× wasted compute). Architecture completely redesigned. See CHANGELOG `[2026-02-23]`. |
| 2026-02-23 | `diffusion_H512L2C128D2_20260223_203349` | H512 **L2** C128 D2 (xattn-only) | Minipile (1x) | Polonez | **20** | None | Diffusion CE train **2.894** / eval **1.433** | **TBD** | **TBD** | 0.515 step/s | [Link](https://wandb.ai/ksopyla/MrCogito/runs/diffusion_H512L2C128D2_20260223_203349) | `arch/diffusion-xattn-only-20260223` | **COMPLETED ✅ — No gradient explosion.** Grad norm peaked at 6.81, ended at 0.23. Train loss 14.19→2.89, eval loss 3.77→1.43. Plateau after epoch 10 — L2 capacity limit. AdaLN-Zero + cosine LR + sparse lm_head all confirmed stable. **Evaluate next:** concept eff. rank, STS-B, MRPC, PAWS. Checkpoint: `Cache/Training/diffusion_H512L2C128D2_20260223_203349/`. |

## Evaluation Experiments (Zero Training Cost)

| Date | Eval Type | Source Checkpoint | Machine | Tasks | Key Scores | WandB | Conclusion |
|---|---|---|---|---|---|---|---|
| 2026-02-22 | `perceiver_decoder_cls` (ViaDecoder) | `perceiver_mlm_H512L6C128_20260208_211633` (from HF Hub) | Odra | mrpc, stsb, qqp, mnli-m, mnli-mm | **MRPC F1:** 82.73% <br> **STS-B P:** 0.650 <br> **QQP F1:** 73.35% <br> **MNLI-m:** 59.75% <br> **MNLI-mm:** 60.90% (ep2) | [WandB](https://wandb.ai/ksopyla/MrCogito) | **ViaDecoder > CLS-Query on all F1/Pearson metrics (+0.65–2.3%).** Consistent improvement confirms classification head was secondary bottleneck. Primary problem remains: concept collapse (rank 5/128). Sets new GLUE baselines. |

**Updated baselines (ViaDecoder, L6 canonical, 2026-02-22):**

| Task | CLS-Query (old) | ViaDecoder (new) | Delta |
|---|---|---|---|
| MRPC F1 | 81.3% | **82.73%** | +1.4% |
| STS-B Pearson | 0.627 | **0.650** | +2.3% |
| QQP F1 | 72.5% | **73.35%** | +0.85% |
| MNLI-m Acc | 59.1% | **59.75%** | +0.65% |
| MNLI-mm Acc | 59.34% | **60.90%** (+ep3 pend.) | +1.56% |

**Full analysis:** [via_decoder_eval_20260222.md](run_reports/via_decoder_eval_20260222.md)

---

## Architecture Overhaul (2026-02-23) — Diffusion Decoder Redesign

**Decision:** Completely rewrite the diffusion decoder. Remove O(N²) self-attention, replace with Perceiver IO-style cross-attention-only decoding, adopt AdaLN-Zero timestep conditioning.

**Root cause of failure (`diffusion_H512L2C128D2_20260221_195554`):**

1. **Architecture contradiction:** Decoder used full token self-attention (O(N²)) — the exact computational pattern the concept bottleneck is designed to replace. Meaningless for long sequences.
2. **Unbounded AdaLN:** Multiplicative conditioning `x * (1 + scale)` with no initialization constraint. When eval_loss → 0.009 (memorization), the remaining LR (2e-4) overshoots the minimum, scale amplifies exponentially → grad_norm → 947.
3. **Linear LR schedule:** At epoch 12, LR was still 2e-4 (40% of 5e-4). Cosine would give 3e-5. The sharp loss landscape post-memorization combined with high LR = guaranteed explosion.
4. **Full logits waste:** `lm_head` applied to all 512 positions, only M masked kept. ~6.6× wasted matmul compute.
5. **Padding positions masked:** `_apply_noise` did not respect `attention_mask`.

**New architecture (`arch/diffusion-xattn-only-20260223`):**

- `DiffusionDecoderLayer`: cross-attention only (O(N·C)), AdaLN-Zero (zero-initialized gates)
- `ConceptDiffusionDecoder`: returns hidden states (no lm_head inside decoder)
- `ConceptEncoderForMaskedDiffusion`: sparse `lm_head` at model level, `label_smoothing=0.1`, padding-safe noise, `t_min=0.1`
- `train_diffusion_multigpu.sh`: LR 3e-4, cosine schedule, grad_accum=2, label_smoothing=0.1

**Full analysis in CHANGELOG:** `[2026-02-23]`

---

## Architecture Overhaul (2026-02-21)

**Decision:** Abandon MLM as primary training objective. Switch to TSDAE (denoising reconstruction) with PosOnly decoder.

**Root cause analysis:** 5 structural misalignments identified in MLM+Perceiver pipeline:
1. [MASK] token pollution in encoder cross-attention (MAE-LM, ICLR 2024)
2. Uncontextualized token embeddings across all encoder layers (static KV)
3. Decoder input-embedding shortcut killing gradient flow (85% positions have no concept gradient)
4. Single CLS query collapsing 128 concepts into 1 weighted mixture
5. GLUE concatenated pair encoding mismatched with single-span pretraining
   
More details in [mlm_perceiver_diagnosis_20260221.md](../4_Research_Notes/mlm_perceiver_diagnosis_20260221.md)

**New architecture implemented:**
- `BiConceptEncoderLayer`: BiXT-style bidirectional cross-attention, O(C*N) preserved
- `DataCollatorForTSDAE`: token deletion (60%), dense labels at all positions
- `ConceptEncoderForMaskedLMPerceiverPosOnly`: dense CE loss (all non-pad positions)
- `ConceptEncoderForSentencePairClassification`: separate encoding, weighted concept pooling
- `ConceptEncoderForSequenceClassificationPerceiver`: weighted concept pooling (replaces CLS query)

**Full analysis:** [mlm_perceiver_diagnosis_20260221.md](../4_Research_Notes/mlm_perceiver_diagnosis_20260221.md)

## Quick Links to Detailed Reports
- [L2 vs L6 Scaling Analysis](../3_Evaluations_and_Baselines/comparative_studies/l2_vs_l6_scaling.md)
- [Baseline Models on GLUE](../3_Evaluations_and_Baselines/canonical_baselines.md)
- [Concept Losses (Kendall-Gal vs Fixed)](run_reports/concept_losses_20260219.md)
- [MLM+Perceiver Deep Diagnosis (Feb 21)](../4_Research_Notes/mlm_perceiver_diagnosis_20260221.md)
- [Diffusion L2 Failure Analysis (Feb 21)](run_reports/diffusion_L2_failure_20260221.md)
