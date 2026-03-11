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
| 2026-02-23 | `diffusion_H512L2C128D2_20260223_203349` | H512 **L2** C128 D2 (xattn-only) | Minipile (1x) | Polonez | **20** | None | Diffusion CE train **2.894** / eval **1.433** | **10.1 / 128** (7.9%) | **MRPC:** 80.0% <br> **STS-B:** 0.138 <br> **PAWS:** 55.98% <br> **SICK-R:** 0.064 <br> **SICK-E:** 57.78% | 0.515 step/s | [Link](https://wandb.ai/ksopyla/MrCogito/runs/diffusion_H512L2C128D2_20260223_203349) | `arch/diffusion-xattn-only-20260223` | **COMPLETED ✅ — Evaluated 2026-02-25.** Concept rank 10/128 (2× better than L6 MLM baseline rank 5/128 but still collapsed). MRPC F1 80.0% matches L2 perceiver baseline (80.6%), but STS-B Pearson 0.138 is near-random — diffusion objective alone does NOT fix semantic collapse. Better singular value distribution (top-1 dominance 0.099 vs ~0.3 for MLM) suggests concepts are more evenly distributed but still not semantically grounded. **Decision:** Diffusion alone insufficient for concept quality. Wait for TSDAE comparison before deciding Track A winner. |

|| 2026-02-26 | `diffusion_H512L6C128D2_20260226_155541` | H512 **L6** C128 D2 (xattn-only) | Minipile (1x) | Odra (3x 3090) | **20** | None (ELBO=True, t_min=0.3) | Diffusion CE train **2.837** / eval **1.418** | **5.74 / 128** (4.5%) | **MRPC:** 78.63% <br> **STS-B:** 0.174 <br> **QQP:** 57.18% <br> **MNLI-m:** 44.25% | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/diffusion_H512L6C128D2_20260226_155541) | — | **TODO 11 — FAILED.** L6 depth + ELBO weighting did not fix collapse. Rank 5.74 WORSE than L2's 10.1. STS-B 0.174 < 0.30 gate → self-reconstruction fundamentally insufficient. |
|| 2026-03-01 | `diffusion_H512L6C128D2_20260301_165308` | H512 **L6** C128 D2 (xattn-only) | Minipile (1x) | Polonez (4x 3090) | **20** | `vicreg` + `t_regs_mst` (fixed=0.02, warmup=2000) ELBO=True, t_min=0.3 | Diffusion CE train **2.841** / eval **1.419** | **5.09 / 128** (4.0%) | Not evaluated (collapsed) | -- | [Link](https://wandb.ai/ksopyla/MrCogito/runs/diffusion_H512L6C128D2_20260301_165308) | — | **TODO 11b — FAILED.** VICReg + t_regs_mst had NO effect on collapse. Rank 5.09 ≈ ELBO baseline (5.74). Decision gate: rank < 20 → close diffusion self-reconstruction permanently. |
|| 2026-03-04 | `prefix_diff_H512L6C128D2_20260304_200437` | H512 **L6** C128 D2 (prefix→suffix diffusion) | Minipile (1x) | Polonez (4x 3090) | **20** | None (clean baseline), ELBO=True, t_min=0.3 | Diffusion CE train **14.57** / eval **7.42** (2× logging bug) | **6.19 / 128** (4.8%) | **MRPC:** 81.25% <br> **STS-B:** 0.337 <br> **QQP:** 74.81% <br> **MNLI-m:** 48.19% | 0.78 step/s | [Link](https://wandb.ai/ksopyla/MrCogito/runs/prefix_diff_H512L6C128D2_20260304_200437) | `arch/prefix-diffusion-20260304` | **TODO 13a — FAILED.** SODA-style prefix generation did NOT fix concept collapse. Rank 6.19/128 (4.8%) ≈ all previous experiments. STS-B 0.337 < 0.50 gate. Eval loss 7.42 (below random 10.82 — model learns something, but weakly). Train loss has 2× gradient_accum reporting bug. MRPC F1 decent (81.25%) but STS-B/MNLI confirm concepts lack semantics. **Root cause:** Prefix→suffix generation is too hard for the bottleneck to learn in 20 epochs with this architecture. AdaLN-Zero gates likely stay near-zero, preventing concept signal from reaching decoder. |
|| 2026-03-08 | `prefix_diffBiXT_T64_H512L6C128D2_20260308_065355` | H512 **L6** C128 D2 (prefix→suffix diffusion, **BiXT**, T64, sentence-boundary split) | Minipile (1x) | Polonez (4x 3090) | **20** | None (clean v2 baseline), ELBO=True, t_min=0.3 | Diffusion CE train **14.93** / best eval **7.425** | **5.74 / 128** (4.5%) | Not evaluated (rank gate fail) | 0.75 step/s | [Link](https://wandb.ai/ksopyla/MrCogito/runs/prefix_diffBiXT_T64_H512L6C128D2_20260308_065355) | `arch/prefix-diffusion-20260304-12-g031e` | **TODO 13b — FAILED.** The hardened prefix-diffusion stack fixed the DDP crash and ran stably to 20 epochs, but concept quality remained collapsed: effective rank **5.74**, global effective rank **4.94**, mean concept similarity **0.576**, max similarity **0.999**. This is slightly worse than TODO 13a (rank 6.19), so **BiXT + token_embedding_dim=64 + sentence-boundary splitting did not rescue random-init prefix generation**. Decision gate: rank < 10/128 and no downstream eval justification → stop further random-init clean prefix variants; only revisit the prefix track with warm-start / pretrained-backbone initialization. |

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
- [Prefix Diffusion BiXT v2 Failure (Mar 8)](run_reports/prefix_diffusion_bixt_v2_20260308.md)
- [L2 vs L6 Scaling Analysis](../3_Evaluations_and_Baselines/comparative_studies/l2_vs_l6_scaling.md)
- [Baseline Models on GLUE](../3_Evaluations_and_Baselines/canonical_baselines.md)
- [Concept Losses (Kendall-Gal vs Fixed)](run_reports/concept_losses_20260219.md)
- [MLM+Perceiver Deep Diagnosis (Feb 21)](../4_Research_Notes/mlm_perceiver_diagnosis_20260221.md)
- [Diffusion L2 Failure Analysis (Feb 21)](run_reports/diffusion_L2_failure_20260221.md)
