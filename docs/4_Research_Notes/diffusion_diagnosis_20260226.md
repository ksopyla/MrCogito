# Diffusion MLM Diagnosis: Root Cause Analysis and Next Steps

**Date:** 2026-02-26
**Author:** Krzysztof Sopyla / AI session analysis
**Status:** Permanent research note
**Related experiments:** `diffusion_H512L2C128D2_20260223_203349` (L2 diffusion), `perceiver_mlm_H512L6C128_20260208_211633` (L6 MLM baseline)
**Related files:** `nn/concept_encoder_diffusion.py`, `nn/concept_encoder.py`, `nn/loss_manager.py`
**Related reports:** [diffusion_L2_eval_20260225.md](../2_Experiments_Registry/run_reports/diffusion_L2_eval_20260225.md), [mlm_perceiver_diagnosis_20260221.md](mlm_perceiver_diagnosis_20260221.md)

---

## Executive Summary

The diffusion L2 run (`diffusion_H512L2C128D2_20260223_203349`) produced concepts that are geometrically better distributed (effective rank 10/128, 2x improvement over MLM baseline) but semantically empty (STS-B Pearson 0.138, near-random). This note traces 5 root causes through architecture analysis, gradient flow, and comparison with SoTA diffusion LMs (LLaDA, MDLM, Mercury), then proposes 4 concrete fixes ranked by expected impact.

---

## 1. Root Cause Analysis (ordered by impact)

### Cause 1: L2 Encoder Too Shallow (DOMINANT)

The L2 encoder has only 2 cross-attention layers operating on **uncontextualized** token embeddings. Token embeddings are raw word vectors + position encoding, computed once before the layer loop and never updated through self-attention. Compositional semantics ("The dog chased the cat" vs "The cat chased the dog") require token-token interactions that 2 layers of concept cross-attention simply cannot capture.

The L6 perceiver_mlm achieves STS-B 0.650 with 6 layers. This 3x depth difference likely accounts for most of the STS-B gap.

**Test:** L6 diffusion ablation (TODO 11) -- same config, 6 encoder layers. If STS-B > 0.50, depth is the bottleneck.

### Cause 2: Missing ELBO Loss Weighting

LLaDA's training loss: `token_loss = F.cross_entropy(...) / p_mask`

MrCogito's training loss: `diffusion_loss = F.cross_entropy(masked_logits, masked_targets)`

Without the 1/t normalization, gradient magnitude varies ~10x across noise levels:
- At t=0.1: ~50 masked positions → small aggregate loss
- At t=0.9: ~460 masked positions → large aggregate loss

The MDLM paper (Sahoo, NeurIPS 2024) derives that the proper ELBO for masked diffusion is a **weighted average of MLM losses** across masking levels, with weight proportional to 1/t. This is mathematically necessary for a tight variational bound.

**Fix:** `weighted_loss = diffusion_loss / t.mean().clamp(min=0.1)` -- 1 line of code.

### Cause 3: t_min=0.1 Wastes Training on Near-MLM Regime

With t ~ Uniform(0.1, 1.0), ~11% of training steps have t<0.2 where the task degenerates to standard 15% MLM. At low masking rates, the decoder's residual stream (from unmasked token embeddings) carries enough information to reconstruct without using concepts. The diagnosis from Feb 21 identified this same pattern in MLM: "at t < 0.15, local context suffices and concepts are unnecessary."

**Fix:** `--t_min 0.3` -- ensures every step puts meaningful pressure on the concept bottleneck.

### Cause 4: Self-Reconstruction Objective Permits Surface-Level Hashing

The encoder sees clean text X and the decoder reconstructs the same text X. With a 128-concept bottleneck at 512 tokens (4:1 compression), the concepts CAN store enough surface-level information (token statistics, positional patterns) to reconstruct without capturing semantics.

**Critical comparison with SoTA:**

| Model | Architecture | Bottleneck | Decoder target |
|---|---|---|---|
| LLaDA / MDLM / Mercury | Single bidirectional Transformer | **None** | Same masked input |
| SODA (Hudson, CVPR 2024) | Encoder + bottleneck + diffusion decoder | **Tight** | **Different view** (novel view synthesis) |
| MrCogito (current) | Encoder + concept bottleneck + diffusion decoder | **128 concepts** | Same text (self-reconstruction) |

LLaDA/MDLM have NO bottleneck -- they process masked tokens directly with full O(N^2) self-attention. They don't need semantic compression because every token can see every other token.

SODA has a tight bottleneck AND generates DIFFERENT content than the encoder saw. This forces genuine semantic compression because pixel-level details don't transfer between views.

MrCogito has a bottleneck but generates the SAME content. The bottleneck is loose enough (4:1 at 512 tokens) to permit surface-level hashing.

**SODA principle for text (prefix generation):** Encode the first 30-50% of a document → generate the remaining 50-70% via diffusion. The decoder CANNOT succeed by memorizing the encoder's input. Forces semantic concepts.

### Cause 5: No Concept Regularization

Deliberately disabled for the baseline (`concept_losses="none"`). The t_regs_mst regularization (implemented, untested) would help push concepts apart, but it's secondary to causes 1-4.

---

## 2. Gradient Flow Analysis

### Forward Pass
```
input_ids (clean) → Encoder (L2/L6 cross-attention) → concepts [B, 128, 512]
input_ids → mask with t → noisy_ids
noisy_ids + concepts + t → Decoder (2 DiffusionDecoderLayers) → hidden [B, L, 512]
hidden[masked_positions] → lm_head → logits → cross_entropy loss
```

### Backward Pass: Where Signal Weakens

1. **AdaLN-Zero gates start at zero.** During early training, `gate_ca ≈ 0` so almost no gradient reaches concepts through cross-attention. The decoder learns its own residual representations first, only gradually pulling from concepts.

2. **Residual shortcut at unmasked positions.** The decoder input is `token_embed(noisy_ids) + pos_embed`. For unmasked positions (~(1-t) fraction), this carries the actual token identity through the residual stream. The decoder can reconstruct unmasked positions WITHOUT using concepts.

3. **No self-attention in decoder.** Each masked position independently queries the 128 concepts. There is no token-token coordination in the decoder. This makes reconstruction harder (good for forcing concept usage) but also limits the decoder's capacity to build coherent outputs.

On the other hand, if we add the self-attention to the decoder we get O(N²) complexity per layer, which is not what we want.

4. **Sparse loss.** Only masked positions contribute to the loss. At t=0.1, only ~50 of 512 positions generate gradients. Effective gradient per concept is diluted by 1/128 (softmax over concepts).

Does it mean that we should get back to the full loss, with all positions contributing to the loss?

### Gradient Magnitude Comparison

| Objective | Positions with gradient | Dilution | Relative signal |
|---|---|---|---|
| MLM (15% sparse) | ~15% | 1/128 per concept | **1x** |
| Diffusion at t=0.5 | ~50% | 1/128 per concept | ~3.3x |
| Diffusion at t=0.9 | ~90% | 1/128 per concept | ~6x |
| TSDAE (all positions, no shortcut) | 100% | 1/128, no shortcut | **~83x** |
| Prefix generation | 50-70% suffix | 1/128, no shortcut | **~40-60x** |

---

## 3. SoTA Comparison: MrCogito vs Diffusion LMs

### LLaDA (Nie et al., 2025)

- **Architecture:** Single bidirectional Transformer (LLaMA with causal mask removed)
- **No bottleneck.** Full O(N^2) self-attention. Every token sees every other token.
- **Loss:** `cross_entropy / p_mask` (ELBO-derived weighting)
- **Masking schedule:** `p_mask ~ Uniform(eps, 1-eps)`, eps=1e-3
- **Scale:** 8B parameters, competitive with LLaMA3 8B
- **LLaDA 2.0:** Scales to 100B via AR→diffusion conversion (knowledge inheritance)

### MDLM (Sahoo et al., NeurIPS 2024)

- **Architecture:** Encoder-only Transformer
- **Training:** Rao-Blackwellized continuous-time ELBO = weighted average of MLM losses
- **Key insight:** The simplified ELBO has an elegant form as weighted MLM. Loss weight is proportional to 1/t.
- **Performance:** Approaches AR perplexity within 15-25% on LM1B and OpenWebText

### Mercury (Inception Labs, 2025)

- **Architecture:** Transformer with discrete-diffusion training objectives
- **Uses:** Time embeddings + adaptive LayerNorm (AdaLN) -- same as MrCogito's decoder
- **12-30 denoising steps** at inference
- **Performance:** 10x faster than AR models at comparable quality

### Key Difference: None of These Have a Bottleneck

All SoTA diffusion LMs operate at the FULL token level. They are essentially "BERT with variable masking rate." The model that processes input IS the model that predicts output. Gradients flow directly.

MrCogito compresses through a concept bottleneck, which creates a fundamentally harder optimization problem. The training objective choice matters FAR more with a bottleneck than without.

---

## 4. Proposed Fixes (Priority Order)

### Fix 1: L6 Diffusion Ablation (TODO 11, 1 GPU-day)

**What:** Run exact same config as L2 but with 6 encoder layers. No code changes.

**Why:** Isolates the dominant confound. If STS-B improves significantly (>0.50), the problem is encoder depth, not the diffusion objective itself.

**Decision logic:**
- STS-B > 0.50 → depth was the bottleneck; keep diffusion, scale encoder
- STS-B < 0.30 → self-reconstruction is fundamentally insufficient; pivot to prefix generation
- Concept rank > 20/128 → geometry improvement scales with depth

### Fix 2: ELBO Loss Weighting + t_min=0.3 (TODO 12, 0.5 day code)

**What:** Weight loss by 1/t (MDLM/LLaDA) and raise t_min to 0.3.

```python
# In ConceptEncoderForMaskedDiffusion.forward():
weighted_loss = diffusion_loss / t.mean().clamp(min=0.1)
```

**Why:** Normalizes gradient magnitude across noise levels. Eliminates trivial near-MLM regime.

### Fix 3: Prefix Generation Training (TODO 13, 3 days code + 5 GPU-days)

**What:** Split documents into prefix (30-50%) and suffix (50-70%). Encoder sees clean prefix → concepts. Decoder generates suffix via diffusion, conditioned on concepts.

**Why (SODA principle):** The decoder CANNOT succeed by memorizing the encoder's input because it generates DIFFERENT text. Forces genuine semantic compression through the bottleneck. Directly trains for the inference-time use case (encode input → generate output).

**Implementation plan:**

| Component | Action | File |
|---|---|---|
| Data collator | `DataCollatorForPrefixGeneration`: split text at random position, return `prefix_ids`, `suffix_ids`, `prefix_attention_mask` | `training/data_collators.py` |
| Model class | `ConceptEncoderForPrefixDiffusion`: encoder processes prefix, decoder generates suffix with diffusion | `nn/concept_encoder_diffusion.py` (extend or new class) |
| Training script | `training/train_prefix_diffusion.py` | New file |
| Shell script | `scripts/train_prefix_diffusion_multigpu.sh` | New file |

**Key architectural decisions:**
- Encoder sees CLEAN prefix (matches inference distribution)
- Decoder uses diffusion (variable masking on suffix tokens, conditioned on concepts via cross-attention)
- Position embeddings for decoder are relative to suffix start (not absolute document position)
- Loss: ELBO-weighted cross-entropy on suffix tokens only

### Fix 4: t_regs_mst Regularization (0.5 day code)

**What:** Add t_regs_mst with weight 0.02 (fixed, not Kendall-Gal). Operates within-sample on concept nearest-neighbor distances.

**Why:** The `combined` loss failed because it operates across-batch. t_regs_mst targets within-sample collapse directly.

**Caution:** Only add AFTER gradient flow is healthy (Fixes 1-3). History shows regularization on a misaligned architecture produces "geometrically diverse but semantically empty" vectors.

---

## 5. Regularization in Other Latent Reasoning Models

| Model | Regularization | Key Insight |
|---|---|---|
| **Seq-VCR** (2025) | Sequential Variance-Covariance at EACH layer | Regularize intermediate layers, not just output. 99.5% on 5x5 multiplication. |
| **EQ-VAE** (2025) | Equivariance to semantic-preserving transforms | Enforce invariance without degrading reconstruction |
| **CORAL** (2025) | Supervised contrastive in bottleneck layer | Separate latent class representations to prevent overlap |
| **Coconut** (Meta, 2024) | Multi-stage curriculum (CoT → latent) | Don't learn language + reasoning simultaneously; stage it |
| **Recurrent Depth** (Geiping, 2025) | Variable-depth training, truncated backprop | Standard next-token loss; no explicit regularization needed |

**Key takeaway for MrCogito:** Seq-VCR's per-layer regularization is applicable -- apply t_regs_mst at each encoder layer, not just the final concept output. Coconut's curriculum approach suggests staging: first learn good concepts (Phase 0-1), then add recursive reasoning (Phase 2).

---

## 6. Key Equations

### ELBO-Derived Diffusion Loss (MDLM)

The proper training objective for masked diffusion is:

```
L = E_t [ (1/t) * E_mask [ cross_entropy(x_masked, predicted) ] ]
```

Where t is the masking rate. The 1/t weighting ensures consistent gradient magnitude regardless of how many tokens are masked.

### Effective Gradient Per Concept

```
Current diffusion:  ~t * (1/C) * upstream_grad  (varies with t)
ELBO-weighted:      ~1 * (1/C) * upstream_grad  (constant)
Prefix generation:  ~(1-prefix_ratio) * (1/C) * upstream_grad  (no shortcut)
```

Where C=128 concepts. The prefix generation case is stronger because there is no residual shortcut from the encoder's input tokens to the decoder.

---

## 7. References

| Paper | Year | Key Finding | Link |
|---|---|---|---|
| SODA (Hudson) | 2024 | Bottleneck diffusion learns semantic representations via novel view synthesis | [CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Hudson_SODA_Bottleneck_Diffusion_Models_for_Representation_Learning_CVPR_2024_paper.html) |
| MDLM (Sahoo) | 2024 | Simplified ELBO = weighted MLM; loss weight ∝ 1/t | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb0b13cc515724ab8015bc978fdde0ad-Abstract-Conference.html) |
| LLaDA (Nie) | 2025 | Masked diffusion LLM at 8B; loss/p_mask weighting | [arXiv](https://arxiv.org/abs/2502.09992) |
| Mercury (Inception) | 2025 | Commercial dLLM; AdaLN + 10x faster than AR | [arXiv](https://arxiv.org/abs/2506.17298) |
| Coconut (Meta) | 2024 | Chain of continuous thought; latent reasoning outperforms CoT | [GitHub](https://github.com/facebookresearch/coconut) |
| Recurrent Depth (Geiping) | 2025 | Prelude+recurrent+coda; 3.5B matches 103B via test-time recurrence | [arXiv](https://arxiv.org/abs/2502.05171) |
| Seq-VCR | 2025 | Per-layer variance-covariance regularization for reasoning | [arXiv](https://arxiv.org/abs/2411.02344) |
| CleanDIFT | 2025 | Train-test mismatch when extracting features from clean vs noisy inputs | [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Stracke_CleanDIFT_Diffusion_Features_without_Noise_CVPR_2025_paper.pdf) |

---

*Created: 2026-02-26*
*Related: [mlm_perceiver_diagnosis_20260221.md](mlm_perceiver_diagnosis_20260221.md), [diffusion_L2_eval_20260225.md](../2_Experiments_Registry/run_reports/diffusion_L2_eval_20260225.md)*
*Next review: after L6 diffusion ablation (TODO 11) results*
