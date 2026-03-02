# Deep Analysis: Why Diffusion + ELBO Self-Reconstruction Fails

**Date:** 2026-03-01
**Author:** Krzysztof Sopyla / AI deep analysis session
**Status:** Permanent research note
**Supersedes:** [diffusion_diagnosis_20260226.md](diffusion_diagnosis_20260226.md) (prior diagnosis correctly identified symptoms but missed the fundamental cause)
**Related experiments:**
- `diffusion_H512L2C128D2_20260223_203349` (L2 diffusion, rank 10.1/128, STS-B 0.138)
- `diffusion_H512L6C128D2` (L6 diffusion + ELBO, rank 5.74/128, STS-B 0.174)
- `perceiver_mlm_H512L6C128_20260208_211633` (L6 MLM baseline, rank 5/128, STS-B 0.650)

---

## 1. Executive Summary

After a layer-by-layer, step-by-step analysis of the full forward/backward pass, comparison with reference implementations (MDLM, LLaDA), and cross-referencing with VQ-VAE collapse literature, the conclusion is:

**The implementation is correct. The ELBO weighting is mathematically equivalent to MDLM/LLaDA. The architecture is sound. The problem is fundamental: self-reconstruction through a concept bottleneck teaches the model to build a positional hash function, not semantic representations.**

L6 encoder depth makes this WORSE (rank 5.74 < L2's 10.1) because a deeper encoder builds a more efficient hash. Regularization (VICReg, t_regs_mst) will increase geometric diversity but not semantic content, exactly replicating the Kendall-Gal failure (rank 95% but GLUE crashed, Feb 19).

The only way forward is to change WHAT the decoder generates, not HOW.

---

## 2. Full Forward Pass Trace (Layer by Layer)

### Step 1: Encoder Input

```
input_ids: [B, 512] — clean text tokens (no masking, encoder always sees full text)
attention_mask: [B, 512] — padding mask
```

### Step 2: Token Embeddings in Encoder

```python
token_embeddings = token_embed(input_ids) + pos_embed(positions)  # [B, 512, H=512]
```

**Critical observation:** Token embeddings are computed ONCE and remain STATIC throughout all encoder layers (in standard mode, no BiXT). The tokens are never updated by self-attention. They serve only as key/value sources for concept cross-attention.

### Step 3: Concept Initialization

```python
concept_repr = concept_embed(arange(128))  # [B, 128, 512] — learned prototypes, same for all inputs
```

Every batch item starts with identical concept vectors. Specialization happens through the layers.

### Step 4: Encoder Layers (L=6 layers)

Each `ConceptEncoderLayer` does:

```
Layer i:
  a. Cross-attention: concepts (Q) attend to tokens (K/V)    → O(C*N) = O(128*512) = 65K ops
     concepts absorb information from token embeddings
  b. Self-attention:  concepts (Q/K/V) attend to each other  → O(C^2) = O(16K) ops
     concepts coordinate and redistribute information
  c. Gated FFN: per-concept nonlinear transformation
     → output: updated concepts [B, 128, 512]
```

**After 6 layers:** concepts = [B, 128, 512]. These 128 vectors are the ONLY information channel between encoder and decoder.

**Why tokens stay static matters:** Because tokens never update through self-attention, the encoder's cross-attention operates on UNCONTEXTUALIZED embeddings. "Bank" in "river bank" and "bank" in "savings bank" have identical key/value representations across ALL 6 encoder layers. Compositional semantics must emerge entirely within the concept self-attention.

### Step 5: Noise Application (Forward Diffusion)

```python
t ~ Uniform(0.3, 1.0)              # [B] — one noise level per sample
noise_mask = (rand < t)             # [B, 512] — each token independently masked with prob t
noisy_ids = input_ids.clone()
noisy_ids[noise_mask] = mask_token_id  # Replace masked tokens with [MASK]
```

At t=0.3: ~150 of 512 tokens masked (30%). At t=0.9: ~460 tokens masked (90%).

### Step 6: Decoder Input

```python
x = decoder.token_embed(noisy_ids) + decoder.pos_embed(positions)  # [B, 512, H]
t_emb = sinusoidal_embed(t) → MLP                                  # [B, H]
```

**Critical observation:** The decoder has its OWN token embeddings (separate from encoder). For each position:
- **Unmasked:** `token_embed(actual_token) + pos_embed(pos)` — carries full token identity
- **Masked:** `token_embed([MASK]) + pos_embed(pos)` — carries ONLY position information

### Step 7: Decoder Layers (D=2 cross-attention-only layers)

Each `DiffusionDecoderLayer` does:

```
AdaLN-Zero modulation from t_emb:
  [scale_ca, shift_ca, gate_ca, scale_ff, shift_ff, gate_ff] = Linear(t_emb)
  All initialized to ZERO → layer starts as identity

Cross-attention (the ONLY attention):
  x_norm = LayerNorm(x) * (1 + scale_ca) + shift_ca
  ca_out = CrossAttention(Q=x_norm, K=concepts, V=concepts)   # O(N*C)
  x = x + gate_ca * ca_out                                    # gate starts at 0!

Gated FFN:
  x_norm = LayerNorm(x) * (1 + scale_ff) + shift_ff
  ff_out = GatedFFN(x_norm)
  x = x + gate_ff * ff_out
```

**NO self-attention between tokens in the decoder.** Each of the 512 positions independently queries the 128 concepts. Position j has NO direct information path to position k.

### Step 8: Sparse Loss Computation

```python
hidden[masked_positions] → lm_head → logits [M, V]    # only at ~t*512 masked positions
loss = cross_entropy(logits, original_tokens[masked]) / t  # ELBO 1/t weighting
```

### Backward Pass: Where the Gradient Signal Goes

Tracing backward from the loss:

1. **lm_head gradient** → hidden states at masked positions ONLY. Unmasked positions get zero gradient.
2. **Through decoder layer 2 residual:** `x = x_input + gate_ca * ca_out + gate_ff * ff_out`
   - Gradient flows to x_input (decoder token embeddings) via residual
   - Gradient to concepts via `gate_ca * cross_attention` — BUT gate_ca starts at zero
   - Gradient to gate_ca, scale_ca, shift_ca (AdaLN parameters) — these DO get gradient
3. **Through decoder layer 1:** Same pattern.
4. **Gradient to concepts:** ONLY through `gate_ca * cross_attention` paths (both layers)
5. **Gradient through encoder:** concepts → 6 encoder layers → token embeddings

**The gradient to concepts is proportional to gate_ca.** During early training, gate_ca ≈ 0, so concepts get negligible gradient. The model first learns to use whatever information is in the decoder's token embeddings (which is nothing for masked positions beyond position).

**Eventually gate_ca must grow** because without concepts, each masked position can only predict from `[MASK] + position` — which gives ~log(V) ≈ 10.8 loss (uniform prior). The initial training loss of 14.19 confirms this. To push loss below 10.8, the decoder MUST open its gates and use concepts.

---

## 3. ELBO Implementation Verification

### MrCogito implementation (verified correct):

```python
per_token_loss = CE(logits, targets, reduction='none')     # [M]
token_weights = 1.0 / t[sample_indices].clamp(min=0.1)     # [M]  — 1/t per token
loss = (per_token_loss * token_weights).sum() / token_weights.sum()
```

### LLaDA implementation (reference):

```python
token_loss = CE(logits[masked], input_ids[masked], reduction='none') / p_mask[masked]
loss = token_loss.sum() / (B * L)
```

### MDLM implementation (reference):

```python
# SUBS parameterization, continuous time (log-linear schedule σ = -log(1-t)):
# weight = dσ/dt / (exp(σ)-1) = [1/(1-t)] / [t/(1-t)] = 1/t
loss_per_pos = -log_p_theta * (dsigma / expm1(sigma))  # [B, L], 0 at unmasked positions
loss = (loss_per_pos * attention_mask).sum() / attention_mask.sum()
```

### Mathematical equivalence proof:

For a single sample with masking rate t and L positions:
- Number of masked tokens: M ≈ t*L
- **LLaDA:** `sum_M(CE/t) / L = (M * mean_CE / t) / L = (tL * mean_CE / t) / L = mean_CE`
- **MrCogito:** `sum_M(CE * 1/t) / sum_M(1/t) = (M * mean_CE/t) / (M/t) = mean_CE`
- **MDLM:** Same derivation as LLaDA for log-linear schedule.

**All three produce the same effective loss: the mean cross-entropy at masked positions, independent of t.** The 1/t weighting exactly cancels with the proportional number of masked tokens. The implementation is mathematically correct.

---

## 4. Why Self-Reconstruction Teaches Hash Functions (The Fundamental Cause)

### Information-theoretic argument

Self-reconstruction asks the model to minimize:

```
L = E_t [ I_reconstruct(X | concepts(X), noisy(X, t)) ]
```

The encoder sees clean X and produces concepts. The decoder sees noisy X (with fraction t masked) and must reconstruct the masked tokens using concepts.

For this objective, the optimal concept representation is one that **maximally compresses the token-level information** needed for reconstruction. With 128 concept vectors of dimension 512, the concept space has 128 × 512 = 65,536 floats of capacity. The input has 512 tokens from a ~50K vocabulary, requiring ~512 × 16 = 8,192 bits.

The capacity (65K floats × 32 bits) vastly exceeds the information content (8K bits). So the bottleneck is NOT tight enough to force lossy compression. The model can learn a nearly LOSSLESS encoding of all token identities.

### What does lossless encoding look like? A hash function.

The encoder learns to distribute token identities across concept dimensions such that:
- **Concept slot i** stores information about tokens at a subset of positions
- **A few dimensions** of each concept vector encode the token identity at those positions
- The decoder's cross-attention learns position-based addressing: "I am position k, attend to concept i to retrieve my token"

This is a distributed lookup table. It needs very few dimensions of the concept space (5-10 effective dimensions), explaining the severe rank collapse.

### Why SODA works but self-reconstruction doesn't

| Property | SODA (vision) | MrCogito (text) |
|---|---|---|
| Encoder input | Image view A | Full clean text |
| Decoder target | Image view **B** (different viewpoint) | Same text (self-reconstruction) |
| What transfers between views? | Semantics (objects, layout) | Everything (exact tokens) |
| What the bottleneck MUST encode | Semantic structure | Token identities |
| Optimal concept representation | Semantic features | Positional hash |

SODA forces semantic compression because pixel-level details of view A are useless for generating view B. The ONLY information that transfers between views is semantic (what objects exist, where they are, their relationships).

Self-reconstruction has no such constraint. ALL token-level information transfers (trivially — it's the same text). So the bottleneck learns the most efficient compression of token identities, not semantics.

### The SODA principle for text = prefix generation

To apply SODA to text: the encoder sees the first 30-50% of a document, and the decoder generates the remaining 50-70% via diffusion. The decoder CANNOT succeed by memorizing the encoder's input because it generates DIFFERENT text. The concepts MUST carry semantic information (topic, entities, narrative direction) because that's all that's shared between prefix and suffix.

```
I(concepts; suffix | prefix) ≈ semantic mutual information
                             ≈ 1-2 bits per suffix token (topic, entities, style)
                             << 16 bits per token (full token identity)
```

This forces the bottleneck to extract ONLY the semantic signal, using ALL 128 concept dimensions to represent rich semantic structure rather than a sparse positional hash.

---

## 5. Why L6 Depth Makes Rank WORSE (Counter-Intuitive)

### The paradox

| Experiment | Encoder depth | Effective rank | STS-B |
|---|---|---|---|
| L2 diffusion | 2 layers | **10.1/128** | 0.138 |
| L6 diffusion + ELBO | 6 layers | **5.74/128** | 0.174 |

L6 has 3x more cross-attention + self-attention capacity. Yet it produces WORSE dimensional utilization.

### Explanation: deeper encoder = more efficient hash

With 2 encoder layers, the concepts can only build a crude hash function. The limited refinement capacity means the hash needs more dimensions to encode 512 token identities → rank 10.

With 6 encoder layers, the concepts undergo 6 rounds of cross-attention and self-attention refinement. This allows a much more efficient hash — the same token information is compressed into fewer dimensions → rank 5.74.

**More encoder capacity → better compression → fewer dimensions needed → lower rank.**

### VQ-VAE parallel (DCVQ, NeurIPS 2025)

The DCVQ paper found the same phenomenon in VQ-VAEs: "VQ-VAEs compress representations into surprisingly low-dimensional subspaces (typically 4-10 dimensions) despite high-dimensional embeddings." This low-dimensionality is INTRINSIC to the learning dynamics when the decoder can reconstruct from a low-dimensional subspace. Our effective rank of 5-10 is exactly in the VQ-VAE collapse range.

Their solution (partitioning the latent space into independent low-dimensional subspaces) is a structural intervention at the bottleneck level. Our equivalent would be to partition concepts into groups that cannot share information — but this alone doesn't fix the semantic emptiness problem.

---

## 6. Why VICReg/t_regs_mst Will Increase Rank But Not Semantics

### Prediction for TODO 11b

VICReg forces all concept dimensions to have equal variance. t_regs_mst forces concepts apart in representation space. Together they will push effective rank toward 30-60/128.

BUT: the concepts will be geometrically diverse while remaining semantically empty. The model will distribute its hash function across 128 dimensions instead of 5. Each dimension will carry a tiny fraction of the token identity information. The result will be:

- Effective rank: HIGH (30-60/128) ← meets the geometric target
- STS-B: LOW (~0.15-0.25) ← still no semantic content
- GLUE: MIXED — MRPC might improve slightly, QQP/MNLI will remain poor

This is exactly what happened with Kendall-Gal (Feb 19): rank 95% but QQP dropped 13.76% and MNLI dropped 10%. **Geometric diversity without semantic content is a false signal.**

### Why regularization cannot fix the fundamental problem

Regularization changes WHERE information is stored (spread across dimensions vs concentrated), not WHAT information is stored (token hashes vs semantics). Only changing the training objective can change what the concepts learn to encode.

---

## 7. AdaLN-Zero: Necessary But Creates a Chicken-and-Egg Problem

### The intended benefit

Zero-initialized gates prevent gradient explosion (proven: the old non-zero-init decoder exploded at epoch 12 in the Feb 21 run). The layer starts as identity and gradually incorporates conditioning.

### The unintended consequence

With gate_ca ≈ 0 during early training, the decoder receives almost no signal from concepts. The loss gradient flows primarily through the residual path to the decoder's own token embeddings and through the AdaLN parameters.

The encoder converges first (it has no AdaLN gates blocking its gradients), building a representation optimized for whatever the decoder's cross-attention pattern converges to. By the time the gates open significantly, the encoder has already settled into a local minimum.

### Why this doesn't prevent hash learning

The gates DO open — loss drops from 14.19 to 2.89, which requires concept usage. But the opening is gradual, and the encoder adapts continuously. The encoder-decoder pair co-evolve toward the nearest loss minimum, which is the hash function.

### Potential mitigation (NOT a fix)

Pre-initialize gate biases to a small positive value (e.g., 0.1 instead of 0.0) so concepts have non-zero influence from step 1. This would change the convergence dynamics but NOT the loss landscape — the hash function minimum would still be the attractor.

---

## 8. The Cross-Attention-Only Decoder: Right Architecture, Wrong Objective

### Why no self-attention is architecturally correct

The purpose of the concept bottleneck is O(C*N) complexity instead of O(N^2). Adding self-attention to the decoder defeats this purpose:

```
Cross-attention only: O(N*C) per layer = O(512*128) = 65K ops
Self-attention added:  O(N^2) per layer = O(512^2) = 262K ops  (4x worse)
At N=2M (target):      O(2M*128) = 256M vs O(2M^2) = 4T  (15,000x worse)
```

### How it affects concept learning

Without self-attention, each masked position independently queries the concepts. There is NO direct information path between positions in the decoder. This means:

1. Concepts are the SOLE information channel for masked token prediction ← GOOD for forcing concept usage
2. Each position must independently reconstruct its token from the same 128 concepts ← creates pressure for concepts to be a shared, queryable memory
3. The most efficient "shared queryable memory" for token reconstruction is... a hash table ← BAD for semantics

The architecture forces concept usage but doesn't force semantic concept content. The objective determines what the concepts learn, not the architecture.

---

## 9. Comparison: Why LLaDA/MDLM Work Without These Problems

### The key difference: NO bottleneck

| Property | LLaDA / MDLM | MrCogito |
|---|---|---|
| Architecture | Single bidirectional Transformer | Encoder → concept bottleneck → decoder |
| Token interactions | Full O(N^2) self-attention | O(C*N) cross-attention only |
| Bottleneck | None | 128 concept vectors |
| Self-reconstruction works? | Yes — rich contextual representations | No — hash function through bottleneck |
| Why? | Every token sees every other token; representations ARE contextual | Must compress 512 tokens into 128 concepts; most efficient compression is hashing |

LLaDA/MDLM don't need semantic compression because they have no bottleneck. Each token's representation is enriched by full self-attention across all other tokens. The "representation" IS the contextual understanding. There's nothing to compress.

MrCogito's entire value proposition is the bottleneck (O(C*N) complexity, fixed-size concept memory). But the bottleneck creates a compression problem, and self-reconstruction's optimal compression is not semantic — it's positional hashing.

**LLaDA/MDLM and MrCogito are solving fundamentally different problems.** Borrowing LLaDA's loss function without recognizing this distinction was a category error.

---

## 10. Revised Root Causes (Ranked by Impact)

### RC1: Self-Reconstruction Permits Positional Hashing (FUNDAMENTAL — not fixable by tuning)

The training objective asks: "encode X → concepts → reconstruct X." The optimal concept representation for this objective is a lossless compression of token identities, which requires only 5-10 dimensions of concept space. This is the dominant cause of both concept collapse and semantic emptiness.

**Evidence:** All self-reconstruction variants collapse:
- MLM: rank 5/128
- Diffusion L2: rank 10.1/128
- Diffusion L6 + ELBO: rank 5.74/128
- Kendall-Gal forced rank to 95% but STS-B crashed to 0.341

### RC2: Deeper Encoder Enables More Efficient Hashing (explains L6 < L2 paradox)

More encoder layers → more refinement → more efficient compression → fewer dimensions needed → lower effective rank. This is the same phenomenon as VQ-VAE dimensional collapse (4-10 dimensions regardless of embedding size).

**Evidence:** L6 rank 5.74 < L2 rank 10.1 despite 3x more capacity.

### RC3: AdaLN-Zero Delays Concept Usage, Biasing Early Convergence

Zero-initialized gates mean concepts contribute nothing initially. The encoder converges to a representation before the decoder fully utilizes it, creating a local minimum that's hard to escape.

**Evidence:** Training loss starts at 14.19 (near log(V)), indicating the decoder initially ignores concepts.

### RC4: Bottleneck Capacity Exceeds Information Requirement

128 concepts × 512 dims = 65K floats >> 512 tokens × 16 bits = 8K bits needed for token reconstruction. The bottleneck is too loose for self-reconstruction to force semantic compression.

**Evidence:** Low rank (5-10/128) confirms only a small fraction of capacity is used.

### RC5: Static Token Embeddings in Encoder (No Token Contextualization)

Encoder cross-attention operates on raw token embeddings (word + position), never updated by self-attention. "Bank" in different contexts has identical encoder K/V representations. Compositional semantics must emerge entirely within concept self-attention (128 concepts × C^2 interactions), which may be insufficient.

**Partial fix available:** BiXT (already implemented) adds reverse cross-attention that updates token embeddings from concepts at each layer, creating contextualized tokens. This doesn't fix RC1 but improves the encoder's representational capacity.

---

## 11. What Must Change: Objective, Not Architecture

### The fix: prefix generation (TODO 13)

```
Current:  Encoder(X) → concepts → Decoder(noisy(X)) → reconstruct X     ← hash function
Needed:   Encoder(prefix) → concepts → Decoder(noisy(suffix)) → generate suffix  ← semantic compression
```

With prefix generation:
1. The encoder sees CLEAN prefix tokens (first 30-50% of document)
2. The decoder generates DIFFERENT tokens (remaining 50-70%) via diffusion
3. The concepts CANNOT store suffix token identities (encoder never saw them)
4. The concepts MUST capture the semantic gist that connects prefix to suffix
5. This forces ALL concept dimensions to be used for rich semantic representation

### Why VICReg is still valuable but only WITH prefix generation

VICReg + t_regs_mst prevent the GEOMETRIC collapse. Prefix generation prevents the SEMANTIC collapse. You need both:
- Without VICReg: prefix generation might still collapse to a few semantic dimensions
- Without prefix generation: VICReg gives geometric diversity but empty semantics

The correct experiment is: prefix generation + VICReg + t_regs_mst (TODO 13 + TODO 11b combined).

### Why TSDAE is a weaker alternative

TSDAE (token deletion + full reconstruction) is partway between self-reconstruction and prefix generation:
- Encoder sees corrupted text (60% tokens deleted) — some information is lost
- Decoder reconstructs ALL tokens, including deleted ones — still self-reconstruction
- The bottleneck must carry information about deleted tokens — some semantic pressure

But TSDAE still provides the encoder with 40% of the original tokens. With 40% of tokens and position information, a hash function can still partially reconstruct. It's less efficient than full self-reconstruction but doesn't force semantic abstraction as strongly as prefix generation (where the decoder generates completely unseen text).

---

## 12. Experimental Recommendations (Updated Priority)

### Immediate (this week):

1. **TODO 11b (VICReg + t_regs_mst):** Run anyway for a clean baseline. Expect rank improvement (30-60/128) but minimal STS-B improvement (<0.30). Confirms the "geometric diversity ≠ semantic content" principle.

2. **TODO 13 (Prefix Generation):** Highest priority implementation. This is the only experiment that changes the fundamental optimization landscape from hashing to semantic compression.

### Implementation notes for prefix generation:

- Split each document at 30-50% (random per sample for robustness)
- Encoder: clean prefix → concepts (standard ConceptEncoder)
- Decoder: diffusion on suffix tokens, cross-attending to concepts
- Position embeddings for suffix: relative to suffix start (pos 0, 1, 2...)
- Loss: ELBO-weighted CE on suffix tokens only (correct implementation already exists)
- VICReg + t_regs_mst with weight 0.02 (combine from the start)
- Data: Minipile documents with sufficient length (filter short documents)

### Decision gates:

| Result | Action |
|---|---|
| STS-B > 0.50, rank > 30 | Prefix generation works. Scale data (TODO 7). |
| STS-B 0.30-0.50, rank > 20 | Partial success. Add contrastive loss (SimCSE). |
| STS-B < 0.30, rank < 20 | Prefix generation alone insufficient. Try prefix + TSDAE combined, or Slot Attention (C5). |

---

## 13. Key Equations

### Information-theoretic formulation

Self-reconstruction:
```
I_needed(concepts; X) = H(X) ≈ 8K bits  (full token information)
I_available(concepts) = 128 × 512 × 32 bits ≈ 2M bits  (concept space capacity)
Compression ratio: 8K / 2M = 0.4%  → bottleneck is VERY loose → hash function
```

Prefix generation:
```
I_needed(concepts; suffix | prefix) = I(prefix; suffix) ≈ semantic mutual info
                                    ≈ 1-2 bits per suffix token ≈ 300-500 bits total
I_available(concepts) = 2M bits
Compression ratio: 500 / 2M = 0.025%  → bottleneck is still loose
BUT: the information is SEMANTIC, not token-level → concepts learn semantics
```

The key insight is not the compression ratio but the NATURE of the information that must pass through the bottleneck.

### Effective rank and hash efficiency

For a hash function that maps 512 tokens to 128 concepts:
```
Minimum dimensions needed = log2(V^N) / (C * bits_per_float)
                          = 512 * 16 / (128 * 32)
                          ≈ 2 dimensions
```

In practice, the hash is approximate (soft attention, not hard indexing), requiring 5-10 dimensions. This exactly matches our observed effective ranks.

---

## 14. References

| Paper | Year | Key Finding |
|---|---|---|
| SODA (Hudson) | CVPR 2024 | Bottleneck diffusion learns semantics only with novel-view synthesis (different target than input) |
| MDLM (Sahoo) | NeurIPS 2024 | Simplified ELBO = weighted MLM; loss weight is 1/t. No bottleneck. |
| LLaDA (Nie) | 2025 | Masked diffusion LLM at 8B; loss/p_mask weighting. No bottleneck. |
| DCVQ (NeurIPS 2025) | 2025 | VQ-VAEs collapse to 4-10 dimensions intrinsically; partitioning helps. |
| SimpleVQ (ICCV 2025) | 2025 | Reparameterize codebook through linear transformation to prevent collapse. |
| IBQ (2024) | 2024 | All-codebook differentiable optimization; 96% utilization via one-hot backprop. |
| Deconstructing Diversity (2025) | 2025 | Information Bottleneck theory for discrete latent models: compression vs diversity pressure. |
| BiXT (NeurIPS 2024) | 2024 | Bidirectional cross-attention unlocks "where+what" bottleneck in Perceiver architectures. |
| Posterior Collapse (Scale-VAE) | 2024 | VAE decoders bypass latent space; Inverse Lipschitz constraints help. |

---

## 15. Addendum: BiXT and Dimension Inversion Analysis

### 15.1 Would BiXT Help? (Contextualized Tokens Through Concepts)

BiXT (Hiller et al., NeurIPS 2024) adds reverse cross-attention at each encoder layer: tokens attend to concepts (`tokens <- cross-attn(Q=tokens, KV=concepts)`), making tokens contextual WITHOUT O(N^2) self-attention. The paper's key finding: this "unlocks a key bottleneck experienced by Perceiver-like architectures and enables the processing and interpretation of both semantics ('what') and location ('where') to develop alongside each other over multiple layers."

#### How BiXT changes the encoder dynamics

**Standard encoder (current):**
```
Layer i: concepts <- cross-attn(Q=concepts, KV=static_tokens)   [O(C*N)]
         concepts <- self-attn(Q/K/V=concepts)                  [O(C^2)]
         concepts <- FFN(concepts)
```
Tokens never change. "Bank" in "river bank" and "savings bank" have identical K/V across all 6 layers. Concepts must discover compositionality entirely through self-attention among themselves.

**BiXT encoder:**
```
Layer i: concepts <- cross-attn(Q=concepts, KV=tokens_i)        [O(C*N)]
         tokens_i <- cross-attn(Q=tokens_i, KV=concepts)        [O(N*C)]  ← NEW
         concepts <- self-attn(Q/K/V=concepts)                  [O(C^2)]
         concepts <- FFN(concepts)
```
Tokens update each layer. After layer 1, "bank" has been enriched by concept feedback that encodes surrounding context. By layer 6, tokens are fully contextualized through the concept bottleneck.

#### Does BiXT fix the hash function problem?

**Verdict: No, but it creates helpful intermediate pressure.**

The dominant gradient is still from the reconstruction loss, which favors positional hashing regardless of whether tokens are contextualized. The encoder-decoder optimization landscape is still: "find the most efficient compression for reconstructing the same text."

However, BiXT introduces a meaningful secondary effect: the reverse cross-attention creates IMPLICIT INTERMEDIATE SUPERVISION. At each layer, the concepts must be useful not only for the final reconstruction (via the decoder) but also for contextualizing tokens (for the next layer's forward cross-attention). Contextualizing "bank" correctly requires semantic understanding of surrounding words — a genuinely semantic task.

The gradient through the reverse cross-attention path is:
```
∂loss/∂concepts_layer_i = (final reconstruction gradient)   ← dominates
                        + (token contextualization gradient)  ← secondary, semantic
```

The secondary gradient is weaker because it passes through several more layers before reaching the loss. But it biases the concept representation toward being contextually useful, not just hash-efficient.

#### BiXT + prefix generation = strong synergy

With prefix generation (encoder sees prefix, decoder generates suffix):
- The encoder must build concepts that capture the SEMANTIC GIST of the prefix
- BiXT ensures token representations are contextualized, so concepts are built from richer token features
- This gives the encoder better raw material for semantic compression

Without BiXT, the encoder's cross-attention operates on raw word+position embeddings. With BiXT, it operates on tokens that have already been enriched by concept feedback. The difference is analogous to: "summarize raw words" vs "summarize contextualized sentence representations."

**Recommendation:** Include BiXT in prefix generation experiments. Cost: ~30-40% more FLOPs per encoder layer (one extra O(N*C) cross-attention), but still O(C*N) total. Given C=128, this is negligible compared to O(N^2) self-attention.

#### BiXT alone with self-reconstruction: marginal improvement

If we add BiXT to the current self-reconstruction setup (without changing the objective), expect:
- Slightly better concept quality (the intermediate token-contextualization gradient helps)
- Marginal STS-B improvement (maybe 0.15 -> 0.25, not enough to pass the 0.50 threshold)
- The fundamental problem (hash function optimality) remains

This is NOT worth running as a standalone experiment. BiXT should be combined with prefix generation.

---

### 15.2 Dimension Inversion: Would Tiny Token Dim Help?

The idea: keep token embeddings at very low dimension (8, 16, 32, 64) while concepts remain at 512. Tokens are just identifiers that concepts attend to, with no need for "self-refinement" in a rich space.

#### Implementation details (as currently coded)

**Encoder side:**
```python
token_embeddings = nn.Embedding(vocab_size, token_dim)         # [V, 8]
token_position_embeddings = nn.Embedding(max_seq, token_dim)   # [512, 8] ← ALSO low-dim!
# Both combined and projected:
tokens = token_projection(word_8d + pos_8d)                    # Linear(8, 512) → rank-8 in 512-dim
```

Both word AND position embeddings live in token_dim space. This is critical: with token_dim=8, positions are ALSO only 8-dimensional. The encoder's cross-attention operates on a rank-8 representation of (word+position).

**Decoder side:**
```python
token_embed = nn.Embedding(vocab_size, token_dim)              # [V, 8]
pos_embed = nn.Embedding(max_seq, hidden_size)                 # [512, 512] ← FULL dim!
x = token_proj(token_embed(noisy_ids)) + pos_embed(pos_ids)   # rank-8 token + full-rank position
```

The decoder's position embeddings are at FULL hidden_size (512), NOT token_dim. This asymmetry means the decoder can distinguish all 512 positions perfectly regardless of token_dim, while the encoder must work with low-dim position representations.

#### Effect on encoder cross-attention (the interesting part)

With token_dim=8, the encoder's cross-attention has:
- Concept queries Q: [128, 512] (full dimensionality)
- Token keys K: [512, 512] but in a **rank-8 subspace** (projected from 8-dim)
- Token values V: [512, 512] same rank-8 subspace

The attention scores Q·K^T are determined by the projection of each 512-dim concept query onto the rank-8 token key subspace. Two concepts that differ only in the 504 dimensions orthogonal to the token subspace will have **identical attention patterns over tokens**.

With 8 attention heads (d_k = 64 per head), each head's attention operates in a 64-dim query space but the key vectors span only an 8-dim subspace within it. This limits each head to ~8 independent "modes" of attention.

**Total attention diversity:** 8 heads × 8 modes = ~64 distinguishable ways for concepts to attend to tokens. Compare to the standard case: 8 heads × 64 modes = ~512 distinguishable patterns.

This means concepts CANNOT maintain 512 independent position-specific attention patterns. They must share attention patterns and aggregate tokens by GROUPS, not individual positions. This is a genuine constraint on the hash function.

#### The information flow analysis

| Token dim | Encoder position resolution | Encoder attention rank | Hash function capability |
|---|---|---|---|
| 512 (current) | Perfect (512 unique positions) | Full rank-512 | Perfect position hashing |
| 64 | Good (64 dims for 512 positions) | Rank-64 | Adequate hashing, some grouping |
| 32 | Moderate (32 dims for 512 positions) | Rank-32 | Crude hashing, significant grouping |
| 16 | Limited (16 dims for 512 positions) | Rank-16 | Hashing very difficult, forced abstraction |
| 8 | Very limited (8 dims for 512 positions) | Rank-8 | Hashing practically impossible, concepts must aggregate by type |

At token_dim=8, the encoder genuinely cannot maintain position-specific concept-token associations. The 8 dimensions must encode BOTH word identity (~16 bits needed for 50K vocab) AND position (~9 bits for 512 positions). With only 8 float dimensions (~256 bits), this is possible in principle but the resulting representations will be dense and entangled, making attention patterns soft and position-unspecific.

#### The critical asymmetry: encoder constrained, decoder unconstrained

With low token_dim, the encoder builds concepts from a rank-limited token space (forced to aggregate by type/feature, not position). BUT the decoder's position embeddings are full 512-dim — it can still do position-specific concept queries.

This creates a mismatch:
- Encoder: "I built concepts that represent semantic groups, not specific positions"
- Decoder: "I want to query for the token at position 42"

The model must resolve this mismatch. Two possible outcomes:

1. **Decoder adapts to semantic queries:** The decoder learns to query concepts by MEANING ("what concept carries information about the verb in this region?") instead of by position. This produces semantic concepts. This is the optimistic outcome.

2. **Encoder builds a crude position hash despite constraints:** The encoder uses its limited rank to approximate position-specific hashing (less precise, using multiple dimensions to jointly encode position). This produces lower-quality hash + higher reconstruction loss. This is the pessimistic outcome.

#### The fundamental tension (same problem, new angle)

With self-reconstruction, the decoder NEEDS precise token identity at each position. If the encoder can't provide it (because token_dim is too low for precise hashing), reconstruction quality drops. The model is caught between:
- "Can't hash precisely" (encoder side, low token_dim)
- "Needs precise information" (decoder side, self-reconstruction demands exact tokens)

This means **low token_dim with self-reconstruction will increase training loss but may not produce semantic concepts** — it might just produce a noisy hash function that's bad at both hashing AND semantics.

#### Dimension Inversion + prefix generation = synergistic

With prefix generation (encoder encodes prefix, decoder generates suffix):
- The encoder doesn't need to hash prefix token positions (the decoder generates DIFFERENT tokens)
- Low token_dim prevents the encoder from memorizing prefix token details
- Concepts are forced to capture semantic gist (topic, entities, narrative direction)
- The decoder's full-dim position embeddings help it organize the generated suffix

In this setting, low token_dim HELPS because:
1. It prevents the encoder from cheating (storing prefix tokens for potential leakage)
2. It forces abstract/semantic concept representations
3. The decoder doesn't need exact tokens from concepts — it generates new text

**With self-reconstruction:** low token_dim is antagonistic (makes the required task harder without changing what's optimal)
**With prefix generation:** low token_dim is synergistic (prevents the wrong solution while enabling the right one)

#### Practical recommendations for token_dim sweep

If combined with prefix generation:

| token_dim | Expected effect | Worth testing? |
|---|---|---|
| 8 | Very aggressive. Encoder can barely distinguish tokens. Might undershoot. | Yes (lower bound) |
| 16 | Strong constraint. Forces semantic grouping. Good starting point. | Yes (recommended) |
| 32 | Moderate constraint. Allows some position specificity. Balanced. | Yes (primary) |
| 64 | Mild constraint. Similar to standard but with memory savings. | Yes (upper bound) |
| 128 | Minimal constraint. Unlikely to change behavior much. | Skip |
| 512 (current) | No constraint. Hash function is easy. | Baseline |

The sweetest spot is likely **token_dim=32**: enough capacity for meaningful word representations (32 dims > 16 bits needed for vocab), but not enough for precise position-specific hashing of 512 tokens. This forces concepts to aggregate by semantic features rather than individual positions.

Parameter savings are also substantial:

| token_dim | Token embed params | Savings vs 512 | Total model savings |
|---|---|---|---|
| 512 | 25.6M | 0% | 0% |
| 64 | 3.2M | 88% | ~25% |
| 32 | 1.6M | 94% | ~28% |
| 16 | 0.8M | 97% | ~29% |
| 8 | 0.4M | 98% | ~30% |

With token_dim=32, the model saves ~24M parameters (28% of total), which can be reallocated to more encoder layers, more concepts, or larger intermediate size.

---

### 15.3 Combined Recommendation: BiXT + Dimension Inversion + Prefix Generation

The strongest configuration combines all three insights:

```
Architecture:
  Encoder: BiXT (bidirectional cross-attention), L=6, token_dim=32, concept_dim=512
  Decoder: Cross-attention only (current), D=2, full position embeddings

Training objective:
  Prefix generation with ELBO-weighted diffusion on suffix

Regularization:
  VICReg + t_regs_mst (weight 0.02, warmup 2000 steps)
```

Why this combination works:
1. **Prefix generation** changes WHAT concepts learn (semantics, not hashes)
2. **BiXT** improves HOW concepts are built (from contextualized tokens, not raw embeddings)
3. **Low token_dim** prevents the encoder from cheating (can't memorize precise token details)
4. **VICReg + t_regs_mst** ensures geometric health (all dimensions used, concepts spread apart)

Each component addresses a different failure mode:

| Failure mode | Fix |
|---|---|
| Hash function learning | Prefix generation (different target) |
| Static token embeddings | BiXT (tokens update from concepts) |
| Bottleneck too loose | Low token_dim (limits information throughput) |
| Dimensional collapse | VICReg + t_regs_mst (geometric regularization) |
| AdaLN-Zero gate delay | (accepted, not worth fixing independently) |

---

*Updated: 2026-03-01 (added sections 15.1-15.3: BiXT analysis, Dimension Inversion analysis, combined recommendation)*
*Created: 2026-03-01*
*Methodology: First-principles analysis following research-methodology skill (information theory, gradient flow, "why-why-why" chain).*
*Replaces: Prior diagnosis assumed depth and ELBO fixes would work. They don't. The objective is the problem.*
*Next action: Implement prefix generation (TODO 13) with BiXT + token_dim=32, then sweep token_dim.*
