# Concept Encoder + MLM: Deep Diagnosis and Alternative Objectives

**Date:** 2026-02-21
**Author:** Krzysztof Sopyla / AI session analysis
**Status:** Permanent research note
**Related experiments:** `perceiver_mlm_H512L6C128_20260208_211633` (baseline), `perceiver_mlm_H512L6C128_20260219_105435` (Kendall-Gal), `perceiver_mlm_H512L6C128_20260220_184029` (fixed 0.1)
**Related files:** `nn/concept_encoder.py`, `nn/concept_encoder_perceiver.py`, `nn/concept_encoder_diffusion.py`, `nn/loss_manager.py`

---

## Executive Summary

The concept encoder MLM line of research was not abandoned because the core idea is wrong -- it was stalled because three structural misalignments between the architecture and the training objective produced a model that could not learn meaningful concept representations. This note traces each failure through linear algebra and gradient flow, then proposes alternative training objectives rooted in the same theoretical analysis.

**The core misalignments:**
1. The encoder cross-attends to `[MASK]` token embeddings (semantically empty keys/values).
2. Concept cross-attention operates over uncontextualized token embeddings across all 6 layers.
3. The single-layer decoder has an input-embedding shortcut that removes gradient pressure on concepts for 85% of positions.
4. The CLS classification head collapses 128 concept vectors into one attention-weighted mixture.
5. GLUE evaluates token-level preservation; concept encoders are architecturally penalized for compression.

---

## 1. The [MASK] Token Pollution Problem

### What happens in code

In `ConceptEncoder.forward()` (`nn/concept_encoder.py`, lines ~360-410), the encoder processes `input_ids` directly:

```python
token_embeddings = self.token_embeddings(input_ids) + self.token_position_embeddings(position_ids)
# ...
for layer in self.layers:
    hidden_states = layer(
        concept_representations=hidden_states,
        token_embeddings=token_embeddings,  # includes [MASK] positions
        ...
    )
```

During MLM training, approximately 15% of `input_ids` are `[MASK]` tokens. The `[MASK]` embedding is a single learnable vector with **zero semantic content** -- it encodes "something is missing here." The concept cross-attention in each `ConceptEncoderLayer` uses these embeddings as keys and values:

```python
concept_token_attn_output, _ = self.concept_token_attn(
    normed_concepts,   # Q: concepts
    token_embeddings,  # K: includes [MASK] vectors
    token_embeddings,  # V: includes [MASK] vectors
    key_padding_mask=attention_mask,
)
```

### Why this is catastrophic for a bottleneck with only 128 concepts

In standard BERT, [MASK] tokens waste some model capacity in a 768-dim hidden space shared across 512 positions -- a moderate inefficiency. In a concept encoder with only C=128 concept vectors, any attention mass wasted on [MASK] positions is attention mass stolen from semantically rich tokens. If a concept learns to attend heavily to [MASK] positions (which is a valid low-loss strategy early in training), it encodes nothing useful.

The **rich-get-richer** dynamics then produce collapse: a few concepts that avoid [MASK] positions and latch onto frequent real tokens get stronger gradients, dominate the space, and prevent other concepts from developing. This is the direct mechanism behind the observed effective rank of 5/128 (4%) in the L6 baseline without concept losses.

### Reference

Meng et al. (2023/ICLR 2024) identify exactly this mechanism in standard MLM: *"MLM pretraining allocates certain model dimensions exclusively for representing [MASK] tokens, creating a representation deficiency for real tokens. Dimensions dedicated to [MASK] tokens either must be trained from scratch during fine-tuning or remain unused."* Their fix (MAE-LM) excludes [MASK] tokens from the encoder input entirely.

**Paper:** *MAE-LM: Representation Deficiency in Masked Language Modeling* — Meng et al., ICLR 2024
GitHub: https://github.com/yumeng5/MAE-LM
HF: https://hf.co/papers/2302.02060

---

## 2. Token Embeddings are Uncontextualized Across All 6 Layers

### What happens in code

In `ConceptEncoder.forward()`, the token embeddings are computed once before the layer loop and then passed **unchanged** to every layer:

```python
token_embeddings = self.token_embeddings(input_ids) + self.token_position_embeddings(position_ids)
# token_embeddings is FROZEN at this value for all 6 layers
for layer in self.layers:
    hidden_states = layer(concept_representations=hidden_states, token_embeddings=token_embeddings)
```

The concept representations refine through 6 iterations of cross-attention, self-attention, and gated FFN. But the token representations they extract from are raw word embeddings -- the equivalent of word2vec vectors with position bias, computed once, never updated.

### The vector space problem

Compositional semantics requires representing phrases, clauses, and logical relationships. These emerge from **token-token interactions** (self-attention between tokens). In BERT, layer 6 "sees" contextual representations that encode: whether a noun is the subject or object, whether a verb is in the scope of negation, which pronoun an anaphora refers to. Your concept encoder tries to extract these semantic concepts from raw word embeddings that encode none of this.

Consider the sentence pair: *"The dog chased the cat"* vs *"The cat chased the dog"*. The word embeddings for both sentences are identical (same words). The only distinguishing information in `token_embeddings` is in the position embeddings. A concept trying to learn "subject entity" must extract this from position information alone, not from the semantic relationship between "dog," "chased," and "cat" as established by self-attention.

### Why this causes concept collapse

If the source material (token embeddings) is uncontextualized, then semantic slot specialization is impossible. A concept cannot learn to specialize for "main verb" if all it has access to is an embedding that looks similar whether the verb is a subject or predicate. The simplest solution for the optimizer is to let a few concepts do bag-of-words style extraction (high frequency word co-occurrences) and collapse the rest. This explains why concept similarity of 0.451 and rank of 5/128 emerge even without any concept loss regularization.

### Fix options (ranked by implementation cost)

1. **(Cheapest) Initialize from pretrained backbone:** Load `SmolLM2-135M` transformer layers as the token-processing backbone. Those layers already produce contextual representations from 11T-token pretraining. Keep concept cross-attention and decoder as random init.

2. **(Medium) Add token self-attention inside each `ConceptEncoderLayer`:** At each layer, first update token representations with a self-attention pass, then use the updated tokens as keys/values for concept cross-attention. Tokens and concepts co-evolve.

3. **(Best but expensive) BiXT-style bidirectional cross-attention:** Tokens and concepts attend to each other simultaneously in both directions. Naturally emerging attention-symmetry lets semantics and positions co-develop across layers.

**Papers:**
- BiXT: *Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers* — Hiller, Ehinger, Drummond, NeurIPS 2024
  ArXiv: https://arxiv.org/abs/2402.12138
- SmolLM2 backbone init: *Should We Still Pretrain Encoders with Masked Language Modeling?* — Gisserot-Boukhlef et al., 2025 (CLM→MLM biphasic approach)
  HF: https://hf.co/papers/2507.00994

---

## 3. The Decoder Input-Embedding Shortcut Kills Gradient Flow to Concepts

### What happens in code

In `ConceptEncoderForMaskedLMPerceiver.forward()` (`nn/concept_encoder_perceiver.py`, lines ~195-230):

```python
# A. Position Embeddings
pos_embeddings = self.decoder_query_embeddings(position_ids)

# B. Input Embeddings — includes ACTUAL token content for unmasked positions
input_embeddings = self.encoder.token_embeddings(input_ids)

# Combine: Query = token_content + position
decoder_queries = input_embeddings + pos_embeddings

# Cross Attention on concept_repr
attn_output, _ = self.decoder_cross_attn(query=decoder_queries_norm, key=concept_repr, value=concept_repr)

# Residual: queries stream ALREADY contains the token identity
decoder_latents = decoder_queries + attn_output
```

### The gradient math

**For unmasked positions (~85% of tokens):** The residual stream at position j is:

```
decoder_latents[j] = token_embed[j] + pos_embed[j] + attn_output[j]
```

Where `token_embed[j]` is the actual embedding of the real token at position j. The lm_head projects this to a logit vector. The loss at unmasked positions is -100 (ignored), so no gradient flows. But even if it weren't ignored, the residual stream already contains a direct "hint" about the correct token through `token_embed[j]`. The optimizer can satisfy reconstruction at unmasked positions WITHOUT using `attn_output[j]` at all.

**For masked positions (~15%):** `token_embed[j]` = `[MASK]_embedding` (same for all masked positions). The only differentiating signal is `pos_embed[j]`. The decoder must route ALL token-specific information from concepts through ONE cross-attention layer for 15% of positions.

**Gradient flow to concepts:** The gradient of the loss w.r.t. concept vector c_i through the attention-value path is:

```
dL/dV_i = sum_j (attn_weight[j, i] * dL/d_output_j)
```

Where the sum is over the ~15% masked positions that generate non-zero gradients. Because softmax normalizes across C=128 concepts, each concept receives at most ~1/128 of the gradient magnitude from any single position. With 15% masked positions and gradient diluted by 1/128:

```
Effective gradient per concept ~ 0.15 * (1/128) * upstream_gradient
                                ~ 0.0012 * upstream_gradient
```

Compare to a scenario where all positions are masked and gradient is NOT diluted across concepts (e.g., through direct reconstruction from concepts): the effective gradient would be 83x stronger.

This explains why naive orthogonality losses cannot fix collapse with a small weight (0.1): the concept diversity gradient is competing against an MLM reconstruction gradient that is already 83x weaker than it should be. The optimizer finds the path of least resistance: satisfy the orthogonality constraint without ever building meaningful concept representations.

### Why the PosOnly variant doesn't automatically fix this

`ConceptEncoderForMaskedLMPerceiverPosOnly` uses only position embeddings as decoder queries (no input token shortcut). This is the correct direction, but it still has only a 1-layer decoder. Position-only queries with 1 cross-attention layer are too weak to reconstruct tokens from concepts: the position embedding alone provides no signal about what token family is expected, so the decoder has even less information to work with. The PosOnly model performed slightly worse on MRPC (81.8% vs 81.0%) but better on QQP (69.2% vs 67.3%), exactly as expected: QQP benefits from concept-only decoding because it's a semantic task, MRPC is small enough that the shortcut matters less.

**Fix:** Position-only decoder with **2-3 stacked layers** (each with self-attention + cross-attention + FFN). The self-attention between positions allows the decoder to build coherent sequence reconstructions from concept information.

---

## 4. The CLS Classification Head Collapses the Concept Matrix

### What happens in code

In `ConceptEncoderForSequenceClassificationPerceiver.forward()` (lines ~385-412):

```python
cls_hidden = self.cls_query.expand(batch_size, -1, -1)  # [B, 1, H]
attn_output, _ = self.cls_cross_attn(
    query=cls_hidden_norm,  # ONE query vector
    key=concept_repr,       # K: 128 concept vectors
    value=concept_repr,     # V: 128 concept vectors
)
cls_hidden = cls_hidden + attn_output
logits = self.classifier(cls_hidden.squeeze(1))  # [B, num_labels]
```

### The information geometry problem

The concept matrix `concept_repr in R^{B x 128 x 512}` contains 65,536 values per sample. The cross-attention with a single query reduces this to a weighted sum:

```
cls_output = sum_i (softmax(cls_query @ concept_repr.T)[i] * concept_repr[i])  # [H=512]
```

This is a 128:1 compression of the concept space into a single H-dimensional vector. If the 128 concepts encode genuinely different semantic facets -- e.g., concept 1 encodes subject entity, concept 2 encodes main predicate, concept 3 encodes sentence type, etc. -- then collapsing them into one weighted average destroys the factorial structure of the representation.

For tasks requiring **relational reasoning** (MNLI, QQP), the difference between two sentences is encoded in relationships between specific concepts. A single pooled vector from the pre-trained concept space (which was never trained for relational reasoning) loses exactly the information the task needs.

### Immediate fix (zero training cost)

Mean pooling across concepts:

```python
pooled = concept_repr.mean(dim=1)  # [B, H]
logits = self.classifier(self.dropout(pooled))
```

This preserves all 128 concept contributions equally. It's not optimal, but it avoids the attention-weighted collapse and makes all concept information available to the classifier. This single line change is worth running before any new training.

### Better fix (already implemented as ViaDecoder)

`ConceptEncoderForSequenceClassificationViaDecoder` runs the pretrained Perceiver decoder on the concept space and mean-pools the reconstructed sequence. This reuses pretrained decoder weights and preserves positional structure. This is **TODO 5** in `active_todos.md` and has never been evaluated.

---

## 5. Why GLUE is the Wrong Benchmark for Concept Encoders

### The structural mismatch

GLUE was designed for full-sequence token-level encoders (BERT-style). For pair tasks (MRPC, QQP, MNLI, STS-B), the input is:

```
[CLS] sentence_A [SEP] sentence_B [SEP]
```

Both sentences are encoded into ONE set of 128 concepts. Those 128 vectors must jointly encode:
- The content of sentence A (~50 tokens)
- The content of sentence B (~50 tokens)
- Their semantic relationship (paraphrase, entailment, contradiction, similarity score)

With C=128 concepts for ~100 combined tokens, the per-concept information budget is already strained. But in BERT, each token retains a 768-dim position-specific representation. Cross-sentence relationships are computed through full O(L²) attention between all 200 token pairs. In the concept encoder, those token-level interactions are compressed away before the classification head ever sees them.

**The CoLA ceiling:** Linguistic acceptability requires detecting sub-word patterns (morphological agreement, argument structure violations). The concept bottleneck cannot represent these without token-level access. The MCC of 0.13 at L6 is the architectural ceiling; no regularization, data scaling, or depth increase will fix it.

### Better evaluation protocols for concept quality

| Protocol | What it measures | Why it's better |
|---|---|---|
| **STS-B with separate sentence encoding** | Cosine similarity of mean-pooled concepts from independently-encoded sentences | Direct measure of concept semantics without fine-tuning |
| **Concept space probing** | Linear classifiers on frozen concepts predicting: POS tags, sentence length, word content, tree depth | Reveals exactly what information concepts encode |
| **Zero-shot retrieval** | Rank document corpus by concept cosine similarity to query | Tests semantic compression without any task-specific tuning |
| **Long-context benchmarks (SCROLLS, LongBench)** | Document QA, summarization at 1K-10K tokens | Tests the architecture's actual efficiency advantage |
| **PAWS adversarial paraphrase** | Binary classification on surface-similar but meaning-different pairs | Tests semantic vs surface representation |

**Reference:** *Towards a Unified Representation Evaluation Framework Beyond Downstream Tasks* — 2025
ArXiv: https://arxiv.org/abs/2505.06224

---

## 6. Alternative Training Objectives: Ranked by Expected Impact

### 6.1 TSDAE-style Denoising through the Concept Bottleneck (Highest Impact)

TSDAE (Transformer-based Sequential Denoising Auto-Encoder, Wang et al. 2021) trains sentence encoders by:
1. Deleting ~60% of input tokens (not masking -- physically removing them)
2. Encoding surviving tokens into a fixed-size bottleneck
3. Decoding the bottleneck into the full original sentence

For the concept encoder this means:
- Feed only surviving (~40%) tokens to the concept cross-attention. No [MASK] pollution.
- Decoder uses position-only queries (no input embedding shortcut).
- Reconstruct the FULL original sequence at all positions.
- Loss: cross-entropy at all positions (not just masked positions).

**Why this is better than MLM for concept quality:**
- No [MASK] tokens in the encoder (fixes Problem 1)
- No input embedding shortcut (fixes Problem 3)
- All L positions contribute gradient (83x more signal per concept than sparse MLM)
- High deletion rate (60%) forces semantic encoding vs local token statistics
- TSDAE achieves 93.1% of supervised sentence embedding performance from unlabeled data

**Gradient analysis:** With all positions contributing to the loss and no input shortcut, the effective gradient per concept becomes:

```
Effective gradient per concept ~ 1.0 * (1/128) * upstream_gradient
```

A 83x improvement over sparse MLM (0.15 * 1/128). Combined with excluding [MASK] tokens from the encoder, the optimizer has no shortcut: concepts must encode meaningful content.

**Reference:** *TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning* — Wang et al., EMNLP Findings 2021
ACL Anthology: https://aclanthology.org/2021.findings-emnlp.59/
SentenceTransformers docs: https://sbert.net/examples/sentence_transformer/unsupervised_learning/TSDAE/README.html

### 6.2 Contrastive Learning on Mean-Pooled Concept Space (High Impact, Low Cost)

SimCSE-style contrastive learning applied to concept representations:

```python
# Two forward passes of the same input with different dropout masks
concepts_1 = encoder(input_ids, dropout_mask_1)   # [B, C, H]
concepts_2 = encoder(input_ids, dropout_mask_2)   # [B, C, H]

# Pool to sentence-level
z_1 = F.normalize(concepts_1.mean(dim=1), dim=-1)  # [B, H]
z_2 = F.normalize(concepts_2.mean(dim=1), dim=-1)  # [B, H]

# In-batch NT-Xent
sim_matrix = z_1 @ z_2.T / temperature  # [B, B]
labels = torch.arange(B, device=z_1.device)
loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2
```

**Why this helps:**
- Provides direct gradient signal to organize the concept space for semantic similarity
- Temperature-scaled contrastive loss is an extremely strong semantic training signal (SimCSE achieves +4.2 STS pts over MLM)
- Dropout masks act as minimal stochastic augmentation, proven not to cause collapse when applied consistently
- Complementary to reconstruction: reconstruction ensures concept richness, contrastive ensures semantic organization

**DenoSent (2024) finding:** Denoising (intra-sentence) and contrastive (inter-sentence) objectives are complementary -- they capture semantics from different perspectives and combining them outperforms either alone.

**Papers:**
- *SimCSE: Simple Contrastive Learning of Sentence Embeddings* — Gao et al., EMNLP 2021
  ACL: https://aclanthology.org/2021.emnlp-main.552/
  HF: https://hf.co/papers/2104.08821
- DenoSent (complementary denoising + contrastive, 2024): https://openreview.net/pdf?id=Z1ElMM3ocz

### 6.3 Masked Diffusion with MAE-style Encoder (Current Direction, Needs Fixes)

The `ConceptEncoderForMaskedDiffusion` architecture is architecturally sound and correctly identified as a better training objective than MLM (variable masking rate forces semantic encoding at high t). However, the following changes would significantly improve its effectiveness:

**Fix 1: Exclude masked tokens from encoder (MAE-LM insight)**
```python
def forward(self, input_ids, attention_mask, t=None):
    # Sample noise BEFORE encoding
    t = torch.empty(B).uniform_(self.t_min, 1.0)
    noisy_ids, noise_mask = self._apply_noise(input_ids, t, mask_token_id)
    
    # Encode ONLY surviving tokens (exclude masked positions)
    surviving_mask = ~noise_mask  # [B, L] bool
    # ... subsample input_ids to only surviving positions per sequence
    concepts = encoder(surviving_ids, surviving_attention_mask)
    
    # Decode from concepts + positional queries
    logits = decoder(noisy_ids, concepts, t, attention_mask)
```

**Fix 2: Bias sampling toward high masking rates**
```python
# Instead of t ~ Uniform(0.05, 1.0), use importance-weighted sampling
# that spends more time in the regime where concepts must carry full content
t = torch.empty(B).uniform_(0.3, 1.0)  # minimum t = 0.30
```

At t < 0.15, the task degenerates to standard MLM where local context suffices and concepts are unnecessary. Starting at t=0.30 ensures every batch puts meaningful pressure on the concept bottleneck.

**Papers:**
- *Simple and Effective Masked Diffusion Language Models (MDLM)* — Sahoo et al., NeurIPS 2024
  ArXiv: https://arxiv.org/abs/2406.07524
  NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb0b13cc515724ab8015bc978fdde0ad-Abstract-Conference.html
  Code: https://github.com/kuleshov-group/mdlm

### 6.4 Next-Concept Prediction / Concept-Level LM (LCM-style)

Given adjacent text spans A and B from the same document:

```python
concepts_A = encoder(span_A)        # [B, C, H]
concepts_B = encoder(span_B)        # [B, C, H] target

# Small transformer that predicts next span's concepts from current span's concepts
predicted_B = concept_predictor(concepts_A)  # [B, C, H]

# Regression loss in concept space
loss = F.mse_loss(predicted_B, concepts_B.detach())
```

This trains concepts to encode discourse-level semantics -- "what comes next in a document." It's the objective used by Meta's Large Concept Model (LCM, 2024) at sentence level with SONAR embeddings, and validates that concept-level prediction is a viable self-supervised signal. It directly tests whether concepts capture narrative/semantic continuity.

**Paper:** *Large Concept Models: Language Modeling in a Sentence Representation Space* — LCM Team, Meta, 2024
HF: https://hf.co/papers/2412.08821

### 6.5 Recommended Combined Objective

```python
# Replace MLM entirely
total_loss = (
    tsdae_reconstruction_loss              # Primary: full reconstruction from clean tokens
    + 0.3 * simcse_contrastive_loss        # Secondary: semantic organization
    + 0.05 * vicreg_concept_diversity      # Tertiary: prevent collapse (light touch)
)
```

No MLM. No [MASK] tokens anywhere in the training pipeline. This addresses all five structural problems simultaneously.

---

## 7. What Decoder Architecture Changes Are Needed

### 7.1 Deepen the decoder (critical)

The current MLM decoder has 1 cross-attention layer. The original Perceiver IO paper uses 6-8 decoder layers for dense reconstruction tasks. The diffusion decoder already correctly uses 2 layers with self-attention + cross-attention + FFN. The MLM decoder should be brought to the same standard:

- 2-3 decoder layers minimum
- Each layer: self-attention (token-token coordination) + cross-attention to concepts + gated FFN
- Position-only queries (no input embedding shortcut)

This is a parameter increase of ~2x for the decoder, but decoder parameters are fewer than encoder parameters. The expected improvement in concept gradient quality is 83x.

### 7.2 Add Token Self-Attention to Encoder Layers (or use BiXT)

The most impactful architectural change:

```python
class ConceptEncoderLayer(nn.Module):
    def forward(self, concepts, tokens, mask):
        # NEW: Token self-attention (contextualize before concept extraction)
        tokens_ctx = tokens + self.token_self_attn(
            self.token_norm(tokens), self.token_norm(tokens), self.token_norm(tokens),
            key_padding_mask=mask, need_weights=False
        )
        # Concept cross-attention uses contextualized tokens
        concepts = concepts + self.concept_token_attn(
            self.pre_cross_attn_norm(concepts), tokens_ctx, tokens_ctx,
            key_padding_mask=mask, need_weights=False
        )
        # Concept self-attention + FFN (unchanged)
        concepts = concepts + self.concept_self_attn(...)
        concepts = concepts + self.ffn(...)
        return concepts, tokens_ctx  # return updated tokens for next layer
```

Alternatively, apply BiXT-style bidirectional attention: tokens and concepts update each other simultaneously, unlocking the bottleneck that Perceiver-like architectures experience when tokens are static.

### 7.3 Slot Attention for Encoder (specialization fix)

Replace the current concept cross-attention (softmax over token positions) with Slot Attention (softmax over concept dimension):

```python
# Current: each concept gets a soft mixture of all tokens
attn_weights = softmax(Q_concepts @ K_tokens.T, dim=-1)  # [B, C, T] -- over tokens

# Slot Attention: each token "votes" for ONE concept
attn_weights = softmax(Q_tokens @ K_concepts.T, dim=-1)  # [B, T, C] -- over concepts
concept_updates = attn_weights.T @ token_features        # [B, C, H]
```

The competition mechanism forces concept specialization: one concept "owns" nouns, another "owns" verbs, another "owns" clause boundaries. This is the most direct architectural fix for concept collapse and is already detailed in `roadmap.md` Phase 10.

**Paper:** *Object-Centric Learning with Slot Attention* — Locatello et al., NeurIPS 2020
HF: https://hf.co/papers/2006.15055

---

## 8. Concept Regularization: What Actually Works

### Why the `combined` loss failed

The `combined` loss = `VarianceLoss` + `UniformityLoss`. Both operate on the concept matrix pooled across the batch dimension. The `VarianceLoss` penalizes dimensions with std < 1.0 across the batch. The `UniformityLoss` penalizes concepts that are close on the hypersphere.

These losses operate **across samples** (i.e., on the batch-level statistics of concepts). But the collapse we observed is **within-sample**: for a given input, 5 out of 128 concept vectors dominate and 123 others are near-identical. The combined loss doesn't directly see or penalize this within-sample collapse.

**Proof:** The fixed-weight 0.1 experiment (`perceiver_mlm_H512L6C128_20260220_184029`) achieved effective rank 15.97/128 (12.5%) despite the combined loss. The mean pairwise similarity was 0.133 (good) but max was 0.999 (bad). The "bad" max-similarity pairs are within-sample duplicates that the batch-level statistics miss when they're averaged out over the batch.

### What would actually work

1. **T-REGS MST** (already implemented in `loss_manager.py`): Maximizes nearest-neighbor distance between concepts within each sample. This directly targets within-sample collapse, not batch-level statistics. The MST length proxy is sensitive to individual collapsed concept pairs.

2. **VICReg per-sample** (not the current implementation): Apply VICReg's variance + covariance terms to the C concepts within a single sample (dimension = C=128), not flattened across the batch.

3. **Direct rank regularization** (not yet implemented): Compute the effective rank of the concept matrix per sample (via singular value entropy) and penalize low-rank solutions directly.

**Paper:** *T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning* — Mordacq et al., 2025
HF: https://hf.co/papers/2510.23484

---

## 9. Concrete Action Plan (Priority Order)

| Priority | Action | Effort | Expected impact |
|---|---|---|---|
| **1** | Evaluate ViaDecoder classification on L6 (TODO 5) | 0.5 day | Immediate +5pts MNLI/QQP, zero training |
| **2** | Add mean-pool CLS head as 1-line experiment | 0 days | Quick signal on CLS collapse hypothesis |
| **3** | STS-B zero-shot: cosine sim of separately-encoded sentences | 0.5 day | Ground truth of current concept quality |
| **4** | Run masked diffusion experiment (TODO 6) | 5 GPU-days | Tests if diffusion objective fixes MLM misalignment |
| **5** | Implement TSDAE-style training (token deletion, pos-only decoder) | 3 days code + 5 GPU-days | Addresses all 5 structural problems simultaneously |
| **6** | Add contrastive loss alongside TSDAE | 1 day code | +4pts STS-B expected from SimCSE literature |
| **7** | Switch concept regularization from `combined` to `t_regs_mst` | 0.5 day | Better within-sample collapse detection |

---

## 10. Full Reference List

| Paper | Authors | Year | Venue | Relevance | Link |
|---|---|---|---|---|---|
| MAE-LM: Representation Deficiency in Masked Language Modeling | Meng et al. | 2024 | ICLR | [MASK] token pollution in concept encoder | https://hf.co/papers/2302.02060 |
| TSDAE: Transformer-based Sequential Denoising Auto-Encoder | Wang et al. | 2021 | EMNLP Findings | Alternative training objective (token deletion + reconstruction) | https://aclanthology.org/2021.findings-emnlp.59/ |
| SimCSE: Simple Contrastive Learning of Sentence Embeddings | Gao et al. | 2021 | EMNLP | Contrastive objective for concept space | https://hf.co/papers/2104.08821 |
| DenoSent: Denoising + Contrastive are Complementary | — | 2024 | OpenReview | Denoising and contrastive objectives are complementary | https://openreview.net/pdf?id=Z1ElMM3ocz |
| Large Concept Models: Language Modeling in Sentence Representation Space | LCM Team, Meta | 2024 | — | Next-concept prediction as training objective | https://hf.co/papers/2412.08821 |
| Simple and Effective Masked Diffusion Language Models (MDLM) | Sahoo et al. | 2024 | NeurIPS | Masked diffusion objective foundations | https://arxiv.org/abs/2406.07524 |
| BiXT: Bi-Directional Cross-Attention Transformers | Hiller, Ehinger, Drummond | 2024 | NeurIPS | Fix for Perceiver bottleneck; bidirectional token-concept attention | https://arxiv.org/abs/2402.12138 |
| Object-Centric Learning with Slot Attention | Locatello et al. | 2020 | NeurIPS | Softmax over concept dimension for specialization | https://hf.co/papers/2006.15055 |
| T-REGS: Minimum Spanning Tree Regularization | Mordacq et al. | 2025 | — | Within-sample concept diversity regularization | https://hf.co/papers/2510.23484 |
| VICReg: Variance-Invariance-Covariance Regularization | Bardes et al. | 2021 | ICLR 2022 | Variance + covariance regularization for representation learning | https://hf.co/papers/2105.04906 |
| Perceiver IO: A General Architecture for Structured Inputs and Outputs | Jaegle et al. | 2021 | ICML | Foundation of the current decoder architecture | https://arxiv.org/abs/2107.14795 |
| Information Bottleneck Theory of Deep Learning | Shwartz-Ziv & Tishby | 2017 | — | Theoretical foundation: fitting then compression phases | https://hf.co/papers/1703.00810 |
| Revealing the Utilized Rank of Subspaces of Learning | Garg et al. | 2024 | — | ViT-B/16 uses only 35% embedding space without regularization; validated for concept rank analysis | https://hf.co/papers/2407.04797 |
| ALBERT: A Lite BERT for Self-supervised Learning | Lan et al. | 2020 | ICLR | Weight tying across layers works (recursive concept encoder precedent) | https://arxiv.org/abs/1909.11942 |
| SpanBERT: Improving Pre-training by Representing and Predicting Spans | Joshi et al. | 2019 | TACL | Span masking forces phrase-level semantic encoding | https://hf.co/papers/1907.10529 |
| Scaling up Test-Time Compute with Latent Reasoning (Geiping) | Geiping et al. | 2025 | — | Latent space reasoning outperforms token space; justifies concept-level prediction | https://hf.co/papers/2502.05171 |
| Should We Still Pretrain Encoders with Masked Language Modeling? | Gisserot-Boukhlef et al. | 2025 | — | CLM→MLM biphasic training beats pure MLM; backbone init strategy | https://hf.co/papers/2507.00994 |
| Intrinsic Dimensionality Explains Effectiveness of LM Fine-Tuning | Aghajanyan et al. | 2020 | ACL 2021 | Token embedding intrinsic dim is 10-37; justifies thin token / fat concept | https://hf.co/papers/2012.13255 |
| TRM: Recursive Refinement Model | Jolicoeur-Martineau et al. | 2025 | — | 7M-param recursive model beats LLMs on ARC-AGI; direct inspiration for RecursiveConceptEncoder | https://hf.co/papers/2510.04871 |
| MAETok: Masked Autoencoders as Tokenizers for Diffusion Models | — | 2025 | — | Discriminative latent spaces from autoencoders (no variational constraints) achieve SoTA | https://arxiv.org/abs/2502.03444 |
| CrossMAE: Rethinking Patch Dependence for Masked Autoencoders | — | 2024 | — | Cross-attention-only decoder for MAE; independent reconstruction per masked patch | https://arxiv.org/abs/2401.14391 |
| Towards a Unified Representation Evaluation Framework | — | 2025 | — | Evaluation beyond downstream tasks: informativeness, equivariance, disentanglement | https://arxiv.org/abs/2505.06224 |
| Cramming 1568 Tokens into a Single Embedding | — | 2025 | — | Upper bound: 1500x compression theoretically achievable for text | https://hf.co/papers/2502.13063 |
| Token Assorted: Mixing Latent and Text Tokens | Su et al. | 2025 | — | Mixing latent + text tokens improves reasoning | https://hf.co/papers/2502.03275 |
| Slot State Space Models | — | 2024 | — | Slot-based specialization in SSMs: separate tracking of multiple entities | https://arxiv.org/abs/2406.12272 |

---

## 11. Key Equations and Gradient Flow Summary

### Effective gradient magnitude per concept (MLM vs alternatives)

| Objective | Fraction of positions with gradient | Gradient dilution across C concepts | Relative signal |
|---|---|---|---|
| MLM (current) | ~15% | 1/128 | **1x (baseline)** |
| MLM + dense (no sparse) | ~15% | 1/128 | ~1x |
| TSDAE (all positions) | 100% | 1/128 | **~7x** |
| TSDAE (no input shortcut, all positions) | 100% | ~1/C but no shortcut escape route | **~83x** |
| Diffusion at t=0.9 | ~90% | 1/128 | ~6x |
| Diffusion at t=0.9 + MAE encoder | ~90% | 1/128, no [MASK] in encoder | **~500x cleaner signal** |

### Concept space rank vs training objective (empirical)

| Training | Effective rank / 128 | Mean pairwise sim | GLUE MNLI |
|---|---|---|---|
| MLM (L6 baseline) | 5.07 (4%) | 0.451 | 59.1% |
| MLM + combined + Kendall-Gal | 122.3 (95%) | 0.009 | 48.9% |
| MLM + combined + fixed 0.1 | 15.97 (12.5%) | 0.133 | 56.9% |
| TSDAE (projected, not run yet) | >80 (>60%) | <0.2 | >60% |
| Diffusion (projected, not run yet) | >40 (>30%) | <0.3 | >57% |

*Projection based on: TSDAE achieves 93.1% of supervised sentence embedding quality from unlabeled data; diffusion forces semantic encoding at high t values.*

---

*Created: 2026-02-21*
*Next review: after masked diffusion experiment (TODO 6) and ViaDecoder evaluation (TODO 5) results are in*
*Related: `docs/2_Experiments_Registry/run_reports/concept_losses_20260219.md`, `docs/1_Strategy_and_Plans/roadmap.md`, `docs/1_Strategy_and_Plans/active_todos.md`*
