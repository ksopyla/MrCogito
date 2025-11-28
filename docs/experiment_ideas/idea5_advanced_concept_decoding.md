# Idea 5: Advanced Concept Decoding (Perceiver IO, Diffusion, MoE)
**Date:** November 27, 2025
**Target**: Break the 81.2% F1 ceiling on MRPC (GLUE) and scale to ARC-AGI levels of abstraction.

## 1. Problem Analysis: The Static Bottleneck
**Current Status:**
*   `weighted_mlm_H512L2C128` (23M params) achieves **81.2% F1** on MRPC after 20 epochs.
*   Training for 100 epochs yields **0% improvement**.
*   Current Decoder: `Output[pos] = WeightedSum(Concepts)` where weights are a **static** parameter matrix `[max_seq_len, concept_num]`.

**Diagnosis:**
The model learns a fixed, global routing template (e.g., "Position 5 always mixes Concept 10 and Concept 50") regardless of the content. It cannot dynamically "query" the concept space based on the input context. This limits the model's ability to perform complex reasoning or handle variable-length dependencies effectively.

---

## 2. Theoretical Frameworks & Solutions

### A. Perceiver IO Decoding (The "Dynamic Query" Paradigm)
**Concept:** Replace static weights with a Cross-Attention mechanism.
*   **Mechanism:** `Output[pos] = Attention(Query=PosEmbed[pos], Key=Concepts, Value=Concepts)`
*   **Benefit:** The decoding becomes dynamic. The model can "ask" the concept workspace for specific features required at a certain position.
*   **Extension:** Inject content into queries (`Query = PosEmbed + UnmaskedTokens`) to allow the decoder to actively search for missing information based on available context.

### B. Mixture-of-Experts (MoE) in Concept Space
**Concept:** Treat the 128 concepts as a specialized "Toolbox" rather than a flat array.
*   **Mechanism:** Instead of dense attention (Softmax over all concepts), use **Top-K Gating** (e.g., Router selects top 4 concepts for this token).
*   **Connection to ARC:** Allows specialization (e.g., "Color Expert", "Geometry Expert", "Movement Expert") without interference.

### C. Recursive & Diffusion Refinement
**Concept:** Reasoning is an iterative process, not a single forward pass.
*   **Recursive:** Re-use the `ConceptEncoderLayer` $K$ times on the concept representations before decoding. Deep reasoning without extra parameters.
*   **Diffusion (LLaDA):** Instead of predicting all masks at once, iteratively "denoise" the concept state.
    *   *Pass 1:* Rough concept sketch.
    *   *Pass 2:* Refine concepts using Self-Attention.
    *   *Pass 3:* Decode to tokens.

---

## 3. Proposed Experiment: "The Dynamic Decoder"

**Goal:** Validate that dynamic decoding outperforms static weighting on small datasets (MRPC).

### Phase 1: Perceiver Decoder (Immediate Action)
Implement `ConceptEncoderForMaskedLMPerceiver` to replace `ConceptEncoderForMaskedLMWeighted`.

**Architecture Changes:**
1.  **Remove:** `self.concept_weights` (Static Parameter).
2.  **Add:**
    *   `self.decoder_query_embeddings` (Learnable Position Embeddings).
    *   `self.decoder_cross_attn` (Standard MultiHeadAttention).
3.  **Forward Pass:**
    *   Encode: `Token -> Concepts` (Existing).
    *   Decode: `Query(Pos) -> CrossAttn(Key=Concepts, Val=Concepts) -> Logits`.

**Success Criteria:**
*   Surpass **81.2% F1** on MRPC (beating the static baseline).
*   Maintain parameter efficiency (~25M params).

### Phase 2: Recursive Refinement
*   Add a loop in the `ConceptEncoder` forward pass to apply the self-attention layer multiple times (e.g., 3 iterations) before returning the final concept state.

### Phase 3: Diffusion Training
*   Implement LLaDA-style training objective: Random masking ratio $t \in [0, 1]$ and predict the masked tokens based on the current partial concept state.

---

## 4. Action Plan
1.  **Implementation**: Create `ConceptEncoderForMaskedLMPerceiver` in `nn/concept_encoder.py`.
2.  **Verification**: Unit test the new decoder forward pass.
3.  **Experiment**: Train on WikiText-103 (Micro config) and evaluate on MRPC.

