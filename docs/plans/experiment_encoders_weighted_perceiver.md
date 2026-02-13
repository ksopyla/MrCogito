# Concept Encoder Experiment Plan v2

**Updated: 2026-02-13** | Previous: 2026-01-17
**Goal:** Close the ~24pt GLUE gap to BERT-Base. Current best: perceiver_mlm L6 = 59.1% MNLI, 81.3% MRPC.

## Status Summary

| Step | Status | Key Finding |
|------|--------|-------------|
| Step 1: L2 baseline on Minipile | **Done** | weighted best on MRPC (82.2%), posonly best on most other tasks |
| Step 1b: Full GLUE eval L2 | **Done** | CoLA ~0 MCC, MNLI ~54%, near-random on inference tasks |
| Step 1c: Scale to L6, 40 epochs | **Done** | +3pts avg, MNLI 59.1%, CoLA still broken (0.13 MCC) |
| Step 1d: Sparse MLM decoding fix | **Done** | Fixed OOM from accelerate fp32 conversion on full logits |

**Diagnosis:** The concept bottleneck is the binding constraint. MLM loss dropped 37% (L2->L6) but downstream only +3pts. Three root causes: (1) too aggressive compression, (2) position information destroyed, (3) insufficient pretraining data.

---

## Phase 1: Engineering Foundation (1-2 days)

Fix performance bottlenecks before running expensive experiments. Expected **2-4x training speedup**.

### 1.1 Enable Flash Attention / SDPA
Set `need_weights=False` on ALL `nn.MultiheadAttention` calls in:
- `nn/concept_encoder.py` (cross-attn line 163, self-attn line 179)
- `nn/concept_encoder_perceiver.py` (decoder cross-attn lines 200, 560; cls cross-attn line 378)
- `nn/concept_encoder_weighted.py` (no attention in decoder, skip)

This is the single most impactful change -- enables SDPA/Flash Attention automatically on PyTorch 2.x.

### 1.2 Enable torch.compile
Add to training script:
```bash
--torch_compile True
--torch_compile_backend "inductor"
```
Test for graph breaks first. Expected 1.5-2x additional speedup.

### 1.3 Fused Cross-Entropy (Liger Kernel)
Replace `CrossEntropyLoss` with `LigerCrossEntropyLoss` in sparse MLM decoding paths. Expected +20% throughput, -60% memory on the loss computation.

### 1.4 Data Loading
Set `dataloader_num_workers=4`, `dataloader_prefetch_factor=2` in training scripts.

---

## Phase 2: Scale Pretraining Data (3-5 days training)

The model sees Minipile 40 times. BERT saw BookCorpus+Wikipedia (~16GB) multiple times with ~3.3B words. We need 10-100x more data.

### 2.1 Dataset: OpenWebText + Wikipedia (recommended first)
- `Skylion007/openwebtext` (~8M samples, 13.5GB) + `wikimedia/wikipedia` 20231101.en (~6.7M, 20GB)
- Total: ~15M samples, ~33GB = 10x Minipile
- Classic BERT-style mix: web text + encyclopedia
- Fits in RAM on Polonez (256GB)

### 2.2 Dataset: FineWeb-Edu sample-10BT (if 2.1 insufficient)
- `HuggingFaceFW/fineweb-edu` sample-10BT config (~10M samples, 10B tokens)
- SoTA data quality (Llama-3-70B filtered)
- Use streaming for memory efficiency

### 2.3 Training Protocol
- Same architecture: perceiver_mlm H512L6C128 (best from current experiments)
- 10-20 epochs on OpenWebText+Wikipedia (vs 40 on Minipile)
- Same LR 3e-4 cosine, effective batch 512
- Target MLM loss: < 2.0 (vs 2.54 on Minipile)
- Evaluate on full GLUE after training

---

## Phase 3: Architectural Improvements (parallel with Phase 2)

### 3.1 Concept-Aware Classification Head (HIGH PRIORITY)

**Problem:** Current classification head (single [CLS] query -> concepts -> classify) discards everything the MLM decoder learned about reconstructing positions and sequences.

**Solution:** Keep the MLM decoder pathway and classify from the reconstructed sequence:
```python
# Instead of: concepts -> cls_query -> classify
# Do: concepts -> decoder(position_queries) -> full_sequence_repr -> pool -> classify

class ConceptEncoderForSequenceClassificationViaDecoder(PreTrainedModel):
    def forward(self, input_ids, attention_mask, labels):
        # 1. Encode to concepts (same as before)
        concept_repr = self.encoder(input_ids, attention_mask).last_hidden_state
        
        # 2. Decode back to sequence using MLM decoder (REUSE pretrained decoder!)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.decoder_query_embeddings(position_ids).expand(batch_size, -1, -1)
        input_embeddings = self.encoder.token_embeddings(input_ids)
        decoder_queries = input_embeddings + pos_embeddings
        
        attn_output, _ = self.decoder_cross_attn(query=decoder_queries, key=concept_repr, value=concept_repr)
        decoder_output = decoder_queries + attn_output
        decoder_output = decoder_output + self.decoder_ffn(self.post_cross_norm(decoder_output))
        
        # 3. Pool and classify (like BERT)
        pooled = mean_pool(decoder_output, attention_mask)
        logits = self.classifier(self.dropout(pooled))
```

**Key insight:** This loads ALL pretrained weights (encoder + decoder) instead of just encoder weights. The decoder knows how to reconstruct positional information from concepts.

**Expected impact:** Significant on position-sensitive tasks (CoLA, RTE). The decoder already learned position->concept mapping during MLM.

### 3.2 Position-Enriched Concepts (MEDIUM PRIORITY)

**Problem:** Concepts have no position. Position only implicit in cross-attention patterns.

**Option A -- Sinusoidal concept positions:**
```python
concept_position_emb = sinusoidal_embedding(concept_num, hidden_size)
concept_repr = concept_embeddings + concept_position_emb
```

**Option B -- Contextual Position Encoding (CoPE-style):**
Let the model learn which tokens increment the concept position counter based on content. This lets concepts address by semantic unit ("i-th noun phrase") not just index.

**Option C -- RoPE on concept self-attention:**
Apply RoPE to the concept self-attention (not cross-attention). This gives concepts a notion of "before/after" relative to each other.

### 3.3 Dimension Inversion: Thin Tokens + Fat Concepts (from original plan Step 2)

**Hypothesis:** Token embeddings can be dramatically smaller (32-128 dim) if concepts aggregate information into larger representations (256-1024 dim).

Implementation:
```python
class ConceptEncoderConfig:
    token_embedding_dim: int = 64    # small
    concept_dim: int = 512           # large (= hidden_size)

class ConceptEncoder:
    self.token_embeddings = nn.Embedding(vocab_size, token_embedding_dim)
    self.token_projection = nn.Linear(token_embedding_dim, concept_dim)
```

**Ablation grid:**
| token_dim | concept_dim | Ratio | Expected effect |
|-----------|-------------|-------|-----------------|
| 512 | 512 | 1:1 | Baseline (current) |
| 128 | 512 | 1:4 | Modest savings |
| 64 | 512 | 1:8 | Strong compression |
| 64 | 1024 | 1:16 | Maximum concept capacity |

### 3.4 Span Masking (from original Idea 3)

Replace random 15% token masking with span masking during pretraining:
```python
# In DataCollator: mask spans of 3-10 consecutive tokens
# Forces concepts to encode multi-token patterns, not just single-token statistics
```

This directly addresses the concept encoder's purpose: concepts should encode phrases/chunks, not individual tokens. Use PMI-Masking (from SpanBERT) for linguistically-motivated spans.

**Combined with higher masking rate:** Research shows 40% masking works for larger models. Span masking + 30-40% rate forces the bottleneck to carry more holistic information.

### 3.5 Gradual Compression (Funnel-style) (LOWER PRIORITY)

Instead of instant bottleneck, compress progressively:
```
Layers 1-2: Full sequence [B, 512, H] (token-level processing)
Layer 3: Pool to [B, 256, H] (2x compression)
Layer 4: Pool to [B, 128, H] (4x = concept level)
Layers 5-6: Concept-level self-attention [B, 128, H]
```

This lets early layers capture token-level patterns before compression destroys them. Directly inspired by Funnel-Transformer.

---

## Phase 4: Concept Losses and Regularization (1-2 days, quick ablation)

Run on L6 perceiver_mlm. Infrastructure ready (`LossManager`).

| Experiment | Config | Expected |
|-----------|--------|----------|
| Baseline | `CONCEPT_LOSSES="none"` | Current results |
| Orthogonality | `CONCEPT_LOSSES="orthogonality" LOSS_WEIGHTING="kendall_gal"` | Diverse concepts |
| VICReg | `CONCEPT_LOSSES="vicreg" LOSS_WEIGHTING="kendall_gal"` | Prevent collapse |
| Combined | `CONCEPT_LOSSES="combined" LOSS_WEIGHTING="kendall_gal"` | Best of both |

---

## Phase 5: Concept Analysis (diagnostic, no training)

Run on L6 perceiver_mlm checkpoint before expensive experiments.

1. **Effective rank:** How many of 128 concepts are actually used? Target > 80%.
2. **Concept correlation matrix:** Are concepts diverse or collapsed? Mean correlation < 0.3 is good.
3. **Attention pattern analysis:** Do concepts specialize on token types (nouns, verbs, function words)?
4. **Dimension utilization:** What fraction of 512 dims carry useful information?

Use existing tools: `analysis/check_model_health.py`, `analysis/concept_analysis.py`.

---

## Execution Priority

| # | Experiment | Days | Expected Impact | Dependencies |
|---|-----------|------|----------------|-------------|
| **1** | Phase 1: Engineering fixes | 1 | 2-4x speedup | None |
| **2** | Phase 5: Concept analysis | 0.5 | Diagnostic guidance | None |
| **3** | 3.1: Classification via decoder | 2 | +5-10pts on CoLA/RTE | None |
| **4** | Phase 2: Scale data (OpenWebText+Wiki) | 5 | +5-10pts avg (hypothesis) | Phase 1 |
| **5** | 3.4: Span masking | 1 code + 5 train | +2-5pts (hypothesis) | Phase 1 |
| **6** | 3.2: Position-enriched concepts | 2 | +3-5pts on CoLA | Phase 1 |
| **7** | Phase 4: Concept losses | 2 | +1-2pts | Any L6 model |
| **8** | 3.3: Dimension inversion | 3 code + 5 train | Unknown (novel) | Phase 1 |
| **9** | 3.5: Gradual compression | 5 code + 5 train | Unknown (novel) | Phase 1, 4 |

**Critical path:** Phase 1 -> [Phase 5 + Experiment 3.1] in parallel -> Phase 2 with span masking -> evaluate

---

## On Server Access

Having direct SSH access to Polonez would enable:
- Automated experiment scheduling (launch next run when current finishes)
- Real-time log monitoring and early stopping decisions
- Quick debugging of OOM/errors without round-trip
- Automated result collection and analysis

This would dramatically accelerate the experiment cycle from ~2 days (manual) to ~hours (automated).

---

## Research Horizon

If the above experiments succeed (closing gap to < 10pts from BERT-Base), the next frontier is:
1. **Concept encoder-decoder for generation** (the original vision) -- use SONAR-LLM's frozen-decoder-as-loss approach
2. **Audio modality** -- concept bottleneck on mel spectrograms, shared concept space with text
3. **Longer context** -- the concept bottleneck enables O(C*N) attention instead of O(N^2), test on 4K-16K sequences
4. **Publication** -- the dimension inversion + concept-aware classification results would be novel contributions
