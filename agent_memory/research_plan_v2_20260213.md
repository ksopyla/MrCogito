# Concept Encoder Research Plan v2 -- Research Notes
## Compiled 2026-02-13

### Diagnostic: Why Current Results Are Stuck

**Current best (L6 perceiver_mlm, 61M params):** MRPC 81.3 F1, SST-2 77.8, MNLI 59.1, CoLA 0.13 MCC
**Gap to BERT-Base (110M):** -23.7 pts average across GLUE

Root causes identified:
1. **Concept bottleneck too aggressive** - 128 concepts x 512 dim compresses too much. MLM loss dropped 37% L2->L6 but downstream only +3pts.
2. **Position information destroyed** - Concepts have NO position embeddings. Position only exists implicitly in how concepts attend to tokens. CoLA (syntactic acceptability) requires token ordering, which is irreversibly lost.
3. **Classification head discards MLM knowledge** - The perceiver classification head uses a single [CLS] query cross-attending to concepts. The MLM decoder uses position queries that reconstruct full sequences. The classification head throws away this reconstruction capability.
4. **Undertrained on tiny data** - Minipile ~1M samples seen 40 times. BERT used 3.3B words. ModernBERT used 2T tokens.
5. **Engineering inefficiency** - Flash Attention not enabled (need_weights not explicitly False), no torch.compile, no fused operations.

---

### Key Papers That Could Change the Game

1. **SONAR-LLM** (2025): Train concept-level model with token-level CE loss through frozen decoder. Eliminates need for diffusion/MSE at concept level.
2. **CoPE (Contextual Position Encoding)** (2024, Meta): Positions conditioned on content. Decoder can attend to "i-th concept" instead of "i-th token". Bridges concept-to-token gap.
3. **Funnel-Transformer**: Gradual compression, not sudden bottleneck. Invest saved FLOPs in deeper bottleneck layers.
4. **"Should You Mask 15%?"** (2022): Higher masking rates (40%+) work for larger models. Span masking needs lower rates. Could train concept encoder with 40% masking.
5. **"Mask More and Mask Later"** (2022): Introduce mask tokens at a LATER layer, not input. Earlier layers process only unmasked tokens. Concept encoder already does this - concepts gather from full input, then reconstruct.
6. **Vision Transformers Need Registers** (2023): Transformers naturally want dedicated storage tokens. Concept slots ARE explicit registers. Adding more registers + proper initialization could help.

---

### Dataset Recommendations

| Dataset | Size | Fit | Strategy |
|---------|------|-----|----------|
| OpenWebText + Wikipedia | ~15M, 33GB | 10x Minipile | Quick win, BERT-style mix |
| C4 realnewslike | ~14M, 15GB | 10x Minipile | High quality, T5-tested |
| FineWeb-Edu sample-10BT | ~10M, 25GB | 7x Minipile | SoTA quality filtering |
| C4 full English | ~365M, 305GB | 240x Minipile | Maximum scale |
| The Pile deduplicated (subsample) | 50-100M | 50-100x | Domain diversity |

---

### Engineering Improvements Checklist

1. **CRITICAL: Set need_weights=False** on ALL nn.MultiheadAttention calls -> enables Flash Attention/SDPA
2. torch_compile=True with inductor backend
3. bf16 already enabled; add tf32=True
4. Fused AdamW (already using adamw_torch_fused)
5. Liger fused cross-entropy (20% throughput, -60% memory)
6. dataloader_num_workers=4, prefetch_factor=2
7. Expected cumulative speedup: 2-4x

---

### New Architectural Ideas

**A. Concept-Aware Classification (don't discard MLM decoder)**
Instead of replacing the MLM head with a classification head, KEEP the decoder and add classification on top:
```
Encoder -> Concepts -> Decoder (position queries -> full sequence) -> Pool -> Classify
```
This preserves position reconstruction learned during MLM.

**B. Hybrid Token-Concept Position Encoding**
Add position information TO concepts:
```
concept_repr = concept_embeddings + concept_position_embeddings  # [B, C, H]
```
Or use CoPE-style context-conditioned positions.

**C. Gradual Compression (Funnel-style)**
Instead of instant bottleneck (512 tokens -> 128 concepts), use progressive pooling:
```
Layer 1-2: 512 tokens -> 256 intermediate
Layer 3-4: 256 intermediate -> 128 concepts  
Layer 5-6: Process at concept level
```

**D. Span Masking for Concept Training**
Replace random 15% token masking with span masking (mask 3-10 consecutive tokens). Forces concepts to encode multi-token patterns, not just single token statistics.

**E. Curriculum Training**
Start with easier MLM (15% masking, short sequences) and gradually increase difficulty (40% masking, longer sequences, span masking).
