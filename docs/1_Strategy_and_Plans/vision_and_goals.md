# MrCogito: Concept Encoder Research Vision and Goals

Build an **audio conversational and reasoning model** grounded in a **concept bottleneck** architecture. Instead of operating on raw tokens (text) or codec frames (audio), the model compresses long input sequences into dense "concept tokens," reasons iteratively in concept space, and generates output (text or speech) via a decoder.

**Milestone: text generative reasoning model.** Before adding audio, we must first prove the concept bottleneck works for text -- producing a generative reasoning model that encodes input, reasons in latent concept space, and generates text output. This is the current focus (SG1 + SG2).

**Inference pipeline (text milestone):**
```
User query (clean text, N tokens)
  → Encoder: cross-attention compresses N tokens into C concepts
  → Reasoning: recursive concept refinement (K iterations, weight-tied)
  → Decoder: generates response from refined concepts (diffusion or autoregressive)
```

**Inference pipeline (audio end goal):**
```
User speech (audio, mel-spectrogram)
  → Audio adapter: maps audio features into concept space
  → Reasoning: recursive concept refinement (shared weights with text)
  → Audio decoder (Talker): generates speech tokens/codec frames from concepts
```

**Core architecture idea:** Cross-attention between C learned concept tokens and N input tokens produces a compact semantic representation (C << N). This yields O(C*N) complexity instead of O(N^2). The concept count C **scales with sequence length** -- it is NOT fixed:

| Sequence length N | Concept count C | Compression ratio | Self-attn O(N^2) | Concept O(C*N) | Speedup |
|---|---|---|---|---|---|
| 512 | 128 | 4:1 | 262K | 65K | 4x |
| 4,096 | 512 | 8:1 | 16.7M | 2.1M | 8x |
| 32,768 | 2,048 | 16:1 | 1.07B | 67M | 16x |
| 262,144 | 4,096 | 64:1 | 68.7B | 1.07B | 64x |
| 1,048,576 | 8,192 | 128:1 | 1.1T | 8.6B | **128x** |

At 1M tokens, full self-attention is computationally impossible. Concept attention with C=8K-16K remains tractable while forcing increasingly abstract, semantic representations.

**Training objective evolution** (reconstruction → generation → reasoning):

| Phase | Objective | What it trains | Evaluation |
|---|---|---|---|
| Phase 0 | Self-reconstruction (MLM/diffusion/TSDAE) | Concept compression quality | STS-B, concept rank, GLUE |
| Phase 1 | **Prefix generation** (encode prefix, generate suffix via diffusion) | Semantic concepts + generative decoder | Suffix perplexity, STS-B |
| Phase 2 | **Variable-depth recursive training** (sample K at train time) | Latent reasoning through iteration | Reasoning benchmarks (GSM8K, ProntoQA) |
| Phase 3 | **Instruction fine-tuning** (encode instruction, generate response) | Task-following generation | Instruction-following benchmarks |
| Phase 4 | **Progressive sequence length** (512 → 4K → 32K → 1M) | Long-context concept abstraction | SCROLLS, LongBench |



**Publication framing:** *"Concept Bottleneck Encoder for Long-Context Reasoning and Multimodal Understanding -- O(C*N) attention with iterative latent reasoning, from text to speech."*
