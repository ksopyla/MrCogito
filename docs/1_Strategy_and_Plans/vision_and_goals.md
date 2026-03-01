# MrCogito: Concept Encoder Research Vision and Goals

Build an **audio conversational and reasoning model** grounded in a **concept bottleneck** architecture. Instead of operating on raw tokens (text) or codec frames (audio), the model compresses long input sequences into dense "concept tokens," reasons iteratively in concept space, and generates output (text or speech) via a decoder.

**Phased milestone path:** The full vision requires six sequential phases, each gated by concrete success criteria. Phase 1 proves the concept bottleneck captures semantics. Phases 2-3 build robust representations and transition to generation. Phase 4 adds instruction following. Phase 5 adds reasoning via recursive concept refinement. Phase 6 extends to audio. The current focus is Phase 1 (SG1).

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

---

## Research Phases

The research progresses through six phases. Each phase has a clear gate that must be passed before the next begins. Phases 1-2 are tightly coupled (both target concept quality) and share SG1. Later phases each have their own sub-goal.

**Phase 1 -- Concept Encoding Proof.**
Cross-attention + MLM/reconstruction objectives prove that concepts capture semantics. Work mainly with encoders and self-reconstruction. Evaluate on STS-B, concept rank, GLUE.
Gate: STS-B > 0.70, effective rank > 64/128.

**Phase 2 -- Representation Excellence.**
New training objectives (TSDAE, diffusion, contrastive, prefix generation), new architectures (recursive, BiXT, dimension inversion), data scaling. Still perceiver-type encoding with different decoding methods. Prefix generation (encode prefix, decode suffix) is a concept quality technique here -- it forces semantic concepts because surface tokens don't transfer across segments.
Gate: STS-B > 0.75, MNLI > 65%, prefix generation loss < 3.0.

**Phase 3 -- Concept-Conditioned Generation.**
Transition from reconstruction to full text generation. Based on proven concept representations, generate coherent responses from concepts via diffusion or autoregressive decoders.
Gate: coherent multi-sentence text generation from concepts demonstrated.

**Phase 4 -- Instruction Following (SFT).**
SFT on instruction data. Encode instruction via concept bottleneck, generate response.
Gate: instruction-following model functional (AlpacaEval, MT-Bench).

**Phase 5 -- Reasoning.**
Recursive concept refinement, variable-depth training, test-time compute scaling. More iterations at inference improve reasoning without retraining.
Gate: reasoning metrics improve with higher K (GSM8K, ProntoQA, HellaSwag).

**Phase 6 -- Audio Modality.**
Audio adapter into frozen concept space. Qwen Thinker-Talker architecture, Moshi full-duplex approach. Concept-to-speech decoder.
Gate: working speech-to-concept-to-speech pipeline.

**Phase dependencies:**

```
Phase 1 (Concept Proof) → Phase 2 (Representation) → Phase 3 (Generation)
                                                       → Phase 4 (SFT) → Phase 5 (Reasoning)
Phase 2 ──────────────────────────────────────────────→ Phase 6 (Audio, also needs Phase 3 decoder patterns)
```

---

## Sub-Goals

| Sub-Goal | Phases | Summary |
|---|---|---|
| **SG1: Text Concept Quality** | 1-2 | Produce concept representations that are semantically rich, geometrically diverse, and generatively useful. Critical-path blocker for everything. |
| **SG2: Text Generation** | 3 | Generate coherent text from concept representations. The transition from "encoder model" to "generative model." |
| **SG3: Instruction Following** | 4 | SFT on instruction data. Encode instruction, generate response via concept bottleneck. |
| **SG4: Concept Reasoning** | 5 | Demonstrate that recursive concept refinement enables test-time compute scaling for reasoning. |
| **SG5: Audio Conversational Model** | 6 | Map audio into concept space. Build a Concept-Talker: encode speech into concepts, reason, decode back to speech. |

---

**Publication framing:** *"Concept Bottleneck Encoder for Long-Context Reasoning and Multimodal Understanding -- O(C*N) attention with iterative latent reasoning, from text to speech."*
