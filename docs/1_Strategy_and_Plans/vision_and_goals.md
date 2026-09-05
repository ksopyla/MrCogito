# MrCogito: Concept Encoder Research Vision and Goals

> **Note (2026-07-26):** This file is the long-term **Vision** and still stands. Day-to-day work is tracked in **[agenda.md](agenda.md)** as small, well-defined increments. The path *within* this Vision is under active exploration and genuinely open — the phases and sub-goals below are an **indicative direction, not a committed schedule**. 

**The bet, in one sentence:** build a foundation model where input is compressed into latent concept vectors, refined by a recursive reasoning core, and decoded back to the target modality — and where multiple such models can eventually cooperate by exchanging concept vectors directly, not text.

Instead of operating on raw tokens (text) or codec frames (audio), the model compresses long input sequences into dense "concept tokens," reasons iteratively in concept space, and generates output (text or speech) via a decoder. Decoding is a **crystallisation interface** (training signal + human-readable answer), not where reasoning happens.

**We do not build everything at once.** The architecture has several unlocks; they are ordered priorities below. Each layer assumes the previous one works. Multimodality and multi-agent are *enabled by* a working reasoning concept core — they are not Stage-0 goals.

---

## Strategic priorities (ordered)

| # | Priority | What it means | Depends on |
|---|---|---|---|
| **1** | **Reasoning concept core** | Encode → recursively refine concepts (weight-tied × K) → decode. Prove concepts carry semantic *and* reasoning state; K is a third compute axis (params, tokens, depth). | — |
| **2** | **~10M-token text context** | Make million-to-10M-token windows tractable via O(C·N) concept attention (C ≪ N), with C scaling with N. Long context is a *consequence* of the bottleneck, not a bolted-on RoPE trick. | Working concept encoder (priority 1) |
| **3** | **Increased reasoning bandwidth** | Reason in continuous concept vectors (~10³–10⁴ bits/step) instead of ~15-bit text tokens. More hypotheses in superposition per step; recursion depth buys thinking time without detokenising. | Concept core that actually carries state (priority 1) |
| **4** | **Latent agent-to-agent communication** | Cooperating model copies exchange concept vectors over a learned channel (not text). Highest scientific ambition *after* the single-model substrate works. | Stable concept space + recursion (priorities 1–3) |
| **5** | **Multimodality** | Vision / audio adapters project into the *same* concept space; one recursive core reasons across modalities. Text first; image then audio as adapters onto a proven core. | Shared concept space that already reasons well on text |

Near-term research focus remains **priority 1** (concept quality + concept-conditioned generation + recursive refinement). Priorities 2–5 are vision context and Stage-later bets — do not pull them into the critical path until the gate for the previous priority is clear.

---

## Why this stack — semantic bandwidth

A text token carries ~15 bits; a d≈2048 fp16 concept vector is on the order of ~32 000 bits. Latent multi-agent work (LatentMAS and related) puts one latent step at hundreds of times a text token for cooperation bandwidth. That quantitative gap is the argument for:

- **Reasoning bandwidth** (priority 3) — continuous CoT / recursive refinement in concept space instead of token CoT.
- **Communication bandwidth** (priority 4) — concept-channel agent cooperation instead of text-MAS.
- **Long context** (priority 2) — O(C·N) instead of O(N²) so 1M–10M tokens stay computable.

Honest caveat from the literature: latent reasoning often shows **parity, not victory**, vs strong token-CoT RL on hard arithmetic; wins more clearly on planning / search. The bet is the *combination*: concept bottleneck + recursive refinement + (later) concept-channel cooperation + multimodal shared bottleneck.

---

## Inference pipelines

**Text milestone (priorities 1–3):**
```
User query (clean text, N tokens)
  → Encoder: cross-attention compresses N tokens into C concepts
  → Reasoning: recursive concept refinement (K iterations, weight-tied)
  → Decoder: crystallises response from refined concepts (AR and/or diffusion)
```

**Latent multi-agent (priority 4, later):**
```
Model A concepts  ⟷  concept channel  ⟷  Model B concepts
  (only the last agent crystallises to text when a human needs it)
```

**Omnimodal end goal (priority 5, later):**
```
Text / image / speech
  → Modality adapters → shared concept space
  → Same recursive reasoning core
  → Modality-specific decoder (text / image / Talker audio)
```

**Core architecture idea:** Cross-attention between C learned concept tokens and N input tokens produces a compact semantic representation (C ≪ N). This yields O(C·N) complexity instead of O(N²). The concept count C **scales with sequence length** — it is NOT fixed. Indicative scaling toward the 10M-token ambition:

| Sequence length N | Concept count C | Compression ratio | Self-attn O(N²) | Concept O(C·N) | Speedup |
|---|---|---|---|---|---|
| 512 | 128 | 4:1 | 262K | 65K | 4× |
| 4,096 | 512 | 8:1 | 16.7M | 2.1M | 8× |
| 32,768 | 2,048 | 16:1 | 1.07B | 67M | 16× |
| 262,144 | 4,096 | 64:1 | 68.7B | 1.07B | 64× |
| 1,048,576 | 8,192 | 128:1 | 1.1T | 8.6B | **128×** |
| ~10M | ~16K–32K | ~300–600:1 | ~10¹⁴ | ~1.6×10¹¹–3×10¹¹ | **orders of magnitude** |

At 1M–10M tokens, full self-attention is computationally impossible. Concept attention with C in the low tens of thousands remains the tractable path while forcing increasingly abstract representations. Exact (N, C) pairs at 10M are a research question — the table is directional.

---

## Research Phases

The research progresses through six phases. Each phase has a clear gate that must be passed before the next begins. Phases 1–2 are tightly coupled (both target concept quality) and share SG1. Later phases each have their own sub-goal.

These phases map onto strategic priorities **1 → 3 first**; priority **2** (long context) is extended inside generation/reasoning once the core works; priorities **4–5** sit after a working text reasoning stack.

**Phase 1 -- Concept Encoding Proof.** *(priority 1)*
Cross-attention + MLM/reconstruction objectives prove that concepts capture semantics. Work mainly with encoders and self-reconstruction. Evaluate on STS-B, concept rank, GLUE.
Gate: STS-B > 0.70, effective rank > 64/128.

**Phase 2 -- Representation Excellence.** *(priority 1)*
New training objectives (TSDAE, diffusion, contrastive, prefix generation), new architectures (recursive, BiXT, dimension inversion), data scaling. Still perceiver-type encoding with different decoding methods. Prefix generation (encode prefix, decode suffix) is a concept quality technique here — it forces semantic concepts because surface tokens don't transfer across segments.
Gate: STS-B > 0.75, MNLI > 65%, prefix generation loss < 3.0.

**Phase 3 -- Concept-Conditioned Generation.** *(priority 1)*
Transition from reconstruction to full text generation. Based on proven concept representations, generate coherent responses from concepts via diffusion or autoregressive decoders. Decoder = crystallisation, not the seat of reasoning.
Gate: coherent multi-sentence text generation from concepts demonstrated.

**Phase 4 -- Instruction Following (SFT).** *(priority 1)*
SFT on instruction data. Encode instruction via concept bottleneck, generate response.
Gate: instruction-following model functional (AlpacaEval, MT-Bench).

**Phase 5 -- Reasoning + long-context stretch.** *(priorities 1, 2, 3)*
Recursive concept refinement, variable-depth training, test-time compute scaling. More iterations at inference improve reasoning without retraining (**reasoning bandwidth**). Extend sequence curriculum toward 256K → 1M, with **~10M as the long-horizon text ambition** once shorter windows are solid.
Gate: reasoning metrics improve with higher K (GSM8K, ProntoQA, HellaSwag); long-context demos (NIH / RULER-class) reported at the lengths we claim.

**Phase 6 -- Multimodality + latent multi-agent.** *(priorities 4–5, only after 1–3)*
- **Agent-comms:** inference-only concept passing first, then trained per-layer fusion (concept channel between model copies).
- **Modalities:** image adapter into the shared concept space; then audio adapter + Talker-style decode. Reasoning stays in concept space — ideally without forcing a text CoT under every audio stream.
Gate: working concept-channel PoC between two instances; working speech/image → concepts → refine → decode path as applicable.

**Phase dependencies:**

```
Phase 1 (Concept Proof) → Phase 2 (Representation) → Phase 3 (Generation)
                                                       → Phase 4 (SFT) → Phase 5 (Reasoning + long context)
                                                                              → Phase 6 (Agent-comms + multimodal)
```

---

## Sub-Goals

| Sub-Goal | Phases | Priority | Summary |
|---|---|---|---|
| **SG1: Text Concept Quality** | 1–2 | 1 | Produce concept representations that are semantically rich, geometrically diverse, and generatively useful. Critical-path blocker for everything. |
| **SG2: Text Generation** | 3 | 1 | Generate coherent text from concept representations. The transition from "encoder model" to "generative model." |
| **SG3: Instruction Following** | 4 | 1 | SFT on instruction data. Encode instruction, generate response via concept bottleneck. |
| **SG4: Concept Reasoning + Bandwidth** | 5 | 1, 3 | Recursive concept refinement enables test-time compute scaling; continuous concepts raise semantic bits per reasoning step. |
| **SG5: Long-Context Text (~10M ambition)** | 5 | 2 | Demonstrate O(C·N) long-context text; grow from 64K/256K/1M demos toward multi-million-token windows. |
| **SG6: Latent Multi-Agent** | 6 | 4 | Concept-channel cooperation between model instances; crystallise to text only when needed. |
| **SG7: Multimodal Concept Space** | 6 | 5 | Map image/audio into the same concept space; shared recursive core; modality-specific decoders. |

---

**Publication framing:** *"Concept Bottleneck Encoder for Long-Context Reasoning and Multimodal Understanding — O(C·N) attention with iterative latent reasoning, from text toward speech and agent-native concept channels."*
