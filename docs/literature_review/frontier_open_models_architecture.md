# Frontier open-weight model architectures — what the 2026 releases actually do

Reviews of the newest open-weight frontier families, read through one question: **which of
[REDACTED]'s mechanisms did they ship, and what is still open?** Our reference design is E18
(Perceiver AR v2: hashed n-gram input → 2 SWA pre-encoder layers → ONE full-causal global read
whose K/V are computed once per sequence → 20 SWA-4096 layers; ≈1 KB/token prefix cache;
Muon), plus the planned concept-slot compression of that read (E18c) and latent message /
write-back hooks (E19/E21).

Sources were collected 2026-09-11 (research-scout sweep + direct reads of the DeepSeek-V4.1-Flash
model card and the Qwen3.8-Next architecture report). Per-claim URLs are inline. Numbers that
were only seen via the scout and not re-read are marked *(scout)*.

Related reviews: [`long_context_architectures_training.md`](long_context_architectures_training.md)
(sparse/linear/hybrid attention, KV compression, 1M recipes),
[`small_lm_training_recipes.md`](small_lm_training_recipes.md) (what ~600M should reach),
[`recurrent_memory_transformers.md`](recurrent_memory_transformers.md) (memory families).

---

## DeepSeek-V4.1-Flash — "Pushing the Limits of KV Cache Compression" (2026-09-10)
[Model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) ·
[V4 tech report arXiv:2606.19348](https://arxiv.org/html/2606.19348) ·
[Engram paper arXiv:2601.07372](https://arxiv.org/html/2601.07372v2) · [Engram code](https://github.com/deepseek-ai/Engram) ·
[V4 HF blog](https://huggingface.co/blog/deepseekv4) · MIT license.

### TL;DR
**The frontier just shipped the E18 skeleton.** V4.1-Flash is a *Causal Encoder-Decoder (CED)*:
a 40-layer transformer split into a **20-layer causal encoder followed by a 20-layer decoder**,
where **the decoder's global KV cache is projected from the final encoder hidden states rather
than from each decoder layer's own hidden states**. That is our "pre-encoder → K/V projection
computed once → read by the stack" identity, at 552B backbone / 8B active (prefill) / 16B active
(decode), 1M context, trained on 45T tokens. With CSA2 (static per-layer Full / Reindex / Reuse
modes sharing main KV and indexer keys across layers, hierarchical sparse indexer, FP4 main KV)
the **global KV cache is 890 bytes per token** — "roughly 1/4 of V4-Flash, 437× less than V1".
SWA layers do not persist their cache at all ("SWA Bounded Replay" recomputes the last `n_win`
tokens). Also in: **Engram conditional memory** (196B params, hashed-token lookup, sparsely
accessed), Single-Pass mHC residual mixing, MTP-style "DSpark" speculative decoding,
`reasoning_effort` 1–100. Sparse attention trained at 64K and extended to 1M at 34T tokens.

### What it means for us
- **Validation of the direction, loss of "first":** an encoder-projected, once-per-sequence
  global K/V read by a whole decoder stack is now a frontier default, at ~1 KB/token. The E18
  pilot number (1 KB/token) is no longer a differentiator on its own; the deck must not lead with it.
- **Where they stop:** the cache is still **one slot per token** (compressed in *bytes* by FP4 and
  by sparse top-k *access*, not in *count*). Nothing pools the prefix into fewer, semantically
  organised slots; nothing exposes the projected cache as an object another model reads;
  nothing trains it under anything but next-token CE + agentic RL. E18c (slots per r tokens) and
  E21 (the projected cache as a trained *message*) sit exactly past their stopping point.
- **Their 20 encoder layers vs our 2.** Their encoder half is as deep as the decoder half.
  Our reach ablation showed one bottom read is used most (arm A) and depth of the *read* does not
  matter (arm B) — but we never varied *encoder* depth beyond 2 SWA layers. Worth one arm.
- **LongBench-V2 45.2 (base, 1-shot)** is their only long-context number on the card; MRCR-1M
  for V4-Flash-Max was 78.7 *(scout, V4-Flash card)*.
- Engram (hashed 2/3-gram tables, O(1) lookup, host-offloaded) is our `TinyHashedEmbedding`
  scaled to 196B — the mechanism is ratified; its *gains at 27B were MMLU +3.4 / ARC-C +3.7 /
  NIAH 84→97 at iso-FLOPs (scout, Engram paper)*.

## DeepSeek-V4 (Pro 1.6T-A49B / Flash 284B-A13B, 2026-04-24 → 2026-08-13)
[V4-Flash card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) · [tech report](https://arxiv.org/html/2606.19348) · [API changelog](https://api-docs.deepseek.com/updates)

### TL;DR *(scout)*
Alternates **CSA** layers (4× sequence-compressed KV + DSA lightning-indexer top-k + local SWA)
with **HCA** layers (128× compressed stream, dense attention over it); MLA-derived KV in FP8/FP4
(~10% of V3.2's KV at 1M); mHC residuals; MTP; **Muon**; 32T+ tokens; post-training SFT → RL →
on-policy distillation. MRCR-1M 83.5 (Pro-Max) / 78.7 (Flash-Max); CorpusQA-1M 62.0 (Pro).
No ≤4B variant. No latent reasoning / latent A2A features; "interleaved thinking" is text.
**Relevance:** the HCA "128× compressed stream" is a coarse *hierarchical* read — a
compression-by-count idea adjacent to E18c, but per-token-pooled without learned queries and
still trained under CE only.

## Qwen3.8 (2.4T-A95B, 27B dense VL; 2026-08) and Qwen3.8-Flash-Next (125B-A6B + 51B n-gram)
[Qwen3.8 blog](https://qwen.ai/blog?id=qwen3.8) · [27B card](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) ·
[2.4T card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) · [GitHub](https://github.com/QwenLM/Qwen3.8) ·
**Flash-Next architecture report [arXiv:2608.30320](https://arxiv.org/html/2608.30320)** (read directly) ·
[Alibaba blog](https://www.alibabacloud.com/blog/qwen-3-8-flash-next-a-new-architecture-towards-ultimate-cost-efficiency_603501)

### TL;DR
Main Qwen3.8 = the Qwen3.5 / Qwen3-Next hybrid: **3× Gated DeltaNet : 1× gated full attention**
(27B: 48 linear + 16 full of 64 layers), GQA on full layers, **262k native → ~1M via YaRN**, MTP
draft head, thinking mode with `reasoning_effort` and `preserve_thinking` (KV reuse across agent
turns). MRCR-v2 256K 8-needle 92.9 (Max). No ≤4B Qwen3.8; Qwen3.5-4B exists.
**Flash-Next** (report, Aug 2026): 125B total / 6B active + **51B n-gram embedding tables held
off-accelerator** (one n-gram layer at layer 2, host-prefetched); 3 GDN : 1 full-attention, the
full layers swapped for **Qwen Sparse Attention (QSA)** at continued-pretraining (micro-block
indexer with compressed keys, O(n²/r)); **Gated Residual** (4-branch widened residual, elementwise
read gate — the stability lever); **Muon** on 2-D weights (embeddings/head/router on AdamW; split
fused params before orthogonalisation; 8 Newton–Schulz steps); refit scaling law → larger LR and
batch; **batch-size warmup unnecessary** (costs 18.8% more steps); stable at 4× optimal LR without
qk-clip. At 1M, QSA 7.6× faster prefill / 4.9× decode vs dense (kernel level).

### Findings that transfer to a 600M from-scratch run
- **Loss and accuracy diverge:** "enlarging the n-gram vocabulary lowers loss monotonically while
  downstream accuracy saturates" — do not size our hashed tables by eval CE alone.
- **SWA-hybrid < GDN-hybrid** on 7/9 benchmarks in their ablation (Tab. 1). Our local stack is
  pure SWA; a GDN local mixer is a credible upgrade for the *local* channel and would leave the
  global read's role (exact retrieval / message) untouched. Note as an arm, not a default.
- **NoPE on full layers** "indistinguishable during pre-training but affects generation quality
  at later stages" — consistent with our tiny-study finding that the NoPE read is not free.
- Muon + wider residual shifts optimal LR/batch *up*; re-fit, do not transfer AdamW HPs.

## GLM-5.3-Flash (Z.ai, 320B-A18B, 2026-08-26)
[Z.ai blog](https://z.ai/blog/glm-5.3-flash) · [HF card](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT) ·
[Z.ai docs](https://docs.z.ai/guides/vlm/glm-5.3-flash) · [GLM-5 report arXiv:2602.15763](https://arxiv.org/abs/2602.15763)

### TL;DR *(scout)*
45 layers: **34 KDA (Kimi Delta Attention, linear) + 11 sparse MLA/DSA layers** (~3:1); sparse
layers keep **top-2048 of 1M** keys via an IndexPool with 4:1 indexer-KV compression; NoPE MLA
(`kv_lora_rank=512`); mHC residuals; MTP; 30T multimodal tokens; 1,048,576 context; vs GLM-5.3
3.01× less attention compute and 4.44× smaller KV. `reasoning_effort` low/high/max. No small
variant, no long-context numbers on the card, no latent A2A.

---

## Cross-family synthesis (what converged, what is open)

| Mechanism | Ours | Shipped by |
|---|---|---|
| Hashed n-gram input tables | `TinyHashedEmbedding` (2×2^17×256) | **Engram** (DeepSeek V4.1, 196B), **Qwen3.8-Flash-Next** (51B) |
| Few global / many local layers | 1 global : 22 local | Qwen 1:4 (GDN), GLM ~1:3 (KDA), Gemma 3 1:5 (SWA) |
| Global K/V computed once from an encoder and read by the stack | E18 pre-encoder → read | **DeepSeek V4.1-Flash CED** (20-layer encoder → projected decoder KV) |
| ~1 KB/token global cache | 1 KB/token (2 kv-heads × 128 × bf16) | **V4.1-Flash 890 B/token** (FP4 + CSA2) |
| Sparse top-k access to the global cache | not yet (candidate for E18 read) | DSA / CSA / QSA / GLM IndexPool |
| Linear-attention local mixer | no (SWA) | Qwen GDN, GLM KDA, Kimi Linear |
| Muon | yes | DeepSeek V4, Qwen3.8-Flash-Next |
| MTP | no (evidence: hurts < 1B) | all three |
| Widened / gated residual (mHC, GR) | no | DeepSeek, GLM, Qwen-Next |
| **Compress the prefix in slot *count*** (learned queries, r tokens → c slots) | E18c (draft) | none |
| **Train the global cache as a message another model reads** | E21 (draft) | none |
| **Latent / looped reasoning in a released model** | E19 hook | none (all reasoning is text; `reasoning_effort` scales tokens) |
| Open from-scratch recipe at 125M–1B with these parts | E18 | none (smallest active: 6–8B) |

**Reading.** Between 2025 and Sep 2026 the field converged on *hybrid local mixer + few
sparse/compressed global layers + hashed n-gram capacity + Muon + MTP*, trained under next-token
CE and agentic RL, with all inter-model communication in text. Our E18 platform is now a
small-scale instance of a frontier-validated skeleton (good for credibility, useless as a moat).
The three cells no one occupies — slot-count compression with learned queries, a
pretrained-as-message global cache, and latent reasoning steps that write back into that cache —
are the same three cells our Vision named. See
[`../4_Research_Notes/strategy_synthesis_latent_channel_20260911.md`](../4_Research_Notes/strategy_synthesis_latent_channel_20260911.md).
