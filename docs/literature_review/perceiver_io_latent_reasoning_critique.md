# Why Perceiver IO was not the foundation of latent reasoning

Cross-paper critique of the original Perceiver / Perceiver IO bet
(Jaegle et al. 2021): *cross-attend the world into a small notebook Z,
self-attend on Z, call that thinking*. Written 2026-09-19 against the
papers and the 2022–2026 record, not against this repo's forks.

Related reviews (do not duplicate):
- BAPO / CoT bandwidth:
  [`reasoning_bandwidth_information_flow.md`](reasoning_bandwidth_information_flow.md)
- Frozen vs writable memory, Coconut / Huginn / Perceiver AR:
  [`recurrent_memory_transformers.md`](recurrent_memory_transformers.md)
- When a latent *set* actually pays for extra compute:
  [`latent_set_refinement.md`](latent_set_refinement.md)
- Rate–distortion / RankMe / Slot Attention:
  [`information_bottleneck_latent_capacity.md`](information_bottleneck_latent_capacity.md)
- Gist / KV compressors:
  [`learned_kv_context_compression.md`](learned_kv_context_compression.md)

**Standing claim.** The Perceiver notebook is a good *compressor* and a
bad *reasoner*. Self-attention on a fixed set is mixing, not a tape.
Modern LLMs reason by writing a re-readable sequence (tokens, or
supervised latent steps). The industrial stack followed that, plus
FlashAttention, not a latent Transformer.

---

## The optical illusion

The architecture *looks* like a student:

1. Cross-attention: the notebook queries the book ("read only what
   matters"). Cost `O(M N)`, `N ≪ M`.
2. Self-attention on Z: the N pages talk to each other ("reason").
   Cost `O(L N²)`, independent of the book.
3. Perceiver IO decode: output queries ask the finished notebook.

That is the right picture for **lossy perception** (ImageNet class,
AudioSet tags, a 64-token visual prompt). It is the wrong picture for
**serial computation**: majority, multi-hop binding, exact recall of a
surprise, anything that needs intermediate results you can look up
later.

The rest of this file is why those two jobs diverged after 2021.

---

## Information flow — what Z actually is

```
X (bytes, frozen)  --K/V-->  CA  <--Q--  Z₀ (learned N×D parameter)
                                      |
                                      v
                               Z ← Z + Attn(Z)   × L   (same array)
                                      |
                    Perceiver:  mean_N(Z) → class
                    IO:         Q_out --CA--> Z → Y
```

Facts from the papers and the released JAX
([deepmind-research/perceiver](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver)):

- **One array, many writes.** Layers have weights. They do not own
  private latents. Residual: `Z ← Z + Attn(LN(Z), …)` then MLP.
- **Encode-once (IO / released encoder) throws the book away.** After
  the first cross-attend, X is never consulted again. Later "reasoning"
  cannot fetch a detail that was not copied on the first pass. The
  original Perceiver paper already measured this: interleaved re-reads
  78.0 vs all cross-attends stacked at the start 73.7 ImageNet top-1
  (Table 6). Even *classification* wanted re-entrant reads.
- **"Attend to what matters" is loss-shaped, not oracle-shaped.** Softmax
  CA is a filter trained by the objective. Class CE keeps class-useful
  texture. It does not keep a password on page 3 because a query on
  page 300 might need it. Gist-token autopsy: details that are
  *surprising* are the first to die
  ([arXiv:2412.17483](https://arxiv.org/abs/2412.17483)).
- **Learned queries are unallocated.** Z₀ is a truncated-normal
  parameter of shape `N×D`. Nothing forces slot `i` to cover span `i`.
  Slot Attention needs *competitive* softmax over slots to stop
  collapse; Perceiver does not. Rank then concentrates (RankMe).
- **Decode queries are not a second notebook.** IO's output array is an
  address book that *reads* a frozen Z. That is retrieval from a
  snapshot, not another thinking stage.
- **A set is not a tape.** Z is permutation-equivariant except for
  learned index embeddings. Self-attention mixes the pages in parallel.
  There is no "step 4 may read what step 3 wrote" unless you *append*
  new positions (CoT, Coconut's chain, LOTUS's parallel blocks with
  gold targets).

Perceiver AR (Hawthorne et al., ICML 2022,
[arXiv:2202.07765](https://arxiv.org/abs/2202.07765)) already quietly
abandoned the "learned notebook" story for language: each latent is
tied to one of the *last N output positions*, queries are those token
embeddings, and both CA and SA are causally masked. That is a narrow
decoder attending a long prefix — closer to YOCO than to Perceiver IO.

---

## Gradient flow — why extra Z-layers often do nothing

- **Residual identity.** `Z_out = Z_in + F(Z_in)`. If F is not needed
  for the loss, its Jacobian can shrink toward 0 and the extra layer is
  an optional path. Veit, Wilber & Belongie (NeurIPS 2016,
  [arXiv:1605.06431](https://arxiv.org/abs/1605.06431)) showed residual
  nets behave like ensembles of *shallower* paths; most gradient in a
  110-layer ResNet came from paths of length 10–34. That measurement
  is conv-nets, analogical for Transformers. Flamingo made it
  industrial: tanh-gates on the new cross-attends init at 0 so those
  layers are **identity at step 0**. Universal Transformers / ALBERT /
  Perceiver weight-sharing did not become the default for the same
  reason: unrolled depth is cheap parameters, not guaranteed extra
  computation.
- **Shared unrolls average the gradient.** ImageNet Perceiver shares
  CA 2–8 and the matching self-attend blocks. `dL/dθ_shared` is the
  *sum* over iterations. That is a regularizer (44.9M vs 326.2M params,
  *same FLOPs*, better val because less overfit — Table 7). It is not
  "the network thinks eight times." First CA stayed unique because
  sharing it was unstable — the one specialized read, then a reused
  mixer.
- **Softmax CA starves most of X.** Gradient into bytes flows only
  through K and V, weighted by attention. A few winners get the
  update; the rest of the book is dead. Fine if the task is "which
  ImageNet class." Lethal if later you need an un-attended token.
- **Mean-pool head is a uniform spray.** Original Perceiver: average
  the N slots, linear to classes. Every slot gets the same
  `dL/dmean`. No per-slot target → redundancy is free (superposition /
  RankMe collapse). Contrast Slot Attention (competitive softmax over
  slots) and LOTUS (gold CoT CE on every latent position).
- **Bypass kills the bottleneck.** If the decoder can see raw X, CE
  routes around Z (Broken ELBO; every "decoder has a local window"
  result since). Then gradient through CA dies and Z becomes
  decoration. IO's exclusive decode from Z is the *right* closure —
  *if* the loss actually needs what Z stored.
- **Latent chains without step CE attenuate.** Chen et al. 2026
  ([arXiv:2606.20075](https://arxiv.org/abs/2606.20075)): outcome-only
  supervision ≈ no-CoT because of gradient attenuation along the latent
  chain. Pfau, Merrill & Bowman (COLM 2024,
  [arXiv:2404.15758](https://arxiv.org/abs/2404.15758)): hidden
  "think in dots" works only with dense supervision of the delay; a
  Markov `z_k → z_{k+1}` is not a tape.

So: extra self-attends on Z are *expressive enough* to mix, and
*easy enough to skip*. That is the opposite of a reasoning primitive.

---

## Theory after Perceiver: mixing is not Turing

Pointers, not re-reviews. Full BAPO writeup:
[`reasoning_bandwidth_information_flow.md`](reasoning_bandwidth_information_flow.md).

- **Constant-depth transformers ⊆ TC⁰** (Merrill & Sabharwal, TACL
  2023). A stack of self-attends on a *fixed* N, however deep, is still
  a constant-depth circuit over those N positions. Depth is not a
  serial tape.
- **CoT is what adds a tape** (Merrill & Sabharwal, ICLR 2024,
  [arXiv:2310.07923](https://arxiv.org/abs/2310.07923)): generating
  intermediate tokens and attending back to them is what makes
  transformers Turing-complete (with enough precision).
- **BAPO Theorem 10** (Schnabel & Tomlinson et al., NeurIPS 2025,
  [arXiv:2505.08140](https://arxiv.org/abs/2505.08140)): adding layers
  or heads at a *fixed* bottleneck does **not** raise effective
  prefix bandwidth. MAJORITY / REACHABILITY / MATCH3 stay hard.
  Theorem 8: chain-of-thought makes any decidable language easy at
  constant `(2,3)` bits/tokens *per step*, by writing a re-readable
  tape. That is the exact negation of "stack more SA on the same Z."
- **Sanford MATCH2 vs MATCH3** ([arXiv:2306.02896](https://arxiv.org/abs/2306.02896)):
  pairwise lookup is easy for attention; three-way binding is not.
  A small latent set that has already softmax-pooled the inputs cannot
  recover a MATCH3 that was never written down as separate addresses.
- **Hahn (TACL 2020)** ([arXiv:1906.06755](https://arxiv.org/abs/1906.06755)):
  a single token's influence vanishes as `n` grows under softmax
  attention. "Attend to what matters" gets *harder* as the book gets
  longer, unless you allocate slots by position.
- **Elhage residual stream** ([transformer-circuits 2021](https://transformer-circuits.pub/2021/framework/index.html)):
  the stream is a finite-width bus. Heads are read/write ports. A
  notebook of 512 vectors *is* a 512-slot bus. You cannot smuggle a
  growing computation through a constant-width bus without writing
  extra positions.

The illusion, restated in this language: Perceiver IO spends its
capacity on a wide parallel mixer over a small set. LLMs that "reason"
spend it on a *long* sequence of writes.

---

## Why it was not adopted (2022–2026 record)

Not one knockout paper. A stack of cheaper alternatives plus an
ecosystem that never formed.

### The original problem got cheaper

Perceiver's pitch was "Transformers cannot look at 50k pixels." ViT
(Dosovitskiy et al. 2021) looked at 196 *patches*. FlashAttention (Dao
et al. 2022, then FA2/FA3) made `O(n²)` in HBM-feasible length, then
GQA / MLA / sparse / sliding-window pushed context to 32k–1M *inside
decoder-only serving*. The industrial answer to "too many tokens" was
**better attention kernels and positional pooling**, not a second
architecture family.

### Decoder-only was already the scaling vehicle

GPT-3 (2020) predates Perceiver IO. The IO paper itself: they “don’t
currently address generative modeling” — the language result is
BERT-style MLM, FLOP-matched to a Transformer encoder, not a GPT.
Wang, Roberts et al. (ICML 2022,
[arXiv:2204.05832](https://arxiv.org/abs/2204.05832)) then ran the
bake-off the Perceiver never entered: at 5B+ / 168B tokens, **causal
decoder + full LM is strongest zero-shot**; encoder-decoder + MLM only
wins after multitask finetune. Same-org revealed preference: Gopher
(Rae et al. 2021, [arXiv:2112.11446](https://arxiv.org/abs/2112.11446))
scaled a decoder-only Transformer, not a Perceiver. Kaplan /
Chinchilla, Megatron, then GQA / PagedAttention / vLLM all assume
**per-layer, per-token causal KV**. Perceiver IO is encode–process–
decode, JAX/Haiku, no drop-in for that object. Nobody pre-trained a
70B Perceiver IO. There is still **no public 1B–7B matched-token
bake-off of Perceiver AR vs a GPT recipe** — a hole, not a hidden
knockout.

### Vision did not pick it either

ImageNet-without-convolutions was a good demo. The winning vision
recipe was ViT / ConvNeXt + JFT or ImageNet-21k + 2D structure. A
model that is *proud* of permutation invariance is fighting the
statistics of photos. Fourier features re-injected the grid that the
architecture had removed.

### Language: Perceiver AR was the real attempt, and still not Llama

PG-19 **test** PPL 28.9 at 60 layers / 2k context vs published
Transformer-XL 36.3 / 36 layers — a test win, not a matched-depth
bake-off. Validation is a tie/slightly worse (45.9 vs XL 45.5). The
authors saw **no gain past ~2k tokens** on PG-19; WikiText-103 is
18.52 vs their own XL 18.42, and 4k–8k context *hurt* vs 2k. They used
extreme cross-attend dropout (0.875–0.96875): the bottleneck is
regularized almost shut. Copy at 131k worked — INDEX is BAPO-easy
(`a=0, b=1`): one raw addressable token is enough if CA can point at
it. That does not imply a reasoning core. Appendix E.3: standard KV
caching does **not** transfer when latent width ≠ input width; they
periodically run full forwards to reset the cache. Latents are rebuilt
every pass from the last N tokens; they are **not** a running memory
across generation
([`recurrent_memory_transformers.md`](recurrent_memory_transformers.md)
§A). The serving stack never grew around this mask pattern.

### The one piece that shipped — then got simplified away

| Year | What shipped | What it kept | What it dropped |
|---|---|---|---|
| 2022 | Flamingo Perceiver Resampler ([arXiv:2204.14198](https://arxiv.org/abs/2204.14198)) | 64 learned queries, one CA into a *frozen* ViT, then a frozen LLM | The latent Transformer as the model. No interleaved re-read. No class-pool head. |
| 2023 | BLIP-2 Q-Former ([arXiv:2301.12597](https://arxiv.org/abs/2301.12597)) | 32 queries + query self-attn | Generative-only alignment. They *added* a contrastive stage because Flamingo-style gen loss was not enough; OPT forgot the language. |
| 2023 | LLaVA ([arXiv:2304.08485](https://arxiv.org/abs/2304.08485)) | — | The bottleneck. Linear map of *all* patches into the LLM. Often better, much simpler. |
| 2023 | MiniGPT-4 ([arXiv:2304.10592](https://arxiv.org/abs/2304.10592)) | optional Q-Former | Removing Q-Former ≈ same AOK-VQA/GQA in their limited-data setup. |
| 2024 | IDEFICS2 ([arXiv:2405.02246](https://arxiv.org/abs/2405.02246)) | 64-latent resampler as **pool** | Flamingo-style xattn *as the LLM*. Measured avg: frozen AR no-Perceiver 51.8 → +Perceiver 60.3; xattn+LoRA 67.3; **fully AR+LoRA 69.5 (chosen)**. Pooling helps; fusion into a Perceiver-shaped LLM loses once the decoder can adapt. |
| 2024 | DeCo / Visual Anchors | — | Q-Former worst on RefCOCO localization (spatial smear). Learned queries are **input-invariant** and drop uncommon patterns. |

So the adopted remnant is: **a small visual compressor in front of a
decoder-only LLM**. The "reason in Z" stack was never the product.

### What actually scaled for long context

DeepSeek HCA pools 128 tokens into one KV entry with softmax weights;
entries **do not talk to each other**. Qwen finite-state layers keep
one raw/sparse layer in four because "no finite-state memory
reproduces exact retrieval." YOCO / CED: a deep *causal encoder*
writes one KV for a decoder — contextualize first, compress second.
LCLM (below): mean-pool blocks, MLP adapter, **no latent self-attn**.

Convergence: **positional compression into KV, no slot mixer, keep a
raw path for INDEX.** That is the opposite of Perceiver IO's learned
queries + deep Z-Transformer + exclusive decode.

---

## LCLM — the 2026 clean-room test of "refine Z"

## End-to-End Context Compression at Scale (LCLM)

2026 · [arXiv:2606.09659](https://arxiv.org/abs/2606.09659) · code:
[github.com/LeonLixyz/LCLM](https://github.com/LeonLixyz/LCLM)

### TL;DR
From-scratch encoder→latent→decoder sweep at 16×, then scale to
0.6B-enc / 4B-dec. Winner: causal encoder, window 1024, **mean pooling**,
**MLP adapter that does not mix latents**. An attention adapter (one
self-attend over the latent sequence, i.e. a baby Perceiver latent
Transformer) **lost on pre-training loss and cost more**.

### Why it matters here
This is the closest published A/B of Perceiver's "then self-attend on
Z" as a language compressor. Under next-token CE it does not pay.
Encoder *depth/window* (contextualize before pooling) does pay. Pooling
operator depends on ratio (concat wins at 4×, mean at 16×). Slots stay
independent.

---

## Flamingo / Q-Former / LLaVA — adoption, then retreat

## Flamingo

NeurIPS 2022 · [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)

### TL;DR
Perceiver Resampler: 64 latents cross-attend a frozen NFNet/ViT. The
LLM stays a decoder-only Transformer with gated cross-attention.
Gates init at 0 (identity). Reasoning, if any, is in the **token
stream**, not in the 64 latents. Their ablation: the resampler beats a
plain Transformer or MLP *as a visual connector* — compressor, not
mind.

## BLIP-2

ICML 2023 · [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)

### TL;DR
Q-Former is a Perceiver-shaped bottleneck (32 queries, self-attn among
queries, CA to frozen ViT). Ablation: drop the contrastive first stage
and it *is* Flamingo's resampler — VQA collapses and OPT catastrophically
forgets. The bottleneck needed an extra objective, not extra Z-depth.

## Visual Instruction Tuning (LLaVA)

NeurIPS 2023 · [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)

### TL;DR
MLP over all patches. The fancy query bottleneck was not load-bearing
once you can afford the extra visual tokens in the LLM context.

---

## LOTUS / Coconut / Pfau — what "latent reasoning" came to mean

Do not merge into this file; see
[`latent_set_refinement.md`](latent_set_refinement.md) and
[`recurrent_memory_transformers.md`](recurrent_memory_transformers.md).

The naming collision is the whole point:

| Name people use | Shape | Supervised how | What it can do |
|---|---|---|---|
| Perceiver "iterative attention" | Mix a **set** of N pages, same N | Final loss only (class / MLM / CE) | Perception, gist, binding if competitive |
| Huginn / Ouro | Loop **depth** on every token | Next-token CE | Extra FLOPs; mixed evidence of CoT-like structure. Huginn probe ([arXiv:2507.02199](https://arxiv.org/abs/2507.02199)): GSM8K with CoT off 3.11 → 4.93 (4 vs 32 recurrences) vs **24.9 / 38.1 with explicit CoT**. |
| Coconut | **Chain** of hidden states | Curriculum from token CoT | Search-ish tasks at small scale; GSM flat; collapse as K grows without step CE |
| LOTUS | Set of latent *blocks*, looped, **gold CoT CE on each** | Parallel process supervision | Matches explicit CoT at 3B; `R=2` is 14.6%, `R=6` is 70% |
| Token CoT (o-series, Thm 8) | Append **tokens** to a tape | Token CE | The thing that actually shipped |

Unsupervised self-attention on Z sits in row 1. It is not rows 3–5.

---

## What I would change in Perceiver IO (2026)

Keep the one load-bearing idea: **asymmetric cross-attention as a
compressor** (`O(M N)`, depth decoupled from M).

Drop the story that the latent Transformer is a mind.

**Adapt — one architecture, not a knob list:**

1. **Z is memory, addressed by position.** One slot per block of `r`
   tokens (DeepSeek HCA / Beacon / LCLM), not N free learned queries.
   Allocation is structural; starvation is no longer a training
   accident.
2. **Contextualize X before the bottleneck.** A windowed / causal
   encoder (LCLM `W=1024`, DeepSeek's deep encoder, BiXT) then pool.
   Perceiver's first CA on raw bytes + Fourier was a vision trick.
   Language tokens that have never talked to their neighbors are a
   bad K/V.
3. **Do not mix slots unless the task is set-binding and each slot has
   a target.** Default adapter = MLP (LCLM). Slot–slot SA is Watch,
   gated on a MATCH/bind exam, not on CE.
4. **Keep a raw/local path for INDEX.** Finite-state / exclusive
   notebooks fail exact retrieval (Qwen; gist "lost if surprise").
   Compression is for gist. Copy needs an addressable token or `r=1`.
5. **Reasoning is a tape.** If you want serial computation, append
   positions: token CoT, or LOTUS-style latent blocks with **per-step
   gold CE**. Encode-once Z + extra SA is BAPO Thm 10. Re-read X
   (original Perceiver interleaved CA) is allowed as *memory refresh*,
   not as a substitute tape.
6. **Close the bypass for the channel you claim.** Decoder must not
   see raw X for the loss that is supposed to train Z. Supervise the
   notebook: prefix reconstruction through a *weak* decoder, or
   per-slot CE — not class-mean and not answer-only.
7. **Carry Z as KV, not as a pooled embedding** (500xCompressor).
   Modern kernels (FlexAttention, GQA/MLA) make a short KV of slots
   the native object; a 512×1024 mean vector is 2016 ResNet-head
   thinking.
8. **Engineering 2026:** PyTorch + FlexAttention, QK-norm, RoPE *on
   slots that have positions*, residual stream on the token decoder.
   Do not rebuild a JAX latent Transformer to chase 2021 ImageNet.

Kill signals for this Adapt: (a) slot–slot SA helps CE but not a
BAPO-hard bind exam — then it is mixing, delete it; (b) exclusive Z
without a raw path fails INDEX at the same `r` that gist survives —
then you have a compressor, not a reasoner, and should name it that
way; (c) extra tied loops without per-slot CE are flat vs `K=1` —
then you have Veit residual skip, not LOTUS.

**Verdicts**

- Asymmetric CA compressor: **Adopt** (already the field's consensus
  remnant).
- Learned unallocated Z₀ + deep latent TF as "reasoning": **Reject**.
- Interleaved re-read of X: **Watch** for memory refresh, not for CoT.
- LOTUS-style supervised set loops: **Adapt** if the exam is
  multi-hop / planning; not as a language-model default.
- Positional KV slots + MLP + local raw path: **Adapt** — this is what
  scaled.

---

## Cross-cutting

1. **Perception ≠ reasoning.** A notebook that must *summarize* a
   photo for a class label should ignore most pixels. A notebook that
   must *prove* a three-hop fact must not. Perceiver optimized the
   first.
2. **Bandwidth is not FLOPs.** 48 latent blocks on 512 slots is a lot
   of multiply-adds and still a constant-width bus (BAPO Thm 10).
3. **The adopted fragment was the read, not the think.** Flamingo
   kept CA-to-queries. LLaVA dropped even that. Nobody kept "48-layer
   GPT-2 on 512 latents" as the mind.
4. **Unsupervised refinement of a set is the attractor that does not
   work for LM.** LCLM MLP > SA; LOTUS `R=2` ≈ fail; Huginn mixed;
   Coconut without step CE collapses. The 2021 aesthetic (iterate Z)
   lost every clean A/B that used next-token CE as the judge.
5. **Missing experiment, not a hidden knockout.** There is still no
   public 1B–7B matched-token Perceiver AR vs GPT recipe on CoT /
   BAPO-hard tasks. The adoption argument is ecosystem + theory +
   VLM ablations, not a single fair language bake-off at modern scale.

---

## Source index

| Paper | URL |
|---|---|
| Perceiver | https://arxiv.org/abs/2103.03206 |
| Perceiver IO | https://arxiv.org/abs/2107.14795 |
| Perceiver AR | https://arxiv.org/abs/2202.07765 |
| Released JAX | https://github.com/google-deepmind/deepmind-research/tree/master/perceiver |
| Veit residual ensembles | https://arxiv.org/abs/1605.06431 |
| FlashAttention | https://arxiv.org/abs/2205.14135 |
| Flamingo | https://arxiv.org/abs/2204.14198 |
| BLIP-2 | https://arxiv.org/abs/2301.12597 |
| LLaVA | https://arxiv.org/abs/2304.08485 |
| Merrill–Sabharwal CoT | https://arxiv.org/abs/2310.07923 |
| Pfau filler tokens | https://arxiv.org/abs/2404.15758 |
| BAPO | https://arxiv.org/abs/2505.08140 |
| Sanford MATCH | https://arxiv.org/abs/2306.02896 |
| Gist failure analysis | https://arxiv.org/abs/2412.17483 |
| LCLM | https://arxiv.org/abs/2606.09659 |
| LOTUS | https://arxiv.org/abs/2606.31779 |
| Coconut | https://arxiv.org/abs/2412.06769 |
| 500xCompressor | https://arxiv.org/abs/2408.03094 |
| Latent-CoT supervision | https://arxiv.org/abs/2606.20075 |
| Wang architecture/objective | https://arxiv.org/abs/2204.05832 |
| Gopher | https://arxiv.org/abs/2112.11446 |
| IDEFICS2 | https://arxiv.org/abs/2405.02246 |
| MiniGPT-4 | https://arxiv.org/abs/2304.10592 |
| Huginn latent-CoT probe | https://arxiv.org/abs/2507.02199 |
| YOCO | https://arxiv.org/abs/2405.05254 |
| DeepSeek-V2 MLA | https://arxiv.org/abs/2405.04434 |
| Gist tokens | https://arxiv.org/abs/2304.08467 |
| Perceiver AR JAX | https://github.com/google-research/perceiver-ar |
