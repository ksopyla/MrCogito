# Recurrent / writable memory transformers — the read/write memory axis

Reviews of transformer architectures that keep a small set of latent "memory" /
"concept" vectors, organized along the single load-bearing axis for MrCogito:
**is the memory read-only (a frozen snapshot computed once) or read+write
(a recurrent state updated as tokens are processed)?**

This is the axis that decides whether the model's compressed context can *evolve*
during long generation / reasoning, or is frozen at encode time — the question
behind the E05/E02-long generation review (concepts are computed once by the
encoder and never updated during decoding; see
`nn/concept_encoder_perceiver.py` `ConceptCausalDecoderStack.forward`).

## How to read this file

The field splits into four families:

- **A. Frozen / read-only latents** (the current MrCogito camp): AutoCompressor,
  Perceiver / Perceiver AR, Landmark Attention. Memory computed once, cross-attended
  without modification.
- **B. Writable + differentiable recurrent memory** (the "running memory" family):
  Block-Recurrent Transformer, Recurrent Memory Transformer (RMT), Coconut, Huginn,
  Ouro, LoopFormer. Memory is the output of one segment/step and the input to the
  next; trained end-to-end by backprop through the unrolled recurrence.
- **C. Writable + non-differentiable external memory**: Memorizing Transformers,
  Compressive Transformer. Write is an append / fixed compression; only the read is
  learned. Can be retrofitted onto a pretrained backbone as retrieval.
- **D. Differentiable in-place memory update (hybrid)**: Infini-attention. A
  compressive memory matrix updated in-place by a linear-attention rule inside the
  same block that does causal attention. Closest published design to "refresh the C
  concepts with a learned rule as you generate" — and the source of the strongest
  "iterated small memory degrades" evidence.

Related reviews already in the repo (do not duplicate): the progenitor
**Memory Transformer (2020)** is reviewed in
[`concept_modeling_encoding.md#memory-transformer`](concept_modeling_encoding.md);
the broad comparative placement (RMT, HMT, Perceiver, CEPE, Transformer-XL) is in
[`concept_transformer_ai_report.md`](concept_transformer_ai_report.md) §II.B / §III.

---

## A. Frozen / read-only latents (current MrCogito camp)

### AutoCompressor — "Adapting Language Models to Compress Contexts"
EMNLP 2023 · https://arxiv.org/abs/2305.14788 · Alexis Chevalier, Alexander Wettig,
Anirudh Ajith, Danqi Chen (Princeton NLP) · code: https://github.com/princeton-nlp/AutoCompressors

#### TL;DR
Fine-tune a pretrained LM to compress a long context segment into a small fixed
number of **summary vectors** that are passed as soft prompts to later segments.

#### The problem that authors want to solve
Long-context LMs are expensive at inference; can a model learn to compress a long
context into a few vectors that preserve enough signal for downstream prediction?

#### The solution, main idea on the intuition level and strong points
A segment is processed and its last `N` (typically 50) hidden states are extracted
as **summary vectors**; in subsequent segments these are prepended as soft prompts.
Memory mechanism = **read-only at inference**: each summary vector is computed once
for its segment and then frozen for downstream use. A "summary accumulation" trick
passes summary vectors from *all* previous segments forward — but they are still
snapshots, not a running state. This is the closest existing analogue to MrCogito's
"C=128 frozen concepts cross-attended by a decoder."

#### The detailed solution, training process, data preparation
Differentiable through the segment chain (BPTT over up to 30,720 tokens), unsupervised
LM objective. Pretrained OPT-1.3B/2.7B and Llama-2-7B fine-tuned into AutoCompressors.
The summary-vector production is a learned skill — cannot be bolted on at inference.

#### The evaluation procedure, evaluation datasets and results
PPL improvements on PG-19 and arXiv; in-context-learning demonstrations compressed to
summary vectors *improve* accuracy over plain-text demos at lower inference cost;
applied to retrieval-augmented LM and passage re-ranking.

#### Relevance to MrCogito
This is functionally what MrCogito's encoder + frozen-concept decoder already is — a
static snapshot compressor. AutoCompressor's documented limitations (summary vectors
leak less than full text; gains saturate as summary-vector count grows; needs
fine-tuning at target segment length) are the same wall MrCogito hits at generation
time: a snapshot helps retrieval/classification but does not sustain open-ended
generation. Verdict: **the baseline we already embody** — evidence that the frozen
camp is genuinely limited for generation, motivating the recurrent alternatives below.

---

### Perceiver / Perceiver AR
ICML 2021 / ICML 2022 · Perceiver: https://arxiv.org/abs/2103.03206 · Perceiver AR:
https://arxiv.org/abs/2202.07765 · Jaegle et al. / Hawthorne, Jaegle et al. (DeepMind) ·
code: https://github.com/google-research/perceiver-ar (JAX/Haiku); HF `deepmind/perceiver-io`

#### TL;DR
Asymmetric cross-attention between a small learned latent array (N positions) and a
large input byte array (M ≫ N) decouples compute from input length.

#### The solution, main idea on the intuition level and strong points
Memory mechanism = **read-only, non-recurrent**. The latent array is initialized as a
learned parameter (Perceiver) or a position-indexed query (Perceiver AR) and is
*recomputed fresh at every forward pass* by attending to the current input. It is
**not carried forward across generation steps.** "Iterative" in the Perceiver title
means alternating cross- and self-attention *within a single forward pass*, not across
generation steps. Perceiver AR is autoregressive but the latents are regenerated at
each position, not a running memory.

#### The detailed solution, training process, data preparation
End-to-end differentiable; latents trained from scratch with the rest of the model.
Cannot be bolted on at inference — the cross-attention *is* the model.

#### The evaluation procedure, evaluation datasets and results
ImageNet, AudioSet, ModelNet40, long-form music generation (Magenta, 65k tokens).
For text LMs the latent bottleneck is hard to train well; Perceiver AR underperforms
standard transformers on LM PPL at equal compute. Dominant where input is huge and
structured (audio, images, point clouds).

#### Relevance to MrCogito
MrCogito's encoder (BiXT cross-attention from learnable concept queries to tokens) is a
Perceiver-style bottleneck — confirming the one-shot-compression design is sound *for
its purpose*. The Perceiver AR result (underperforms on text LM PPL) is a direct
warning that a frozen latent bottleneck is an uphill battle for pure language modeling,
consistent with E05's weak generation.

---

### Landmark Attention — "Random-Access Infinite Context Length for Transformers"
NeurIPS 2023 · https://arxiv.org/abs/2305.16300 · Amirkeivan Mohtashami, Martin Jaggi (EPFL) ·
code: https://github.com/epfml/landmark-attention

#### TL;DR
Insert one **landmark token per block** whose representation summarizes the block; the
model learns to attend to landmarks to decide which blocks to retrieve — random-access
"infinite" context.

#### The solution, main idea on the intuition level and strong points
Memory mechanism = **read-only KV-cache retrieval, not a learned latent update.**
Landmarks live in the KV cache alongside their blocks; the only "write" is appending
new KV blocks and computing their landmark summary as part of the standard attention
forward. No in-place update of past memories — past KV is frozen. Essentially learned
sparse retrieval over a frozen KV store.

#### The detailed solution, training process, data preparation
Differentiable end-to-end (block-selection is a soft attention over landmarks); trained
on LLaMA-7B at 32k. Training context length decouples from inference context length.

#### The evaluation procedure, evaluation datasets and results
Scales LLaMA 4k → 32k; matches full-attention quality on long-context benchmarks at
lower memory. "Infinite" is marketing — demonstrated to 32k; KV cache still grows
linearly (with selective offload).

#### Relevance to MrCogito
A different route to "long context through a bottleneck": rather than compressing into
C concept vectors, learn to *retrieve* frozen KV blocks. Orthogonal to MrCogito's
lossy-compression claim (Landmark preserves everything, just sparse-attends) — useful
as a contrast, not a template.

---

## B. Writable + differentiable recurrent memory (the "running memory" family)

### Block-Recurrent Transformer
NeurIPS 2022 · https://arxiv.org/abs/2203.07852 · DeLesley Hutchins, Imanol Schlag,
Yuhuai Wu, Ethan Dyer, Behnam Neyshabur (Google/IDSIA) · ports:
https://github.com/lucidrains/block-recurrent-transformer-pytorch

#### TL;DR
Apply a single transformer layer **recurrently across blocks** of tokens, where the
layer's input includes both the current token block and a large set of **state
vectors** carried from the previous block, and its output includes the updated state
vectors for the next block — "an LSTM cell scaled up by several orders of magnitude."

#### The problem that authors want to solve
Give a transformer a persistent, differentiable working memory that accumulates
information across blocks without O(N²) attention over the whole history.

#### The solution, main idea on the intuition level and strong points
Memory mechanism = **read AND write, fully differentiable.** State vectors are part of
the layer's output and are fed back as input on the next block. Read = self-attention +
cross-attention from tokens to state; **write = LSTM-style gated update** of the state
vectors, produced as the layer's output. The paper explicitly says the model is
"trained to control both memory operations and sequence representation processing."
**This is the closest existing design to a "running C-concept memory."**

#### The detailed solution, training process, data preparation
Standard BPTT through the unrolled recurrence: within a block, computation is parallel;
across blocks, gradients flow back through time. Trained from scratch on PG-19, arXiv
papers, GitHub code. Memory operations are *learned* (LSTM-style gates). Cannot be
bolted on at inference.

#### The evaluation procedure, evaluation datasets and results
Outperforms Transformer-XL by a wide margin on PG19/arXiv/GitHub at ~half the
wall-clock time; effective memory over ~60k tokens; linear complexity in sequence
length.

#### Relevance to MrCogito
The single most important comparison for the "should concepts be updatable" question.
The **gated write** is the known antidote to the collapse risk (see Infini-attention
below): a running concept memory must update via gates, not overwrite. Limitation:
trained from scratch in the original — for MrCogito, warm-starting from E02-long +
adding a Block-Recurrent-style gated state would be the faithful test. Verdict:
**Adapt** — the architectural template for a writable-concept variant.

---

### Recurrent Memory Transformer (RMT)
NeurIPS 2022 · https://arxiv.org/abs/2207.06881 · Aydar Bulatov, Yuri Kuratov,
Mikhail Burtsev · code: https://github.com/booydar/recurrent-memory-transformer
(official, HF-wrapper); AAAI 2024 follow-up (1M tokens on pretrained BERT):
https://ojs.aaai.org/index.php/AAAI/article/view/29722/31239

#### TL;DR
Add special `[mem]` tokens to the input/output; the model writes the segment summary
into the output `[mem]` slots and reads the previous segment's `[mem]` slots at the
input — recurrence with **no architectural change**, just by treating segment `t-1`'s
output memory tokens as segment `t`'s input memory tokens.

#### The solution, main idea on the intuition level and strong points
Memory mechanism = **read AND write, fully differentiable.** The model writes memory by
producing output `[mem]` representations; reads by attending to input `[mem]` tokens.
Memory state is the small set of `[mem]` vectors. Crucially, RMT is implemented as a
**wrapper around any HF encoder/decoder** — the underlying transformer is unchanged.
**This is essentially "what if MrCogito's C concept vectors were `[mem]` slots refreshed
by the decoder every K generated tokens."**

#### The detailed solution, training process, data preparation
BPTT across segments. Originally trained from scratch, but the AAAI 2024 follow-up shows
the wrapper **can be fine-tuned onto a pretrained model** (applied RMT-style memory to a
pretrained BERT, scaled to 1M+ tokens). So unlike Block-Recurrent, RMT *can* be
retrofitted onto a pretrained backbone with fine-tuning.

#### The evaluation procedure, evaluation datasets and results
On par with Transformer-XL at small memory sizes; outperforms at longer sequences;
algorithmic/reasoning tasks (copying, associative recall) show clear benefits; the AAAI
follow-up demonstrates 1M-token context.

#### Relevance to MrCogito
The **cheapest faithful test of the writable-concept idea**: warm-start from E02-long
(pretrained backbone) and add `[mem]`/concept slots the decoder refreshes every K
tokens, fine-tuning only the memory machinery — no from-scratch training. Limitations:
fixed `[mem]` slot count; writing is lossy compression; high-density long contexts
overwhelm small memory. Verdict: **Adapt** — the practical on-ramp.

---

### Coconut — "Training Large Language Models to Reason in a Continuous Latent Space"
COLM 2025 · https://arxiv.org/abs/2412.06769 · Shibo Hao, Sainbayar Sukhbaatar, DiJia Su,
Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian (Meta/FAIR + UCSD) · code:
https://github.com/facebookresearch/coconut

#### TL;DR
Replace discrete chain-of-thought tokens with **continuous thought** vectors — the
model's last hidden state is fed back as the next input embedding, looping the LLM on
itself in latent space until it switches back to emitting tokens.

#### The solution, main idea on the intuition level and strong points
Memory mechanism = a **single hidden-state vector** (or small set per position) that is
**read AND written each step** of the latent loop — the canonical "running latent state"
model. Exit from the latent loop is controlled by a learned `<bot>`/`<eot>`-style
decision. Headline claim: continuous thought can encode *multiple alternative next
steps* (implicit BFS in latent space).

#### The detailed solution, training process, data preparation
**Multi-stage curriculum** — start from full CoT SFT, gradually replace explicit CoT
tokens with continuous-thought steps, then fine-tune end-to-end with the continuous
loop. BPTT through the latent steps; fully differentiable except the discrete exit
decision (curriculum + teacher forcing). Must be trained from a strong CoT baseline;
cannot be bolted on at inference.

#### The evaluation procedure, evaluation datasets and results
Beats CoT on ProsQA, ProntoQA (logical reasoning with substantial search); better
accuracy/latency trade-off than CoT. Demonstrated at 1.4B scale; BFS interpretation is
partly conceptual; interpretability concerns (no human-readable trace).

#### Relevance to MrCogito
Coconut's "memory" is a single hidden state, not a *set* of C concept vectors — but its
**curriculum (start from explicit token traces, gradually replace with continuous-latent
steps) is the most concrete published recipe for training a running-latent system
without collapse.** It overlaps directly with E08's reasoning-trace distillation, so the
writable-concept idea and E08 compose. Verdict: **Adapt (methodology)** — borrow the
curriculum, not the single-vector state.

---

### Huginn — "Scaling by Thinking in Continuous Space"
ICLR 2025 · https://arxiv.org/abs/2502.05171 (model) · skeptical analysis:
https://arxiv.org/abs/2507.02199 · Tom Goldstein group (UMD) + SEAL-RG · model
`tomg-group-umd/huginn-0125` (3.5B) · code: https://github.com/seal-rg/recurrent-pretraining

#### TL;DR
A **depth-recurrent** transformer that reuses the *same block of layers* `T` times at
inference before emitting each token; `T` can be increased at test time to scale compute.

#### The solution, main idea on the intuition level and strong points
Memory mechanism = the hidden-state vector at each position is **read AND written** by
each iteration of the shared block — the running-state version of Universal Transformer.
Unlike RMT, recurrence is *within a single forward pass* (depth-recurrence), not across
segments.

#### The detailed solution, training process, data preparation
Pretrained from scratch with per-token random recurrence depth; fully differentiable BPTT
through the unrolled depth.

#### The evaluation procedure, evaluation datasets and results
3.5B model on 800B tokens; matches/beats fixed-depth basins at higher test-time depth.
The analysis paper (2507.02199) probes whether genuine latent CoT emerges — **mixed
evidence** (some structured emergence, not the clean interpretable CoT hoped for).
"When Depth Adds Nothing" (OpenReview) shows regimes where added depth doesn't help.

#### Relevance to MrCogito
Cautionary tale: even at 3.5B / 800B tokens, depth-recurrent latent reasoning gives
ambiguous gains and the community is skeptical it produces interpretable reasoning.
Reinforces that a writable/iterated concept memory needs careful evaluation (the
agenda's "audit-resistant depth bench") — don't assume iteration = reasoning.

---

### Ouro — "Scaling Latent Reasoning via Looped Language Models"
arXiv:2510.25741 (Oct 2025) · https://arxiv.org/abs/2510.25741 · code:
https://github.com/rkstgr/LoopLM · project: https://ouro-llm.github.io/ · stability
follow-ups: https://arxiv.org/abs/2605.18797 (Ouro-SFT/STARS), https://arxiv.org/abs/2509.23314

#### TL;DR
A family of **pretrained Looped Language Models (LoopLM)** that reuse shared layers `K`
times per token; positions reasoning as a *third scaling axis* (vs. parameters and
data). Ouro-"Thinking" uses K=4 recurrent steps at inference.

#### The solution, main idea on the intuition level and strong points
Depth-recurrent running latent state (same family as Huginn), read+written each loop.
Contribution is showing pretraining-with-looping is feasible at scale; introduces a
"gate training" mechanism that learns when looping helps.

#### The detailed solution, training process, data preparation
Pretrained from scratch with looping + gate training.

#### The evaluation procedure, evaluation datasets and results
Reports math/reasoning gains at 4 recurrent steps. **Follow-up papers introduce
Ouro-SFT / Ouro-STARS specifically to address instability across recurrent steps** —
direct evidence that bare recurrence has stability problems.

#### Relevance to MrCogito
The stability follow-ups are the key takeaway: **looped latent state is unstable by
default and needs dedicated stabilization** (sandwich-RMSNorm, entropy depth allocator,
gate training). For MrCogito this means a writable concept memory must ship with
anti-collapse machinery from day one — and the agenda already earmarks exactly these
(Ouro's recipe) for E08.

---

### LoopFormer — "Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation"
ICLR 2026 · https://arxiv.org/abs/2602.11451 · project: https://loopformer.github.io/
(code not yet located)

#### TL;DR
Adds **shortcut modulation** (borrowed from diffusion models) so the same looped
transformer can run at variable depth (K from 1..max) at inference — elastic depth —
without quality loss at low K.

#### The solution, main idea on the intuition level and strong points
Depth-recurrent running latent state (same family as Huginn/Ouro). Variable-length
trajectory training + shortcut-consistency loss so the model is trained across many K
simultaneously; fully differentiable.

#### Relevance to MrCogito
Mostly relevant as evidence that the depth-recurrent/looped family is an active ICLR-2026
research direction with unresolved stability/elasticity questions — not a ready template.
The agenda already cites LoopFormer as evidence that "bare recursion amplifies collapse."

---

## C. Writable + non-differentiable external memory

### Memorizing Transformers
ICLR 2022 · https://arxiv.org/abs/2203.08913 · Yuhuai Wu, Markus Rabe et al. (Google) ·
paper page: https://huggingface.co/papers/2203.08913

#### TL;DR
Augment attention with an **external non-differentiable kNN memory** of past (key,
value) pairs; retrieve top-k via approximate kNN and gate them into attention.

#### The solution, main idea on the intuition level and strong points
Memory mechanism = **read = differentiable attention over retrieved KVs; write =
non-differentiable append** of new (K, V) pairs into the external index. The model never
*learns* to write — it appends everything; only the retrieval gate is learned. This is
the canonical "memory as a non-differentiable append" and the **opposite design pole
from MrCogito's C=128 concepts.**

#### The detailed solution, training process, data preparation
No backprop through memory writes — memory is a data structure, not a learned state.
Crucially this means memory **can be bolted on at inference**: take any pretrained
transformer, wrap a layer with kNN external memory, fine-tune briefly (the paper does
this on top of a pretrained model).

#### The evaluation procedure, evaluation datasets and results
PPL improvements on PG19, arXiv, GitHub; ablation shows the gate is essential (without
it, retrieved noise hurts).

#### Relevance to MrCogito
A contrast pole: this family preserves everything (append-only) and learns only to
retrieve — it sidesteps collapse by never compressing, at the cost of growing memory and
no learned curation. It would break MrCogito's compression invariant (the architecture
invariants rule out "retrieval over raw tokens"), so it is a **Reject** as a design, but
useful as the "no-bottleneck" counterfactual.

---

### Compressive Transformer
ICLR 2020 · https://arxiv.org/abs/1911.05507 · Jack Rae, Anna Potapenko, Siddhant Jayakumar,
Tim Lillicrap (DeepMind) · port: https://github.com/lucidrains/compressive-transformer-pytorch

#### TL;DR
Extension of Transformer-XL where the oldest attention memories are **compressed**
(pooling or 1D conv, optionally trained) before being moved into a second "compressed
memory" cache, enlarging the effective window beyond the raw KV cache.

#### The solution, main idea on the intuition level and strong points
Memory mechanism = **read = standard attention over raw + compressed memory; write =
deterministic movement of old KV into the compressed cache with a fixed (or
separately-trained) compression function.** The compression op is *not* learned
end-to-end against the LM loss in the basic version (training it gave only modest
gains) — so essentially "non-differentiable write at the boundary of two caches."

#### The detailed solution, training process, data preparation
Compression function is fixed (pooling/conv) or trained as an auxiliary autoencoder
objective — not backprop through the LM loss. Can be retrofitted.

#### The evaluation procedure, evaluation datasets and results
SOTA on PG19, WikiText-103, Enwik8 at the time; ablation: compressed memory gives
meaningful PPL gains over Transformer-XL.

#### Relevance to MrCogito
A middle ground: compress old memory, but with a fixed/non-learned op. Suggests that even
a non-learned compression of aging memory helps long-context PPL — a cheap baseline if
MrCogito ever wants a "compressed cache of old generated tokens" alongside the C concepts
without a learned write-back.

---

## D. Differentiable in-place memory update (hybrid)

### Infini-attention — "Leave No Context Behind"
Google, 2024 · https://arxiv.org/abs/2404.07143 · Tsendsuren Munkhdalai, Manaal Faruqui,
Siddharth Goyal (Google) · HF reproduction (failure analysis):
https://huggingface.co/blog/infini-attention

#### TL;DR
Fuse a **compressive memory matrix** (updated by a linear-attention rule) into a standard
causal attention block, so the same block does both masked local attention and a
*differentiable* in-place update of a long-term memory state.

#### The solution, main idea on the intuition level and strong points
Memory mechanism = **both differentiable.** Read = σ(query · memory_key) over the
compressive matrix; **write = `memory ← memory + σ(key) · value`** (linear-attention
in-place update). A learned gate combines local causal attention with the long-term
memory output. **This is the closest published design to "refresh the C concept vectors
with a learned differentiable rule as more tokens are generated."**

#### The detailed solution, training process, data preparation
Differentiable end-to-end; trained from scratch on 1B-param Llama-style models on long
sequences.

#### The evaluation procedure, evaluation datasets and results
**CRITICAL failure-mode evidence:** Hugging Face's independent reproduction
(huggingface.co/blog/infini-attention) reports that **"long context performance decreases
as the number of times we compress the memory"** increases, and that the gating parameter
is highly sensitive. HF could not reproduce the original paper's pretraining results.

#### Relevance to MrCogito
The single most decision-relevant paper for the "should concepts be updated?" question.
It confirms two things at once: (1) a differentiable in-place memory update is
*architecturally feasible* inside an attention block; (2) **repeated differentiable
compression of a small memory state degrades long-context performance** — i.e., the
naïve writable-memory fix can make things *worse*, the same collapse failure MrCogito
already fights. Caveat: Infini-attention uses a *linear-attention* update (restricted
expressiveness); whether a richer gated update (Block-Recurrent-style) collapses the
same way at modern scale is untested publicly — a research gap. Verdict: **Adapt with
extreme caution** — the existence proof that the mechanism is buildable, and the warning
that it must be gated/stabilized.

---

## Cross-cutting findings for MrCogito

1. **The read/write axis is the decision.** MrCogito currently sits in family A
   (frozen/read-only, AutoCompressor/Perceiver camp). The "should concepts be updated?"
   question is exactly: *should the C concept set move from family A to family B
   (writable, differentiable, gated recurrence)?* Families C and D are intermediate
   options.
2. **The write must be trained, never bolted on at inference.** Every differentiable
   recurrent-memory design (Block-Recurrent, RMT, Coconut, Huginn, Ouro, Infini-attention)
   trains the write operator end-to-end. Untrained write-back at inference produces
   garbage. So a writable-concept variant is a *new training regime*, not an inference
   tweak.
3. **Recurrence can amplify collapse — MrCogito's central enemy.** Strongest evidence:
   Infini-attention's HF reproduction (repeated compression degrades performance).
   Corroborating: Ouro's stability follow-ups (Ouro-SFT/STARS), Huginn's skeptical
   analysis, Dong et al. 2021 (arXiv:2102.11242). The known antidotes are **gated
   writes** (Block-Recurrent) + **stabilization machinery** (Ouro's sandwich-RMSNorm /
   entropy depth allocator) + **curriculum** (Coconut's explicit-trace → latent
   transition). MrCogito's agenda already earmarks these for E08.
4. **The cheapest on-ramp is RMT-on-a-pretrained-backbone.** The AAAI 2024 follow-up
   fine-tuned RMT-style `[mem]` tokens onto a pretrained BERT and reached 1M tokens.
   MrCogito could warm-start from E02-long (STS-B 0.714) and add a small `[mem]`/concept
   slot set the decoder refreshes every K tokens, fine-tuning only the memory machinery.
5. **There is a research gap = MrCogito's opportunity.** No published work runs a
   controlled "frozen C-concept set vs running C-concept set" comparison at modern scale
   for generation coherence/repetition. AutoCompressor (frozen) and RMT (running) are the
   closest pair but from different groups, at small scale, on different benchmarks. A
   controlled A/B is exactly what MrCogito's methodology is built to do.
6. **Diagnostic before training.** Before any writable-memory experiment, measure the
   symptom directly: suffix cross-entropy and free-running repetition/distinct-n as a
   function of position relative to the window K. A rising curve past K quantifies that
   "frozen memory + K-window" cannot sustain generation and justifies the experiment;
   a flat curve falsifies the hypothesis.
