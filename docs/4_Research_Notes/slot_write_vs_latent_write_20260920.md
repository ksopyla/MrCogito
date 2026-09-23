# Slot write vs latent write — which compressed object should carry the long range?

**Written:** 2026-09-20 · dated research note (append-only: later notes may supersede the
*interpretation*, never the record) · author: Cursor session for KS.

**Question asked (KS, 2026-09-17/20).** E21's "slots" and a Perceiver-style latent vector are
both called a notebook. Are they the same object? Which one should carry the long range, and why?
Analyse the literature, gradient flow, information flow, bandwidth, and theory.

**Grounding.** Code: `nn/perceiver_ar_lm.py` (exclusive compressed read / `KVCompressor`),
`nn/perceiver_concept_lm.py` (positional concept array + latent transformer),
`nn/concept_encoder_perceiver.py` (classic Perceiver IO decode). Ledger: E22, E25, E26, E27,
E28, E29. Literature reviews: [`reasoning_bandwidth_information_flow.md`](../literature_review/reasoning_bandwidth_information_flow.md),
[`information_bottleneck_latent_capacity.md`](../literature_review/information_bottleneck_latent_capacity.md),
[`learned_kv_context_compression.md`](../literature_review/learned_kv_context_compression.md),
[`latent_set_refinement.md`](../literature_review/latent_set_refinement.md),
[`perceiver_io_latent_reasoning_critique.md`](../literature_review/perceiver_io_latent_reasoning_critique.md).
Prior notes: [E22 root cause](e22_root_cause_20260912.md),
[concept channel cold start](concept_channel_cold_start_20260913.md),
[perceiver revisit](perceiver_revisit_synthesis_20260912.md),
[E18/E21 vs dense diagnosis](e18_e21_dense_baseline_diagnosis_20260916.md).

---

## 0. Verdict in six lines

1. **"Slots" and "latents" are not two names for one object.** There are three, and the argument
   about which is "better" only makes sense once they are separated.
2. **A uniform mean of a token block is the wrong sufficient statistic for lookup**, regardless of
   how wide the vector is. It is a good statistic for gist and for shrinking a KV cache.
3. **A free learned-query latent array over the whole sequence is the wrong bandwidth story**
   (fixed `C`, slots compete for the whole book) and its "reasoning" stage is optional by
   construction.
4. **The write mechanism matters more than the container.** Positional allocation + a *learned
   query inside the block* + contextualise-before-pooling is the combination that shipped at scale;
   the container (hidden-space latent vs K/V slot) is secondary.
5. **Our own four attempts to repair the pooled write all failed** (E26 prefix AE, E27 hybrid key
   spans, E28/E29 paying packed exams). Frozen-mean `r=16` carries copy; it does not carry lookup.
6. **Slot-to-slot "reasoning" is gated, not next.** Every clean A/B of it under language CE
   (LCLM; our E22) says it does not pay until the slots already carry addressable facts.

---

## 1. Three objects, not two

| # | object | where it is written | who reads it | does it mix? |
|---|---|---|---|---|
| **A** | **KV slot** (E21 exclusive compressed read) | mean/pool of `r` tokens in the global layer's K/V space (`KVCompressor`, `nn/perceiver_ar_lm.py:855`) | the same layer's attention, exclusive of raw prefix | **no** — `message_extra_slot_attends` defaults to 0 |
| **B** | **positional latent array** (E22 concept LM) | `z0_j = mean(h_enc[block j]) + XAttn_c(q_learned → block j)`, one slot per 16 tokens (`nn/perceiver_concept_lm.py:237`) | a latent transformer, then a segment-confined decoder | **yes** — 4 causal layers over the array |
| **C** | **classic Perceiver latents** | `C` learned queries cross-attend the **whole** sequence (`nn/concept_encoder_perceiver.py:68`) | Perceiver IO decode queries, or a class head | optional self-attn on the array |

A and B share "one page per 16 tokens". A and C share "attention-pooled". B and C share "a
transformer over the array". **None of the three is the same object**, and the project has run all
three: A in E25–E29, B in E22, C in E01–E05 and in the concept-MLM line.

```
tokens ─► windowed mix
             │
    ┌────────┼──────────────────┬────────────────────┐
    ▼        ▼                  ▼                    ▼
 mean of 16 K/V          mean + 1 learned       C learned queries
 (A: cache page)          query per block        over whole book
    │                     (B: latent page)       (C: free latents)
    ▼                          │                    │
 answer attends K/V      latent transformer    class head / IO decode
 no page↔page talk       page↔page talk        optional self-attn
```

## 2. Bandwidth — what each can carry (BAPO)

BAPO ([arXiv:2505.08140](https://arxiv.org/abs/2505.08140)) separates two channels from the prefix
to the answer-emitting position: `b` = how many **raw** tokens it may address, `a` = the bits in a
summary it may read. The classes:

- **INDEX / copy** is BAPO-easy `(a≈0, b=1)`: one addressable gist is enough. Our DNA `far_copy`
  confirms it — 16-token means still copy a 64-bit span.
- **MATCH / lookup of one fact among `n`** needs `a · b = Ω(n)` — a **count of independently
  addressable items**, not a fatter summary. DNA `recall_single` confirms the failure: `r=16` frozen
  mean **0 bits**, `r=1` identity **43.08 bits** (E25).
- **MAJORITY** needs `a = Ω(log n)` bits with `b = 0`.

This is the core of the answer. A mean-pooled block is **one** address whose content is a mixture.
Compression ratio `r` therefore sets an *upper bound on addressable item count* (`n/r`), while the
mixture sets the *quality of each address* — and the mixture is what kills lookup. Widening the slot
(`head_dim`, `g`, hidden size) does not change the count.

BAPO **Theorem 10** also matters here: adding layers or heads at a *fixed* bottleneck does **not**
raise effective prefix bandwidth. That is the theoretical reason a deeper latent transformer over a
fixed array is not a bandwidth fix.

## 3. Capacity — count vs width, and the right sufficient statistic

From [`information_bottleneck_latent_capacity.md`](../literature_review/information_bottleneck_latent_capacity.md):

- **Rate–distortion view.** The bottleneck keeps `I(Z;Y)` and drops the rest of `I(X;Z)`
  ([Tishby 1999](https://arxiv.org/abs/physics/0004057)). A uniform mean minimises squared
  reconstruction error over the block — it is the optimal *typical* summary. A lookup key is not a
  typical summary: the answer needs the identity of one item, which a mean deliberately averages
  away. Wrong sufficient statistic, not insufficient width.
- **Count starvation for exact rehearsal.** ICAE: `512→128` works, `k=64/32` fail lossless; random
  text BLEU 0.2 at 4× ([arXiv:2307.06945](https://arxiv.org/abs/2307.06945)). Fine-KV synthetic
  recall: **93.9 → 40.6 (4×) → 13.8 (16×) → 11.9 (32×)** ([arXiv:2412.17483](https://arxiv.org/abs/2412.17483)).
  `r=16` sits in the regime where published compressors lose exact recall.
- **Width is usually not the binding constraint.** MLA beats MHA at ~GQA-2.25 cache
  ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434)); 500xCompressor finds KV-of-slots ≫
  embeddings and `16→4` slots flat while `4→1` drops ([arXiv:2408.03094](https://arxiv.org/abs/2408.03094));
  gist tokens are saturated at `k=1–5` for instruction gist ([arXiv:2304.08467](https://arxiv.org/abs/2304.08467)).
- **Allocated width ≠ used bits.** RankMe ([arXiv:2210.02885](https://arxiv.org/abs/2210.02885)) and
  *Fixing a Broken ELBO* ([arXiv:1711.00464](https://arxiv.org/abs/1711.00464)) both show identical
  objectives at wildly different effective rates. `D` dimensions are a superposition packing budget,
  not `D` independent facts.
- **No published `(K, D)` grid at matched compute** for concept sets. So "try wider slots" is not a
  research program; it is a safe retread.

Consequence for A vs C: **C's problem is that `C` does not scale with `N`** (a fixed 128 slots over a
whole book), and **A's problem is that each slot is a mixture**. B fixes the first (one slot per
16 tokens) and inherits the second (mean inside the block).

## 4. Gradient flow

This is where the two writes differ most, and it is measurable in our own runs.

**A — uniform mean.** `∂L/∂k_i = (1/r) · ∂L/∂k̄` for every token in the block. Every token receives
the *same* credit; there is no routing, no per-token target, no way for the loss to say "this token
in the block mattered". Frozen `identity_slots` also removes `u`/`delta` from the graph entirely.

**A′ — learned pool** (`softmax(h_j · u)` + zero-init `delta`). This *can* route, and at init it
*is* the mean. Under answer-only next-token CE the cheap optimum is a document mean: E25 learned
`u`/`delta` scored **0 bits** on INDEX that the frozen mean carried (53.82 bits @1024). Training
made the write worse, not better — the same shape as Fine-KV's "lost if surprise".

**B/C — learned-query cross-attention.** Softmax CA is a filter trained by the objective: a few
winner tokens get the update, the rest of the block gets ~0 gradient. That is acceptable for "which
class is this", lethal for "there is a password on page 3". Classic Perceiver's mean-pool head adds a
second uniform spray (`dL/dmean` identical for every slot) — no per-slot target, so redundancy is
free and RankMe collapse is expected.

**B/C — zero-init residual gates in series.** Our own measurement
([concept channel cold start](concept_channel_cold_start_20260913.md)): on the concept path two
zero-init residual writers sit in series (`pooler.wo`, `xattn.wo`), each one's gradient proportional
to the other. At step 0 the answer logits are **bit-identical** under a perturbation of the far
evidence (`max |Δlogits| = 0.000e+00`), even though the evidence *did* reach the slots
(`max |Δz| = 1.68` on an array norm of 45) and the mask *did* expose exactly those slots. Seeding
both gates recovered **17× more information at matched steps (3/3 seeds)**. So part of the classic
Perceiver failure is an initialisation artefact, not an architectural limit — but even repaired, the
channel recovered only **15.4%** of what raw access recovers (0.2137 of 1.3863 nats; 42% vs 100%
accuracy).

**Both — bypass starvation.** If the decoder can still see raw tokens, CE routes around the notebook
and its gradient dies. This is the project's standing failure (E22; E18 arm C; Broken ELBO). The
exclusive mask fixes *read-side* starvation. It does not fix a write that cannot be addressed.

**Slot-to-slot mixing is an optional path by construction.** Residual identity
(`Z_out = Z_in + F(Z_in)`) means an unused block's Jacobian can shrink to zero — Veit et al. showed
most gradient in a 110-layer ResNet came from paths of length 10–34
([arXiv:1605.06431](https://arxiv.org/abs/1605.06431)). Weight-tied unrolls sum the gradient, which
is a parameter-sharing regulariser, not "the network thinks K times". LCLM measured this directly at
scale: an attention adapter over the latent sequence **lost on pre-training loss and cost more** than
a non-mixing MLP ([arXiv:2606.09659](https://arxiv.org/abs/2606.09659)).

## 5. Information flow and theory

**A (slots).** `tokens → windowed mix → linear average of K/V → one exclusive attend → windowed
stack → head`. There is no second stage. Far content reaches the answer only through that single
read. Cost per token: `g·dh/r` keys.

**B (positional latents + transformer).** `tokens → encoder → per-block write → array → page-to-page
mix → decoder cross-attend`. Richer, and the only one of the three that can, in principle, compute
over the notebook.

**C (classic Perceiver).** `C` queries attend the whole book, then self-attend the array. Two
published properties hurt it:
- **Encode-once throws the book away.** After the first cross-attend, `X` is never consulted, so
  nothing can be fetched later that was not copied on the first pass. The original Perceiver paper
  measured this: interleaved re-reads 78.0 vs all cross-attends at the start 73.7 ImageNet top-1.
- **Learned queries are unallocated.** `Z0` is a parameter of shape `N×D`; nothing forces slot `i` to
  cover span `i`. Slot Attention needed *competitive* softmax over slots to stop collapse
  ([arXiv:2006.15055](https://arxiv.org/abs/2006.15055)); Perceiver does not have it.

**Theory, three pointers.**
- Constant-depth transformers ⊆ TC⁰; a stack of self-attends on a *fixed* `N` is still a
  constant-depth circuit over those `N` positions. **A set is not a tape.** CoT adds a tape
  ([Merrill & Sabharwal, arXiv:2310.07923](https://arxiv.org/abs/2310.07923)); BAPO Thm 8 makes the
  same point with constant per-step bandwidth.
- **Hahn (TACL 2020)** ([arXiv:1906.06755](https://arxiv.org/abs/1906.06755)): a single token's
  influence under softmax attention vanishes as `n` grows. "Attend to what matters" gets *harder* as
  the book gets longer **unless slots are allocated by position**.
- **Sanford** ([arXiv:2306.02896](https://arxiv.org/abs/2306.02896)): MATCH2 is easy for attention,
  MATCH3 is not. A set that has already softmax-pooled its inputs cannot recover a MATCH3 that was
  never written down as separate addresses.

## 6. What the literature actually shipped

- **DeepSeek-V4 HCA** pools 128 tokens into one KV entry with learned per-token weights and
  learnable positional biases; entries **do not talk to each other**; CSA supplies a fine-grained
  sparse read for exact recall ([arXiv:2606.19348](https://arxiv.org/html/2606.19348)). Positional
  allocation, learned pool, no mixer, raw path kept for INDEX.
- **Qwen3.8-Flash-Next** keeps one raw/sparse layer in four because "no finite-state memory
  reproduces exact retrieval" ([arXiv:2608.30320](https://arxiv.org/html/2608.30320)).
- **LCLM** (from-scratch encoder→latent→decoder sweep at 16×, scaled to 0.6B/4B): winner is causal
  encoder with window ≥ 256, **mean pooling**, **MLP adapter that does not mix latents**; mean > CLS
  pooling; a latent self-attention adapter lost on pre-training loss
  ([arXiv:2606.09659](https://arxiv.org/abs/2606.09659)).
- **Activation Beacon / Fine-KV**: interleaved > end-append; beacons match full FT on LongBench-32k
  and NIAH-128k; soft-prompt compressors (ICAE, AutoCompressor) collapse at 32k
  ([arXiv:2401.03462](https://arxiv.org/abs/2401.03462), [arXiv:2412.17483](https://arxiv.org/abs/2412.17483)).
- **Perceiver lineage retreated.** Flamingo kept a 64-latent resampler *as a compressor in front of*
  a decoder-only LLM ([arXiv:2204.14198](https://arxiv.org/abs/2204.14198)); BLIP-2's Q-Former needed
  an extra contrastive stage ([arXiv:2301.12597](https://arxiv.org/abs/2301.12597)); LLaVA dropped
  the bottleneck entirely ([arXiv:2304.08485](https://arxiv.org/abs/2304.08485)); IDEFICS2 measured
  frozen Perceiver-shaped fusion at 60.3 vs fully autoregressive + LoRA at **69.5**
  ([arXiv:2405.02246](https://arxiv.org/abs/2405.02246)). The adopted remnant is the read, not the
  think.
- **Perceiver AR** (the language attempt) reported **no gain past ~2k tokens** on PG-19, used extreme
  cross-attend dropout (0.875–0.96875), and its latents are rebuilt every pass rather than carried
  ([arXiv:2202.07765](https://arxiv.org/abs/2202.07765)).
- **Looped/supervised latent sets.** LOTUS matches explicit CoT at 3B — but `R=2` is 14.6% vs `R=6`
  70.0%, `c=1` is 49.7% vs `c=25` 70.0%, and **answer-only supervision without per-slot CE is 63.3%**
  ([arXiv:2606.31779](https://arxiv.org/abs/2606.31779)). One loop is not the trained computation;
  loops without per-slot targets are not the win. Pfau et al.: hidden delay works only with dense
  supervision of the delay ([arXiv:2404.15758](https://arxiv.org/abs/2404.15758)).

**Convergence:** positional compression into KV, a learned pool *inside* the block, contextualise
before pooling, no slot mixer by default, and a raw/indexed path kept for exact retrieval.

## 7. Our ledger evidence

| run | write | what it tested | result |
|---|---|---|---|
| E25 | `r=16` frozen mean, inplace | INDEX copy @1024 | **PASS** 53.82 bits vs 0.75×E18 47.34 |
| E25 | `r=16` frozen mean | MATCH lookup @512 | **0 bits** (E18 47.94) |
| E25 | `r=16` **learned** pool | INDEX @800 | **0 bits** — learning the pool destroyed a working write |
| E25 | `r=1` identity | MATCH @512 | **PASS** 43.08 bits |
| E26 | `r=16` + weak prefix AE (`L = L_answer + λ L_AE`) | MATCH @512 | **1.03 bits** vs gate ≈35.5; AE key_acc 0.66, RankMe 14.6 |
| E27 | `r=16` mean + raw identity keys on key spans | MATCH @512 | **3.03 bits** vs ≈35.9; RankMe 1.12; anchors-only 0.342 ≈ real 0.350 |
| E28 | `r=16` exclusive, CogitoProbe-bits @1024 | paying packed facts | **K1** — dense itself 6.5% / 0.21 bits on a 40-bit prize; exclusive 0 |
| E29 | `r=16` exclusive, CogitoProbe-bind @1024 | `(entity, attr, value)` | hop **4.4% / 0.078 bits**; `attr_color` 2.4% |
| E22 | 1 slot/16 tok + learned query + latent transformer | from-scratch LM @32k | array **live** (Δ_none 0.25 nats) and **diverse** (RankMe 265), but far-slot marginal **0.05 nats flat 1k→32k** and **arm A = arm C** |

Three readings of this table:

1. **The pooled write carries gist/copy and not lookup**, at exactly the ratio the literature
   predicts (16× is Fine-KV's collapse point).
2. **Both published "repairs" of the pooled write failed here** — a weak-decoder prefix AE (Fine-KV's
   recommendation) and hybrid identity key anchors. So the queue note's ranked levers 1 and 2 are now
   *falsified*, not pending.
3. **E22 shows the container is not the problem either.** The array was alive and diverse; the
   objective paid 0.05 nats for far content, so the latent transformer had nothing to reason over.
   That is the "objective must pay for the channel" law, not a latent-vs-slot verdict.

## 8. Verdicts

- **Uniform mean-pooled KV slot as the long-range object** — **Reject** for MATCH / binding /
  reasoning; **Keep** as an INDEX/gist cache and as a KV-cache compressor. It is a compressed cache,
  and should be named that way.
- **Fixed-`C` free learned-query latents over the whole sequence** — **Reject.** `C` must scale with
  `N`; unallocated queries collapse or duplicate; the latent transformer is an optional path.
- **Positional slots with a learned query *inside* each block, in K/V space, exclusive read** —
  **Adapt.** This is the pattern that shipped (HCA). It fixes allocation and lets the pool select
  rather than average.
- **Contextualise before pooling** (windowed encoder depth/window before the write) — **Adopt.** It
  is the one ingredient both LCLM and the DeepSeek split agree on, and E18b's failure was a 1-layer
  encoder producing unaddressable keys.
- **Slot-to-slot mixing / latent transformer as a language default** — **Watch / Reject.** LCLM
  measured it losing under CE; E22 measured it having nothing to mix. Gate it on a set-binding or
  multi-hop exam and on per-slot targets.
- **Tied loops with per-slot supervision (LOTUS-shaped)** — **Adapt, but gated.** Only after the
  channel is load-bearing on the exam the loops are supposed to help. `R=2` ≈ fail; answer-only
  supervision ≈ no-CoT.
- **Keeping a raw or indexed path for exact retrieval** — **Adopt** as a product constraint
  (Qwen's 1-in-4 layer; Fine-KV "lost if surprise"). A finite compressed notebook does not replace
  exact retrieval at this ratio.

## 9. Recommended bet (for `experiment-design`, not a frozen spec)

**One coherent claim.** If the exclusive read's write becomes a **learned query inside each block**
(HCA-style softmax pool with a learnable positional bias) over **pre-encoder-contextualised** states,
then MATCH-class lookup at `r=16` recovers a majority of uncompressed performance where frozen mean,
learned mean, prefix AE, and hybrid key anchors all failed — **because** the failure was the
sufficient statistic (mixture vs address), not the container or the ratio.

**Falsifiable gates.** S0 dense solvable on the same rows. S1 MATCH bits ≥ 0.75 × uncompressed
reference at `r=16` with the pool frozen at init = mean. S2 the pool must actually move: ablate it
back to the mean on the trained checkpoint and the score must collapse. S3 copy/INDEX must not
regress.

**Kill.** Pool ablation ≈ real (the pool learned nothing) → the block is not addressable at any pool,
and `r=16` is recorded as a gist-only ratio. Do not then sweep width, and do not add a latent
transformer on top.

**Diagnostics that decide it cheaply.** Effective rank of the slot matrix (RankMe) and slot-to-slot
cosine on a checkpoint; `message_override none/swapped/raw`; per-position bins so the number is not
diluted by local tokens.

## 10. Open tensions and doc drift

- [`e21_improvement_queue.md`](e21_improvement_queue.md) (undated mutable note) still ranks "prefer
  exclusive slots + weak prefix AE" and "hybrid `(a,b)` anchors" as levers 1–2. **E26 and E27 ran and
  both missed their gates.** The note's *reasoning* (literature levers) stands; its *recommendation*
  is superseded by the result and should be marked obsolete.
- The queue note's ranking also predates the E28/E29 Wave B outcomes, which removed the "no 32k-only
  length story" hypothesis rather than confirming it.
- Handoff: marking those sections is a `docs-hygiene` job, not this note's.

## 11. Sources

| source | URL |
|---|---|
| BAPO (bandwidth `a`/`b`, Thm 8/10) | https://arxiv.org/abs/2505.08140 |
| ICAE (count starvation) | https://arxiv.org/abs/2307.06945 |
| Fine-KV gists (4×/16× synthetic recall) | https://arxiv.org/abs/2412.17483 |
| Activation Beacon | https://arxiv.org/abs/2401.03462 |
| Gist tokens (gist saturated at `k=1`) | https://arxiv.org/abs/2304.08467 |
| 500xCompressor (KV-of-slots) | https://arxiv.org/abs/2408.03094 |
| MLA (width overprovision) | https://arxiv.org/abs/2405.04434 |
| RankMe | https://arxiv.org/abs/2210.02885 |
| Fixing a Broken ELBO | https://arxiv.org/abs/1711.00464 |
| Slot Attention | https://arxiv.org/abs/2006.15055 |
| LCLM (mean > CLS; no latent mixer) | https://arxiv.org/abs/2606.09659 |
| LOTUS (loops need per-slot CE) | https://arxiv.org/abs/2606.31779 |
| Pfau filler tokens | https://arxiv.org/abs/2404.15758 |
| Merrill–Sabharwal CoT / TC⁰ | https://arxiv.org/abs/2310.07923 |
| Sanford MATCH2 vs MATCH3 | https://arxiv.org/abs/2306.02896 |
| Hahn (token influence vanishes) | https://arxiv.org/abs/1906.06755 |
| Veit residual ensembles | https://arxiv.org/abs/1605.06431 |
| DeepSeek-V4 HCA/CSA | https://arxiv.org/html/2606.19348 |
| Qwen3.8-Flash-Next (finite-state limits) | https://arxiv.org/html/2608.30320 |
| Perceiver / IO / AR | https://arxiv.org/abs/2103.03206 · https://arxiv.org/abs/2107.14795 · https://arxiv.org/abs/2202.07765 |
| Flamingo · BLIP-2 · LLaVA · IDEFICS2 | https://arxiv.org/abs/2204.14198 · https://arxiv.org/abs/2301.12597 · https://arxiv.org/abs/2304.08485 · https://arxiv.org/abs/2405.02246 |
| TSDAE (decoder confinement) | https://arxiv.org/abs/2104.06979 |
