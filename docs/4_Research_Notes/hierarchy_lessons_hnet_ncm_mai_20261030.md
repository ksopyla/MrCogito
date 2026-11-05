# New architectures and what they teach us: H-Net at scale, Next-Concept Modeling, MAI-Thinking-1, Latent Sparse Attention (30 Oct 2026, LSA added 4 Nov)

Research synthesis requested by the author (30 Oct): what to learn from three recent sources, whether they are
relevant to the vision, and what they suggest for (a) our capability testing and (b) our architecture.
**Revised the same day after reading the primary papers and code** (first version was built partly on search
summaries; corrections are marked "corrected"). **4 Nov:** the author added *Latent Sparse Attention* (§1.4), "aligned
with our goal"; it changes step 3 of the bet and adds three capability changes (marked "added 4 Nov").

Builds on (not repeated here): [notebook_writer_compression_20261016.md](notebook_writer_compression_20261016.md)
(BLT, H-Net 2025, DLCM, Large Concept Models, rate-matched notebook, candidates A1–A10 / P1–P4),
[e33_multihop_review_label_free_20261018.md](e33_multihop_review_label_free_20261018.md) (written chains, step
removal), the [capability checks](../engineering_specs/capability_checks.md) and
[text checks](../engineering_specs/text_capability_checks.md) protocols, and E32
([two-level concept memory](../experiments_specs/ahead/E32_two_level_concept_memory.md)).

## How the sources were read (trust level)

| source | read from | trust |
|---|---|---|
| **Next-Concept Modeling (NCM)**, Kumbong, Khare, … Arora, Mirhoseini, Ré, [arXiv 2610.14162](https://arxiv.org/abs/2610.14162), 9 Oct 2026 | **full paper incl. appendices A–J** (Hugging Face paper mirror) + **code** [HazyResearch/ncm](https://github.com/HazyResearch/ncm) @ `c702742` (14 Oct 2026): quantizer, generator, all eval configs | primary |
| **H-Net**, Hwang, Wang, Gu, *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*, [arXiv 2507.07955](https://arxiv.org/abs/2507.07955) — the published method behind Cartesia's post | **full paper** (method, results, ablations, appendix E) + **code** [goombalab/hnet](https://github.com/goombalab/hnet) @ `3673fe1` (30 Sep 2025): chunking module, model, ratio loss, configs | primary |
| Cartesia, *From Tokens to Bytes I: Scaling Hierarchical Networks*, [blog](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1), 28 Oct 2026 (+ [part II](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2)) | **search excerpts only** — cartesia.ai is blocked from the session; the improved H-Net of the post is **not released** (the public repo is the Sep 2025 version) | secondary |
| **LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts** (NVIDIA), [arXiv 2601.18089](https://arxiv.org/abs/2601.18089) — the MoE layer MAI-Thinking-1 uses | **full paper** (Hugging Face paper mirror) | primary |
| Microsoft AI, *MAI-Thinking-1: Building a Hill-Climbing Machine*, [report](https://microsoft.ai/pdf/mai-thinking-1.pdf) (p. 6 = architecture) | **secondary summaries only** — microsoft.ai is blocked; no mirror found on GitHub or Hugging Face | secondary |
| **Latent Sparse Attention (LSA)**, Kubicek, Caciolai, Peng, … Hoefler, Whittington (Prefrontal AI, ETH Zürich), [arXiv 2610.42147](https://arxiv.org/abs/2610.42147), 30 Oct 2026 — added 4 Nov | **full paper** (main body; appendices A–E, F in part) via the Hugging Face paper mirror + **code** [Prefrontal-AI/latent-sparse-attention](https://github.com/Prefrontal-AI/latent-sparse-attention) @ `97b82d6` (3 Nov 2026): native-training layer, Ward clustering, two-path summaries, configs | primary |

Every Cartesia-post and MAI number below must be re-checked against the primary page before it enters a spec.

## Brief

- **All four move the heavy work onto a shorter, coarser representation.**
  - H-Net runs its deep network on learned chunks.
  - NCM generates coarse codes first.
  - LatentMoE runs experts in a 4× narrower latent.
  - LSA searches content clusters before it reads keys.

  This is outside evidence for the vision's direction. None reaches our ratios: H-Net's deepest level sees about
  one position per 9 bytes ≈ 2 tokens, while we write one note per 6 tokens and aim for 16–128. LSA keeps every
  key. None of them targets our use, which is reading and reasoning over a long book through compressed notes.
- **"Next-concept modeling" is now a published name** (Stanford, arXiv 2610.14162) for something different from
  ours. Their "concepts" are the coarse levels of a code used to *generate* text fast; ours are notes used to *read
  and reason*. Any MrCogito paper must cite it and state the difference. The two are complementary: theirs is a
  candidate for our decoder (the crystallisation step).
- **The strongest lesson for our notebook comes from H-Net's ablations:** at equal compression, **fixed-width
  pooling clearly loses to learned boundaries** (0.780 vs 0.755 bits per byte, lower is better), while **how a
  chunk is squeezed does not matter** (keeping one vector, mean, max and cross-attention are on par) — the
  encoder in front does the compressing. And **recurrent (Mamba) encoders compress better than attention**
  even on ordinary tokens. Our notebook spends a fixed 32 notes on every 256-token page and writes them with
  attention layers: both choices are the ones H-Net found weakest.
- **NCM's coarse levels hold the gist, not the facts:** fixing a story's first three levels keeps its theme while
  names and objects change. Its "down" step is plain averaging over fixed regions, and the whole hierarchy is
  ~5× *larger* than the token ids — only the coarse levels compress. Nested dropout (train with only the first
  levels sometimes) is the reusable trick.
- **Latent Sparse Attention (added 4 Nov) is the closest of the four to the vision's 10M-token goal.** It groups
  each 512-token block's keys into content clusters of ~8 (no new parameters: the attention's own keys, Ward
  merging), and a question scores one centre per cluster, then reads only the best clusters' keys plus a window.
  Trained from scratch on next-token loss alone (0.5B, 128k), it matches full attention (validation loss 1.085 vs
  1.085). Distilled into 27B models it holds accuracy to 10M tokens, with one attention layer 72× faster to
  prefill. It is a working, large-scale version of E32's "read the coarse level, then fetch the fine notes". It is
  also a third independent result that **grouping by content beats grouping by position**. It keeps the full key
  cache, so its memory still grows with the text. Our notes compress that memory about 48× — the two combine.
- **Architecture — one bet, revised:** the coarse-to-fine notebook, now in the order the evidence supports:
  1. ordered notes (cheap; gives the rate–distortion curve);
  2. **content-placed notes** with H-Net's router, smoothing and ratio loss (the best-evidenced change for the
     writer);
  3. **(changed 4 Nov) a content-clustered index over the notes, LSA-style**, for reading at 128k–1M. It needs no
     trained coarse writer; residual levels move to Watch.

  Two cheap H-Net levers ride along: start the raw-text shortcut at zero, and try a recurrent writer.
- **Far context only pays on long documents.** In LSA's native run every attention variant sits within 0.004
  of the others at 4k. Only a 12B-token phase on 128k-token books, papers and code repositories separates them:
  a window-only model ends 0.069 nats worse than full attention. This matches our own law that the objective must
  pay for the channel; E22 measured ~0.05 nats of far-context value on ordinary text. Long natural documents are a
  data lever we have not used: the text checks train on short stories plus generated facts.
- **Capability testing — eleven changes (§3):**
  - an **efficiency multiplier** against dense (LatentMoE gives the primary definition: invert the baseline's
    fitted scaling curve);
  - a ranking must hold at **two sizes** before an architecture is parked or crowned;
  - a **frontier-shaped control**: a few local layers per global layer, run both with full attention and with
    LSA as the **ceiling past 32k**, where our dense ceiling stops;
  - a **global-constraint exam** (Sudoku-like: a refining model beat left-to-right 86 % vs 1 % in NCM);
  - **out-of-distribution and surface-noise robustness** as a headline column (H-Net's 2-stage model was about
    2× as robust to noisy text as BPE);
  - the notebook's **rate–distortion** and **note budget** curves;
  - score against the number of sequential steps;
  - a gist-vs-fact swap probe;
  - **answer-not-in-the-book** questions;
  - **revised facts** whose old value is still quoted elsewhere (LSA's PaperTrail).
- **First moves need no training (§6):** the note budget curve, the rate–distortion curve, LSA-style clustering
  of an existing checkpoint's notes, and the efficiency multiplier all run on checkpoints we already have.

## 1. Per source

### 1.1 Next-Concept Modeling (Stanford / Hazy Research, arXiv 2610.14162) — paper + code

- **Thesis.** Generate text coarse-to-fine: a frozen multiscale residual VQ-VAE turns a sequence into K levels of
  discrete codes; a transformer predicts all codes of one level in parallel, given the cumulative reconstruction
  of the earlier levels. K ≪ L sequential steps.
- **Technique, as implemented** (`models/quant.py`, `config/eval/*.yaml`):
  - Levels (TinyStories): 12 levels of 1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128 codes for 128 tokens;
    4,096 codes per level (12 bits). **641 codes ≈ 7,700 bits for a 128-token story, vs ~1,700 bits of BPE-8192
    ids — the hierarchy expands the text ~4.6×.** Only the first levels compress: levels 0–2 are 25 codes
    (~5 tokens per code, ~300 bits). Sudoku: 12 levels of 1 → 89 codes with only 16 codes per level (4 bits).
  - "Down" = **area interpolation (average pooling over equal regions)**; "up" = linear interpolation, plus a
    VAR-style residual conv (`Phi`, blend ratio 0.5). Only level 0 has learned conv down/up-samplers. **Coarse
    codes are positional region averages, not content-placed.**
  - Residuals computed on the stop-gradient encoder output; codebooks updated by moving average (decay 0.95–0.99)
    with dead-code restarts; the encoder gets gradient through a straight-through copy of the final
    reconstruction. Code comment: *"Straight-through estimator at level 0 only; at higher levels it diverges
    within ~100 steps."*
  - **Nested dropout (`fine_drop`)**: with probability 0.5 per batch, the decoder gets the reconstruction cut at a
    uniformly drawn level (min keep 1).
  - Generator: block-diagonal attention per level (each level's block sees only itself, its input already sums
    the earlier levels; 1.4× faster than block-causal and best quality); codebook-tied output heads; level 0
    drawn from a unigram (unconditional) or predicted from the prompt (Sudoku). **Code corruption** (replace
    input codes with random codes of the same level; mean rate ~0.5, decreasing with level): moderate rates
    "greatly improve" quality, high rates hurt. Level-loss weighting barely matters (Gen. PPL 18.9–19.4).
  - **The repository is evaluation-only** (no training loop); weights on Hugging Face. The quantizer is reusable
    (MIT); training would be ours to write.
- **Evidence (~100M generators, one tokenizer per dataset).** LM1B: Gen. PPL 93.1 at 12 steps vs DUO 97.3 at 128,
  21× faster wall-clock. XSum: ROUGE-L 28.0 at 8 steps vs AURORA-LM 26.6 at 128 (31× faster); 29.6 with MBR.
  WMT14 De–En: 22.6 BLEU at 11 steps (AR 25.2 at 64). **Sudoku (25M, greedy): hard 86.1 % at 12 steps vs AR
  1.0 %, FMLM+ 81.3 % at 16.** Training positions: LM1B 13.4B vs 66B for the diffusion baselines (4.9×). More
  levels help with diminishing returns (4 → 32). Scaling 97M → 725M improves.
- **Probing (App. H, the result that matters most for us).** Fixing the first three levels of a TinyStories story
  and resampling the other nine keeps the **broad theme** ("a girl seeking warmth and a garment that moves")
  while **names, objects and wording change**; swapping two level-1 codes changes characters and events. Coarse
  codes carry gist, not facts.
- **Authors' limitations.** Fixed depth and level lengths ("an input-dependent schedule would adapt compute");
  separately trained tokenizer ("end-to-end training could benefit"); ≤ 725M; artifacts at coarse transitions.
  Both of their future-work items are what H-Net provides.
- **Verdict: Adapt** (nested dropout for ordered notes; constraint exam). **Watch** as a future parallel decoder.
- **For publication.** The name overlaps with our framing; cite it and draw the line. NCM's concepts are *generated*
  coarse codes over ≤ 1,024 tokens. Ours are *written and read* notes over 16k–1M-token books, used for lookup and
  multi-step reasoning.

### 1.2 H-Net (arXiv 2507.07955) — paper + code; Cartesia's scaled version (blog, excerpts)

- **Thesis.** A U-Net-like network that learns its own chunk boundaries end to end (dynamic chunking) and runs its
  deep network on the chunks beats a compute- and data-matched BPE Transformer; nesting two chunking stages
  helps more.
- **Technique, as implemented** (`hnet/modules/dc.py`, `hnet/models/hnet.py`, `hnet/utils/train.py`):
  - Encoder (4 Mamba-2 layers at the byte level) → **router**: `p_t = ½(1 − cos(W_q x_t, W_k x_{t−1}))`,
    boundary where `p_t ≥ 0.5`, `W_q = W_k = I` at init (pure "how different is this position from the
    previous one") → keep only boundary vectors → main network (2-stage Large: 26 Transformer layers at width
    1536) → **smoothing** `z̄_t = P_t ẑ_t + (1 − P_t) z̄_{t−1}` (an EMA, run with the Mamba scan kernel) →
    repeat each chunk vector until the next boundary, times a straight-through confidence → **plus a
    full-resolution shortcut from the encoder, `Linear(x̂)`, whose weight starts at exactly zero** → decoder
    (4 Mamba-2 layers).
  - **Ratio loss** `N/(N−1)·((N−1)FG + (1−F)(1−G))` (F = share of boundaries, G = mean boundary probability),
    weight 0.03; targets 6 bytes per chunk (1 stage) or (3, 3) → 9 (2 stages). Per-stage learning-rate
    multipliers (e.g. 3.0 / 1.7 / 0.9, outer stages higher); an RMSNorm at the end of every sub-network so the
    deep path does not drown the shortcut.
- **Evidence (100B FineWeb-Edu tokens, bytes per batch and FLOPs per byte matched).**
  - XL (1.3B): 2-stage 0.715 bits per byte vs BPE Transformer 0.730; zero-shot average 58.2 vs 55.5; the Large
    2-stage (55.5) matches the XL Transformer (55.5). It crosses the Transformer after 30B bytes and the gap
    widens.
  - Noisy-text robustness score 42.8 vs 22.2 (BPE), from clean training data only.
  - Chinese and code beat BPE; DNA reaches isotropic perplexity with 3.6× less data.
- **Ablations (the part we use).**
  - **Fixed-width pooling** ("H-Net pool") at the same ratio: 0.780 vs 0.755 (Large) — "fixed-width chunking is
    not effective".
  - **Smoothing** is essential: without it the compression ratio swings and quality drops. Cosine routing and
    the straight-through step help.
  - **Mamba encoders/decoders beat Transformer ones even on BPE-token inputs**: the authors attribute it to the
    recurrent state's "inductive bias for compression".
  - **How a chunk is squeezed does not matter** (App. E.1): keeping the boundary vector, mean, max and
    cross-attention are on par — "the encoder is trained to implicitly compress necessary context into vectors
    at boundaries".
  - It beats a FLOP-matched MoE of similar parameter count; a hybrid main network scales better.
  - Training is up to 2× slower than an isotropic model.
- **What it learns.** Stage 1 cuts at spaces and word starts; stage 2 at words and multi-word groups ("the
  backbone", "such as").
- **Cartesia's Oct 2026 post (excerpts only, corrected).** A 3.67B two-stage H-Net on a reproduction of SmolLM3's
  stage-1 mix reaches SmolLM3's (3.08B) six-task average after ~1T vs ~11T tokens, with "~7.6 bytes per main
  position" reported. Part II: loss-to-loss advantage ~1.3× on the training mix, ~2× downstream, up to ~5× out of
  domain; offline chunk pipelines do not match. Not a matched ablation (different sizes and data); the improved
  architecture is not public.
- **Verdict: Adapt** (content-placed notes with router + smoothing + ratio loss; zero-start raw shortcut;
  recurrent writer). **Watch** for audio (learned chunking over codec frames).

### 1.3 LatentMoE (NVIDIA, arXiv 2601.18089) — paper; and MAI-Thinking-1 (summaries)

- **LatentMoE (read in full).** Project each token `d → ℓ` (α = d/ℓ = 4), route on the *full* vector, run the
  routed experts in the latent, project back; spend the saved bandwidth on α× more experts and α× higher top-k.
  Quality holds for α ≤ 4 (an "intrinsic feature rank" floor); at 95B total / 8B active, MMLU-Pro 29.3 → 34.9 at
  equal active parameters; adopted by Nemotron-3. Its **Effective Parameter Multiplier** is the primary
  definition of the efficiency multiplier we want: fit `f(N)` to a baseline family, `N_eff = f⁻¹(score)`,
  `λ = N_eff / N` (1.35× for their 1T variant).
- **MAI-Thinking-1 (secondary).** ~35B active / ~1T total (one outlier summary: 38B / 480B); 5 sliding-window
  layers (512, RoPE) : 1 global layer without position code; LatentMoE 8 of 512 experts, interleaved with small
  dense FFNs, no shared experts; zero-initialised attention output gains; 256k context; 30T + 3.55T tokens.
  Process: scaling ladders at fixed tokens per parameter, an Efficiency Gain metric (same construction as
  LatentMoE's multiplier), and a reported small-scale ranking that reversed at 23B active.
- **Verdict.** **Adopt** the efficiency multiplier and the frontier-shaped control. **Reject** LatentMoE at
  30–100M. **Reject** zero-initialised outputs on *new memory reads* (our cold-start law) — note H-Net zero-inits
  the opposite path, the raw shortcut, which is consistent with that law (§4).

### 1.4 Latent Sparse Attention (Prefrontal AI / ETH Zürich, arXiv 2610.42147) — paper + code (added 4 Nov)

- **Thesis.** At ultra-long context the cost of sparse attention is *finding* the keys: today's indexers score every
  key or block (at 10M tokens the indexer is 98 % of a sparse layer's prefill). Cluster the keys by content in a
  learned latent space and let each query score the clusters instead. Flat search is a constant-factor saving;
  tree search costs O(log T) per query.
- **Technique, as implemented** (`src/lsa/training/native/lsa.py`, `src/lsa/core/ward.py`, `config.py`):
  - Index keys are L2-normalised and summed in runs of 4 ("pool keys"). Each closed **cluster block** of 512 keys
    is merged bottom-up with **Ward's method** into 64 buckets: ~8 keys on average, at most 16, never changed
    afterwards. Every round merges all mutual-cheapest pairs in parallel. A bucket's centre is its normalised key
    sum, stored with its key positions (GLM at 10M: 246 MiB per layer next to a 10 GiB key/value cache).
  - A query scores the centre of every bucket in the blocks before its own (`s = c · q`). It takes the top-p
    buckets and expands them best-first into k long-range keys, plus 4 sink keys and a 512-key window (the
    window covers the query's own, not-yet-clustered block). One index list is shared by all heads; the attention
    kernel is unchanged.
  - **Tree**: buckets are merged again into node buckets (branching 4) and searched by beam search. It is
    O(T log T) to prefill instead of O(T²).
  - **D-LSA** (pre-trained models): 64-dimensional index projections of the position-free key/query dims (from a
    PCA initialisation, or PCA alone = training-free). Distilled by KL divergence to the frozen layer's
    head-averaged attention (top-2,048 target) on 352M tokens of 16k–1M documents; the base model is frozen.
  - **N-LSA** (from scratch): no new parameters. It clusters the attention's **own keys**, uses one key/value
    head and **no position code**, and scores a bucket by its best head. **The selection carries no gradient**;
    the next-token loss trains the projections only through the keys actually read. No auxiliary loss, no
    teacher.
  - **Two-path read (in the code, not in the paper; default `kind: multires`)**: every closed bucket the query
    does not read enters the same softmax as **one summary term**. Its value is the mean value; its logit is
    `log n + s·q·μ + s²|q|²σ²/2`, a second-order estimate of the bucket's log-sum-exp. Gradients reach unread keys
    and values through the summary statistics. Nothing is dropped; unread content is seen coarsely. No results
    for it are reported.
- **Evidence.**
  - **Distilled (GLM-5.3-Flash, Qwen-3.8-27B; 4k → 10M tokens):** knowledge and reasoning stay inside the
    baselines' 95 % confidence intervals. RULER stays within 3–5 points to 1M. At 10M, D-LSA (Flat) on Qwen beats
    its own full attention: RULER 63.9 vs 59.9, PaperTrail 29.1 vs 19.8.
  - **Speed, one layer at 10M:** prefill 22.7× (GLM, vs its sparse attention) and 72.4× (Qwen, vs full); decode
    4.6× and 42.5×. End to end, time to first token is 2.2× and 9.2× faster.
  - **Index size:** 246 MiB per layer vs 331 (GLM) and 642 (QSA).
  - **Block-size ablation:** positional blocks of 8 / 16 / 64 keys at the same key budget fall earlier the larger
    the block — "a larger block … can hide a single relevant key". Even 8-key blocks trail semantic buckets at
    long context.
  - **Native (0.5B hybrid: 24 layers, every 4th attention, the rest Gated DeltaNet; 100B FineWeb-Edu at 4k, then a
    12B-token decay at 128k):**
    - validation loss: N-LSA 1.085 = full 1.085; PISA (tree over blocks) 1.100, NSA (block sparse) 1.117,
      SWA 1.154;
    - RULER at 128k: N-LSA 32.2 vs full 34.7; NSA 19.2, PISA 14.6, SWA 3.0;
    - 1.6× faster per layer at 128k, 1.24× faster end to end; peak memory 6.8 vs 5.8 GiB per layer.
    - At 4k all methods sit within 0.004 of each other: the far-context gain only appears on long documents.
  - **Key budget:** single-needle retrieval holds down to k = 16 keys. MultiKey 2/3 and common-word extraction
    (aggregation) fail first and fall to ≤ 5 % at 10M for every method, full attention included. Tree search loses
    accuracy past 1M on GLM (RULER 55.8 vs 66.8 flat at 10M).
- **PaperTrail (their benchmark).** Fabricated "trails" are inserted into 1,246 real 2026 papers that postdate both
  models. A result is introduced, cited, then two to three weeks later revised and renamed, while some later
  papers still quote the old value.
  - Tiers: 1 = one paper, 2 = two papers, 3 = the whole trail.
  - Six questions per sample have no answer in the papers (absent entity, false premise, near miss); they are
    scored apart.
  - Four controls: the question alone, hosts without insertions, insertions alone, background alone. Answers
    exist only in the insertions (controls score 0 % / 0 % / 100 % / 0 %).
  - Scoring: numbers within 2 % relative tolerance; the cited papers by F1.
  - Tier 3 falls to ≤ 19 % at 10M for every method.
- **Limits (authors).** The key cache still grows with the text — LSA lowers the cost of *finding* keys, not of
  *storing* them, and is "orthogonal to methods that compress the KV cache". Native training only at 0.5B / 128k
  with flat search. Tree search vs flat beyond 1M is open. Kernels target H200 / B300 (FlashAttention-3).
- **MrCogito mapping.** The reader at scale (vision priority 2). The coarse level of E32. The dense ceiling past
  32k. Exams (abstain, revised facts, real-text filler).
- **Verdict: Adapt** — the clustered index over *our notes* (step 3 of the bet) and N-LSA as the long-length
  control. **Watch** the two-path summary read until it has published results.

## 2. Relevance to the vision

| vision priority | H-Net | NCM | LatentMoE / MAI | LSA (added 4 Nov) |
|---|---|---|---|---|
| 1 Reasoning concept core | **supports**: the deep network on learned chunks beats token-level at 1.3B (paper) and reportedly 3.7B / 1T (post) | decode side; per-level targets echo our per-round targets | — | — (keeps raw keys; no concepts) |
| 2 ~10M-token context | weak: ~9 bytes per deepest position | none (≤ 1,024 tokens) | the frontier still pays full attention in 1 of 6 layers at 256k; at 1M–10M our O(C·N) read is the open niche | **strongest support**: a content-clustered coarse-then-fine read holds accuracy to 10M at O(log T) per query; its open cost (memory grows with T) is exactly what our notes compress |
| 3 Reasoning bandwidth | — | **supports**: coarse-to-fine refinement solves global constraints where left-to-right fails | — | — |
| 4 Agent-to-agent | — | discrete codes with exact bit counts suit a channel | compress before communicating (same motive) | a bucket index is a natural "table of contents" another agent could query (E33f fetched notes; same idea) |
| 5 Multimodality | **Watch**: dynamic chunking is modality-agnostic (DNA 3.6×) | the recipe comes from images and audio codecs | — | — |

**Convergence.** All put compute on a shorter coarse sequence and keep a cheap path back to full resolution
(H-Net's shortcut, NCM's reconstruction, LatentMoE's up-projection, LSA's window plus expanded buckets).
Multi-resolution is the shared pattern.

**Convergence (added 4 Nov): group by content, not by position — three independent results.**
- H-Net: fixed-width pooling loses to learned boundaries.
- LSA: positional blocks lose to semantic buckets at the same key budget, more so the larger the block.
- NCM: positional region averages carry only a story's gist.

Our notebook groups by position (a fixed page) and lets learned queries pick within it. Placement (step 2) and the
clustered index (step 3) are the two places to stop grouping by position.

**Tension: placement vs squeezing.** Our E30 → E31 story credited E31's *latents* (cross-attention queries) over
E30's *weighted averages*. H-Net's App. E.1 says the squeeze is irrelevant when the encoder before it is good,
and its ablations say *placement* and *a recurrent encoder* matter. Consistent with our own law 7 (the writer
reads bare token vectors) and with E31's gain coming with two-way page context: **the writer's encoder and the
note budget per page are the levers, not the pooling operator.**

**Tension: end-to-end vs staged.** H-Net trains everything jointly (with smoothing and a ratio loss to keep it
stable); NCM freezes its tokenizer and says end-to-end "could benefit". Our E36 found reconstruction-first
training lifted bits per entry from 32–48 to 114–145. Resolution: stage the writer's reconstruction, then train
end to end with notes-only questions, keeping H-Net's stabilisers for any learned placement.

**Gist vs facts.** NCM's probe shows reconstruction-trained coarse codes keep a story's theme and lose its
names. Our notes must hold facts. Coarse notes can route; fine notes must store — the E32 split, now with
outside evidence.

**Data: far context needs long documents (added 4 Nov).** LSA's native comparison is flat at 4k (all within 0.004)
and opens only in a 128k-token decay on long books, papers, code repositories and agent traces (window-only
1.154 vs full attention 1.085). H-Net and NCM never test far context. Our E22 law ("the objective must pay for the
channel") and the text checks' question-density finding (1 question per document switched no notebook on; 8 switched
every one on) say the same: the read learns only what the data pays for. Generated facts pay for lookup; long
natural documents would pay for general far-context use. LSA's LongDocs recipe (public sources, equal share per
domain, extra weight at the longest lengths) is a template.

**Our own evidence agrees with the robustness results:** on unseen sentence shapes dense lookup falls from
37–48 % to 0–1 % while the notebook model keeps 64 % (one run, text checks finding 6); H-Net's 2-stage model
is ~2× more robust to noisy text than BPE.

## 3. Capability testing — proposed changes

| # | change | from (primary unless marked) | why for us | cost | verdict |
|---|---|---|---|---|---|
| T1 | **Efficiency multiplier vs dense**: fit dense's score (or loss) against training tokens on its checkpoints (10/25/50/75/100 %) or sizes; a model's multiplier = dense tokens needed for its score ÷ its own tokens. Per task: dense tokens to pass ÷ the model's | LatentMoE effective parameter multiplier; MAI Efficiency Gain (secondary) | "pass / fail" hides how much cheaper; the vision's claim is an efficiency claim | analysis only | **Adopt** |
| T2 | **Two-size rule**: parking an architecture or changing the champion needs the same order at two sizes | MAI ranking flip (secondary); NCM and H-Net both show gaps that *widen* with data/size | E30 parked and E31 crowned at ≤ 50M; E33's "four rounds = one" is a 30M fact | one extra size per decision | **Adopt** (author decision) |
| T3 | **Frontier-shaped control** (revised 4 Nov): same size; local layers with **one global layer in 4–6**, which has a single key/value head and no position code; run twice — global layer full attention (≤ 32k), and **N-LSA** (k ≈ 1–2k keys + window 512) as the **ceiling past 32k** | Gemma-style hybrids; MAI (secondary); **LSA's own testbed is this layout** (every 4th layer attention, MQA, NoPE) | isolates what the notebook replaces; it is the real competitor; our dense ceiling stops at 32k in the length ladder, N-LSA reads 128k+ at near-full-attention quality | config on the perceiver stack for the full arm (global read positions + `global_nope`); M for N-LSA (Ward + bucket read on flex attention; their kernels are H200-only) | **Adopt** (full arm); **Adapt** (N-LSA arm) |
| T4 | **Global-constraint exam** (DNA: 4×4 Latin square / small Sudoku given as facts; text: a seating or schedule puzzle from fact sentences), status `calibrating` | NCM Sudoku (25M: 86 % hard vs AR 1 %, greedy, 12 steps) | every current exam has dense as its ceiling, so the board can show "matches" but never "beats"; rule 3 (2026-10-07) lets a non-dense model be a task's named reference | M | **Adapt** |
| T5 | **Robustness column**: in-distribution vs unseen-sentence-shape score, plus a **surface-noise** variant (case flips, dropped spaces, repeated characters in the document) | H-Net robustness score; Cartesia part II (excerpts) | our clearest notebook-vs-dense contrast is out of distribution | paraphrase2 exams exist; noise is a data transform | **Adopt** |
| T6 | **Rate–distortion curve of the notebook**: read each exam with 1/2/4/8/16/32 notes per page; report notes needed to pass and the share of notes ever read (NCM's codebook use) | NCM nested dropout, App. G | measures compression honestly; none exists for chunk latents | eval flag + analysis | **Adopt** |
| T7 | **Score vs sequential steps** for any refining model (loop rounds, levels), with wall-clock | NCM NFE reporting | turns "the loop does nothing" into a curve | eval loop | **Adopt** where loops exist |
| T8 | **Gist-vs-fact swap probe**: keep a page's first k notes, replace the rest with another document's; does the answer keep the topic but lose the fact? | NCM App. H (fix coarse, resample fine) | says what ordered or coarse notes actually hold | eval only | **Adopt** with the bet |
| T9 | **Answer-not-in-the-book questions** (added 4 Nov): an absent entity, a false premise, a near miss (a name one letter off a real one); the right reply is an abstain word; scored apart from the main score | PaperTrail not-in-corpus questions | none of our exams asks it; a notebook that "always finds something" would pass every current exam and still hallucinate; tests whether a read can return "nothing matches" | S (generator + a reserved answer word) | **Adopt** (text checks) |
| T10 | **Revised facts with stale echoes** (added 4 Nov): a fact is stated, later revised (and the entity renamed), and later passages still quote the old value; ask for the current value, and at tier 3 for the chain old name → new name → value | PaperTrail trails | T4 Latest (as the spec describes it) interleaves moves but has no later sentence re-quoting an old place as if current; that is where a content index retrieves the wrong, more often repeated value | S–M | **Adopt** as a T4 / T5 variant |
| T11 | **Key / note budget curve** (added 4 Nov): read each exam with a top-k of 16, 64, 256 … notes per question (selection by score), with the T6 curve as its per-page twin | LSA App. A (single needle holds at k = 16; multi-key and aggregation fail first) | tells how selective a read can be per task class before a sparse reader is built; the measurement is free on today's dense note read | eval flag | **Adopt** |

## 4. Architecture — the bet (revised)

### The coarse-to-fine notebook

- **Novel claim.** Notes ordered by importance and placed by content, then indexed by content clusters, give a
  per-document note budget that follows the facts. They also give a rate–distortion curve and a read whose cost
  per question grows like log N instead of N. Memory stays ~48× below a key/value cache, which LSA alone cannot
  offer. All of this comes at no loss at full notes. Nobody combines compressed notes with a semantic index: LSA
  names KV compression as orthogonal and combinable, and does not do it.
- **How the ideas connect.**
  - NCM: nested dropout; coarse = gist.
  - H-Net: learned boundaries beat fixed pooling; router + smoothing + ratio loss make placement trainable; a
    recurrent encoder compresses better; the squeeze operator is irrelevant.
  - LSA: content clusters beat positional blocks; the coarse level needs no trained writer (normalised key sums);
    a no-gradient selection trains fine on next-token loss; a tree gives O(log N).
  - Ours: E32 (coarse routes, fine stores), the 13 Sep cost analysis ("a selective read and a non-global latent
    stack are requirements" for 1M–10M), the notebook note's ratio curriculum (P1) and variable-rate notes (A7),
    E36's reconstruction-first writer.
  - Cross-domain: **progressive (embedded) coding** in image compression (any prefix of the bitstream decodes to
    the best image at that rate); **chunking** in memory psychology (H-Net's stated inspiration); and
    **hierarchical clustering for nearest-neighbour search** (the inverted-file / tree indexes of vector
    databases) — LSA is that index made differentiable around, not through, the selection.
- **Staging (one change per step, each gated).**
  1. **Ordered notes** (no new module; cheapest; gives the curve). On the no-loop text notebook (E31c) at 30M: with
     probability 0.5 per batch (NCM's setting) draw a cutoff k from {1, 2, 4, 8, 16} and hide notes k+1…32 from
     every reader; otherwise all 32. Two seeds.
  2. **Content-placed notes** (best-evidenced change; promoted from step 3). Notes are the writer's states at
     boundaries chosen by an H-Net router over the page stream (`W_q = W_k = I` at init), with EMA smoothing on
     the read side, a ratio loss (weight ~0.03) holding the document's average notes per token at today's budget,
     and the boundary vector itself as the note (no pooling operator). Pages with dense facts get more notes,
     filler fewer. Notes-only questions stay in the mix (law 1: the objective must pay for the channel).
  3. **Clustered note index (changed 4 Nov; replaces residual levels).** Apply N-LSA to the *notes* rather than
     raw keys.
     - Every closed block of notes (e.g. 512 notes ≈ 3k tokens) is Ward-merged by its notes' own position-free read
       keys into buckets of ~8 notes. Each bucket keeps its normalised key sum and its note ids.
     - A question scores the bucket centres and expands the best buckets into k notes. The last 256 raw tokens
       (today's window) and a few sink notes are always read.
     - The selection carries no gradient. No new parameters.
     - Above ~1M tokens, a second Ward level (tree search).
     - Optional two-path variant: unread buckets enter the softmax as one summary term each (LSA's `multires`
       code).

     The coarse level of E32 becomes a clustering of the fine notes, not a second writer.
- **Ride-along levers from H-Net's code (each a flag, each measured on its own run when used).**
  - **Raw shortcut starts at zero.** H-Net's full-resolution shortcut `Linear(x̂)` is zero-initialised so the
    compressed path carries the signal first. Our analogue: a learned gate on the reader's last-256-token raw
    read, starting at 0, while the note read keeps its warm start (our cold-start law). Directly targets the
    project's recurring failure: "if the model can still see the raw text, it ignores the notebook".
  - **Recurrent writer.** A small gated linear recurrence (Mamba-2-style) over the token stream as the writer's
    encoder instead of page-local attention; it also gives the writer context beyond the page (law 7).
- **Success (step 1).** With all 32 notes: within 5 points of the base on every exam the base passes, at 4k and
  16k. With the first 8 notes: far lookup at 16k ≥ 75 %, or — if the first notes hold gist, as NCM's probe
  predicts — the first 4 notes alone send ≥ 90 % of the read's attention to the right page.
- **Kill (step 1).** The first 8 ordered notes score no more than 10 points above the base reading a random 8 of
  its 32, or any full-notes exam loses more than 5 points.
- **Success / kill (step 2).** Same exams at the same average notes per token: crowded lookup (several facts per
  page) ≥ 10 points above fixed placement and no full-exam loss > 5 points; kill if the ratio swings by more than
  ±25 % over the last half of training (H-Net's instability signature) or crowded lookup does not move.
- **Success / kill (step 3, added 4 Nov).** Trained at 16k, read at 128k:
  - Success: with the clustered read touching ≤ 1/8 of the notes, far lookup, keyed and latest are within 3
    points of the dense note read at every length where the dense note read passes.
  - Success: the read per question costs ≤ 1/8 of the dense note read's FLOPs at 128k.
  - Kill: lookup more than 5 points under the dense note read at 16k (selection hides the fact), or the
    expanded set misses the answer's note in more than 10 % of questions (an oracle check — the answer's
    note is known on the exams).
  - First, run it read-only on an existing checkpoint (cluster the trained notes, no retraining): if
    training-free clustering already holds lookup, step 3's training run is only for the gradient through
    the selection.
- **Early diagnostics.**
  - The rate–distortion curve (T6) and the budget curve (T11).
  - Effective rank per note position.
  - Attention mass on the right page from the first notes.
  - The swap probe (T8).
  - Boundary density on fact sentences vs filler (step 2).
  - Bits per note under 8/4/2-bit rounding.
  - Bucket purity: the share of a bucket's notes that come from the same fact sentence or entity (step 3).
- **Cost.**
  - Step 1: two 30M runs (about 2–3 h each on 4 × 3090) plus reads at six cutoffs.
  - Step 2: M–L effort (a router, an EMA read, a ratio loss, variable notes per page in the write and the mask);
    H-Net reports up to 2× slower training.
  - Step 3: M effort. LSA's Ward merge and bucket read are plain PyTorch in their training code (Apache-2.0); a
    flex-attention mask or gathered keys suffice at our sizes. Their fast kernels need H200 / B300, and N-LSA
    peaks at ~17 % more memory per layer.
- **Risks.** Late notes are trained half as often under nested dropout; ordering may follow position, not
  importance; boundaries learned from next-token loss follow word structure, not facts (H-Net's stage 2 chunks
  are words and phrases) — notes-only questions must pay for placement; variable notes per page complicate the
  closed-window read and batching. Step 3: notes are already compressed, so a bucket of 8 notes spans ~50 tokens
  of content — the "hidden single key" effect LSA measured for large blocks may hit earlier; LSA's native result is
  0.5B at 128k with a weak absolute RULER (32 vs 35 for full attention); a no-gradient selection never learns to
  *look for* a note it does not already score well (our E18 "used but not useful" risk) — the notes-only questions
  and the two-path summaries are the two remedies.
- **Verdict: Adapt.** Hand to `experiment-design` on the author's go-ahead.

### Smaller items

| item | verdict | note |
|---|---|---|
| Coarse-to-fine parallel decoder (NCM) for the crystallisation step | **Watch** | if non-autoregressive decoding is revived, use this recipe (residual multiscale codes, nested dropout, code corruption, straight-through only at the coarsest level), not prefix diffusion (failed 2026-03) |
| Per-step supervised targets for refinement | **already ours** | NCM supervises every level; our per-round targets (E33q, written chains) did the same |
| Latent chain as coarse-to-fine codes (code the written reasoning chain; predict its levels instead of writing it) | **Watch** | a priority-3 route after step removal; NCM's coarse = gist finding warns the coarse levels would not hold the bridge entity |
| Pooling operator for notes (cross-attention latents vs weighted averages vs picking a position) | **stop optimising** | H-Net App. E.1: on par when the encoder is good; spend effort on the writer's encoder and placement |
| Residual levels (NCM-style coarse notes that store what finer ones miss) | **Watch** (was step 3) | LSA shows the coarse level can be a clustering of fine notes, with no second writer; revisit only if the clustered index fails |
| Two-path summary read (unread buckets as one softmax term: mean value, second-order log-sum-exp logit) | **Watch** | in LSA's code as the default native mode but with no published result; the cleanest gradient route to notes the selection skips |
| N-LSA over raw keys (no notes) | **Adopt as control (T3)** | the strongest "keep everything, read selectively" competitor; our notes must beat it on memory and match it on exams |
| Recurrent local layers (Gated DeltaNet / Mamba-2) around the global read | **Watch** | LSA's 0.5B testbed (3 Gated DeltaNet : 1 attention) and H-Net's encoders both use them; a second source for the recurrent-writer lever |
| LatentMoE / sparse experts | **Reject** at 30–100M | revisit only above ~1B |
| Zero-initialised outputs on new memory reads | **Reject** | contradicts our cold-start law; zero-start belongs on the *raw shortcut* instead |
| Byte-level input | **Watch** | not needed for the text checks; relevant for audio |

## 5. What I am not sure about

- The Cartesia post and the MAI report are still unread at the source; their numbers conflict between excerpts.
- H-Net's wins are at mild compression (~2 tokens per deepest position); nothing shows learned placement at our 6+
  tokens per note. NCM's gains are at ≤ 1,024 tokens with one tokenizer per dataset.
- Nested dropout on a *task-trained* writer (not a reconstruction tokenizer) is untested anywhere.
- Our out-of-distribution contrast (64 % vs 0–1 %) is one run, one held-out shape per relation.
- A constraint exam may be unlearnable at 30M from scratch for every model; calibrate before it gates anything.
- LSA (added 4 Nov):
  - Its distilled results sit on two 2026 models we cannot run; its native result is a single 0.5B run, without
    seeds.
  - Clustering *compressed notes* (each already ~6 tokens of content) is untested: buckets of 8 notes summarise
    ~50 tokens, where LSA's block ablation shows coarser units start hiding single facts.
  - The two-path summary read is code only, with no reported result.

## 6. Recommended order of work

Cheapest and most informative first. Nothing here is launched; each step that trains goes through
`experiment-design` first.

| # | what | needs training? | answers |
|---|---|---|---|
| 1 | **Note budget curve (T11)** and **rate–distortion curve (T6)** on the newest text-notebook checkpoints (E33 / E31c): top-k notes per question, and the first / a random k notes per page | no | how selective a reader can be per task; whether a sparse or clustered read can work at all |
| 2 | **Clustered read on existing notes** (step 3 read-only): Ward-cluster a checkpoint's notes per closed block, read the best buckets plus the raw window, and compare with the dense note read at 16k–128k, plus the oracle check that the answer's note is in the expanded set | no | whether LSA-style clustering of *compressed* notes keeps lookup; how much of step 3 a training run must add |
| 3 | **Efficiency multiplier (T1)** from dense checkpoints already on the text board | no | how much cheaper each architecture is, not just pass / fail |
| 4 | **Exam additions**: answer-not-in-the-book (T9), revised facts with stale echoes (T10), surface-noise variant and robustness column (T5), gist-vs-fact swap probe (T8) | no (generator and scorer work) | gaps none of today's exams cover |
| 5 | **Ordered notes (bet step 1)**: two 30M seeds on the text notebook | yes, ~2–3 h per run on 4 × 3090 | the rate–distortion curve from one writer; a free coarse index |
| 6 | **Frontier-shaped control (T3)**, full-attention arm (≤ 32k) | yes | the real competitor at equal size |
| 7 | **Clustered note index trained (bet step 3)**, if step 2 above shows a gap the gradient could close | yes | O(log N) note reads at 128k–1M |
| 8 | **Content-placed notes (bet step 2)** with H-Net's router, smoothing and ratio loss; zero-start raw shortcut and recurrent writer as separate flags | yes, M–L effort | facts get notes, filler does not |
| 9 | **N-LSA ceiling arm (T3)** and the **global-constraint exam (T4)**, both calibrating first | yes | a reference past 32k; the first exam where concepts could beat dense |
| 10 | **Two-size confirmation (T2)** of any park or champion decision these produce | yes | whether the ranking survives scale |

**Still to read at the source.** The Cartesia *From Tokens to Bytes* posts and the MAI-Thinking-1 report. Both
hosts (www.cartesia.ai, microsoft.ai) are blocked from the cloud session; allowing them in the environment's network
settings, or adding the PDF to the repository, would let the secondary numbers in §1.2 and §1.3 be checked.

## Sources

- NCM: https://arxiv.org/abs/2610.14162 · code https://github.com/HazyResearch/ncm · weights
  https://huggingface.co/hazyresearch/ncm-checkpoints · data https://huggingface.co/datasets/hazyresearch/ncm-datasets
- H-Net: https://arxiv.org/abs/2507.07955 · code https://github.com/goombalab/hnet · weights
  https://huggingface.co/cartesia-ai
- Cartesia posts (unread at source): https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1 ·
  https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2 · excerpt sources https://digg.com/ai/b4j5q79q,
  https://github.com/alphaonedev/ai-memory-mcp/issues/8292
- LatentMoE: https://arxiv.org/abs/2601.18089 · Nemotron 3: https://arxiv.org/abs/2512.20856
- Latent Sparse Attention: https://arxiv.org/abs/2610.42147 · code
  https://github.com/Prefrontal-AI/latent-sparse-attention · SGLang fork https://github.com/Prefrontal-AI/sglang ·
  related: NSA https://arxiv.org/abs/2502.11089 · Routing Transformer https://arxiv.org/abs/2003.05997 · Reformer
  https://arxiv.org/abs/2001.04451 · RULER https://arxiv.org/abs/2404.06654
- MAI-Thinking-1 (unread at source): https://microsoft.ai/pdf/mai-thinking-1.pdf · summaries
  https://hyper.ai/en/papers/MAI-Thinking-1 · https://ritvik19.medium.com/papers-explained-577-mai-thinking-1-e5afeca9bbfc ·
  https://www.alphaxiv.org/abs/2606.mai-thinking-1
- Lineage: VAR https://arxiv.org/abs/2404.02905 · RQ-VAE https://arxiv.org/abs/2203.01941 · nested dropout
  https://arxiv.org/abs/1402.0915 · Matryoshka representations https://arxiv.org/abs/2205.13147 · BLT
  https://arxiv.org/abs/2412.09871 · SpaceByte https://arxiv.org/abs/2404.14408 · DLCM https://arxiv.org/abs/2512.24617
