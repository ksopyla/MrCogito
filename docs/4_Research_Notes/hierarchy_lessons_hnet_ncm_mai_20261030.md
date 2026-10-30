# Three new architectures and what they teach us: H-Net at scale, Next-Concept Modeling, MAI-Thinking-1 (30 Oct 2026)

Research synthesis requested by the author (30 Oct): what to learn from three recent sources, whether they are
relevant to the vision, and what they suggest for (a) our capability testing and (b) our architecture.

Builds on (not repeated here): [notebook_writer_compression_20261016.md](notebook_writer_compression_20261016.md)
(BLT, H-Net 2025, DLCM, Large Concept Models, rate-matched notebook, candidates A1–A10 / P1–P4),
[e33_multihop_review_label_free_20261018.md](e33_multihop_review_label_free_20261018.md) (written chains, step
removal), the [capability checks](../engineering_specs/capability_checks.md) and
[text checks](../engineering_specs/text_capability_checks.md) protocols, and E32
([two-level concept memory](../experiments_specs/ahead/E32_two_level_concept_memory.md)).

## How the sources were read (trust level)

| source | read from | trust |
|---|---|---|
| Next-Concept Modeling (NCM), Kumbong, Khare, … Arora, Mirhoseini, Ré (Stanford), [arXiv 2610.14162](https://arxiv.org/abs/2610.14162), 9 Oct 2026 | full paper text (main body; appendix C–H not read) via the Hugging Face paper mirror | primary |
| Cartesia, *From Tokens to Bytes I: Scaling Hierarchical Networks*, [blog](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1), 28 Oct 2026 (+ [part II](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2)) | **search-result excerpts only** (cartesia.ai is blocked from the session); method background from the H-Net paper [arXiv 2507.07955](https://arxiv.org/abs/2507.07955) | secondary; numbers conflict between excerpts (below) |
| Microsoft AI, *MAI-Thinking-1: Building a Hill-Climbing Machine*, [report](https://microsoft.ai/pdf/mai-thinking-1.pdf) (p. 6 = architecture), June 2026 | **secondary summaries only** (microsoft.ai blocked): [hyper.ai](https://hyper.ai/en/papers/MAI-Thinking-1), [Papers Explained 577](https://ritvik19.medium.com/papers-explained-577-mai-thinking-1-e5afeca9bbfc), [alphaXiv](https://www.alphaxiv.org/abs/2606.mai-thinking-1) | secondary; the architecture facts agree across 3+ summaries, one outlier |

Re-check every Cartesia and MAI number against the primary page before it enters a spec.

## Brief

- **All three move work into a compressed, coarser representation and win by doing so.** H-Net runs its main
  network on learned byte chunks (claimed ~10× data efficiency vs a 3B tokenized model); NCM generates text as
  coarse codes first and fine codes last (10–31× faster than diffusion LMs at equal or better quality, Sudoku far
  above autoregressive models); MAI computes its experts inside a compressed latent. This is outside evidence that
  the vision's direction is right. None of them targets our ratio (4–128 tokens per concept) or our use (reading
  and reasoning over a 1M-token book), so the niche is still ours.
- **"Next-concept modeling" is now a published name.** NCM's "concepts" are the coarse levels of a residual code
  used for fast *generation*; ours are notes used for *reading and reasoning*. Cite it and state the difference
  in any publication; the two are complementary (an NCM-style decoder is a candidate for our "crystallisation"
  step).
- **Capability testing — four changes worth adopting:** report an **efficiency multiplier** against dense (how
  much more training dense needs to match), require the ranking to **hold at two sizes** before parking or
  crowning an architecture, add a **frontier-shaped dense control** (5 sliding-window layers : 1 global layer
  without position code, the MAI / Gemma layout), and add a **global-constraint exam** (Sudoku-like) — the first
  task where a refining concept core could *beat* dense instead of only matching it.
- **Architecture — one bet: the coarse-to-fine notebook.** Write each page's notes in order of importance
  (NCM's nested dropout), so the first few notes are a usable summary and later ones add detail; then make the
  levels residual; then let a learned boundary router (H-Net) decide where fine notes go. One writer then gives
  the rate–distortion curve, E32's coarse index and variable-rate notes. Step 1 is one config-sized change on the
  no-loop text notebook.

## 1. Per source

### 1.1 Next-Concept Modeling (Stanford / Hazy Research, arXiv 2610.14162)

- **Thesis.** Generate text coarse-to-fine: a frozen multiscale residual VQ-VAE turns a sequence into K levels of
  discrete codes (short coarse levels, a full-length last level); a transformer predicts all codes of one level in
  parallel, given the cumulative reconstruction of the earlier levels. K ≪ L sequential steps.
- **Technique.**
  - Tokenizer: encoder `z = E(x) ∈ R^{L×d}`; level k downsamples the residual to `ℓ_k` vectors
    (`ℓ_1 < … < ℓ_K = L`), quantizes each to its nearest entry of codebook `V_k`, upsamples back with a
    reconstruction operator `R_k` and adds it to `ẑ^{(k)}`; the next level codes `z − ẑ^{(k)}`.
  - **Nested dropout:** in training a random cutoff k keeps only the first k levels; the reconstruction
    cross-entropy and VQ losses are taken at that cutoff, so **every prefix of levels is a usable approximation**.
  - Generator: one forward pass trains all levels (block-diagonal attention per level); **conditioning
    corruption** (replace a fraction of conditioning codes with random codes of the same level, keep true codes as
    labels) for robustness to its own errors; level weights `w_k ∝ ℓ_k^{-α}`. Conditional generation by a
    prefix-LM mask.
- **Evidence (~100M generators, one tokenizer per dataset).** LM1B: better generative perplexity than the best
  diffusion LM (DUO) at 21× less wall-clock. OpenWebText: competitive at 10×. XSum: ROUGE-L 28.0 at 8 steps vs 26.6
  for the best diffusion LM at 128 (31× faster); with MBR selection 29.6. WMT14 De–En: 22.6 BLEU at 11 steps (AR
  25.2 at 64). **Sudoku (25M): hard boards 86.1 % at 12 steps, vs AR 1.0 %, best diffusion 81.3 %.** 4.9× fewer
  training tokens than diffusion baselines on LM1B (tokenizer included). 97M → 317M → 725M improves at a fixed
  token budget. Tokenizer rebuilds text near 100 %, codebook use 62–90 %. Probes: coarse levels hold global
  structure, fine levels detail.
- **Weak points.** Short, fixed lengths (LM1B 128, OWT 1024 tokens); a separate tokenizer per dataset; quality
  judged by GPT-2-large perplexity; "concept" is defined as "what the coarse levels carry", not shown to be
  semantic; no long context; the only reasoning task is Sudoku.
- **Code.** [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (MIT), data
  [hazyresearch/ncm-datasets](https://huggingface.co/datasets/hazyresearch/ncm-datasets). Lineage: next-scale
  prediction for images (VAR, [2404.02905](https://arxiv.org/abs/2404.02905)), residual quantization (RQ-VAE,
  [2203.01941](https://arxiv.org/abs/2203.01941)), nested dropout ([1402.0915](https://arxiv.org/abs/1402.0915)).
- **Mapping.** Writer (ordered, multi-rate notes with an exact bit count), decoder (a parallel crystallisation
  step), evaluation (constraint reasoning; steps-vs-quality curves).
- **Verdict: Adapt** (nested-dropout residual notes; global-constraint exam). **Watch** as a future decoder.
- Full review: [concept_modeling_encoding.md § Next-Concept Modeling](../literature_review/concept_modeling_encoding.md#next-concept-modeling-fast-parallel-coarse-to-fine-text-generation).

### 1.2 Cartesia, From Tokens to Bytes I (H-Net scaled)

- **Thesis.** A byte-level hierarchical network that learns its own chunk boundaries (dynamic chunking) and runs
  its main network on the chunks is more compute- and data-efficient than a tokenized transformer, at billions of
  parameters and ~1T tokens.
- **Technique (H-Net, 2507.07955, plus the post).** Encoder layers over bytes → a **routing module** that marks a
  boundary where adjacent encoder states are dissimilar (cosine) → the main network runs only on boundary
  positions → a smoothing / de-chunk step brings states back to byte resolution → decoder layers. **Two stages**
  of chunking (bytes → chunks → larger chunks). An **auxiliary ratio loss** holds the compression at a target; the
  post reports a target of about one main-network position per 8 bytes, ~7.6 reached (≈ 40 % fewer positions
  than a Llama-3-tokenized model on the same text).
- **Evidence (from excerpts).** A 3.67B two-stage H-Net, trained on a best-effort reproduction of SmolLM3's stage-1
  data mix, reaches SmolLM3's (3.08B) final six-task average after ~1T tokens vs ~11T (the post says ">10×
  compute and data efficiency"; it also cautions that sizes and data differ). Part II: in a loss-to-loss analysis
  the advantage grows from ~1.3× on the training mix to ~2× on downstream tasks, and from ~2× to ~5× on code
  and out-of-domain languages; pipelines that fix chunks offline do not match end-to-end learned chunks.
  **Conflict:** one excerpt quotes a 1.28–2.06× pretraining compute gain; the 10× is likely the comparison with
  SmolLM3's run, not a matched ablation. No weights released yet.
- **Mapping.** Writer placement (where notes go), the hierarchy (E32), the audio adapter (priority 5).
  The compression is mild (~1.7 BPE tokens per position) — it validates *trainable learned compression*, not
  our 6–128× ratios.
- **Verdict: Adapt** (learned note placement with a ratio loss, step 3 of the bet). **Watch** for audio
  (learned chunking over codec frames).

### 1.3 MAI-Thinking-1 (Microsoft AI), architecture page and process

- **Thesis.** A frontier reasoning model built by a "hill-climbing machine": many small, measured improvement
  loops over data, architecture, infrastructure, RL and evaluation, each validated on scaling ladders.
- **Architecture (p. 6, via summaries).** Sparse MoE, ~35B active / ~1T total (one outlier summary says 38B / 480B,
  192 experts — treat as unreliable). **Attention 5 local : 1 global**; local = 512-token sliding window with RoPE;
  **global = no position code (NoPE)**; 256k context. **LatentMoE** (from NVIDIA): a shared down-projection before
  the all-to-all dispatch, experts compute in the compressed latent, outputs projected back; routing from the
  original representation; 8 of 512 experts; no shared experts. High-sparsity MoE layers **interleaved with small
  dense FFNs**. Attention output starts at zero (output RMSNorm gains zero-initialised). Pretraining 30T tokens +
  3.55T mid-training.
- **Process.** Scaling ladders at a constant tokens-per-parameter ratio; an **Efficiency Gain** metric (fit a
  scaling law to the baseline, express a candidate as the compute multiplier the baseline would need to match it);
  single-source data ablations (one source upsampled to 50 %) and in-mixture ladder ablations. A reported lesson:
  a small-scale ranking of two data mixtures **reversed** at 23B active / ~20T tokens (secondary source). A critic
  notes the report gives no matched per-component ablations.
- **Mapping.** The dense control (the competitor we must beat is this layout, not full attention everywhere); our
  length-invariant, position-free note matching (E31 / E31c) agrees with NoPE global layers; evaluation process.
- **Verdict: Adopt** the evaluation practice and the frontier-shaped control. **Reject** LatentMoE at our scale
  (30–100M). **Reject** zero-initialised outputs on any notebook read: it contradicts our own cold-start law (two
  zero gates in series leave only a document embedding) — fine for one gate on full attention, not for a new
  memory path.

## 2. Relevance to the vision

| vision priority | H-Net | NCM | MAI |
|---|---|---|---|
| 1 Reasoning concept core | **supports**: a main network on learned chunks beats token-level at 3.7B / 1T | supports on the decode side; per-level supervised refinement echoes our own finding that each loop round needs its own target | neutral |
| 2 ~10M-token context | weak: ratio ~1.7 tokens per position | none (≤ 1k tokens) | **sharpens the niche**: the frontier still pays full attention in 1 of 6 layers at 256k; at 1M–10M that is the cost our O(C·N) read removes |
| 3 Reasoning bandwidth | — | **supports**: coarse-to-fine refinement of a whole answer solves global constraints where left-to-right fails (Sudoku 86 % vs 1 %) | — |
| 4 Agent-to-agent | — | discrete codes with exact bit counts suit a messaging channel | LatentMoE compresses before communication (same motive, different scale) |
| 5 Multimodality | **Watch**: dynamic chunking is modality-agnostic (also reported on DNA) — the natural writer for audio frames | the recipe comes from images (VAR) and audio codecs (residual VQ) | — |

**Convergence.** Three independent groups put the heavy compute on a shorter, coarser sequence and keep a cheap
path back to full resolution (H-Net de-chunk, NCM reconstruction operator, LatentMoE up-projection). That is our
encode → reason → crystallise pipeline. **Multi-resolution is the shared pattern:** two chunking stages, K code
levels, 512-token local + global. Our notebook has one resolution (32 notes per page) plus the raw window.

**Tension: end-to-end vs staged.** H-Net learns boundaries end to end and says offline chunking loses; NCM trains
the tokenizer first and freezes it. Our own E36 (frozen-reader writer) found reconstruction-first training at
growing spans lifted the bits per entry from 32–48 to 114–145, and the 18 Oct review found task-free continuation
alone erases note content. Resolution: train the hierarchy with nested-dropout reconstruction first, then
unfreeze end to end with notes-only questions (the staging LCLM also reports).

**Our own evidence agrees with Cartesia part II:** on unseen sentence shapes dense lookup falls from 37–48 % to
0–1 % while the notebook model keeps 64 % (one run, text checks finding 6). Compression models may generalise
better out of distribution — worth a headline column, not a footnote.

## 3. Capability testing — proposed changes

| # | change | from | why for us | cost | verdict |
|---|---|---|---|---|---|
| T1 | **Efficiency multiplier vs dense**: per task, dense's tokens to pass ÷ the model's; on story loss, the training tokens dense needs to reach the model's final loss (read off dense's checkpoints at 10/25/50/75/100 %) | MAI Efficiency Gain; Cartesia tokens-to-match | "passes / fails" hides *how much cheaper*; the vision's claim is an efficiency claim | analysis only, existing checkpoints | **Adopt** |
| T2 | **Two-size rule**: parking an architecture or changing the champion needs the same order at two sizes (30M and 100M text tiers, or 10M and 50M DNA) | MAI ranking flip | E30 was parked and E31 crowned at ≤ 50M; E33's "four rounds = one" is a 30M fact | one extra size per decision | **Adopt** (author decision) |
| T3 | **Frontier-shaped dense control**: same size, 5 sliding-window layers (window 512, RoPE) : 1 full global layer without position code | MAI, Gemma-style hybrids | isolates exactly what the notebook replaces (the global layers); it is the real competitor; NoPE global is the known length-transfer recipe, so it is fair at 128k | config on the existing perceiver stack (global read positions + `global_nope`) | **Adopt** |
| T4 | **Global-constraint exam** (DNA: a 4×4 Latin square / small Sudoku given as facts in the book; text: a seating or schedule puzzle from fact sentences), status `calibrating` | NCM Sudoku (25M: 86 % hard vs AR 1 %) | every current task has dense as its ceiling, so the board can show "matches dense" but never "beats dense"; constraint solving is where the vision says latent refinement should win; rule 3 (2026-10-07) already lets a non-dense model be a task's named reference | M (generator + floors audit) | **Adapt** |
| T5 | **Out-of-distribution gap as a headline column**: every model's in-distribution vs unseen-sentence-shape score side by side; later a held-out filler domain | Cartesia part II (advantage 2× → 5× out of domain) | our strongest notebook-vs-dense contrast so far is out of distribution | paraphrase2 exams exist; board column | **Adopt** |
| T6 | **Rate–distortion curve of the notebook**: read each exam with 1/2/4/8/16/32 notes per page; report notes needed to pass and the share of notes ever read | NCM nested dropout + codebook use | the notebook note says no such curve exists for chunk latents; it measures compression honestly | eval flag + analysis | **Adopt** (pairs with the bet) |
| T7 | **Score vs sequential steps** for any refining model (loop rounds, levels): accuracy at 1…R, plus wall-clock | NCM's NFE reporting | turns "the loop does nothing" into a curve; needed for priority 3 claims | eval loop | **Adopt** where loops exist |

## 4. Architecture — the bet

### The coarse-to-fine notebook

- **Novel claim.** One writer whose notes are *ordered by importance* gives, at once: the rate–distortion curve,
  a coarse index for two-level reading (E32), and an exact bit count per note — and it costs nothing at full
  notes. Surprising if true because today's 32 notes per page are exchangeable; ordering them is a free
  hierarchy.
- **How the ideas connect.** NCM: nested dropout over levels and residual coding; H-Net: learned boundaries with a
  ratio loss, two stages; ours: E32's coarse-then-fine fetch, the notebook note's ratio curriculum (P1),
  variable-rate notes (A7) and two-level notebook (A8), E36's reconstruction-first writer.
  Cross-domain hook: **progressive (embedded) coding** in image compression (JPEG 2000 / embedded zerotree): any
  prefix of the bitstream decodes to the best image at that rate. Nested dropout trains a writer to be an
  embedded code.
- **Staging (one change per step, each with its own gate).**
  1. **Ordered notes** (no new module). On the no-loop text notebook (E31c) at 30M: with probability 0.5 per page
     sample a cutoff k ∈ {1, 2, 4, 8, 16} (log-uniform) and hide notes k+1…32 from every reader; otherwise keep
     all 32. The learned note queries fix the order. Two seeds.
  2. **Residual levels.** Level 1 writes a few notes per page; a reconstruction operator rebuilds the page's token
     states from them; level 2 writes from what is left (state minus reconstruction), and so on. Losses:
     reconstruction at a sampled cutoff (~20 % weight, the E36 recipe) + notes-only questions (the channel must
     be paid for).
  3. **Learned placement.** The finest level's notes go where an H-Net-style router marks boundaries (dissimilar
     adjacent writer states), with a ratio loss holding notes per token at the page budget averaged over the
     document — dense facts get more notes, filler fewer.
- **Success (step 1).** With all 32 notes: within 5 points of the base on every exam the base passes, at 4k and
  16k (no-harm). With only the first 8 notes: far lookup at 16k ≥ 75 %.
- **Kill (step 1).** The first 8 ordered notes score no more than 10 points above the base model reading a random 8
  of its 32 notes (ordering bought nothing), or any full-notes exam loses more than 5 points.
- **Early diagnostics.** Far lookup vs notes read (T6); effective rank per note position (the first notes should
  carry the most); page routing accuracy from the first note alone at 16k; bits per note under 8/4/2-bit
  rounding.
- **Cost.** Step 1: two 30M runs on the text notebook (about 2–3 h each on 4 × 3090, the E31c figure in the
  notebook note) plus exam reads at six cutoffs.
- **Risks.** Nested dropout halves how often late notes are trained — late facts on a page may suffer (watch
  crowded lookup); ordering may follow page *position* (first notes = first sentences) rather than importance —
  the effective-rank and per-position read probes show which; H-Net boundaries learned from next-token loss
  follow word structure, not facts (law 1: the objective must pay for the channel), so step 3 needs the
  notes-only questions in the mix.
- **Verdict: Adapt.** Hand to `experiment-design` on the author's go-ahead.

### Smaller items

| item | verdict | note |
|---|---|---|
| Coarse-to-fine parallel decoder (NCM) for the crystallisation step | **Watch** | if non-autoregressive decoding is revived, use this recipe (residual multiscale codes, nested dropout, conditioning corruption), not prefix diffusion (failed 2026-03) |
| Per-step supervised targets for refinement | **already ours** | NCM supervises every level; our E33q / written-chain runs showed loop rounds need their own targets. Convergent, not new |
| Latent chain as coarse-to-fine codes (code the written reasoning chain with a small residual VQ; predict the levels instead of writing the chain) | **Watch** | a priority-3 route after step removal; does not obviously fix 3-hop generalisation, which is the open problem |
| Local 512 window + NoPE global around the notebook | **Adopt as control (T3)** | the notebook's raw window is 256; matching the control's window keeps the comparison clean |
| LatentMoE / sparse experts | **Reject** at 30–100M | revisit only above ~1B |
| Zero-initialised attention outputs | **Reject** on read paths | contradicts our cold-start law |
| Byte-level input | **Watch** | not needed for text checks (4k-word vocabulary); relevant for audio |

## 5. What I am not sure about

- The Cartesia and MAI numbers come from excerpts and summaries, not the pages. The "10×" vs "1.3–2×" Cartesia
  gap and the MAI expert count conflict between sources.
- NCM's gains are at ≤ 1k tokens and ~100M parameters, with one tokenizer per dataset; whether nested dropout
  helps a *task-trained* writer (not a reconstruction tokenizer) is untested anywhere.
- Our out-of-distribution contrast (64 % vs 0–1 %) is one run with one held-out shape per relation.
- A global-constraint exam may be unlearnable for every model at 30M from scratch; NCM's Sudoku result used
  25M with a dedicated tokenizer. Calibrate before it gates anything.

## Sources

- Next-Concept Modeling: https://arxiv.org/abs/2610.14162 · code https://github.com/HazyResearch/ncm · data
  https://huggingface.co/datasets/hazyresearch/ncm-datasets · project page
  https://scalingintelligence.stanford.edu/pubs/ncm/
- Cartesia: https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1 ·
  https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2 · H-Net paper https://arxiv.org/abs/2507.07955 ·
  summaries https://digg.com/ai/b4j5q79q, https://github.com/alphaonedev/ai-memory-mcp/issues/8292
- MAI-Thinking-1: https://microsoft.ai/pdf/mai-thinking-1.pdf · https://hyper.ai/en/papers/MAI-Thinking-1 ·
  https://ritvik19.medium.com/papers-explained-577-mai-thinking-1-e5afeca9bbfc ·
  https://www.alphaxiv.org/abs/2606.mai-thinking-1 · https://yage.ai/share/mai-thinking-1-hill-climbing-en-20260603.html
- Lineage: VAR https://arxiv.org/abs/2404.02905 · RQ-VAE https://arxiv.org/abs/2203.01941 · nested dropout
  https://arxiv.org/abs/1402.0915 · Matryoshka representations https://arxiv.org/abs/2205.13147 · BLT
  https://arxiv.org/abs/2412.09871 · DLCM https://arxiv.org/abs/2512.24617
