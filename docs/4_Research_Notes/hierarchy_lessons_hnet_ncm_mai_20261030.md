# Three new architectures and what they teach us: H-Net at scale, Next-Concept Modeling, MAI-Thinking-1 (30 Oct 2026)

Research synthesis requested by the author (30 Oct): what to learn from three recent sources, whether they are
relevant to the vision, and what they suggest for (a) our capability testing and (b) our architecture.
**Revised the same day after reading the primary papers and code** (first version was built partly on search
summaries; corrections are marked "corrected").

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

Every Cartesia-post and MAI number below must be re-checked against the primary page before it enters a spec.

## Brief

- **All three move the heavy work onto a shorter, coarser sequence** — H-Net runs its deep network on learned
  chunks, NCM generates coarse codes first, LatentMoE runs experts in a 4× narrower latent. This is outside
  evidence for the vision's direction. None reaches our ratios (H-Net's deepest level sees about one position per
  9 bytes ≈ 2 tokens; we write one note per 6 tokens and aim for 16–128) or our use (reading a long book).
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
- **Capability testing:** report an **efficiency multiplier** against dense (LatentMoE gives the primary
  definition: invert the baseline's fitted scaling curve); require a ranking to hold at **two sizes** before
  parking or crowning; add a **frontier-shaped dense control**; add a **global-constraint exam** (Sudoku-like,
  where a refining model beat left-to-right 86 % vs 1 % in NCM); make **out-of-distribution and surface-noise
  robustness** a headline column (H-Net's 2-stage model was twice as robust to noisy text as BPE).
- **Architecture — one bet, revised:** the coarse-to-fine notebook, now in the order the evidence supports:
  ordered notes (cheap, gives the rate–distortion curve), then **content-placed notes** with H-Net's router,
  smoothing and ratio loss (the best-evidenced change), then residual levels. Two cheap H-Net levers ride along:
  start the raw-text shortcut at zero, and try a recurrent writer.

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

## 2. Relevance to the vision

| vision priority | H-Net | NCM | LatentMoE / MAI |
|---|---|---|---|
| 1 Reasoning concept core | **supports**: the deep network on learned chunks beats token-level at 1.3B (paper) and reportedly 3.7B / 1T (post) | decode side; per-level targets echo our per-round targets | — |
| 2 ~10M-token context | weak: ~9 bytes per deepest position | none (≤ 1,024 tokens) | the frontier still pays full attention in 1 of 6 layers at 256k; at 1M–10M our O(C·N) read is the open niche |
| 3 Reasoning bandwidth | — | **supports**: coarse-to-fine refinement solves global constraints where left-to-right fails | — |
| 4 Agent-to-agent | — | discrete codes with exact bit counts suit a channel | compress before communicating (same motive) |
| 5 Multimodality | **Watch**: dynamic chunking is modality-agnostic (DNA 3.6×) | the recipe comes from images and audio codecs | — |

**Convergence.** All put compute on a shorter coarse sequence and keep a cheap path back to full resolution
(H-Net's shortcut, NCM's reconstruction, LatentMoE's up-projection). Multi-resolution is the shared pattern.

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

**Our own evidence agrees with the robustness results:** on unseen sentence shapes dense lookup falls from
37–48 % to 0–1 % while the notebook model keeps 64 % (one run, text checks finding 6); H-Net's 2-stage model
is ~2× more robust to noisy text than BPE.

## 3. Capability testing — proposed changes

| # | change | from (primary unless marked) | why for us | cost | verdict |
|---|---|---|---|---|---|
| T1 | **Efficiency multiplier vs dense**: fit dense's score (or loss) against training tokens on its checkpoints (10/25/50/75/100 %) or sizes; a model's multiplier = dense tokens needed for its score ÷ its own tokens. Per task: dense tokens to pass ÷ the model's | LatentMoE effective parameter multiplier; MAI Efficiency Gain (secondary) | "pass / fail" hides how much cheaper; the vision's claim is an efficiency claim | analysis only | **Adopt** |
| T2 | **Two-size rule**: parking an architecture or changing the champion needs the same order at two sizes | MAI ranking flip (secondary); NCM and H-Net both show gaps that *widen* with data/size | E30 parked and E31 crowned at ≤ 50M; E33's "four rounds = one" is a 30M fact | one extra size per decision | **Adopt** (author decision) |
| T3 | **Frontier-shaped dense control**: same size, 5 sliding-window layers (window 512, RoPE) : 1 full global layer without position code | Gemma-style hybrids; MAI (secondary) | isolates what the notebook replaces; it is the real competitor; fair at 128k | config on the perceiver stack (global read positions + `global_nope`) | **Adopt** |
| T4 | **Global-constraint exam** (DNA: 4×4 Latin square / small Sudoku given as facts; text: a seating or schedule puzzle from fact sentences), status `calibrating` | NCM Sudoku (25M: 86 % hard vs AR 1 %, greedy, 12 steps) | every current exam has dense as its ceiling, so the board can show "matches" but never "beats"; rule 3 (2026-10-07) lets a non-dense model be a task's named reference | M | **Adapt** |
| T5 | **Robustness column**: in-distribution vs unseen-sentence-shape score, plus a **surface-noise** variant (case flips, dropped spaces, repeated characters in the document) | H-Net robustness score; Cartesia part II (excerpts) | our clearest notebook-vs-dense contrast is out of distribution | paraphrase2 exams exist; noise is a data transform | **Adopt** |
| T6 | **Rate–distortion curve of the notebook**: read each exam with 1/2/4/8/16/32 notes per page; report notes needed to pass and the share of notes ever read (NCM's codebook use) | NCM nested dropout, App. G | measures compression honestly; none exists for chunk latents | eval flag + analysis | **Adopt** |
| T7 | **Score vs sequential steps** for any refining model (loop rounds, levels), with wall-clock | NCM NFE reporting | turns "the loop does nothing" into a curve | eval loop | **Adopt** where loops exist |
| T8 | **Gist-vs-fact swap probe**: keep a page's first k notes, replace the rest with another document's; does the answer keep the topic but lose the fact? | NCM App. H (fix coarse, resample fine) | says what ordered or coarse notes actually hold | eval only | **Adopt** with the bet |

## 4. Architecture — the bet (revised)

### The coarse-to-fine notebook

- **Novel claim.** One writer whose notes are ordered by importance *and* placed by content gives, at once, the
  rate–distortion curve, a coarse index for two-level reading (E32), and a per-document note budget that follows
  the facts — at no cost at full notes.
- **How the ideas connect.** NCM: nested dropout, residual levels, coarse = gist. H-Net: learned boundaries beat
  fixed pooling; router + smoothing + ratio loss make placement trainable; a recurrent encoder compresses
  better; the squeeze operator is irrelevant. Ours: E32 (coarse routes, fine stores), the notebook note's ratio
  curriculum (P1), variable-rate notes (A7), two-level notebook (A8), E36's reconstruction-first writer.
  Cross-domain: **progressive (embedded) coding** in image compression — any prefix of the bitstream decodes to
  the best image at that rate; and **chunking** in memory psychology, which H-Net names as its inspiration.
- **Staging (one change per step, each gated).**
  1. **Ordered notes** (no new module; cheapest; gives the curve). On the no-loop text notebook (E31c) at 30M: with
     probability 0.5 per batch (NCM's setting) draw a cutoff k from {1, 2, 4, 8, 16} and hide notes k+1…32 from
     every reader; otherwise all 32. Two seeds.
  2. **Content-placed notes** (best-evidenced change; promoted from step 3). Notes are the writer's states at
     boundaries chosen by an H-Net router over the page stream (`W_q = W_k = I` at init), with EMA smoothing on
     the read side, a ratio loss (weight ~0.03) holding the document's average notes per token at today's budget,
     and the boundary vector itself as the note (no pooling operator). Pages with dense facts get more notes,
     filler fewer. Notes-only questions stay in the mix (law 1: the objective must pay for the channel).
  3. **Residual levels.** A coarse level (e.g. one note per page) plus finer levels that encode what it misses,
     with nested dropout across levels and ~20 % reconstruction (E36 recipe). The coarse level becomes E32's index.
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
- **Early diagnostics.** The curve (T6); effective rank per note position; attention mass on the right page from
  the first notes; the swap probe (T8); boundary density on fact sentences vs filler (step 2); bits per note under
  8/4/2-bit rounding.
- **Cost.** Step 1: two 30M runs (about 2–3 h each on 4 × 3090) plus reads at six cutoffs. Step 2: M–L effort (a
  router, an EMA read, a ratio loss, variable notes per page in the write and the mask); H-Net reports up to 2×
  slower training.
- **Risks.** Late notes are trained half as often under nested dropout; ordering may follow position, not
  importance; boundaries learned from next-token loss follow word structure, not facts (H-Net's stage 2 chunks
  are words and phrases) — notes-only questions must pay for placement; variable notes per page complicate the
  closed-window read and batching.
- **Verdict: Adapt.** Hand to `experiment-design` on the author's go-ahead.

### Smaller items

| item | verdict | note |
|---|---|---|
| Coarse-to-fine parallel decoder (NCM) for the crystallisation step | **Watch** | if non-autoregressive decoding is revived, use this recipe (residual multiscale codes, nested dropout, code corruption, straight-through only at the coarsest level), not prefix diffusion (failed 2026-03) |
| Per-step supervised targets for refinement | **already ours** | NCM supervises every level; our per-round targets (E33q, written chains) did the same |
| Latent chain as coarse-to-fine codes (code the written reasoning chain; predict its levels instead of writing it) | **Watch** | a priority-3 route after step removal; NCM's coarse = gist finding warns the coarse levels would not hold the bridge entity |
| Pooling operator for notes (cross-attention latents vs weighted averages vs picking a position) | **stop optimising** | H-Net App. E.1: on par when the encoder is good; spend effort on the writer's encoder and placement |
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

## Sources

- NCM: https://arxiv.org/abs/2610.14162 · code https://github.com/HazyResearch/ncm · weights
  https://huggingface.co/hazyresearch/ncm-checkpoints · data https://huggingface.co/datasets/hazyresearch/ncm-datasets
- H-Net: https://arxiv.org/abs/2507.07955 · code https://github.com/goombalab/hnet · weights
  https://huggingface.co/cartesia-ai
- Cartesia posts (unread at source): https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1 ·
  https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2 · excerpt sources https://digg.com/ai/b4j5q79q,
  https://github.com/alphaonedev/ai-memory-mcp/issues/8292
- LatentMoE: https://arxiv.org/abs/2601.18089 · Nemotron 3: https://arxiv.org/abs/2512.20856
- MAI-Thinking-1 (unread at source): https://microsoft.ai/pdf/mai-thinking-1.pdf · summaries
  https://hyper.ai/en/papers/MAI-Thinking-1 · https://ritvik19.medium.com/papers-explained-577-mai-thinking-1-e5afeca9bbfc ·
  https://www.alphaxiv.org/abs/2606.mai-thinking-1
- Lineage: VAR https://arxiv.org/abs/2404.02905 · RQ-VAE https://arxiv.org/abs/2203.01941 · nested dropout
  https://arxiv.org/abs/1402.0915 · Matryoshka representations https://arxiv.org/abs/2205.13147 · BLT
  https://arxiv.org/abs/2412.09871 · SpaceByte https://arxiv.org/abs/2404.14408 · DLCM https://arxiv.org/abs/2512.24617
