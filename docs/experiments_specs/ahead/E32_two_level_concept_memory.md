# E32 — Two-level concept memory (cheap fine slots + stateful coarse latents, question-driven fetch)

- **Status:** idea / parked. Written on 2026-09-27 so the design is not lost. Do not start
  before the E30 vs E31 limits study ([E31b](E31b_e30_vs_e31_limits.md)) reports. That study
  fixes the numbers this spec leaves open: tokens per entry, fine writer, m, and the curriculum.
- **Serves:** Vision priorities 1–2. A small reasoning model reads a 1M–10M-token context
  through a compact concept memory. The memory has to be cheap at 1M, content-addressed,
  length-invariant, and able to hold many facts, not one.
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-27 · closed —
- **Assessment it follows from:** [E30 vs E31 assessment](../../4_Research_Notes/e30_vs_e31_assessment_20260927.html)
  (criteria, scorecard, 1M / 10M maths) and the E31 [Result](E31_sliding_window_latent_memory.md#result).

> One bet: keep what each architecture proved and drop what each got wrong.
> - **Fine level:** E30-style slots written from main-path states with a 64-token reach. They are cheap (≈ 0.1 M-parameter writer), exact and robust at ≤ 2k, and they sit at about N/6.
> - **Coarse level:** E31-style stateful latents that read *pages of fine slots* in two-way context. The coarse level is the index and the concept layer, at about N/48–N/64.
> - **Both levels are length-invariant:** no absolute address, and slot keys are rotated at the reading position.
> - **The read:** the question reads the coarse level densely, picks the top-k pages, and reads only those pages' fine slots.
> - **Cost:** reading is O(N/48 + k·page) per token instead of O(N/6). Only the coarse index has to sit on the GPU.

## What each predecessor contributes (evidence)

| keep | from | evidence | drop |
|---|---|---|---|
| Cheap writer on main-path states, 64-token reach | E30_ctx | 99 % lookalike-1k and lookup-2k, 3 seeds; writer ≈ 0.13 M params; 0.33 s per 128k row | E30's 16-token reach (the 13-letter salience wall) |
| Length-invariant address (no window-start sinusoid; slot keys RoPE'd at the reading position) | E31-LI | lookup 97–98 % at 32k and 80–81 % at 128k after training on books of at most 16k (2 seeds); 4-hop chain 82 % at 128k | absolute read (both E30_ctx and E31's original address fall to chance by 16k) |
| Stateful latents with two-way page context | E31 | flat per-letter accuracy; shuffled chain 98 %; chain transfers 16× in length | 5 reader entries per latent (0.83 N entries: barely compressed) |
| Lookup → harder task curriculum, step 5e-5 at 2k, 8k/16k stages | E31 protocol | chain from scratch fails with the invariant memory; from lookup weights it transfers | training only at 2k |
| Windowed global read (`message_raw_window`) | E31 ladder | the memory is the only long path; cost linear | full causal raw keys |

## Hypothesis
If the fine level is E30-LI slots (N/6, main-path states, 64-token reach) and the coarse
level is E31-LI latents that read *pages of fine slots* (one latent per 48–64 tokens), and
the question reads the coarse level first and then only the fine slots of its top-k pages
(k ≤ 16), then:
- at 128k, lookup and the 4-hop chain keep ≥ 0.9× the single-level E30-LI / E31-LI accuracy;
- **multi-fact recall** (16–64 facts in the book) keeps ≥ 75 %;
- the per-token read touches ≤ N/40 entries;
- at 1M the GPU-resident memory is ≤ 1/40 of the tokens.

**Because** the coarse level only has to *route* (which page), not *store* the fact. E31b
tests directly whether a latent at 48 tokens can route. The fine level stores the exact
bits at the ratio already proven to work (6 tokens per entry).

## The architectural bet
```
ids ─► main causal path (E18 platform, SWA + windowed global read)
   │
   ├─ fine write   (E30-LI): per window W=256/stride 192, K=32 queries pool main-path layer-0
   │               states (64-token causal reach) → fine slots F  (N/6), key RoPE'd at read pos
   │
   ├─ coarse write (E31-LI): per page P = 384 tokens (= 64 fine slots), K_c latents
   │               (1 per 48–64 tokens) read the page's fine slots with 2 bidirectional
   │               layers + competition → coarse latents Z (N/48–N/64), one reader entry each
   │
   └─ read (every reading token, or after QUERY in the probes):
        1. attend to Z (all pages, dense, length-invariant) → page scores s_p
        2. top-k pages (k = 8–16; straight-through / Gumbel or score-weighted fusion,
           HiLS-style, with a teacher KL from the dense fine read during training)
        3. attend to the fine slots of those pages (+ Z) → answer
```
- **Streaming writer:** windows are written one at a time and only slots are kept (43 MB
  per 1M tokens at our width). This removes the 27 GB activation wall at 1M found in
  the E31 cost analysis.
- **Text mode:** a token reads only pages that closed before it (the E31 "closed windows"
  rule). The probe's single QUERY read is a special case of this.
- **Reasoning loop (later, E33):** repeated read → update → read over Z, for shuffled
  multi-hop chains. That is the "reason in concept space" part of the vision.

## Maths at 1M (30M platform: 1 KV head × 64 dims, 256 B per entry per reading layer)
| memory | entries at 1M | GPU bytes | entries read per token |
|---|---|---|---|
| dense full attention | 1M × 4 layers | 1 GB | 1M |
| E30-LI / E31-LI m1 (single level, N/6) | 175k | 43 MB | 175k |
| E31-LI m5 (N/1.2) | 870k | 213 MB | 870k |
| **E32 coarse (N/48) + top-16 pages × 64 fine** | 22k coarse + 175k fine (CPU) | **5.4 MB GPU** + 43 MB host | **23k** |
| E32 coarse (N/64) | 16k | 4 MB | 17k |
At 10M the E32 read touches ≈ 220k entries (coarse N/48) + 1k fine. A third level
(N/384) is needed there to keep the per-token read ≤ 50k.

## Open questions E31b must answer first (and how they set E32)
1. **How many tokens can one latent / one reader entry cover?** The E31b ratio sweep runs
   6 → 64 tokens and separates write compression from read bandwidth. It sets the coarse
   ratio (48 or 64) and whether coarse latents need m > 1.
2. **Which fine writer?** E30-LI vs E31-LI-m1 at the same N/6 (head-to-head suite, length
   ladder, hard exams). If E31-LI-m1 ≈ E30-LI, the fine level is E30-LI, because it is cheaper.
3. **Does E30-LI go long?** If E30 with the length-invariant address holds to 128k like
   E31-LI, the fine level is settled.
4. **Multi-fact capacity:** does recall8 / recall16 hold at length? This sets whether top-k
   pages are needed to keep precision or only to save cost.

### What E31b has answered so far (28 Sep, first-letter accuracy)
- **(1) Ratio:** the current writers work at 6–12 tokens per entry, and 24 tokens with 4
  entries per latent. **48 fails** for E31 at every read bandwidth and window tried, and for E30.
  The coarse level cannot be a 48-token latent of today's design. It needs a new writer (e.g.
  latents over fine slots, as proposed here) or a coarser 24-token level plus a third level.
- **(2) Fine writer:** it depends on the exam.
  - E30-LI keeps multi-fact recall at length (recall8: 84 % at 128k).
  - E31-LI-m1 is better with decoys (89 % at 128k) and on recall16 at 4k–16k.
  - m1 ≥ m5 everywhere it was measured.
  - Keep both candidates until the seed-0 replicates land.
- **(3) E30-LI goes long on lookup** (83 % at 128k after the 16k stage): yes.
- **New constraint: one read does not do multi-hop.** Neither writer follows 4 hops beyond the
  training length, and parallel chains are at chance. The E33 read → update → read loop must
  come with E32, not after it.
- **Order matters (28 Sep):** the length-invariant boundary address drops slot order, and the
  in-order chain fails. A *scaled* order code (slot keys at a scaled distance before QUERY)
  restores it. The E31 chain then transfers (seed 1: 92 % at 2k, 72 % at 64k, first letter), at
  some cost to lookup transfer before the curriculum stages. E32's levels should carry this
  order code.
- **Coarse level:** latent width 1024 lifts 48 tokens per latent from ≈ 40 % to 84 % (lookup-1k).
- **Metric:** gate on first-letter accuracy (or free-running exact match), never the
  teacher-forced mean, on any exam with several candidate answers.

## Success / kill (draft; freeze after E31b)
- **S1 (routing):** at 32k and 128k, the correct page is in the top-16 for ≥ 95 % of
  questions on lookup / recall16 (measured directly from page scores).
- **S2 (accuracy):** lookup, chain-4 and recall16 at 128k are ≥ 0.9× the single-level
  N/6 memory; at training length they are ≥ 0.95×.
- **S3 (cost):** entries read per token ≤ N/40 at 128k. At 1M a streamed forward fits one
  3090, and its time per token is within 1.5× of local-only.
- **Kill:** routing recall@16 < 80 % at 32k after the curriculum. That means the coarse
  latents do not index, and the next bet goes to a learned retrieval head on fine slots
  (ANN-style), not concepts.

## Follow-ups (not in E32)
- E33 reasoning loop over coarse latents (shuffled k-hop, the BAPO REACHABILITY family).
- Text rung (L7) and a 10M three-level run.
- Delta-rule / matrix latents for streaming updates.
