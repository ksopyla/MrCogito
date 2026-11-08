# E37 — Concept memory: an encoder, a reasoner and a decoder read for 10M-token context

- **Status:** draft, exploration program (author, 2026-11-07: "explore architectures, change them, test on AWS,
  fix, improve … LSA results as the reference target, with a different architecture"). Stage A code built and
  unit-tested 2026-11-07 (`nn/concept_memory.py`, commit 95ee75d); GPU check and runs pending compute.
- **Serves:** the Vision (compress context into concepts, reason in concept space) and its priority 2 (~10M tokens)
  with priority 1 (reasoning over the concepts). New family: a different encode / reason / decode object from E33's
  notebook (concepts are content clusters of the reading layer's own keys with links back to their tokens, not
  writer notes per page).
- **Implementation plan:** this spec's "Plan" section until a separate `E37_plan.md` is needed.
- **Owner / dates:** Krzysztof Sopyla · opened 2026-11-07 · closed —
- **Evidence behind it:** E33p stage 1 (2026-11-07, 30M, LongDocs-lite at 16k): LSA's read ported to our model
  recovers 99 % of the window→dense gap, our notes-as-index read 39 %, and the limits were measured (empty notes,
  unused budget, summaries crowding exact tokens, fade with length; analysis page
  `docs/3_Evaluations_and_Baselines/e33p_vs_lsa.html`). Literature scout of 2026-11-07 (chat): SPLA (2601.22379),
  RetroInfer (2505.02922), HiLS (2607.02980), HSA / RAMba (2504.16795), Cramming (2502.13063), Huginn (2502.05171),
  Looped Models Done Right (2610.23036), TRM (2510.04871); LSA (2610.42147) and its code.

## Hypothesis
If each past token is read at the resolution its contribution deserves — exactly for the content concepts with the
largest estimated attention mass, and through one calibrated second-order summary for every other concept — a 100M
model matches full attention and LSA's read on long natural documents (≥ 90 % of the window→dense gap at 16k, no
fade to 64k), at LSA's budget of 512 exact tokens per query; and a small looped **reasoner** over the concepts adds
value that neither full attention nor LSA has (≥ 3 points of the gap, or ≥ 0.01 nats on far-copy tokens).

## The three blocks
At the 100M scale (width 896, 3 local prelude layers, the reading layer + 1 local layer looped 4 times, coda; one
KV head of 64, no position code in the reading layer), 16k-token rows:

1. **Encoder — memory with links.** The layers below the reading layer turn tokens into states x, keys k, values v.
   Every closed block of 512 keys is grouped into **64 concepts by content**: farthest-point start, then three
   rounds of Sinkhorn-balanced soft k-means on the unit keys (no gradient). A concept keeps the positions of its
   member tokens (its links) and **additive sufficient statistics**: count n, mean key μ_k, mean value μ_v,
   isotropic key variance σ², mean state z. At 16k: 30 closed blocks, 1,920 concepts of ~8 tokens. Statistics add,
   so concepts of concepts are exact sums (long pages, stage C).
2. **Reasoner — thinking over the concepts** (arm e37b). Two tied transformer layers of width 512, looped twice
   with the input added back each loop, over the concept states z. A concept attends to the concepts of its own and
   earlier blocks in the same document (causal by block), so it can connect what it holds with what came before.
   Its output is added to the concept's value (zero at the start: e37b begins as e37a). 6.8M parameters (+7 %).
3. **Decoder read — exact where it matters, coarse elsewhere.** Each query reads one softmax over: its document's
   first 4 tokens (sinks); the raw tokens of its own block and the block before (block-aligned window, 512–1,023
   tokens); the member tokens of the concepts with the largest estimated mass (best query head, best concept first,
   until 512 tokens); and **one summary entry per other concept** of older blocks, with
   logit = log n + q·μ_k/√d + |q|² σ²/(2d) (the exponential kernel's moment-generating function to second order)
   and value μ_v (+ reasoner). Every past token of the document is counted exactly once. With pure concepts
   (identical member keys) the read equals full attention; with everything opened it equals full attention for any
   keys (tests T3, T4).

**Analogies.** The fast multipole method of N-body physics (near field exact, far field by multipole moments; the
summary logit is a monopole + quadrupole term); a library's catalogue (cards grouped by subject, a label per box,
open the boxes that matter); the hippocampal indexing theory of memory (an index of links back to the full traces,
consolidated offline — the reasoner).

## Builds-on
- **Foundation:** `PerceiverARLM` with `par_global_read = "concept"` (new, `nn/concept_memory.py`), reusing
  `nn/exact_read.py` (member groups, per-query choice table, the flex `mask_mod`), the LSA testbed
  (`scripts/run_e33n.py`, `infra/aws/jobs/e33n_run.sh` with multi-GPU and spot checkpoints, `data/bin_token_dataset.py`),
  the evaluation suite (`evaluation/long_doc_loss.py`, `run_lsa_suite.py` with the 2026-11-07 RULER fix,
  `flow_probe.py`), and the E33p instrumentation (`nn/flow_stats.py`, new `reading_concept`).
- **Init / checkpoint:** random init, the LSA testbed protocol.
- **Baseline to beat:** E33p stage 1 at 30M (16k, positions ≥ 2k, share of the window→dense gap): LSA-native 99 %
  (127 % at 64k), notes-open 39 % (21 % at 64k). At 100M the controls are retrained in stage A.
- **Materially new:** vs LSA — balanced Sinkhorn concepts instead of Ward merges; every unread concept stays in the
  read as a calibrated summary with gradient to its members (LSA drops them in the paper; its code's `multires` mode
  has mean values and no published result); opening by estimated mass instead of cosine; a reasoner over the
  concepts. Vs SPLA — content clusters instead of positional blocks, softmax summaries instead of a linear-attention
  residual, trained from scratch at 100M. Vs E33 — no writer, no page notes; the concepts are the reading layer's own
  keys grouped by content, so the index is calibrated by construction (the notes' failures: empty notes, blurry keys,
  unused budget, summaries crowding the exact tokens).

## Why this is not a safe retread
It is not a knob on E33's notebook or a copy of LSA: the memory is a mixture of exact tokens and second-order kernel
summaries chosen per query by estimated mass, with a reasoning network that works on the summaries. The summary math
is checked exact in the limits (tests), the reasoner is the Vision's "reason in concept space" made concrete.

## Arms
Stage A, 100M (`m100`), all with E33's loop (4 passes) and one KV head, no position code on the reading layer:
- **dense** (ceiling), **window** (floor: the last 512 tokens only), **lsa** (LSA's read, our port: E33p's LSA-native),
- **notes_open** (best E33-family read on natural long text, E33p; the parent session's catalogue read joins if ready),
- **e37a** (encoder + decoder read), **e37b** (+ reasoner).
- **Compression ladder** (30M home screen): **e37a_c32** and **e37a_c128** are e37a with 16 and 4 concepts per
  512-key block (32 and 128 keys per concept), with the same 512 opened tokens. Only the far field gets coarser, so
  the ladder measures how much of the read the summaries carry, and whether few rich summaries are enough. A 10M
  context needs few, rich summaries. K1's "summaries off" fallback is the end of this ladder.
- **Original LSA** (their trainer `lsa native train`, the `qwen35_0_8b_gpt2/lsa_mqa8` preset scaled to ~100M,
  Gated DeltaNet hybrid) — stage B, only if their stack runs on our GPUs (torch 2.13, CUTLASS DSL; kernels target
  H200 / B300; a reference path exists). Our port reproduces their read exactly (E33p T12), so stage A already
  compares the read; stage B adds their backbone.

## Training protocol
Stage A screening: FineWeb-Edu GPT-2 tokens at 4k, 1.0B tokens (warmup 1 %, constant LR 6.4e-4), then LongDocs-lite
at 16k, 0.2B tokens (fresh optimizer, 3 % warmup, 1 − sqrt decay to 0.1 ×). 128 rows of 4k per step. Stage B: the
full m100 protocol (3.0B + 0.4B). Stage D (data): 5 % synthetic retrieval and multi-hop documents in the long decay
(HiLS's recipe), a length curriculum 4k → 16k → 64k (SkyLadder), related-document packing.

## Evaluation
Long-document loss by position at 4k–128k (share of the window→dense gap on positions ≥ 2k; 128k reported raw), the
copy split (far-copy tokens), RULER at 4k and 16k, core benchmarks, the flow probe (recall of dense's top-32 keys in
what each read opens; copy-source hit rate; attention by distance), tokens per second and peak memory.

## Success criteria (set BEFORE running)
- **S1 (the read works):** e37a recovers ≥ 90 % of the gap at 16k (ceiling demonstrated: dense 100 %, LSA-native
  99 % at 30M).
- **S2 (no fade):** e37a's share at 64k ≥ its share at 16k − 5 points.
- **S3 (the reasoner adds):** e37b ≥ e37a + 3 points at 16k, or ≥ 0.01 nats better on far-copy tokens, at equal
  tokens.
- **S4 (no harm):** loss on positions < 512 at 16k within 0.01 nats of the window control; core benchmarks within
  1 point of dense on average.
- **S5 (cost):** tokens per second at 16k ≥ 0.8 × the LSA arm's.

## Kill criteria (set BEFORE running)
- **K1:** e37a < 60 % of the gap at 16k after the decay → rerun with summaries off (is the coarse path hurting?)
  before any further stage.
- **K2:** e37b ≤ e37a on both S3 measures → the reasoner is shelved for stage B.
- **K3 (health):** loss spikes > 0.5 that do not recover within 1k steps, or a trunk loss more than 0.1 above the
  window control at 25 % of the run → stop and debug.

## Plan
- **Stage 0 (laptop + one GPU, free):** unit tests (done, 9 + the E33p suites, 43 pass); tiny pipeline check of
  e37a / e37b / lsa through trunk and decay (done, 2026-11-07); GPU check: flex = reference read, forward and
  backward, at the tiny size and 4k / 16k at 100M; speed and peak memory against the LSA arm (S5's projection).
- **Stage 0 result (2026-11-07, Polonez GPU 1, RTX 3090):** 18 GPU tests pass (flex = the reference read, forward
  and backward, worst relative gradient difference 4e-4). Speed went through three passes (training tokens per
  second, from `infra/aws/jobs/speed_check.sh`; LSA arm in brackets):
  1. d106c24: e37a 3.8k [14.8k] at 100M / 4k, 1.9k [16.5k] at 30M / 16k — the score_mod's captured bias buffers
     carried gradients, accumulated with atomic adds into the same summary entries by every query and head;
  2. 4b1c908 (bias buffers detached): 12.3k at 100M / 4k, 12.0k at 30M / 16k; the reasoner arm 4.5k at 16k — the
     forward kernel still 4.4× LSA's, one score_mod over all keys;
  3. 63feb08 (two flex calls merged by log-sum-exp: raw tokens without score_mod, summaries alone): **e37a 12.6k
     (0.85 ×) and e37b 11.5k (0.78 ×) at 100M / 4k; at 30M / 16k e37a 13.1k (0.79 ×), e37b 11.3k (0.69 ×) [16.5k]**;
     peak memory within 1.5 GB of the LSA arm.
  S5 (≥ 0.8 × LSA) is met by e37a at 100M and missed narrowly by e37b; the remaining gap is the summary call's
  score_mod, the per-pass clustering and the merge. Further kernel work waits for the read to prove itself.
- **30M home screen (2026-11-07, while the AWS node runs E33p stage 1 and no H100 is free):** E33p's exact
  stage-1 protocol (1.5B FineWeb-Edu at 4k, 0.2B LongDocs-lite at 16k, 64 rows of 4k per step, seed 42), so the
  runs compare directly with E33p's six seed-42 controls (`infra/aws/jobs/e37_home.sh`):
  - e37a on Polonez GPUs 0+1 (from 10:21 UTC);
  - e37b on Odra GPUs 0+1 (from 11:51 UTC);
  - e37a_c128 on Polonez GPU 2 (from 11:53 UTC);
  - e37a_c32 started on Odra GPU 2 at 11:53 UTC but was stopped at 12:12 UTC, because with three cards busy the
    card reached 80 °C (the author's heat rule for Odra while Polonez trains). It is queued on Polonez GPUs 0+1 for
    when e37a finishes.
  
  At the controls' 30M scale the share of the gap at 16k is 99 % for LSA-native, 34 % for pages_open, 31 % for
  notes_open and 4 % for e33. Three GPUs would break the 64-row step, so the reasoner arm uses two.
- **30M home screen result, e37a and e37b (2026-11-08; e37a decay done 23:25 UTC Nov 7, e37b 01:11 UTC Nov 8; RULER
  16k for e37a pending; the ladder pending).** Share of the window→dense gap, positions ≥ 2k (long-document eval, 8
  rows per source and length; LSA-native in brackets): e37a 96.0 / 100.0 / 98.7 / 86.4 % at 4k / 16k / 32k / 64k
  [96.5 / 99.2 / 100.2 / 127.0]; e37b 95.6 / 99.8 / 99.6 / 89.0 %. At 128k (gap 0.012, raw loss): window 2.278,
  dense 2.267, LSA 2.170, e37a 2.333, e37b 2.326 (2.7M scored tokens).
  - **S1 met** (100 % ≥ 90 %). **S2 missed** (86.4 % < 95 %; 128k worse than the window control).
  - **S3 missed, K2 met → the reasoner is shelved for stage B.** e37b − e37a at 16k: −0.2 points; on far-copy tokens
    (copy split at 16k, 4-gram test, 438k far tokens) −0.0045 nats (95 % document bootstrap −0.0073 … −0.0016,
    156 docs; 8-gram −0.0047), real but under the 0.01 bar, and +0.004 nats worse on never-seen tokens.
  - **S4 split:** core average 39.8 (e37a) / 38.7 (e37b) vs dense 38.8 (met); loss on positions < 512 at 16k
    2.6322 / 2.6311 vs window 2.6052 (+0.027, missed; dense +0.020, LSA +0.015 also miss it).
  - **Exact far copy works:** e37a recovers 99.8 % of the far-copy gap at 16k (window 2.098 → dense 0.673 nats on
    far tokens; e37a 0.676), 104.6 % on near-copy tokens, 99.2 % on never-seen tokens.
  - **RULER 4k / 16k: 0.000 on all 13 tasks — an eval bug, not a result.** The controls score (mean at 4k: LSA
    0.139, single-needle 0.97; pages_open 0.105; dense 0.091; window 0.029). Cause (2026-11-08, Odra GPU 0, e37b's
    checkpoint): the RULER generator lays a row out with document id −1 (left padding, and the decoding slots up
    to a 256 multiple); a padding query reads nothing in either flex call of the concept read, so the
    log-sum-exp merge gave 0/0 = NaN there, which reached the real tokens through 0 × NaN in the next layers'
    attention tiles. Training and the long-document eval have no padding, so their numbers stand. Fixed in
    `_attend_concepts_flex` (and the reference path): a padding query reads zero; test T7
    (`test_padding_slots_read_zero_and_leave_real_tokens_unchanged`). On the GPU with the fix: no NaN, loss on
    positions 2k–4k 2.7853 / 2.7865 / 2.7855 for the three padded layouts vs 2.7855 plain. RULER to be rerun.
  - **Summaries-off probe (eval only, same rows; 01:41–01:49 UTC Nov 8, Polonez GPU 0):** e37a with the concept
    summaries switched off recovers 86.5 % at 16k (vs 100 % with them) but 104.7 % at 64k (vs 86.4 %); at 128k
    (arxiv, doab, gutenberg; hansard has no 128k row and the probe stopped before usgpo) loss 2.930 vs 3.131 with
    summaries, LSA 2.947, window 3.031, dense 3.044. The summaries carry ~13 points at the training length, and
    their growing number (2,048 at 16k → 16,384 at 128k) breaks the read past it: the fade is the summaries'.
- **Stage A:** six arms at 100M with the screening protocol, on one 4 × L40S spot node (each arm on all four
  GPUs, ~8–10 h, ≈ $60 per arm; ≈ $360) or H100s when available. Gate: S1–S5, K1–K3.
- **Stage B (if S1 holds):** the full m100 protocol for e37 (a or b), lsa and dense (+ original LSA if its stack
  runs), ≈ $450.
- **Stage C:** long pages — concepts of concepts (additive statistics) over pages of 4k–16k tokens, two-level
  choice; train at 16k, read at 128k–1M.
- **Stage D:** data and protocol (synthetic retrieval / multi-hop share, length curriculum).
- **Launch:** `E33N_EXPERIMENT=E37 bash infra/aws/jobs/e33n_run.sh e37a m100` (the screening token budget via the
  run's `--tokens`), arms in `scripts/run_e33n.py`.
- **New foundation code:** `nn/concept_memory.py`; `par_global_read = "concept"` and the `concept_*` fields in
  `PerceiverARConfig`; trainer arguments; `FlowRecorder.reading_concept`; `tests/test_concept_memory.py`.

## Risks
- The summary entries can take most of the attention when queries are diffuse (calibrated, so they take what dense
  attention would give those tokens; but their values are means). Measured by the read's mass split; K1's fallback
  is summaries off.
- The hard membership carries no gradient: the clustering follows the keys, which learn through the read.
- The kernel is near-dense at 16k (as LSA's port and E33p's read): the summary columns add C / S ≈ 12 % to the
  attention. A gather kernel is an engineering step after the bet works.
- The block-causal reasoner is quadratic in concepts (1,920 at 16k is cheap; ~2M at 10M is not): stage C restricts
  it to each concept's retrieved neighbours.

## Result
- (filled in by experiment-track)
