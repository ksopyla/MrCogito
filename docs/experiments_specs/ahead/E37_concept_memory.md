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
  - **RULER rerun with the fix (e7a0b18), e37b (Odra GPUs 0+1, 07:42–11:13 UTC Nov 8; 16k tasks split over the two
    cards, merged; 100 documents per task):** mean of the 13 tasks 0.113 at 4k and 0.103 at 16k. At 4k: LSA 0.139,
    pages_open 0.105, dense 0.091, notes_open 0.047, window 0.029, e33 0.024 (the controls ran RULER at 4k only;
    e33 at 16k 0.016). Single needle (niah_single_1) 0.57 at 4k and 0.49 at 16k (LSA 0.97, pages_open 0.78, dense
    0.68 at 4k); multi-value 0.155 / 0.185 (LSA 0.103). The old zero result is kept as `eval_ruler_padnan.json`.
  - **RULER rerun with the fix, e37a (Polonez GPU 2, 12:53–17:39 UTC Nov 8; 100 documents per task):** mean of
    the 13 tasks **0.158 at 4k and 0.151 at 16k**, the best of all 30M models at 4k (LSA 0.139, e37b 0.113,
    pages_open 0.105, dense 0.091) and level from 4k to 16k (e37b 0.113 → 0.103, e33 0.024 → 0.016). Single needle
    0.94 / 0.90 (LSA 0.97 at 4k), multi-key-1 0.31 / 0.41, multi-value 0.19 / 0.21; the variable-tracking,
    word-frequency and multi-key-2/3 tasks stay near zero, as for LSA and dense at 4k. The reasoner (e37b) does not
    help RULER at 30M: −0.045 at 4k, −0.048 at 16k.
  - **Summaries-off probe (eval only, same rows; 01:41–01:49 UTC Nov 8, Polonez GPU 0):** e37a with the concept
    summaries switched off recovers 86.5 % at 16k (vs 100 % with them) but 104.7 % at 64k (vs 86.4 %); at 128k
    (arxiv, doab, gutenberg; hansard has no 128k row and the probe stopped before usgpo) loss 2.930 vs 3.131 with
    summaries, LSA 2.947, window 3.031, dense 3.044. The summaries carry ~13 points at the training length, and
    their growing number (2,048 at 16k → 16,384 at 128k) breaks the read past it: the fade is the summaries'.
- **Compression ladder, e37a_c128 (4 concepts of 128 keys per block, same 512 opened tokens; Polonez GPU 2, decay
  done 09:10 UTC Nov 8, long-document eval 09:10–09:15 UTC; RULER and core then run with the padding fix):** share
  of the window→dense gap 85.8 / 80.9 / 71.5 / 65.0 % at 4k / 16k / 32k / 64k (e37a, 8 keys per concept: 96.0 /
  100.0 / 98.7 / 86.4); 128k raw loss 2.286 (e37a 2.333, window 2.278, dense 2.267, LSA 2.170).
  - Coarse concepts cost ~19 points at the training length: four broad concepts choose a less precise 512-token open,
    and their summaries are coarser.
  - The fade remains (−16 points from 16k to 64k; e37a −14), although c128 has ~512 summaries at 64k, fewer than
    e37a's ~2,048 at 16k, where e37a is at 100 %. So the number of summaries alone does not explain the fade.
    Hypothesis: the cause is the total weight the far field claims (each summary claims its members' estimated
    mass, and that total grows with length whatever the grouping). Tested by the fade probe below.
  - RULER (13 tasks, mean, padding fix): 0.110 at 4k, 0.056 at 16k (e37a 0.158 at 4k). Core (lm-eval, accuracy):
    within ±0.02 of e37a except sciq 0.592 (e37a 0.626).
- **Compression ladder, e37a_c32 (16 concepts of 32 keys per block; trained on Polonez GPUs 0+1 after the Odra
  heat stop, decay done 15:21 UTC Nov 8, evals 15:21–15:45 UTC):** share of the window→dense gap 95.0 / 94.2 /
  88.1 / 55.2 % at 4k / 16k / 32k / 64k (e37a 96.0 / 100.0 / 98.7 / 86.4; c128 85.8 / 80.9 / 71.5 / 65.0); 128k raw
  loss 2.417 (e37a 2.333, c128 2.286). RULER 4k mean 0.105 (e37a 0.158, c128 0.110). Core within ±0.02 of e37a
  except sciq 0.500 (e37a 0.626).
  - The ladder at the training length: 8 keys per concept 100 %, 32 keys 94 %, 128 keys 81 %. Each 4× coarser
    grouping costs more than the one before; the open of 512 tokens chosen by coarser concepts is less precise.
  - As trained, c32 fades the most past 16k (55 % at 64k): its 16 summaries per block each claim 32 keys' mass.
    With the length scale it recovers to 111 % at 64k and 2.160 at 128k (below).
- **Fade probe (eval only, no retraining; e37b's checkpoint, Odra GPUs 0+1, 11:14–11:52 UTC Nov 8; same rows as
  its long-document eval, 8 per source; shares paired by source: 5 sources at 16k / 64k, 4 at 128k).** The flex
  summary call patched in memory: **cap K** keeps each query and head's K strongest summaries (by the summary logit);
  **scale** multiplies the summaries' total weight by ρ = min(1, 15,360 / N), N the keys the summaries stand for
  (15,360 = 30 blocks at the end of a 16k row), so nothing changes up to 16k. Self-test: with the probe off the
  hidden states equal the original read's (difference 0), and the 16k loss equals the official eval's (2.3492).

  | variant | 16k share | 64k share | 128k loss |
  |---|---|---|---|
  | e37b as trained | 99.8 % | 89.0 % | 2.326 |
  | cap 2,048 (keeps 45 % of the summaries at 64k, 24 % at 128k) | 99.8 % | 95.1 % | 2.288 |
  | cap 512 (46 % at 16k, 12 % at 64k, 6 % at 128k) | 99.6 % | 121.8 % | 2.204 |
  | **scale** | 99.8 % (unchanged) | **127.8 %** | **2.155** |
  | LSA-native / dense / window | 99.2 / 100 / 0 % | 127.0 / 100 / 0 % | 2.170 / 2.267 / 2.278 |

  - **The fade is the summaries' growing total weight:** holding it at the training length's removes the fade and
    puts e37b level with LSA-native at 64k and 0.015 nats better at 128k. Capping the number of summaries helps only
    as far as it removes weight.
  - Only 512 summaries per query and head are needed at 16k (cap 512: −0.2 points).
  - Caveats: one checkpoint, chosen on the same rows it is scored on, 32–40 rows per length. To confirm: e37a,
    e37a_c128 and e37a_c32 (not used to choose), more rows, and RULER past 16k (the scale acts only there).
  - **Built into the read (2026-11-08):** `concept_sum_len` in `PerceiverARConfig` and the trainer (0 = off, the
    default, so old checkpoints read as trained; set it to the training length, 16,384 here), applied at the merge
    of the two flex calls and in the reference path; it counts only the blocks wholly inside the query's document.
    Eval switch: `--concept_sum_len 16384` in `evaluation/long_doc_loss.py` and `evaluation/run_lsa_suite.py`.
    Tests T8 (values, and no change up to the length) and the GPU flex = reference case `e37a_sum_len` (10 GPU
    tests pass, Odra GPU 0, 12:37 UTC).
  - **Confirmed with the built-in scale (`--concept_sum_len 16384`, official rows; e37b Odra GPU 1 12:35–12:52 UTC,
    e37a and e37a_c128 Polonez GPU 2 12:36–12:53 UTC):** share of the gap at 16k / 32k / 64k, as trained → with
    the scale, and the 128k loss:
    - e37a (not used to choose): 100.0 → 100.0 / 98.7 → 100.9 / **86.4 → 129.0 %**; 128k **2.333 → 2.143**;
    - e37b (reproduces the probe): 99.8 → 99.8 / 99.6 → 101.2 / 89.0 → 127.9 %; 128k 2.326 → 2.154;
    - e37a_c128: 80.9 → 80.9 / 71.5 → 72.1 / 65.0 → 88.0 %; 128k 2.286 → 2.180;
    - e37a_c32 (not used to choose; Polonez GPU 0 15:37–15:45 UTC): 94.2 → 94.2 / 88.1 → 91.0 / **55.2 → 111.2 %**;
      128k **2.417 → 2.160**;
    - LSA-native: 99.2 / 100.2 / 127.0 %; 128k 2.170 (window 2.278, dense 2.267).
    With the scale, e37a passes S2 (129 % at 64k ≥ 100 % − 5) and is ahead of LSA-native at 64k and 128k. The
    scale is an eval-time rule found after the runs, not part of the trained read S2 was set for: it is the default
    for the next stage (set `concept_sum_len` to the training length), where S2 is tested as designed.
    The scale helps all four concept models past 16k and changes nothing at or below it; the coarser the
    grouping, the more it recovers.
  - **RULER at 32k, e37b with and without the scale (five needle tasks, 50 examples each; Odra GPUs 0+1,
    after 12:52, done 16:56 (on) and 17:03 UTC (off), Nov 8):** mean 0.168 off → **0.227 on**; the same five tasks score 0.257 at 4k and 0.249 at
    16k. Per task, off → on: single_1 0.48 → 0.44, single_2 0.02 → 0.16, single_3 0.10 → 0.14, multikey_1
    0.14 → 0.26, multivalue 0.100 → 0.135. The scale keeps most of the 16k retrieval at 2× the training length;
    without it a third is lost. No control has RULER at 32k yet (stage A scores 16k).
- **Stage A:** six arms at 100M with the screening protocol, on one 4 × L40S spot node (each arm on all four
  GPUs, ~8–10 h, ≈ $60 per arm; ≈ $360) or H100s when available. Gate: S1–S5, K1–K3.
  - **Launch (2026-11-08, author's go-ahead for ≈ $360):** one g6e.12xlarge spot node, `infra/aws/jobs/e37_multi.sh
    42 e37a lsa dense window notes_open e37b` (most important first), hard lifetime 52 h (≤ ≈ $400 at ~$7.6/h).
    Spend before it: ≈ $320 of $2,000. The concept arms train with the length scale on (`concept_sum_len` 16,384,
    set by `scripts/run_e33n.py`; inert at ≤ 16k) and are also scored with it off. Evals per arm over all four
    GPUs (`infra/aws/jobs/e37_evals.sh`: RULER 16k split in three task groups; at 30M the node's evals took 2.3 h
    with mostly one GPU busy). Estimate per arm at 100M: trunk ~4–5 h, decay ~1.2 h, evals ~1.75 h.
  - Eval-script trial before the node's first evals (Odra, `E37_EVAL_QUICK=1`, e37b's 30M checkpoint): the RULER task
    groups were named `GROUPS`, which bash reserves (the user's group ids; assignments are ignored), so the 16k
    lanes ran group ids as task names. Renamed (3833ca1, pulled on the node about 17:10 UTC, before its evals; only
    that file changed). The quick check's long length moved from 2k to 8k (07ca3a5; RULER's hotpot QA generator
    never fits its documents below ~4k). Trial passed 17:33–17:40 UTC: 13 tasks at both lengths, merge, marker.
  - **e37a at 100M (trunk 13:42–17:14, decay 17:14–18:20, evals 18:20–19:58 UTC Nov 8; 1,907 trunk steps of 128
    rows, last train loss 3.33; the 30M e37a ended at 3.35 after 5,722 steps of 64 rows and 1.5B tokens):**
    long-document raw loss on positions ≥ 2k (5 sources; 4 at 128k), trained with the length scale, scored with
    it on / off: 16k 2.334 / 2.334, 32k 2.442 / 2.451, 64k **2.417 / 2.490**, 128k **2.148 / 2.321** (30M e37a with
    the scale: 2.349, 2.453, 2.426, 2.143). The scale trained in works as at eval time. RULER mean of 13 tasks
    **0.066 at 4k, 0.055 at 16k** (30M e37a 0.158 / 0.151; single needle 0.27 / 0.38 vs 0.94 / 0.90). Core within
    ±0.02 of the 30M e37a except arc_easy 0.400 (0.384) and sciq 0.589 (0.626).
    - At the screening budget the 100M model is barely better than the 30M one (0.015 nats at 16k) and much
      worse at RULER: 1.0B tokens in 1,907 steps is fewer tokens and a third of the 30M screen's steps. The 100M
      controls (same budget) decide whether this is the budget or the read; shares of the gap wait for window
      and dense.
  - **lsa (our port) at 100M (trunk 19:59–23:08, decay 23:08–23:57 UTC Nov 8, evals 23:57–01:02 UTC Nov 9; last
    train loss 3.33, as e37a):** long-document raw loss on positions ≥ 2k (same 5 sources; 4 at 128k): 16k 2.332,
    32k 2.437, 64k 2.399, 128k 2.130 — below e37a at every length by 0.002 / 0.005 / 0.018 / 0.018 nats (30M LSA:
    2.350, 2.455, 2.429, 2.170). RULER mean of 13 tasks **0.104 at 4k, 0.092 at 16k** (30M LSA 0.139 at 4k);
    single needle 0.70 / 0.73 (30M LSA 0.97), multi-key-1 0.16 / 0.10, multi-value 0.04 / 0.04. Core mean of the
    7 accuracy tasks 0.391 (e37a 0.397): arc_challenge 0.251, arc_easy 0.395, hellaswag 0.281, lambada 0.157, piqa
    0.578, sciq 0.575, winogrande 0.503.
    - The budget explains part of e37a's RULER drop: LSA also falls at 100M under the screening budget (0.139 →
      0.104 at 4k, single needle 0.97 → 0.70). But the concept read falls more (0.158 → 0.066; single needle
      0.94 → 0.27) and now trails LSA by 0.038 / 0.037, the reverse of 30M (+0.019 at 4k). On loss the two are
      within 0.02 nats, LSA ahead past 32k; e37a keeps a slight core edge (+0.006). One guess, untested: at a
      third of the 30M screen's optimizer steps, the concept read's grouping (Sinkhorn assignments) has not yet
      learned to route single needles. Window and dense decide the gap shares.
  - **dense, window, notes_open at 100M (evals done 04:45, 08:40 and 15:42 UTC Nov 9; read from the node over ssh,
    the S3 login having expired):** raw loss on positions ≥ 2k, same 5 sources (4 at 128k), and the share of the
    window → dense gap:

    | length | e37a | lsa | dense | window | notes_open |
    |---|---|---|---|---|---|
    | 4k | 2.493 (94.7 %) | 2.485 (100.1 %) | 2.485 | 2.629 | 2.532 (67.2 %) |
    | 16k | 2.334 (**99.1 %**) | 2.332 (99.9 %) | 2.332 | 2.523 | 2.411 (58.7 %) |
    | 32k | 2.442 (99.4 %) | 2.437 (102.2 %) | 2.441 | 2.646 | 2.544 (49.8 %) |
    | 64k | 2.417 (**113.7 %**) | 2.399 (127.1 %) | 2.436 | 2.574 | 2.503 (52.0 %) |
    | 128k | 2.148 | 2.130 | 2.212 | 2.257 | 2.237 |

    Dense fades past its 16k training length (gap 0.191 at 16k, 0.045 at 128k), so the shares past 32k are
    inflated; at 128k e37a and LSA beat dense by 0.06–0.08 nats. RULER mean of 13 tasks at 4k / 16k (single
    needle): dense **0.192 / 0.138** (0.87 / 0.79), notes_open 0.116 / 0.072 (0.64 / 0.63), lsa 0.104 / 0.092
    (0.70 / 0.73), e37a 0.066 / 0.055 (0.27 / 0.38), window 0.023 / 0.006 (0.12 / 0.04). Share of the window →
    dense RULER gap at 4k / 16k: notes_open 55 % / 50 %, lsa 48 % / 65 %, e37a **25 % / 37 %**. Core means 0.391–0.401
    for all five (a tie). Training samples per second at 16k (decay) / 4k (trunk), node of 4 × L40S: dense
    5.71 / 24.8, window 5.60 / 23.5, lsa 4.23 / 21.5, e37a 3.09 / 19.3, notes_open 2.99 / 19.6.
  - **Stage-A gates for e37a (e37b's S3 / K2 wait for its evals):** **S1 met** (99.1 % ≥ 90 %); **S2 met** with the
    length scale (113.7 % at 64k ≥ 94.1 %); **S4 split** — core 39.7 vs dense 39.4 (met), loss on positions < 512 at
    16k 2.602 vs window 2.572 (+0.030, missed; dense +0.022, lsa +0.018, notes_open +0.016 miss it too); **S5 missed
    at 16k** (0.73 × LSA's samples per second; 0.89 × at 4k); K1 not hit; last trunk loss 3.33 as LSA (K3 clear).
  - **The budget does not explain the RULER drop.** From 30M to 100M at the screening budget, dense rose on RULER
    at 4k (0.091 → 0.192) and so did notes_open (0.047 → 0.116), while the two reads that pick a few blocks fell
    (LSA 0.139 → 0.104, e37a 0.158 → 0.066). The loss on long text says the concept read works as well as LSA and
    dense (S1, S2); retrieval of a single exact needle says its top-k routing is the weak part at 100M. Next
    measurement (cheap, eval only, on a saved checkpoint): the flow probe — does e37a's opened set contain the
    needle's block at 100M (it did at 30M, single needle 0.94)?
    - **Correction (2026-11-10, stage B):** at the full budget e37a's single needle is back to 0.96 / 0.96 and its
      RULER mean is 0.185 / 0.136, so the screen's drop was the budget for the read that picks blocks (a third of
      the optimizer steps), not a defect of the routing; dense needed fewer steps for it. The full-budget controls
      decide the comparison.
- **Stage B (if S1 holds):** the full m100 protocol for e37 (a or b), lsa and dense (+ original LSA if its stack
  runs), ≈ $450.
  - **Go-ahead (author, 2026-11-09 ~20:30 UTC):** all the needed models at 100M on the full token budget (3.0B trunk +
    0.4B decay = 3.4B tokens, ~21 per parameter of the 162M models; stage A gave 1.2B, ~7 per parameter), the original
    LSA and our two best, for generation tests and evaluation with LSA's protocol; up to $3,000; H100 and L40S in
    parallel; first results on Nov 10; act without asking. Spend before it ≈ $570.
  - **Machines** (quotas, us-east-1: G spot 64 vCPU = one g6e.12xlarge; P on demand 64 vCPU = four one-H100
    p5.4xlarge; P spot 8, unusable; other regions ≤ 8; a G-spot increase to 96 is pending since Nov 8):
    one 4 × L40S spot node `e37-b-l40s` (`infra/aws/jobs/e37_full.sh 42 e37a lsa`, launched when the stage-A node
    ends, 32 h cap ≈ $240); one H100 each for notes_open, dense and window (`e37_full.sh 42 ARM`, 40–44 h caps,
    ≈ $300 each), retrying every 5 min (no p5.4xlarge capacity in any zone at 20:31 UTC); the fourth H100 for the
    original LSA (their trainer, ~160M preset, our shards and budget; smoke test first). Ours: e37a (lead) and
    notes_open (best E33-family read at 100M on RULER, 0.116 at 4k); e37b is out (no gain at 30M).
  - **Evals:** e37_evals.sh on each node (long-document loss 4k–128k, RULER 13 tasks at 4k and 16k, core). LSA's
    protocol adds RULER at 8k, 32k and 64k and held-out FineWeb-Edu loss at 4k: a separate pass on the saved
    checkpoints after training (RULER past 16k costs hours per model with uncached decoding).
  - **No H100 at all (20:31 UTC Nov 9 → 12:30 UTC Nov 10, every 5 min, every zone):** the original LSA moved to the
    L40S slot (its node job runs their stack on sm89 with their Triton kernels, `infra/aws/jobs/lsa_orig_node.sh`);
    `e37-b-l40s` was terminated at 12:49 UTC right after e37a's evals (lsa had trained 5 minutes; its S3
    checkpoints resume later). Slot order: e37a → original LSA → lsa → the parent session's catalogue arm
    (catalogue_reuse, branch e37-catalogue, its own budget). Quota requests pending: G spot 96 (us-east-1, us-east-2,
    us-west-2), P on demand 64 (us-west-2), 32 (us-east-2).
  - **e37a, full budget (trunk 22:28–09:00, decay 09:00–11:06, evals 11:06–12:43 UTC Nov 9–10; 4 × L40S spot,
    ≈ $105):** raw loss on positions ≥ 2k, 5 sources (4 at 128k): 4k 2.320, 16k **2.166**, 32k 2.254, 64k 2.250,
    128k **1.990** (screen 2.493, 2.334, 2.442, 2.417, 2.148: −0.16 to −0.19 nats at every length); with the length
    scale off 2.166 / 2.317 / 2.156 at 16k / 64k / 128k (the scale still carries 0.07–0.17 nats past 16k). Loss on
    positions < 512 at 16k 2.388 (screen 2.602). RULER mean of 13 tasks **0.185 at 4k, 0.136 at 16k** (screen 0.066 /
    0.055; screening dense 0.192 / 0.138); single needle **0.96 / 0.96** (screen 0.27 / 0.38), single-2 0.58 / 0.14,
    multi-key-1 0.29 / 0.26, multi-value 0.12 / 0.15; variable tracking, word frequency and multi-key-2/3 stay near
    zero. Core mean of 7 accuracy tasks 0.416 (screen 0.397): arc_challenge 0.247, arc_easy 0.422, hellaswag 0.303,
    lambada 0.201, piqa 0.599, sciq 0.633, winogrande 0.510. Shares of the gap wait for the full-budget window and
    dense.
  - **Original LSA, full budget (launched 14:01:16 UTC Nov 10; node `lsa-orig-l40s` i-00da766f90443a5a1, 4 × L40S
    spot, launched 13:02:55 UTC, 24 h cap):** their code (github.com/Prefrontal-AI/latent-sparse-attention, pinned at
    5ebea732d639576d2a7579b0de40e6252f9001f5) in its own venv from their uv.lock (torch 2.13.0+cu129, torchtitan
    @56a721b, flash-linear-attention 0.5.2, causal-conv1d 1.7.0 built on the node); torchtitan's Trainer through their
    `trainer_config`, nothing in their checkout edited (`scripts/lsa_orig.py`, `infra/aws/jobs/lsa_orig_node.sh` →
    `lsa_orig_run.sh`). **Model:** their `qwen35_0_8b_gpt2/pretrain_fineweb_lsa.yaml` (lsa_mqa8, FineWeb trunk) with
    the Qwen3.5 body shrunk to our size: hidden 896, 12 layers (9 Gated DeltaNet + LSA at layers 3, 7, 11: their 3 : 1),
    Gated DeltaNet 7 key / 7 value heads of 128, SwiGLU 2,240, tied GPT-2 embeddings (50,304), their GPT-2 inits;
    attention layer exactly theirs (8 query heads over 1 KV head, d 128, no RoPE, output gate, QK-norm, `kind: lsa`,
    `coarse: false`, `fast_read: true`, window 512, block 512, pool 4, buckets 8 / cap 16, 4 sinks, top_k 1024, max
    over query heads); **162,632,318 parameters** (45.07M embedding, 117.56M other). **Data and budget:** our m100
    rows (data/bin_token_dataset.py, the 32 FineWeb-Edu shards; LongDocs-lite) in our HF trainer's order
    (`torch.randperm(N, seed 42)`, checked against `Trainer.get_train_dataloader`; the same 128 × 4k rows every step
    on 1 or 4 GPUs: the first 30 trunk losses repeat the smoke's to 4 decimals); positions restart at every document
    (torchtitan then resets the Gated DeltaNet there), labels across a document start ignored; trunk 5,722 steps
    (3.0B), warmup 57, constant; decay 763 steps of 32 × 16k from the trunk's weights (fresh optimizer), warmup 22,
    1 − sqrt over 740 to 0.1 ×; AdamW (0.9, 0.95), wd 0.1 on matrices (their scope), clip 1.0, fp32 weights under
    bf16 autocast, peak **6.4e-4** (ours, so the LR is no difference between arms; inside their 0.55B sweep 5e-4 /
    1e-3; their 5e-4 × 1,024 / 896 = 5.7e-4); their FSDP over the 4 GPUs, micro-batch 4 × 4k / 1 × 16k.
    **Deviations, all forced:** their fused LSA read has no document mask, so in the packed 4k trunk the 3 LSA layers
    read across documents (as in their own FineWeb trunk); on the L40S their LSA layer runs their Triton read
    (gather_sparse) and, because lsa_select's tiles need 105–168 KiB of shared memory and the L40S has 99 KiB
    (measured), their reference selection (their CUDA kernels are sm_90 only); evals in their venv with lm-eval 0.4.11
    (ours) put ahead of their 0.4.13; rows padded with <|endoftext|> to a multiple of 512 (causal, never read back),
    and RULER generation without left padding (their read has no padding mask). **Smoke (200 + 100 steps, 13:17–14:00
    UTC):** trunk loss 11.01 → 4.99, decay 5.55 → 3.73, no NaN, 18.5 GiB a GPU; **75.3k tokens/s at 4k and 71.7k at
    16k for the node** (18.8k / 17.9k a GPU); every eval ran end to end (long-document loss at 4k and 128k, val,
    core, RULER 4k / 16k; batched generation = one by one on the real model). **Evals on the node:** our models' set
    (`evaluation/lsa_orig_eval.py`, which runs `long_doc_loss.py`'s scoring and `run_lsa_suite.py`'s lm-eval protocol
    on their model): long-document loss 4k–128k (8 rows a source, plus `open_off` = their LSA layers at top_k 0),
    FineWeb-Edu val at 4k, core, RULER 13 tasks at 4k and 16k; 8k / 32k / 64k in the later pass for every model.
    **Gate passed:** 12.6 h of training + ~1 h of evals → expected finish **≈ 03:45–04:15 UTC Nov 11** (trunk ≈
    01:05, decay ≈ 02:40), ≈ $105 at $7.24 / h. Results: `s3://…/runs/e37/lsa_orig_m100_tok3.0e9_seed42/`
    (eval_longdoc.json, eval_val.json, eval_core.json, eval_ruler.json; checkpoints every 500 / 200 steps, a relaunch
    resumes). No H100 came free (20:44 UTC Nov 9 → 12:30 UTC Nov 10, every 5 min).
  - **Original LSA, result (trunk 14:01–01:00, decay 01:00–02:32, evals 02:32–03:17 UTC Nov 10–11; node gone 03:18,
    ≈ 14.3 h ≈ $107; job exit 0):** same size (162.6M), data, token budget and node evals as full e37a. Raw loss on
    positions ≥ 2k, 5 sources (4 at 128k): 4k 2.284, 16k **2.135**, 32k 2.215, 64k 2.201, 128k **1.925** — 0.03–0.07
    nats below e37a at every length (2.320 / 2.166 / 2.254 / 2.250 / 1.990); positions < 512 at 16k 2.369 (e37a 2.388).
    Their read off (`open_off`, LSA layers at top_k 0, Gated DeltaNet unchanged): 2.432 / 2.334 / 2.386 / 2.079 at
    4k / 16k / 64k / 128k, so the sparse read carries 0.15–0.20 nats. RULER mean of 13 tasks **0.139 at 4k, 0.082 at
    16k** (e37a 0.185 / 0.136); single needle (repeated-sentence haystack) **0.10 / 0.04** (e37a 0.96 / 0.96), but
    single-2 (essay haystack, number) 0.63 / 0.46 (e37a 0.58 / 0.14) and single-3 0.19 / 0.06 (e37a 0.13 / 0.03);
    multi-key-1 0.30 / 0.27, multi-value 0.16 / 0.04, common words 0.14 / 0.02, the rest near zero as for e37a. Core
    mean 0.420 (e37a 0.416, a tie): arc_challenge 0.240, arc_easy 0.415, hellaswag 0.309, lambada 0.202, piqa 0.590,
    sciq 0.665, winogrande 0.522. FineWeb-Edu val at 4k 3.162 after the decay (3.093 after the trunk: the long-document
    decay costs 0.07 nats on short web text). **Reading:** the original LSA predicts long text a little better, e37a
    retrieves a planted fact far better; their single-needle collapse on the repeated-sentence haystack (while the
    essay haystack works) is unexplained — the eval saved no generations, so the next check is a few RULER samples.
    Their body is 9 Gated DeltaNet (recurrent) + 3 LSA layers against our local-attention layers around one looped
    reading layer, so the loss gap is not the read alone; the lsa arm (their read in our body) separates the two.
  - **lsa (our port: LSA's top-k read in our body), full budget (trunk 03:40–13:05, decay 13:06–14:46, evals
    14:47–15:52 UTC Nov 11; node `e37-b-l40s-lsa` i-0e44d4a99d1d8f0ae, 4 × L40S spot, self-shutdown 15:54, ≈ 12.3 h ≈
    $90; job exit 0):** training loss at the end of the trunk 3.095 (e37a 3.092; trunk eval 3.121), of the decay 2.333
    (e37a 2.330; decay eval 2.255 vs 2.253). Raw loss on positions ≥ 2k, 5 sources (4 at 128k): 4k 2.322, 16k
    **2.169**, 32k 2.254, 64k 2.245, 128k **1.994** — a tie with e37a (2.320 / 2.166 / 2.254 / 2.250 / 1.990, within
    0.005 everywhere), and 0.03–0.07 nats above the original LSA (2.284 / 2.135 / 2.215 / 2.201 / 1.925); positions
    < 512 at 16k 2.384 (e37a 2.388). Its read off (`open_off`): 2.505 / 2.389 / 2.492 / 2.435 / 2.126, so the top-k
    read carries 0.13–0.24 nats (the original's 0.15–0.22). RULER mean of 13 tasks **0.255 at 4k, 0.209 at 16k** — the
    best of the three (e37a 0.185 / 0.136, original LSA 0.139 / 0.082; screen lsa 0.104 / 0.092). Single needle
    0.96 / 0.95 (e37a 0.96 / 0.96, original 0.10 / 0.04); the gain is on the essay haystacks: single-2 (number)
    **0.88 / 0.84** (e37a 0.58 / 0.14, original 0.63 / 0.46), single-3 (UUID) **0.55 / 0.27** (e37a 0.13 / 0.03,
    original 0.19 / 0.06), and word frequency 0.27 / 0.09 (e37a 0.01 / 0.06); multi-key-1 0.26 / 0.27, multi-value
    0.12 / 0.14, multi-query 0.05 / 0.04, QA 0.10 / 0.09 (HotpotQA) and 0.03 / 0.01 (SQuAD), common words 0.06 / 0.01,
    variable tracking and multi-key-2/3 near zero as for the others. Core mean 0.416 (e37a 0.416, original 0.420, a
    tie): arc_challenge 0.266, arc_easy 0.413, hellaswag 0.305, lambada 0.212, piqa 0.600, sciq 0.596, winogrande
    0.522. No notebook in this arm (`has_notebook: false`), so no notes-off file. W&B: trunk and decay runs synced
    from the node's offline logs (15:57 UTC). **Reading:** with the body held fixed, LSA's exact top-k read and our
    notebook predict long text equally well, so the original LSA's 0.03–0.07-nat edge comes from its Gated DeltaNet
    body, not its read. On retrieval the top-k read is clearly ahead of the notebook: the same single-needle score and
    far better on the essay haystack with numbers and UUIDs at 16k (0.84 vs 0.14; 0.27 vs 0.03), i.e. the exact read
    finds facts the notebook's summaries blur. And the original LSA's single-needle collapse is not the read itself
    (0.96 in our body): its body or its code path. Next: the notebook plus an exact top-k read (the notes_open arm
    asks a nearby question), and dense / window at the full budget for the shares of the gap.
  - **LSA protocol, later pass at home (author's go-ahead 19:50 UTC Nov 11): RULER at 8k and the validation loss.**
    Checkpoints fetched from S3 onto the servers by one-hour presigned URLs (sizes checked); the original LSA in its
    own venv on Odra (their pinned repo and uv.lock, torch 2.13 cu129; causal-conv1d left out, no CUDA toolkit there;
    lsa_select's reference selection, as on the L40S). RULER 8k, 13 tasks × 100 documents, one lm-eval call per task
    (`infra/aws/jobs/e37_ruler_later.sh`): e37a Polonez GPUs 0–1 19:55–21:03, lsa Polonez GPUs 2–3 then 2 alone
    (Polonez's 3-card cap) 20:02–21:20, original LSA Odra GPUs 0 + 2 20:15–20:59 UTC; 3090 time a task ≈ 575 s
    (e37a), 380 s (lsa), ~200 s a GPU (original). **RULER mean 4k / 8k / 16k: lsa 0.255 / 0.241 / 0.209, e37a 0.185 /
    0.187 / 0.136, original LSA 0.139 / 0.119 / 0.082** — the order holds at every length. At 8k: single needle 0.96 /
    0.95 / 0.09 (lsa / e37a / original); single-2 0.92 / 0.60 / 0.68; single-3 0.53 / 0.24 / 0.10; multi-key-1 0.22
    for all three; multi-value 0.17 / 0.15 / 0.12; word frequency 0.12 / 0.02 / 0.00; the rest near zero. e37a's
    essay-number needle holds to 8k (0.58 → 0.60) and falls by 16k (0.14), while the top-k read keeps it (0.88 →
    0.92 → 0.84): the notebook's summaries lose a planted number between 8k and 16k. **Validation loss**
    (`evaluation/val_loss.py` for ours, `lsa_orig_eval.py val` for the original; the m100 trainer's eval rows,
    1,047,441 FineWeb-Edu tokens at 4k and 1,310,640 LongDocs-lite tokens at 16k, the same rows for all three):
    FineWeb-Edu at 4k after the trunk **3.093 original / 3.105 e37a / 3.111 lsa**, after the decay 3.162 / 3.187 /
    3.193; LongDocs-lite at 16k after the decay 2.193 / 2.228 / 2.232 — the original LSA 0.01–0.04 nats lower
    everywhere, our two within 0.006 of each other (the body, not the read, as in the long-document loss). The
    trainer's own last evals came before the end of each phase (trunk at step 5,500 of 5,722: lsa 3.121; decay at
    600 of 763: 2.255), so they sit above these final-weight numbers. Results in S3 next to each run's node evals
    (`eval_ruler_later.json`, `later/`, `eval_val.json`). RULER 32k / 64k not run: the home cards belong to the E33t
    capability checks until ≈ Nov 14; on AWS (one-L40S spot nodes) ≈ 15 GPU-h for 32k and ≈ 30 for 64k, waiting for
    the author.
  - **notes_open (E33p: the closed notebook with tied note keys and an exact top-64 open read), full budget (trunk
    06:39–16:58, decay 16:58–19:18, evals 19:18–21:43 UTC Nov 12; node `e37-b-l40s-notes-open` i-0146e9d2c4732ba1a,
    4 × L40S spot, us-east-1, self-shutdown 21:45, ≈ 15.1 h ≈ $110; job exit 0; W&B online, both runs finished):** the
    trunk is a tie with e37a (trainer eval 3.119 / 3.119 at step 5,500, training loss 3.092 / 3.092), the decay is
    not (trainer eval at step 600 2.348 vs 2.253). Raw loss on positions ≥ 2k, 5 sources (4 at 128k): 4k 2.379, 16k
    **2.282**, 32k 2.393, 64k 2.367, 128k **2.104** — 0.06–0.14 nats above e37a (2.320 / 2.166 / 2.254 / 2.250 /
    1.990) and the lsa port; the full budget gained 0.13–0.15 nats over its screen (2.532 / 2.411 / 2.544 / 2.503 /
    2.237), as the other arms did, so the gap to e37a is the read, not the budget. Its notebook carries 0.10–0.12 nats
    to 32k (notes off 2.497 / 2.388 / 2.488), 0.065 at 64k and 0.016 at 128k; the open read alone 0.07–0.10 (open
    off 2.476 / 2.364 / 2.466 / 2.434 / 2.179), the gist 0.015–0.04. Positions < 512 at 16k 2.370 (e37a 2.388, the
    lowest of all: no harm near the front). RULER mean of 13 tasks **0.066 at 4k, 0.018 at 16k** — below its own
    screen (0.116 / 0.072) and far below e37a (0.185 / 0.136) and lsa (0.255 / 0.209); single needle 0.18 / 0.04
    (screen 0.64 / 0.63), single-2 0.17 / 0.01, multi-key-1 0.10 / 0.01, the rest near the floor. Core mean 0.411
    (e37a 0.416, a tie): arc_challenge 0.252, arc_easy 0.420, hellaswag 0.304, lambada 0.193, piqa 0.598, sciq 0.611,
    winogrande 0.502. **Reading:** the closed notebook with 64 opened notes uses 16k of text worse than the concept
    read or the top-k read, and with three times the tokens it retrieves less, not more; it is not in the later pass,
    and the E33-family read stays behind e37a at 100M.
  - **LSA protocol, later pass on AWS (author 2026-11-12: "Yes, 32k and 64k"): RULER at 32k and 64k** for e37a, lsa
    and the original LSA; for the parent session's catalogue model (catalogue_reuse_n4, branch e37-catalogue-later)
    also RULER 8k and the validation loss. Six one-L40S nodes (g6e.xlarge spot first; g6e.2xlarge spot and g6e.xlarge
    on demand in us-east-2 as fallbacks), `infra/aws/jobs/e37_ruler_later_node.sh`: one lm-eval call per (length,
    task), pair files to S3 every 10 min, a relaunch skips finished pairs, a DONE marker per node; 06:16 UTC Nov 12 →
    02:27 UTC Nov 13 (e37a 32k done 20:37, e37a 64k 22:50, catalogue 64k 22:37, lsa 01:31, original 14:17,
    catalogue val + 8k + 32k 02:27). Uncached decoding at 64k, measured per document: e37a 140 s, lsa 88 s,
    catalogue 175 s, original 24.6 s, so **64k uses the first 25 documents of each task** (all 100 × 13 would be
    30–60 h a model; ± 0.1 on a task near 0.5), 8k and 32k all 100 (`limit_per_length` in each merged file). Three
    spot interruptions (original 13:42, e37a 64k and catalogue 17:26 UTC), relaunched 13:54, 18:36 (g6e.2xlarge) and
    19:03 (us-west-2), only the pair in work lost; ≈ 90 node-hours ≈ $120–150. **RULER mean of 13 tasks, 4k / 8k /
    16k / 32k / 64k: lsa 0.255 / 0.241 / 0.209 / 0.174 / 0.151, catalogue 0.231 / 0.197 / 0.183 / 0.125 / 0.101,
    e37a 0.185 / 0.187 / 0.136 / 0.106 / 0.088, original LSA 0.139 / 0.119 / 0.082 / 0.040 / 0.028** — the same order
    at every length. Single needle at 32k / 64k: e37a 0.96 / 0.92, lsa 0.96 / 0.96, catalogue 0.98 / 0.96, original
    0.03 / 0.04. Essay haystack, number (single-2): lsa **0.55 / 0.28**, original 0.11 / 0.04, catalogue 0.08 / 0.04,
    e37a 0.06 / 0.00; UUID (single-3): lsa 0.20 / 0.04, catalogue 0.07 / 0.00, original 0.04 / 0.00, e37a 0.01 /
    0.00. Multi-key-1: lsa 0.31 / 0.20, catalogue 0.19 / 0.12, original 0.16 / 0.08, e37a 0.14 / 0.04; multi-value:
    catalogue 0.15 / 0.06, lsa 0.14 / 0.15, e37a 0.09 / 0.02, original 0.01 / 0.00; HotpotQA at 64k 0.16–0.20 for all
    but the catalogue (0.08); the rest near zero. Catalogue at 8k: single needle 0.98, single-2 0.53, single-3 0.32,
    multi-query 0.18, multi-value 0.18. **Catalogue validation loss:** FineWeb-Edu at 4k after the trunk 3.112,
    after the decay 3.200; LongDocs-lite at 16k after the decay 2.250 (e37a 3.105 / 3.187 / 2.228, lsa 3.111 / 3.193 /
    2.232, original 3.093 / 3.162 / 2.193). **Reading:** past 16k every read loses the essay needles first; LSA's
    exact top-k read keeps the most (the number needle 0.55 at 32k against ≤ 0.11 for the others), the planted
    sentence holds near 0.96 to 64k for all three of our reads and never for the original LSA (its body or code
    path, not the read). The catalogue read sits between lsa and e37a on retrieval at every length and 0.02 nats
    above e37a on long-text loss (positions ≥ 2k 0.02–0.07, LongDocs val 0.022).
  - **dense (ceiling: the reading layer attends to every earlier token, no position code), full budget (trunk
    22:14–06:22, decay 06:23–07:39, evals 07:39–08:03 UTC Nov 12–13; node `e37-b-l40s-dense` i-092a42761dc19cc7c,
    4 × L40S spot, us-east-1d, launched 22:12 by the queue when the notes_open node freed the G-spot slot,
    self-shutdown 08:04, ≈ 9.9 h ≈ $72; job exit 0; W&B online, both runs finished):** trainer evals a tie with e37a
    (trunk 3.119 / 3.119, decay at step 600 2.252 / 2.253). Raw loss on positions ≥ 2k, 5 sources (4 at 128k): 4k
    2.319, 16k **2.163**, 32k 2.256, 64k 2.332, 128k **2.194** (e37a 2.320 / 2.166 / 2.254 / 2.250 / 1.990, lsa 2.322
    / 2.169 / 2.254 / 2.245 / 1.994): a tie to 32k (twice its 16k training length), then worse (+0.08 at 64k, +0.20
    at 128k) — full attention without positions spreads thin where the block reads stay put. Positions < 512
    at 16k 2.394 (e37a 2.388). RULER mean of 13 tasks **0.210 at 4k, 0.175 at 16k** — between e37a (0.185 / 0.136)
    and the catalogue / lsa (0.231 / 0.183, 0.255 / 0.209); single needle 0.77 / 0.71 (e37a 0.96 / 0.96), single-2
    0.88 / 0.61, single-3 0.09 / 0.14, multi-key-1 0.26 / 0.35, multi-query 0.12 / 0.17, multi-value 0.18 / 0.19,
    common words 0.17 / 0.03, word frequency 0.12 / 0.02. Core mean 0.422 (e37a 0.416, original 0.420, a tie):
    arc_challenge 0.259, arc_easy 0.416, hellaswag 0.304, lambada 0.197, piqa 0.596, sciq 0.655, winogrande 0.528.
    **Reading:** at the full budget the notebook (e37a) and the top-k read (lsa) reach the dense ceiling on long-text
    loss to 32k and beat it past 32k; on retrieval the top-k read and the catalogue beat it at 4k and 16k, and e37a is
    below it on the mean but above it on the planted sentence. The shares of the gap wait for the full-budget window
    (queued on the next free L40S slot).
  - **window (floor: the reading layer sees only its block-aligned 512-token window, no long-range read), full
    budget (trunk 08:34–17:11, decay 17:12–18:29, evals 18:29–18:56 UTC Nov 13; node `e37-b-l40s-window`
    i-09c4804bef64d838d, 4 × L40S spot, us-east-1, launched 08:31, self-shutdown 19:00, ≈ 10.5 h ≈ $77; job exit 0):**
    raw loss on positions ≥ 2k, 5 sources (4 at 128k): 4k 2.461, 16k **2.351**, 32k 2.449, 64k 2.397, 128k 2.086.
    **The full-budget window → dense gap: 0.142 / 0.187 / 0.193 nats at 4k / 16k / 32k** (screen 0.144 / 0.191 /
    0.205: the budget lowered both ends by about the same 0.17–0.20, so the long-range value of the context did not grow).
    Positions < 512 at 16k 2.367, the lowest with the original LSA (2.369) and notes_open (2.370). RULER mean of 13
    tasks 0.047 at 4k, 0.011 at 16k (single needle 0.13 / 0.04: the floor; only what falls in the last 512 tokens).
    Core mean 0.421 (dense 0.422): arc_challenge 0.255, arc_easy 0.420, hellaswag 0.304, lambada 0.212, piqa 0.614,
    sciq 0.615, winogrande 0.530.
  - **Shares of the full-budget window → dense gap (positions ≥ 2k, the 5 sources every arm has; script
    `full_gap.py` in the session scratchpad, the same pairing as the screen's):**

    | arm | 4k | 16k | 32k | gain over window at 64k / 128k (nats) |
    |---|---|---|---|---|
    | e37a (concept read) | 99 % | **99 %** | 101 % | 0.147 / 0.096 |
    | lsa (LSA's top-k read, our port) | 98 % | 97 % | 101 % | 0.153 / 0.092 |
    | catalogue_reuse_n4 | 84 % | 86 % | 90 % | 0.127 / 0.027 |
    | notes_open (E33p) | 58 % | 37 % | 29 % | 0.030 / −0.018 |
    | original LSA (their body) | 125 % | 115 % | 121 % | 0.197 / 0.161 |
    | dense | 100 % | 100 % | 100 % | 0.065 / −0.108 |

    Past 32k dense is no ceiling (it trained at 16k without a position code and falls below the window model at
    128k), so 64k and 128k are given as the gain over the window model. The original LSA passes 100 % because its
    Gated DeltaNet body is better everywhere (validation loss 0.01–0.04 nats lower; trunk FineWeb-Edu 3.093 vs 3.105),
    not only on far context.
  - **Criteria at the full budget:** **S1 met** — e37a 99 % of the gap at 16k (≥ 90 %), with lsa 97 %; the
    catalogue model 86 % (below S1's bar), notes_open 37 % (inside K1's < 60 % zone for an E33-family read). **S2:**
    as a share, undefined past 32k (no ceiling); in nats e37a's gain over the window model is 0.185 / 0.195 / 0.147 /
    0.096 at 16k / 32k / 64k / 128k — it holds to 32k and shrinks past it (the lsa port the same: 0.181 / 0.195 /
    0.153 / 0.092), while dense loses all of it by 128k. **S4 missed on the short positions, met on core:**
    positions < 512 at 16k e37a 2.388 vs window 2.367 (+0.021; lsa +0.017, dense +0.027, catalogue +0.033, notes_open
    +0.003, original +0.002); core mean e37a 0.416 vs dense 0.422 (−0.6 points, within 1), lsa and catalogue 0.416,
    notes_open 0.411 (−1.1, just outside). **Reading:** at 100M on 3.4B tokens our concept read and the top-k read
    use far context as well as full attention to 32k and keep a gain to 128k where full attention loses it; both pay
    ~0.02 nats on the first 512 positions that the window model and the original LSA do not.
  - **Checks before any claim (author 2026-11-13 ~20:50 UTC, relayed by the "100M model review and next steps"
    session: "Yes, start checks"; AWS spend before them ≈ $1,265 to mid-Nov 13 by Cost Explorer, before credits;
    budget $2,000):**
    1. **Fair full-attention control, `dense_rope`:** the dense arm with RoPE on the reading layer (par_global_nope
       false, rope_theta 500k as the local layers; arch-10m 5f9cb36; commands differ from dense's only in that flag,
       same 96.5M non-embedding parameters). Every read here, dense included, has no position code on the reading
       layer, which is why dense spreads thin past 32k. Full m100 protocol, seed 42, 4 × L40S spot, the stage-B
       evals; also the long-document loss at an eval-time NTK-style base on every layer (4.28e6 = 500k × 8^(64/62),
       16k → 128k), reported as a separate row. ≈ $75.
    2. **Second seeds** for e37a and the lsa port (seed 7: initial weights and row order; the data are the same),
       full m100 protocol on 4 × L40S, the same evals. ≈ $105 each.
    3. **Needles with cached decoding** (branch e37-needles = e37-long + arch-10m; `infra/aws/jobs/e37_needles_node.sh`
       on one-L40S spot nodes): RULER at 64k on all 100 documents per task and at 128k, plus RULER 4k cached to set
       against the stage-B old-loop 4k, for e37a, lsa, catalogue, dense, dense_rope; the original LSA at 64k and
       128k on its own uncached path. Pairs in each run's `needles/` folder, merged into `eval_ruler_needles.json`
       (the old-loop 64k@25 pairs stay in `later/`). The essay-haystack tasks (single-2 number, single-3 UUID) are
       reported apart from the filler-text ones. ≈ $110.
    Order on the one 4-GPU slot (G spot quota): the 64k stage (catalogue, then the LSA port) → dense_rope → e37a seed 7
    → lsa seed 7; the needle nodes run beside it.
  - **64k stage at 100M (the "10M context architecture analysis" session's request, author: "Yes, run it"; recipe
    of e33n-lsa-testbed a729ca7, branch e37-long = it + the jobs + cached decoding): 0.1B LongDocs-lite tokens in
    rows of 65,536 from the stage-B decay final, 191 steps of 8 rows (micro 1 × accum 2 × 4 GPUs), lr 6.4e-5 →
    6.4e-6, gradient checkpointing; one g6e.12xlarge spot node per model, 8 h hard stop; files with a `_long` suffix
    and `long_evals/` in the run folder.** RULER with the cached decoding; the old loop on the same model at 4k as a
    check (`eval_ruler_long.json["cache_check"]`).
    - **catalogue_reuse_n4 (node `e37-long-catalogue-reuse-n4` i-07e15db9d75d8c8ac, launched 19:12, training
      19:16–20:54, tests 20:54–21:37 UTC Nov 13, ≈ 2.4 h ≈ $18; ~22 GB of 46 GB a card, ~32 s a step):** loss on
      positions ≥ 2k, stage B → 64k stage: 4k 2.342 → 2.339, 16k 2.189 → 2.187, 32k 2.275 → 2.264, 64k 2.270 →
      **2.232**, 128k 2.058 → **1.943** (4 sources; e37a 1.990, lsa 1.994, original LSA 1.925 at stage B); notebook
      off unchanged (−0.002 to −0.005), so the gain is the catalogue read's. Positions < 512 at 16k 2.400 → 2.399;
      core 0.416 → 0.416. RULER mean of 13 tasks 4k / 8k / 16k / 32k: 0.231 / 0.197 / 0.183 / 0.125 → 0.224 / 0.191
      / 0.180 / **0.158**; essay haystack at 32k number 0.08 → 0.19, UUID 0.07 → 0.21; single needle 0.99 at every
      length. Cache check at 4k: 10 of 13 tasks identical, three differ by one document (multi-key-2 0.01 vs 0.00,
      multi-value 0.18 vs 0.1825, single-2 0.73 vs 0.72), mean 0.2240 cached vs 0.2226 old loop — bf16 ties, not a
      different decoding; the 4k drop against stage B (0.231) is the 64k stage's, not the cache's.
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
