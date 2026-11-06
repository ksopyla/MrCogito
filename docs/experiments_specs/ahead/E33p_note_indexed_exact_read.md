# E33p — note-indexed exact read: the notes point, the reading layer opens the raw tokens behind them

- **Status:** **approved 2026-11-06** ("Aligned, build it") on the alignment page
  [e33p_architecture.html](../../3_Evaluations_and_Baselines/e33p_architecture.html), with its "my reading" choices
  as shown there and the two open questions settled as recommended: (1) a chosen note whose member tokens all lie
  inside the query's window is skipped and the next-best note takes its place, so every one of the m places opens new
  tokens; (2) the opened set is a **hard cap of 512 tokens per query block** (best notes first; a note that would cross
  the cap is cut at the cap).
- **Serves:** vision priority 2 (long context through the notebook) on the way to priority 4 (agent-to-agent: the
  notes become the table of contents another agent can query, the raw pages what it asks for). Direct follow-up of
  E33n's finding.
- **Implementation plan:** E33p_plan.md *(after alignment, by `implementation-plan`)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-11-05 · closed —
- **Evidence behind it:** E33n run report
  [e33n_s30_lsa_testbed_20261105.md](../../2_Experiments_Registry/run_reports/e33n_s30_lsa_testbed_20261105.md) (the
  copy split); LSA (arXiv 2610.42147: cluster centres address, the read goes to the original keys); NSA (DeepSeek's
  native sparse attention: compressed + selected-block + window branches in one softmax); Landmark Attention (page
  weight × in-page weight); the "indexed notebook" proposal of 2026-11-04 (chat, recorded in cogito-mem).

## Hypothesis
Using the notebook as an **index** — each note's address key tied to the reading layer's own raw keys of the tokens it
summarises, and each block of queries opening the raw keys and values of the tokens behind its best notes — recovers
**≥ 40 %** of the window − dense far-context gap on natural long text at 30M (E33n's notes alone: 6 %), with a read
budget per query that does not grow with the document's length.

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (perceiver_ar, the E31c page writer, the closed note read in
  `attend_message`, E33n's no-gradient top-k threshold `note_topk_threshold`, the eval-only `raw` override that lets
  the reader see uncompressed prefix keys/values); the LSA testbed (`data/bin_token_dataset.py`, `scripts/run_e33n.py`,
  `infra/aws/jobs/e33n_run.sh`); evals `evaluation/long_doc_loss.py`, `evaluation/copy_split.py`,
  `evaluation/run_lsa_suite.py`. One trainer entrypoint; new arms are config.
- **Init / checkpoint:** random init, E33n's protocol unchanged (same data order and seed), so E33n's trained controls
  are the controls here.
- **Baseline to beat (E33n s30, 2026-11-05, 16k, positions ≥ 2k, 64 rows per source):** dense ceiling 2.4811 ·
  window control 2.6146 · full note read (E33) 2.6073 · top-64 note read (E33n) 2.6064. G = 0.1335. Far-copy tokens
  (9.6 %): window − dense 1.361 nats, notes close 1.6–1.8 % of it.
- **Materially new:** the reading layer gets **exact** access to past tokens, chosen by a learned index. The notes
  change role from the only copy of the content to address + gist. The address keys are tied to the reader's key space,
  so "the query matches the note" means "the query matches the note's tokens on average" (q·mean k = mean q·k). Every
  earlier notebook arm (E30, E31, E33, E33n) read compressed notes only; the `raw` override was an eval-only ceiling
  that opened everything.

## The architectural bet
The writer is unchanged: E31c, 32 notes per closed 256-token page, stride 192. Its slot competition gives each page
token an assignment over the page's notes (A[t, n], a softmax over notes — soft k-means). The reading layer (no RoPE,
one KV head) reads, per query i in query block b (128 queries):

1. **window:** the last 512 raw tokens (as in every arm);
2. **notes (gist):** every note of every closed page (E33's full read);
3. **opened tokens (new):** the block's best m notes are chosen by the block score
   s_b(n) = Σ_{i∈b} softmax_n(q_i · κ_n), computed without gradient. κ_n is the note's **tied address key**:
   κ_n = Σ_t A[t, n] k_t / Σ_t A[t, n], where k_t is the reading layer's own raw key of token t. The raw keys and
   values of the chosen notes' member tokens (t whose largest assignment is n) join the read. Tokens already inside
   i's window are not repeated, and only closed pages open (causality as in E33);
4. **one softmax** over window ‖ notes ‖ opened tokens. A note's key in that softmax is its address key κ_n, so the
   read trains the index (the assignment A and the raw key projection) through the gist entries. Values: the writer's
   note value for gist entries, the raw value for opened tokens.

- **Budget:** m = 64 notes per query block, about 8 member tokens each, so about 512 opened tokens, the same as the
  window. The read per query is about 512 + 512 + (number of notes), flat in length apart from the note count, which
  E33n's top-64 can cap later.
- **Arms (30M, E33n's s30 shape: width 576, 2 pre + reading + 5 stack layers, loop 4, FFN 1472):**
  - **notes-open (main):** opens the member tokens of the 64 best notes.
  - **pages-open (coarser):** opens the 2 best pages whole (page score = the sum of its notes' block scores), again
    512 tokens. This tells whether the finer grouping matters.
  - **LSA-native (published reference, added 2026-11-06 at the author's request):** the same 30M backbone with LSA's
    native read in the reading layer instead of the notebook: the reading layer's own keys of each closed key block
    clustered by Ward merging into buckets of about 8 (no parameters, no gradient), a query block scores one centre
    per bucket, opens the best buckets' original keys within the same 512-token cap, plus LSA's sinks, and (LSA's
    default two-path read) every unread bucket enters the softmax as one averaged term. Ported from LSA's Apache-2.0
    code with attribution. E33p vs LSA-native isolates the question "is our learned writer a better index than
    clustering the keys?".
  - **controls:** E33n's dense ceiling, window control and full note read — **retrained with the instrumentation
    below** so training dynamics are comparable (same seed and data order; their E33n numbers double as a
    reproducibility check of G).
- **Information budget:** in the read, one note is a single 64-number key and 64-number value (one key/value head),
  made from a 512-number writer state that summarises about 8 tokens; it cannot return which of 50,257 tokens came
  next. An opened token keeps its full per-token state; the 30M dense model shows that this size can copy
  when it can see the text (it closes all of G).

## Why this is not a safe retread
It is not another notebook-size or read-rule knob. It changes what the notebook is (an index into kept text instead
of the text's only copy), adds exact retrieval with a learned, tied index (LSA's mechanism, with our trained writer
as the clusterer instead of LSA's bottom-up merges), and is tested on the far-copy tokens that E33n identified as the
whole of the far-context value at this size.

## Training protocol
Identical to E33n s30: FineWeb-Edu GPT-2 1.5B tokens at 4k (constant LR 1e-3 after 1 % warmup), then LongDocs-lite
0.2B tokens at 16k (3 % warmup, 1 − sqrt decay to 0.1 ×, fresh optimizer); 0.26M tokens per step.

## Evaluation
E33n's suite on the same rows: long-document loss by position at 4k–128k (64 rows per source at 16k and 64k, with a
bootstrap over documents); the copy split (near / far / no copy); RULER per length (4k, 16k); core benchmarks (SDPA);
and two switches at eval inside each new model: notes off, and **opened tokens off** (the exact read's own value).

## Instrumentation (added 2026-11-06 at the author's request; identical in every arm)
Goal: compare training dynamics across arms — do the layers do similar work and get similar learning signals, how
information flows through the network, and whether the reading layers pick the same things.

**A. During training** (every 200 steps, computed inside that step on the GPU, one host sync when logged, ≤ 3 % of
average step time; W&B keys under `flow/`, `grad/`, `read/`, `select/`, `writer/`):
1. **Information flow per layer** (pre layers, reading layer, each stack layer and each loop round): residual-stream
   norm and the relative update of each sublayer, ‖Δh‖ / ‖h‖, for attention and feed-forward separately.
2. **Learning signal per layer:** gradient norm and update-to-weight ratio ‖ΔW‖ / ‖W‖ (after the optimizer step) for
   each layer's attention and feed-forward weights, the writer, the embeddings and the output layer.
3. **Where the reading layer reads:** attention mass and entropy per key source — window, notes, opened tokens,
   LSA's unread-bucket terms, dense raw keys — split by query position (0–512, 512–2k, 2k–8k, 8k–16k).
4. **Selection** (opening arms): share of query blocks that hit the 512-token cap, distinct pages opened per block,
   window-only notes skipped, and how much the opened set changes between loop rounds (overlap of round r with r − 1).
5. **Writer** (notebook arms): assignment entropy, note usage balance, address-key norm against raw-key norm.
6. **Logit lens:** the next-token loss read from each layer's residual through the final norm and output layer, on
   the last 1,024 positions of one logged row — where in the depth the prediction forms.

**B. Across arms, after training** (`evaluation/flow_probe.py`, the same 16 held-out 16k rows for every arm, at the
trunk's end and the decay's end, and on intermediate checkpoints kept every 1,000 steps):
1. **Do they pick the same things:** for each query at position ≥ 2k, the dense ceiling's reading-layer top-32 keys
   against the tokens each opening arm opened (recall of the dense choice, and the dense attention mass on the opened
   set); for the notebook arms, the same with note mass mapped to pages.
2. **Copy-source hit rate:** on far-copy tokens (the copy split), does the opened set contain the earlier occurrence
   that the copy needs?
3. **Do the layers behave alike:** linear CKA between every pair of arms' layer outputs on the same tokens, and
   within an arm across loop rounds.
4. **Per-layer loss (logit lens) and attention mass by distance** for the reading layer, on all probe rows.
Outputs: JSON per arm and checkpoint in the registry, plus a comparison script that draws the curves.

## Success criteria (set BEFORE running)
- **S1 (exact read carries far context):** the notes-open arm recovers ≥ 40 % of G at 16k on positions ≥ 2k
  (G = 0.1335 from E33n's controls; the ceiling is demonstrated: the 30M dense model closes 100 %).
- **S2 (it is the copying):** on far-copy tokens (4-gram test) it closes ≥ 30 % of their 1.361-nat gap (notes alone:
  2 %).
- **S3 (no dilution with length):** its share of G at 64k is no more than 5 points below its share at 16k.
- **S4 (no harm):** loss on positions < 512 at 16k within 0.01 nats of the window control; RULER at 4k not below the
  window control's.

## Kill criteria (set BEFORE running)
- **K1:** after the long decay both new arms recover < 15 % of G at 16k → stop E33p.
- **K2 (speed):** a training step of the notes-open arm is slower than 60 % of the E33 arm's throughput (26.7k tokens/s
  on one RTX 3090) after the kernel work in stage 0 → fix the kernel before stage 1.
- **K3 (health):** loss spikes > 0.5 that do not recover within 1k steps, or a trunk loss more than 0.1 above the window
  control at 25 % of the run → stop that run and debug.

## Plan
- **Stage 0 (free; Mac + one 3090):** correctness tests — the tied key equals the assignment-weighted mean of raw
  keys; membership and dedupe across overlapping pages (stride 192); causality (only closed pages, window tokens not
  repeated); with m ≥ every note, the opened read equals the `raw` override restricted to closed pages; gradient
  reaches A and the key projection, none through the selection. Then a speed and memory check at 4k and 16k (K2).
- **Stage 1:** six arms at 30M with the instrumentation — notes-open, pages-open, LSA-native, and the three retrained
  controls — about 15–20 RTX 3090 hours each (about 100 in all: two days on the home servers, or about 36 H100 hours
  ≈ $250 on AWS; the author picks), then E33n's evals and the cross-arm probe. Gate: S1–S4, K1.
- **Stage 2 (only if S1 and S2 hold, and with the author's approval):** 100M on an AWS H100 against new 100M controls,
  within E33n's unspent stage-2 budget (≤ $850).
- **Launch:** `E33N_DATA=… E33N_RUNS=… E33N_MICRO_TRUNK=4 E33N_MICRO_DECAY=1 CUDA_VISIBLE_DEVICES=<gpu> bash
  infra/aws/jobs/e33n_run.sh notes_open s30` (and `pages_open`), new arms in `scripts/run_e33n.py`.
- **New foundation code:** `lm_open` (`none` | `notes` | `pages`), `lm_open_k`, `lm_note_key` (`writer` | `tied`),
  and the eval switch for the opened branch, in the closed-read path of `nn/perceiver_ar_lm.py` (a flex `mask_mod`
  over raw tokens with a per-query-block selection table; block-sparse at 128-token blocks); tests in
  `tests/test_note_open.py`.

## Risks
- One selection per 128-query block is coarser than LSA's per-query choice; per-query selection is a later option if
  the block choice misses.
- The member rule (largest assignment) is a hard cut: no gradient flows through membership itself; values still flow
  through the opened raw tokens, and keys through κ.
- The opened read needs the raw keys and values of closed pages in memory (the dense ceiling's cost at 16k); the
  re-run-the-local-writer idea that avoids storing them is not part of E33p.
- Overlapping pages (stride 192 for 256-token pages) can make two notes claim one token; opened tokens are deduped by
  position.

## Result
- (filled in by experiment-track)
