# E39 — Living concepts: dynamic spans and document-wide concepts in the reading layer (150M)

- **Status:** approved for build, pre-flight checks and training (author, 2026-11-22: "I like the dynamic span
  grouping; it needs to be fast on contextualised token vectors. At the same time, please remember our 10M context
  goal and reasoning in latent space; the looping seems helpful" · "lets start with the same training protocol as
  other 150M" · "Start Stages 1 and 2 together, I've raised the budget to $5000"). Spec written 2026-11-22
  19:22 UTC. Nothing is built yet.
- **Changed by the author, 2026-11-22 (before any run):** the first version of this spec put a 30M screen on the
  home servers before the paid run (kill mark K1). Asked whether to keep it, the author answered "Skip it". The
  screen is withdrawn; an early reading inside the paid run (K2a) takes over part of its job.
- **Serves:** the 10M-context goal (author, 2026-11-20: full attention is the ceiling up to 64k, the target is the
  lengths it cannot reach) and Vision priority 1 (reasoning in latent space: concepts that exist for the whole
  document are what a later reasoner would work on).
- **Implementation plan:** `E39_plan.md` (to be written by the build session before code, `implementation-plan`).
- **Owner / dates:** Krzysztof Sopyla · opened 2026-11-22 · closed —
- **New family, not a flavour of E37:** the author asked to "treat it as a new architecture". E39 was free on every
  branch and worktree on 2026-11-22 (checked `origin/dev`, `arch-10m`, `eng-100m`, `e37-sft`, `e33t-capability`,
  the card branches).
- **Evidence behind it** (all measured on the four ~150M models of E38, seed 42, same data and recipe; card
  `docs/release/hf_card_concept_memory_150m/EVALUATION.md`):
  - **Where concept memory loses.** On RULER it ties the topic index (our port of Latent Sparse Attention) on single
    needles (1.00 / 1.00 at 4k–16k) and loses when one answer needs several far items: all values of one key
    0.34 against 0.64 at 8k and 0.24 against 0.39 at 16k; several keys at once 0.32 against 0.69 and 0.11 against
    0.43; the most frequent words 0.16 against 0.42 and 0.02 against 0.37. Mean of 13 tasks 0.439 against 0.540 at 8k.
  - **On plain long text the two tie** (64k spans: 2.293 against 2.299 nats per token), so the grouping is not
    losing information in bulk; it loses when several exact items are needed together.
  - **The reading layer works as a sharp pointer.** In the trained weights of the catalogue and full-attention
    models (concept memory's own weights were not read) its largest possible attention logit is 23–24, against
    16–18 for the local layers and 8 untrained, so it is built to copy a few places, not to average many.
  - **What the two groupings do differently.** Concept memory groups each 512-token block by content into 64 equal
    groups, token by token, so the tokens of one multi-token fact can land in different groups and a one-off fact
    shares a group with about 7 unrelated tokens. The topic index pools 4 adjacent tokens before it merges, so a
    fact's neighbours stay together; it has no summaries. Whether this is what costs concept memory the multi-item
    tasks is **a hypothesis**, tested by pre-flight check P2 before any paid run.
  - **30M ladder (E37):** concepts of 32 keys kept 94 % of the window→dense gap at 16k and concepts of 128 keys
    81 % (8 keys: 100 %); there the coarser concept was also the unit that is opened, so the open got less precise
    too. E39 keeps the opened unit at about 8 tokens and makes only the summaries coarser. At 150M, switching the
    summaries off costs 0.023 nats at 16k and 0.068 at 64k: most of the gain is the word-for-word reads.
  - **Two-level search lost once:** at 30M the catalogue's page-first search recovered 3.2 points less of the
    window→dense gap than its flat search at 16k and 15.7 points less at 64k (E33t stage A). Those pages were
    positional (256 adjacent tokens); a page key says little about its content.
  - **The block-bound concept reasoner gave no gain** at 30M (E37b: −0.2 points at 16k). It is not in this bet.

## Hypothesis
If the reading layer groups its memory the way text is written — **contiguous spans** cut where the meaning shifts,
then **document-wide concepts** that each span joins or founds — instead of equal-size content groups inside fixed
512-token blocks, then a 150M model on the unchanged recipe (a) closes at least half of concept memory's gap to the
topic index on the retrieval tasks that need several far items, (b) keeps concept memory's long-document loss up to
64k and loses less past it, and (c) gives each reader a memory whose number of entries grows more slowly than the
document — because a fact's tokens are opened together, a one-off fact gets a group of its own, and everything said
about one thing across the document is reachable through one entry.

## Builds-on
- **Foundation (reused unchanged):** the single entrypoint `scripts/run_e33n.py` (size `m120t`), `PerceiverARLM`
  (`nn/perceiver_ar_lm.py`: shared 384-wide table and gated input map, 3 local layers, reading layer + 1 local layer
  looped 4 times with shared weights, 8 more local layers, tied factorised head), the exact-read machinery
  (`nn/exact_read.py`: `GroupIndex`, `groups_from_spans`, `choose`, `OpenTable`, `ChoiceList`, `open_term`), and the
  concept read's statistics, summary logit, length scale and two-call flex merge (`nn/concept_memory.py`:
  `summary_logit`, `summary_log_scale`, `attend_concepts`, `_attend_concepts_flex`).
- **Init / checkpoint:** random init, seed 42, as the other four models. No warm start: the keys must grow up with
  the new grouping.
- **Baseline to beat:** concept memory at 150M, `Meridian21Lab/concept-memory-140m-11.5bt` (arm `e37a`, run
  `e33n_m120t_e37a_muon0.01_tok8.0e9_seed42`, S3 `runs/e38/`). **Measured ceiling for retrieval:** the topic index,
  `Meridian21Lab/topic-index-140m-11.5bt` (arm `lsa`), same data, seed, budget and layers. Full attention
  (`full-attention-140m-11.5bt`) is the reference inside 64k, not an opponent.
- **Materially new:**
  1. the groups are **contiguous and of variable length** (1–16 tokens), cut by one backward-looking comparison per
     token; no k-means, no balance constraint, no fixed number of groups per block;
  2. **concepts live for the whole document** and are formed online (a span joins or founds), with a causal
     version of every concept per 512-token stride; there is no bank and no fixed count;
  3. a **two-level choice over content clusters** (concepts, then their spans), chosen by match and not by size;
  4. a summary stands for **what was not opened** of its concept (remaining count), so a large concept can be opened
     in part without losing or double-counting the rest.
  Zero new weights: the model has the same 141,506,208 parameters as concept memory.
- **Not new, kept on purpose:** the 512-token window rule (own stride + the one before, raw), 4 sinks, 512 opened
  tokens per reader and pass, one softmax over sinks ‖ window ‖ opened ‖ summaries, the second-order summary logit,
  the length scale, the loop of 4 passes, the U-net skips.

## The architectural bet
One coherent change: **how the reading layer's memory is grouped and searched.** Everything below happens inside the
reading layer (layer 3), without gradient through the grouping; the statistics carry gradient to the member tokens'
keys and values exactly as today.

Shapes are given for a 16k row (S = 16,384, d = 64, 14 query heads, 1 key/value head).

### 1. Spans (replaces the per-block balanced k-means)
- **Signal:** the reading layer's keys of the **first loop pass** (after layers 0–2, the pass marker, `wk` and the
  key norm), as unit vectors k̂ [B, S, 64]. Sinks (the first 4 tokens of a document) and padding are never members.
- **Cut rule:** a new span starts at token t when cos(k̂_t, k̂_{t−1}) < θ, or the document changes, or a 512-token
  stride starts. One shifted dot product for the whole row.
- **Bounds, by construction:** a run longer than 16 tokens is split every 16; a stride with more than 96 natural
  cuts keeps its 96 lowest-cosine ones. So every span has 1–16 tokens, never crosses a stride or a document, and a
  stride holds at most 128 spans (32 forced + 96 natural).
- **Only tokens that can be read from memory form spans.** A token is read as memory by readers of its own
  document two strides later, so in each stride only the last document can matter, and only if it runs on that
  far; shorter documents form no spans (rows of thousands of tiny documents exist in the data). A new document
  restarts the concept slots.
- **θ is held by a slow controller during training** so that the average span is **8 tokens** (the granularity of
  concept memory's 8 keys per concept): after each training step θ moves a small step against the error of the
  observed cut rate (target 1 cut in 8 tokens), from counters summed over all cards. θ is a saved buffer and is
  **frozen at test time**, where the number of spans then follows the text: more in dense passages, fewer in
  repetitive ones. Without the controller the granularity would follow the keys' drifting geometry.
- **Span statistics** (per pass, from that pass's keys and values, additive): n, Σk, Σ|k|², Σv, and the unit
  centre û = normalise(Σk̂).
- **Shapes:** 32 strides × 128 slots = 4,096 span slots, about 2,048 used. The open table T is
  [B, 16,384, 4,096] int16 (134 MB per row; 2.1 GB at 64k, the catalogue model's size).
- **Cost:** replaces ~400 small kernels per call × 4 calls per forward by a handful of kernels once per forward.

### 2. Living concepts (document-wide, formed once per forward)
- **Rule:** strides are processed in reading order. Every span of stride s is compared with the concepts of its own
  document as they stood after stride s − 1 (cosine of unit centres). It **joins** the best one if the cosine is at
  least λ, otherwise it **founds** a concept. Spans of the same stride that would found near-duplicates are merged
  by three parallel rounds of the leader rule inside the stride (earliest unresolved span leads; spans within λ of a
  leader follow it). A concept's centre is the normalised sum of its members' unit keys, so it moves as spans join.
- **λ is held by the same kind of controller** so that on average **1 span in 8 founds** a concept during training
  (about 64 tokens per concept at the training lengths); saved, frozen at test time, where the growth of the
  concept count with length is then a property of the text (readout R1).
- **Capacity:** C_max = S / 16 concept slots per row (1,024 at 16k), allocated in order; when full, spans join their
  nearest concept whatever the cosine. Concepts never mix documents.
- **Formed once, used by all four passes:** the span cuts and the span→concept assignment come from the first pass's
  keys; each pass recomputes the statistics from its own keys and values with the fixed assignment (scatter-adds).
- **Causal versions:** a reader in stride τ sees its own stride and the previous one as raw tokens, and the memory as
  it stood after stride τ − 2. So each concept has one **version per stride**: cumulative sums over the strides,
  [B, 32, 1,024, 64] for Σk and Σv (2.1M numbers each at 16k; 134 MB each at 64k), with gradient. A reader never sees
  a statistic that contains a token it must not see, and cached decoding gives the same numbers as a full forward.
- **Cost:** one loop over the strides (32 at 16k, 128 at 64k) of about 30 small kernels each, once per forward.

### 3. The read (per reader, per pass; no gradient in the choice)
1. **Concepts first:** score every visible concept by the estimated attention weight of an average member,
   max over the 14 query heads of q·μ_k/√d + |q|²σ²/(2d) (today's summary logit **without** log n, so size does
   not win), and keep the best **32**.
2. **Then their spans:** rank the visible spans of those 32 concepts by the same score on the span's own
   statistics, best first, and open them until **512 tokens** are held (`choose`, up to 256 spans, the last one cut
   short). A test-time switch lets all visible spans compete (flat search), to measure what the first level costs.
3. **One softmax** over: the document's 4 sinks ‖ the raw window ‖ the opened tokens ‖ **one summary per visible
   concept**, with key μ_k and value μ_v of its version and logit bias log(n − n_opened) + |q|²σ²/(2d) + the length
   scale. A concept opened whole drops out; a concept opened in part keeps a summary for the rest (same mean
   assumed). Every past token of the document is counted once: in the window, opened, or inside one summary.
- **Shapes:** summary entries 32 versions × 1,024 slots = 32,768 keys, of which a reader sees one version (about
  256 non-empty at 16k; concept memory shows 1,920); the remaining-count table is [B, 16,384, 1,024] int16.
- **Flex path:** unchanged structure — raw call (window, sinks, opened) and summary call merged by their
  log-sum-exp; the summary call's block mask selects the reader's version.

### 4. What stays out
No reasoner over the concepts (stage 3 of the design, a later spec), no concept tree (stage 4), no learned writer,
no index loss, no change to the loop, the skips or any other layer.

### Configuration (one new value of `par_global_read`; no fork)
`par_global_read = "living"` with `living_stride 512`, `living_span_max 16`, `living_span_cap 128`,
`living_span_target 8`, `living_found_target 0.125`, `living_keep 32`, `living_levels 2` (1 = spans only: every span
is its own concept, flat search by the same match score, one summary per unopened span), `living_search
two_level | flat` (test switch), `living_score match | mass` (`mass` = the existing read's rule, the estimated
attention mass with log n; a test switch, and what the regression test against the existing read uses),
`living_servo on | off`, plus the existing `concept_sinks 4`, `concept_top_k 512`, `concept_variance`,
`concept_summaries`, `concept_sum_len`. Arms in `scripts/run_e33n.py`: `living` and `living_spans`.

## Why this is not a safe retread
- Not a retune of concept memory: the grouping principle changes from "equal content clusters per block" to
  "contiguous spans + open-ended document-wide concepts", and the search from one level to two.
- Not the topic index: that has a fixed 64 buckets per block from Ward merges of 4-token pools, a flat cosine
  choice and no summaries.
- Not the catalogue's lost two-level search: the first level here is a content cluster, like the coarse cells of an
  inverted-file vector index (search a few cells, then their vectors), not a positional page.
- Analogies it stands on: text segmentation by dips in adjacent similarity (TextTiling) and H-Net's dynamic chunking,
  whose ablation found content-dependent boundaries better than fixed-width pooling
  (`docs/4_Research_Notes/hierarchy_lessons_hnet_ncm_mai_20261030.md`); one-pass leader clustering with additive
  cluster features (BIRCH), which is also the route to a concept tree; the fast-multipole split kept from E37
  (near field exact, far field by moments).

## Arms
| arm | what | where | role |
|---|---|---|---|
| `living` | spans + living concepts (the bet) | 150M on AWS | the model |
| `living_spans` | spans only (`living_levels 1`) | no run planned | first build milestone; kept as a config for attribution later |

Not in this spec (each ≈ one more 150M run, about $350): a 150M run of spans alone (only if the result needs
attribution), a second seed, the concept reasoner, the concept tree.

## Pre-flight checks (home GPU, $0, before any training)
On the released concept-memory model (the author's own weights, public Hub repository or S3 `runs/e38/`, about
0.6 GB), reading-layer keys and queries captured on held-out documents.
- **P1 — is there a span signal?** Distribution of the adjacent-key cosine at 16k on the seven long sources; θ for
  a mean span of 8; span-length histogram; mean cosine at lag 1 against lag 8. Starting value of θ.
- **P2 — is the diagnosis right?** On the RULER prompts concept memory fails at 8k and 16k (all values of one key,
  several keys at once), for the tokens of each expected answer in the haystack: in how many groups they sit; what
  share the last reader before the answer opens, in any pass; what share of that reader's attention goes to opened
  tokens, summaries and window. The same coverage with the E39 rules simulated on the same keys and queries (spans
  + match, flat and two-level). Three outcomes: answers are **not opened** (grouping or choice: E39's target);
  **opened but outweighed by summaries** (crowding: E39 helps only through fewer summaries; reported before
  training); **opened and attended, answer still wrong** (use, not memory: K0).
- **P3 — how do concepts grow?** The leader rule on 16k, 64k and 128k held-out documents for λ from 0.5 to 0.9:
  concepts against tokens, share of single-span concepts, the largest concept's share of tokens. Starting value of
  λ, and the reachable mark for readout R1.
- **P4 — speed and parity** of the built read on one home card: tokens per second at 4k and 16k against the
  concept-memory arm at the same size; the unit tests below.

## Success criteria (set BEFORE running; 150M, seed 42, the E38 tests unchanged)
Ceilings are the measured scores of the topic index (retrieval) and of concept memory (long text) on the same data.
Run-to-run noise of one RULER task at 100 prompts is ±0.09; of the mean of 13 about ±0.02.
- **S1 (several far items, the bet):** mean of four RULER cells — all values of one key and several keys at once,
  at 8k and 16k — **≥ 0.40**. Concept memory 0.25 (0.34, 0.24, 0.32, 0.11); the topic index 0.54 (0.64, 0.39,
  0.69, 0.43). 0.40 is half of the gap, three times the noise of a four-cell mean.
- **S2 (retrieval overall):** RULER mean of 13 **≥ 0.49 at 8k and ≥ 0.41 at 16k** (concept memory 0.439 / 0.373;
  topic index 0.540 / 0.439).
- **S3 (long text kept):** loss on the last 4,096 tokens of the held-out spans **≤ 2.136 at 16k and ≤ 2.313 at 64k**
  (concept memory 2.126 / 2.293; topic index 2.129 / 2.299; catalogue 2.147 / 2.333).
- **S4 (single lookups kept):** both single-number needles ≥ 0.95 at 4k, 8k, 16k and 32k (concept memory 0.98–1.00);
  one number among distractor keys at 16k ≥ 0.80 (0.91).
- **S5 (past the training length):** on the 15 held-out 128k documents, the whole 127k of context is at most
  **0.031** nats worse than the last 61k alone (concept memory +0.060; topic index +0.031; full attention +0.014).
- **S6 (general ability unchanged):** BLiMP ≥ 79.6 (80.56) and the 7-task mean ≥ 45.2 (46.2).
- **S7 (cost):** training speed at least 0.85 × concept memory's on the same node: ≥ 90k tokens per second at 4k,
  ≥ 59k at 16k, ≥ 18k at 64k (105.6k / 69.7k / 21.6k on 4 × L40S).

The bet **succeeds** with S1 and S3 and S4; S2, S5, S6, S7 are reported as met or missed. No "beats" claim follows
from any of them: inside 64k these are no-harm and gap-closing checks.

## Readouts that decide the next step (not pass marks)
- **R1 (toward 10M):** with frozen thresholds, concepts per document at 128k against 64k on the 15 held-out
  documents (linear growth = 2.0×). At most 1.6× → concepts alone carry the memory to long documents and the
  concept reasoner comes next; above 1.8× → the concept tree comes first. The mark is re-read after P3.
- **R2 (what the first level costs):** S1's cells and the 64k loss with the flat search switched on at test time.
- **R3 (what the summaries carry):** the 16k and 64k loss with summaries off (concept memory: +0.023 / +0.068).
- **R4:** the grouping's own numbers through training — mean span length, share of spans at 16 tokens and at 1,
  spans per stride, share of strides that hit the cap, concepts per document, share of single-span concepts, the
  largest concept's share, θ and λ, tokens opened per reader, share of attention on opened / summaries / window.

## Kill criteria (set BEFORE running)
- **K0 (pre-flight):** P1 shows no span signal (mean adjacent cosine less than 0.05 above the lag-8 cosine), or P2
  shows the failing answers are already opened and attended → stop before the build's training and report; the
  weak spot is not the grouping.
- **K1 — withdrawn** (the 30M screen at home; author, 2026-11-22: "Skip it"). The code is not reused.
- **K2a (150M, step 2,000 of stage 1, about $20 spent):** held-out loss more than **0.05** nats above concept
  memory's own run at the same step (read from its W&B run; the optimizer test before that run measured 2.283 at
  step 2,000) → stop the node and inspect. Added when the screen was withdrawn: it is the first evidence that the
  model learns with the new grouping.
- **K2 (150M, end of stage 1):** held-out loss above **2.155** (concept memory 2.134; the four models span
  2.134–2.145) → stop before stage 2 and inspect. Stage 1 is 53 % of the run's cost.
- **K3 (speed):** below 0.70 × concept memory's speed over the first 500 steps on the node (< 74k tokens per
  second at 4k) → stop and fix; do not pay for a slow run.
- **K4 (money):** the run stops if its own cost would pass **$450**, and nothing launches if the ledger total would
  pass the **$5,000** cap (`MC_BUDGET_USD`, checked by `infra/aws/keep_running.sh` before every launch).

## Plan
- **Data:** `m21mix_v2` (SmolLM2 tokens), unchanged: 8.0B tokens in 4k rows, 3.0B in 16k rows, 0.5B in 64k rows,
  the same shards in the same order as the other four models.
- **Compute:** home servers (Odra, Polonez) for the checks, the build's GPU tests and a short pipeline check, under
  the cooling rules; one 4 × L40S spot node for the 150M run, about 40–45 hours and $320–360 with its tests.
- **Steps:** 15,259 × 128 rows of 4k, 5,722 × 32 rows of 16k, 954 × 8 rows of 64k; Muon peak 0.01 (a tenth in the
  64k stage) + AdamW for vectors, bf16, flex attention, `--fast`, seed 42.
- **Launch (150M):** `infra/aws/jobs/e38_board.sh 42 muon:0.01 living` through `infra/aws/keep_running.sh` with
  `MC_SPOT=1 MC_BUDGET_USD=5000`; a thin wrapper may set the experiment tag and the S3 folder to `runs/e39/`. W&B
  name "150M living concepts, Muon".
- **Order of work:**
  1. Read the spend (`infra/aws/status.sh`) and the servers' state; P1–P3 on the released model.
  2. `E39_plan.md`, then the build: spans only first (a swap of the labelling behind the existing read), then the
     concepts, versions, two-level choice and remaining-count summaries; cached decoding and the long-length list
     path for the tests.
  3. Unit tests, then P4 on a home card.
  4. A pipeline check on a home card, not a screen: a few hundred steps of each of the three stages at a small
     size and the tests on the result, so the paid run cannot die on a crash (as E38's plan step 3).
  5. The 150M run on AWS if K0 passes; K3, K2a and K2 read during the run; the E38 tests on the 64k model.
  6. Capability checks core package for the new read on the home servers (`capability-checks`), results to the
     ledger and the NAS; `experiment-track` for the registry and the run report.
- **New foundation code:** a reusable module for the grouping and its index (`nn/living_concepts.py`), selected by
  `par_global_read = "living"` in `PerceiverARLM`; additions to `nn/concept_memory.py` / `nn/exact_read.py` only
  where the read needs the versioned summaries and the remaining count. No new training script.
- **Unit tests (CPU reference path and GPU flex path):** causality (changing a token changes no earlier output, and
  no reader two strides back or less sees it through the memory); every past token counted exactly once; identical
  member keys → the summary read equals full attention; spans-only with given labels and `living_score mass` equals
  the existing concept read with the same labels; cached decoding equals the full forward at 4k and 16k; the bounds (span ≤ 16, ≤ 128 per
  stride, no span across a stride or a document, concepts ≤ C_max, overflow joins); padding readers read zero;
  thresholds frozen in eval and moving toward their targets in training, equal on all cards; flex equals the
  reference path.

## Risks
- **The diagnosis may be wrong** (P2): the topic index may win on multi-item tasks because it has no summaries to
  share the softmax with, not because its groups are contiguous. E39 reduces the summaries about 8× at 16k, so it
  moves in that direction too, but would not isolate it; R2 and R3 are there to tell.
- **The first level may hide the right span** (the catalogue's lesson). Mitigations: content clusters, 32 concepts
  kept, the score's variance term, the flat test switch.
- **Coarser summaries** (about 64 tokens per concept against 8) may give back part of the 0.023–0.068 nats the
  summaries carry; S3 allows 0.010 at 16k and 0.020 at 64k.
- **The thresholds are in cosine units of keys that move during training.** The controllers hold the two ratios in
  training; at test time the thresholds are fixed and R4 shows whether real documents stay inside the bounds.
- **Stage 1 trains on 4k rows of short documents**, where few readers have any memory two strides back; the
  memory is mostly shaped in stages 2 and 3, as for the other models.
- **Three changes land together** (spans, concepts, choice by match). That is the bet as approved; attribution comes
  from the test switches, and from a spans-only run only if needed.
- **No small-scale screen** (the author's choice, for speed): the first evidence that a model learns with the new
  grouping comes from the paid run itself. The readings inside the run bound what a bad run can cost: about $20 at
  K2a, about $180 at K2.
- **One seed.** Differences of a few points on single RULER tasks are not established.
- **Kernel work:** the versioned summary call and the remaining-count table are new flex masks; the open table
  doubles (128 slots per stride). S7 and K3 guard the cost.

## Budget
Cap **$5,000** (author, 2026-11-22). A commit of 2026-11-20 quotes about $3,089 spent; the build session re-reads the
ledger before any launch. This spec's own AWS cost: one 150M run, cap $450.

## Result
—
