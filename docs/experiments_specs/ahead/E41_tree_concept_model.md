# E41 — Tree concept model: a tree of learned summaries, read top-down, trained with NCM's signals and a next-concept loss (150M)

- **Status:** approved by the author 2026-11-26 (after the alignment page: "Build and train 150M, use AWS and Odra or
  Polonez if needed ... train and evaluate, then optimize if needed, iterate, your goal is to beat all the previous
  models and all the current small SOTA models, you don't need to keep our past architecture, invent and change keep
  the NCM idea and JEPA, remember of our vision 10M context and agent2agent communication. You can spent max $2000").
  Spec written 2026-11-26 20:55 UTC. In build.
- **Serves:** Vision priority 2 (about 10M tokens of context: a read whose size does not grow with the text) and
  priority 4 (agent-to-agent: the tree is the thing one agent hands another), and the meridian21 goal.
- **Implementation plan:** [E41_plan.md](E41_plan.md)
- **Alignment page:** `docs/3_Evaluations_and_Baselines/e41_architecture.html` (published 2026-11-26 as "Tree Concept
  Model"; the author answered it with the go-ahead above).
- **Owner / dates:** Krzysztof Sopyla · opened 2026-11-26 · closed —
- **ID:** a new family (the author: a new architecture). IDs in use on every branch on 2026-11-26 (`refs/heads` and
  `refs/remotes/origin`): E37 (a, b, c, e, f, s), E38, E39, E39a, E40. E41 is the next unused number.

## What the author asked for (the idea, in his words)
"the NCM ladder is definitely something we could reuse to build the hierarchical summaries for memory. Then those
summaries could be reread and help to steer where the reader should focus, going deeper into the tree, from trunk
through branches to leaves, all those things should be learnt (not averaging any more). Then the thin decoder should
have access to all for the paths in this tree, from coarse summaries to the tokens themselves - in this way, we will
have concepts (coarse summaries) and the indices of specific tokens with information." Added: "next concept
prediction loss could support better summaries formation and better concept formation, and ncm style decoder";
"go all in with the new arch".

## What we know (measured, 150M, card protocol unless said)
- **The scoreboard** (nats per token on the last 4,096 tokens of real long documents, lower is better):

  | model | 16k | 64k | 128k | RULER 13-task mean 4k / 8k / 16k / 32k |
  |---|---|---|---|---|
  | recall run (concept memory, stages 2–3 on `m21mix_v4`) | 2.125 | 2.291 | 2.461 | .567 / .527 / .392 / .362 |
  | concept memory (released) | 2.126 | 2.293 | 2.451 | .488 / .439 / .373 / .319 |
  | topic index | 2.129 | 2.299 | 2.426 | .541 / .540 / .439 / .390 |
  | full attention | 2.139 | 2.344 | 2.421 | .459 / .407 / .255 / .202 |
  | living concepts (E39) | 2.161 | 2.388 | 2.487 | .436 / .387 / .269 / .201 |

  Held-out loss at the end of stages 2 / 3: recall run 1.996 / 1.938. BLiMP 79.9, 7-task core 46.37.
- **The summaries carry almost nothing alone** (E37f's table, released model at 64k): whole document 2.301, last 1k
  tokens only 2.444, summaries only 2.534 (worse than the cut), summaries off 2.364. The far past is worth 0.142;
  next to the 512 opened tokens the summaries add 0.063. A summary's value is the plain mean of about 8 tokens'
  64-number values; no weights write it, and training never needs it because the best 512 tokens are always opened.
- **One flat level does not reach 10M tokens.** A reader scores one summary per 8 tokens: 8,000 at 64k, 1.3M at 10M.
  Past the training length the summaries' count swamps the read (patched by the length scale).
- **Rereads work.** Four tied passes over the read is the released recipe; E39's flat search at test time brought
  single lookups back (0.48 → 1.00 at 16k), so the reader's choice matters more than the grouping.
- **Outside evidence.** NCM (arXiv 2610.14162): a residual ladder trained with nested dropout puts the gist in the
  coarse levels; its "down" is plain averaging and its tokenizer is frozen (both named as limits). Next-concept
  losses on top of next-token loss: NCP (2609.10715, abstract only: the same loss with 51 % of the tokens), CoCoMix
  (2502.08524: 21.5 % fewer tokens; predicted concept fed back into the stream), Future Summary Prediction
  (2510.14751: ARC-E 71.8 → 76.6 at 8B). Text JEPA without a generative anchor collapses (2607.23531); squared-error
  concept prediction drifts to the average (Large Concept Models, 2412.08821). BEST-RQ (2202.01855): predicting the
  codes of a fixed random projection is a collapse-free target.

## Hypothesis
Concept memory's far past is one flat level of plain averages that training never needs. If the memory becomes a
**tree of summaries** (statistics add exactly, so a parent is the sum of its children), a **reasoner over the tree's
concepts** corrects every summary's key and value and predicts the next passage's concept codes, and training
**closes the book at a random depth**, then the summaries carry the document and can be routed. Falsifiable, against
the recall run on the same data and steps: 64k documents at least 0.010 nats better (≤ 2.281), the RULER 13-task
mean at least the best of our models at every length, and a read bounded to about 1,500 entries loses at most 0.010
at 64k against the model's own flat read and reads 1M-token documents.

## Builds-on
- **Foundation:** `PerceiverARLM` and the shared trainer; the concept read's building blocks (`nn/concept_memory.py`,
  `nn/exact_read.py`); the E39 branch's shared flex helpers and recall recipe (merged into `tree-concept`). New,
  config-selectable: `par_global_read = "tree"` in `nn/concept_tree.py`, arm `tree` in `scripts/run_e33n.py`.
- **Body:** size `m120t` unchanged (13 token layers of width 896, reading layer 3 looped four times with layer 4,
  14 query heads, one key/value head of 64). The reasoner adds about 7M weights (two tied layers of width 512,
  looped twice): about 148.5M in all, under the board's 150M cap.
- **Init, run 1:** concept memory's stage-1 final (`runs/e38/e33n_m120t_e37a_muon0.01_tok8.0e9_seed42/trunk/…/final`),
  then stages 2–3 on `m21mix_v4`, exactly the recall run's recipe (5,722 + 954 steps, Muon 0.01, seeds). Every new
  part starts as an identity (reasoner outputs zero; the tree read with every branch followed is the concept read),
  so step 0 is concept memory's stage 1. Why not from scratch first: stage 1 reads 4k rows, where the tree has one
  level and the far past is six blocks; the recall run and E39a set the precedent, it costs about $150 instead of
  $320, and the baseline is like for like. **Run 2** (after a pass): the same recipe from scratch, with the new
  signals on from the first step and the weights moved toward a thinner decoder if the home screen favours it.
- **Baseline to beat:** the recall run (table above).
- **Materially new:** levels above the concepts; learned corrections to summaries; a top-down read with budgets;
  closed-book cuts at any depth; a next-concept loss on the tree; a plan fed back to the tokens.

## The architectural bet
**The tree (writer).** Level 1 is concept memory's concepts: 64 per closed 512-token block, grouped by meaning, each
with links to its member tokens and the statistics n, mean key, key spread, mean value, mean state. Level 2 groups a
block's 64 concepts into 8 by meaning. Every higher level groups the 64 nodes of 8 neighbouring containers into 8
(containers of 4k, 32k, 256k, 2M tokens). Statistics add, so every merge is exact and weight-free; a level first
seen at 10M tokens needs no new weights. A container that holds two documents never closes (its children stay
readable).

**The reasoner (concept space).** A small transformer over the nodes of level 2 and above (about 1,150 at 64k). A
node sees nodes that closed no later than itself, in its document. From its state it writes, zero at the start:
a **key correction** and a **value correction** for the node (the summary is the average plus what the reasoner
adds); the same for level-1 concepts through their parent; and a **plan**, one vector added to the token stream of
the block that follows.

**The read (looped reader).** Top-down, per query and per pass: score the readable nodes of a level, follow the best
`B` deeper, and read every other node as one summary entry. At the leaves open up to 512 tokens word for word.
Every past token is represented once. Two ways to score a node: its own estimate (`descent`, needed past 1M) and the
sum of its concepts' estimates (`lookahead`, exact routing, affordable to 128k). With every node followed the read is
the concept read exactly.

**Training signals.**
1. Next word (kept).
2. **Closed-book cuts** (NCM's nested dropout; E37f's one-level version generalised): each training row draws its
   read. 40 % as normal (everything followed, 512 tokens); 20 % a bounded tree read (16 followed per level, 512
   tokens); 40 % cut: nothing below a drawn level may be read, no tokens opened (level 1 = E37f's closed book).
3. **Next concept** (NCM's generator loss in JEPA form): after each block the reasoner predicts the concept codes of
   the next block, coarse (its 8 level-2 nodes) and fine (its 64 concepts). Codes are the nearest entries of a
   slowly updated codebook over a fixed random projection of the node states (collapse-free; cross-entropy, not
   squared error). Weight 0.1 each.
4. **Parent predicts children**: a node's own estimate is trained toward the sum of its concepts' estimates for the
   batch's real queries (weight 0.1). This is what teaches `descent` routing.

Deferred to a second run (each a flag): rebuilding a passage from its own summaries, swapped branches, the
bag-of-words future target, the thin-decoder weight split.

## Why this is not a safe retread
E37f tests one closed-book level on plain averages; E39 regrouped the leaves; E33t's two-level search lost as a
test-time switch and its trained arm was deferred; E37's reasoner was shelved with the book always open. E41 is the
first model here whose summaries are written by a network, trained to stand alone at every depth, and routed.
Analogies: a Laplacian pyramid / predictive coding (a level stores what the level above did not predict); Barnes–Hut
and fast multipole codes (far field by moments, opened where it matters); BEST-RQ targets.

## Success criteria (set before running; card protocol `scripts/release_evals.sh`, paired by document)
- **S1 (long text, decides):** 64k ≤ **2.281** and 16k ≤ **2.120** (the recall run 2.291 and 2.125), the paired
  differences' 95 % intervals below zero. Reachable: the summaries add 0.063 at 64k today while carrying nothing
  alone; S1 asks for a sixth more.
- **S2 (lookups, decides):** RULER 13-task mean ≥ **.567 / .540 / .439 / .390** at 4k / 8k / 16k / 32k (the best of
  our models at each length), at least three of the four.
- **S3 (the notebook carries the document):** summaries only minus the last-1k cut at 64k ≥ **+0.030** (released
  model −0.091; E37f asks its own arm for +0.057).
- **S4 (a read that does not grow):** at 64k the bounded read (16 per level, 512 tokens) within **0.010** of the
  model's flat read; 128k ≤ **2.421** (full attention, the best of six); a 1M-token document read on one 24 GB card,
  loss reported.
- **S5 (no harm):** held-out loss at the end of stage 3 ≤ **1.948**; BLiMP ≥ 79.6; 7-task core ≥ 45.7; speed ≥ 0.85 ×
  the recall run's (59k tokens/s in stage 2).
- The bet **succeeds** with S1 and S2. S3 and S4 qualify the 10M claim.
- **Reported, no marks:** cross-model bits per byte at 61k against Granite 4.0 350M (0.776), Gemma 3 270M (0.805),
  SmolLM2-135M (0.832), ours 0.795; RULER 4k / 8k against Granite (.62 / .54); the list exam; a two-agent hand-over
  of the tree's top levels; board place. General quizzes (ARC-E 49–51 against SmolLM2's 64) are a data-budget gap
  that 11.5B tokens cannot close; not claimed.

## Kill criteria
- **K1:** held-out loss at step 1,000 of stage 2 above **2.227** (the recall run 2.177 + 0.05) → stop the node.
- **K2:** stage-2 speed over the first 300 steps below 0.70 × the recall run's (48.6k tokens/s) → stop, fix or report.
- **K3 (money):** $220 for a stages-2–3 run, $400 for a from-scratch run, **$2,000 for E41 in all**.
- **K4:** identity check fails (step-0 loss differs from concept memory's stage-1 final by more than 1e-3) → no launch.

## Reading the result
- S1 and S2 met → run 2 from scratch (the claim "trained from scratch" and the signals' effect in stage 1), then the
  1M and agent-to-agent tests, then the compression ladder on the tree's levels.
- S2 met, S1 missed but S3 met → the summaries carry more, the mix of reads needs tuning: next run changes the row
  draw and the reasoner's size.
- S3 missed → corrections on averages are not enough; next run adds the rebuild signal and a wider value per node.

## Plan
- **Data:** `m21mix_v4` in S3 (stages 2–3: 3.0B tokens at 16k, 0.5B at 64k).
- **Compute:** home cards for the tests, the identity check, the pipeline check and the speed check (Odra 3 cards,
  Polonez cards 0, 1, 3); AWS spot `g6e.12xlarge` (4 × L40S) for the 150M run, about 20 h and $150; release tests
  on home cards.
- **Launch:** `infra/aws/jobs/e41_tree.sh` (seeds the run folder with stage-1 weights, then
  `e38_board.sh 42 muon:0.01 tree` with `E38_EXPERIMENT=E41 E38_MIX=m21mix_v4`), through `infra/aws/keep_running.sh`
  with a money guard at K3.
- **New foundation code:** `nn/concept_tree.py`, wiring in `nn/perceiver_ar_lm.py`, trainer arguments, arm `tree`,
  tests `tests/test_concept_tree.py` (+ CUDA), eval switches, cached decoding.

## Log
- **2026-11-26 20:55–21:48 UTC, build and checks (home cards, $0).**
  - Tests: 45 on the laptop, 62 with the GPU tests on an RTX 3090 (Odra and Polonez): every node followed = the
    concept read; flex = the reference forward and backward for bounded, cut and training rows; gradient
    checkpointing agrees; cached decoding = the full forward to 6e-7; the streamed read = the table read.
  - **K4 (identity) passed** on the released concept-memory weights: hidden states identical at 16k and 64k
    (difference 0.00), loss equal (`scripts/e41_tree_view.py check`, Odra 21:16–21:21 UTC).
  - **The untrained tree read on the released model** (loss over whole rows of held-out books, nats per token; the
    far read is worth 0.071 at 16k and 0.102 at 64k on these rows):

    | read | 16k: loss above flat | summary entries | 64k: loss above flat | summary entries |
    |---|---|---|---|---|
    | every node followed (flat) | 0 | about 930 | 0 | about 4,000 |
    | 64 followed per level | +0.0006 | 474 | +0.0060 | 824 |
    | 16 followed per level | +0.0069 | 173 | +0.0266 | 286 |
    | 8 followed per level | +0.0124 | 87 | +0.0530 | 176 |
    | 16 per level, scored by a node's own estimate | +0.0106 | 173 | +0.0404 | 291 |
    | closed book: concept summaries only | +0.0437 | 990 | +0.0743 | 4,015 |
    | level-2 summaries only | +0.0659 | 124 | +0.1105 | 504 |
    | level-3 summaries only | +0.0714 | 40 | +0.1573 | 84 |
    | no summaries (opened tokens only) | +0.0160 | — | +0.0462 | — |
    | no far read | +0.0706 | — | +0.1015 | — |

    The tree works as a read without training (64 per level keeps 94 % of the far read at 64k with a fifth of the
    entries); summaries of level 2 and above alone are worse than no far read, the state the cut rows train on.
  - **Speed** (one RTX 3090, 16k rows, gradient checkpointing; `speed_check.sh`): concept memory 7,278 tokens/s,
    `tree_read` 6,931 (0.95 ×), `tree_noaux` 6,548 (0.90 ×), `tree` 6,435 (0.88 ×): above K2 and S5's 0.85 ×.
  - **Pipeline check** (`e38_pipeline_check.sh 0.01 tree m120t`, Odra card 2): the three training stages with
    evaluation and the resume passed (21:17–21:35 UTC; the 64k stage ran under gradient checkpointing with the
    auxiliary losses). Its test stage found a bug: generation casts the model to bf16 and the codebook comparison
    mixed fp32 with bf16 (RULER crashed). Fixed in `4f897796` (codebooks compared in fp32; no code targets outside
    training), with a regression test; the test stage is rerun on the check's model.
  - Row draw as launched: 40 % as normal, 20 % bounded (4, 8, 16 or 64 nodes per level, drawn evenly; half of them
    scored by a node's own estimate), 40 % cut at a level drawn evenly (no token opened; half of the level-1 cuts
    open 64 tokens).
- **2026-11-26 21:48:43 UTC, run 1 launched.** Arm `tree` (148.9M), stages 2–3 on `m21mix_v4` from concept memory's
  stage-1 final (copied server side to `runs/e41/e33n_m120t_tree_muon0.01_tok8.0e9_seed42/trunk/` at 21:38 UTC,
  laptop admin profile). Branch `e41-run1` at `4f897796`. Keeper `Cache/keepers/e41-tree.sh` (at most 8 launches,
  30 h per node) launched spot g6e.12xlarge `i-0881ec244a58d5d8c` in us-east-1c (1a and 1d had no capacity), up to
  $7.87/h. Watcher `Cache/keepers/e41-tree-watch.sh` (from `infra/aws/watch_run.sh`): stops the node above 2.227
  at the 5th held-out evaluation of stage 2 (K1), reports below 48.6k tokens/s (K2), stops at $220 for this run
  (K3). Family spend before the launch $3,684 (the E39a node still running); E41 spend so far $0.

- **2026-11-26 22:00 UTC, the pipeline check's test stage passes with the fix** (`scripts/e41_check_tests.sh`, Odra
  card 2: long-document loss, RULER quick form with cached decoding and its old-loop check, core).
- **2026-11-26 21:50–22:57 UTC, the baselines for S3 and S4: the released concept memory read through the tree, card
  protocol** (Odra cards 0–1; `evaluation/long_context_use.py` on the tree view; nats per token on the last 4,096
  tokens; files `docs/2_Experiments_Registry/results/evaluations/E41_released_tree_view/`):

  | read (no training for it) | 16k, 120 documents | 64k, 55 documents |
  |---|---|---|
  | whole document, every node followed | 2.1257 | 2.2933 |
  | 64 followed per level | +0.0009 | +0.0141 |
  | 16 followed per level | +0.0086 | +0.0506 |
  | 16 per level, scored by a node's own estimate | +0.0231 | +0.0906 |
  | no summaries (opened tokens only) | +0.0225 | +0.0675 |
  | closed book: concept summaries only (cut 1) | +0.2066 | +0.2358 |
  | level-2 summaries only (cut 2) | +0.2774 | +0.3107 |
  | level-3 summaries only (cut 3) | +0.2908 | +0.3931 |
  | no far read | +0.2710 | +0.2869 |
  | last 1k tokens only (the document cut) | +0.0942 | +0.1477 |

  The whole-document figures reproduce the card (2.126 and 2.293). Notebook value at 64k (document cut minus
  summaries only): concepts −0.088 (E37f measured −0.091), level 2 −0.163, level 3 −0.245: the untrained upper
  levels are worse than the untrained concepts. S4 asks the trained model to bring "16 per level" from +0.051 to
  within 0.010 of its flat read.
- **2026-11-26 21:51 UTC, stage 2 started on the node.** First log lines: next-word loss on the mixed rows 2.49 →
  2.42, next-concept loss off chance (coarse 6.90 → 6.56, chance 6.93), route 0.17; 52k tokens/s (0.75 × the recall
  run; a home card gave 0.88 ×). Likely cause: with four cards every step waits for the slowest card, and a bounded
  row is the slow kind (1.3 × a flat row; a cut row is 2–8 × faster). For run 2 the read type is drawn per step,
  the same on every card.

## Result
<Filled in after, by experiment-track.>
