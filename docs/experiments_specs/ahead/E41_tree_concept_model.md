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

- **2026-11-26 23:05 UTC, K2 passed:** median 51.9k tokens/s over stage-2 log lines 4–33 (mark 48.6k; the recall run
  69.7k). Step 200: held-out loss 2.339 (the recall run 2.308 at the same step).
- **2026-11-26 23:02 UTC, two home retrofits started** (Odra cards 0–1, `scripts/e40_train.sh`, E40's rows, order and
  schedule from the released model, the reasoner at its own AdamW rate 6e-4): `tree` (run 1's recipe) and
  `tree_soft` (70 % as normal, 15 % bounded, 15 % cut). Reference: E40's control. They pick run 2's row draw.
- **2026-11-26 23:04–23:33 UTC, the first lookup past 128k on any 150M model** (released concept memory through the
  tree view, Polonez card 0, `scripts/e41_long_needle.py`: RULER's number needle in held-out text, 4 depths per
  length, peak 9.1 GB): at 64k the flat read finds 3 of 4, 16 per level 3 of 4, 64 per level 2 of 4, 16 per level by
  a node's own estimate 0 of 4 (first digit right 4 of 4); at 256k every read finds 0 of 4 (72–106 s per row). The
  released model does not look up past its training length whichever way it reads; the trained tree model gets
  the same test (S4), and rows longer than 64k in training (stage 4) are the likely requirement.

- **2026-11-27 00:52:50 UTC, run 1 stopped by K1** (the watcher, by itself). Held-out loss on the normal read at
  steps 200 … 1,000 of stage 2: 2.339, 2.326, 2.290, 2.260, **2.236** against the recall run's 2.308, 2.255, 2.223,
  2.195, 2.177: behind by 0.031, 0.071, 0.068, 0.065, 0.059 (mark 2.227). Node `i-0881ec244a58d5d8c` terminated;
  3.07 h, **$24.13**. Next-concept loss fell from chance to about 3.0 (coarse) and 5.3 (fine); route 0.16 → 0.31.
- **Why, from the home retrofits (released model, E40's rows and schedule, held-out loss on the normal read at step
  286 and 572; E40's control 1.9275 and 1.9257):**

  | recipe | step 286 | step 572 | against the control |
  |---|---|---|---|
  | closed-book and bounded rows only, no reasoner (`tree_read`, Polonez) | 1.9295 | — | +0.002 |
  | the reasoner, plan and both losses, every row normal (`tree_flat`, Polonez) | 1.9487 | — | +0.021 |
  | everything, run 1's recipe (`tree`, Odra) | 1.9509 | 1.9458 | +0.023, +0.020 |
  | everything, fewer closed-book rows (`tree_soft`, Odra) | 1.9493 | 1.9416 | +0.022, +0.016 |

  The closed-book rows cost the normal read almost nothing; the reasoner with its plan and losses is what hurts
  (new weights that start at zero move fast under Muon and AdamW, and the next-concept loss keeps moving the states
  the corrections are read from). `tree_read` also trains faster than the control (8.7k against 7.3k tokens/s on
  one card: a cut row is cheap).
- **2026-11-27 00:44 UTC, the reasoner taken apart on home cards** (three retrofits, first reading about 01:30 UTC):
  values only, no plan, no losses (`tree_vals`, Odra card 1, replacing `tree_soft`); the reasoner and plan without
  the losses (`tree_noaux`, Polonez card 3); the full reasoner with losses of weight 0.02 that do not reach the
  reasoned states (`tree_gentle`, Polonez card 0, replacing `tree_flat`).
- **2026-11-27 00:54 UTC, run 2 queued:** arm `tree_read_step` (141.5M, no new weights: the tree read, closed-book
  and bounded rows 40 / 20 / 40, the read drawn per step so that every card of a step holds the same kind of row),
  stages 2–3 from concept memory's stage-1 final on `m21mix_v4`, branch `e41-run2` at `21fd2c05`, S3
  `runs/e41/e33n_m120t_tree_read_step_muon0.01_tok8.0e9_seed42`. Keeper `Cache/keepers/e41-read.sh`, watcher
  `Cache/keepers/e41-read-watch.sh` with the same marks (K1 2.227, K2 48.6k tokens/s, $220). The first launch
  attempt met the spot limit while run 1's node was shutting down; the keeper retries every 10 minutes.

- **2026-11-27 01:04:27 UTC, run 2 launched** on spot g6e.12xlarge `i-01b73cd7e0f8c41d8` (us-east-1c, up to $7.87/h).
- **2026-11-27 01:28 UTC, the reasoner's cost by part** (home retrofits, held-out loss on the normal read at step
  286; control 1.9275):

  | what the model has besides the tree read and the closed-book rows | step 286 | against the control |
  |---|---|---|
  | nothing (`tree_read`) | 1.9295 (step 572: 1.9273, control 1.9257) | +0.002 |
  | the reasoner, value corrections only (`tree_vals`) | 1.9298 | +0.002 |
  | the reasoner with key corrections and the plan (`tree_noaux`) | 1.9358 | +0.008 |
  | the same with both losses at weight 0.02, kept off the reasoned states (`tree_gentle`) | 1.9368 | +0.009 |
  | the same with both losses at weight 0.1 into the reasoned states (`tree`, run 1) | 1.9509 (step 858: 1.9417, control 1.9241) | +0.023, +0.018 |

  Value corrections are free; key corrections and the plan together cost 0.006; the losses pushed into the states
  cost 0.015 more and recover slowly. Training loss on the same rows: `tree_vals` equals `tree_read` so far (paired
  difference +0.0003 over 310 steps), so the value corrections have not helped the cut rows yet either; the mixed
  rows cost 0.054 against all-normal rows at the start and 0.046 over the next 280 steps.
- **2026-11-27 01:30 UTC, two more retrofits** in place of `tree` and `tree_gentle`: values and the plan from the
  reasoned state (`tree_vp`, Polonez card 0: is it the plan or the keys?), and values with a plan read from the
  next-concept predictor, which predicts the block that reads the plan and is the only thing the prediction loss
  trains (`tree_pred`, Odra card 0).

- **2026-11-27 02:06 UTC, run 2: K2 passed and the first evaluation ties the recall run.** Median 66.3k tokens/s
  (0.95 × the recall run; run 1 had 51.9k: the per-step draw works). Held-out loss at step 200: **2.3077** (the
  recall run 2.3081).
- **2026-11-27 02:16 UTC, it is the plan.** Step 286 against the control's 1.9275: values and the plan from the
  reasoned state (`tree_vp`) 1.9369 (+0.009); values and a plan from the predictor (`tree_pred`) 1.9509 (+0.023, as
  bad as run 1's recipe); values only at step 572 1.9273 (+0.002, equal to no reasoner); keys, values and plan at
  step 572 1.9316 (+0.006). A vector added to every token's stream is what the model does not tolerate; corrections
  to the summaries' keys and values are. Next (started 02:17 UTC): key and value corrections without a plan
  (`tree_kv`, Polonez card 0) and the same with both losses shaping the reasoned states (`tree_np`, Odra card 0),
  the next-concept loss as the author meant it, without the plan.

- **2026-11-27 02:59–03:38 UTC, home readings.** Key and value corrections without a plan (`tree_kv`) 1.9291 at
  step 286 (+0.002); the same with both losses shaping the reasoned states (`tree_np`) 1.9303 (+0.003): without the
  plan the next-concept loss does not hurt. Neither has helped the cut rows yet (training loss on the same rows
  against `tree_read`: `tree_vals` +0.0002 over 960 steps, `tree_np` +0.0014 over 290). `tree_read` finished its 16k
  part (02:45 UTC): a steady +0.002 behind the control at all five evaluations.
- **2026-11-27 03:38 UTC, what 93.7M tokens of closed-book rows did** (`tree_read` retrofit after its 16k part, 120
  documents at 16k, against the released model; Polonez card 1):

  | read | released | after the retrofit | change |
  |---|---|---|---|
  | whole document | 2.1257 | 2.1229 | −0.003 |
  | 16 followed per level | 2.1344 | 2.1292 | −0.005 |
  | 16 per level, by a node's own estimate | 2.1488 | 2.1397 | −0.009 |
  | concept summaries only | 2.3323 | 2.3070 | −0.025 |
  | level-2 summaries only | 2.4032 | 2.3695 | −0.034 |
  | level-3 summaries only | 2.4165 | 2.3757 | −0.041 |
  | no far read | 2.3968 | 2.3856 | −0.011 |
  | last 1k tokens only | 2.2200 | 2.2150 | −0.005 |

  Every read improved and the normal read did not pay, but the notebook's value is still negative (−0.092 against
  the cut, from −0.112): at the released model's tiny final rates the summaries learn slowly.
- **2026-11-27 03:32 UTC, run 3 launched** next to run 2 (the second spot slot came free when E39a finished): arm
  `tree_np_step` (148.9M: the reasoner with key and value corrections, both losses at 0.1 shaping the reasoned
  states, no plan; the read drawn per step), branch `e41-run3` at `7124da59`, S3
  `runs/e41/e33n_m120t_tree_np_step_muon0.01_tok8.0e9_seed42`, spot g6e.12xlarge `i-054b0dd1a34c11be6` (us-east-1d).
  Keeper `Cache/keepers/e41-np.sh`, watcher `Cache/keepers/e41-np-watch.sh`, the same marks.
- **2026-11-27 03:37 UTC, run 2 passed K1:** held-out loss at steps 200 … 1,000: 2.3077, 2.2616, 2.2332, 2.2025,
  **2.1863** (the recall run 2.3081, 2.2554, 2.2226, 2.1947, 2.1773; mark 2.227). Behind by −0.000, 0.006, 0.011,
  0.008, 0.009 on the normal read. Its training loss on the same rows is 0.061 above the recall run's at the start
  and 0.051 by step 1,000 (the cost of the cut and bounded rows; the number to watch fall).
- **2026-11-27 03:44 UTC, wide summaries built** (commit `c8720da6`, arm `tree_wide`). Why: a summary hands a
  reader one value of 64 numbers, shared by all 14 heads: about 8 numbers per token of the text it stands for. The
  reasoner's corrections to that value have not helped the cut rows, so the limit may be the room, not the content.
  With `tree_wide` a writer (one zero-initialised linear map, 0.8M weights, its own AdamW rate) turns a node's mean
  state (896 numbers) into a value per head (14 × 64), added to the shared value in the reference, flex, streamed and
  cached reads: 14 times the room, and each head reads its own slice with its own attention. A fresh model is still
  the concept model. Tests: 174 on the laptop, 23 on a GPU (flex = reference forward and backward for three wide
  cases, causality, the three reads and the cached decoder agree). Screen: home retrofit on Odra card 1 (queued
  03:48 UTC behind `tree_vals`, whose 64k part was dropped: value corrections alone are a null result), compared
  with `tree_read` on the same rows.
- **2026-11-27 03:45 UTC, home cards.** Another session started an evaluation on Polonez card 1, so the queued 64k
  part of `tree_read` there was cancelled (by process id) to keep Polonez at three busy cards. Polonez has been
  under load since 23:35 UTC: `tree_kv` ends about 05:35 UTC, then a cooldown of at least 20 minutes.
- **2026-11-27 04:00 UTC, the next step is a stage on rows of 262,144 tokens.** The released model finds a planted
  number at 64k (3 of 4) and never at 256k (0 of 4, every read): no row longer than 65,536 tokens was ever trained,
  because the concept read ranks every concept for every query (the cost per row grows with the square of its
  length). The tree's bounded read is the part of this design that removes that limit, and "memory on text at 256k"
  is the first rung of the author's roadmap. Plan: (1) time a real training step at 64k, 128k and 256k through the
  bounded read (`scripts/e41_long_step.py`); (2) rows: long documents (books, PDFs, government and parliament
  records of at least 262,144 tokens) and planted-fact recall rows at 256k (`scripts/build_reasoning_rows.py`, a new
  set `recall_256k`); (3) a home screen from the released weights: does the number at 256k and 1M get found;
  (4) on AWS as stage 4 of the best stages-2-3 run.

## Result
<Filled in after, by experiment-track.>
