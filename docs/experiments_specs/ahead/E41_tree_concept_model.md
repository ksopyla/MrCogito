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
- **2026-11-27 04:20 UTC, why no row was longer than 64k** (a real training step of the released weights through
  the tree, one RTX 3090, bf16, gradient checkpointing; `scripts/e41_long_step.py`): 64k rows 19 s a step with the
  whole read and 10.4 GB; 128k rows 62–65 s and 20.6 GB; 256k rows do not fit (one allocation of 15.9 GB). The
  bounded read as built bounds what a query *reads*, not what it *scores*: `choose_chunk` still scores every concept
  of the row for every query (the summed-mass score needs them all), so cost and memory grow with the square of the
  row whatever the budget.
- **2026-11-27 04:25–04:38 UTC, the top-down read built** (`tree_read_sparse`, config `tree_sparse`, commits
  `287f1acc`, `0db415c6`, `272dafbb`). A query scores, level by level from the top, only the nodes it may reach: the
  children of the nodes it followed one level up (packed lists: a node has 8 children on average and up to 64) and
  the level's roots (nodes whose parent is not closed yet, or never will be). It is the same read as the streamed
  one for bounded rows chosen by a node's own estimate: 18 comparison cases (one to five followed per level, a bound
  per row, cuts at levels 2–4, no tokens, a small token cap, wide summaries, token updates, packed documents), output
  and gradients; a model with `tree_sparse` gives the bounded model's logits and never reads ahead; with gradients
  on, every chunk of queries is its own gradient checkpoint, so a step's memory does not grow with the row. A
  `tree_sparse` model draws only bounded rows in training (an as-normal row follows 4 × the budget; a cut row keeps
  its bound above the cut; every row by descent). 199 tests on the laptop, 30 on a GPU.
  One forward of the released weights, 16 followed per level, one RTX 3090 (`scripts/e41_read_profile.py`):

  | row | table read (flex) | top-down read | peak memory, top-down |
  |---|---|---|---|
  | 16k | 0.9 s | 1.0 s | 1.3 GB |
  | 64k | 10.1 s | 4.4 s | 2.9 GB |
  | 256k | does not fit (streamed: 72–106 s) | 19.2 s | 8.2 GB |

  The cost per token is now flat in the row's length (73 µs at 256k, 66 µs at 64k).
- **2026-11-27 04:16 UTC, rows of 262,144 tokens built at home** (Odra, 8 minutes, `scripts/e41_xl_data_build.sh`,
  recipe `data/mix_recipes/m21_mix_xl1.json`): single documents that long from books (93 rows), US government
  publications (130) and open-access academic books (48); 153 recall rows (`recall_256k`: several books in a row,
  25 planted items a row on average, 3,853 asked items, the answer's distance 162k tokens at the median); packed
  rows of PDFs, code and papers. The home copies of the other long sources hold no document that long (PDFs stop at
  154k tokens there).
- **2026-11-27 04:38 UTC, home screens at their later evaluations** (held-out loss, steps 286 … 1,430): value
  corrections 1.9298, 1.9273, 1.9265, 1.9251, 1.9233; the reasoner with its plan and no extra losses 1.9358, 1.9316,
  1.9299, 1.9272, 1.9240; key and value corrections 1.9291, 1.9267, 1.9256 (running); those with both losses
  1.9303, 1.9280, 1.9276 (running). The plan's cost fades with training (+0.006 at step 286, +0.001 at step 1,430
  against value corrections alone); nothing with a reasoner is ahead of the closed-book rows alone yet.
- **2026-11-27 04:42 UTC, a 256k training row does not fit a 24 GB card** (top-down read; peak 20.2 GB, then out of memory). It is not the far read (a 256k forward peaks at
  8.2 GB): gradient checkpointing keeps every block's input, 262,144 × 896 fp32 numbers = 0.94 GB each, about
  19 GB for the 13 layers and the reading layer's loop passes. So the 256k stage runs on the 48 GB AWS cards and the
  home screen at rows of 131,072 (set `recall_128k`, recipe `m21_mix_xl128.json`, built on Odra 04:48–04:53 UTC).
- **2026-11-27 04:44 UTC, the 256k mix on AWS:** the long sources of m21mix_v2 copied to `data/m21mix_xl1/` (64 files,
  7.2 GB, server side from the laptop); a CPU node (`m21xl1-data`, c7i.8xlarge `i-036d239be4b2a713c`, 3 h cap)
  builds `recall_256k` at full scale and writes mix.json. Jobs: `infra/aws/jobs/m21xl1_data_build.sh` and
  `infra/aws/jobs/e41_long.sh` (e33n_run.sh's `long` phase on that mix from a finished model: the run's folder is
  seeded with it as both stage 1 and stage 2).
- **2026-11-27 04:48 UTC, wide summaries, first reading: the writer's rate was far too high.** Held-out loss at
  step 286: 1.9413 (control 1.9275, closed-book rows alone 1.9295); training loss on the same rows +0.022 above
  the closed-book rows alone, already +0.017 over the first ten steps. A zero-initialised map trained by AdamW at
  6e-4 moves every weight by about its rate on the first steps whatever the gradient, and this one feeds every
  summary's value. Stopped 04:49 UTC and relaunched with 5e-5 for the writer (`tree_wide_lr5e-5`, Odra card 1,
  04:49:41 UTC).
- **2026-11-27 04:51 UTC, run 3 stopped by hand** (the reasoner with key and value corrections and both losses,
  no plan): held-out loss 2.3304 and 2.3244 at steps 200 and 400 against run 2's 2.3077 and 2.2616, nearly flat,
  so the 2.227 mark at step 1,000 was out of reach an hour before the watcher would have tripped it. Speed 62.7k
  tokens/s (0.90 × the recall run). $10.29. Its next-word loss on training rows is only about 0.02 above run 2's
  on the same steps while its held-out loss (the normal read) is 0.06 above: the corrections help or do not hurt
  the closed-book rows and cost the normal read. The home retrofit of the same arm was within 0.003 of the
  closed-book rows alone; the difference is the rates (at home the token layers train 100 times slower, so the
  extra losses barely move them). Two runs with a reasoner have now failed this mark and the run without one
  passed it: the reasoner as built is out of the recipe. Run 2 at steps 1,200 and 1,400: 2.1732, 2.1632.
- **2026-11-27 04:52 UTC, the top-down read's own backward** (commit `015b8bbc`): a gather's backward allocates
  a tensor the size of its source, so every chunk of queries paid for the whole row several times and a training
  step's cost grew with the square of the row again. Now a chunk is computed without a graph, recomputed alone
  for its backward, and its gradients to the row's keys, values and nodes go through one buffer per tensor. 64k
  training rows on one RTX 3090: 16–31 s a step by the budget drawn (23–39 s before; the whole read through the
  flex kernel 19 s).
- **2026-11-27 04:53 UTC, the home screen of the long stage** (Odra card 0, `Cache/e41/retro/tree_xl128`): arm `tree_xl` (the top-down read; 80 % of the steps a
  bounded read, 20 % cut; evaluation at 64 followed per level by descent) from the released weights on rows of
  131,072 tokens (mix `m21xl128_home`: books 629 rows, government publications 306, academic books 220, recall rows
  153, packed PDFs, code and papers), 360 steps of one row, constant rates Muon 3e-4 and AdamW 2e-5, evaluation
  every 60 steps. A 128k training row through the top-down read: 34–37 s and 16.6 GB on one RTX 3090 (3.9k
  tokens/s, the same per token as at 64k; the table read took 62–65 s and 20.6 GB). First ten steps: loss 1.545.
- **2026-11-27 04:53 UTC, the 256k mix is on S3** (`data/m21mix_xl1/`, built in 9 minutes): books 386 rows, government
  publications 435, academic books 217, `recall_256k` 305 rows (64 held out), packed PDFs 4,574, code 2,287, papers
  1,143, 64k multi-hop 229 and lookup 114; every source covers its share of 0.3B tokens.
- **2026-11-27 05:12 UTC, run 4 launched: the stage on 256k rows** from the recall run's final weights (its folder
  seeded with them as stages 1 and 2): arm `tree_xl`, mix `m21mix_xl1`, 0.3B tokens = 286 steps of four rows
  (1.05M tokens a step, twice the 64k stage's), the rate starting at 1e-3 as the 64k stage did; branch `e41-run4` at
  `d4c58cce`, S3 `runs/e41/e33n_m120t_tree_xl_muon0.01-xl_tok8.0e9_seed42`, spot g6e.12xlarge
  `i-075b008f5141b1f63` (us-east-1d, 14 h cap); "long start" 05:14:19 UTC. Keeper `Cache/keepers/e41-xl.sh`, watcher
  `Cache/keepers/e41-xl-watch.sh` (started by `Cache/ops/start_run4.sh`; no loss mark, money cap $150). After the
  tests the job hides one number in 64k … 1M tokens (`scripts/e41_long_needle.py`, 8 depths) for the trained model
  and, at 256k and 1M, for the model it continued.
- **2026-11-27 05:16 UTC, the released model's needle through the top-down read** (8 depths a length; all seven
  digits right / first token right; Polonez card 3):

  | text | 16 followed per level | 64 followed per level |
  |---|---|---|
  | 64k | 0 of 8 / 8 of 8 | 3 of 8 / 8 of 8 |
  | 128k | 0 of 8 / 7 of 8 | 2 of 8 / 8 of 8 |
  | 256k | 0 of 8 / 6 of 8 | 0 of 8 / 6 of 8 |

  (The whole read at 64k: 3 of 4.) It finds where the number is at every length (the first token) and then loses
  it: the choice by a node's own estimate was never trained, and the later digits need the same tokens opened again.
- **2026-11-27 05:07–05:18 UTC, two more speed changes to the top-down read** (commits `d4c58cce`, `51ebb281`):
  a gradient-checkpoint recomputation of the layer takes the forward pass's read output (it is hit on every call,
  but saved little: the cost is in the backward); and a chunk holds several whole blocks of queries, each read
  against its own window and its own last nodes (fewer, larger operations). 64k training rows on one RTX 3090:
  11–12 s a step, 20 s with 64 followed per level (16–17 s and 29 s before; the table read 19 s); a 64k forward
  4.0 s, a 32k forward 1.7 s. Run 4 has the first change, not the second.
- **2026-11-27 05:22 UTC, run 2 against the recall run, every evaluation so far** (held-out loss, the normal read): steps 200 … 1,800: −0.000, +0.006, +0.011, +0.008, +0.009, +0.008,
  +0.009, +0.007, +0.010; its training loss stays 0.05 above on the same steps (the closed-book rows). The gap is
  steady, not closing: closed-book rows through all of stage 2 cost the normal read about 0.008. The first pass
  mark (64k documents 0.010 better than the recall run) is not in reach of this recipe; its case is the bounded and
  closed-book reads.
- **2026-11-27 05:28 UTC, streamed reading** (`PerceiverARLM.hidden_states_streamed`, commit `9fb01b7d`): a row too
  long for one forward is read in windows (default 65,536 tokens, each starting 8,192 before the last one's end).
  One forward keeps every layer's states of the whole row (4.2 GB at 98k, about 33 GB at 1M); only the reading
  layer looks further back than about 4k tokens, and what it needs of the past is its keys and values (64 numbers
  each per token and loop pass), so those are carried from window to window and the top-down read runs over the
  whole row. Tiny model: the single forward's states to 1e-6 once the overlap covers the local layers' reach (no
  overlap: off by 3.0). 150M model, fp32: states within 0.5 % on average and the same next token at 99.6–100 % of
  positions (bf16: 96 % on random tokens): the remainder is discrete choices (which concept a key joins, which
  nodes are followed) turning rounding differences into a few different reads, not the overlap (2,048 and 8,192
  give the same). `scripts/e41_long_needle.py --stream_from N` reads long rows this way.
- **2026-11-27 05:43 UTC, home screens finished:** key and value corrections without a plan (`tree_kv`) 1.9291,
  1.9267, 1.9256, 1.9242, 1.9225 at steps 286 … 1,430: the lowest of the reasoner screens at the end (value
  corrections 1.9233, the plan without extra losses 1.9240). Wide summaries at 5e-5: 1.9311 at step 286, training
  loss +0.0005 against the closed-book rows alone over 230 steps (no harm, no gain yet).
- **2026-11-27 05:50 UTC, run 4's first steps:** 15.1k tokens/s on four L40S at rows of 262,144 (a step of four rows
  in 70 s); training loss 2.11, 2.18 over the first 20 steps; expected end of training about 10:50 UTC.
- **2026-11-27 05:55 UTC, the first sign that long rows teach the read** (the 128k home screen after 50 steps,
  6.5M tokens; one number hidden at 8 depths, all seven digits right):

  | text | released model, 64 per level | after 50 steps, 64 per level | 16 per level (both) |
  |---|---|---|---|
  | 64k | 3 of 8 | 6 of 8 | 0 of 8 |
  | 128k | 2 of 8 | 1 of 8 | 0 of 8 |

  Its held-out loss at step 60 is 2.566 (its own rows; no baseline yet). 512k on the released model: 0 of 6.
  A second screen at the cloud run's rate (Muon 1e-3, AdamW 6e-5, constant) is queued on Polonez card 0 for
  06:03 UTC, after the 20-minute cooldown (`Cache/e41/retro/tree_xl128_lr1e-3`).
- **2026-11-27 06:06 UTC, why walking down the tree loses single facts** (the 128k screen's first checkpoint, 50 steps, on Polonez: 64k 7 of 8, 128k 2 of 8, 256k 0 of 8 with 64
  followed per level). With 64 followed per level a query follows half of the level-3 nodes of a 64k text, a
  quarter at 128k and an eighth at 256k, and the needle's success follows those shares. A node of thousands of
  tokens is a mean over them: one sentence moves its key by a fiftieth of the sentence's own match, less than the
  spread between nodes, so above the block level the choice by a node's own estimate is close to chance for a
  single fact. A concept (about 8 tokens) keeps it. So the walk down is the right bound for *summaries* and the
  wrong tool for *finding tokens*.
- **2026-11-27 06:10–06:20 UTC, `tree_scan` built** (commits `c450ff67`, `161e71e2`, arm `tree_xl_scan`): the
  tokens are opened from the best concepts of the whole row, the choice the concept read makes (every closed
  concept scored; in the top-down read one matrix product per head with a running maximum, the variance term
  folded into the product: 8 numbers per token of the row, no table); the followed nodes bound the summaries only.
  In the table, streamed, top-down and cached reads; the top-down read equals the streamed one in the same 18
  cases with the scan on. Its cost per query grows with the row (unlike the walk down) with a small constant; an
  index over quantised concept keys (the NCM ladder's codes as an inverted file) is the way to make it flat later.
  The released model with it, untrained (8 depths, all digits / first token): 64k 3 of 8 / 3 of 8 (the walk down:
  3 of 8 / 8 of 8); 128k 4 of 8 / 8 of 8 (2 of 8 / 8 of 8). Eight needles do not separate the two.
- **2026-11-27 06:25 UTC, Polonez: a kernel fault.** `general protection fault ... pagecache_get_page` inside an
  ext4 write by my second 128k screen (started 06:03:55 UTC on card 0), at about its first checkpoint save. Every
  GPU job on the machine stalled (all cards at 0 %), the needle test on card 3 is in uninterruptible state and
  cannot be stopped, and at 05:21 UTC a CPU-only data build had segfaulted there. Cards were at 37–64 °C. My other
  processes there were stopped by process id (06:32 UTC) and nothing new is started on Polonez until the author
  has looked (reboot, memory check). Lost: the scan needle test at 256k, the higher-rate screen (22 minutes).
- **2026-11-27 06:33 UTC, the two reads side by side at 128k rows on Odra:** the walk down (`tree_xl128`, card 0,
  since 04:53 UTC, a step in about 17 s: it ends about 07:40 UTC) and the scan (`tree_xl128_scan`, card 1, the same
  rows and rates, started 06:33:13 UTC). The screen at raised rates was stopped for it after one evaluation (the
  closed-book rows alone at Muon 3e-3: held-out loss 2.1625 at step 120, from 1.93: that rate throws the finished
  model far off, so the screen needs a gentler one). Every checkpoint (each 50 steps, and step 0) gets the needle
  test at 64k and 128k with 16 depths on card 2 (`scripts/e41_ck_watch.sh`; results `Cache/e41/evals/tree_xl128*/
  needle.log` on Odra).
- **2026-11-27 06:52 UTC, run 5 launched on demand: the 256k stage with the scan** (arm `tree_xl_scan`; everything else run 4's: the recall run's final weights, mix `m21mix_xl1`, 0.3B
  tokens). The account's spot quota for these cards is two nodes (both in use) and its on-demand quota is eight, so
  this one runs on demand ($10.49/h; no capacity in any zone at 06:41 UTC, a node at 06:52 UTC): g6e.12xlarge
  `i-00df522eef49da934` (us-east-1c, 14 h cap), branch `e41-run5` at `850bcc20`, S3
  `runs/e41/e33n_m120t_tree_xl_scan_muon0.01-xl_tok8.0e9_seed42`, keeper `Cache/keepers/e41-xls.sh`, watcher
  `Cache/keepers/e41-xls-watch.sh` (cap $180), started by `Cache/ops/start_long.sh`. Its scan scores in bf16 (the
  branch predates the change to fp32 below).
- **2026-11-27 06:52 UTC, the AWS login on the laptop expired** (SSO token; `aws sso login --profile m21-research`
  is the author's to run; a phone notification was sent 06:57 UTC). The three nodes train on (each ends its own
  job and has a hard cap in hours: run 2 30 h, runs 4 and 5 14 h); until the login is back the keepers cannot
  relaunch an interrupted spot node and the watchers cannot read S3 or enforce the money caps. Run 5's launch
  line was followed by a failed wait for the node (the token had just gone): the node is assumed up.
- **2026-11-27 06:39–06:55 UTC, needle readings with 16 depths** (Odra card 2, all seven digits right; the card
  sits at 78 °C next to card 1 and the heat guard keeps pausing it, so these come slowly):

  | model | 64k | 128k |
  |---|---|---|
  | the walk-down screen after 50 steps | 9 of 16 | 3 of 16 |
  | the walk-down screen after 100 steps | 8 of 16 | 2 of 16 |
  | released weights with the scan, untrained | 5 of 16 | 1 of 16 |

  The untrained scan at 128k was 4 of 8 on Polonez with other needles: eight or sixteen needles do not carry a
  comparison; the trend over checkpoints has to.
- **2026-11-27 06:57 UTC, the scan's scores in fp32** (commit `1a5d5772`): the choice is among tens of thousands
  of concepts and bf16 steps are 0.125–0.25 at these scores (the table read scores in fp32 too).
- **2026-11-27 07:00 UTC, `tree_span` (experimental, the top-down read's forward only; commit `59ebacd6`):** an
  opened token brings the n tokens after it (each position once, the same cap, inside the document and before the
  window). Why: the released model finds where the number starts at every length (first token right in 6–8 of 8
  up to 256k) and then loses the digits. Each later digit must win the choice again by a weak cue (the digit
  before it, a common pattern in 128k of text), while the place itself keeps matching the question; a fact is a
  run of tokens, so the index should find the place and the reader read on from it. Zero-shot test on the
  released weights queued on Odra card 2 (16 depths, 64k and 128k, a run of 4).
- **2026-11-27 07:02 UTC, the baseline on the same 16 needles:** the released weights through the walk-down read (64 followed per level) find the number in 2 of 16 at 64k
  and 0 of 16 at 128k. So 50–100 steps on 128k rows took the walk-down read from 2 to 8–9 of 16 at 64k and from
  0 to 2–3 of 16 at 128k; the scan, untrained, starts higher than the walk down (5 and 1 of 16).
- **2026-11-27 07:31 UTC, the scan learns the long lookup much faster than the walk down** (the two 128k screens, the same rows and rates, the same 16 needles; all seven digits right, and
  in brackets the first token right):

  | | 64k | 128k |
  |---|---|---|
  | walk down, untrained | 2 (9) | 0 (16) |
  | walk down, 50 steps | 9 (14) | 3 (16) |
  | walk down, 100 steps | 8 (14) | 2 (16) |
  | scan, untrained | 5 (8) | 1 (15) |
  | scan, 50 steps | **14 (14)** | **6 (16)** |

  After 6.5M tokens of 128k rows the scan variant reads the whole number in 14 of 16 texts at 64k and 6 of 16 at
  128k. Run 5 (the scan at 256k) is the run to watch; run 4 is its control.
- **2026-11-27 07:30 UTC, `tree_span` does nothing without training:** a run of 4 tokens after every opened token,
  on the released weights: the walk down 1 of 16 at 64k and 0 at 128k (2 and 0 without it), the scan 1 and 1 (5 and
  1 without it). Fewer concepts fit the cap and the model has never read such runs. Parked: it would need its own
  training screen, and the scan variant already learns the later digits.
- **2026-11-27 07:54 UTC, the two 128k screens, later checkpoints** (16 needles, all seven digits right):

  | steps | walk down, 64k | walk down, 128k | scan, 64k | scan, 128k |
  |---|---|---|---|---|
  | 0 | 2 | 0 | 5 | 1 |
  | 50 | 9 | 3 | 14 | 6 |
  | 100 | 8 | 2 | 12 | **11** |
  | 150 | 7 | 3 | | |
  | 200 | 10 | 2 | | |

  Held-out loss of the screens' own rows at steps 60, 120, 180: the walk down 2.5663, 2.5610, 2.5631; the scan
  2.5554, 2.5497 (0.011 lower at both). The walk down stays at 2–3 of 16 at 128k through 200 steps; the scan is at
  11 of 16 after 100. The walk-down screen was stopped at step 200 (07:54 UTC, its question answered) and card 0
  runs the scan screen's step-150 checkpoint through longer tests: 32 needles at 64k, 128k and 256k in one
  forward, 8 needles at 512k and 1M read window by window, then 256k again window by window on the same needles
  (`Cache/e41/evals/scan_ck150/` on Odra).
- **2026-11-27 08:24 UTC, the number is found at 256k** (the scan screen's step-150 checkpoint: the released weights after 150 rows of 131,072 tokens, 19.7M
  tokens; 32 needles a length, one forward, the model's own read: 64 followed per level for the summaries, every
  concept scored for the tokens; Odra card 0, peak 8.7 GB):

  | text | all seven digits | first token |
  |---|---|---|
  | 64k | 29 of 32 | 29 of 32 |
  | 128k | 22 of 32 | 32 of 32 |
  | 256k | **22 of 32** | 31 of 32 |

  The released model was at 0 of 8 at 256k in every read, and no row here was longer than 131,072 tokens. The
  screen's checkpoints on 16 needles (64k / 128k): 5 / 1 untrained, 14 / 6 at step 50, 12 / 11 at 100, 14 / 10 at
  150, 16 / 12 at 200; its held-out loss 2.5554, 2.5497, 2.5511 at steps 60, 120, 180. The AWS login came back at
  08:00 UTC (the author); all three nodes, keepers and watchers had survived the gap: run 4 at step 140 of 286
  (held-out 2.4528 at step 100), run 5 at step 50 (16–19k tokens/s, the scan costs no speed), run 2 at step 2,980
  of stage 2 (2.1041 at step 2,800).
- **2026-11-27 09:25 UTC, a million tokens on a home card** (the same step-150 checkpoint, read window by window: 65,536-token windows, 8,192 overlap, the far read's
  keys and values carried along; one RTX 3090, peak 5.9 GB; 8 needles a length):

  | text | all seven digits | first token | seconds a text |
  |---|---|---|---|
  | 512k | 5 of 8 | 8 of 8 | 122 |
  | 1M | **3 of 8** | 8 of 8 | 327 |

  It finds where the number is in every text at 1M tokens, eight times the longest row it has trained on, and
  reads all of it in 3 of 8. The screen's later checkpoints on 16 needles (64k / 128k): 13 / 10 at step 250,
  14 / 9 at 300: 128k has levelled at 9–12 of 16 since step 100.
- **2026-11-27 08:25 UTC, run 4's watcher left on one failed S3 read** ("no long log 3 h after the start": its
  speed was never read because the stage logs fewer than 33 lines, and the rule then treated a single failed read
  as a missing log). Fixed (`infra/aws/watch_run.sh`: only when the log was never read; `WATCH_SPEED_LINES` for
  short stages, commit `27b804eb`); the watchers of runs 4 and 5 restarted 08:31 UTC.
- **2026-11-27 10:16 UTC, the lookup was bought with 0.018 on every read** (the scan screen's final weights, 360 rows of 131,072 tokens at Muon 3e-4 and AdamW 2e-5, constant:
  three times the released model's final rates; the card's long-document test, loss per token on the last 4,096
  tokens, 120 documents at 16k and 55 at 64k, against the released model on the same documents; Odra card 1):

  | read | released, 16k | scan screen, 16k | released, 64k | scan screen, 64k |
  |---|---|---|---|---|
  | the model's own read of the whole document | 2.1257 | 2.1444 (+0.019) | 2.2933 | 2.3090 (+0.016) |
  | every node followed (the released model's read) | 2.1257 | 2.1461 (+0.020) | 2.2933 | 2.3098 (+0.017) |
  | the last 1k tokens only | 2.2200 | 2.2385 (+0.019) | 2.4410 | 2.4597 (+0.019) |
  | concept summaries only | 2.3323 | 2.3373 | 2.5291 | 2.5287 |
  | no far read | 2.3968 | 2.4105 | 2.5802 | 2.5946 |

  The loss is the same 0.018–0.019 with a kilobyte of context as with the whole document, so it is not the far
  read: the token layers drifted off their optimum at a rate three times the one they had settled at (what far
  context is worth is unchanged: 0.151 at 64k against 0.148). The screen at ten times that rate lost 0.23 in 120
  steps, the retrofits at the final rates lost nothing. The bounded own read costs nothing against every node
  followed (−0.002 / −0.001). The screen's held-out loss ended at 2.5367 (2.5554 at step 60); its needle on 16
  needles levelled at 12–14 of 16 (64k) and 9–10 of 16 (128k) from step 250 on.
- **2026-11-27 10:16 UTC, streamed reading checked on the task:** the step-150 checkpoint at 256k, the same 32
  needles, in one forward and window by window: 22 of 32 both ways (first token 31 of 32 both ways).
- **2026-11-27 10:18 UTC, the same screen at the released model's final rates** (Muon 1e-4, AdamW 6e-6:
  `Cache/e41/retro/tree_xl128_scan_lr1e-4`, Odra card 1, 360 rows; needle at 64k, 128k and 256k at every
  checkpoint on card 2): does the lookup still come, without the drift. Next on card 0: only the reading layer's
  attention training (arm `tree_xl_scan_read`, `--train_only_params layers.3.attn.`) at the first screen's rates.
  The two 256k cloud runs end at a tenth of their starting rate (1e-4, the released model's final rate), so their
  own no-harm reading is the one that counts for them.
- **2026-11-27 10:23 UTC, the scan screen's final weights on 32 needles:** 64k 26, 128k 23, 256k 14 (first token 27, 32, 30); the step-150 checkpoint had 29, 22, 22. The lengths it
  trained at hold; the length beyond them fell back (22 to 14 of 32) with 210 more rows at that rate. Its 512k and
  1M tests were dropped to free card 0 for the next screen: only the reading layer's attention trains
  (`tree_xl_scan_read`: 1.72M of 141.5M parameters; the first screen's rates, rows and order;
  `Cache/e41/retro/tree_xl128_scan_read`, started 10:23:25 UTC), needle at 64k, 128k, 256k at every checkpoint.
- **2026-11-27 10:42 and 12:11 UTC, the two 256k cloud runs finished training** (286 steps of four rows of
  262,144 tokens from the recall run's final weights; Muon from 1e-3 to 1e-4, AdamW from 6e-5 to 6e-6). Held-out
  loss on the 256k rows at steps 100 and 200: run 4 (the walk down chooses what is opened) 2.4528, 2.4349; run 5
  (every concept of the row scored) 2.4305, 2.4144.
- **2026-11-27 13:31 UTC, their tests on the nodes, against the recall run's own** (the model each continued; the
  same scripts, files `long_evals/` under each run on S3; copies in `Cache/e41/aws/`):

  | loss per token on whole held-out documents (five sources, 8 rows each) | recall run | run 4, walk down | run 5, every concept scored |
  |---|---|---|---|
  | 4k | 2.0655 | 2.0688 (+0.003) | 2.0676 (+0.002) |
  | 16k | 1.8899 | 1.8940 (+0.004) | 1.8912 (+0.001) |
  | 32k | 1.9040 | 1.9114 (+0.007) | 1.9069 (+0.003) |
  | 64k | 1.8811 | 1.8916 (+0.010) | 1.8840 (+0.003) |
  | 128k | 1.7746 | 1.7713 (−0.003) | **1.7567 (−0.018)** |

  | RULER, mean of 13 tasks, 100 documents each | recall run | run 4 | run 5 |
  |---|---|---|---|
  | 4k | 0.572 | 0.528 | 0.555 |
  | 8k | 0.526 | 0.467 | 0.508 |
  | 16k | 0.405 | 0.377 | 0.398 |
  | 32k | 0.360 | 0.304 | 0.362 |

  Core benchmarks: level within a point on all seven for both (ARC-Easy 0.491 / 0.482 / 0.487, HellaSwag 0.301 /
  0.299 / 0.299, PIQA 0.625 / 0.633 / 0.626, WikiText bits per byte 0.964 / 0.967 / 0.966); LAMBADA perplexity
  50.3 / 53.0 / 52.7.
  Run 5 keeps the ordinary reading to 64k within 0.003 and reads 128k documents 0.018 better (the last quarter of
  the document: 1.7000 against 1.7367); RULER is level on the mean, with the tasks moved: one planted number among
  distractor numbers down (16k: 0.44 to 0.11; 32k: 0.59 to 0.38), the second key of several at 8k down (0.62 to
  0.38), several queries and several values up (32k: 0.155 to 0.280 and 0.378 to 0.400). Run 4 is behind on both
  tables: the walk down costs short-range reading as well as the long lookup.
- **2026-11-27 13:33 UTC, both nodes stopped by hand** (`i-075b008f5141b1f63` spot, `i-00df522eef49da934` on
  demand; keepers and watchers stopped by PID). Every test above was done; what still ran was the 4k check of
  cached decoding against the old loop, which takes a top-down model 13.5-16.5 s per request (829 and 613 of 1,300
  done, 2 h and 2.5 h left) on one card of a four-card node. Spend: run 4 $64.5, run 5 $69.7. The final weights
  are on S3 and were fetched to Odra through signed links (`Cache/ops/to_odra.sh`, `Cache/e41/aws_models/`); the
  planted-number tests run there (`scripts/e41_needle_battery.sh`). Fix: `E37_EVAL_NO_OLDLOOP=1` in
  `infra/aws/jobs/e37_evals_long.sh`, set by `e41_long.sh`, whose needle part is now optional (commit `8712585c`).
- **2026-11-27 13:46 UTC, the two follow-up screens on 128k rows ended** (360 rows each, Odra; 16 numbers per
  checkpoint at 64k / 128k / 256k):

  | checkpoint | the released model's final rates (Muon 1e-4, AdamW 6e-6) | only the reading layer's attention trains (3e-4 / 2e-5) | first screen, all weights at 3e-4 / 2e-5 |
  |---|---|---|---|
  | 50 | 9 / 9 / 10 | 5 / 9 / 8 | 14 / 6 / - |
  | 100 | 12 / 10 / 7 | 9 / 9 / 5 | 12 / 11 / - |
  | 150 | 10 / 10 / 9 | 8 / 9 / 7 | 14 / 10 / 22 of 32 |
  | 200 | 14 / 9 / 11 | not tested | 16 / 12 / - |

  Held-out loss on the 128k rows, last evaluation: 2.5333 (final rates), 2.5449 (reading layer only), 2.5367
  (first screen). The lookup also comes at the final rates, more slowly, and with the lowest held-out loss of the
  three; the reading layer alone learns it least, so the token layers' queries and keys are part of what has to
  move. The reading-layer screen's later checkpoints were not tested (card 2 given to the tests below).
- **2026-11-27 13:41 UTC, run 6 queued** (`e41-xls3`, branch `e41-run6` at `8712585c`, spot): run 5 with one
  change, the starting rate a third of run 5's (Muon 3e-4 to 3e-5, AdamW 1.8e-5 to 1.8e-6: where the first home
  screen learned the lookup, and decaying so the ordinary reading does not drift). Why: run 5 reads 16 of the
  first 32 planted numbers at 64k and the first home screen 29 of 32; the cloud runs started at ten times the
  released model's final rate, and their RULER lookups at 8k-32k fell where the recall run was strong. The node
  stops with its tests (no needle part, no old-loop check). Waiting for a spot slot: the stopped node still counted.
- **2026-11-27 13:52:53 UTC, run 6 launched** on a spot g6e.12xlarge (`e41-xls3`, keeper and watcher in
  `Cache/keepers/`, cap $180).
- **2026-11-27 14:05 UTC, planted numbers of the two cloud models** (Odra cards 0 and 1,
  `scripts/e41_needle_battery.sh`: 32 seven-digit numbers per length, each text in one forward, the model's own
  read; `Cache/e41/evals/aws_xls`, `aws_xl`). All tokens of the number right, and in brackets the first token:

  | model | 64k | 128k | 256k |
  |---|---|---|---|
  | run 5, cloud, 256k rows, every concept scored | 16 (16) | 14 (15) | 10 (15) |
  | run 4, cloud, 256k rows, walk down | 6 (16) | 5 (13) | 1 (20) |
  | first home screen, 128k rows, step 150 | 29 (29) | 22 (32) | 22 (31) |
  | first home screen, final | 26 (27) | 23 (32) | 14 (30) |

  Run 5 trained on rows of 256k and reads fewer planted numbers at every length than a home screen that never saw
  a row beyond 128k. The two fail differently: the home screen finds the place in nearly every text and loses a
  later token in a third; run 5 does not find the place in half the texts (first token 15-16 of 32) and reads
  nearly every number whose place it found (16 of 16, 14 of 15, 10 of 15). The released model untrained for this
  read found the first token in 6 of 8 at 256k, so run 5 finds places less often than the model it started from.
  With the fall of RULER's single planted number at 16k and 32k this points at the start of the cloud stage (Muon
  1e-3, ten times the released final rate) moving the queries and keys that find a place; the decay then
  restored the ordinary reading and not that. The 512k and 1M parts of both batteries were stopped (the cards
  were needed, and 256k already answers).
- **2026-11-27 14:06 UTC, two home screens on the rate alone** (Odra, `scripts/e41_home_screen.sh`, arm
  `tree_xl_scan`, 360 rows of 131,072 from the recall run's final weights, a 10-step warm-up and a fall to a tenth;
  16 numbers per checkpoint at 64k, 128k and 256k on card 2): `xl128_scan_d3e-4` (card 1: Muon 3e-4, AdamW 2e-5,
  run 6's rates on 128k rows) and `xl128_scan_d1e-3` (card 0: Muon 1e-3, AdamW 6e-5, run 5's rates on 128k
  rows). If the second loses the first token as run 5 did, it is the rate; if it learns the lookup like the first
  home screen, it is the 256k rows.
- **2026-11-27 14:08 UTC, queued on card 2** (`scripts/gpu_when_free.sh`, one at a time): where a missed number's
  place went (`scripts/e41_needle_why.py`: for each answer token, whether the read opened its source position) on
  the first home screen's step-150 weights and on run 5's; and the recall run's own read on the same 32 numbers at
  64k and 128k (the baseline row).
- **2026-11-27 16:08 UTC, CORRECTION to the 14:05 entry: run 5 did not lose the place, the score undercounted
  it.** The test counted a number as read only when the first predicted token was the number's first token. The
  planted sentence reads "... is: 1234567." and the question ends "... is": run 5's model says an opening quote,
  a colon, "the" or a comma first in half the texts (16 texts at 64k: ` "` 6, `:` 3, ` the` 2, `,` 1) and then
  copies every digit. `scripts/e41_needle_why.py` on run 5's final weights, the battery's own 32 texts (Odra card
  2): at 64k the number's place was opened for every answer token of every text and all 224 later tokens are
  right; at 128k the place was opened for 223 of 224 later tokens and 208 are right. Counted as RULER counts
  (the number is in what the model goes on to say; here: every token after the first right, the first given):

  | read as "all tokens after the first" | 64k | 128k |
  |---|---|---|
  | run 5, cloud, 256k rows, every concept scored | 32 of 32 | 23 of 32 |
  | first home screen, 128k rows, step 150 | 32 of 32 | 22 of 32 (its first token was right in all 32) |

  So run 5 reads planted numbers at 64k and 128k as well as the best home screen, and the 14:05 entry's reading
  ("does not find the place in half the texts", "finds places less often than the model it started from", the
  start rate as the cause) is withdrawn. What stays measured: both cloud runs' fall on RULER's lookups among
  distractors at 8k-32k (generated and string-matched, so not this artifact), and the recall run's own read on
  the old score (64k 9 of 32 with the first token right in 16; 128k 3 of 32 with the first token right in 28: its
  later tokens fail, which is a reading failure). The 15 wrong later tokens of run 5 at 128k had their place
  opened: opened and not used, not a routing miss.
  The test now reports both scores and what was said first (`digits_after_first`, `said_first_instead`; commit
  `5bd6ac85`); every earlier planted-number figure in this log is the strict score and undercounts models that
  open with a quote or colon. Re-measured with both scores on Odra card 0 (`Cache/e41/evals/fair/`): the recall
  run's own read, run 5's final and step-150 weights, run 4's final, the first home screen's step 150.
- **2026-11-27 15:41 UTC, the two rate screens' first checkpoint** (step 50, 16 numbers, strict score with the
  first token in brackets, 64k / 128k / 256k): gentle (3e-4) 10 (11) / 9 (16) / 12 (15); run 5's rates (1e-3)
  14 (14) / 13 (16) / 9 (13). Held-out loss on the 128k rows at steps 60, 120, 180, 240: gentle 2.5503, 2.5442,
  2.5416; run 5's rates 2.5916, 2.5946, 2.5676, 2.5459 (thrown 0.04 off at the start, closing as the rate falls).
  The high-rate screen was stopped by hand at step 240 (16:11 UTC): its question went with the correction above,
  and the server needed the room (below). The gentle one runs on (card 1) as run 6's forecast.
- **2026-11-27 15:01-16:12 UTC, card 2 on Odra.** (1) Two checkpoint evaluators and a queued test started on it in
  the same seconds and the test ran out of memory at 128k: the evaluators waited for an empty card without a lock.
  Fixed: `scripts/gpu_when_free.sh` holds a per-card lock while its command runs, `scripts/e41_ck_watch.sh` takes
  its card through it (commits `22409638`, `0be25c8d`). (2) With cards 0 and 1 both training, card 2 read 81 C the
  moment its job ran and 69-70 C paused: the heat guard held it at a half duty cycle but could not keep the
  reading under 80. Card 2 was left idle and its queue moved to card 0 (freed by stopping the high-rate screen);
  the guard is back at pause 78 C / resume 70 C, checked every 5 s.
- **Run 6 stands** (256k rows at a third of the start rate; step 80 at 15:43 UTC, 13-14k tokens/s): its reason is
  now only the RULER lookups among distractors and the size of the drift, not a lost lookup.

## Result
<Filled in after, by experiment-track.>
