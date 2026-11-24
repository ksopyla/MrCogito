# E40 — Depth-mixed read: DeepCrossAttention's query/key/value split for the concept read (150M retrofit)

- **Status:** asked for by the author (2026-11-24: "Asses value of it and Spin another experiment if needed to test
  the recent findings by google about DeepCrossAttention and how this can help in our concept model, we have a few
  skip attention but the concepts should be the main source of important information and now I'm not sure if they
  actually are"). Spec written 2026-11-24 ~06:45 UTC. Nothing built. Home cards only, no AWS money.
- **Serves:** the vision's claim that concepts carry the document (10M-context goal; reasoning in latent space
  needs concepts that hold content, not only pointers), and a cheap read on a 2025–2026 trunk upgrade.
- **New family:** a different object from E39 (it changes how every layer, or the reading layer, takes its input
  from depth, not how the memory is grouped). E40 was free on every branch on 2026-11-24 (all `refs/remotes/origin`
  and local heads checked).
- **Owner / dates:** Krzysztof Sopyla · opened 2026-11-24 · closed —

## Source (literature review: `docs/literature_review/depth_wise_residual_connections.md`)

DeepCrossAttention (DCA), Heddes et al., Google Research, ICML 2025, [arXiv 2502.06785](https://arxiv.org/abs/2502.06785);
Google Research blog, 2026-11-23. Each layer's input is no longer the plain residual sum of all earlier layer outputs:
a learned, per-dimension weighting of them (GRN-v2/v3), and three such mixes per layer, one each for the query, key
and value inputs of attention. Measured gains (C4, 8-DCA): 75M 27.88 → 26.44 perplexity, 179M 21.57 → 20.46,
449M 17.17 → 16.76; gains shrink with width (−2.82 at width 64, −0.39 at 1024). Ablation (6 layers, LM1B): static
per-layer weights −0.18, per-dimension −0.37, input-dependent −0.02, **the query/key/value split −0.35**. Retrofit:
adding DCA to a trained model and training 60k more steps gave −0.19 against −0.02 for the plain model. Limits: 128-token
sequences only, no long context, no downstream language tasks, 34–54 % fewer steps per second in their code, largest
model 449M, no independent replication at scale. The closest rival, MUDDFormer ([2502.12170](https://arxiv.org/abs/2502.12170)),
is the same idea tested to 2.8B.

## What we already know about our own wiring (released `concept-memory-140m-11.5bt` and its siblings)

- **The far past is used and it is the content.** 64k documents, last 4,096 tokens scored: whole document 2.293 nats
  per token, last 1k only 2.441 (gain 0.148), another document's far past 2.399.
- **The concepts' own vectors carry part of it.** Summaries off (exact reads kept) costs 0.068 at 64k and 0.023 at
  16k, so the summaries matter more the longer the document. The far read off costs 0.287 (overstated: the model never
  trained without it). Never measured: **the summaries alone, nothing opened**: how much the concepts carry by
  themselves.
- **The skips do not bypass the concepts; they amplify the read.** Trained U-net skip weights in the catalogue and
  full-attention models (concept memory's own were not read): reading layer → stack 5 +0.66 / +0.77; stack 0 →
  stack 4 −0.26 / −0.23; the other four between −0.04 and +0.10. Same pattern as DCA's learned weights (input and
  the last layers kept, middle layers pushed negative).
- **The multi-item failure was the training data, not the wiring** (E38 recall rows, 2026-11-23): one-hour
  fine-tunes took RULER several-keys at 16k from 0.11 to 0.86.

## Hypothesis

In concept memory the reader's question and the memory's keys and values are all computed from one stream, the input
of the reading layer. If the keys and values may draw on a different mix of depths than the question (early layers
for exact identity: names, numbers, code words, the things the read copies; later passes for the question), the
concept read carries more of the far past: the far-context gain rises and the read's own share of it rises.
Falsifiable: a retrofit that starts exactly at the released model and trains equal tokens against a paired control.

## Arms (each starts as the released model, identical outputs at step 0)

The depth columns are the stream increments of every earlier layer run (c_0 = the stream after the embedding,
c_j = x_j − x_{j−1}), so their plain sum is the current stream exactly, whatever α, β and the U-net skips did. A
GRN-v3 mix g = Σ_j (b_j + relu(w·c_j)) ⊙ c_j with b = 1 and w = 0 is then the identity at step 0.

1. **Control:** the released model, trained on the same rows, nothing added.
2. **Depth-mixed read:** only the reading layer (all four loop passes) gets three mixes, for its query, key and value
   inputs, over x0, L0–L2 and the earlier passes' L3/L4 increments. Everything else unchanged. About 3 × (columns × 896)
   new weights per pass, under 0.1 % of the model.
3. **Google's recipe:** 2-DCA on every layer run (query/key/value mixes over x0, the last two increments and the sum of
   the rest), as the paper's k-DCA. Tells whether the trunk gains at our width (896) and depth (13 layers, 19 runs).

Same rows, same order, same seed, same schedule (the released stage-3 final learning rate, held constant; the new
weights by AdamW at that rate), same tokens per arm: the build session picks the count that fits ≤ 12 h per arm on
one RTX 3090, at least 150M tokens, from the released recipe's stage-3 mix (`m21mix_v2`, the 16k and 64k rows).

## Pre-flight (free, before any training)

- **P1 — the concepts alone:** add a "summaries only" mode (nothing opened, every concept a summary) to
  `evaluation/long_context_use.py` and score the released model on the 64k and 16k documents next to its existing
  modes. This answers the author's question directly: how much of the 0.148 the concept vectors carry by themselves.
- **P2 — concept memory's own skip weights:** read σ, α, β of `concept-memory-140m-11.5bt` (cached on Odra).
- **P3 — identity:** each arm's logits equal the released model's at step 0 (fp32, max abs difference < 1e-4).
- **P4 — speed:** tokens per second of arms 2 and 3 against the control on one home card at 16k.

### Pre-flight results (2026-11-24, Odra, $0; build session)

Released weights: `Cache/e38_board/e37a_m120t_final` (the folder the release card was scored from). Files in Odra
`~/dev/MrCogito-e40/Cache/e40/preflight/`.

- **P1 (18:01–18:18 UTC; `long_context_use.py --modes off,nosum,sumonly`, the release settings, contexts 1k and
  whole only).** The card's numbers reproduce exactly (64k: whole document 2.293, last 1k 2.441, summaries off
  +0.068, memory off +0.287; 16k: 2.126, +0.094, +0.023, +0.271). **Summaries only** (nothing opened, every old
  concept a summary): 64k 2.529 (+0.236 [0.176, 0.306] against the whole read), 16k 2.332 (+0.207). Share of the
  memory read's benefit (memory off − whole) that the summaries carry alone, (off − sumonly) / (off − whole):
  **18 % [15, 21] at 64k, 24 % [21, 27] at 16k**. Summaries alone score worse than the last-1k cut (2.529 against
  2.441), while removing them from the full read costs 0.068: they work as a complement to the word-for-word reads,
  not as a carrier of the far past by themselves. By source at 64k (1k / whole / off / summaries only): US GPO 2.021 /
  1.642 / 2.192 / 2.098; arXiv 1.221 / 0.930 / 1.404 / 1.301; DOAB 2.300 / 2.148 / 2.593 / 2.535; books 2.987 /
  2.916 / 3.046 / 3.007; Hansard 2.593 / 2.533 / 2.650 / 2.629.
- **P2 (`scripts/depth_weights.py`).** U-net skips σ (into ← from): layer 9 ← reading layer **+0.738**, layer 8 ←
  layer 4 (looped with the reading layer) **−0.296**, layer 7 ← 5 −0.076, layers 10–12 ← 2, 1, 0: +0.024, +0.015,
  −0.015. Input mix α / β: reading layer 0.529 / +0.094, its looped partner 0.564 / +0.007 (each loop pass keeps
  ~30 % of the stream it got), layers 0–2 0.71 / −0.29, 0.53 / +0.21, 0.58 / −0.01, layers 5–12 0.66–1.05, β −0.31
  to +0.25. Loop markers' norms 1.81, 2.34, 1.64, 0.82. Same pattern as the sibling models and DCA's learned
  weights: the read's output is amplified into the upper stack, nothing routes around it.
- **P3 (18:20 UTC; `scripts/e40_identity.py`, fp32, 2 held-out rows each at 16k and 64k).** Both mixed arms equal
  the control exactly: max |logit difference| 0 (logits at every 16th position, final hidden states everywhere).
  PASS (by construction: the mixes are written as stream + Σ (b' + gate) ⊙ column, exact zeros at the start).

### Build decisions (2026-11-24, build session)

- **Component:** `nn/depth_mix.py`, `par_depth_mix: none | read | dca2` (arms `e40_read`, `e40_dca2` in
  `scripts/run_e33n.py`); cached decoding supports it (`nn/incremental.py`). Block wiring as the paper's code: the
  query mix also feeds the feed-forward input (g_q + attention); the stream continues as mix(x) + attention +
  feed-forward, so each column is a run's increment. 2-DCA columns as the paper's text (input, sum of the middle,
  last two; the unofficial code drops the middle instead).
- **The gate weight w:** the paper starts it at 0 under relu, where its gradient is 0, so it could never learn; the
  unofficial code starts it at noise (std 0.01), which breaks the exact start. Here: w = 0 and a relu whose gradient
  at 0 is 1, so the start is exact and w learns. rms-normalised columns feed the gate (as the unofficial code).
- **Data at home, no AWS:** the 16k and 64k phases' sources of `m21mix_v2` rebuilt on Odra from the public sources
  at small scales (`scripts/e40_data_build.sh`, 18:03–18:13 UTC, 602 MB): the same files in the release build's
  order and the same held-out documents, so the rows are a subset of what the release run read (a replay, the same
  for every arm). One pass holds 130M tokens of 16k rows and 62M of 64k rows.
- **Schedule:** 1,430 steps of 4 rows of 16k (93.7M tokens, the stage-2 mix), then 920 steps of one 64k row
  (60.3M tokens, the stage-3 mix), 154M tokens per arm, ending on 64k rows (a 16k-only fine-tune undid the 64k
  stage's calibration on 2026-11-23). 65,536 tokens per step, so the new weights take ~2,350 steps. Rates constant:
  Muon 1e-4 on the matrices, AdamW 6e-6 on the rest (the released stage 3's final rates).
- **Deviation: the new weights' rate is 1e-3, not 1e-4.** Adam moves a weight by at most its rate per step, so at
  1e-4 a mix that starts at 1 could move by at most 0.24 over the run, too little to learn the depth patterns the
  paper reports; 1e-3 is the released stage 3's starting matrix rate. Same rate for both mixed arms. K2 guards harm.

## Pass marks (set before running; paired differences on the same documents, 95 % bootstrap intervals)

- **S1 (the read carries more):** arm 2's far-context gain at 64k (loss with the last 1k only minus loss with the whole
  document) at least 0.010 nats above the control's, interval above zero. 0.010 is about 7 % of the gain and the size
  of the paper's own retrofit effect (−0.19 perplexity on 18.98, ~1 %).
- **S2 (it is the concepts):** arm 2's cost of switching the summaries off at 64k rises against the control's, and
  its summaries-only score (P1 mode) is better than the control's.
- **S3 (Google's recipe at our size):** arm 3's whole-document loss at 16k and 64k at least 0.010 nats below the
  control's.
- **S4 (no harm):** RULER 4k–16k 13-task mean within 0.02 of the control's; BLiMP and the 7-task core within 1 point.
- **S5 (cost):** arm 2 at least 0.95 × the control's speed. Arm 3's speed is reported, not judged.

## Kill marks

- **K1:** P3 fails after one fix → stop; the retrofit is not a clean test.
- **K2:** at the halfway point, arm 2's held-out loss is more than 0.02 above the control's → stop arm 2.
- **K3:** both arms within ±0.005 nats of the control on whole-document loss and far-context gain at the end → DCA
  does not help this model as a retrofit; record it (verdict Watch) and stop. No from-scratch run.

## If it passes

A from-scratch 150M run of the winning arm on the current recipe (the recall mix, `m21mix_v4`), compared with the E38
recall run, as its own spec revision and with the author's go-ahead (about $450 on AWS).

## Compute

Odra's three RTX 3090s, one arm per card, about 12 h, then the release tests (`scripts/release_evals.sh`, long-document
modes `off,nosum` plus the new summaries-only mode) on each arm. Polonez only if Odra is taken, at most three busy
cards (card 2 is faulty until the author reboots it). AWS: none.

## Not in scope

Input-dependent weights beyond GRN-v3's one scalar per column (the paper's ablation found them worth −0.02);
attention over depth (Kimi's Attention Residuals); any change to the concept grouping (E39 tests that).
