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
