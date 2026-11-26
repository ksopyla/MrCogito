# E37f — Closed-book practice: train concept memory's summaries to carry the far past (150M retrofit)

- **Status:** asked for by the author (2026-11-26, "Yes, write it", answering "Write the spec for the closed-book
  practice test?" after the E38–E40 analysis). Spec written 2026-11-26 ~09:40 UTC. Nothing built. Home cards only,
  no AWS money. Waiting for the alignment page's checklist.
- **Serves:** the vision's claim that concepts carry the document, not only point at it. At 10M tokens the reader
  can reread 512 tokens, 0.005 % of the book, so everything else must come through the summaries; today they carry
  less than cutting the document.
- **ID:** a flavour of E37 (concept memory): the same model, trained with a mask on what the read may open, plus the
  E37 reasoner that already exists in the code. E37 letters in use on every branch (2026-11-26, all
  `refs/heads` and `refs/remotes/origin`): a, b, c, e, s. d was never used; f follows the last letter in sequence.
- **Owner / dates:** Krzysztof Sopyla · opened 2026-11-26 · closed —

## What we know (the E40 control: the released `concept-memory-140m-11.5bt` + 154M more tokens, 64k documents)

Nats per token on the last 4,096 tokens of 55 documents of 64k (16k: 120 documents), lower is better:

| Run of the same model | 64k | 16k |
|---|---|---|
| Whole document, as trained (512 tokens opened + summaries + window) | 2.301 | 2.128 |
| Only the last 1k tokens given (the document cut) | 2.444 | 2.221 |
| Only the last 4k tokens given | 2.392 | 2.169 |
| Another document's far past, own last 4k (the swap) | 2.406 | 2.179 |
| **Summaries only** (nothing opened; window + summaries) | **2.534** | **2.333** |
| Summaries off (opened tokens + window) | 2.364 | 2.149 |
| Far read off (sinks + window only) | 2.582 | 2.397 |

- **The far past is worth 0.142 at 64k (0.094 at 16k)**, the cut minus the whole document. All of it is this
  document's content: another document's far past is worth nothing (2.406 against 2.392 with the last 4k only).
- **The summaries alone are worse than no far past at all:** 0.091 worse than the cut at 64k, 0.112 at 16k. E40's
  "share" (18 % on the card, 16.8 % on the control) measured against the far-read-off run, a state the model never
  trained in (0.138 worse than the cut), so it flattered the summaries. This spec measures against the cut.
- **Next to the opened tokens the summaries do add something:** switching them off costs 0.063 at 64k (0.022 at 16k).
  The averages already hold about 0.06 nats the 512 opened tokens lack: the measured hint that the pass mark below is
  reachable.
- **Why they carry so little (code, `nn/concept_memory.py`):** a summary's value is μ_v, the plain mean of about 8
  members' 64-dimension values, plus the reasoner's update, and the reasoner is off in the release. Nothing learned
  writes it. Its gradient reaches the members' keys and values, but training never needs it: the best 512 tokens are
  always opened, so the loss is low without the summaries holding anything. A student who may always reread the book
  writes a lazy notebook.
- **The reasoner was tried and shelved, with the book open** (e37b, 30M, 2026-11-08): −0.0045 nats on far-copy
  tokens (bar 0.01), shelved by E37's K2. It was trained with exact reads always present, so it had nothing to add.
- **Related failures point the same way.** E39 (living concepts, 2026-11-26) let the reader open 32 of 210–880
  concepts and long text got worse: what is not opened arrives as averages. E38's recall rows confused similar keys:
  the read finds the neighbourhood, not the item.

## Hypothesis

The summaries carry little because training never needs them. If a third of the training rows open nothing and a
third open only 64 tokens, the loss on those rows can only fall through the summaries, the gradient through their
means reaches the member tokens' keys and values, and the layers below the reading layer learn to write values
whose averages hold the document. Then **summaries alone beat the document cut at 64k by at least 0.057 nats** (40 %
of the far past's 0.142), up from −0.091, while the model as normally run loses at most 0.010. A reasoner trained the
same way says whether plain averages are also too thin: if it adds at least 0.02 more, room binds as well as the
incentive. Falsifiable: a retrofit from the released weights, paired by document with E40's control.

## Arms (each starts as the released model; same rows, order, seed and schedule as E40)

1. **Control:** E40's control, reused (the released model trained on E40's 154M tokens, nothing changed). Its
   release tests are on file (`docs/2_Experiments_Registry/results/evaluations/E40_control/`). P0 checks the reuse.
2. **Closed-book practice:** each training row draws its opening budget: **nothing opened** (1/3), **64 tokens**
   (1/3), **512 tokens as trained** (1/3). Summaries, window and sinks are always on. All queries and all four passes
   of a row use the row's budget. The draw is seeded by the row's index, so arm 3 sees the same draws. Held-out
   evaluation and every test run as trained (512). No new weights.
3. **Closed-book practice + reasoner:** arm 2 plus E37's reasoner as e37b had it: two tied transformer layers of
   width 512, 8 heads, looped twice over the concepts' mean states (896 dimensions, 14 times the value's room), its
   output added to each summary's value and to the members' values (zero at the start, so step 0 is the released
   model). About 6.8M new weights (+5 %), in their own AdamW group at 6e-4 (the released recipe's AdamW peak),
   constant, no weight decay.

Why thirds: no prior favours a weighting; a third of rows stays exactly as trained (the no-harm guard), and 64
tokens practises scarcity (at 64k it is 0.1 % of the document; at 10M, 512 tokens is 0.005 %).

Schedule (E40's): 1,430 steps of 4 × 16k rows (stage-2 mix) then 920 steps of one 64k row (stage-3 mix), 154M
tokens; Muon 1e-4 and AdamW 6e-6 constant (the released stage 3's final rates); E40's home-built `m21mix_v2` rows.

## Pre-flight (free, before training)

- **P0 — the control is reusable:** E40's control weights and data rows are still on Odra; the first 50 steps of a
  fresh control reproduce E40's training losses (within 1e-3). If not, the control is retrained first (+9 h).
- **P1 — the small-budget baseline:** score the control at 64k and 16k with 64 tokens opened (a new test switch,
  `open64`) and summaries only, so S2 has its reference.
- **P2 — identity:** arm 3's logits equal the released model's at step 0 (fp32, max abs difference < 1e-4); arm 2's
  first 512-budget row gives the control's step-0 loss.
- **P3 — speed:** tokens per second of arms 2 and 3 against the control at 16k, including the recompiles that
  switching budgets may cause.

## Pass marks (set before running; paired by document against the control, 95 % bootstrap intervals)

- **S1 (the notebook carries the document; decides success):** arm 2's **notebook value** at 64k (loss with the last
  1k only minus loss with summaries only) at least **+0.057 nats**, interval above +0.03. The control: −0.091. Ceiling:
  0.142, the whole far past as read with opened tokens. (In E40's share this is about 69 %; a 40 % share would still
  be worse than the cut.)
- **S2 (scarce budgets degrade gently):** at 64k with 64 tokens opened, arm 2 closes at least half of the control's
  gap between 64 and 512 tokens opened (judged only if that gap is at least 0.02).
- **S3 (room binds too):** arm 3's notebook value at 64k at least 0.020 above arm 2's, interval above zero.
- **S4 (no harm, as normally run):** 64k and 16k whole-document loss at most 0.010 above the control's; RULER 4k–16k
  13-task mean within 0.02; BLiMP and the 7-task core within 1 point.
- **S5 (cost):** arm 2 at least 0.95 × the control's speed. Arm 3's speed is reported, not judged.

## Kill marks

- **K1:** P2 fails for arm 3 after one fix → stop arm 3; arm 2 goes on.
- **K2:** at the halfway point (step 1,144 of the 16k part, 75M tokens) an arm's held-out loss, run as trained, more
  than 0.02 above the control's (1.9228) → stop that arm.
- **K3:** at the end both arms' notebook value at 64k below +0.010 → plain or reasoned averages of about 8 tokens
  cannot hold the document even when training needs them; record it and stop.

## Reading the result

- S1 met, S3 missed: the missing piece was the incentive and averages suffice at this size → closed-book practice
  goes into a from-scratch 150M run on the recall recipe (`m21mix_v4`) against the E38 recall run (about $450, with
  the author's go-ahead), then the compression ladder.
- S1 and S3 met: the reasoner joins that run.
- S1 missed but the notebook value clearly positive, S3 met: room binds → next bet a learned writer with room scaled
  to content (one summary per fixed number of tokens).
- K3: the notebook cannot be averages → the learned writer, or an attention-reconstruction loss (below), is next.

Readouts, no marks: notebook value at 16k and 128k (15 documents); summaries-off cost as trained; E40's share;
RULER per task (several keys at once, similar keys); the attention share on summaries as trained; concept sizes.

## Sources

- AutoCompressors ([2305.14788](https://arxiv.org/abs/2305.14788)) and gist tokens ([2304.08467](https://arxiv.org/abs/2304.08467))
  train summary vectors by making them the only access to the past: summaries carry content when training forces the
  model to use them. Reviewed in `docs/literature_review/learned_kv_context_compression.md`.
- Compressive Transformer ([1911.05507](https://arxiv.org/abs/1911.05507)): mean pooling beat a compressor trained
  only by the language-model loss (0.982 against 0.996 bits per character); an attention-reconstruction loss was
  best (0.973). Our means are a fair start; arm 3's learned update could hurt for the same reason.
- Gist-token failure analysis ([2412.17483](https://arxiv.org/abs/2412.17483)): compression loses surprising,
  distinct items first ("lost if surprise"), the names and numbers an average blurs.
- Hypothesis, not a claim: in hyperdimensional computing a sum of k random d-dimension codes keeps each member
  recoverable while k is small against d. Eight members in 64 dimensions is near the edge, so the encoder must learn
  to write values that survive averaging.

## Why this is not a retread

E10–E17 made concepts the only channel from scratch, and under next-word loss the model routed around them. Here
the read is trained and worth 0.142 nats to this model, closed-book rows are a third of training, not the only
regime, and the gradient lands in the members' keys and values through the means. e37b's reasoner had the book
always open. E40 changed the wiring; this changes what training asks of the summaries.

## Compute

Odra, two cards at a time (the heat rule: card 2 cannot stay under 80 °C next to a busy card 1), arms 2 and 3 side by
side with `scripts/heat_guard.sh`: about 9 h for arm 2 (the control took 8.5 h), 10–14 h for arm 3 (P3 decides),
then the release tests of each (`scripts/release_evals.sh`, long-document modes `off,nosum,sumonly,open64`). About
one day. Polonez only if Odra is taken (at most three busy cards). AWS: none.

## Plan

- **Launch:** E40's retrofit launcher and pipeline (`scripts/e40_train.sh`, `scripts/e40_pipeline.sh`) with arms
  `e37f_closed` and `e37f_reason` added to `scripts/run_e33n.py`'s arm table, outputs under `Cache/e37f/`, the
  control read from `Cache/e40/runs/e37a`.
- **New foundation code (reusable, config-selectable):**
  1. `--concept_open_mix "0:1,64:1,512:1"` — training rows draw the concept read's opening budget (0 = nothing
     opened, the `sumonly` path); one draw per micro-batch (one row at micro 1, as both parts run), seeded by the
     row's index; evaluation always as configured.
  2. `--concept_reasoner_lr` — the reasoner's own AdamW group in a Muon run (as `--depth_mix_lr`); the warm start
     accepts the reasoner's weights as missing (built with zero output).
  3. `evaluation/long_context_use.py` mode `open<k>` (opening budget k at test); `scripts/e40_longctx_summary.py`
     adds the notebook value (cut 1k − summaries only) next to the share.
- **Not in scope:** an attention-reconstruction loss; a learned writer; more concepts per block (fewer tokens per
  summary: a room knob, a follow-up); a from-scratch run (only if S1 passes, with the author's go-ahead).

## Result
- Run report: —
- Verdict: —
