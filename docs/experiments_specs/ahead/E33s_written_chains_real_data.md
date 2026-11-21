# E33s — Written reasoning from scratch: a 30M confirmation, then 100M against dense (real data deferred)

- **Status:** re-scoped 2026-10-24, draft awaiting the author's go-ahead. Asked for by the author on 24 Oct: "plan
  the 30M from-scratch confirmation (written chains for every task from step 0, exit losses off, 3 seeds + dense on
  the same data) plus the small code fixes, then the 100M comparison. Show me the plan and cost before launching
  anything." The real-data design of 21 Oct is kept below as Phase C, deferred.
- **Serves:** the Vision (reason in latent space over compressed memory) and the author's 18 Oct direction:
  methods that need no generator labels, only available text, instruction and reasoning data.
- **Implementation plan:** [E33s_written_chains_real_data_plan.md](E33s_written_chains_real_data_plan.md) (section 11
  is the re-scope plan; sections 1–10 the real-data design and its pre-flight)
- **Alignment page:** [e33s_architecture.html](../../3_Evaluations_and_Baselines/e33s_architecture.html) (real-data
  version, with the 23 Oct hold notice)
- **Owner / dates:** ksopyla · opened 2026-10-21 · re-scoped 2026-10-24

## Re-scope, 24 Oct 2026: the text world first, from scratch

### Why
- **Real multi-hop QA cannot resolve the 100M gates.** In the 22 Oct ceiling probe, a pretrained SmolLM2-135M
  (2T tokens) was fine-tuned on the same written-chain stages. It reached only MuSiQue 4.7 % exact (floor 1.0 %) and
  HotpotQA 18.3 % (floor 7.0 %); plan section 10 has the details.
- **The synthetic world does resolve them, but only for warm starts so far.** Every result at the goal comes from
  0.1–0.3B-token continuations of one from-scratch E33 run (E33 on the bridge data, 1.35B tokens). That run's
  notebook switches on in 2 of 3 seeds from scratch.
- **The best label-free result.** The written route (19 Oct, Odra): latest 78 / 92 %, compose 93 / 98 %, deduce
  100 / 100 % at 4k / 16k; count 17 %.
- **Nothing scales before this holds from scratch.** No 100M or AWS run should start until the recipe is shown
  to work from random weights.

### The claim (one bet, two scales)
Train the E33 notebook model from random weights on the text world, with every training answer written as
reasoning from the first step, and remove the written steps in stages at the end (iCoT). Train the dense model on
the same data and schedule.

- **The 30M claim.** The model reaches the T4–T7 goal on latest, compose and deduce at 4k without generator labels.
  Its notebook switches on in every seed, it keeps the scores at 16k, and its language is no worse than dense.
- **The 100M claim.** Scale keeps all of that, and it beats dense where dense cannot read (past 4k) or cannot join
  facts (compose, deduce).

What 100M changes for the held-out tests is measured, with no pass mark set in advance: new sentence shapes, a third
hop, count. These numbers decide what the AWS run must add.

### Phase 0 — code fixes and one cheap check (laptop + 1 GPU-hour scale)
Details in plan section 11. Each fix is config-selectable, and its default keeps old runs bit-identical.

1. **The read target sees the order term** (`message_loop_read_order`: `content` = today | `full`).
   - Today `read_target_loss` and `capture_note_mass` (`nn/perceiver_ar_lm.py`) score a softmax without the E33d /
     E33e note terms. For the order-on-demand versions (E33j, E33m, the written route) the latest target therefore
     trains a read the model does not use.
   - **Check:** rerun stage 3 of the written route (0.1B tokens from its stage-2 checkpoint on Odra) with `full`.
   - **Adopt** for Phase A if latest at 4k reaches ≥ 83 % (written route 78 %; 400 items, SE ≈ 2 points) with
     compose and deduce within 2 points. Otherwise Phase A runs with `content`.
2. **Dense limited to its last 4,096 tokens** in `evaluation/real_multihop_eval.py`. Gate S3 of Phase C assumes it.
   Needed only when Phase C returns.
3. **`message_loop_span` as a trainer argument** (env var and launcher), so the AWS plan's "reading layer + 2 layers
   looped" can be launched. Default 1, as today.
4. **String flags parsed as booleans** in `PerceiverARConfig`. Today `bool("false")` is True for
   `lm_note_age_learn` when a config is built directly (`count_params`, the laptop bench). Server runs are not
   affected.
5. **100M pre-flight.** 50 steps each of `e33m_plain` and `dense_plain` at the main-tier shape on one 3090: memory
   at the planned micro-batch, and measured tokens per second. These replace the estimates below.

### Phase A — 30M from scratch (Polonez + Odra, about 15 h wall)
- **Arms:**
  - `e33m_plain`, seeds 0, 1, 2. It is the written route's architecture: hop naming 0.3, per-round read target 0.5,
    order on demand, count heads. Its loop exit losses are already off (`message_loop_exit_aux` 0).
  - `dense_plain`, seed 0, on the same data and schedule.
- **Data:** the written route's mix, frozen exams `text_checks_v2q8` (exams v2 + v2.1):
  - every task written as reasoning: compose ×3, deduce ×3, lookup, keyed, quote, latest, count;
  - 8 questions and 4 bridge questions per world document, 30 % story tokens;
  - read and naming targets read off the written text by the rule.

  One new build per schedule stage, all with seed 7:
  - stage 1, `all` steps written, 1.6B tokens (Phase B reads it all, Phase A the first 1.2B);
  - stage 2, `drop1`, 0.2B tokens;
  - stage 3, `none`, 0.2B tokens.
- **Schedule** (1.5B tokens, the screen-tier budget):
  - stage 1: 1.2B tokens from random weights, step size 1e-3, cosine;
  - stages 2 and 3: 0.15B tokens each from the previous final, step size 3.3e-4 (as R2 and the written route).
- **Exams:**
  - stage-3 finals at 1k / 4k / 16k, with the evidence-removed twin, notebook off, harder, paraphrase, crowded,
    paraphrase2 and steps splits;
  - stage-1 learning-curve checkpoints at 10 % steps, 4k only, as the switch-on detector.

### Phase A gates (median of 3 seeds, 4k, id split, stage-3 finals; set before running)
The ceilings measured for these gates:
- **Our warm-started written route** (same architecture, same data family): latest 78 %, compose 93 %, deduce 100 %,
  far lookup 99 % at 4k.
- **Dense on the bridge data:** latest 92 %, far lookup 81 % at 4k. On compose and deduce dense has no ceiling (it
  stays at guessing).

Gates:
- **A-S1, the recipe holds from scratch:** compose ≥ 70 %, deduce ≥ 70 %, latest ≥ 60 % (≈ 77 % of its warm-start
  ceiling). The evidence-removed twin must sit at its floor.
- **A-S2, the notebook switches on every time:** far lookup at 4k ≥ 60 % in 3 of 3 seeds (from-scratch E33
  today: 2 of 3).
- **A-S3, length:** at 16k, compose and latest each keep ≥ 80 % of the seed's own 4k score.
- **A-S4, no harm to language:** story loss within 2 % of dense on the same data.
- **Reported, no gate:** count, 3-hop compose, new sentence shapes, crowded facts, far quotes; dense's stage-3 scores.

Kill rules:
- **A-K1:** at 60 % of stage 1 (0.72B tokens), far lookup at 4k is under floor + 10 in all 3 seeds. Read targets
  from step 0 do not switch the notebook on. Stop. Phase B then uses the two-phase recipe (plain pretraining, then
  the written stages), the route all warm-start results took.
- **A-K2:** after stage 3, median compose < 30 % or median deduce < 30 %. The recipe does not work from scratch: no
  from-scratch 100M, and Phase B uses the two-phase recipe.

Go to Phase B: A-S1 and A-S2 pass, or A-S1 passes in 2 of 3 seeds and the third seed's failure is understood from
its curve.

### Phase B — 100M against dense (main tier, about 1.5 days wall)
- **Arms:** `e33m_plain` and `dense_plain` at the main tier (100M, rows of 4,096, 96 rows per step), seed 0 each.
  The recipe is Phase A's; if Phase A failed, it is the two-phase recipe.
- **Step sizes:** tuned per arch at the main tier as the protocol requires (3 runs × 0.1B tokens, held-out loss
  only).
- **Schedule** (2B tokens, the main-tier budget): stage 1 1.6B, stages 2 and 3 0.2B each.
- **Exams:** 1k / 4k / 16k, plus 64k for E33; the same splits and controls as Phase A.

### Phase B gates (set before running)
The ceilings are our Phase A medians and dense at 4k.

- **B-S1, no regression with scale:** E33 100M ≥ the Phase A median − 5 points on far lookup, latest, compose and
  deduce at 4k.
- **B-S2, dense compared where it can read:** at 4k, E33 is within 5 points of dense on far lookup and latest, and
  above it on compose and deduce. At 16k, E33 keeps ≥ 80 % of its 4k score; dense is not compared past 4k.
- **B-S3, switch-on:** far lookup ≥ 60 % at 4k by 25 % of stage 1.
- **B-K1:** at 25 % and again at 50 % of stage 1, far lookup at guessing. Stop and diagnose before spending the rest.
- **Measured, no gate (the scale question):** new sentence shapes (compose at 30M: 5 %), 3-hop compose (0 %), count
  (17 %).
  - If they move at 100M, the AWS 500M run keeps this recipe and adds natural text.
  - If they stay flat, the next step is architecture or data for those three (for example, the question-reading
    loop control, or varied phrasings), not size.

### Compute (estimates from measured 30M throughput; the Phase 0 pre-flight replaces the 100M figures)
Measured rates: `e33m_plain` ≈ 20k tokens/s per 3090 (written route, 19 Oct: 19.8–22.2k); `dense_plain` ≈ 60k.
Assumed at 100M: ≈ 6.5k and ≈ 20k (three times slower; dense 50M measured 41.5k).

| item | GPU-hours (3090) | wall |
|---|---|---|
| Phase 0: read-order check (0.1B tokens + exams) and 100M pre-flight | ≈ 5 | ≈ 3 h, with the data builds (CPU) in parallel |
| Phase A: 3 × E33 30M (≈ 21 training + 3 exams each) | ≈ 72 | ≈ 14–16 h on 6 GPUs |
| Phase A: dense 30M (≈ 7 + 1.5) | ≈ 8.5 | on the 7th GPU, inside the same window |
| Phase B: tuning (E33 3 × 4.3, dense 3 × 1.4) | ≈ 17 | ≈ 4 h |
| Phase B: E33 100M, 2B tokens (cap 120) + exams | ≈ 85 + 8 | ≈ 24 h on Polonez's 4 GPUs (heat bursts included) |
| Phase B: dense 100M, 2B tokens + exams | ≈ 28 + 3 | ≈ 11 h on Odra's 3 GPUs |
| **total** | **≈ 225** | **about 3–4 days of machine time, a week with reviews** |

On AWS (one p5.48xlarge, 8 × H100, about $55 per hour, an H100 taken as about 3 × a 3090), Phase B would be about
$350 and save about a day. The infrastructure for it does not exist yet, so Phases A and B run on our servers and the
AWS credits stay for the ~500M run.

### Out of scope for this re-scope
- New architecture parts (question-reading loop control, two key/value heads, note proportions).
- Real data (Phase C).
- A notebook-free control, and an E33 run without targets on the same data (R1 / R2 already measured the targets at
  30M).
- More dense seeds: dense has never composed at 30M (5 runs). Add two dense seeds (≈ 17 GPU-h) only if dense's
  stage-3 compose is above 20 %.

## Phase C (deferred) — the 21 Oct real-data design

Kept as written. It returns only with a ceiling that can be resolved, for example a pretrained reader at a larger
size, or HotpotQA bridge questions only.

## Hypothesis

**Claim.** Take a 100M E33 notebook model: a looped reader over compressed notes. Train it on real multi-hop QA
and maths data whose answers write their reasoning out, with three ingredients:
- the per-hop targets are read off that written reasoning by a rule;
- the written steps are removed in stages (iCoT);
- at exam time it answers with no steps.

**Predictions.**
- **At 4k:** it answers MuSiQue, HotpotQA and 2WikiMultiHopQA at least as well as a dense 100M model trained
  identically.
- **At 16k–64k:** it keeps ≥ 80 % of that score when the same evidence is buried in distractor paragraphs. The
  dense model, with a 4k window, falls to its no-evidence floor.

**Why.** The synthetic pilot showed the mechanism:
- each loop round learns to fetch one hop from the notes;
- the notes keep content whatever its distance.
R2 reached compose 92 % and deduce 99 % at 4k with no generator labels, the same as with them (94 / 98 %).

## Builds-on

- **Foundation (reused, config only):**
  - `PerceiverARLM` with the latent-memory writer (E31c / E33);
  - the loop's naming lens and per-round read target (E33q / E33qn: `message_loop_name_lens`,
    `message_loop_read_aux`);
  - the `read_targets` row column and its collator;
  - `scripts/pretokenize_mix.py`, which splits whole long documents into consecutive rows;
  - the text-checks runner (`scripts/run_text_checks.py`, main tier) and the shared bash launcher;
  - the written-chain rule (`parse_written_chain` in `scripts/build_text_checks_data.py`).
- **Init:** random, both arms, trained from scratch, as the project's architecture comparisons require. Phase P
  pretrains on general long-document text; phase M branches from phase P's final.
- **Baseline to beat:** a dense 100M model (`dense_plain`, main tier) on the identical data, tokens and stage
  schedule, measured inside this experiment.
- **Pilot evidence (30M, synthetic world, 19 Oct):**
  - R2, `written_r2s3_none`: compose 92 / 97 % and deduce 99 / 100 % at 4k / 16k, from written chains only.
  - R1, naming without read targets: compose 8 %, so read targets matter.
  - E33qn, generator labels: 94 / 98 %.
- **Materially new vs R2:**
  - real data: Wikipedia paragraphs, human-written questions, numbers;
  - reasoning text from the datasets' own decompositions, supporting facts and equation steps;
  - 100M scale;
  - long-context QA padded with real distractors;
  - a dense model trained under the identical written-chain schedule.

## The architectural bet

One claim, three parts.

1. **Supervision through text only.** Every training answer is written as reasoning:
   `"<step>. <step>. So <answer>."`. The steps come from the dataset itself:
   - MuSiQue's per-hop supporting paragraphs and sub-answers;
   - HotpotQA's and 2Wiki's supporting-fact sentences;
   - GSM8K-Aug's equation steps.

   A rule, extended from `parse_written_chain` to real text, reads off each step:
   - **its output word:** the step's sub-answer; the entity it shares with the next step; or the result of an
     equation;
   - **where it sits in the context:** the step sentence found in the document.

   These give the loop's round-r naming target and read target. No generator and no model-made label is used.
2. **Internalisation (iCoT, step removal).** Three stages of 0.15B tokens each, on the same documents:
   - all steps written;
   - the first step dropped;
   - the plain answer.

   The per-round targets stay attached to the hidden steps, as in R2 and SIM-CoT. At exam time no step is
   written.
3. **Notebook loop vs dense, under identical supervision.**
   - The E33 model reads its context through the raw window plus the notes, over 4 tied rounds.
   - The dense model sees 4k tokens with full attention and no loop. It learns the written chain explicitly in
     stages 1–2 and must answer directly in stage 3.

**Out of scope** (follow-ups only after a positive result):
- hop counts beyond those trained (E33c-style control);
- the key-gradient stop;
- latest / count;
- E36's frozen reader;
- a paraphrase-robust step matcher beyond entity overlap.

## Why this is not a safe retread

- Not a retune of R2: it moves the supervision from a synthetic generator to real reasoning text, at 4× scale,
  against a dense model under the same supervision.
- It asks whether a looped notebook can internalise written reasoning over long real documents. That question
  is open in the literature:
  - Coconut, CODI, SIM-CoT and LOTUS internalise reasoning over short contexts, not over a compressed memory;
  - long-context compressors (ICAE, AutoCompressors, gist tokens) do not reason in loops.
- **Analogy:** Vygotsky's internalisation, where a child's spoken self-guidance becomes inner speech. Here the
  written steps are the private speech, and the loop rounds are the inner speech that replaces them.

## Success criteria (set before running; all on stage-3 checkpoints, answers only)

**Ceiling C (measured inside the experiment).** The dense arm's stage-1 checkpoint, generating its written
chain at 4k. This is the best this scale and data reach when the reasoning is written out.

- **S1, no harm at 4k.** E33s F1 ≥ dense F1 − 2 points, on MuSiQue-Ans dev and on HotpotQA dev.
- **S2, the latent chain.**
  - E33s MuSiQue F1 at 4k ≥ 70 % of ceiling C. Coconut's GPT-2 reached 79 % of written CoT on GSM8K.
  - E33s ≥ E33 answer-only (the same tokens with no chains) + 5 F1 on MuSiQue.
- **S3, length.**
  - E33s keeps ≥ 80 % of its own 4k F1 at 16k and at 64k, on MuSiQue and HotpotQA.
  - The dense model at 16k / 64k (its last 4k tokens visible) is within 5 F1 of its evidence-removed floor.
- **S4, secondary.** GSM8K exact answer: reported for E33s, dense and E33 answer-only. No gate if all are
  below 5 %.

## Kill criteria (set before running)

- **K1.** Pre-flight rule coverage below 60 % of MuSiQue + HotpotQA-bridge training questions (§Plan P0). Then
  rescope to the datasets whose chains are verbatim sentences: HotpotQA and 2Wiki.
- **K2.** After phase P, E33's held-out language loss is more than dense's + 0.10 nats: the notebook model is a
  worse language model at 100M. Stop before phase M and report.
- **K3.** After stage 3, E33s MuSiQue F1 at 4k is below its evidence-removed floor + 5: the recipe does not
  transfer to real text. Stop.
- **K4.** E33s − E33 answer-only is below +1 F1 on MuSiQue at 4k: written chains add nothing on real data.

## Plan

### P0 — rule coverage (CPU, no GPU)

Build the written chains for every training question. Measure:
- the share the rule parses into per-hop targets;
- for MuSiQue and HotpotQA, how often the rule's located step matches the dataset's own supporting
  paragraph / sentence.

Targets: coverage ≥ 80 % and location agreement ≥ 90 %. Below 60 % coverage, K1 applies.

### P1 — 30M pipeline pilot (Polonez, 2 GPUs, ~4 h)

Screen tier with the 16k tokenizer: 0.3B phase-P tokens, then 3 × 0.05B phase M. This is a pipeline check, not
a result. Pass: the MuSiQue dev F1 at 4k is above its evidence-removed floor, and the exams and controls run
end to end.

### Data

- **Tokenizer:** a 16,384-token BPE, trained on the phase-P mix with the existing tokenizer trainer.
  - At 100M: FFN 1,728 for E33 and 1,856 for dense, 100.5M parameters each (fit_params, main tier).
  - A 32k vocabulary would thin the FFN to 1,152.
- **Phase P, 1.5B tokens, both arms identical.** Rows of 4,096 tokens, whole documents split into consecutive
  rows. New recipe `data/mix_recipes/e33s_realtext_4k_v1.json`:

  | source | share | status |
  |---|---|---|
  | English Wikipedia (`wikimedia/wikipedia`, 20231101.en) | 40 % | new source; the paragraphs MuSiQue and HotpotQA come from |
  | `nemotron_cc_v21_hq` (web) | 30 % | already in the repo's recipes |
  | `pg19_chunked` (books) | 15 % | already in the repo's recipes |
  | `finepdfs_long` (documents) | 15 % | already in the repo's recipes |

- **Phase M, 0.45B tokens, three stages of 0.15B: all → drop1 → none.**

  | data | train rows | steps | per-hop targets |
  |---|---|---|---|
  | MuSiQue-Ans train (`dgslibisey/MuSiQue`) | 19.9k (2–4 hops) | per hop, the sentence of the supporting paragraph that contains the sub-answer | naming = sub-answer; read = that sentence |
  | HotpotQA distractor train (`hotpotqa/hotpot_qa`) | 90.4k | the supporting-fact sentences, bridge first | naming = the shared entity, then the answer; comparison questions name only the answer |
  | 2WikiMultiHopQA train (`framolfese/2WikiMultihopQA`) | 167.5k | the supporting-fact sentences, in evidence order | as HotpotQA |
  | GSM8K-Aug train (`whynlp/gsm8k-aug`) | 385.6k | the equations | naming = each result (first 3 steps, then the answer at round 4); no read target |
  | phase-P text replay | 30 % of tokens | — | language only |

  - **Padding.** Each QA context is padded with distractor paragraphs from other questions of the same dataset,
    to a random length up to 4,096 tokens. The supporting paragraphs sit at random positions.
  - **Exam data.** The dev and test sets never enter training.

### Arms

Phase M branches from each arm's phase-P final.

| arm | model | phase-M data |
|---|---|---|
| the bet | `e33qn_plain`, main tier (naming lens 0.3 + per-round read target 0.5, 4 tied rounds) | written chains, 3 stages |
| the matched dense control | `dense_plain`, main tier | the same written chains and stages (no loop: the chain is learned explicitly in stages 1–2) |
| the no-chain control | `e33qn_plain` with naming and read targets off (= `e33x_plain`) | the same documents, answers only, same tokens |

### Exams (frozen, built once from dev / test)

- **Items:**
  - MuSiQue-Ans dev: all 2,417;
  - HotpotQA distractor dev: 2,000 sampled;
  - 2Wiki dev: 2,000 sampled;
  - GSM8K test: 1,319.
- **Lengths.** Every QA item at 4k, 16k and 64k tokens, padded with distractor paragraphs from the dev pool.
  The supporting paragraphs are spread over early, middle and late depths.
- **Controls.**
  - Evidence removed: the supporting paragraphs replaced by distractors; this is the floor.
  - Notebook off, for the E33 arms.
  - Stage-1 checkpoints answering with their chain written out: ceiling C for dense, and the same for E33s.
- **Scoring.** Greedy answer of at most 24 tokens. Exact match and token F1 with SQuAD normalisation, over the
  answer and its aliases. GSM8K: exact number.
- **Secondary.** The text-checks board exams, rebuilt with the 16k tokenizer, as a no-harm sanity check.

### Compute

| item | where | estimate |
|---|---|---|
| P0 | CPU | — |
| P1 pilot | Polonez, 2 GPUs | ~8 GPU-h |
| phase P, E33 | Polonez, 4 GPUs | ~90 GPU-h, ~23 h wall in bursts under 8 h with 20-minute pauses |
| phase P, dense | Odra, 3 GPUs (coordinate with the board session), or Polonez after E33 | ~90 GPU-h |
| phase M | 3 arms × 0.45B tokens | ~80 GPU-h |
| exams | both servers | ~10 GPU-h |
| **total** | | **~280 GPU-h, about 4 days wall on both servers** |

The phase-P estimate comes from the main tier's cap of 120 GPU-h for 2B tokens.

### Launch (shared runner and launcher, config only)

```
# phase P (per arm; dense: --arches dense_plain on Odra)
uv run python scripts/run_text_checks.py plan --phase train --tier main --arches e33qn_plain --host polonez \
  --gpus 0 1 2 3 --data $T/e33s_realtext_4k_v1 --out Cache/text_checks/e33s_pretrain_e33 --tokens 1.5e9 \
  --extra_env PAR_LOOP_NAME_LENS=0 PAR_LOOP_READ_AUX=0
# phase M, stage k (all | drop1 | none), each from the previous stage's final
uv run python scripts/run_text_checks.py plan --phase train --tier main --arches e33qn_plain --host polonez \
  --gpus 0 1 2 3 --data $T/e33s_realqa_<stage> --out Cache/text_checks/e33s_m<k>_<stage> --tokens 1.5e8 \
  --extra_env MODEL_NAME_OR_PATH=<previous final> LEARNING_RATE=<main-tier LR / 3>
```

### New foundation code (reusable; built via research-implement after go-ahead)

1. **Real multi-hop data builder:** `data/real_multihop.py` with `scripts/build_real_multihop_data.py`. It
   holds:
   - loaders for the four datasets;
   - chain writers;
   - the rule (`parse_written_chain` extended to sub-answers and equations);
   - distractor padding;
   - the stage flag;
   - frozen exam items.

   It writes rows with `read_targets` in the existing format, plus a manifest the loader already reads.
2. **A `wikipedia_en` source in `pretokenize_mix.py`,** plus the phase-P recipe and a 16k tokenizer.
3. **A real-QA exam:** `evaluation/real_multihop_eval.py`. Greedy answers, EM / F1 per dataset × length, and
   the evidence-removed and notebook-off controls. It reuses `text_checks_eval`'s model loading and
   loop / notebook overrides.
4. **The runner's fit at a data-defined vocabulary:** fit_params' `vocab` taken from the data's tokenizer.

## Result
*Filled in after, by experiment-track.*
