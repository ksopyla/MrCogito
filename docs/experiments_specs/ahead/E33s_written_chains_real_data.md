# E33s — Multi-hop on real text, taught by written reasoning, at 100M

- **Status:** draft (awaiting the author's alignment and go-ahead)
- **Serves:** the Vision (reason in latent space over compressed memory) and the author's 18 Oct direction:
  methods that need no generator labels, only available text, instruction and reasoning data.
- **Implementation plan:** [E33s_written_chains_real_data_plan.md](E33s_written_chains_real_data_plan.md)
  *(follows alignment)*
- **Alignment page:** [e33s_architecture.html](../../3_Evaluations_and_Baselines/e33s_architecture.html)
- **Owner / dates:** ksopyla · opened 2026-10-21

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
