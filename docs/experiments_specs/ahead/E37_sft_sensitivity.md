# E37 run — instruction tuning as a sensitivity test for the 100M memories

- **Status:** approved (the author, 2026-11-12: "Yes, both" — add the three tests and write this spec; "schedule the SFT training")
- **Serves:** the 100M comparison of E37's full protocol (E33p LSA port, E33t catalogue read, E37a concept memory,
  E33p notes-open, the original LSA). Not a new architecture: a post-training stage run on all five, so it is a *run*
  of E37, not a new ID.
- **Owner / dates:** the E33n/E33t session (worktree `MrCogito-sft`, branch `e37-sft`, cut from `arch-10m`) ·
  opened 2026-11-12
- **Builds-on:** the five decay-final checkpoints of E37's m100 protocol (3.0B FineWeb-Edu tokens at 4k, then 0.4B
  LongDocs-lite tokens at 16k, seed 42), `data/bin_token_dataset.py`, `scripts/run_e33n.py`,
  `scripts/lsa_orig.py`, `evaluation/run_lsa_suite.py`.

## Hypothesis

A short, matched instruction-tuning stage (about 200M tokens, ~6 % of pretraining) changes the long-context scores
of the five 100M models by **different** amounts, because a memory that is read through learned addresses (the
notebooks) can be taught to be used better than one it was never asked to use on questions. Falsified if every
model's long-context gain is the same within noise, or if no model gains at all.

## What is new against the base comparison

The base models were judged only on prompts they never saw the shape of (RULER's questions, quizzes). This stage
teaches every model the same question → answer shape on the same data, then repeats every test. The readout is not
the absolute score but the **change per architecture**: does tuning move the ranking (the LSA port leads the needle
puzzles at every length; the original LSA leads text prediction; general knowledge is tied)?

## Arms (identical data, tokens, schedule, seed)

| arm | base checkpoint (S3 `runs/e37/…`) | trainer |
|---|---|---|
| LSA port (E33p) | `e33n_m100_lsa_tok3.0e9_seed42` decay final | ours |
| Catalogue read, 64 notes per page (E33t n4) | `e33n_m100_catalogue_reuse_n4_tok3.0e9_seed42` decay final | ours |
| Concept memory (E37a) | `e33n_m100_e37a_tok3.0e9_seed42` decay final | ours |
| Notes-open (E33p) | `e33n_m100_notes_open_tok3.0e9_seed42` decay final | ours |
| Original LSA (the authors' code) | `lsa_orig_m100_tok3.0e9_seed42` decay final | theirs (`scripts/lsa_orig.py`, Odra's venv) |

## Data

**Instruction set: `HuggingFaceTB/smol-smoltalk`** (Apache-2.0). It is the SFT set Hugging Face built for
SmolLM2-135M and -360M-Instruct, i.e. for models of our size: 460k English conversations, ~412M GPT-2 tokens
(measured on its 24k-row test split: 895 tokens per conversation on average, 74 % of them answer tokens). Its
sources: short synthetic chats (Smol-Magpie-Ultra, ~72 %), LongAlign long-document question answering (~10 %,
3.5k conversations, ~11k tokens each), OpenHermes, self-OSS-instruct code, summarising, rewriting and
constraint-following. Why this one:
- built for sub-1B models (shorter chats, no function calling, little math), so the stage teaches format a 100M
  model can learn;
- it carries real long-document QA, the part that asks the memory to work;
- no benchmark-derived sets (unlike FLAN-style mixes), so the quizzes stay clean; nothing from SQuAD, HotpotQA,
  bAbI or SWDE, so RULER's QA tasks, BABILong and SWDE stay held out.

Not included: math chain-of-thought (MetaMathQA, NuminaMath; SmolLM2 dropped them below 1B), Tulu 3 / FLAN mixes
(contain the quiz benchmarks), DPO / UltraFeedback (the follow-up below).

**Mix per arm (~200M tokens processed):**
- every LongAlign conversation that fits 16,384 tokens (~83 %; longer ones dropped, never cut);
- a seeded random sample of the other conversations, up to 180M tokens in all;
- 10 % replay of LongDocs-lite training text (16k rows, loss on every token) so long-document prediction does not
  drift.

**Format:** plain text with GPT-2's existing tokens (no new embeddings): `<|endoftext|>` opens a conversation, each
turn is `<|system|>\n…`, `<|user|>\n…` or `<|assistant|>\n…` followed by `\n<|end|>\n`. Loss only on assistant text and
its `\n<|end|>` (so the model learns to stop). Whole conversations packed into 16,384-token rows, padded with
`<|endoftext|>`; the document mask keeps packed conversations apart in our models (the original LSA's fused read takes
no document mask, as in its own trunk).

**Held-out:** smol-smoltalk's test split (24k conversations, 21.7M tokens): loss on answer tokens, before and after.

## Schedule

- **LR pre-flight** (LSA port only, 20M tokens each): peak 5e-5, 1.5e-4, 4.5e-4. Keep the one with the lowest
  held-out answer loss; all five arms then use it, so the LR is not a difference between arms.
- From the decay-final weights, fresh optimizer: warmup 3 %, 1−sqrt decay to 0.1× over the rest (the decay's shape),
  AdamW (0.9, 0.95), weight decay 0.1, clip 1.0, 16 rows × 16,384 = 262k tokens per step, 763 steps, seed 42.
- **Loss weighting (decided 2026-11-12):** the answer-token mean over the whole optimizer step, in both trainers
  (torchtitan's own reduction; ours is switched to it for this phase only, auxiliary losses unchanged). The
  per-row mean our trainer used before would have given the long-document conversations (one per row, ~200 answer
  tokens each) about 23 % of the gradient instead of about 1 %, and only in four of the five arms. Under the token
  mean they are a small part of the signal; if S1 fails only on the long tests, LongAlign's per-conversation
  weighting is the first follow-up arm.
- The held-out answer loss is scored on 256 of the test split's ~1,365 rows (about a fifth of its tokens).

## Evaluation (same prompts before and after)

- held-out instruction loss (answer tokens of the smol-smoltalk test split);
- general knowledge: the seven zero-shot quizzes (HellaSwag, ARC easy/challenge, PIQA, WinoGrande, SciQ, LAMBADA);
- needle puzzles: RULER's 13 tasks at 4k and 16k, 100 prompts each;
- the three tests added 2026-11-12 for a wider view (run on the base models too):
  - **BLiMP** — grammar: 67 sets of minimal sentence pairs, all 67k pairs;
  - **SWDE** — reading a real web page: pull one field out of it, all ~1.1k validation pages;
  - **BABILong tasks 1–3** — one to three facts scattered through book text, at 2k, 4k, 8k and 16k, 100 each;
- long-document text prediction at 4k, 16k and 32k (24 GB home cards; 64k+ stays on AWS).

The base models already have core and RULER 4k/16k from AWS; they get the three new tests and the held-out
instruction loss at home. A second pass with the prompts wrapped in the chat format is optional (after the first
readout).

## Success criteria (set before running)

- **S1, the stage unlocks something:** at least one arm's RULER-16k mean or BABILong mean rises by ≥ 5 points
  (a 1,200–1,300-prompt mean has a standard error of about 1 point).
- **S2, sensitivity differs by architecture:** the best- and worst-gaining arms differ by ≥ 5 points in gain on
  RULER-16k or on BABILong.
- **No harm (per arm):** general knowledge within 1 point, BLiMP within 1 point, long-document prediction at 16k
  worse by at most 0.03. An arm that breaks this is reported as harmed by tuning.

## Kill criteria (set before running)

- **K1:** the best pre-flight LR moves the held-out answer loss by less than 0.02 → the stage is too weak; stop and
  rethink before tuning five models.
- **K2:** every arm's gain on every long-context test is within ±2 points → "100M models are insensitive to this
  tuning"; record it, and do not start RL on them.

## Plan

Home servers only (no AWS spend), as the capability checks free the cards: Polonez at most three cards with 5.5 h
bursts and 20 min cooldowns, Odra under 80 °C, the original LSA on Odra (its venv is only there). A lane starts on a
card only after the capability queue on it has written `QUEUE-END` (an idle card between that queue's jobs or heat
rests is not free; that session's advice, 2026-11-12 20:50 UTC). Polonez's lanes share three slots: a lane takes its
card when the card is free and fewer than three cards are busy (capability queues included), and gives it back while
it waits for the LR pick. On Odra the original LSA's lane waits for the capability queues on cards 0 and 1 and for
those two cards under 78 °C (card 2 keeps its own queue and heat guard).

Revised 2026-11-13 04:45 UTC (the capability session's new times: Polonez cards free a day earlier, Odra card 2 a
day later; times are its estimates, worst case in brackets):

| Polonez slot | free from | lanes, in the order they get a slot |
|---|---|---|
| 1 (card 0 or 2; card 2's queue ended 04:35) | Nov 13 ~04:50 | the first to claim: the LR pre-flights + pick + LSA port, or concept memory |
| 2 (card 3's queue ends) | Nov 13 ~07:30 (20:00) | the other of those two |
| 3 (card 1's queue ends) | Nov 13 ~11:00 (Nov 14 00:00) | catalogue read n4 |
| any slot, card 3 | after the LR pre-flights claim card 0 (~07:45) and a slot frees (~08:00, when concept memory gives its card back to wait for the LR) | notes-open, moved from Odra |

| Odra card | free from | lane |
|---|---|---|
| 0 (after Odra's card 0 and 1 queues end; temperature checked on cards 0–1 only) | Nov 14 ~17:00 – Nov 15 00:00 | the original LSA |

Notes-open moved to Polonez on 2026-11-13 06:04 UTC: its checkpoint was fetched from S3 at 06:03 (all eight files
the S3 sizes, the weights byte-identical to Odra's copy), its lane waits on card 3 until the LR pre-flights hold
card 0 (so it never takes their slot), and Odra's card-1 fallback lane was stopped before it had claimed a card.
Its base tests now fill the hours the other lanes spend waiting for the LR.

The capability session agreed (2026-11-13 ~04:50 UTC) to the original LSA on Odra card 0 once its card 0–1 queues
end, card 1 left empty while its card 2 runs (more heat pauses for its last seed, a few hours, against 1–1.5 days of
card 0). The servers cannot reach each other, so the LR pick is copied to Odra by hand.
The catalogue arm's code comes from `e37-catalogue-later` (merged into `e37-sft`).

| step | what | card-hours (estimate) |
|---|---|---|
| 0 | build the mix and held-out set on each server (CPU); pull the five checkpoints from S3 | 0 |
| 1 | base models: BLiMP, SWDE, BABILong, held-out instruction loss | ~15 |
| 2 | LR pre-flight on the LSA port; memory check of 16k rows on a 24 GB card | ~3 |
| 3 | instruction tuning, five arms | ~50 |
| 4 | tuned models: the whole evaluation list above | ~50 |

About 120 card-hours. With the 2026-11-13 times: the four Polonez arms done ~Nov 15 08:00 UTC, the original LSA
~Nov 15 17:00 – Nov 16 00:00.

## Risks

- A 16k training row may not fit a 24 GB card at micro-batch 1 for every arm: fall back to gradient checkpointing,
  never to shorter rows (the long conversations are the point).
- Our models' answer decoding is uncached (cost grows with length squared), so BABILong at 16k and RULER at 16k
  dominate the evaluation time.
- One seed per arm: gains under ~3 points are not separable from noise.

## Follow-ups (out of scope)

- DPO on UltraFeedback after this stage (SmolLM2's second step): the first preference-learning test.
- RL (GRPO) on the needle puzzles, where the base models already score 10–25 %: needs cached decoding first.

## Result

**Interim, 2026-11-13 20:15 UTC (two of five arms partly scored; not a verdict).**

- **LR pick (Polonez card 0, 06:39–09:47 UTC):** held-out answer loss of the LSA port after 20M tokens: 5e-5 1.908,
  1.5e-4 1.806, **4.5e-4 1.714** (base 2.215). Picked 4.5e-4; copied to Odra 09:47. **K1 not hit** (best move 0.50,
  bar 0.02).
- **Tuning (200M tokens, 763 steps, ~6 h a model with the burst rests):** LSA port 10:56–16:10, notes-open
  10:41–17:10, concept memory 10:43–17:19 UTC (all on Polonez); held-out eval loss fell smoothly, no NaN.
- **Scores so far** (base → tuned; loss lower is better, accuracy higher is better):

| test | LSA port | notes-open |
|---|---|---|
| held-out answer loss (smol-smoltalk test, 16k rows) | 2.215 → 1.389 | 2.237 → 1.386 |
| FineWeb-Edu val loss at 4k | 3.193 → 3.296 | 3.195 → 3.293 |
| LongDocs-lite val loss at 16k | 2.232 → 2.331 | 2.323 → 2.432 |
| BLiMP | 78.8 → 78.4 % | 77.9 → 78.3 % |
| SWDE (contains) | 32.9 → 41.2 % | 27.0 → 35.0 % |
| BABILong 1–3, mean of 2k–16k | 5.3 → 3.8 % | 2.6 → (16k pending; 2k–8k at the floor as before) |
| core, mean of 7 (acc_norm where defined) | 41.6 (AWS) → 42.6 | (pending) |

  Concept memory and catalogue n4 base: held-out answer loss 2.222 / 2.243, BLiMP 79.2 / 77.8 %, SWDE 37.6 / 29.6 %,
  BABILong mean 5.7 / 2.8 %.
- **Reading so far:** the stage teaches the format (answer loss −0.83 nats), adds 8 points on SWDE for both arms
  (same size: no architecture difference yet), gives nothing on BABILong (at the floor before and after), keeps
  BLiMP and core, and costs ~0.10 nats of plain-text prediction at 4k and 16k (3× the 0.03 no-harm bar; the
  official long-document test runs last). RULER 4k / 16k pending (S1, S2).
- **Polonez card 2 fault (2026-11-13 17:39:31 UTC):** Xid 62 (internal micro-controller halt) the second concept
  memory's first tuned test started, after its 20 min rest; the card reports power "Unknown Error" and clocks "System
  is not in ready state"; any CUDA copy to it spins, and the stuck processes survive SIGKILL. Needs a reboot. The
  tuned concept-memory checkpoint is saved and intact (final at 17:19). Catalogue n4 waits: the faulted card still
  counts as busy (it sits in P2 at 55 °C), so Polonez stays at three cards.
