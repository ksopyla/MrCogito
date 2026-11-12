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
bursts and 20 min cooldowns, Odra under 80 °C (two cards), the original LSA on Odra (its venv). A lane starts on a card
only after the capability queue on it has written `QUEUE-END` (an idle card between that queue's jobs or heat rests
is not free; that session's advice, 2026-11-12 20:50 UTC). Expected (UTC, worst case of that queue):

| card | free from | lane |
|---|---|---|
| Polonez 0 (cap: after Polonez 1's queue ends) | ~Nov 13 15:00 | the three LR pre-flights back to back, the pick, then the LSA port |
| Polonez 1 (cap: after Polonez 2's queue ends) | ~Nov 14 00:00 | catalogue read n4 |
| Polonez 2 (after Polonez 3's queue ends) | ~Nov 14 04:00 | concept memory |
| Odra 0 / Odra 1 | ~Nov 14 17:00–23:00 | notes-open / the original LSA |

The servers cannot reach each other, so the LR pick is copied to Odra by hand (it is needed there a day later).
The catalogue arm's code comes from `e37-catalogue-later` (merged into `e37-sft`).

| step | what | card-hours (estimate) |
|---|---|---|
| 0 | build the mix and held-out set on each server (CPU); pull the five checkpoints from S3 | 0 |
| 1 | base models: BLiMP, SWDE, BABILong, held-out instruction loss | ~15 |
| 2 | LR pre-flight on the LSA port; memory check of 16k rows on a 24 GB card | ~3 |
| 3 | instruction tuning, five arms | ~50 |
| 4 | tuned models: the whole evaluation list above | ~50 |

About 120 card-hours, 1.5–2 days on six cards after the capability checks end.

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

*(not run yet)*
