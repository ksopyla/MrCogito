# E33f — Notes on the wire (E33 trained as two agents: a question reads its document only through the notes)

- **Status:** run 1 done 2026-10-16 (Polonez); main pass mark met, late-fact mark missed. The author asked for "a clear separation between memory,
  writing to the memory, and sending to another agent ... based on e33 ... work until you reach the goal", and on
  16 Oct: "my main goal was firstly to train the model which could write and read from memory, the e33 was one of
  the most promising". Code and run by the E33 session.
- **Serves:** the agent-to-agent goal (E35): agent A writes the notebook, agent B answers from it. E33 is already
  split at its notes for far facts. E33f makes the split the training condition, so the notebook alone must carry
  everything a question needs.
- **Owner / dates:** ksopyla · opened 2026-10-16

## Why now (measured)
- The E35 transmitters (16 message vectors, broadcast and question-steered, 0.5B tokens each) carried nothing:
  the receiver's answer surprise was the same with the real, an empty or another document's message.
- The notes-budget exam (`scripts/notes_budget_exam.py`, no training) splits E33 at its notebook. Agent B reads
  another document of the same length plus agent A's notes. Far facts cross, so E33 already works as two agents
  for them. Two weak points:
  - late facts cross less well, even when A closes its last page: E33 learned to keep recent text in its raw window,
    not in its notes;
  - the read is diffuse: the answer looks at dozens of notes, and B's first query picks the wrong ones.

  E33 on the bridge data, 40 items per cell, exact answers, far facts / late facts (16 Oct, Polonez). "Own text" is
  the normal exam. "A's notes" means B reads another document plus all of A's notes. "Best 16" keeps only the 16
  notes the answer looked at most (an upper bound for any selection). "B asks" keeps the 128 notes that B's first
  read scores best.

  | task | own text | A's notes | best 16 | best 64 | B asks, 128 | no notes |
  |---|---|---|---|---|---|---|
  | far lookup 4k | 74 / 100 | 70 / 54 | 59 / 46 | 78 / 46 | 22 / 23 | 0 / 0 |
  | far lookup 16k | 93 / 92 | 93 / 92 | 93 / 92 | 96 / 100 | 11 / 0 | 0 / 0 |
  | keyed 4k | 70 / 85 | 67 / 69 | 37 / 46 | 48 / 46 | 22 / 8 | 0 / 0 |
  | keyed 16k | 96 / 92 | 96 / 100 | 93 / 100 | 93 / 100 | 11 / 0 | 0 / 0 |
  | quote 4k | 0 / 62 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

  - At 16k, 16 of 2,720 notes (2,048 numbers, 4 KB at 16 bits) carry a far fact as well as the whole document.
  - At 4k the read is more spread out: 64 notes are needed.
  - Late facts and quotes lose most when B cannot see A's text.
  - B's first read asks for the wrong notes: the loop finds the fact in later rounds.

## Hypothesis
If every question can read its document only through the notes during training, E33 learns to put all a question
needs into the notes and to read them sharply. Then:
1. the split costs nothing on far facts;
2. late facts cross;
3. a small number of notes carries the answer.

## The change (one setting, warm start)
- `message_boundary_token_id` = the "Question" token. Every "Question" starts a new side (`nn/perceiver_ar_lm.py`,
  the E21 boundary machinery). This is already in-tree; checked on a tiny model on 16 Oct:
  - a question's raw keys and local window stop at its boundary. With notes off, a question's logits do not
    depend on any document token;
  - the closed-window notes stay readable across the boundary (the closed rule ignores sides);
  - the window under the boundary writes only from the document side and closes at the last document token, so
    the tail page is readable at once (no late-fact gap).
- Training rows are unchanged (bridge data, up to 8 questions + 4 bridge questions per document). Each question
  is its own side, so a later question cannot read an earlier question's answer as raw text either.
- **Init:** E33 on the bridge data (`q8b_e33_s0_polonez`, checkpoint 7038), full state dict. The boundary adds
  no parameters.
- **Arch:** `e33f_plain` in `evaluation/text_checks.py`. `message_boundary_token: "Question"` is resolved to the
  data tokenizer's id by the runner. The recipe card is E33's, with a third of its step size (warm start).
- **Budget:** 0.3B tokens (about 1,570 steps × 96 rows), about 80 min on Polonez's 4 GPUs.

## Baseline (E33 on the bridge data, 400 items per cell, own text)
| task | 4k | 16k |
|---|---|---|
| far lookup | 79 % | 90 % |
| keyed | 73 % | 88 % |
| latest | 22 % | 4 % |
| quote | 25 % | 0 % |

## Success / kill
- **Pass:**
  - split far lookup and keyed at 4k and 16k ≥ 90 % of E33's own-text score;
  - late lookup at 4k ≥ 80 % (E33 split: 54 %; own text: 100 %);
  - oracle top-16 notes keep ≥ 90 % of the all-notes score;
  - held-out story loss within 0.03 of E33's (1.525).
- **Kill:** split far lookup at 4k below 60 % of E33's own-text score after 0.3B tokens. Then the notes cannot
  carry what the raw window did at this scale, and the wire needs raw text alongside the notes.

## Measures
- The text-checks exam: the boundary applies at exam time, so this is the split exam; plus notebook-off.
- The notes-budget exam (oracle and request selections, 1–128 notes), against E33's.

## Not in scope
- **A top-K note read in training** (each query reads only its K best notes, so the one-message request is
  exact). This is E33g if E33f's read stays diffuse.
- Compressing the notebook for broadcast (E35).

## Result
**Run:** Polonez, 16 Oct 2026, warm start from E33 checkpoint 7038, 1,568 steps (0.3B tokens), 20:23–21:30 UTC.
The first launch crashed at step 1: a compile bug in the global-read mask with more than one side per document,
fixed by a constant mask-tag stride (same mask).

**Notes-budget exam** (40 items per cell, exact answers, far / late facts). E33f's own exam is already split:
the boundary applies.

| task | E33 own text | E33 split | E33f split | E33f best 16 | E33f best 64 |
|---|---|---|---|---|---|
| far lookup 4k | 74 / 100 | 70 / 54 | 74 / 69 | 70 / 38 | 74 / 54 |
| far lookup 16k | 93 / 92 | 93 / 92 | 96 / 92 | 100 / 92 | 100 / 92 |
| keyed 4k | 70 / 85 | 67 / 69 | 70 / 69 | 26 / 62 | 56 / 92 |
| keyed 16k | 96 / 92 | 96 / 100 | 96 / 100 | 93 / 100 | 96 / 92 |
| quote 4k | 0 / 62 | 0 / 0 | 11 / 15 | 4 / 0 | 15 / 8 |
| quote 16k | 0 / 0 | 0 / 0 | 15 / 8 | 11 / 15 | 26 / 8 |

**One request per read** (B sends its round-r queries, A returns its K best notes, B re-reads with everything it
received and asks again; 4 rounds). All answers, exact; the number of distinct notes sent is in brackets.

| task | model | all notes | K = 2 | K = 4 | K = 8 |
|---|---|---|---|---|---|
| far lookup 16k | E33 | 92 | 40 (4) | 90 (8) | 95 (15) |
| far lookup 16k | E33f | 95 | 75 (4) | 98 (8) | 98 (14) |
| keyed 16k | E33 | 98 | 32 (4) | 95 (8) | 88 (15) |
| keyed 16k | E33f | 98 | 70 (4) | 95 (8) | 98 (15) |
| far lookup 4k | E33 | 65 | 30 (4) | 55 (8) | 62 (13) |
| far lookup 4k | E33f | 68 | 45 (4) | 70 (8) | 68 (13) |
| keyed 4k | E33 | 68 | 20 (4) | 57 (7) | 48 (13) |
| keyed 4k | E33f | 72 | 40 (4) | 70 (8) | 65 (13) |
| quote 16k | E33f | 12 | 0 (4) | 12 (8) | 25 (16) |

**Reading:**
- A one-message request (B's first read only) fails for both models: 0–30 %. One request per read of the loop
  works: about 8 of 2,720 notes (1,024 numbers, 2 KB at 16 bits) answer a far fact in a 16k-token document as
  well as the whole notebook.
- The notes needed did not grow from 4k to 16k (8 at both).
- E33f carries more per note than E33: with 4 notes sent, 75 vs 40 % on lookup and 70 vs 32 % on keyed at 16k.
- E33f quotes a little through the notes: 11–15 %, where E33 split and E33 on its own text get 0 % on far quotes.
- Late lookup at 4k rose from 54 to 69 % (13 late items: noisy), short of the 80 % pass mark.
- Wire cost: B's requests are the larger part. One query per head (576 numbers) per round per answer token, about
  2,300 numbers per answer token over 4 rounds. They have not been compressed yet.

**Full text-checks exam** (400 items per cell, exact %, all items / far facts; E33f is split by construction, E33
reads its own text):

| task | 1k E33f | 1k E33 | 4k E33f | 4k E33 | 16k E33f | 16k E33 |
|---|---|---|---|---|---|---|
| lookup | 34 / 33 | 55 / 35 | 74 / 77 | 79 / 72 | 91 / 91 | 90 / 90 |
| keyed | 25 / 26 | 45 / 21 | 63 / 66 | 73 / 69 | 91 / 92 | 88 / 90 |
| quote | 6 / 5 | 33 / 0 | 15 / 16 | 25 / 0 | 23 / 24 | 0 / 0 |
| latest | 15 | 95 | 4 | 22 | 6 | 4 |
| count | 31 | 28 | 29 | 26 | 17 | 19 |
| compose / deduce | 8 / 7 | 10 / 9 | 8 / 4 | 8 / 10 | 8 / 7 | 9 / 7 |

Held-out story loss 1.516 (E33 1.525).

**Long documents** (E33f, one request per read, 12 items per cell, exact %):

| length | notes in A's notebook | lookup, all notes | lookup, ~8 notes sent | keyed, all | keyed, ~8 sent |
|---|---|---|---|---|---|
| 32k | ~5,500 | 83 | 92 | 92 | 75 |
| 64k | ~10,900 | 92 | 100 | 100 | 100 |
| 128k | ~21,800 | 100 | 92 (7 notes) | — | — |

**Against the pass marks:**
- Far lookup and keyed split ≥ 90 % of E33's own text at 4k and 16k: **met** (77 vs 72, 91 vs 90; 66 vs 69,
  92 vs 90).
- Late lookup at 4k ≥ 80 %: **missed** (about 69 %; E33 split 54 %, E33 own text about 93 %).
- Best 16 notes keep ≥ 90 % of the all-notes score: met at 16k and for lookup at 4k (70 of 74), **missed** for keyed
  at 4k (26 of 70).
- Story loss within 0.03: **met** (1.516 vs 1.525).

**What the split costs:** near facts. At 1k every fact sits in the raw window E33 reads, and E33f cannot see A's
text. Lookup drops 55 → 34, keyed 45 → 25, quote 33 → 6, latest 95 → 15. Latest stays broken at every length.

**Agent-to-agent demo** (`scripts/notes_a2a_demo.py`, 16 Oct 23:19–23:23 UTC, Polonez, 3 documents × 12 questions
per cell, greedy answers). The two agents are separate:
- agent A runs only the writer: a 1M-token document becomes 174,752 notes in 0.45 s;
- agent B runs the reader on a row holding only its question;
- per answer token and per read of B's loop, B sends one query per head (585 numbers, 1.2 KB) and A returns its 4
  best notes (256 bytes each).

| document | lookup | keyed | quote | notes B fetched | down / up | text |
|---|---|---|---|---|---|---|
| 4k | 69 % | 67 % | 17 % | 13–17 | 3–4 KB / 22 KB | 16 KB |
| 16k | 97 % | 86 % | 36 % | 15–18 | 4 KB / 23 KB | 65 KB |
| 64k | 100 % | 94 % | 22 % | 16–21 | 4–5 KB / 23 KB | 260 KB |
| 256k | 100 % | 100 % | 28 % | 17–21 | 4–5 KB / 23 KB | 1 MB |
| 1M | 100 % | 97 % | (rerun pending) | 20 | 5 KB / 22 KB | 4.2 MB |

- Controls at every length: no notes 0 %, another document's notebook 0 %.
- B reading A's whole notebook (no fetching) scores 92–100 % up to 256k, but only 50 % (lookup) and 67 % (keyed) at
  1M. Fetching a few notes beats reading 174,752.
- The page that states the fact was among the fetched notes for 100 % of the questions.
- Asking only for the first answer token gives 0 % everywhere: each token of a name needs its own fetch.
- Quotes come back partly right (e.g. 5 of 6 words).

**Verdict:** E33f is the first trained model where agent A's notebook and agent B's reader are separate, and B answers
far facts from a handful of fetched notes. A 128k-token document becomes about 8 notes per question (1,024 numbers).
Near facts are the open gap.

