# E33f — Notes on the wire (E33 trained as two agents: a question reads its document only through the notes)

- **Status:** opened 2026-10-16, run queued on Polonez. The author asked for "a clear separation between memory,
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
_(to fill)_
