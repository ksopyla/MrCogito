# E33d — Notes with a page distance (per-head age bias on the notebook read)

- **Status:** approved 2026-10-14 (author: "yes please, e33d is ok"). Code and laptop check by the
  E33 reasoning-fixes session; the text runs go to the session that runs the text checks.
- **Serves:** the Vision (compress long context into notes and read them back) and the text
  roadmap (prove the notebook on text). Order is the first capability the E33 notebook lacks by
  design: it cannot tell which of two notes is newer.
- **Implementation plan:** [E33d_note_page_distance_plan.md](E33d_note_page_distance_plan.md)
- **Alignment page:** [e33a_architecture.html](../../3_Evaluations_and_Baselines/e33a_architecture.html),
  sections 2–3 (where position lives today, the "latest" example at 1k / 4k / 16k, the proposal).
- **Owner / dates:** ksopyla · opened 2026-10-14

## Hypothesis
If the notebook read adds to every note's score a learned, signed, per-head penalty proportional
to log(1 + pages between the note and the reading token), starting at zero, then E33 learns which
of two same-content notes is newer. Far "latest" at 4k rises from 22 % to at least 60 %, while far
lookup at 16k stays within 5 points of E33's 92 %. The reason: the distance enters as a separate
term added to the score. The content match is untouched, and a head with slope 0 reads exactly as
today. On DNA, order codes placed inside the 64-number key (rotation) competed with content and
cost long-range lookup (scaled order code 47 % vs 95 % at 128k).

## Builds-on
- **Foundation:**
  - `nn/perceiver_ar_lm.py`: the global read (`attend_message`, both attention backends) gets an
    additive bias on the note columns; the global layer's `Attention` gets one parameter per query
    head (`note_age_slope`);
  - `nn/latent_memory.py`: **unchanged**. It already returns each note's position (`slot_pos`, the
    expected position of what its latent read), which the text setup computes and never uses;
  - shared trainer and launcher: one pass-through setting (`--lm_note_age`, `PAR_LM_NOTE_AGE`);
  - `evaluation/text_checks.py`: new arches `e33d` and `e33d_plain` (= E33 + `lm_note_age=log`).
  No new trainer and no new model file. Defaults keep every existing model and checkpoint
  bit-identical.
- **Init / checkpoint:** random init, as every text-checks model. All slopes start at 0, so step 0
  is exactly E33.
- **Baseline to beat:** E33 run 1 on the 8-questions recipe, `q8r_e33_s0_odra.odra` (30M plain,
  1.5B tokens, 13 Oct 23:12 → 14 Oct 08:13 UTC):
  - far facts at 4k / 8k / 16k: lookup 70 / 83 / 92 %, keyed 71 / 86 / 90 %, quote 71 / 84 / 89 %;
  - latest 95 % at 1k (the last move sits inside the 256-token raw window), 22 % at 4k (guessing
    20 %), about 5 % beyond 4k;
  - crowded lookup 44 % at 4k; count 26 %; story loss 1.520 nats per token;
  - E33 run 2 (running, done about 17:35 UTC 14 Oct) adds the seed spread.
  Ceiling for latest at this scale: dense learned it in 2 of 3 runs, 77–82 % at 4k.
- **Materially new:** the first order code on the text notebook. It is a relative page distance,
  added outside the key (ALiBi-style, attention with linear biases, here on a log scale). It is
  not a rotation of the key, as the DNA "read", "scaled" and "mixed" note positions were, and not
  an absolute page number (`lm_addr = window_start`).

## The architectural bet
For a reading token at position t and a note n, in query head h of the global read:

```
score(t, n, h) = q_slot,h(t) · k_n / √64  −  s_h · log(1 + max(0, pos(t) − pos(n)) / 192)
                 (content: unchanged)        (new: page distance)
```

- **pos(n):** the expected position of what note n's latent read, already computed by the writer
  (rounded, in its document). Notes on the same page are therefore ordered too.
- **pos(t):** the reading token's own position in its document. Tokens on the open page, and
  generated tokens, carry "now" automatically; no page number is stored anywhere.
- **s_h:** one learned number per query head of the global read (9 at 30M), starting at 0, signed.
  A positive slope prefers newer notes (latest), a negative one older notes ("where did Anna live
  first?"), and 0 is pure content lookup.
- **Unit:** one page step (192 tokens). **Log scale:** 20 pages back (a 4k row) costs log 21 = 3.0
  per unit of slope, 85 pages back (16k) 4.5. The penalty keeps growing beyond the training length
  but slowly; this is the main guard for length transfer.
- **Unchanged:** raw keys (rotated, the last 256 tokens), the writer and its proportions, the loop
  (the global layer is tied, so one set of slopes serves all 4 rounds), the exits, the data.

Not in scope: absolute page addresses, rotating note keys, the writer's input and proportions (the
notebook-proportions sweep), and a learned non-monotone distance curve (a follow-up if the log
curve is too rigid).

## Why this is not a safe retread
The DNA study tried order **inside** the key: true-distance rotation, squeezed distances, and a
frequency split. All three traded long-range lookup for order ("order and content pull against
each other in one key"). This bet moves order **out of** the key into a per-head score term,
so each head chooses content, recency or primacy. Analogy: human free recall shows a recency
gradient on top of cue matching. The cue picks the memory, and recency breaks ties among similar
memories.

## Success criteria (set BEFORE running)
- **Order:** far latest at 4k ≥ 60 % (median of 2 seeds). Dense's 77–82 % is the ceiling at this
  scale.
- **Order beyond training length:** far latest at 8k and 16k ≥ 40 % (dense has nothing past 4k).
- **No harm:** far lookup, keyed and quote at 16k each within 5 points of E33 (median of E33 runs 1–2);
  story loss ≤ E33 + 0.01 nats per token.

## Kill criteria (set BEFORE running)
- **Early readout, 0.6B tokens:** far lookup at 4k below 54 % in both seeds (E33 had 64 % at this
  point) → stop; the bias is hurting the notebook switching on.
- **Full run, 1.5B tokens:** far lookup at 16k more than 5 points below E33 → reject, whatever latest
  does.
- **Full run:** latest at 4k ≤ 30 % in both seeds → the age term is not enough; the order problem
  lies elsewhere (for example in the writer), stop this line.

Diagnostics logged with each exam: the 9 slopes (signs and sizes), the read's mass on notes vs raw
keys at answer positions, and the saved latest answers (which of Anna's places is picked).

## Plan
- **Data:** Text World v2 training data with 8 questions per document (the current recipe),
  1.5B tokens; exams v2 + v2.1 at 1k–16k, exactly as E33's runs.
- **Compute:** Odra 3 × RTX 3090 after the E31c runs (about 01:00 UTC 15 Oct), or Polonez in bursts
  under 10 h. About 7 h training + 1.6 h exams per seed at 30M; 2 seeds. Early readout at 0.6B
  tokens. Measured on Polonez (14 Oct): a training step costs 0.51 s vs E33's 0.49 s, same memory.
- **Step size:** E33's tuned 1e-3 is borrowed (the change adds 9 numbers). The text-checks protocol
  gives every architecture its own tuning budget; the text session decides whether to run it.
- **Launch:** the text-checks runner with arch `e33d_plain`, e.g.
  `uv run python scripts/run_text_checks.py plan --phase train --tier screen --arches e33d_plain ...`,
  i.e. E33's settings plus `PAR_LM_NOTE_AGE=log` on `scripts/train_concept_pretraining_multigpu.sh`.
- **New foundation code:** config `lm_note_age` ("none" | "log" | "linear") and `lm_note_age_unit`
  (0 = the page step); the per-head slope in the global read; the bias in the flex and SDPA paths.
  Defaults "none" / 0 keep every model bit-identical.

## Result
<Filled in AFTER, by experiment-track. Link out; do not paste full results here.>
