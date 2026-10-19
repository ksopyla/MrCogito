# E33e — Notes ordered among their matches (rank of newer matching notes on the notebook read)

- **Status:** running from 2026-10-16. The author asked for training fixes and the next runs for T4–T7
  (16 Oct: "analyze the results, propose some fixes in training, and schedule next runs"). Code, the
  exam-time check and the runs are by the E33 session.
- **Serves:** the Vision (compress long context into notes and read them back) and the text roadmap
  (prove the notebook on text). Order is the capability the E33 notebook lacks by design. E33d (page
  distance) showed that a recency term cannot add it without erasing far facts.
- **Alignment page:** [e33a_architecture.html](../../3_Evaluations_and_Baselines/e33a_architecture.html),
  section 8 ("the order toy", rank mode), and section 11 (the review, weak points 1 and 2).
- **Owner / dates:** ksopyla · opened 2026-10-16

## Hypothesis
If the notebook read charges each note by **how many newer notes match the question about as well as
the question's best note**, then E33 learns "where does Anna live now?" from its notes. A lone fact has
no newer match, so it pays nothing at any length. Far lookup keeps E33's level, while far "latest" at 4k
rises from 20–22 % (the guessing floor is 20 %) to at least 60 %.

Why E33d could not do this:
- On the log page-distance scale, Anna's last move and the one before it differ by about 0.47 × the
  slope, so a 2-nat preference needs a slope near 4.
- That slope erases every old single fact.
- The fixed-slope check on E33d run 1 measured this. A slope of +1 lifted latest at 16k from 2 to
  33 %, but halved far lookup (93 to 52 %).
- A slope of +10 on 4 of the 9 heads kept lookup at 93 % but left latest at 2 %. A recency head reads
  the newest page, which holds someone else's move, not Anna's.

## Builds-on
- **Foundation:**
  - `nn/perceiver_ar_lm.py`:
    - `note_rank_logcount` computes the count from the read's own note scores, without gradient;
    - `attend_message` adds the bias in both attention backends, reusing E33d's per-head parameter `note_age_slope`;
    - config `lm_note_age="rank"`, plus `lm_note_rank_margin` (1.0) and `lm_note_rank_sharp` (2.0);
  - one pass-through setting in the shared trainer and launcher: `PAR_LM_NOTE_AGE=rank`;
  - `evaluation/text_checks.py`: arches `e33e` and `e33e_plain`.
  
  There is no new trainer and no new model file. Defaults keep every existing model and checkpoint bit-identical.
- **Init:** random, as every text-checks model. Every strength starts at 0, so step 0 is exactly E33,
  and the parameter count equals E33d's (29.86M).
- **Baseline to beat:** E33 (3 runs, 8-question recipe, 30M plain input, 1.5B tokens):
  - latest at 4k: 20–22 %;
  - far lookup at 4k / 16k: 70–76 / 92–93 % in the runs whose notebook switched on;
  - story loss: 1.519–1.522.
  
  E33d (2 runs) is the same picture, with latest at 16–21 %.
- **Ceiling for latest at this scale:** dense learned it in 2 of 3 runs, 77–82 % at 4k.
- **Materially new:** an order code that is relative to the **matches**, not to the page distance.
  Close relatives:
  - CoPE (contextual position encoding, arXiv 2405.18719), which counts gated tokens instead of
    positions;
  - stick-breaking attention (arXiv 2410.17980), where the newest match takes the mass first.
  
  It is not a rotation of the key, not an absolute page number, and not a distance.

## The architectural bet
For reader t, query head h, visible note c on page p(c) (same document, page closed at or before t):

```
l(t,c)      = q_slot,h(t) · k_c / √64                     (the read's own note score, unchanged)
m(t)        = max over visible notes of l(t,c)             (the best match)
gate(t,c)   = σ(sharp · (l(t,c) − m(t) + margin))          (≈ 1 for notes within ~margin of the best)
count(t,p)  = Σ gate(t,c') over visible notes c' on pages newer than p
score(t,c)  = l(t,c) − s_h · log(1 + count(t, p(c)))       (s_h: per head, starts at 0)
```

- Anna's ten moves all match, so the newest pays 0, the one before ≈ log(1 + 32·σ(2)), and so on.
  The penalty does not grow with document length.
- A lone fact (lookup, quote, keyed) has no newer match, so it pays nothing however many pages follow.
- The count is computed without gradient, so it costs no extra memory in training. Only s_h learns.
- Cost:
  - one extra note-score pass per loop round;
  - flex kernel blocks of 64 instead of 128, because the page gather overflowed the 3090's shared memory;
  - measured on Odra at 30M, 4k, micro-batch 4: 0.54 s per step against 0.51 s for E33d, peak memory 15.4 GB (equal).

## Exam-time check (no training, run first)
The rank term with one fixed strength for every head, put on E33d run 1's trained weights. The margin
and sharpness are chosen on fresh development items (new seeds, never the exam set). The setting is
then confirmed on the exam set.
- **Pass (the mechanism can work with trained scores):** latest at 16k above 33 % (the best fixed page
  distance) while far lookup at 16k stays within 5 points of 93 %.
- A miss is not a kill. These scores were never trained to be counted, which is what the training run tests.

## Success / kill (training run, `e33e_plain`, 1.5B tokens, one seed first)
- **Pass:** latest ≥ 60 % at 4k and ≥ 40 % at 16k; far lookup within 5 points of E33 (at 4k and
  16k); story loss ≤ 1.53.
- **Stop the line:** latest ≤ 30 % at 4k on a run whose notebook switched on, with the learned
  strengths near 0. Order is then not what blocks latest.
- **Read also:**
  - the learned per-head strengths (logged with every exam);
  - the learning curve for latest (dense switches it on at 0.9–1.05B tokens);
  - the "picked" histogram: a reader with order picks Anna's newest place, while one without spreads over all 11.

## Not in scope (follow-ups after a signal)
- a gradient through the count (CoPE-style learned gates);
- per-head margins;
- combining this with the bridge-question data (the T5/T7 line, a separate run of E33 with only the data changed);
- the 100M tier.

## Result
**Exam-time check (16 Oct 07:53–08:45 CEST, Odra; no training).**

The gate was chosen on 1,200 fresh development questions (latest, lookup, keyed and quote at 4k and 16k; never the exam set). The default gate (margin 1, sharpness 2) counted look-alikes and cost keyed up to 15 points at strength 2. The tighter gate (margin 0.5, sharpness 4) kept far lookup and lost keyed 7.

On dev, E33d run 1 with the tighter gate at strengths 1 / 2 / 3 / 5:

| strength | latest at 16k | far lookup at 16k |
|---|---|---|
| none (as trained) | 5 % | 94 % |
| 1 | 22 % | 95 % |
| 2 | 37 % | 95 % |
| 3 | 39 % | 92 % |
| 5 | 45 % | 90 % |

On the exam set (400 per cell), strength 2, tighter gate:

| model | latest at 16k | far lookup at 16k | keyed |
|---|---|---|---|
| E33 run 1 (never trained with any order term) | 5 % → **41 %** | 92 → 91 % | −2 points |
| E33d run 1 | 2 % → **30 %** | 93 → 92 % | −5 points |

- **Verdict against the bar set before the check** (latest at 16k above 33 %, far lookup within 5 points): pass on E33 run 1, a narrow miss on E33d run 1.
- **Against the fixed page distance:** that bought 33 % on E33d run 1 at the cost of far lookup 93 → 52 %.
- **Reading:** order is the missing piece for latest, and the read's existing scores find the matches well enough to count them.
- **Training gate:** the training run uses the tighter gate.

**Training run 1** (`q8r_e33e_s0_odra`, Odra, from 16 Oct about 08:45 CEST): _(to fill)_
