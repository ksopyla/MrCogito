"""Hand-written text of the text capability board: the status brief and each round's verdict.

Numbers on the board come from the ledgers; this file holds only what a person has to judge:
the one-line status, where we are, what is next, and per round the verdict and observations.
Plain words, every number with its meaning (research-comms). Update it whenever a round lands.
"""

STATUS = {
    "as_of": "2026-10-12 01:40 UTC",
    "headline": "At 30M and 1.5B tokens, dense learns retrieval only in some runs: 2 of 4 seeds learned something "
                "(one with the input tables, one without), so the plain-input gain was seed luck, not a fix. The E33 "
                "loop run reads facts at 16k where dense cannot, but it is also one seed.",
    "where": [
        "Dense plain input, seed 1 (Odra, 11 Oct 05:36-08:43 UTC): every task at guessing. Seed 0 learned four tasks.",
        "Dense with the tables: seed 0 passed quote, seed 1 nothing. So far no input-layer effect can be told apart "
        "from the seed.",
        "E33 loop, plain input, seed 0: lookup 82 % and keyed 77 % at 16k (dense seed 0: 0 %); notebook off → 0 %.",
    ],
    "next": [
        "Done 12:15 UTC: E31c notebook, plain input, mixed tasks — lookup 16 % at 4k, 0 % at 16k; the same with "
        "the notebook off, so without the loop the notebook goes unused (one seed).",
        "Then the single-task comparison (author, 11 Oct): for T1 quote, then T2 lookup, ... T7 deduce, train dense, "
        "E31c and E33 loop on training data of that task only (30M, plain input, 1.5B tokens) and score that task at "
        "1k-16k. About 12.5 h per task, 3.5 days in all; data builds run ahead on the CPU.",
    ],
}

# The calibration ladder at a glance (spec §16.1): one row per run, newest last. Times UTC.
LADDER = [
    {"run": "30M round, four models", "model": "dense, no-memory control, E31c notebook, E33 loop · 30M",
     "tokens": "0.6B (3,173 steps, 0.9 passes)", "exams": "v1 (guessing 6-50 %)",
     "result": "Nothing: every model at guessing on every task at 4k.",
     "when": "9 Oct 15:48 → 10 Oct 03:07", "state": "done"},
    {"run": "Budget run", "model": "dense · 30M", "tokens": "1.5B (7,931 steps, 2.3 passes)",
     "exams": "v1 (guessing 6-50 %)",
     "result": "Lookup 47 % (25 %), latest 37 % (20 %), keyed 13 % (6 %), switched on at 0.6-0.9B tokens. No pass.",
     "when": "10 Oct 05:50 → 08:16", "state": "done"},
    {"run": "Step 1", "model": "dense · 30M", "tokens": "1.5B (7,888 steps, 1.5 passes)",
     "exams": "v2 (guessing 4-9 %)",
     "result": "Quote passes: 95 % (8 %), switched on at 0.75-1.1B tokens. Lookup and keyed at guessing; "
               "latest 26 % at 1k only (9 %).",
     "when": "10 Oct 08:17 → 10:52", "state": "done"},
    {"run": "Step 2", "model": "dense · 50M (12 layers, width 576)", "tokens": "1.5B (7,888 steps, 1.5 passes)",
     "exams": "v2 (guessing 4-9 %)",
     "result": "Lookup 19.5 % (8 %), keyed 15 % (6 %), rising slowly; latest 25.5 % at 1k only; quote 0 %. No pass.",
     "when": "10 Oct 11:24 → 15:17", "state": "done"},
    {"run": "Step 1, second seed", "model": "dense · 30M, seed 1", "tokens": "1.5B (7,888 steps)",
     "exams": "v2 (guessing 4-9 %)",
     "result": "Nothing: every task at guessing at 4k (seed 0 passed quote). Latest 19 % at 1k only.",
     "when": "10 Oct 15:18 → 17:54", "state": "done"},
    {"run": "Plain input", "model": "dense · 30M, no word-pair/triple tables", "tokens": "1.5B (7,888 steps)",
     "exams": "v2 (guessing 4-9 %)", "result": "Do the input tables delay retrieval?",
     "when": "10 Oct 18:08 → ~21:05", "state": "running"},
    {"run": "Plain input", "model": "dense · 30M, no word-pair/triple tables", "tokens": "1.5B (7,888 steps)",
     "exams": "v2 (guessing 4-9 %)",
     "result": "Quote passes 86 % (8 %); lookup 39 % (8 %), keyed 35 % (6 %), latest 32 % (9 %), count 19 % (9 %).",
     "when": "10 Oct 18:08 → 20:55", "state": "done"},
    {"run": "Round: E33 loop, plain", "model": "E31c notebook + reread loop · 30M, plain input",
     "tokens": "1.5B (7,888 steps)", "exams": "v2, 1k-16k",
     "result": "Lookup 82 % and keyed 77 % at 16k (pass; dense 0 %); 63.5 / 55.5 % at 4k (dense 39 / 35 %). "
               "Latest 79.5 % at 1k only; quote 25 %.",
     "when": "10 Oct 21:00 → 11 Oct 05:35", "state": "done"},
    {"run": "Plain input, second seed", "model": "dense · 30M, plain, seed 1", "tokens": "1.5B",
     "exams": "v2 (guessing 4-9 %)", "result": "Nothing: every task at guessing (seed 0 learned four). The plain-input gain was seed luck.",
     "when": "11 Oct 05:36 → 08:43", "state": "done"},
    {"run": "Round: E31c (plain)", "model": "E31c notebook · 30M, plain input", "tokens": "1.5B, mixed tasks",
     "exams": "v2, 1k-16k",
     "result": "Lookup 16 %, keyed 15 % at 4k, 0 % at 16k; notebook off gives the same, so it is unused. Latest 87 % at 1k.",
     "when": "11 Oct 08:44 → 12:15", "state": "done"},
    {"run": "Single-task comparison", "model": "dense, E31c, E33 loop · 30M, plain input",
     "tokens": "1.5B each, training data of one task", "exams": "that task, 1k-16k",
     "result": "T1 quote, then T2 lookup, T3 keyed, T4 latest, T5 compose, T6 count, T7 deduce: 21 runs.",
     "when": "11 Oct 12:15 → ~14 Oct", "state": "T1 done; T2 lookup running"},
]

# Plain names for the architectures (first mention = name + what it is).
NAMES = {
    "dense": ("Dense", "full attention over the whole document — the reference"),
    "local": ("No-memory control", "sees only the last 256 tokens (E31c with its notebook removed)"),
    "e31c": ("E31c notebook", "local attention plus a notebook of summaries of past text"),
    "e31c_loop": ("E33 loop", "E31c plus a read–think–reread loop (4 rounds)"),
    "dense_plain": ("Dense, plain input", "full attention, tiny token embedding only (no word-pair/triple tables)"),
    "local_plain": ("No-memory control, plain input", "last 256 tokens only; tiny token embedding only"),
    "e31c_plain": ("E31c notebook, plain input", "local attention plus the notebook; tiny token embedding only"),
    "e31c_loop_plain": ("E33 loop, plain input", "E31c plus the read-think-reread loop; tiny token embedding only"),
}

# Result rounds, keyed by "<run>.<host>" (the ledger file name). Rounds not listed here but present in the
# ledgers are shown under calibration.
ROUNDS = {
    "r1v1_screen_odra.odra": {
        "title": "30M models on the fixed exams (data v1)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "9 Oct 2026 15:48 UTC → 10 Oct 03:07 UTC",
        "verdict": "No model learned any task: every score sits at or near the guessing rate at the 4k training length. "
                   "Dense passes nothing either, so the round says nothing about the architectures — the budget is too small.",
        "observations": [
            "Language is learned well and evenly: held-out story loss 1.67–1.72 nats per token (lower is better); "
            "E33 loop lowest, the notebook model and the control 1.5 % behind dense.",
            "The copy probe: no model predicts a just-read passage when it is repeated (its surprise barely drops), "
            "so name-to-fact retrieval cannot start yet.",
            "Dense is half-way: with the fact in the document it is far less surprised by the right answer than with "
            "the fact removed (lookup 0.6 vs 3.5 nats per answer piece at 4k, lower = surer), yet its pick is still a "
            "guess. It knows the answer is a word from this document, not whose — the stage before copying in the "
            "literature. The memory models barely use the document at 4k (1.8–1.9 vs 2.3–2.4).",
            "First faint signal: quoting a sign rises late in training for dense and both notebook models "
            "(0 → 15–18 % exact at 4k), still below the 25 % guessing rate of picking one of four signs.",
            "At 1k the no-memory control answers 'where does X live now' 60 % of the time (guessing 20 %): in a "
            "short document the last move is often in its 256-token window. At 4k it falls to 5 %.",
            "The control's 19 % on picking one of sixteen homes at 4k (guessing 6 %) has the same cause: a third of the "
            "questions put the fact in the last tenth of the document, inside its window. Expected, not a skill.",
        ],
    },
    "r2_budget_dense_odra.odra": {
        "title": "Calibration ladder: dense 30M for 1.5B tokens (old exams, data v1)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "10 Oct 2026 05:50 UTC → 08:16 UTC (training 2.2 h, exams 17 min)",
        "verdict": "On the way, not passing: between 0.6B and 0.9B tokens dense learned to find whose fact is asked "
                   "— lookup 47 % at 4k (guessing 25 %), latest 37 % (20 %), keyed 13 % (6 %). Nothing reaches the "
                   "75 % pass mark; quote, compose, count and deduce stay at guessing.",
        "observations": [
            "The skill appears between steps 3,168 and 4,752 (0.6 → 0.9B tokens): lookup 25 → 46 % at 4k, then "
            "flattens (48 %, 47 %) over the last 40 % of training, while the step size decays to zero.",
            "Shorter documents are easier: lookup 64 % at 1k vs 47 % at 4k; latest 54 % vs 37 %; keyed 29 % vs 13 %.",
            "With the fact removed it never answers (0 % on every retrieval task): the gains are reading, not guessing.",
            "Graded pick (probability on the right candidate) tracks the same rise: lookup 0.24 → 0.52, latest "
            "0.29 → 0.50 (guessing 0.25 and 0.20).",
            "Quoting a 4-6 word sign is still at guessing at 4k (25 %; 34 % at 1k), and the copy probe still finds no "
            "copying of a repeated passage (1.80 → 1.64 nats per token): it finds facts before it copies text.",
            "Language keeps improving with tokens: held-out story loss 1.54 nats per token, down from 1.69 at 600M.",
        ],
    },
    "cal1_dense30m_v2_odra.odra": {
        "title": "Calibration ladder step 1: dense 30M for 1.5B tokens (new exams, data v2)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "10 Oct 2026 08:17 UTC → 10:52 UTC (training 2.2 h, exams 26 min)",
        "verdict": "Quote passes — the first calibrated task: 95 % at 4k against 8 % guessing, 92 % with 24 signs, "
                   "76 % with a sentence form never seen in training. Lookup and keyed stay at guessing; latest is "
                   "above guessing only in short documents.",
        "observations": [
            "Quote switches on abruptly: 0 % at 0.75B tokens, 47 % at 0.94B, 78 % at 1.1B, 94 % from 1.3B — the same "
            "window in which the old-exam run learned lookup.",
            "With the sign removed it never answers (0 %): it reads the right person's sign and copies it word for "
            "word.",
            "Lookup and keyed sit at guessing (6-7 % vs 8 % and 6 %), although it is far less surprised by the right "
            "home with the fact present (0.81 vs 3.98 nats per answer piece): it knows the answer is a place in "
            "this document, not whose.",
            "Latest: 26.5 % at 1k, 16.5 % at 2k, 11 % at 4k (guessing 9 %) — it tracks moves only when they are close.",
            "The copy probe on repeated stories still finds no copying (1.80 → 1.56 nats per token): copying a named "
            "sign comes before copying a whole passage.",
            "Language is the same as on the old data: held-out story loss 1.56 nats per token (1.54 on v1).",
        ],
    },
    "cal2_dense50m_v2_odra.odra": {
        "title": "Calibration ladder step 2: dense 50M for 1.5B tokens (new exams, data v2)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "10 Oct 2026 11:24 UTC → 15:17 UTC (training 3.4 h, exams 32 min)",
        "verdict": "Bigger did not mean further: lookup and keyed rise above guessing (19.5 % and 15 % at 4k vs 8 % and "
                   "6 %) but quote, which the 30M model passed on the same data, stays at 0 %. Nothing passes.",
        "observations": [
            "Lookup and keyed grow slowly over the whole run (lookup 10 % at 0.6B tokens, 14 % at 1.05B, 20 % at "
            "1.2B) — no sudden switch like the 30M model's quote.",
            "Quote is not learned at all: the right sign is no less surprising with the sign in the document than "
            "without it (3.27 vs 3.27 nats per token).",
            "Shorter documents are easier again: lookup 31 % at 1k, keyed 23 % at 1k.",
            "With the fact removed it never answers (0 %); latest 25.5 % at 1k and 10.5 % at 4k (guessing 9 %).",
            "Language is better than at 30M: held-out story loss 1.49 vs 1.56 nats per token.",
            "The copy probe still finds no copying (1.82 → 1.64 nats per token).",
            "Step size 1e-3, chosen by held-out loss over 5e-4 (3.186) and 2e-3 (4.380): 3.014.",
        ],
    },
    "cal1s1_dense30m_v2_odra.odra": {
        "title": "Calibration ladder step 1, second seed: dense 30M for 1.5B tokens (data v2, seed 1)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "10 Oct 2026 15:18 UTC → 17:54 UTC (training 2.2 h, exams 26 min)",
        "verdict": "Nothing switched on: every task at the guessing rate at 4k. The same model, data and tokens with "
                   "seed 0 passed quote at 95 %, so at 30M and 1.5B tokens learning a retrieval skill is a matter "
                   "of luck.",
        "observations": [
            "Quote 0 % at every length; the right sign is no less surprising with the sign present (3.27 vs 3.28 "
            "nats per token): no reading of signs at all.",
            "Latest is the only signal: 19 % at 1k, 9 % at 2k, 7.5 % at 4k (guessing 9 %) — moves that sit close "
            "to the question.",
            "Language is the same as seed 0: held-out story loss 1.554 vs 1.556 nats per token, so the seeds differ "
            "in what they retrieve, not in how well they write.",
            "The copy probe finds no copying (1.79 → 1.58 nats per token).",
        ],
    },
    "cal3a_plain30m_v2_odra.odra": {
        "title": "Diagnostic: dense 30M without the word-pair/triple tables, 1.5B tokens (data v2)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "10 Oct 2026 18:08 UTC → 20:55 UTC (training 2.3 h, exams 27 min)",
        "verdict": "Seed 0 only — a second seed (11 Oct) learned nothing, so this is seed luck, not the input layer. With plain input it learned four tasks — quote passes (86 % "
                   "at 4k, guessing 8 %), lookup 39 % (8 %), keyed 35 % (6 %), latest 32 % (9 %) — where the same model "
                   "with the tables learned at most one. One seed; a second is queued.",
        "observations": [
            "Quote switches on first (10 % → 74 % between 0.75B and 0.9B tokens), then lookup, keyed and latest climb "
            "together from 1.05B tokens and are still rising at the end (lookup 27 → 38 → 39 %).",
            "Shorter documents are easier: lookup 49.5 %, keyed 45 %, latest 49.5 % and quote 97.5 % at 1k.",
            "With the fact removed it never answers (0 % on every retrieval task): it reads, it does not guess.",
            "Count rises too (19 % at 4k vs 9 % guessing); composing two facts and chaining rules stay at guessing.",
            "Language is better without the tables: held-out story loss 1.53 vs 1.56 nats per token, with the tables' "
            "4.2M parameters moved into the feed-forward layers (width 1,472 vs 1,088).",
            "The copy probe still finds no copying of a whole passage (1.72 → 1.49 nats per token), the largest drop so far.",
            "Read past the training length it fails: lookup 18 % at 8k and 0 % at 16k; quote 20 % at 8k (scored 11 Oct).",
        ],
    },
    "r2v2p_screen30m_odra.odra": {
        "title": "Architecture round on plain input: 30M, 1.5B tokens (data v2)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "from 10 Oct 2026 21:00 UTC (E33 loop done 11 Oct 05:35 UTC; E31c and the control queued)",
        "verdict": "The E33 loop reads facts far past its 4k training length: lookup 82 % and keyed 77 % at 16k "
                   "(guessing 8 % and 6 %), both passing, where dense with the same input, data and tokens scores 0 %. "
                   "Its notebook carries it: switched off, lookup and keyed fall to 0 % at 16k.",
        "observations": [
            "Longer is better for E33 on lookup and keyed (63.5 → 76 → 82 % from 4k to 16k): with more filler between "
            "facts, each notebook page holds fewer of them — a guess to test.",
            "Dense reads well only inside its training length: lookup 39 % at 4k, 18 % at 8k, 0 % at 16k.",
            "Latest goes the other way: 79.5 % at 1k (a pass), 60 % at 2k, 13 % at 4k, about 2 % at 8k-16k — the last "
            "move must sit near the question.",
            "Quote is weaker than dense (25 % vs 86 % at 4k) and gone past 4k: word-for-word signs do not survive the "
            "notebook.",
            "Held-out sentence forms and 16 people instead of 12 still pass at 16k: lookup 83 % and 79 %.",
            "Language is slightly better than dense: story loss 1.52 vs 1.53 nats per token.",
            "Training took 7.0 h on 3 GPUs (58k tokens/s, a 20-minute cooldown after 6 h), 3× dense.",
            "E31c without the loop (12:15 UTC 11 Oct) does not use its notebook: lookup 16 % at 4k and 0 % at 16k, "
            "the same with the notebook off. Latest 87 % at 1k. So the reread loop is what makes the notebook read.",
        ],
    },
    "pt_quote_30m_odra.odra": {
        "title": "Single-task T1 quote: dense, E31c, E33 loop trained on quote only (30M, plain input, 1.5B tokens)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "11 Oct 2026 12:15 UTC → 12 Oct 01:04 UTC (dense 2.5 h, E31c 2.7 h, E33 7.6 h incl. exams)",
        "verdict": "Nobody learned quote from memory. Dense stayed at guessing (8 % at 4k); E31c and E33 score 33 % at "
                   "1k and 25 % at 4k but 0 % from 8k, and exactly the same with the notebook off — they read only "
                   "the signs that fall in their last 256 tokens. Training on one task did not make dense learn.",
        "observations": [
            "The window effect: about a third of the questions put the sign late in the document, inside the 256-token "
            "window, which matches the 25-33 % and the 0 % at 8k-16k.",
            "Dense with mixed training learned quote in one seed of four (95 %); quote-only training did not help it.",
            "Next: T2 lookup (dense from 01:04 UTC 12 Oct), where E33 had its mixed-training win.",
        ],
    },
}

SUPERSEDED = {
    "r1_screen.polonez": "30M round on the first-draft exams (v0), Polonez, 7 Oct: all four models trained "
                         "(dense 05:21, no-memory 06:08, notebook 06:56, E33 loop 09:20 UTC); the exams were cut "
                         "off by the Polonez shutdown that day. Superseded by the v1 round (shortcuts fixed).",
}
