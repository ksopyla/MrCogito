"""Hand-written text of the text capability board: the status brief and each round's verdict.

Numbers on the board come from the ledgers; this file holds only what a person has to judge:
the one-line status, where we are, what is next, and per round the verdict and observations.
Plain words, every number with its meaning (research-comms). Update it whenever a round lands.
"""

STATUS = {
    "as_of": "2026-10-10 15:35 UTC",
    "headline": "Each run switches on a different retrieval task, and none switches on general copying: the 30M model "
                "learned quote (95 %), the 50M model learned lookup and keyed partly (20 % and 15 % at 4k vs 8 % and "
                "6 %) but not quote. Bigger did not mean further. A second 30M seed is running.",
    "where": [
        "Ladder step 2, dense 50M, 1.5B tokens, exams v2 (Odra, 10 Oct 11:24-15:17 UTC): lookup 19.5 % and keyed "
        "15 % at 4k, rising slowly over the whole run; quote 0 %; better language (story loss 1.49 vs 1.56 at 30M).",
        "Ladder step 1, dense 30M, same data and tokens (08:17-10:52 UTC): quote passes (95 % vs 8 %); lookup and "
        "keyed at guessing.",
        "On the old exams the 30M model learned lookup (47 % vs 25 %) and not quote. No run copies a repeated "
        "passage (copy probe), so each run seems to build its own task-specific retrieval rather than one copy skill.",
    ],
    "next": [
        "Running on Odra since 15:18 UTC: dense 30M with a second seed on v2 (same data and tokens), scores about "
        "17:55 UTC — does quote switch on again, or something else?",
        "Proposed next (author's call): the dense model without the word-pair/triple input tables, and training "
        "documents with several questions each instead of one (answers are about 0.1 % of training tokens).",
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
     "exams": "v2 (guessing 4-9 %)", "result": "Does the same model and data switch on the same task?",
     "when": "10 Oct 15:18 → ~17:55", "state": "running"},
]

# Plain names for the architectures (first mention = name + what it is).
NAMES = {
    "dense": ("Dense", "full attention over the whole document — the reference"),
    "local": ("No-memory control", "sees only the last 256 tokens (E31c with its notebook removed)"),
    "e31c": ("E31c notebook", "local attention plus a notebook of summaries of past text"),
    "e31c_loop": ("E33 loop", "E31c plus a read–think–reread loop (4 rounds)"),
    "dense_plain": ("Dense, plain input", "diagnostic: dense without the n-gram input layer"),
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
}

SUPERSEDED = {
    "r1_screen.polonez": "30M round on the first-draft exams (v0), Polonez, 7 Oct: all four models trained "
                         "(dense 05:21, no-memory 06:08, notebook 06:56, E33 loop 09:20 UTC); the exams were cut "
                         "off by the Polonez shutdown that day. Superseded by the v1 round (shortcuts fixed).",
}
