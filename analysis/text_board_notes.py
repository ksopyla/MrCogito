"""Hand-written text of the text capability board: the status brief and each round's verdict.

Numbers on the board come from the ledgers; this file holds only what a person has to judge:
the one-line status, where we are, what is next, and per round the verdict and observations.
Plain words, every number with its meaning (research-comms). Update it whenever a round lands.
"""

STATUS = {
    "as_of": "2026-10-10 05:50 UTC",
    "headline": "No model has learned any task yet, the dense reference included. The training budget is too small "
                "for these exams, so no architecture comparison is possible so far.",
    "where": [
        "Exams fixed (data v1): four of seven tasks could be passed by shortcuts in the first draft; an audit of all "
        "29,400 questions now finds every shortcut at the guessing rate.",
        "30M round on v1 finished on Odra (9–10 Oct): four models, 600M tokens each, all at the guessing rate at 4k.",
        "None of the four can yet copy back a passage it has just read — the skill every task builds on.",
    ],
    "next": [
        "Running since 10 Oct 05:50 UTC on Odra: the dense model alone at 30M for 1.5B tokens (7,931 steps, about "
        "2.2 h + exams), scored along the way — step 1 of the calibration ladder.",
        "Then dense at 50M on the same tokens (capacity or steps?), then 100M with more tokens; the four-model "
        "round at the first budget where dense passes.",
    ],
}

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
}

SUPERSEDED = {
    "r1_screen.polonez": "30M round on the first-draft exams (v0), Polonez, 7 Oct: all four models trained "
                         "(dense 05:21, no-memory 06:08, notebook 06:56, E33 loop 09:20 UTC); the exams were cut "
                         "off by the Polonez shutdown that day. Superseded by the v1 round (shortcuts fixed).",
}
