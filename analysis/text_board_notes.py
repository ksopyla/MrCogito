"""Hand-written text of the text capability board (the mission-control page of the text checks).

Numbers on the board come from the ledgers; this file holds only what a person has to judge:
  STATUS        what runs now, what is next, what blocks us;
  PROTOCOL_LOG  every change to exams, recipe or reading rules, and what it did to earlier results;
  CLAIMS        every conclusion drawn, with its state (holds / single run / open / retracted);
  LADDER        every run with its standing (in use / calibration / history / running / queued) and why;
  ROUNDS        per run: verdict and observations.
Never delete a run or a claim: change its standing or state and say why (skill `text-checks`, "Board").
Plain words, every number with its meaning (research-comms). Update it whenever a run lands or a rule changes.
"""

# The main table (top of the board): the comparison in use — plain input, 30M, 1.5B tokens, exams v2, one question
# per training document — in two setups. Only runs whose standing is "in use" (LADDER below) belong here.
SCOREBOARD = {
    "setups": [
        {"key": "mixed", "title": "Mixed training (all seven tasks in one model)",
         "runs": ["cal3a_plain30m_v2_odra.odra", "cal3a_s1_plain30m_v2_odra.odra", "r2v2p_screen30m_odra.odra"]},
        {"key": "single", "title": "Single-task training (one model per task) — paused", "prefix": "pt_"},
    ],
    # single-task work that was queued with no run folder yet (in order); paused 2026-10-13, see PROTOCOL_LOG
    "single_queue": [],
}

# Mission control: what is running, what is next, what blocks us. Times UTC.
STATUS = {
    "as_of": "2026-10-13 11:35 UTC",
    "headline": "Trust the measurements, not yet the ranking. Every score on this board is measured correctly, with "
                "the reading rules of 13 Oct applied to all of them. What is missing is reliable learning: dense learns "
                "a skill in only some runs at this budget, and now E33's one strong win did not repeat either (trained "
                "on lookup only, it read no far facts). No architecture comparison is settled until training is reliable.",
    "where": [
        "Settled: E31c without the loop does not use its notebook (three runs). Dense at 30M / 1.5B learns by luck "
        "(two of four runs learned nothing).",
        "E33's strong signal (far facts at 16k: lookup 83 %, keyed 78 %, mixed training) did not repeat when E33 was "
        "trained on lookup only (13 Oct 11:12 UTC): 0 % on far facts at every length; its 62-83 % came from facts "
        "inside its window. Dense trained the same way: 48 % on far facts at 4k.",
        "Reading rules of 13 Oct: quote, lookup and keyed are read on facts placed early or in the middle of the "
        "story; latest's floor is 20 %; dense is not compared past 4k. Nothing had to be retrained.",
    ],
    "next": [
        "Running on Odra since 11:12 UTC: the ten finished models scored on the added exams (crowded facts, more "
        "unseen sentence shapes, one-step compose and deduce; built 07:34 UTC) — no retraining; about 13:45 UTC.",
        "Then the reliability fix: dense 30M, 8 questions per training document, three runs of about 3 h each (done "
        "about 22:45 UTC). If dense learns in at least 2 of 3, the same recipe goes to E33 next — its notebook reading "
        "looks as luck-dependent as dense's retrieval.",
        "Odra stopped suddenly at 09:52 UTC 12 Oct (3.6 h into training, nothing logged: power loss or hard freeze) "
        "and was back at 07:18 UTC 13 Oct; E33's lookup run resumed from half-way.",
    ],
}

# Protocol log: every change to the exams, the training recipe or the reading rules, newest last. Each entry says what
# it did to the results already on the board: "re-read" (same measurements, read with new rules — no retraining),
# "history" (older exams or recipe — kept, not compared with current runs), or "planned".
PROTOCOL_LOG = [
    {"date": "2026-10-06", "version": "exams v0", "kind": "exams",
     "change": "First exams and training data (published on Hugging Face).",
     "why": "A from-scratch text testbed with facts, questions and a length ladder.",
     "effect": "history",
     "effect_text": "The 7 Oct round on Polonez never finished its exams. A model-free audit (8 Oct) then found five "
                    "shortcuts (e.g. lookup answerable 98 % by copying the only place name).",
     "learned": "Every exam needs a model-free shortcut audit before any model is trusted on it."},
    {"date": "2026-10-08", "version": "exams v1", "kind": "exams",
     "change": "The five shortcuts fixed; four candidates per question in most tasks.",
     "why": "v0 exams could be passed without reading.",
     "effect": "history",
     "effect_text": "Runs on v1 (the 30M round, 600M tokens; the 1.5B-token dense run) are kept as history: guessing "
                    "rates of 20-50 % made small gains unreadable, and v2 changed the exams.",
     "learned": "600M tokens taught no model anything; at 1.5B tokens dense switched on retrieval at 0.6-0.9B."},
    {"date": "2026-10-10", "version": "exams v2", "kind": "exams",
     "change": "Every guessing rate below 10 % (12 signs, 12 homes, 10 moves …); candidate places start with different "
               "letters; a graded score (probability on the right candidate).",
     "why": "Small progress near guessing was invisible at 25-50 % guessing rates.",
     "effect": "current",
     "effect_text": "All current runs use v2. v1 results are not compared with v2 results.",
     "learned": "A frozen exam with every rate below 10 % shows when a skill switches on."},
    {"date": "2026-10-10", "version": "plain input", "kind": "recipe",
     "change": "Architecture rounds run without the word-pair/triple input tables (plain input).",
     "why": "Dense without the tables learned four tasks where with them it learned one.",
     "effect": "current",
     "effect_text": "Retracted as a finding on 11 Oct 08:43 (a second run learned nothing), kept as the setup for "
                    "consistency. No rerun needed: every model in the comparison has the same input.",
     "learned": "One run is not evidence at this budget: which skills switch on is a matter of the seed."},
    {"date": "2026-10-11", "version": "single-task series", "kind": "recipe",
     "change": "One model per task (dense, E31c, E33), seven tasks in sequence, one run each.",
     "why": "Remove task interference and see who wins each task.",
     "effect": "paused",
     "effect_text": "Paused 13 Oct after quote and lookup: with one run each it measures luck (dense trained on quote "
                    "only learned nothing, where two mixed-training runs passed quote). Its results stay on the board.",
     "learned": "More examples of one task did not make dense learn it; the luck is not about task data."},
    {"date": "2026-10-13", "version": "reading rules", "kind": "reading",
     "change": "Quote, lookup and keyed are read on facts placed early or in the middle (far); facts at the very end "
               "(near) are shown apart. Latest's floor is 20 %. \"Learned\" = 10 points or 3 standard errors above the "
               "floor. Leaders only at the 4k training length; dense is not compared past it.",
     "why": "Protocol review: near facts sit inside every model's short window, so a model with no memory scored "
            "there; a reader ignoring whose move it is already gets 1 in 5 on latest; dense has no method for reading "
            "past 4k.",
     "effect": "re-read",
     "effect_text": "Every v2 result re-read from the stored per-position scores — nothing retrained. Changed "
                    "conclusions: E31c's quote, lookup and keyed scores were all near facts (0 % far); E31c's and "
                    "E33's latest at 4k are below the 20 % floor; dense's 0 % at 16k is not a memory comparison.",
     "learned": "Read memory only on facts beyond the short window, and compare long lengths only against a "
                "reference that can read them."},
    {"date": "planned", "version": "8 questions per document", "kind": "recipe",
     "change": "Training documents end with 8 questions instead of 1 (exams unchanged, byte-identical).",
     "why": "Answers are about 0.1 % of training tokens; dense learns retrieval late and only in some runs.",
     "effect": "planned",
     "effect_text": "If dense learns in at least 2 of 3 runs, the architecture comparison is redone on this recipe "
                    "with two runs per model; the 1-question rounds become history (\"earlier recipe\").",
     "learned": ""},
    {"date": "planned", "version": "exams v2.1", "kind": "exams",
     "change": "New exams added, nothing existing changed: crowded facts (length without spreading the facts), "
               "more unseen sentence shapes, one-step compose and deduce.",
     "why": "Length also spread the facts out (E33 did better at 16k than at 1k); one unseen sentence shape per fact "
            "type is thin; compose and deduce showed no signal at all.",
     "effect": "planned",
     "effect_text": "Implemented and audited on the laptop 13 Oct (a rule reader answers 100 %, shortcuts stay at guessing; "
                    "existing exams and training documents byte-identical). Built on Odra and scored on the existing "
                    "checkpoints when Odra is back — no retraining.",
     "learned": ""},
]

# Claims register: every conclusion we have drawn, with its state. holds = repeated or overwhelming within its runs;
# single run = one run, needs a second; open = not testable yet; retracted = disproved (kept, with the date).
CLAIMS = [
    {"claim": "E31c without the loop does not use its notebook.", "state": "holds", "since": "2026-10-11",
     "evidence": "Three runs (mixed, quote-only, lookup-only): 0 % on far facts at every length, and the same scores "
                 "with the notebook switched off.",
     "settle": "—"},
    {"claim": "Dense at 30M / 1.5B tokens learns retrieval only in some runs.", "state": "holds", "since": "2026-10-11",
     "evidence": "Four mixed-training runs: one passed quote only, one learned four tasks, two learned nothing; the "
                 "quote-only run learned nothing.",
     "settle": "The 8-questions-per-document recipe: does dense learn in 2 of 3 runs?"},
    {"claim": "E33's notebook reads far facts past the training length.", "state": "single run", "since": "2026-10-13",
     "evidence": "Mixed training (11 Oct): lookup 83 % and keyed 78 % at 16k on far facts; notebook off: 0 %. Not "
                 "repeated: E33 trained on lookup only (13 Oct) read 0 % of far facts at every length, its 62-83 % all "
                 "from facts inside its window — so whether the notebook learns to read looks like luck, as for dense.",
     "settle": "E33 on the reliable recipe (8 questions per document), two runs."},
    {"claim": "E33 keeps reading when the fact is phrased in a sentence shape never seen in training; dense does not.",
     "state": "single run", "since": "2026-10-13",
     "evidence": "Lookup on far facts at 4k: E33 64 % with training shapes, 64 % with the unseen shape (its near "
                 "facts, read by short-range attention, drop 62 → 31 %); dense 37-48 % → 0-1 % (three dense runs).",
     "settle": "A second E33 run; more unseen shapes (exams v2.1)."},
    {"claim": "Word-for-word quotes do not survive the notebook.", "state": "holds", "since": "2026-10-12",
     "evidence": "E31c and E33, two runs each: quote 0 % on far facts; their 25-33 % came from signs inside the window.",
     "settle": "—"},
    {"claim": "Dense tracks the latest move better than the memory models.", "state": "single run", "since": "2026-10-13",
     "evidence": "Latest at 4k: dense 32 % in one of two runs; E31c 17 % and E33 13 %, both below the 20 % floor.",
     "settle": "Dense learning reliably (8-questions recipe), then two runs per model."},
    {"claim": "Memory models beat rereading on long documents.", "state": "open", "since": "2026-10-13",
     "evidence": "Dense scores 0 % past 8k on every task, but it has no method for reading past its training length, "
                 "and the no-memory control has not been run on v2.",
     "settle": "The no-memory control and a dense model with a long-text method on the same exams."},
    {"claim": "E33 reads better at 16k than at 1k because the facts are less crowded.", "state": "open",
     "since": "2026-10-11",
     "evidence": "E33 on far facts: about 30 % at 1k, 73-93 % at 16k. A guess about the cause.",
     "settle": "The crowded-facts exam (v2.1) on the existing E33 checkpoint."},
    {"claim": "Nobody composes two facts or chains rules at 30M / 1.5B.", "state": "holds", "since": "2026-10-10",
     "evidence": "Every run at guessing on compose and deduce, including the probability on the right candidate.",
     "settle": "One-step versions (v2.1) to see whether any part is learned."},
    {"claim": "Plain input (no word-pair/triple tables) makes dense learn.", "state": "retracted", "since": "2026-10-11",
     "evidence": "Seed 0 learned four tasks (10 Oct); seed 1 learned nothing (11 Oct 08:43).",
     "settle": "—"},
    {"claim": "A 50M dense learns more than 30M at equal tokens.", "state": "open", "since": "2026-10-10",
     "evidence": "One 50M run: lookup and keyed rose slowly (20 % and 15 % at 4k), quote not at all.",
     "settle": "After the recipe fix, if ever needed."},
]

# All runs, oldest first. standing: "in use" (feeds the scoreboard), "calibration" (dense alone; sets the recipe and
# budget; valid evidence about learnability), "history" (older exams or recipe: kept, not compared with current runs),
# "running" / "queued" / "blocked". `why` says what the standing means for this run. `id` joins the ledger.
LADDER = [
    {"id": "r1_screen.polonez", "start": "2026-10-07", "data": "v0", "standing": "history",
     "run": "First round, four models", "model": "dense, no-memory control, E31c, E33 loop · 30M",
     "tokens": "0.6B", "exams": "v0", "result": "Exams cut off by the Polonez shutdown.",
     "when": "7 Oct", "why": "Exams v0 had five shortcuts; superseded by v1."},
    {"id": "r1v1_screen_odra.odra", "start": "2026-10-09", "data": "v1", "standing": "history",
     "run": "30M round, four models", "model": "dense, no-memory control, E31c notebook, E33 loop · 30M",
     "tokens": "0.6B (3,173 steps, 0.9 passes)", "exams": "v1 (guessing 6-50 %)",
     "result": "Nothing: every model at guessing on every task at 4k.",
     "when": "9 Oct 15:48 → 10 Oct 03:07", "why": "Older exams (v1) and a budget too small for any model to learn; "
                                                  "taught us the token budget."},
    {"id": "r2_budget_dense_odra.odra", "start": "2026-10-10", "data": "v1", "standing": "history",
     "run": "Budget run", "model": "dense · 30M", "tokens": "1.5B (7,931 steps, 2.3 passes)",
     "exams": "v1 (guessing 6-50 %)",
     "result": "Lookup 47 % (25 %), latest 37 % (20 %), keyed 13 % (6 %), switched on at 0.6-0.9B tokens. No pass.",
     "when": "10 Oct 05:50 → 08:16", "why": "Older exams (v1); set the 1.5B-token budget."},
    {"id": "cal1_dense30m_v2_odra.odra", "start": "2026-10-10", "data": "v2", "standing": "calibration",
     "run": "Step 1", "model": "dense · 30M", "tokens": "1.5B (7,888 steps, 1.5 passes)", "exams": "v2 (guessing 4-9 %)",
     "result": "Quote passes: 96 % on far facts (8 %). Lookup and keyed at guessing; latest at 4k below its 20 % floor.",
     "when": "10 Oct 08:17 → 10:52", "why": "Dense with the input tables; evidence of the seed lottery."},
    {"id": "cal2_dense50m_v2_odra.odra", "start": "2026-10-10", "data": "v2", "standing": "calibration",
     "run": "Step 2", "model": "dense · 50M (12 layers, width 576)", "tokens": "1.5B (7,888 steps, 1.5 passes)",
     "exams": "v2 (guessing 4-9 %)",
     "result": "Lookup and keyed learned a little (20 % and 18 % on far facts at 4k), quote 0 %. No pass.",
     "when": "10 Oct 11:24 → 15:17", "why": "Bigger, with the input tables; one run."},
    {"id": "cal1s1_dense30m_v2_odra.odra", "start": "2026-10-10", "data": "v2", "standing": "calibration",
     "run": "Step 1, second seed", "model": "dense · 30M, seed 1", "tokens": "1.5B (7,888 steps)",
     "exams": "v2 (guessing 4-9 %)", "result": "Nothing learned (seed 0 passed quote).",
     "when": "10 Oct 15:18 → 17:54", "why": "Evidence of the seed lottery."},
    {"id": "cal3a_plain30m_v2_odra.odra", "start": "2026-10-10", "data": "v2", "standing": "in use",
     "run": "Plain input", "model": "dense · 30M, no word-pair/triple tables", "tokens": "1.5B (7,888 steps)",
     "exams": "v2 (guessing 4-9 %)",
     "result": "Quote passes (84 % far); lookup 37 %, keyed 33 % (far facts), latest 32 % (floor 20 %), count 19 %.",
     "when": "10 Oct 18:08 → 20:55", "why": "The dense reference of the mixed-training comparison (seed 0)."},
    {"id": "r2v2p_screen30m_odra.odra", "start": "2026-10-10", "data": "v2", "standing": "in use",
     "run": "Round: E33 loop and E31c, plain", "model": "E33 loop, E31c notebook · 30M, plain input",
     "tokens": "1.5B (7,888 steps)", "exams": "v2, 1k-16k",
     "result": "E33: lookup 64 % and keyed 56 % on far facts at 4k, 83 % and 78 % at 16k; notebook off: 0 %. "
               "E31c: 0 % on far facts.",
     "when": "10 Oct 21:00 → 11 Oct 12:15", "why": "The memory models of the mixed-training comparison."},
    {"id": "cal3a_s1_plain30m_v2_odra.odra", "start": "2026-10-11", "data": "v2", "standing": "in use",
     "run": "Plain input, second seed", "model": "dense · 30M, plain, seed 1", "tokens": "1.5B",
     "exams": "v2 (guessing 4-9 %)", "result": "Nothing learned (seed 0 learned four tasks).",
     "when": "11 Oct 05:36 → 08:43", "why": "The dense reference of the mixed-training comparison (seed 1)."},
    {"id": "pt_quote_30m_odra.odra", "start": "2026-10-11", "data": "v2", "standing": "in use",
     "run": "Single task: quote", "model": "dense, E31c, E33 loop · 30M, plain", "tokens": "1.5B each, quote only",
     "exams": "quote, 1k-16k", "result": "Nobody learned quote on far facts (dense 6 %, E31c and E33 0 %).",
     "when": "11 Oct 12:15 → 12 Oct 01:04", "why": "Single-task series (paused); one run per model."},
    {"id": "pt_lookup_30m_odra.odra", "start": "2026-10-12", "data": "v2", "standing": "in use",
     "run": "Single task: lookup", "model": "dense, E31c, E33 loop · 30M, plain", "tokens": "1.5B each, lookup only",
     "exams": "lookup, 1k-16k",
     "result": "Dense 48 % on far facts at 4k (0 % past 4k); E31c and E33 0 % on far facts — both read only facts "
               "inside their window (E33 near facts 62-83 %). E33's mixed-training win did not repeat.",
     "when": "12 Oct 01:04 → 13 Oct 11:12", "why": "Single-task series (paused); E33 was cut off half-way by the Odra "
                                                   "outage and resumed 07:19 UTC 13 Oct."},
    {"id": None, "start": "2026-10-13", "data": "v2.1", "standing": "running",
     "run": "Added exams on finished models", "model": "E33, E31c, dense (two runs), and the single-task models",
     "tokens": "no training", "exams": "v2.1: crowded facts, more unseen shapes, one step",
     "result": "Does crowding explain why E33 reads better at 16k? Does E33's reading of unseen shapes hold?",
     "when": "11:12 → ~13:45", "why": "Scores existing checkpoints; nothing is retrained."},
    {"id": None, "start": "planned", "data": "v2", "standing": "queued",
     "run": "Reliability fix", "model": "dense · 30M, plain, three seeds", "tokens": "1.5B, 8 questions per document",
     "exams": "v2, 1k-4k (+ 16k)", "result": "Does dense learn in at least 2 of 3 runs?",
     "when": "~13:45 → ~22:45 (13 Oct)", "why": "Decides the recipe for the architecture comparison."},
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
    "pt_lookup_30m_odra.odra": {
        "title": "Single-task T2 lookup: dense, E31c, E33 loop trained on lookup only (30M, plain input, 1.5B tokens)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "12 Oct 2026 01:04 UTC → 13 Oct 11:12 UTC (dense done 03:37, E31c 06:14; E33 cut off half-way by the "
                 "Odra outage at 09:52 UTC 12 Oct, resumed 07:19 UTC 13 Oct)",
        "verdict": "Dense learned lookup inside its training length (48 % on far facts at 4k, guessing 8 %); neither "
                   "memory model read its notebook: E31c and E33 score 0 % on far facts at every length. E33's "
                   "mixed-training win (83 % at 16k) did not repeat.",
        "observations": [
            "E33 learned lookup only through short-range attention: 83 % at 1k, 62 % at 4k on facts at the end of the "
            "story, 0 % on facts early or in the middle, 0 % at 8k-16k; its learning curve on far facts stays at 0 %.",
            "E31c the same, weaker: near facts 41-62 %, far facts 0 %.",
            "Dense's lookup switched on between 0.6B and 0.9B tokens (far facts 12 → 37 %), then 48 % at the end; "
            "with an unseen sentence shape it falls to 1 %.",
            "Language: story loss E33 1.511, dense 1.520, E31c 1.554 nats per token (lower is better).",
            "Training on lookup alone did not make the notebook read: the same E33 with mixed training did (one run "
            "each), so the notebook switching on looks like luck at this budget, as dense's retrieval does.",
        ],
    },
    "cal3a_s1_plain30m_v2_odra.odra": {
        "title": "Dense 30M plain input, second seed, 1.5B tokens (data v2, mixed training)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "11 Oct 2026 05:36 UTC → 08:43 UTC (training 2.4 h)",
        "verdict": "Nothing learned: every task at guessing at 4k. Seed 0 of the same setup learned four tasks, so the "
                   "plain-input gain was seed luck.",
        "observations": [
            "Across four dense 30M / 1.5B mixed-training runs, two learned something (one with the input tables, one "
            "without) and two learned nothing.",
        ],
    },
}

SUPERSEDED = {
    "r1_screen.polonez": "30M round on the first-draft exams (v0), Polonez, 7 Oct: all four models trained "
                         "(dense 05:21, no-memory 06:08, notebook 06:56, E33 loop 09:20 UTC); the exams were cut "
                         "off by the Polonez shutdown that day. Superseded by the v1 round (shortcuts fixed).",
}
