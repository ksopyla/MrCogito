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
        {"key": "q8", "title": "Current recipe: mixed training, 8 questions per training document",
         "runs": ["q8_dense30m_s0_odra.odra", "q8_dense30m_s1_odra.odra", "q8_dense30m_s2_odra.odra",
                  "q8r_e33_s0_odra.odra", "q8r_e33_s1_odra.odra", "q8r_e31c_s0_odra.odra", "q8r_e31c_s1_odra.odra", "q8r_e31c_s2_polonez.polonez",
                  "q8r_e33_s2_odra.odra", "q8r_e33d_s0_polonez.polonez", "q8r_e33d_s1_polonez.polonez",
                  "q8r_e33d_s2_polonez.polonez"],
         "families": ["dense", "e31c", "e33", "e33d"]},
        {"key": "mixed", "title": "Earlier recipe (history): mixed training, 1 question per document",
         "runs": ["cal3a_plain30m_v2_odra.odra", "cal3a_s1_plain30m_v2_odra.odra", "r2v2p_screen30m_odra.odra"]},
    ],
    # families queued on a setup with no run folder yet (every task)
    "queued": {"q8": ["e31c", "e33d"]},
    "single_queue": [],
}

# Mission control: what is running, what is next, what blocks us. Times UTC.
STATUS = {
    "as_of": "2026-10-16 06:30 UTC",
    "headline": "E31c, the notebook model without the loop, is the best architecture so far: on the current recipe it "
                "learned to read far facts in 3 of 3 runs, better than dense at 4k and still at 16k. Nobody tracks the "
                "latest move beyond 2k yet.",
    "where": [
        "Checks without training (16 Oct 05:53-06:45 UTC, Odra, E33 session). (1) Ranking each note among the notes "
        "that match the question about as well as its best one, one fixed strength on trained weights (exam set): E33 "
        "run 1 latest at 16k 5 → 41 % with far lookup 92 → 91 %; E33d run 1 2 → 30 % with lookup 93 → 92 %; keyed "
        "−2 to −5. A fixed page distance had bought 33 % only by halving far lookup. The match gate was chosen on fresh "
        "questions, not the exams. (2) E33 run 1 with its loop stopped after 1, 2 or 3 rounds scores like 4 rounds on "
        "every task (within 2 points): the loop does no task work. (3) By true count, every model gets 0 visits right "
        "most often and bunches the rest: no model counts.",
        "Started 16 Oct (author: fixes for T4-T7 and the next runs): E33e, E33 plus that ranking with strengths "
        "learned from 0 (Odra); dense, then E33, on the bridge-question data — compose and deduce training documents "
        "also ask the single steps of their chain (Polonez).",
        "Fixed page distance on E33d run 1 (exams only, 15 Oct 16:06-16:20 UTC, E33 session): +1 on every read head "
        "lifts latest at 16k from 2 % to 33 % (the last move picked 7 % → 47 %) but halves far lookup (93 → 52 %); +3 "
        "and +10 erase far facts; +10 on 4 of 9 heads keeps far facts and leaves latest at guessing. The order signal "
        "is there; it has to be trained, from slopes that do not start at 0.",
        "E31c run 3 (Polonez, 15 Oct 14:18 UTC): notebook switched on again, all three fact tasks — far facts at 4k / "
        "16k lookup 77 / 92 %, keyed 70 / 91 %, quote 68 / 90 %; learned between 0.6B and 0.9B tokens. Crowded lookup 45 % "
        "at 4k (dense 54-58 %); latest 22 % (floor 20 %); story loss 1.564.",
        "E33 run 3 (15 Oct 11:14 UTC): notebook switched on for lookup and keyed (far facts at 4k / 16k: 76 / 93 %, "
        "68 / 89 %; learned by 0.6B tokens) but not for quotes (far 0 % at every length; near quotes 77-100 % up to "
        "4k). E33's notebook now reads in 2 of 3 runs on this recipe; E31c's in 2 of 2. Latest 20 % (floor 20 %); "
        "story loss 1.519.",
        "E33d stopped (15 Oct 10:49 UTC): run 2 switched its notebook on too (far lookup 75 % at 4k, 93 % at 16k) and "
        "latest stayed at guessing (16 % at 4k, floor 20 %; 1-2 % at 8k-16k). Both switched-on runs are under the stop "
        "mark set before the runs (latest ≤ 30 % at 4k); the 9 page-distance slopes, which start at 0, never grew "
        "(largest 0.14 and 0.09; telling near-identical notes apart needs about 3-5). Whether a larger page distance "
        "would fix latest is untested.",
        "E33d run 1 (E33 plus a page distance on its notes; Polonez, 15 Oct 02:48 UTC): notebook switched on, the "
        "earliest of any run (far lookup 50 % at 0.3B tokens); far facts at 4k / 16k lookup 75 / 93 %, keyed 68 / 93 %, "
        "quote 69 / 82 %. Latest unchanged: 21 % at 4k (floor 20 %), 2 % at 8k-16k. The 9 page-distance slopes stayed "
        "near 0 (largest 0.14): the model did not use them. Story loss 1.521 (E33 1.520).",
        "E31c run 2 (15 Oct 01:49 UTC): far facts at 4k / 16k — lookup 97 / 99 %, keyed 92 / 97 %, quote 88 / 87 %; "
        "crowded lookup 75 % at 4k (dense 54-58 %, E33 44 %); latest 24 % (floor 20 %); story loss 1.564 again.",
        "E31c (notebook, no loop) run 1 on the new recipe (14 Oct 21:44 UTC): its notebook switched on — far facts at "
        "4k / 16k: lookup 75 / 93 %, keyed 72 / 91 %, quote 77 / 90 %; notebook off 0 %. About E33 run 1's level, at a "
        "third of the training time (2.4 h vs 7 h). Same weak spots: crowded facts (lookup 43 % at 4k) and latest (23 %, "
        "floor 20 %). Story loss 1.564 (lower is better; dense 1.537-1.548, E33 1.520). On the earlier recipe E31c "
        "never used its notebook.",
        "E33 run 2 on the new recipe (14 Oct 17:34 UTC): far facts 13-15 % at 4k and 16k (guessing 6-8 %), rising "
        "from 0 % at 0.9B tokens to 13 % at 1.5B — the notebook was switching on as training ended; facts near the "
        "question 66-100 %, notebook off 0 %.",
        "E33 run 1 on the new recipe (14 Oct 08:13 UTC): far facts at 4k / 16k — lookup 70 / 92 %, keyed 71 / 90 %, quote "
        "71 / 89 % (passes at 8k-16k); it learned them by 0.6B tokens, earlier than dense. Weaker where facts are "
        "crowded (lookup 44 % at 4k, dense 54-58 %) and on latest (22 %, floor 20 %).",
        "New recipe (8 questions per training document), three dense runs: lookup and keyed learned in 3 of 3 (46-63 % "
        "on far facts at 4k), latest and count in 2 of 3, quote in 1 of 3 (94 %, a pass). With 1 question per "
        "document two of four runs had learned nothing.",
        "Settled: E31c without the loop does not use its notebook (three runs). Dense at 30M / 1.5B learns by luck "
        "(two of four runs learned nothing).",
        "E33's strong signal (far facts at 16k: lookup 83 %, keyed 78 %, mixed training) did not repeat when E33 was "
        "trained on lookup only (13 Oct 11:12 UTC): 0 % on far facts at every length; its 62-83 % came from facts "
        "inside its window. Dense trained the same way: 48 % on far facts at 4k.",
        "Added exams (scored 12:05 UTC on ten finished models, no retraining): most of E33's long-document gain was "
        "the facts being spread out — with every fact packed into 1,024 tokens it reads lookup 33 % at 16k (83 % "
        "spread), about its 1k level, and dense matches it at 4k (42 % vs 39 %). With three more unseen sentence "
        "shapes E33 reads lookup 30-45 %, dense 1-4 %. One-step compose and deduce: every model at guessing.",
        "Reading rules of 13 Oct: quote, lookup and keyed are read on facts placed early or in the middle of the "
        "story; latest's floor is 20 %; dense is not compared past 4k. Nothing had to be retrained.",
    ],
    "next": [
        "Polonez: dense on the bridge data from 06:08 UTC 16 Oct (about 2.5 h with exams), then E33 on the same data "
        "after a 20-minute pause (about 6.5 h): done about 15:30 UTC.",
        "Odra: E33e from about 06:40 UTC 16 Oct (about 8.5 h with exams): done about 15:00 UTC.",
        "Open for the author: whether to start the notebook-proportions sweep; give E33 more tokens or keep the "
        "equal budget.",
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
    {"date": "2026-10-13", "version": "exams v2.1", "kind": "exams",
     "change": "New exams added, nothing existing changed: crowded facts (length without spreading the facts), "
               "more unseen sentence shapes, one-step compose and deduce.",
     "why": "Length also spread the facts out (E33 did better at 16k than at 1k); one unseen sentence shape per fact "
            "type is thin; compose and deduce showed no signal at all.",
     "effect": "current",
     "effect_text": "Added to the v2 builds on Odra 07:34-07:49 UTC 13 Oct (identical files in both; existing exams and "
                    "training documents byte-identical, 558 outputs compared) and scored on ten finished models 11:12-12:05 "
                    "UTC — no retraining. Shown under each run as other exam sets.",
     "learned": "E33's 16k gain was mostly the facts being spread out; compose and deduce fail even at one step; E33 "
                "reads unseen sentence shapes only partly (30-45 %)."},
    {"date": "2026-10-13", "version": "8 questions per document", "kind": "recipe",
     "change": "Training documents end with 8 questions instead of 1 (exams unchanged, byte-identical).",
     "why": "Answers are about 0.1 % of training tokens; dense learned retrieval late and only in some runs.",
     "effect": "current",
     "effect_text": "Adopted 20:40 UTC 13 Oct after three dense runs: lookup and keyed learned in 3 of 3 (46-63 % on far "
                    "facts at 4k), latest and count in 2 of 3, quote in 1 of 3. The 1-question rounds are history "
                    "(\"earlier recipe\"); E33 and E31c are retrained on this recipe with their own step sizes.",
     "learned": ""},
    {"date": "2026-10-16", "version": "bridge-question data (runs of E33 and dense)", "kind": "recipe",
     "change": "Compose and deduce training documents end with up to 4 more questions: the single steps of their chain "
               "asked on their own (\"Who is the teacher of Lumo?\", \"What is Pim?\", \"What is a wump like?\"), "
               "after the task's own questions. Every other document, and every exam, is unchanged.",
     "why": "Every model, dense included, guesses on compose and deduce, even at one step; training only ever asks "
            "for the final answer. On DNA the loop joined two facts only after a one-step stage.",
     "effect": "planned",
     "effect_text": "A separate data build (text_checks_v2q8b4, 16 Oct; same tokenizer, same exams). Used only by the "
                    "runs marked \"bridge data\"; the 8-questions recipe stays the reference until they report.",
     "learned": ""},
]

# Claims register: every conclusion we have drawn, with its state. holds = repeated or overwhelming within its runs;
# single run = one run, needs a second; open = not testable yet; retracted = disproved (kept, with the date).
CLAIMS = [
    {"claim": "With 1 question per training document, E31c without the loop does not use its notebook.",
     "state": "holds", "since": "2026-10-11",
     "evidence": "Three runs (mixed, quote-only, lookup-only): 0 % on far facts at every length, and the same scores "
                 "with the notebook switched off. Narrowed 14 Oct: on the 8-questions recipe it does (next claim).",
     "settle": "—"},
    {"claim": "On the reliable recipe, E31c without the loop uses its notebook.", "state": "holds",
     "since": "2026-10-14",
     "evidence": "Run 1: far facts at 4k / 16k lookup 75 / 93 %, keyed 72 / 91 %, quote 77 / 90 %; notebook off 0 %. "
                 "Learned between 0.3B and 0.9B tokens. Run 2: lookup 97 / 99 %, keyed 92 / 97 %, quote 88 / 87 %; "
                 "learned by 0.6B. Run 3: lookup 77 / 92 %, keyed 70 / 91 %, quote 68 / 90 %. 3 of 3 runs. 2.4 training hours vs E33's 7 h; story loss 1.564 vs E33 1.520.",
     "settle": "—"},
    {"claim": "On the reliable recipe, E31c beats dense on far facts at the 4k training length.", "state": "holds",
     "since": "2026-10-15",
     "evidence": "Far lookup 75 %, 97 %, 77 % vs dense 56-63 % (3 runs); keyed 72 %, 92 %, 70 % vs 46-54 %; quote 77 %, "
                 "88 %, 68 % vs 3-94 %. Crowded facts: runs 1 and 3 43 % and 45 % (below dense 54-58 %), run 2 75 % "
                 "(above). Latest: E31c 22-24 %, dense 18-82 % — dense is better there.",
     "settle": "—"},
    {"claim": "Dense at 30M / 1.5B tokens with 1 question per training document learns retrieval only in some runs.",
     "state": "holds", "since": "2026-10-11",
     "evidence": "Four mixed-training runs: one passed quote only, one learned four tasks, two learned nothing; the "
                 "quote-only run learned nothing.",
     "settle": "—"},
    {"claim": "With 8 questions per training document, dense learns lookup and keyed reliably.", "state": "holds",
     "since": "2026-10-13",
     "evidence": "Three of three runs (seeds 0, 1, 2): lookup 56 %, 63 %, 56 %; keyed 46 %, 54 %, 46 % on far facts at "
                 "4k; seed 1 learned nothing with 1 question. Latest (77 %, 18 %, 82 %) and count (30 %, 17 %, 38 %) "
                 "switched on in 2 of 3, quote in 1 of 3 (94 %).",
     "settle": "—"},
    {"claim": "When E33's notebook switches on, it reads far facts past the training length.", "state": "holds",
     "since": "2026-10-14",
     "evidence": "Both runs where it switched on: new recipe run 1, far facts at 16k lookup 92 %, keyed 90 %, quote 89 %; "
                 "earlier recipe, lookup 83 %, keyed 78 % at 16k; notebook off 0 % in both.",
     "settle": "—"},
    {"claim": "E33's notebook switches on only in some runs at 30M / 1.5B tokens.", "state": "holds",
     "since": "2026-10-14",
     "evidence": "New recipe: 2 of 3 runs (run 2 far facts 13-15 %, rising from 0 % at 0.9B tokens to 13 % at 1.5B; "
                 "run 3 lookup and keyed yes, quote no). Earlier recipe: 1 of 2 (mixed training yes, lookup-only no). "
                 "Dense's lookup on the new recipe: 3 of 3; E31c's notebook: 2 of 2.",
     "settle": "Whether more tokens make it reliable."},
    {"claim": "On the reliable recipe, when E33's notebook switches on, it beats dense on far lookup and keyed at the "
              "4k training length.", "state": "holds",
     "since": "2026-10-15",
     "evidence": "Lookup: run 1 70 %, run 3 76 % vs dense 56-63 % (3 runs); keyed 71 %, 68 % vs 46-54 %. Run 2 did not "
                 "(13-15 %: its notebook had not switched on). Quote: run 1 71 %, run 3 0 % (dense 3-94 %). Not on "
                 "crowded facts (lookup 44 %, 46 % vs 54-58 %) or latest (20-22 % vs 18-82 %).",
     "settle": "—"},
    {"claim": "E33 partly keeps reading when the fact is phrased in a sentence shape never seen in training; dense "
              "does not.",
     "state": "single run", "since": "2026-10-13",
     "evidence": "Lookup on far facts at 4k: E33 64 % with training shapes, 64 % with the first unseen shape, 30 % "
                 "(45 % at 16k) with three more unseen shapes (exams v2.1); its near facts, read by short-range "
                 "attention, drop 62 → 31 %. Dense 37-48 % → 0-4 % on every unseen shape (three dense runs).",
     "settle": "New recipe: run 1 first unseen shape 74 % on lookup at 4k, three more shapes 20-27 %; run 3 76 % "
               "and 25 %; dense 0-2 %; run 2's notebook did not switch on. Two runs agree; E31c reads 76-80 % and "
               "33-36 % — a property of the notebook, not the loop."},
    {"claim": "Word-for-word quotes do not survive the notebook.", "state": "retracted", "since": "2026-10-14",
     "evidence": "Held on the earlier recipe (E31c and E33, two runs each: quote 0 % on far facts). On the new recipe E33 "
                 "quotes far facts 71 % at 4k and 89 % at 16k (14 Oct 08:13 UTC), 0 % with the notebook off.",
     "settle": "—"},
    {"claim": "Dense tracks the latest move better than the memory models.", "state": "single run", "since": "2026-10-13",
     "evidence": "Latest at 4k: dense 32 % in one of two runs; E31c 17 % and E33 13 %, both below the 20 % floor.",
     "settle": "Dense learning reliably (8-questions recipe), then two runs per model."},
    {"claim": "A learned page distance on E33's notes, starting at 0 (E33d), is learned within 1.5B tokens and "
              "lets E33 track the latest move beyond its 256-token window.",
     "state": "retracted", "since": "2026-10-15",
     "evidence": "E33 has no position on its notes: latest 95-96 % at 1k (moves inside the window), 20-22 % at 4k in "
                 "both new-recipe runs (floor 20 %). E33d adds one learned slope per read head, starting at 0. Marks "
                 "set before any run: pass = far latest ≥ 60 % at 4k and ≥ 40 % at 8k-16k on a switched-on run, no "
                 "harm to lookup / keyed / quote at 16k (within 5 points of E33 run 1) or story loss (≤ 1.53); stop = "
                 "latest ≤ 30 % at 4k on every switched-on run. Run 1 (switched on): latest 21 % at 4k, 2 % at 8k-16k "
                 "— at the stop mark; slopes stayed near 0 (largest 0.14); its wrong latest answers at 4k spread "
                 "evenly over the person's earlier places (no sense of which move came last). Lookup / keyed at 16k kept (93 / 93 %), quote 82 % (7 "
                 "points under E33 run 1), story loss 1.521."
                 " Run 2 (switched on): latest 16 % at 4k, 1-2 % at 8k-16k; slopes near 0 (largest 0.08); lookup / "
                 "keyed at 16k 93 / 90 %, quote 77 %, story loss 1.521. Stop mark met on both switched-on runs. "
                 "What this disproved: a slope starting at 0 gets learned (it stayed ≤ 0.14; about 3-5 is needed). "
                 "Not tested: whether a large page distance fixes latest.",
     "settle": "—"},
    {"claim": "A large page distance on E33's notes (fixed, not learned) fixes latest.", "state": "retracted",
     "since": "2026-10-15",
     "evidence": "E33d run 1's weights, slopes overwritten, exams only (400 items per cell; latest exact, others far "
                 "facts, at 4k / 8k / 16k). As trained: latest 21 / 2 / 2 %, lookup 75 / 87 / 93 %. All heads +1: latest "
                 "28 / 19 / 33 % (last move picked 34 / 36 / 47 %), lookup 56 / 51 / 52 %, quote 45 / 29 / 10 %. All +3: "
                 "latest 31 / 14 / 4 %, far facts 0 %. All +10: latest 19 / 0 / 0 %, far facts 0 %. Heads 5-8 at +10: "
                 "latest 18 / 3 / 2 %, far facts unchanged. No fixed value fixes latest without destroying far facts on "
                 "this model; a mild distance does carry order (+1 lifts latest at 16k 2 → 33 % with no retraining).",
     "settle": "A design question for the author: train the distance from a non-zero start."},
    {"claim": "Memory models beat rereading on long documents.", "state": "open", "since": "2026-10-13",
     "evidence": "Dense scores 0 % past 8k on every task, but it has no method for reading past its training length, "
                 "and the no-memory control has not been run on v2.",
     "settle": "The no-memory control and a dense model with a long-text method on the same exams."},
    {"claim": "E33 reads better at 16k than at 1k mostly because the facts are spread out, not because of length.",
     "state": "single run", "since": "2026-10-13",
     "evidence": "Crowded-facts exam (every fact within 1,024 tokens of the asked one): E33 lookup 33 % at 16k and 39 % "
                 "at 4k, against 83 % and 64 % with spread facts and 28 % at 1k; keyed 20 % at 16k (78 % spread). Its "
                 "notebook holds one fact per page far better than several.",
     "settle": "A second E33 run; a notebook with more room per page would be a design question, not a protocol one."},
    {"claim": "At 4k, dense matches E33 once the facts are crowded together.", "state": "single run",
     "since": "2026-10-13",
     "evidence": "Earlier recipe: crowded lookup at 4k dense 42 % vs E33 39 %, keyed 39 % vs 18 %. New recipe (E33 run 1, "
                 "dense 3 runs): lookup 44 % vs 54-58 %, keyed 28 % vs 46-51 %. With spread facts E33 leads. At 16k E33 "
                 "still reads crowded facts (43 %), dense nothing. E33 run 3: lookup 46 %, keyed 34 % at 4k — the "
                 "same picture. E31c: 43 %, 75 %, 45 % — run 2 is the exception.",
     "settle": "—"},
    {"claim": "Nobody learns compose or deduce at 30M / 1.5B — not even one step.", "state": "holds",
     "since": "2026-10-13",
     "evidence": "Every run at guessing on two-step compose and deduce (10 Oct on), and on the one-step versions "
                 "(exams v2.1, 13 Oct): dense 9-10 %, E33 6-8 %, E31c 4-10 % (guessing 8 %).",
     "settle": "The reliable recipe first; these relations may need more data or a larger model."},
    {"claim": "What latest lacks on the notebook is order, and ranking notes among their matches adds it without "
              "costing far facts.", "state": "single run", "since": "2026-10-16",
     "evidence": "No training, one fixed strength (2), exam set: E33 run 1 latest at 16k 5 → 41 %, far lookup 92 → "
                 "91 %; E33d run 1 2 → 30 %, lookup 93 → 92 %; keyed −2 to −5 (look-alike names count as matches). A "
                 "fixed page distance bought 33 % at the cost of far lookup 93 → 52 %. Gate chosen on fresh questions.",
     "settle": "E33e training run (Odra, 16 Oct): latest ≥ 60 % at 4k with far lookup within 5 points of E33."},
    {"claim": "E33's loop does no task work on text: 1 round scores like 4.", "state": "single run",
     "since": "2026-10-16",
     "evidence": "E33 run 1 with the loop stopped after 1 / 2 / 3 rounds: every task within 2 points of the trained 4 "
                 "rounds (latest at 4k 22 / 23 / 22 vs 22 %, far lookup at 16k 91 / 91 / 91 vs 92 %, compose 8 %, deduce "
                 "8 %). Its gain is story loss only (1.520 vs E31c 1.564).",
     "settle": "Rounds exams on a second E33 run; a task for the rounds (round targets) after the bridge data."},
    {"claim": "No model counts: they know zero and roughly how many, not the number.", "state": "holds",
     "since": "2026-10-16",
     "evidence": "Accuracy by true count at 4k, 8 runs: 0 visits right 38-81 %, other counts 0-65 % with no fall-off "
                 "with size; notebook models within one of the truth 55-65 % of the time.",
     "settle": "—"},
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
    {"id": "cal3a_plain30m_v2_odra.odra", "start": "2026-10-10", "data": "v2", "standing": "history",
     "run": "Plain input", "model": "dense · 30M, no word-pair/triple tables", "tokens": "1.5B (7,888 steps)",
     "exams": "v2 (guessing 4-9 %)",
     "result": "Quote passes (84 % far); lookup 37 %, keyed 33 % (far facts), latest 32 % (floor 20 %), count 19 %.",
     "when": "10 Oct 18:08 → 20:55", "why": "Earlier recipe (1 question per document): the dense reference of that comparison, seed 0."},
    {"id": "r2v2p_screen30m_odra.odra", "start": "2026-10-10", "data": "v2", "standing": "history",
     "run": "Round: E33 loop and E31c, plain", "model": "E33 loop, E31c notebook · 30M, plain input",
     "tokens": "1.5B (7,888 steps)", "exams": "v2, 1k-16k",
     "result": "E33: lookup 64 % and keyed 56 % on far facts at 4k, 83 % and 78 % at 16k; notebook off: 0 %. "
               "E31c: 0 % on far facts.",
     "when": "10 Oct 21:00 → 11 Oct 12:15", "why": "Earlier recipe (1 question per document): the memory models of that comparison."},
    {"id": "cal3a_s1_plain30m_v2_odra.odra", "start": "2026-10-11", "data": "v2", "standing": "history",
     "run": "Plain input, second seed", "model": "dense · 30M, plain, seed 1", "tokens": "1.5B",
     "exams": "v2 (guessing 4-9 %)", "result": "Nothing learned (seed 0 learned four tasks).",
     "when": "11 Oct 05:36 → 08:43", "why": "Earlier recipe (1 question per document): the dense reference of that comparison, seed 1."},
    {"id": "pt_quote_30m_odra.odra", "start": "2026-10-11", "data": "v2", "standing": "history",
     "run": "Single task: quote", "model": "dense, E31c, E33 loop · 30M, plain", "tokens": "1.5B each, quote only",
     "exams": "quote, 1k-16k", "result": "Nobody learned quote on far facts (dense 6 %, E31c and E33 0 %).",
     "when": "11 Oct 12:15 → 12 Oct 01:04", "why": "Earlier recipe, single-task series (paused 13 Oct); one run per model."},
    {"id": "pt_lookup_30m_odra.odra", "start": "2026-10-12", "data": "v2", "standing": "history",
     "run": "Single task: lookup", "model": "dense, E31c, E33 loop · 30M, plain", "tokens": "1.5B each, lookup only",
     "exams": "lookup, 1k-16k",
     "result": "Dense 48 % on far facts at 4k (0 % past 4k); E31c and E33 0 % on far facts — both read only facts "
               "inside their window (E33 near facts 62-83 %). E33's mixed-training win did not repeat.",
     "when": "12 Oct 01:04 → 13 Oct 11:12", "why": "Earlier recipe, single-task series (paused 13 Oct); E33 cut off by the outage, resumed."},
    {"id": None, "start": "2026-10-13", "data": "v2.1", "standing": "history",
     "run": "Added exams on finished models", "model": "E33, E31c, dense (two runs), and the single-task models",
     "tokens": "no training", "exams": "v2.1: crowded facts, more unseen shapes, one step",
     "result": "Crowded facts: E33 lookup 33 % at 16k (83 % spread), dense matches it at 4k. Three more unseen shapes: "
               "E33 30-45 %, dense 1-4 %. One-step compose and deduce: everyone at guessing.",
     "when": "13 Oct 11:12 → 12:05", "why": "Existing checkpoints scored on the added exams; nothing retrained. Scores "
                                            "appear under each run's other exam sets."},
    {"id": "q8_dense30m_s0_odra.odra", "start": "2026-10-13", "data": "v2", "standing": "in use",
     "run": "Reliability fix, run 1", "model": "dense · 30M, plain, seed 0", "tokens": "1.5B, 8 questions per document",
     "exams": "v2 + v2.1, 1k-4k",
     "result": "Lookup 56 %, keyed 46 % (far facts), latest 77 % (floor 20 %), count 30 % at 4k; quote at guessing; "
               "learned steadily from about 0.5B tokens.",
     "when": "13 Oct 12:05 → 14:58", "why": "The dense reference of the current recipe."},
    {"id": "q8_dense30m_s1_odra.odra", "start": "2026-10-13", "data": "v2", "standing": "in use",
     "run": "Reliability fix, run 2", "model": "dense · 30M, plain, seed 1", "tokens": "1.5B, 8 questions per document",
     "exams": "v2 + v2.1, 1k-4k",
     "result": "Lookup 63 %, keyed 54 % (far facts at 4k); latest 18 % (floor 20 %) and count 17 % not learned at 4k; "
               "quote at guessing. The same seed learned nothing with 1 question.",
     "when": "13 Oct 14:58 → 17:49", "why": "The dense reference of the current recipe."},
    {"id": "q8_dense30m_s2_odra.odra", "start": "2026-10-13", "data": "v2", "standing": "in use",
     "run": "Reliability fix, run 3", "model": "dense · 30M, plain, seed 2",
     "tokens": "1.5B, 8 questions per document", "exams": "v2 + v2.1, 1k-4k",
     "result": "Quote 94 % and latest 82 % (both pass), lookup 56 %, keyed 46 % (far facts), count 38 % at 4k.",
     "when": "13 Oct 17:49 → 20:40", "why": "The dense reference of the current recipe."},
    {"id": None, "start": "2026-10-13", "data": "v2", "standing": "calibration",
     "run": "Step size for E33 and E31c", "model": "E33 loop, E31c notebook · 30M, plain",
     "tokens": "4 × 60M each, 8 questions per document", "exams": "none (held-out loss only)",
     "result": "Both best at 1e-3 (E33 held-out loss 2.56 vs 2.77 / 3.07 / 4.41 at 5e-4 / 2e-3 / 4e-3; E31c 2.78 vs "
               "2.91 / 3.55 / 4.81) — the value they had borrowed from dense.",
     "when": "13 Oct 21:08 → 23:05", "why": "Spec §7: every architecture gets its own step size, same tuning budget."},
    {"id": "q8r_e33_s0_odra.odra", "start": "2026-10-13", "data": "v2", "standing": "in use",
     "run": "Round, new recipe: E33 run 1", "model": "E33 loop · 30M, plain, seed 0",
     "tokens": "1.5B, 8 questions per document", "exams": "v2 + v2.1, 1k-16k",
     "result": "Far facts at 4k / 16k: lookup 70 / 92 %, keyed 71 / 90 %, quote 71 / 89 %; notebook off 0 %. Crowded "
               "lookup 44 % at 4k; latest 22 % (floor 20 %); count 26 %.",
     "when": "13 Oct 23:12 → 14 Oct 08:13", "why": "The first memory model on the reliable recipe."},
    {"id": "q8r_e33_s1_odra.odra", "start": "2026-10-14", "data": "v2", "standing": "in use",
     "run": "Round, new recipe: E33 run 2", "model": "E33 loop · 30M, plain, seed 1",
     "tokens": "1.5B, 8 questions per document", "exams": "v2 + v2.1, 1k-16k",
     "result": "Notebook did not switch on in time: far facts 13-15 % at 4k and 16k, rising from 0 % at 0.9B tokens; "
               "near facts 66-100 %; latest 20 % (floor 20 %); count 24 %.",
     "when": "14 Oct 08:33 → 17:34", "why": "The repeat of E33 run 1."},
    {"id": "q8r_e31c_s0_odra.odra", "start": "2026-10-14", "data": "v2", "standing": "in use",
     "run": "Round, new recipe: E31c run 1", "model": "E31c notebook · 30M, plain, seed 0",
     "tokens": "1.5B, 8 questions per document", "exams": "v2 + v2.1, 1k-16k",
     "result": "Notebook switched on: far facts at 4k / 16k lookup 75 / 93 %, keyed 72 / 91 %, quote 77 / 90 %; "
               "notebook off 0 %. Crowded lookup 43 % at 4k; latest 23 % (floor 20 %); count 24 %; story loss 1.564.",
     "when": "14 Oct 17:54 → 21:44", "why": "The architecture comparison on the reliable recipe."},
    {"id": "q8r_e31c_s1_odra.odra", "start": "2026-10-14", "data": "v2", "standing": "in use",
     "run": "Round, new recipe: E31c run 2", "model": "E31c notebook · 30M, plain, seed 1",
     "tokens": "1.5B, 8 questions per document", "exams": "v2 + v2.1, 1k-16k",
     "result": "Notebook switched on again: far facts at 4k / 16k lookup 97 / 99 %, keyed 92 / 97 %, quote 88 / 87 %; "
               "crowded lookup 75 % at 4k; latest 24 % (floor 20 %); count 28 %; story loss 1.564.",
     "when": "14 Oct 22:04 → 15 Oct 01:49", "why": "The repeat of E31c run 1."},
    {"id": "q8r_e33_s2_odra.odra", "start": "2026-10-15", "data": "v2", "standing": "in use",
     "run": "Round, new recipe: E33 run 3 (Odra)", "model": "E33 loop · 30M, plain, seed 2",
     "tokens": "1.5B, 8 questions per document", "exams": "v2 + v2.1, 1k-16k",
     "result": "Notebook switched on for lookup / keyed (far at 4k / 16k: 76 / 93 %, 68 / 89 %) but not for quote (0 % "
               "far; near 77-100 % to 4k). Crowded lookup 46 % at 4k; latest 20 % (floor 20 %); story loss 1.519.",
     "when": "15 Oct 02:09 → 11:14", "why": "Run 1 switched on, run 2 did not."},
    {"id": "q8r_e31c_s2_polonez.polonez", "start": "2026-10-15", "data": "v2", "standing": "in use",
     "run": "Round, new recipe: E31c run 3 (Polonez)", "model": "E31c notebook · 30M, plain, seed 2",
     "tokens": "1.5B, 8 questions per document", "exams": "v2 + v2.1, 1k-16k",
     "result": "Notebook switched on: far facts at 4k / 16k lookup 77 / 92 %, keyed 70 / 91 %, quote 68 / 90 %; crowded "
               "lookup 45 % at 4k; latest 22 % (floor 20 %); story loss 1.564.",
     "when": "15 Oct 11:11 → 14:18", "why": "Settles E31c's switch-on rate before it becomes the main line."},
    {"id": None, "start": "2026-10-15", "data": "v2", "standing": "history",
     "run": "E33d run 1 with a fixed page distance (Odra; exams only, run by the E33 reasoning-fixes session)",
     "model": "E33d run 1's weights, slopes overwritten: all heads +1, +3, +10; 4 of 9 heads +10",
     "tokens": "none (no training)", "exams": "v2 id: latest / lookup / keyed / quote at 4k-16k",
     "result": "+1: latest at 16k 2 → 33 %, far lookup 93 → 52 %; +3 / +10: far facts 0 %; 4 heads at +10: far facts "
               "kept, latest at guessing. No fixed value works; the order signal is there.",
     "when": "15 Oct 16:06 → 16:20", "why": "Approved by the author 15 Oct; settles the open fixed-page-distance claim."},
    {"id": "q8r_e33d_s0_polonez.polonez", "start": "2026-10-14", "data": "v2", "standing": "in use",
     "run": "E33d run 1 (Polonez)", "model": "E33d · 30M, plain, seed 0",
     "tokens": "1.5B, 8 questions per document", "exams": "v2 + v2.1, 1k-16k; latest picks and slopes logged",
     "result": "Notebook switched on (far lookup 50 % already at 0.3B): far facts at 4k / 16k lookup 75 / 93 %, keyed "
               "68 / 93 %, quote 69 / 82 %; latest 21 % at 4k, 2 % at 8k-16k; slopes near 0; story loss 1.521.",
     "when": "14 Oct 19:37 → 15 Oct 02:48", "why": "Does a page distance let E33 track the latest move?"},
    {"id": "q8r_e33d_s1_polonez.polonez", "start": "2026-10-15", "data": "v2", "standing": "in use",
     "run": "E33d run 2 (Polonez); stopped after it, no run 3", "model": "E33d · 30M, plain, seed 1",
     "tokens": "1.5B each, 8 questions per document", "exams": "v2 + v2.1, 1k-16k; latest picks and slopes logged",
     "result": "Notebook switched on: far facts at 4k / 16k lookup 75 / 93 %, keyed 67 / 90 %, quote 66 / 77 %; "
               "latest 16 % at 4k, 1-2 % at 8k-16k; slopes near 0; story loss 1.521. E33d stopped.",
     "when": "15 Oct 03:08 → 10:49 (Polonez, 4 GPUs, same 96-row batch; one 20-minute heat pause)",
     "why": "Requested by the author through the E33 reasoning-fixes session (14 Oct); spec E33d_note_page_distance. "
            "Step size borrowed from E33's tuning (1e-3): the same model plus 9 numbers that start at 0."},
    {"id": None, "start": "2026-10-16", "data": "v2", "standing": "history",
     "run": "Checks without training (Odra; exams only, E33 session)",
     "model": "E33d run 1 + the rank-among-matches term (fixed strength 1-5, gate sweep); E33 run 1 with 1-4 loop rounds",
     "tokens": "none (no training)",
     "exams": "fresh development questions (latest / lookup / keyed / quote, 4k and 16k); v2 + v2.1 exams for the rounds",
     "result": "Rank term (strength 2, exam set): latest at 16k E33 5 → 41 %, E33d 2 → 30 %, far lookup within 1 point. "
               "Rounds: 1, 2, 3 rounds = 4 rounds on every task.",
     "when": "16 Oct 05:53 → 06:45 UTC", "why": "Author: fixes for T4-T7 (16 Oct); the cheapest tests first."},
    {"id": "q8r_e33e_s0_odra.odra", "start": "2026-10-16", "data": "v2", "standing": "running",
     "run": "E33e run 1 (Odra)", "model": "E33e · 30M, plain, seed 0 (E33 + rank among matching notes)",
     "tokens": "1.5B, 8 questions per document", "exams": "v2 + v2.1, 1k-16k; latest picks and strengths logged",
     "result": "—", "when": "16 Oct from about 06:40 UTC",
     "why": "Author: fixes for T4-T7 (16 Oct); spec E33e_note_rank. Gate set on fresh questions, never the exams."},
    {"id": "q8b_dense_s0_polonez.polonez", "start": "2026-10-16", "data": "v2 + bridge", "standing": "running",
     "run": "Dense on the bridge data (Polonez)", "model": "dense · 30M, plain, seed 0",
     "tokens": "1.5B, 8 questions + up to 4 bridge questions per compose / deduce document", "exams": "v2 + v2.1, 1k-16k",
     "result": "—", "when": "16 Oct from 06:08 UTC",
     "why": "Is one-step compose / deduce learnable at all once its steps are taught? The cheapest learner first."},
    {"id": "q8b_e33_s0_polonez.polonez", "start": "2026-10-16", "data": "v2 + bridge", "standing": "queued",
     "run": "E33 on the bridge data (Polonez)", "model": "E33 · 30M, plain, seed 0",
     "tokens": "1.5B, as the dense run", "exams": "v2 + v2.1, 1k-16k",
     "result": "—", "when": "after the dense run and a 20-minute pause",
     "why": "Does the loop chain the taught steps (two-step compose above guessing)?"},
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
    "e33d": ("E33d, notes with a page distance", "E33 plus one slope per read head that lowers older notes' scores "
                                                  "by log(1 + pages back); starts at 0, so it begins as E33"),
    "e33d_plain": ("E33d, notes with a page distance", "E33 plus a learned page distance on the notes (9 numbers, start "
                                                        "at 0); plain input"),
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
            "Added exams (13 Oct): with every fact packed into 1,024 tokens E33 reads lookup 39 % at 4k and 33 % at 16k "
            "(spread: 64 % and 83 %), keyed 18 % and 20 % (56 %, 78 %) — the long-document gain was mostly the facts "
            "being spread out. Three more unseen sentence shapes: lookup 30 % / 45 %. One-step compose and deduce: at "
            "guessing.",
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
    "q8_dense30m_s0_odra.odra": {
        "title": "Reliability fix, run 1: dense 30M with 8 questions per training document (1.5B tokens, data v2)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "13 Oct 2026 12:05 UTC → 14:58 UTC (training 2.3 h, exams and learning curve 0.6 h)",
        "verdict": "Learned four tasks, more than any 1-question run: on far facts at 4k lookup 56 % and keyed 46 % "
                   "(guessing 8 % and 6 %), latest 77 % (floor 20 %), count 30 % (9 %). Quote stayed at guessing.",
        "observations": [
            "The skills rose steadily from about 0.5B tokens (lookup 15 → 33 → 52 % at 0.45B, 0.75B, 1.05B tokens) "
            "instead of switching on late; latest jumped between 0.75B and 0.9B (11 → 46 %).",
            "The best 1-question run (seed 0) had lookup 37 %, keyed 33 %, latest 32 %, count 19 %; two of four had nothing.",
            "Crowded facts at 4k read as well as spread ones (lookup 57 %, keyed 46 %).",
            "Unseen sentence shapes still fail (0-2 %): more questions taught the training shapes, not the meaning.",
            "Quote at guessing (10 %) where two 1-question runs passed it: copying a 4-6 word sign is a different "
            "skill, and 8 questions per document did not bring it.",
            "Compose, deduce and their one-step versions at guessing.",
            "Language: story loss 1.548 nats per token (1-question runs 1.53-1.56).",
        ],
    },
    "q8_dense30m_s1_odra.odra": {
        "title": "Reliability fix, run 2: dense 30M, seed 1, 8 questions per training document (1.5B tokens, data v2)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "13 Oct 2026 14:58 UTC → 17:49 UTC (training 2.3 h)",
        "verdict": "Lookup and keyed learned again: 63 % and 54 % on far facts at 4k (guessing 8 % and 6 %). Latest "
                   "(18 %, floor 20 %) and count (17 %) did not switch on at 4k; quote at guessing. With 1 question per "
                   "document the same seed learned nothing.",
        "observations": [
            "Lookup and keyed rose between 0.75B and 1.2B tokens (lookup 16 → 30 → 43 → 59 %), later than run 1 but "
            "steadily.",
            "Latest learned only in short documents (21 % at 1k) and count only up to 2k (28 % at 1k): which of the "
            "harder tasks switch on still depends on the seed.",
            "Crowded facts read as well as spread ones (lookup 58 %, keyed 51 %); unseen sentence shapes still fail (1 %).",
            "Language: story loss 1.544 nats per token (run 1: 1.548).",
        ],
    },
    "q8_dense30m_s2_odra.odra": {
        "title": "Reliability fix, run 3: dense 30M, seed 2, 8 questions per training document (1.5B tokens, data v2)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "13 Oct 2026 17:49 UTC → 20:40 UTC (training 2.3 h)",
        "verdict": "Five tasks learned, two passing: quote 94 % and latest 82 % at 4k, lookup 56 % and keyed 46 % (far "
                   "facts), count 38 %. With runs 1 and 2 this settles the recipe: lookup and keyed in 3 of 3 runs.",
        "observations": [
            "Quote switched on first (52 % at 0.45B tokens, 91 % from 0.75B), latest at 1.05B, lookup and keyed late "
            "(7 → 52 % between 1.05B and 1.2B tokens) — later than runs 1 and 2, but within the budget.",
            "Crowded facts read as well as spread ones (quote 93 %, lookup 54 %, keyed 50 %); unseen sentence shapes "
            "still fail (0-1 %).",
            "Compose, deduce and their one-step versions at guessing, as in every run.",
            "Language: story loss 1.537 nats per token (runs 1 and 2: 1.548, 1.544).",
        ],
    },
    "q8r_e33_s0_odra.odra": {
        "title": "Round on the new recipe: E33 loop, run 1 (30M plain, 8 questions per training document, 1.5B tokens)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "13 Oct 2026 23:12 UTC → 14 Oct 08:13 UTC (training 7.0 h, exams 1.6 h)",
        "verdict": "E33 reads far facts better than dense and better the longer the document: at 4k / 8k / 16k lookup "
                   "70 / 83 / 92 %, keyed 71 / 86 / 90 %, quote 71 / 84 / 89 % (passes at 8k-16k); dense, three runs, "
                   "56-63 % and 46-54 % at 4k and nothing past 4k. With its notebook off: 0 %.",
        "observations": [
            "It learned early: lookup 64 %, keyed 55 % and quote 36 % already at 0.6B tokens; dense reached its "
            "lookup between 0.75B and 1.2B.",
            "Quote far back in the story now survives the notebook (89 % at 16k), retracting the earlier-recipe "
            "finding that it did not.",
            "Short documents are its weak spot again: far facts 27-34 % at 1k, 46-49 % at 2k; with every fact packed "
            "into 1,024 tokens lookup 44 % at 4k and 43 % at 16k (dense 54-58 % at 4k).",
            "Unseen sentence shapes: the first one 74 % on lookup at 4k, three more 20-27 % (dense 0-2 %).",
            "Latest: 95 % at 1k (moves inside the window), 22 % at 4k (floor 20 %), about 5 % past 4k; dense learned it "
            "in two runs of three (77-82 %).",
            "Count 26 % at 4k and still 22 % at 16k; harder keyed (64 people) 8 %; compose and deduce at guessing.",
            "Language: story loss 1.520 nats per token (dense on this recipe 1.537-1.548).",
        ],
    },
    "q8r_e33_s1_odra.odra": {
        "title": "Round on the new recipe: E33 loop, run 2 (30M plain, 8 questions per training document, 1.5B tokens)",
        "where": "Odra · 3 × RTX 3090",
        "dates": "14 Oct 2026 08:33 UTC → 17:34 UTC (training 7.0 h, exams 1.6 h)",
        "verdict": "The notebook switched on too late: far facts 13 % (quote, lookup) and 15 % (keyed) at 4k, 13-16 % at "
                   "16k — not learned — where run 1 read 70-92 %. Facts near the question it reads well (66-100 %).",
        "observations": [
            "Far facts stayed at 0 % until 0.9B tokens, then 10 % at 1.2B and 13-15 % at 1.5B: the notebook was switching "
            "on as the step size decayed to zero. Run 1 had switched on by 0.6B.",
            "Notebook off: 0 % on far facts, so the 13-15 % is the notebook's.",
            "Everything else matches run 1: latest 96 % at 1k and 20 % at 4k (floor 20 %), count 24 % at 4k, compose and "
            "deduce at guessing, story loss 1.522 nats per token (run 1: 1.520).",
            "With run 1, E33's notebook switched on in 1 of 2 runs on this recipe; dense's lookup in 3 of 3.",
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

# ---- added 2026-10-15 for the dashboard layout (docs/engineering_specs/capability_boards.md) ----------------------

# Decisions waiting for the author: shown at the top of the board. Remove an item when it is decided.
DECISIONS = [
    "Make E31c (notebook, no loop) the main line? It switched on in 3 of 3 runs at a third of E33's training time.",
    "Start the notebook-proportions sweep (approved in principle 14 Oct) — on E31c instead of E33?",
    "Latest: train the page distance from a non-zero start (the E33 session's proposal)?",
]

# One entry per architecture family compared on the board: plain name, what it is in one line, the standing analogy
# (research-comms), and the current one-line verdict. Keys are the scoreboard families.
ARCHS = {
    "dense": {"name": "Dense", "what": "Full attention over the whole document; the reference.",
              "analogy": "rereads any page of the book at any time", "verdict":
              "Learns lookup and keyed reliably on the current recipe; cannot read past its 4k training length."},
    "e31c": {"name": "E31c notebook", "what": "Short-range attention plus a notebook of summaries of past text.",
             "analogy": "sees the last pages and keeps a notebook", "verdict":
             "Best so far: reads far facts in 3 of 3 runs, also at 16k; weak on crowded facts and on latest."},
    "e33": {"name": "E33 loop", "what": "E31c plus a read-think-reread loop over the notebook.",
            "analogy": "rereads its notebook several times before answering", "verdict":
            "As good as E31c when its notebook switches on, but it did in 2 of 3 runs, at 3× the training time."},
    "e33d": {"name": "E33d page distance", "what": "E33 plus a learned penalty on older notes (page distance).",
             "analogy": "E33 with page numbers in the notebook", "verdict":
             "Stopped: the page distance was never learned; latest stayed at guessing."},
}

# What changed the results: one entry per change of recipe, architecture, exam or reading rule whose effect was
# measured. effect: helped / hurt / no effect / mixed / measurement (a new exam or rule, the models did not change).
# metrics: (what is measured, before, after, unit) — unit "%" (0-100) or "runs" ("k/n").
CHANGES = [
    {"date": "2026-10-15", "kind": "architecture", "change": "Fixed page distance +1 on E33d run 1 (no retraining)",
     "compared": "E33d run 1 as trained vs the same weights with every slope set to +1", "effect": "mixed",
     "metrics": [("latest at 16k", 2, 33, "%"), ("far lookup at 16k", 93, 52, "%")],
     "note": "The order signal is there, but a fixed value trades far facts for it; +3 and +10 erase far facts."},
    {"date": "2026-10-15", "kind": "architecture", "change": "A learned page distance on E33's notes (E33d)",
     "compared": "E33 (3 runs) vs E33d (2 runs), current recipe", "effect": "no effect",
     "metrics": [("latest at 4k, middle run", 20, 19, "%"), ("far quote at 16k, best run", 89, 82, "%")],
     "note": "The 9 slopes start at 0 and never grew (largest 0.14), so nothing changed except a little quote loss."},
    {"date": "2026-10-15", "kind": "architecture", "change": "A read-think-reread loop on the notebook (E33 vs E31c)",
     "compared": "E31c (3 runs) vs E33 (3 runs), current recipe", "effect": "hurt",
     "metrics": [("runs that read far facts", "3/3", "2/3", "runs"), ("far lookup at 4k, middle run", 77, 70, "%"),
                 ("training hours on 3 GPUs", 2.4, 7.0, "h")],
     "note": "Slightly better language (story loss 1.519 vs 1.564), but less reliable and 3× slower to train."},
    {"date": "2026-10-13", "kind": "recipe", "change": "8 questions per training document instead of 1",
     "compared": "the same models, earlier recipe vs current recipe", "effect": "helped",
     "metrics": [("E31c runs that read far facts", "0/3", "3/3", "runs"),
                 ("dense runs that learned anything", "2/4", "3/3", "runs"),
                 ("E33 runs that read far facts", "1/2", "2/3", "runs")],
     "note": "Answers were 0.1 % of training tokens; more questions per document turned luck into a reliable skill."},
    {"date": "2026-10-13", "kind": "recipe", "change": "Training E33 on lookup only instead of mixed tasks",
     "compared": "E33, mixed training vs lookup-only training, earlier recipe", "effect": "hurt",
     "metrics": [("far lookup at 16k", 83, 0, "%")],
     "note": "Its notebook never switched on with one task; dense trained the same way still learned (48 % at 4k)."},
    {"date": "2026-10-13", "kind": "exam", "change": "Crowded facts (exams v2.1): every fact within 1,024 tokens",
     "compared": "E33, spread facts vs crowded facts at 16k", "effect": "measurement",
     "metrics": [("far lookup at 16k", 83, 33, "%")],
     "note": "Most of E33's long-document gain was the facts being spread out; the models did not change."},
    {"date": "2026-10-13", "kind": "rules", "change": "Read memory on far facts only (reading rules)",
     "compared": "E31c, all facts vs facts beyond the short window", "effect": "measurement",
     "metrics": [("E31c quote at 4k, earlier recipe", 24, 0, "%")],
     "note": "Facts at the end sit inside every model's short window; E31c's apparent scores were all near facts."},
    {"date": "2026-10-10", "kind": "recipe", "change": "Plain input (no word-pair / triple tables)",
     "compared": "dense with vs without the tables, earlier recipe", "effect": "no effect",
     "metrics": [("dense runs that learned anything", "1/1", "1/2", "runs")],
     "note": "The first run learned four tasks, the second nothing: seed luck, not the input. Kept for consistency."},
]
