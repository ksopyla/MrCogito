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
    "as_of": "2026-10-18 11:20 UTC",
    "headline": "Three of four goal tasks reach 70 % on the main exam at 4k: latest 88 % (two order arms agree), compose 94 % "
                "and deduce 99 % (three runs), all with privileged training labels and only on trained shapes. Count stays "
                "at guessing.",
    "where": [
        "17-18 Oct night (Odra): E33g (E33r + four fixed order heads) failed latest (22 % at 4k) but lifted compose to "
        "69 / 88 / 92 % at 4k / 16k / 32k and deduce to 24 % at 4k, against 61 / 82 % and 17 % for the matched control "
        "E33rb. Probes on its weights: order on every head with a wider match picks the newest move 50 % at 4k but costs "
        "lookup and compose, so E33i adds a learned per-question switch (training since 22:21 UTC). E33h (count heads) "
        "and E33q (the loop asks with the name it found) train beside it. Polonez (multi-hop session): E33ok's key-gradient "
        "stop made the notes keep the property word (probe 8-13 → 33 %), deduce two-step 31 % in its probe; denser-mix "
        "runs with and without order heads report around 02:00 UTC.",
        "Goal set by the author (17 Oct): 70 % exact on T4-T7 with a notebook model. Roadmap agreed with the other "
        "sessions (research note text_t4_t7_roadmap_20261017): order heads for latest and a summing read for count "
        "(this session, Odra); hop supervision plus a content objective for compose and deduce (multi-hop session, "
        "Polonez); decode-and-append and the writer signal (agent-memory sessions).",
        "Order term on E33r without training (Odra, 17 Oct 18:20-18:35 UTC, 200 items per cell): latest at 4k 20 → 28 % "
        "with strength 2 on every head, 24 % with four order heads (1, 2, 4, 8); at 16k 2 → 7 / 4 %. Far lookup and "
        "compose lose up to 11 points with every head ranked, about nothing with four order heads. Much weaker than the "
        "same check on E33 run 1 (5 → 41 % at 16k): E33r's scores were not trained to be counted. Training decides.",
        "Two-step questions, first success (E33r, Polonez, 17 Oct 08:27-12:12 UTC, E33 session): E33 on the bridge "
        "data, continued 0.3B tokens with each loop round's read pushed in training onto the page holding the next "
        "fact of the chain (no hint at exam time). Two-step compose (\"Where does the teacher of X live?\") 30 / 60 / "
        "76 % at 1k / 4k / 16k, guessing 8 %; the same continuation without the read target (E33x) 9 / 7 / 11 %. Far "
        "lookup at 16k 89 % (start 90 %). Deduce still at guessing (13 % at 4k): the read finds the page of \"Every "
        "kavi is shiny\" 95 % of the time, but the note does not say which property. Quotes lost (0 %). One run.",
        "Two follow-ups broke the notebook (Polonez, 17 Oct 10:52-16:17 UTC): pushing each round onto single notes "
        "instead of pages (E33o) erased far facts (lookup 0 % at every length); adding a read-back of each note's "
        "words (E33b, run 2; run 1 stopped after its random head scrambled the reader) brought lookup back only to "
        "15-21 %. Deduce at guessing in both.",
        "E33e (E33 plus ranking among matching notes, learned strengths) finished on Odra (17 Oct 13:38 UTC, resumed "
        "after the 16 Oct outage; exams 16:48 UTC): latest 19 % at 4k, 0 % at 16k, under the pass mark of 60 %. The "
        "strengths start at 0 and stayed small (largest 0.15; the no-training check used 2). Far facts fine: lookup "
        "4k / 16k 81 / 93 %, quote 80 / 94 %. One run.",
        "Agent to agent (E35 and E33f, Polonez, 16 Oct 17:00-21:35 UTC, E33 session). (1) Three trained 16-vector message "
        "channels carried no facts (0.5B tokens each): broadcast, steered by B's question, and B frozen with the loss on "
        "answers only. B's answer surprise on far lookup at 4k was 2.31 / 2.32 / 2.32 with the real message against 2.35 / "
        "2.32 / 4.74 empty and 2.32 / 2.32 / 2.33 with another document's: the frozen-B run teaches the answer format, not "
        "the facts. (2) E33 split at its notebook, no training: B reads another document plus A's notes and answers far "
        "lookup at 16k 93 % (own text 93 %). With one request per read of the loop (B sends its round queries, A returns "
        "its 4 best notes), about 8 of 2,720 notes give 90 %. A single request after B's first read fails (0-30 %). "
        "(3) E33f, E33 fine-tuned 0.3B tokens with every question cut off from the document's raw text (full exam, 400 "
        "items): far lookup 4k / 16k 77 / 91 % (E33 on its own text 72 / 90 %), far quote 16 / 24 % (0 / 0 %), near facts "
        "lose (latest at 1k 15 vs 95 %). With per-round requests about 8 notes answer far lookup at 16k-128k (92-100 %, "
        "12 items per length): a 128k-token document becomes about 1,000 numbers per question. (4) Demo with the agents "
        "separate (A runs only the writer, B only the reader on its question row, 16 Oct 23:19 UTC): far lookup 97-100 % "
        "from about 20 fetched notes (5 KB down, 22 KB of requests up) on documents of 16k to 1M tokens, keyed 86-100 %, "
        "0 % with no notes or another document's; B reading the whole 1M notebook gets only 50 %. (5) E33f continued for "
        "0.6B more tokens: far quotes 4k / 16k 76 / 95 % through the notes alone (run 1 16 / 24 %, E33 on its own text "
        "0 / 0 %), far lookup and keyed unchanged, near facts and latest unchanged, story loss 1.498. Its demo: lookup "
        "97-100 %, keyed 86-100 %, word-for-word quotes 92-100 % from 15-21 fetched notes at 16k-1M tokens.",
        "Odra stopped answering on 16 Oct (about 18:05 UTC, likely a power cut) and was back on 17 Oct (about 09:05 "
        "UTC); E33e resumed from its last checkpoint.",
        "E33 on the bridge-question data (Polonez, 16 Oct 09:17-17:00 UTC, two bursts): stopped at the 24 GPU-hour cap at "
        "90 % of its tokens (about 1.35B; 0.33 steps per second against dense's 1.27). Notebook switched on: far lookup at "
        "4k / 16k 79 / 91 % (notebook off 23 / 0 %). Latest 22 % at 4k (floor 20 %), 4 % at 16k. Compose and deduce still "
        "at guessing (5-12 %, guessing 8 %): the loop did not join the two steps. Story eval loss 1.411 (dense on the same "
        "data 1.432). Exams on checkpoint 7038; not yet in the ledger. One run.",
        "Dense on the bridge-question data (Polonez, 16 Oct 06:08-08:57 UTC): the best dense run so far on every "
        "retrieval task — far facts at 4k lookup 81 %, keyed 80 %, quote 99 %, latest 92 %, count 50 %, crowded lookup "
        "82 % (dense on the old data: 56-63 %, 46-54 %, 3-94 %, 18-82 %, 17-38 %, 54-58 %), learned by 0.6B tokens; story "
        "loss 1.554 (old data 1.537-1.548). Compose and deduce still at guessing (one-step compose 12 %, guessing 8 %). "
        "Asked the bridge questions themselves it answers \"who is the teacher of X\" 69 % at 4k (old data 0 %): it "
        "knows both steps and cannot join them. One run.",
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
        "Odra: E33g (order heads) then E33h (count heads), 0.3B-token warm starts from E33r, about 1.5 h each.",
        "Polonez (multi-hop session): E33rb and dense on the reasoning data; then the content fix for deduce.",
        "Before the AWS run: the changes in section 14 of the E33 page (exit-loss cost, far-recall data from step 0, a "
        "switch-on detector, a compute-matched loop check).",
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
    {"claim": "Told which move is the newest and given a rank-among-matching-notes order term, the notebook learns latest.",
     "state": "holds", "since": "2026-10-18",
     "evidence": "E33j (E33q + every head ranked × a learned per-question switch, the last move as read and naming target): "
                 "latest 90 / 96 % at 4k / 16k (quick exam). Without the order term (E33q on the same data) 20 %; order on "
                 "demand without the labels (E33i) 22 %: both parts are needed.",
     "settle": "Two arms agree: E33j (order switched per question) 88 / 95 % and E33jf (always on) 88 / 94 % at 4k / 16k, "
               "evidence removed 0 %. A page-distance penalty instead (E33jd) reaches only 37 / 28 %. Open: a second seed, "
               "and whether it holds without the last-move labels."},
    {"claim": "Both 90 % recipes run a fixed number of hops: on 3-hop compose they stop after two.", "state": "single run",
     "since": "2026-10-18",
     "evidence": "Harder split, 3-hop compose: E33q 0 %, E33q seed 1 2-3 %, denser-mix E33ok 3 %. The pick is bridge 2's home "
                 "80-89 % of the time (the right one 2-4 %); E33r 24 / 33 %. Round embeddings are not the cause (any map, "
                 "even zero, scores the same). Deduce depth 3 works (66-92 %) because its chain ends on content.",
     "settle": "A per-round control over the question words (MAC-style), or training on 3 hops (the author's call)."},
    {"claim": "Training each loop round to name its hop's word lifts compose and deduce above 90 % on the id split.",
     "state": "holds", "since": "2026-10-18",
     "evidence": "E33q: E33r + a naming loss (round r names hop r's output word through the shared LM head, decoder "
                 "weights detached), 0.3B tokens: compose 94 / 98 %, deduce 99 / 100 % at 4k / 16k, evidence removed 0 %. "
                 "The feedback built beside it is unused (removing it at exam time changes nothing). Fails on new "
                 "sentence shapes (compose 6 %) and on more hops than trained (3-hop compose 0 %).",
     "settle": "Settled on the id split: E33q seed 1 (compose 94 / 98 %, deduce 99.5 / 99.5 %) and the naming-only arm "
               "E33qn (93.5 / 97 %, 98 / 99.5 %) repeat it. Open: new sentence shapes and hop counts."},
    {"claim": "On the reasoning data, 30M dense learns neither compose nor deduce, so E33r's compose comes from the "
              "loop and its read target.", "state": "single run", "since": "2026-10-17",
     "evidence": "Dense warm-started on the same 0.3B tokens as E33r: compose 8 %, deduce 6.5 % at 4k (one step 11 / 8 %), "
                 "while lookup 84 %, latest 92 %; E33r compose 60 %.",
     "settle": "A second dense seed, or dense with the same read-target-like supervision (attention to the fact)."},
    {"claim": "Compose fails at forming the second question, not at having the bridge in memory.", "state": "single run",
     "since": "2026-10-17",
     "evidence": "Oracle, no training, 150 items per cell: appending the gold first fact as text moves E33r's compose by "
                 "+6 points at most (58 → 64 % at 4k) and E33 bridge's by 0; writing the gold bridge into the question "
                 "gives 81 / 89 % (E33r) and 76 / 93 % (E33 bridge) at 4k / 16k. E33r decodes its own steps worse than "
                 "it composes inside (step 1 55-58 %), so writing its own answers back hurts (53 / 63 %).",
     "settle": "A loop whose second round is given the bridge it named (round target + re-embedding) reaches ≥ 70 %."},
    {"claim": "Deduce fails at storage: the notes do not keep the property.", "state": "single run", "since": "2026-10-17",
     "evidence": "Oracle: deduce stays at 7-15 % even with the gold category written into the question, on E33r and E33 "
                 "bridge. Property-word probe on the notes 8-13 % (place names 59-80 %).",
     "settle": "A content objective for the writer (read-back, address/content split) lifts the property probe ≥ 50 %."},
    {"claim": "Order on demand (every head ranked, scaled by a learned per-question switch) lets a notebook model "
              "learn latest.", "state": "retracted", "since": "2026-10-18",
     "evidence": "Probes on E33g: ordering every head with a wide match picks the newest move 50 % at 4k without training, "
                 "at a large cost to lookup and compose when applied to every question.",
     "settle": "E33i (Odra, 18 Oct): latest 22 % at 4k, the switch never opened. Next: E33j, where the latest "
               "question's last move is the read target and the naming target."},
    {"claim": "Fixed order heads (rank among matching notes) let a notebook model learn latest.", "state": "retracted",
     "since": "2026-10-18",
     "evidence": "Learned strengths from 0 never grew (E33d 0.14, E33e 0.15 then 0.09). Without training a strength "
                 "of 2 took E33 run 1 from 5 to 41 % at 16k, but E33r only from 20 to 28 % at 4k (2 → 7 % at 16k).",
     "settle": "E33g (Odra, 17 Oct): latest 22 % at 4k after a 0.3B warm start, under the 30 % kill line. Compose and "
               "deduce rose instead (69 / 24 % vs the matched control's 61 / 17 %)."},
    {"claim": "A summing read (count heads) lets a notebook model count.", "state": "single run",
     "since": "2026-10-18",
     "evidence": "A softmax read averages: every notebook run is at 22-33 % on count at 4k, dense 50 % at best.",
     "settle": "E33h (answer loss only): 23 % at 4k. E33k gives the sum the true number of visits."},
    {"claim": "At 30M and 0.5B tokens, a learned 16-vector message from agent A's notebook carries no facts to agent B.",
     "state": "holds", "since": "2026-10-16",
     "evidence": "E35 round 1 on Polonez, one run each: broadcast, steered by B's question, and B frozen with the loss on "
                 "answers only. 0 % exact on every fact task at 1k-128k. B's surprise is the same with the real message "
                 "and with another document's (far lookup at 4k: 2.31 vs 2.32, 2.32 vs 2.32, 2.32 vs 2.33). The receiver "
                 "takes about 85 bits per vector even when the vectors are optimised per text (Cramming probe).",
     "settle": "A transmitter trained much longer, or one that sends the notes themselves (next claim)."},
    {"claim": "E33's notes are a working agent-to-agent channel: B answers far facts from about 8 of A's notes, fetched "
              "with one request per read of the loop.", "state": "single run", "since": "2026-10-16",
     "evidence": "Notes-budget exam (no training), 40 items per cell. B reads another document plus A's notes. Far lookup "
                 "at 16k: own text 93 %, all of A's notes 93 %, 8 notes fetched by B's per-round requests 90 % (E33) and "
                 "98 % (E33f, trained with the questions cut off from the text). Keyed at 16k: 95 % from 8 notes, both "
                 "models. At 4k: 55-70 %, as with all notes. A single request after B's first read: 0-30 %.",
     "settle": "Repeat on another E33 seed and at 32k-128k (running)."},
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
    {"claim": "Trained on answers alone, no model learns compose or deduce at 30M / 1.5B, not even one step.",
     "state": "holds", "since": "2026-10-13",
     "evidence": "Every run at guessing on two-step compose and deduce (10 Oct on), and on the one-step versions "
                 "(exams v2.1, 13 Oct): dense 9-10 %, E33 6-8 %, E31c 4-10 % (guessing 8 %). Also with the single steps "
                 "taught (bridge data, dense and E33) and with E33's loop exits on answers only (E33x 7 % at 4k). "
                 "Supervising where each round reads changes this for compose (next claim).",
     "settle": "—"},
    {"claim": "Told in training where each loop round should read, E33 learns two-step compose, but not deduce.",
     "state": "single run", "since": "2026-10-17",
     "evidence": "E33r: E33 on the bridge data + 0.3B tokens with each round's read pushed onto the page of the next "
                 "fact (training only). Two-step compose 30 / 60 / 76 % at 1k / 4k / 16k (guessing 8 %); without the "
                 "target (E33x) 9 / 7 / 11 %. Far lookup at 16k 89 % (start 90 %). Deduce 17 / 13 / 9 %: the read "
                 "finds the right page 95 % of the time, the note lacks the property word (probe 11 %, guessing 8 %). "
                 "Quotes 0 %. The same target on single notes (E33o) erased far facts.",
     "settle": "A second E33r run; deduce once notes keep property words; the target switched off for the last part "
               "of training. Oracle (17 Oct): with the gold bridge in the question E33r reaches 81 / 89 %."},
    {"claim": "What latest lacks on the notebook is order, and ranking notes among their matches adds it without "
              "costing far facts.", "state": "single run", "since": "2026-10-16",
     "evidence": "No training, one fixed strength (2), exam set: E33 run 1 latest at 16k 5 → 41 %, far lookup 92 → "
                 "91 %; E33d run 1 2 → 30 %, lookup 93 → 92 %; keyed −2 to −5 (look-alike names count as matches). A "
                 "fixed page distance bought 33 % at the cost of far lookup 93 → 52 %. Gate chosen on fresh questions.",
     "settle": "Not by training it from 0 (next claim); a fixed strength in training is untested."},
    {"claim": "E33e: ranking among matching notes, with strengths learned from 0, fixes latest.", "state": "retracted",
     "since": "2026-10-17",
     "evidence": "E33e run 1 (Odra, 1.5B tokens): latest 19 % at 4k, 0 % at 16k (pass mark 60 % at 4k, floor 20 %). The "
                 "9 strengths stayed small (largest 0.15 mid-run, 0.09 at the end; the no-training check used 2). Far "
                 "facts unharmed: lookup 4k / 16k 81 / 93 %, keyed 75 / 91 %, quote 80 / 94 %. Same pattern as E33d.",
     "settle": "—"},
    {"claim": "E33's loop does no task work on text: 1 round scores like 4.", "state": "single run",
     "since": "2026-10-16",
     "evidence": "E33 run 1 with the loop stopped after 1 / 2 / 3 rounds: every task within 2 points of the trained 4 "
                 "rounds (latest at 4k 22 / 23 / 22 vs 22 %, far lookup at 16k 91 / 91 / 91 vs 92 %, compose 8 %, deduce "
                 "8 %). Its gain is story loss only (1.520 vs E31c 1.564).",
     "settle": "Rounds exams on a second E33 run; a task for the rounds (round targets) after the bridge data."},
    {"claim": "Taught the single steps, dense answers each of them but still cannot join two.", "state": "single run",
     "since": "2026-10-16",
     "evidence": "Dense on the bridge data: \"who is the teacher of X\" 69 % at 4k (dense on the old data 0 %); \"where "
                 "does X live\" (lookup) 81 %; \"where does the teacher of X live\" (one step) 12 %, guessing 8 %.",
     "settle": "E33 on the same data (running): does the loop join the two steps?"},
    {"claim": "Bridge questions make dense learn retrieval earlier and better.", "state": "single run",
     "since": "2026-10-16",
     "evidence": "Every retrieval task above all three old-data dense runs at 4k (lookup 81 vs 56-63 %, keyed 80 vs "
                 "46-54 %, latest 92 vs 18-82 %), learned by 0.6B tokens. One seed: the old-data runs show how much the "
                 "seed decides at this budget.",
     "settle": "A second seed on the bridge data."},
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
    {"id": "q8r_e33e_s0_odra.odra", "start": "2026-10-16", "data": "v2", "standing": "in use",
     "run": "E33e run 1 (Odra)", "model": "E33e · 30M, plain, seed 0 (E33 + rank among matching notes)",
     "tokens": "1.5B, 8 questions per document", "exams": "v2 + v2.1, 1k-16k; latest picks and strengths logged",
     "result": "Latest 19 % at 4k, 0 % at 16k (pass mark 60 %): failed. Strengths stayed near 0 (largest 0.15). Far facts "
               "at 4k / 16k lookup 81 / 93 %, keyed 75 / 91 %, quote 80 / 94 %; compose and deduce at guessing; story "
               "loss 1.520.",
     "when": "16 Oct 06:47 UTC → outage → resumed 17 Oct, finished 13:38; exams 16:48 UTC (Odra, 3 GPUs, 7.7 h)",
     "why": "Author: fixes for T4-T7 (16 Oct); spec E33e_note_rank. Gate set on fresh questions, never the exams."},
    {"id": "q8b_dense_s0_polonez.polonez", "start": "2026-10-16", "data": "v2 + bridge", "standing": "in use",
     "run": "Dense on the bridge data (Polonez)", "model": "dense · 30M, plain, seed 0",
     "tokens": "1.5B, 8 questions + up to 4 bridge questions per compose / deduce document",
     "exams": "v2 + v2.1, 1k-16k; bridge questions asked directly (diagnostic, CPU)",
     "result": "Far facts at 4k lookup 81 %, keyed 80 %, quote 99 %; latest 92 %; count 50 %; compose 8 % and one-step "
               "12 % (guessing 8 %); deduce 8 %. Bridge asked directly: teacher-of 69 %, category property 45 %.",
     "when": "16 Oct 06:08 → 08:57 UTC",
     "why": "Is one-step compose / deduce learnable at all once its steps are taught? The cheapest learner first."},
    {"id": "q8b_e33_s0_polonez.polonez", "start": "2026-10-16", "data": "v2 + bridge", "standing": "in use",
     "run": "E33 on the bridge data (Polonez)", "model": "E33 · 30M, plain, seed 0",
     "tokens": "about 1.35B (stopped at the 24 GPU-hour cap, 90 %)", "exams": "v2 + v2.1, 1k-16k",
     "result": "Far lookup 4k / 16k 79 / 91 %, keyed 73 / 88 %, latest 22 / 4 %, quote 25 / 0 %; compose and deduce at "
               "guessing. Split at its notebook (B reads only A's notes): far lookup at 16k 93 %.",
     "when": "16 Oct 09:17 → 17:00",
     "why": "Does the loop chain the taught steps? No: two-step compose stays at guessing."},
    {"id": "q8b_e33f_s0_polonez.polonez", "start": "2026-10-16", "data": "v2 + bridge", "standing": "in use",
     "run": "E33f, notes on the wire (Polonez)", "model": "E33 warm start, each question reads its document only "
                                                       "through the notes · 30M, plain, seed 0",
     "tokens": "0.3B fine-tune (1,568 steps)", "exams": "v2 + v2.1, 1k-16k (split by construction)",
     "result": "Split by construction. Far lookup 4k / 16k 77 / 91 % (E33 on its own text 72 / 90 %), keyed 66 / 92 % "
               "(69 / 90 %), far quote 16 / 24 % (0 / 0 %); near facts lose (lookup at 1k 34 vs 55 %, latest 15 vs 95 %). "
               "About 8 notes fetched by per-round requests answer far lookup at 16k-128k (92-100 %, 12 items). Story "
               "loss 1.516 (E33 1.525).",
     "when": "16 Oct 20:23 → 21:30",
     "why": "Agent to agent: make the notebook carry everything a question needs, in few notes."},
    {"id": "q8b_e33f_s0_cont_polonez.polonez", "start": "2026-10-16", "data": "v2 + bridge", "standing": "in use",
     "run": "E33f continued (Polonez GPUs 1-3)", "model": "E33f run 1 + 0.6B tokens of split training · 30M, plain, "
                                                         "seed 0",
     "tokens": "0.6B more (0.9B of split training)", "exams": "v2 + v2.1, 1k-16k (split by construction)",
     "result": "Far quote 4k / 16k 76 / 95 % through the notes alone (run 1 16 / 24 %, E33 on its own text 0 / 0 %); far "
               "lookup 74 / 93 %, keyed 70 / 92 %; near facts and latest unchanged (latest at 1k 15 %). Story loss 1.498.",
     "when": "16 Oct 23:23 → 17 Oct ~03:00",
     "why": "Does more split training close the near-fact gap and carry quotes?"},
    {"id": "reason_e33rb_s0.polonez", "start": "2026-10-17", "data": "v2 + bridge, reasoning phase", "standing": "in use",
     "run": "E33rb, read target + read-back (Polonez GPUs 0-1, multi-hop session)",
     "model": "E33r + zero-init note read-back 0.1 · 30M, plain", "tokens": "0.3B continuation", "exams": "v2 + v2.1",
     "result": "No harm, small gains: compose 61 / 82 % at 4k / 16k (E33r 60 / 76 %), deduce 17 / 12 % (13 / 9 %), lookup "
               "74 / 89 %, latest 20.5 %, count 22 %; story loss 1.514. The property-word probe stayed at 12 %: the "
               "read-back head learned token frequency, not the property.",
     "when": "17 Oct 18:24 → about 20:00 UTC; extra exams still running",
     "why": "Deduce: do the notes keep the property word (probe 11 %) without losing places?"},
    {"id": "reason_dense_s0.polonez", "start": "2026-10-17", "data": "v2 + bridge, reasoning phase", "standing": "in use",
     "run": "Dense on the reasoning data (Polonez GPUs 2-3, multi-hop session)",
     "model": "dense · 30M, plain, from the bridge-data dense", "tokens": "0.3B continuation", "exams": "v2 + v2.1",
     "result": "Compose 8 %, deduce 6.5 % (one step 11 / 8 %), with every fact visible: dense learns neither join in "
               "this budget, so it is no ceiling for T5 / T7. Lookup 84 %, keyed 83 %, latest 92 % (16k 1 %), count "
               "46 % at 4k.",
     "when": "17 Oct 18:24 → about 20:00 UTC",
     "why": "Is deduce learnable at 30M with this data when every fact is visible? Sets the ceiling for T5/T7."},
    {"id": None, "start": "2026-10-17", "data": "v2", "standing": "history",
     "run": "Decode-and-append oracle (Odra GPU 2, agent-memory session)",
     "model": "E33r and E33 bridge, no training", "tokens": "none", "exams": "compose / deduce, 4k and 16k, 150 per cell",
     "result": "E33r compose at 4k / 16k: as is 58 / 79 %; gold first fact appended 64 / 84 %; own steps appended 53 / 63 %; "
               "gold bridge written into the question 81 / 89 % (E33 bridge model 10 → 76 / 93 %). Deduce 7-15 % in "
               "every mode, gold bridge included.",
     "when": "17 Oct 18:28 → 18:52 UTC",
     "why": "Idea 4: with the first hop written into the document, can the second hop reach 70 %?"},
    {"id": "order_e33g_s0.odra", "start": "2026-10-17", "data": "v2 + bridge, reasoning phase", "standing": "in use",
     "run": "E33g, order heads (Odra GPUs 0-1)",
     "model": "E33r + four fixed order heads (rank among matching notes at 1, 2, 4, 8) · 30M, plain",
     "tokens": "0.3B continuation from E33r", "exams": "v2 + v2.1, 1k-16k",
     "result": "Latest failed: 95 / 71 / 22 / 4 / 2 % at 1k / 2k / 4k / 8k / 16k (kill line 30 % at 4k); the picks spread "
               "over the asked person's 11 places. Compose rose: 69 / 88 / 92 % at 4k / 16k / 32k (matched control E33rb "
               "61 / 82 %), deduce 24 / 19 % (17 / 12 %), lookup 76 / 88 %, keyed 73 / 87 %; story eval loss 1.393 "
               "(E33r 1.400). Exams at 64k paused for E33i (resumable).",
     "when": "17 Oct 18:46 → 20:25 training; exams 20:46 → 22:15 UTC (id split to 32k)",
     "why": "T4: order as a separable score term that cannot fade. Pass: latest ≥ 60 % at 4k, compose and far lookup "
            "within 5 points of E33r. Kill: latest ≤ 30 % at 4k."},
    {"id": "count_e33h_s0.odra", "start": "2026-10-17", "data": "v2 + bridge, reasoning phase", "standing": "in use",
     "run": "E33h, count heads (Odra)", "model": "E33r + two count heads (a summing read) · 30M, plain",
     "tokens": "0.3B continuation from E33r", "exams": "v2 + v2.1, 1k-16k",
     "result": "Count failed: 25 / 27 / 23 / 22 / 20 % at 1k-16k (kill line 35 %), the answer loss alone did not teach the "
               "summing read. Compose 67 / 88 %, deduce 17 / 8 %, latest 22 % at 4k / 16k.",
     "when": "17 Oct 22:05 → 18 Oct 01:00 UTC training (Odra GPU 1); exams running",
     "why": "T6: add matches up instead of averaging them. Pass: count ≥ 50 % at 4k (dense's best). Kill: ≤ 35 %."},
    {"id": None, "start": "2026-10-17", "data": "v2", "standing": "history",
     "run": "Order probes on E33g (Odra GPU 1, no training)",
     "model": "E33g final with other order settings at exam time", "tokens": "none",
     "exams": "latest / lookup / keyed / compose, 4k and 16k, 200 per cell",
     "result": "Content-judged match: latest 24 / 4 % (no change). Every head ranked, margin 0.5: strength 3 → 26 %, 8 → 29 %. "
               "Every head at 4, margin 2: newest picked 50 % at 4k (exact 33 %), 35 % at 16k (exact 14 %), but lookup "
               "80 → 54 %, compose 68 → 44 %. Margin 4: no better on latest, worse elsewhere.",
     "when": "17 Oct 21:46 → 22:30 UTC",
     "why": "Why did E33g's order heads not pick the newest move? The match set was too narrow (the home fact is the "
            "best match for 'live'), and order on every question costs the other tasks: order must be switched on per "
            "question (E33i)."},
    {"id": "order_e33i_s0.odra", "start": "2026-10-18", "data": "v2 + bridge, reasoning phase", "standing": "in use",
     "run": "E33i, order on demand (Odra GPU 0)",
     "model": "E33g with every head ranked (strength 4, margin 2) × a learned per-position switch (starts at σ(−2)) · "
              "30M, plain", "tokens": "0.3B continuation from E33g (its order strengths reset)", "exams": "v2 + v2.1",
     "result": "Latest failed: 22 % at 4k (kill line 35 %); the switch barely moved (bias −2.03 from −2). Deduce rose "
               "to 85 / 93 % at 4k / 8k (E33g 24 %), compose 71 / 84 %, lookup 77 %, keyed 74 % (exams to 8k so far).",
     "when": "17 Oct 22:21 → 18 Oct 01:10 UTC training (Odra GPU 0); exams running",
     "why": "T4: let 'now' questions turn order up while every other token keeps it off. Pass: latest ≥ 60 % at 4k with "
            "compose and lookup within 5 points of E33g. Kill: latest ≤ 35 % at 4k."},
    {"id": None, "start": "2026-10-18", "data": "v2 + bridge, reasoning phase, latest chain", "standing": "running",
     "run": "E33j, naming + order on demand, latest supervised (Odra GPU 1)",
     "model": "E33q + every head ranked (strength 4, margin 2) × a learned per-question switch · 30M, plain",
     "tokens": "0.3B continuation from E33q", "exams": "v2 + v2.1",
     "result": "Latest solved on the id split: 88 / 95 % at 4k / 16k (400 items; quick exam 90 / 96 %), the newest move "
               "picked 97 / 99 % (E33q 20 / 2 %); evidence removed 0 %. Compose 96 / 98 %, deduce 99 / 100 %, lookup 99 / "
               "100 %, count 22 / 25 %. The switch opens per question (bias −2.02, weight norm 0.46).",
     "when": "18 Oct 01:56 → 09:20 UTC training (Odra GPU 1, one 20-min cooldown); quick exam 09:35 UTC",
     "why": "T4: the data now gives each latest question a one-step chain (the asked person's last move), so the read "
            "target and the naming tell round 1 which move answers; the order term gives the read a way to find it. "
            "Pass: latest ≥ 60 % at 4k with compose and deduce within 5 points of E33q. Kill: latest ≤ 35 % at 4k."},
    {"id": None, "start": "2026-10-18", "data": "v2 + bridge, reasoning phase, latest chain + count visits + count naming",
     "standing": "queued", "run": "E33k2, the count decoded (Odra)",
     "model": "E33k + count loss 1.0 + the rounds name the count word (naming lens) · 30M, plain",
     "tokens": "0.3B continuation from E33k", "exams": "v2 + v2.1", "result": "—",
     "when": "18 Oct, after the count-naming data build", "why": "T6: E33k counts but does not say the number. Pass: "
     "count ≥ 50 % at 4k. Kill: ≤ 35 %."},
    {"id": "latest_e33jf_s0.polonez", "start": "2026-10-18", "data": "v2 + bridge, reasoning phase, latest chain", "standing": "in use",
     "run": "E33jf, rank term always on, latest supervised (Polonez GPUs 0-1)",
     "model": "E33q + every head ranked (strength 4, margin 2), no switch · 30M, plain",
     "tokens": "0.3B continuation from E33q", "exams": "v2 + v2.1, 1k / 4k / 16k",
     "result": "Latest 93 / 88 / 94 % at 1k / 4k / 16k (evidence removed 0 %): the switch is not needed. Compose 96 / 98 %, "
               "deduce 98 / 100 %, keyed 98 / 100 %; count fell to 11 / 12 %. Extra exams still running.",
     "when": "18 Oct 07:44 → 09:50 UTC training (Polonez); exams to about 11:30", "why": "T4: E33i's switch never opened; order always on."},
    {"id": "latest_e33jd_s0.polonez", "start": "2026-10-18", "data": "v2 + bridge, reasoning phase, latest chain", "standing": "in use",
     "run": "E33jd, page-distance heads, latest supervised (Polonez GPUs 2-3)",
     "model": "E33q + four fixed log page-distance heads (1, 2, 4, 8), five order-free · 30M, plain",
     "tokens": "0.3B continuation from E33q", "exams": "v2 + v2.1, 1k / 4k / 16k",
     "result": "Latest only partly: 92 / 37 / 28 % at 1k / 4k / 16k (evidence removed 0 %). A recency penalty by page distance "
               "is not enough; the rank among matching notes is. Compose 96 / 98 %, deduce 99 / 100 %, lookup 98 / 100 %.",
     "when": "18 Oct 07:44 → 10:40 UTC (Polonez)", "why": "T4: the page-distance alternative to the rank term."},
    {"id": None, "start": "2026-10-18", "data": "v2 + bridge, reasoning phase, latest chain",
     "standing": "running", "run": "E33q on the latest-supervised data (Odra GPU 2)",
     "model": "E33q (naming, no order term) · 30M, plain", "tokens": "0.3B continuation from E33q", "exams": "v2 + v2.1",
     "result": "Quick exam (200 items): latest still 20 / 2 % at 4k / 16k (the newest move picked 29 %) even with the last "
               "move as read target and naming target: without an order signal in the read, the supervision cannot be "
               "met. Compose 96 / 100 %, deduce 99 / 100 %, lookup 100 / 100 %, count 24 / 26 %.",
     "when": "18 Oct 02:35 → 05:15 UTC training; quick exam 05:45 UTC; full exams running",
     "why": "The ablation beside E33j: do the read target and the naming alone teach latest, or is the order term needed?"},
    {"id": None, "start": "2026-10-18", "data": "v2 + bridge, reasoning phase, latest chain + count visits",
     "standing": "running", "run": "E33k, count heads told the true number (Odra)",
     "model": "E33h + each count head's sum at a count question supervised with the number of visits (0.1) · 30M, plain",
     "tokens": "0.3B continuation from E33h", "exams": "v2 + v2.1",
     "result": "Quick exam: count still 26 / 22 % at 4k / 16k, but the count heads now count — their sum at the decision "
               "position is within 0.66 of the true number on average (rounded exact 49 %, 4k exams). The sum is not "
               "decoded into the number word. Deduce 46 % at 4k (E33h 17 %), compose 70 / 92 %, lookup 76 / 87 %.",
     "when": "18 Oct 02:28 → 08:30 UTC (Odra GPU 0, micro-batch 2 × 48); quick exam 08:46",
     "why": "T6: give the summing read a direct signal. Pass: count ≥ 50 % at 4k. Kill: ≤ 35 %."},
    {"id": "query_e33q_s1.polonez", "start": "2026-10-18", "data": "v2 + bridge, reasoning phase (output-word spans)",
     "standing": "in use", "run": "E33q seed 1 (Polonez, multi-hop session)", "model": "E33q, second seed · 30M, plain",
     "tokens": "0.3B continuation from E33r", "exams": "v2 + v2.1; round-embedding probes",
     "result": "Replicates: compose 94 / 98 %, deduce 99.5 / 99.5 %, lookup 98.8 / 99.2 %, keyed 97.8 / 99 % at 4k / 16k; "
               "story loss 1.523. Harder: deduce depth 3 87 / 94 %, compose 3 hops 2 / 0 %; paraphrase compose 6 %.",
     "when": "18 Oct, exams done 07:33 UTC", "why": "Confirm E33q before it becomes the base."},
    {"id": "query_e33qn_s0.polonez", "start": "2026-10-18", "data": "v2 + bridge, reasoning phase (output-word spans)",
     "standing": "in use", "run": "E33qn, naming only (Polonez, multi-hop session)",
     "model": "E33q without the round feedback · 30M, plain", "tokens": "0.3B continuation from E33r",
     "exams": "v2 + v2.1",
     "result": "Same as E33q: compose 93.5 / 97 %, deduce 98 / 99.5 %, lookup 98 / 99 %, keyed 98 / 99 %. The simpler base "
               "recipe; the feedback contributes nothing.",
     "when": "18 Oct, exams done 07:33 UTC", "why": "Is the naming loss alone the recipe?"},
    {"id": "dgate_e33rb_s0.polonez", "start": "2026-10-18", "data": "deduce bridge-only gate", "standing": "in use",
     "run": "Deduce gate (Polonez, multi-hop session)", "model": "E33rb + 0.1B tokens of deduce documents asking only their "
     "one-fact bridges · 30M, plain", "tokens": "0.1B continuation", "exams": "v2 + v2.1",
     "result": "The property step learns when asked densely: \"What is a kavi like?\" 12 → 44 %.",
     "when": "17-18 Oct night", "why": "Can this writer store a property word at all when paid for it? Yes."},
    {"id": "r2out_e33ok_s0.polonez", "start": "2026-10-18", "data": "denser reasoning mix (deduce 4 of 12, 8 bridges)",
     "standing": "in use", "run": "E33ok on the denser mix (Polonez, multi-hop session)",
     "model": "E33o + key-gradient stop, from the E33ok final · 30M, plain", "tokens": "0.6B continuation (0.9B from E33r)",
     "exams": "v2 + v2.1",
     "result": "Goal reached for T5 / T7 by a second recipe: compose 95 / 97 %, deduce 98 / 99 %, lookup 97 / 98 %, keyed "
               "95 / 100 % at 4k / 16k; story loss 1.493 (E33q 1.523); property word in the notes 72 % (probe). Latest 20 %, "
               "count 24 %. Harder: deduce depth 3 66 / 78 % (E33q 85 / 92 %), compose 3 hops 3 / 2 %.",
     "when": "18 Oct, finished 04:43 UTC", "why": "The key-stop line on denser data, without the naming loss."},
    {"id": "r2out_e33gk_s0.polonez", "start": "2026-10-18", "data": "denser reasoning mix (deduce 4 of 12, 8 bridges)",
     "standing": "in use", "run": "E33gk = E33ok + order heads (Polonez, multi-hop session)",
     "model": "E33ok + E33g's fixed order heads · 30M, plain", "tokens": "0.6B continuation", "exams": "v2 + v2.1",
     "result": "Same as E33ok (compose 95 / 96 %, deduce 97 / 99 %, count 27 / 28 %): the order term adds nothing once the "
               "reads are supervised this well.",
     "when": "18 Oct, finished 04:43 UTC", "why": "Isolates E33g's order heads on top of the key stop."},
    {"id": "query_e33q_s0.odra", "start": "2026-10-17", "data": "v2 + bridge, reasoning phase (output-word spans)",
     "standing": "in use", "run": "E33q, the loop asks with the name it found (Odra GPU 2)",
     "model": "E33r + rounds name their hop through the LM head (0.3) and pass the name, re-embedded, to the next round "
              "(gate from 0) · 30M, plain",
     "tokens": "0.3B continuation from E33r", "exams": "v2 + v2.1, 1k-16k",
     "result": "Goal reached for T5 / T7 on the id split: compose 94 / 98 / 97 % at 4k / 16k / 128k, deduce 99 / 100 %, "
               "lookup 98 / 100 %, keyed 98 / 99 % (E33r 60 / 76, 13 / 9, 74 / 87, 70 / 83); evidence removed 0 %. "
               "Latest 20 %, count 24 %, quote 26 % unchanged; story loss 1.523 (E33r 1.516). The same exam with the "
               "round feedback removed scores the same: the naming does the work. Notebook off at 4k: compose 6 %, deduce 8 %, "
               "lookup 23 %. Learned fast: deduce 96 % and compose 71 % by 0.07B tokens. Not on new sentence shapes "
               "(paraphrase compose 6 %, as every model) or more hops (harder compose 0 %, E33r 25 %).",
     "when": "17 Oct 19:24 → 22:10 training (Odra GPU 2); exams 22:10 → 18 Oct ~01:15 UTC",
     "why": "T5: the oracle put the gold bridge in the question and E33r went 58 → 81 % at 4k. Pass: compose ≥ 70 % at "
            "4k, far lookup within 5 points. Kill at mid-run: compose < 60 % or lookup < 65 % at 4k, < 75 % at 16k. "
            "Design reviewed by the multi-hop session."},
    {"id": "reason_e33ok_s0.polonez", "start": "2026-10-17", "data": "v2 + bridge, reasoning phase (output-word spans)",
     "standing": "in use", "run": "E33ok, note target + key-gradient stop (Polonez, multi-hop session)",
     "model": "E33o + the writer's key gradient stopped (a note's key learns addressing only) · 30M, plain",
     "tokens": "0.3B continuation from E33r", "exams": "v2 + v2.1",
     "result": "Deduce 36 / 35 % at 4k / 16k, the best so far (E33rb 17 / 12 %; one step 40 / 35 %); the notes keep the "
               "property word (probe 8-13 → 33 %). Costs compose 42 / 48 % (61 / 82 %), lookup 65 / 52 %, keyed 56 / 51 %; "
               "still climbing at the end. Latest 21 %, count 24 %, story loss 1.511.",
     "when": "17 Oct, exams 22:37 UTC",
     "why": "A loss on where reads land pulled note content away (E33o); with the key's gradient stopped, the answer "
            "trains the values of exactly the notes read. Next (multi-hop session): a separate address read for keys "
            "(lm_key_grad=address) so keys can carry the person's name again."},
    {"id": "reason_e33x_s0.polonez", "start": "2026-10-17", "data": "v2 + bridge, reasoning phase", "standing": "in use",
     "run": "E33x, loop exits on answers (Polonez)", "model": "E33 bridge warm start, loop exit losses off · 30M, plain",
     "tokens": "0.3B continuation (1,787 steps)", "exams": "v2 + v2.1, 1k-16k",
     "result": "Two-step compose 9 / 7 / 11 % at 1k / 4k / 16k (guessing 8 %); far lookup 4k / 16k 75 / 91 %; quotes 0 %.",
     "when": "17 Oct 09:20 → 10:37, exams 12:51 UTC (2 GPUs)",
     "why": "Multi-hop diagnosis step 1, the base rung: is it only data and a stuck loop? No."},
    {"id": "reason_e33r_s0.polonez", "start": "2026-10-17", "data": "v2 + bridge, reasoning phase", "standing": "in use",
     "run": "E33r, read target per round (Polonez)", "model": "E33x + each round's read pushed onto the next fact's page "
                                                         "(weight 0.5, training only) · 30M, plain",
     "tokens": "0.3B continuation (1,787 steps)", "exams": "v2 + v2.1, 1k-16k",
     "result": "Two-step compose 30 / 60 / 76 % at 1k / 4k / 16k, the first model above guessing; deduce 17 / 13 / 9 %; far "
               "lookup 4k / 16k 67 / 89 %, keyed 61 / 85 %; quotes 0 %; story loss 1.516.",
     "when": "17 Oct 08:27 → 09:18, exams 12:12 UTC (4 GPUs)",
     "why": "Supervise where each round reads (the agent-to-agent request, ported). One run."},
    {"id": "reason_e33o_s0.polonez", "start": "2026-10-17", "data": "v2 + bridge, reasoning phase", "standing": "in use",
     "run": "E33o, read target on single notes (Polonez)", "model": "E33x + each round pushed onto the notes that read the "
                                                                "hop's output word · 30M, plain",
     "tokens": "0.3B continuation", "exams": "v2 + v2.1, 1k-16k",
     "result": "Far facts erased: lookup and keyed 0 % at every length; compose and deduce at guessing.",
     "when": "17 Oct 11:14 → 12:50, exams 14:04 UTC (2 GPUs)",
     "why": "Can a note-level target fix deduce? No: it broke the notebook."},
    {"id": None, "start": "2026-10-17", "data": "v2 + bridge, reasoning phase", "standing": "history",
     "run": "E33b run 1, note read-back (Polonez)", "model": "E33o + a head that names each note's words (0.2, random start)",
     "tokens": "stopped at about half", "exams": "none",
     "result": "Stopped: the random head scrambled the warm-started reader (compose two-step 47 → 12 % first-token pick, "
               "bridge lookup 77 → 13 % at its first check).",
     "when": "17 Oct 11:41 UTC", "why": "Make notes keep property words for deduce; restarted with the head at 0."},
    {"id": "reason_e33b2_s0.polonez", "start": "2026-10-17", "data": "v2 + bridge, reasoning phase", "standing": "in use",
     "run": "E33b run 2, note read-back (Polonez)", "model": "E33o + read-back head from 0 (weight 0.1) · 30M, plain",
     "tokens": "0.3B continuation", "exams": "v2 + v2.1, 1k-16k",
     "result": "Far lookup 10 / 15 / 21 % at 1k / 4k / 16k; compose at guessing; deduce 27 % at 1k, 14 % at 4k.",
     "when": "17 Oct 10:52 → 13:03, exams 14:17 UTC (2 GPUs)",
     "why": "Does the read-back repair E33o's notes? Only a little."},
]

# ---- recipes (added 2026-10-18): the leaderboard of trained recipes, one row per recipe -------------------------------
# A recipe = architecture + training signals + data + where it started. `runs` are ledger ids (seeds of the same recipe);
# numbers come from the ledgers. `signals` keys SIGNALS; "privileged" signals use annotations only a generator has (which
# fact answers, which word each step finds, how many) — the board marks them so a reader never mistakes a supervised
# routine for an emergent skill.
SIGNALS = {
    "answers": ("answers", "arch", "trained on next-token loss with answers in the text (the default)"),
    "scratch": ("from scratch", "arch", "trained from random weights, 1.5B tokens"),
    "warm": ("warm start", "arch", "continued from another trained model"),
    "dense-data": ("task-dense data", "data", "reasoning mix: compose and deduce ×3, bridge questions asked directly"),
    "read": ("read target", "privileged", "told which fact each loop round should read (the generator knows the chain)"),
    "name": ("hop naming", "privileged", "told the word each loop round should find (the generator knows the chain)"),
    "keystop": ("key stop", "arch", "the writer's key gradient is stopped: keys learn addressing, values keep content"),
    "order": ("order heads", "arch", "the notebook read ranks newer matching notes (fixed strengths, no labels)"),
    "count": ("count heads", "arch", "a summing read beside the notebook read (no labels)"),
    "count-labels": ("count labels", "privileged", "told how many visits the answer counts"),
    "latest-labels": ("last-move labels", "privileged", "told which move answers a latest question"),
}
RECIPES = [
    {"key": "dense", "name": "Dense", "what": "full attention, the reference", "from": "from scratch, 1.5B (+0.3B "
     "reasoning data)", "signals": ["scratch", "answers"], "runs": ["q8b_dense_s0_polonez.polonez", "reason_dense_s0.polonez"]},
    {"key": "e31c", "name": "E31c notebook", "what": "local attention + notebook, no loop", "from": "from scratch, 1.5B",
     "signals": ["scratch", "answers"],
     "runs": ["q8r_e31c_s0_odra.odra", "q8r_e31c_s1_odra.odra", "q8r_e31c_s2_polonez.polonez"]},
    {"key": "e33bridge", "name": "E33 loop", "what": "notebook + read-think-reread loop (bridge-question data)",
     "from": "from scratch, 1.35B", "signals": ["scratch", "answers"], "runs": ["q8b_e33_s0_polonez.polonez"]},
    {"key": "e33r", "name": "E33r", "what": "each loop round told which fact to read", "from": "E33 loop + 0.3B",
     "signals": ["warm", "dense-data", "read"], "runs": ["reason_e33r_s0.polonez", "reason_e33rb_s0.polonez"]},
    {"key": "e33q", "name": "E33q / E33qn", "what": "+ each round names the word its hop finds", "from": "E33r + 0.3B",
     "signals": ["warm", "dense-data", "read", "name"],
     "runs": ["query_e33q_s0.odra", "query_e33q_s1.polonez", "query_e33qn_s0.polonez"]},
    {"key": "e33j", "name": "E33j", "what": "+ order on demand; latest told its last move", "from": "E33q + 0.3B",
     "signals": ["warm", "dense-data", "read", "name", "order", "latest-labels"], "runs": ["latest_e33j_s0.odra"]},
    {"key": "e33jf", "name": "E33jf", "what": "E33j with order always on (no switch)", "from": "E33q + 0.3B",
     "signals": ["warm", "dense-data", "read", "name", "order", "latest-labels"], "runs": ["latest_e33jf_s0.polonez"]},
    {"key": "e33ok", "name": "E33ok, denser mix", "what": "key stop + read target on notes, denser reasoning data",
     "from": "E33r + 0.9B", "signals": ["warm", "dense-data", "read", "keystop"],
     "runs": ["r2out_e33ok_s0.polonez", "r2out_e33gk_s0.polonez"]},
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
    "e33e_plain": ("E33e, rank among matching notes", "E33 plus a learned bonus for newer notes among those that match "
                                                       "the question (9 strengths, start at 0); plain input"),
    "e33f_plain": ("E33f, notes on the wire", "E33 warm start; every question reads its document only through the "
                                              "notes (agent to agent); plain input"),
    "e33x_plain": ("E33x, loop exits on answers", "E33 bridge model continued with a reasoning-heavy mix, loop exit "
                                                  "losses off; plain input"),
    "e33r_plain": ("E33r, read target per round", "E33x plus, in training, each loop round's read pushed onto the page "
                                                  "of the next fact in the chain; plain input"),
    "e33o_plain": ("E33o, read target on notes", "E33x plus each round pushed onto the single notes that read the "
                                                 "hop's answer word; plain input"),
    "e33b_plain": ("E33b, note read-back", "E33o plus a training head that names each note's words; plain input"),
    "e33rb_plain": ("E33rb, read target + read-back", "E33r plus a zero-start head that names each note's words; plain input"),
    "e33ok_plain": ("E33ok, key stop", "E33o with the writer's key gradient stopped; plain input"),
    "e33gk_plain": ("E33gk, key stop + order heads", "E33ok plus E33g's fixed order heads; plain input"),
    "e33g_plain": ("E33g, order heads", "E33r plus four fixed order heads (rank among matching notes); plain input"),
    "e33h_plain": ("E33h, count heads", "E33r plus two count heads (a summing read); plain input"),
    "e33i_plain": ("E33i, order on demand", "E33g with every head ranked, scaled by a learned per-question switch; plain input"),
    "e33q_plain": ("E33q, hop naming", "E33r plus: each loop round names the word its hop finds; plain input"),
    "e33qn_plain": ("E33qn, hop naming only", "E33q without the round-to-round feedback; plain input"),
    "e33j_plain": ("E33j, last move supervised + order on demand", "E33q plus order on demand, latest told its last "
                                                                    "move; plain input"),
    "e33jf_plain": ("E33jf, last move supervised + order always on", "E33j without the switch; plain input"),
    "e33jd_plain": ("E33jd, last move supervised + page distance", "E33q plus four page-distance heads; plain input"),
    "e33k_plain": ("E33k, count labels", "E33h with the count heads told the true number; plain input"),
    "e33k2_plain": ("E33k2, count decoded", "E33k plus the rounds naming the count word; plain input"),
    "e33m_plain": ("E33m, all four", "E33j plus count heads told the true number; plain input"),
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

# The author's goal for the text checks (17 Oct 2026): shown under the headline with the best score so far per task.
GOAL = {"set": "2026-10-17", "target": 0.70, "tasks": ["latest", "compose", "count", "deduce"], "lengths": [4096, 16384],
        "text": "70 % exact on T4-T7 with a notebook model (30M, the current exams)."}

# Decisions waiting for the author: shown at the top of the board. Remove an item when it is decided.
DECISIONS = [
    "Privileged signals: the 90 % recipes are told which fact each loop round reads and which word it finds (labels "
    "only the data generator has). Keep them as part of the method, or require the next step to learn from answers "
    "alone?",
    "3 hops: both recipes stop after the two hops they were trained on. Train on 3-hop chains, or first build a "
    "per-round control over the question (the multi-hop session's proposal)?",
    "From-scratch confirmation of the simplest 90 % recipe (E33qn: hop naming) at 1.5B tokens, 3 seeds (about 7 h × 3 "
    "GPUs each), once latest and count have a working fix?",
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
    {"date": "2026-10-18", "kind": "architecture", "change": "Order by page distance instead of rank among matches (E33jd)",
     "compared": "E33jf (rank term, always on) vs E33jd (four page-distance heads), same start, data and labels",
     "effect": "hurt", "metrics": [("latest at 4k", 88, 37, "%"), ("latest at 16k", 94, 28, "%")],
     "note": "A recency penalty cannot tell the asked person's last move from anyone's; the rank among matches can."},
    {"date": "2026-10-18", "kind": "recipe", "change": "Latest told its last move, with order on demand (E33j)",
     "compared": "E33q on the same last-move-labelled data, without vs with the order term", "effect": "helped",
     "metrics": [("latest at 4k", 20, 90, "%"), ("latest at 16k", 2, 96, "%"), ("compose at 4k", 96, 96, "%")],
     "note": "Quick exam, one run. Order alone (E33i) and labels alone both stayed at about 20 %."},
    {"date": "2026-10-18", "kind": "recipe", "change": "Latest told its last move (read target + naming), no order term",
     "compared": "E33q vs E33q + 0.3B tokens on the last-move-labelled data", "effect": "no effect",
     "metrics": [("latest at 4k", 20, 20, "%"), ("compose at 4k", 94, 96, "%")],
     "note": "Without an order signal in the read the supervision cannot be met. With order: E33j / E33jf / E33jd."},
    {"date": "2026-10-18", "kind": "recipe", "change": "Count heads told the true number (E33k)",
     "compared": "E33h vs E33h + 0.3B tokens with count labels", "effect": "mixed",
     "metrics": [("count at 4k", 23, 26, "%"), ("deduce at 4k", 17, 46, "%")],
     "note": "The heads now count (sum within 0.66 of the truth) but the answer is not decoded; E33k2 names the number."},
    {"date": "2026-10-18", "kind": "architecture", "change": "Order on demand: every head ranked × a learned switch (E33i)",
     "compared": "E33g vs E33i (from E33g, + 0.3B tokens)", "effect": "no effect",
     "metrics": [("latest at 4k", 22, 22, "%"), ("deduce at 4k", 24, 85, "%")],
     "note": "The switch never opened (bias −2 → −2.03); deduce rose, latest did not."},
    {"date": "2026-10-18", "kind": "recipe", "change": "Key stop + denser reasoning data (E33ok, multi-hop session)",
     "compared": "E33r vs E33ok on the denser mix (+0.9B tokens in all)", "effect": "helped",
     "metrics": [("compose at 4k", 60, 95, "%"), ("deduce at 4k", 13, 98, "%"), ("far lookup at 16k", 87, 98, "%")],
     "note": "A second route to the goal without the naming loss; the notes keep property words (probe 72 %)."},
    {"date": "2026-10-18", "kind": "architecture", "change": "Round-to-round name feedback (E33q vs E33qn)",
     "compared": "E33q (naming + feedback) vs E33qn (naming only), same start and data", "effect": "no effect",
     "metrics": [("compose at 4k", 94, 94, "%"), ("deduce at 4k", 99, 98, "%")],
     "note": "The naming loss does the work; the feedback gate stayed near 0."},
    {"date": "2026-10-18", "kind": "recipe", "change": "Each loop round names its hop's word (E33q)",
     "compared": "E33r vs E33r + 0.3B tokens with the naming loss (and an unused feedback)", "effect": "helped",
     "metrics": [("compose at 4k", 60, 94, "%"), ("deduce at 4k", 13, 99, "%"), ("far lookup at 16k", 87, 100, "%")],
     "note": "One run; id split only — new sentence shapes and extra hops do not transfer."},
    {"date": "2026-10-17", "kind": "architecture", "change": "Stop the writer's key gradient (E33ok)",
     "compared": "E33r + 0.3B tokens: page read target + read-back (E33rb) vs note read target + key stop (E33ok)",
     "effect": "mixed",
     "metrics": [("deduce at 4k", 17, 36, "%"), ("compose at 4k", 61, 42, "%"), ("far lookup at 16k", 89, 52, "%")],
     "note": "The first model whose notes keep property words; keys that can only reuse the content read lose the name."},
    {"date": "2026-10-17", "kind": "architecture", "change": "Fixed order heads on the notebook read (E33g)",
     "compared": "E33r + 0.3B tokens: with read-back (E33rb, matched control) vs with four fixed order heads (E33g)",
     "effect": "mixed",
     "metrics": [("compose at 4k", 61, 69, "%"), ("deduce at 4k", 17, 24, "%"), ("latest at 4k", 20, 22, "%")],
     "note": "Order sharpened selection among chain facts but did not pick the newest move; one seed each."},
    {"date": "2026-10-17", "kind": "exam", "change": "Oracle: the bridge written into the question vs appended as text",
     "compared": "E33r, compose at 4k, as is vs gold first fact appended vs gold bridge in the question",
     "effect": "measurement",
     "metrics": [("compose at 4k, first fact appended", 58, 64, "%"), ("compose at 4k, bridge in the question", 58, 81, "%")],
     "note": "The second question is the missing link; memory already has the bridge. The models did not change."},
    {"date": "2026-10-17", "kind": "recipe", "change": "Tell each loop round where to read, in training (E33r)",
     "compared": "E33 bridge model + 0.3B tokens, without (E33x) vs with the per-round read target", "effect": "helped",
     "metrics": [("two-step compose at 4k", 7, 60, "%"), ("two-step compose at 16k", 11, 76, "%"),
                 ("far lookup at 16k", 91, 89, "%")],
     "note": "First model above guessing on a two-fact question; deduce unchanged (9 → 13 % at 4k). One run each."},
    {"date": "2026-10-17", "kind": "recipe", "change": "Read target on single notes instead of pages (E33o)",
     "compared": "E33r (page target) vs E33o (note target), same continuation", "effect": "hurt",
     "metrics": [("far lookup at 4k", 67, 0, "%"), ("two-step compose at 4k", 60, 7, "%")],
     "note": "The note-level push erased the notebook; a read-back head on top (E33b) brought lookup back to 15 %."},
    {"date": "2026-10-17", "kind": "architecture", "change": "Rank among matching notes, strengths learned (E33e)",
     "compared": "E33 (3 runs) vs E33e (1 run), current recipe", "effect": "no effect",
     "metrics": [("latest at 4k, middle run", 20, 19, "%"), ("far lookup at 4k, middle run", 70, 81, "%")],
     "note": "The strengths start at 0 and stayed small (largest 0.15); the no-training check had used 2."},
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
