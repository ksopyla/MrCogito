"""Capability checks — what we know is wrong, what changed, what we learned, and what is training now.

The dashboard reads this file next to the ledger so every number on it carries its trust status:
  * an **open issue** marks the results it touches ("caution": shown, the verdict may still change) and says what to
    trust meanwhile and which run will settle it;
  * a **resolved issue** stays as history: what was wrong, what changed, what we learned — the record of past runs,
    so an old number is explained instead of silently replaced;
  * **runs in flight** say what is training, where, since when and which issue it settles.

Rules (author, 2026-10-13): a finding is recorded here in the same commit that finds it — never only in chat;
results are never deleted, they are re-labelled; a result is *trusted* when it matches the task definition
(`same`), runs on seeds 0–2 and no open issue touches it. Process: docs/engineering_specs/capability_checks.md
("Trust: open issues and history"), skill `capability-checks`.
"""
from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)
class Issue:
    id: str
    date: str                  # when it was found (CEST)
    title: str                 # plain words, one line
    severity: str              # critical | high | medium | low
    status: str                # open | rerun running | resolved
    effect: str                # caution (shown, verdict may change) | invalid (never evidence) | note (no cell marked)
    tasks: tuple[str, ...]     # tasks it touches (empty = none on the board)
    models: tuple[str, ...]    # models it touches (empty = every model)
    meaning: str               # what it means for reading the board now
    learned: str               # the lesson, for the history
    fix: str                   # what changed / what will settle it
    runs: tuple[str, ...] = ()  # run folders (RUNS) that settle it


ISSUES: tuple[Issue, ...] = (
    # ---- open: results shown with a caution mark until the named run lands
    Issue("decode-cost-cap", "2026-11-12", "The picked candidate at 128k is not decoded yet (cost cap)",
          "low", "open", "note", ("C1.edge-1k", "C1.keyed4-1k", "C5.pchain2-1k", "C5.path2-1k", "C5.path3-1k"),
          ("e33t_catalogue_n4",),
          "Since 2026-11-12 21:46 UTC (v4-draft-2026-11-12.2) a ladder reads 128k teacher-forced but does not decode "
          "the answer, so on the candidate-scored tasks the 128k point is hollow, 'not read, cost cap', and counts as "
          "the floor: it can only be lost. The verdict does not depend on it: the catalogue read already loses "
          "C1.edge-1k and C3.recall8-1k at the training length and C5.path2-1k at 2k–16k, and at 128k the champion "
          "is below the pass mark on the C5 tasks (path2 35 %, pchain2 9 %), so no capped C5 cell can be lost. C5.path2-1k seed 0's 128k was decoded "
          "in full (started before the cap). The first capped rung: C5.path2-1k seed 1 at 128k, written "
          "2026-11-13 01:53:49 UTC (first letter 67 %, picked not decoded). The 128k reference, C5.path2-1k seed 0's "
          "full decode, ended 2026-11-13 05:31:28 UTC: on its 64 rows candidate 0.1828125, exact 0.015625 (1 of 64), "
          "first_greedy 0.875, at 296 s a row decoded against 25 s read (5.3 GPU-h); raw ladder.json in the NAS "
          "archive e33t_core.polonez.",
          "The greedy decode costs one forward per answer letter; on a slow long-context read that is 12–15 forwards "
          "per row (33 s against a 2.7 s read at 32k), about 8 GPU-h a ladder at 128k. Time a decode on a model that "
          "learned before planning a ladder budget.",
          "Once the cached decoding (branch kv-cache, nn/incremental.py) lands on dev, extend it on e33t-capability to "
          "the capability catalogue model (sink keys in the sliding-window step; exactness from the layout, no page "
          "closing inside the answer span, instead of raw window − page, which is 0 here; a _greedy_answers adapter), "
          "each with an equality test: letter for letter against the uncached decode at 32k, and the exact aggregate "
          "scores of C5.path2-1k seed 0's full 128k decode on the same 64 rows. Then re-read the capped rungs with "
          "length_ladder.py --decode_cap off."),
    Issue("recipeA-seeds", "2026-10-13", "With answer exits, the loop learns 2-hop reasoning on 3 of 5 seeds",
          "medium", "open", "caution", ("C5.pchain2-1k",), ("e33a_loop",),
          "Picked candidate per seed with answer exits: 94 / 94 / 4 / 94 / 10 % (seeds 0–4); without exits 98 / 95 / 93 % "
          "(seeds 0–2, Polonez 2026-10-14). The median passes, but the registered variant fails 2 seeds in 5; the "
          "no-exit loop has not failed yet.",
          "A median over 3 seeds hides a 1-in-3 failure; headline claims need more seeds and the per-seed scores.",
          "Author's decision: make the no-exit loop the registered E33 variant on recipe A (and run its seeds 3–4).",
          ("v4_fix_c",)),
    Issue("catalogue-startup", "2026-11-13", "The catalogue read either learns a task or never starts",
          "high", "open", "caution",
          ("C0.copy-1k", "C1.edge-1k", "C1.keyed4-1k", "C1.lookup-1k", "C1.lookup-2k", "C3.recall8-1k",
           "C5.pchain2-1k"),
          ("e33t_catalogue_n4",),
          "Every catalogue-read job so far either learns to 97–99 % or stays at chance for its whole budget with "
          "nothing reaching the answer (flow 0.000), never in between. From random init 8 of 26 first stages started: "
          "C5.path2-1k and C5.path3-1k stage 1 on all three seeds (some only after 15–19k of their 19.2k steps), "
          "C1.edge-1k and C5.pchain2-1k on one seed of three (edge seed 1: 99 % at 1k, seeds 0 and 2 at chance); "
          "C0.copy-1k (seeds 0–1, ended 2026-11-13 08:01 UTC), C1.lookup-1k (seeds 0–1, ended 09:56 UTC), C1.keyed4-1k "
          "(seeds 0–1, ended 12:17 UTC: picked candidate 26 / 30 %, a guess gets 25 %), C1.lookup-2k (seeds 0–2 at "
          "both 1e-4 and 5e-5) and "
          "C3.recall8-1k (seeds 0–2) never started. So the read can learn a content lookup from scratch, and its "
          "losses at the training length are failures to start that depend on the seed, not a ceiling: they count as "
          "lost under the protocol, never as 'cannot'. A plausible cause, not tested: the note scores learn only "
          "where the read's attention already lands, and the read learns only through the answer, so evidence "
          "outside the answer's 256-token window gives no signal until a page is opened by chance.",
          "A read that must be found before it can be learned can fail by never starting; report how many seeds "
          "started, not only the median, and try a longer budget or a written warm-up before a 'cannot'.",
          "Not queued. More seeds or a longer budget on the missed tasks would settle it; a written warm-up stage would "
          "be a protocol change for every model. Seed 2 of C0.copy-1k, C1.lookup-1k and C1.keyed4-1k is still to come "
          "(Odra card 2)."),
    Issue("dense-lr-lookup", "2026-10-13", "Dense fails the 2k lookup only at our standard step size",
          "high", "open", "caution", ("C1.lookup-2k", "C1.lookup-16k"), ("dense", "e33t_catalogue_n4"),
          "Dense scores 24–29 % at step 1e-4 but passes at 5e-5 (the step E31's 2k runs use). 'Dense cannot do long "
          "lookup' is our setting, not dense's limit. The E33t catalogue read (2026-11-13) also stays at chance at "
          "1e-4 for the full 7,200 steps: 27 / 29 / 27 % first letter on seeds 0–2 (Odra, ended 11-13 00:32, 00:32 "
          "and 02:22 UTC). Its C1.lookup-16k curriculum starts with the same 2k stage at 5e-5, the second step size, "
          "and that ends at chance too: 22.7 / 28.5 / 21.9 % first letter on seeds 0–2 after the full 7,200 steps "
          "(ended 11-13 06:08, 06:08 and 11:40 UTC). So for the catalogue read the miss is not "
          "the step size: it fails the 2k lookup at both 1e-4 and 5e-5, where dense and E31 pass at 5e-5. Its "
          "C1.lookup-16k stages 2–3 (8k, 16k) would have started from these chance weights; the author dropped them "
          "(2026-11-13, marked on Odra 10:44 UTC), so its C1.lookup-16k stays not run.",
          "One step size for every model is fair only if it suits every model; check a second step size before a "
          "'cannot'.",
          "Write 5e-5 into C1.lookup-2k and rerun dense, E31 and E33 on it (not queued yet). The E33t catalogue "
          "read's part is settled by its C1.lookup-16k stage 1 (e33t_core on Odra): chance at 5e-5 on seeds 0–1.",
          ("e33t_core",)),
    Issue("chain4-counting", "2026-10-13", "The in-order chain can be answered by counting hop markers",
          "medium", "open", "caution", ("C4.chain4-1k",), (),
          "The chain is the first 4 hop blocks in reading order, so a reader that counts them answers without the "
          "keys. Dense 100 % vs memory models 66–71 % (the same on every seed) may be counting vs no slot order, not "
          "composition.",
          "An exam whose answer sits at a fixed position measures position, whatever its name says.",
          "A diagnostic (per-position accuracy, shuffled order with decoy chains) before the gap is read as composition."),
    Issue("keyed-forgiving", "2026-10-13", "The picked-candidate score is forgiving on long names",
          "medium", "open", "caution", ("C1.keyed4-1k",), (),
          "With 32-letter values and 4 candidates, a copy with only ~38 % of letters right still picks the right fact "
          "75 % of the time. Passes here say 'found the right fact', not 'copied it'.",
          "How forgiving the nearest-candidate score is depends on name length; report per-letter accuracy beside it.",
          "Score a fixed 8-letter prefix or require a margin (not done yet)."),
    Issue("path-early-stop", "2026-10-13", "Written-path runs stopped early on teacher-forced accuracy",
          "low", "resolved", "note", ("C5.path2-1k", "C5.path3-1k"), (),
          "Path runs stopped after 300–2,900 steps at 85–96 %. The passes stand; few-point gaps between models on "
          "these tasks are not meaningful.",
          "Stop and extend on the task's own score, on rows that are not the reported ones.",
          "New runs decide on the task's own score on 128 validation rows (stop rule, v4-draft-2026-10-13.2)."),
    Issue("ladder-noise", "2026-10-13", "Long-book points use 64 questions each",
          "medium", "open", "note", (), (),
          "Each long-book point has about ±5 points of noise, so the 5-point 'lost' margin is about one standard error. "
          "Both 'lost' items against the champion (keyed lookup at 32k, lookup at 2k) look like noise.",
          "Size the evaluation for the decision it feeds.",
          "256-question long-book reads for the headline tasks; a noise-aware no-harm margin (not done yet)."),
    Issue("public-blog-chain", "2026-10-13", "The 2026-09-28 blog post cites a chain result that did not start from scratch",
          "high", "open", "note", (), (),
          "'82 % at 128K' on the 4-hop chain came from a run started from lookup weights. From scratch the memory "
          "models score ~67 % and dense 100 %.",
          "Public numbers go through the same rules as the board.",
          "A dated correction note on the post (author's decision)."),
    Issue("hf-card-template", "2026-10-13", "The dataset-card script would overwrite the live Text World card",
          "medium", "open", "note", (), (),
          "The card template in the publish script is still the v0 card; the live v2 card text exists only on the "
          "E33 branch.",
          "Card sources live in the repo next to the generator they describe.",
          "Merge the v2 card source and update the template before any re-publish."),
    # ---- resolved: the history of what was wrong and what changed
    Issue("recipeA-compute", "2026-10-13", "The loop had about 3× the compute per training step on 2-hop reasoning",
          "high", "resolved", "note", ("C5.pchain2-1k",), ("dense", "e31_li_m1"),
          "Settled 2026-10-14 (Polonez): with 3× the steps (57.6k, the loop's compute) dense stays at guessing (14 / 21 / "
          "15 %, floor 25 %); E31 rises above guessing on 2 seeds but fails (37 / 49 / 8 %, median 37 %). The loop "
          "passes at a third of their steps: its advantage is not extra compute.",
          "Compare architectures at matched compute, not matched steps, before saying one cannot learn something.",
          "Dense and E31 reran with 57.6k second-stage steps (matched compute, no extension).",
          ("v4_fix_a", "v4_fix_b")),
    Issue("recipeA-exits", "2026-10-13", "The loop also got the answer loss after every round of re-reading",
          "medium", "resolved", "note", ("C5.pchain2-1k",), ("e33a_loop",),
          "Settled 2026-10-14 (Polonez): without answer exits the loop passes on all 3 seeds (98 / 95 / 93 %). The "
          "extra signal is not the source of its advantage; if anything the exits made it less reliable.",
          "Training aids specific to one architecture need their own ablation before the comparison is fair.",
          "The loop without answer exits, seeds 0–2 (shown as 'E33 loop, no answer exits').", ("v4_fix_b", "v4_fix_c")),
    Issue("unique-memory", "2026-10-14", "On the fixed unique-fact exam, the memory models fail",
          "medium", "open", "note", ("C6.unique-256",), ("e31_li_m1", "e33a_loop"),
          "With the blocks shuffled, dense passes (98–100 %) but E31 (34–46 %) and the E33 loop (39–47 %) fail on every "
          "seed; their old 'pass' was the first-block shortcut. Finding the one fact that appears once (aggregation) "
          "is a real gap of the memory models at 256 tokens.",
          "A shortcut can hide a missing capability, not only fake one.",
          "None needed for the board; a candidate research question (aggregation over the notebook)."),
    Issue("first-block", "2026-10-13", "Look-alike, decoy and unique-fact exams always put the answer first",
          "critical", "resolved", "invalid",
          ("X.lookalike-128-v3", "X.lookalike-1k-v3", "X.decoy8-1k-v1", "X.unique-256-v3", "X.unique-1k-v1"), (),
          "Every recorded C2 and C6.unique result is invalid: copying the first block answered 400 of 400 rows. They "
          "are kept as flawed tasks. Reruns on the fixed exams (Polonez, 2026-10-13): look-alikes still pass for every "
          "model (1k: one E31 and one E33 seed failed); the unique fact separates dense from the memory models.",
          "Before an exam goes live, check where its answer sits (position statistics over generated rows).",
          "The generator shuffles the blocks (--shuffle_blocks); C2 / C6 rerun for dense, E31, E33, seeds 0–2.",
          ("v4_fix_a",)),
    Issue("chain-floor", "2026-10-13", "The parallel-chain guessing floor was 25 %, not 8 %",
          "high", "resolved", "note", ("C1.edge-1k", "C5.path2-1k", "C5.path3-1k", "C5.pchain2-1k"), (),
          "Answering 'the node before a chain end' of a random chain is right 1 time in 4. Scores of 8–25 % (dense "
          "15 %, E31 9 % on 2-hop reasoning) are guessing. The 75 % pass mark still separates learning from guessing.",
          "Measure a floor with simple guessing rules on generated rows; never derive it by counting candidates.",
          "Floors set to 25 % (C1.edge, C5) and 48.5 % (C4.chain4-1k first letter)."),
    Issue("budget-names", "2026-10-13", "The '4×' and '16×' training budgets were the same 19,200 steps",
          "critical", "resolved", "note", (), (),
          "Calibration rounds 3–5 that tried 'a 16× budget' never trained longer. Their verdicts stand at 19.2k steps.",
          "A budget's name must state its steps.",
          "Budget comments corrected (TRAIN_19K); the calibration log says what was really run."),
    Issue("stop-rule", "2026-10-13", "Early stop and extension read teacher-forced accuracy on the reported rows",
          "critical", "resolved", "note", (), (),
          "Teacher forcing sat at 82–93 % on the reasoning plateau, so those runs never got their extension; path runs "
          "stopped early. Verdicts stand; the caps were the same for every model.",
          "In-run decisions read the task's own score on separate rows.",
          "Stop rule from v4-draft-2026-10-13.2: own score on 128 validation rows; --extend off for fixed budgets."),
    Issue("pchain-stale", "2026-10-13", "The 3- and 4-hop reasoning tasks kept their pre-freeze recipe",
          "low", "resolved", "note", ("C5.pchain3-1k", "C5.pchain4-1k"), (),
          "Their definitions still had 16-letter names and no written schedule after the 2-hop recipe was frozen.",
          "A recipe written as prose drifts; derive related tasks from one definition.",
          "Rewritten on the frozen 2-hop recipe (v4-draft-2026-10-13.1)."),
    Issue("recipe-b", "2026-10-07", "Writing the path out makes multi-hop easy for every model",
          "medium", "resolved", "note", ("C5.path2-1k", "C5.path3-1k"), (),
          "With the path written, k hops become k lookups: dense, E31 and E33 all pass, so these tasks test chained "
          "lookup, not reasoning inside the model.",
          "Separate 'composition inside the model' (final answer only) from 'chained lookup with a scratchpad'.",
          "Recipe A (C5.pchain*, final answer only) and recipe B (C5.path*, written path) are separate tasks."),
    Issue("old-pchain", "2026-10-05", "The first parallel chains could be guessed at ~40 %",
          "critical", "resolved", "invalid", ("X.pchain2-1k-v1", "X.pchain3-1k-v1", "X.pchain4-1k-v1"), (),
          "Every arm, dense included, sat on the guessing floor; all E31b / E33a parallel-chain numbers are invalid.",
          "Score multi-candidate exams on the picked candidate and print the guessing floor beside every score.",
          "C5 rebuilt: chain overhang, hop count in the question, picked-candidate score."),
    Issue("suite-step", "2026-10-04", "A different step size looked like a lost capability",
          "medium", "resolved", "note", ("C1.lookup-2k",), (),
          "E33a's 2k lookup at 1e-4 looked worse than E31 at 5e-5; at the same step size it was not.",
          "Compare two models only at the same step size.",
          "Results at a non-default step size are labelled 'settings differ' and never count."),
    Issue("warm-start", "2026-10-04", "Battery hard exams started from the run's own lookup weights",
          "high", "resolved", "note", (), (),
          "Length-battery chains, recall and decoys were not trained from scratch; they are shown as 'schedule differs' "
          "and never count.",
          "From scratch only; a curriculum is allowed only when written in the task, the same for every model.",
          "Rule 1 and rule 2 of capability checks v4."),
    Issue("shortcut-tasks", "2026-10-04", "Shuffled chains without decoys, triple-match and majority had shortcuts",
          "high", "resolved", "invalid", ("X.shuffled2-1k", "X.shuffled3-1k", "X.match3-1k", "X.majority-1k"), (),
          "Their scores do not show the capability they were named after.",
          "Look for a no-reasoning answer before trusting an exam.",
          "Kept as flawed tasks; replaced by the parallel chains."),
)
ISSUE_BY_ID = {i.id: i for i in ISSUES}


@dataclass(frozen=True)
class Run:
    folder: str        # the study folder; its ledger (when pulled) is `<folder>.<host>.json`
    host: str
    started: str       # CEST, or "queued"
    eta: str           # estimate, CEST
    what: str
    settles: tuple[str, ...]  # issue ids
    jobs: int


RUNS: tuple[Run, ...] = (  # finished runs stay listed until the next launch replaces them
    # E33t catalogue read (`e33t_catalogue_n4`), the v4 core package, phase core_e33t_catalogue_n4; times in UTC
    # (Odra's clock, queue logs and file times are Europe/Warsaw, one hour ahead in November; Polonez runs on UTC)
    Run("e33t_core", "odra", "2026-11-11 20:40 UTC",
        "cards 0–1 about 2026-11-14 17:00 – 11-15 00:00 UTC, card 2 about 11-16 00:00–12:00 UTC (from 11-13 04:40 UTC, "
        "worst case, every job at its full budget; the lookup curriculum's 8k and 16k stages, never counted here, were "
        "dropped by the author on 11-13 (marked 10:44 UTC), so the queues end here instead of hours later; card 2 runs "
        "about 1.7 times slower under the heat guard, 4.7 against 2.8 s a step, which pauses its job at 79 °C from "
        "11-12 21:38 UTC and at 78 °C since 11-13 01:59 UTC: peaks 79 °C, paused about 40 % of the time; early "
        "stops pull it in)",
        "E33t catalogue read, core package part 1: C0–C4 and the C1.lookup-16k curriculum's 2k stage (its 8k and "
        "16k stages dropped), seeds 0–2, ladders to 128k",
        (), 48),
    Run("e33t_core", "polonez", "2026-11-11 20:40 UTC (queued behind other evals)",
        "finished 2026-11-13 08:36:27 UTC, all 21 jobs (QUEUE-END: card 2 04:34:53, card 3 07:43:38, card 1 "
        "08:36:27 UTC); pulled 08:40 UTC",
        "E33t catalogue read, core package part 2: the C5 curricula (pchain2, path2, path3), seeds 0–2, ladders",
        (), 21),
    Run("v4_fix_a", "polonez", "2026-10-13 08:38", "finished 2026-10-13 17:39",
        "C2 / C6 on the fixed exams (dense, E31, E33, seeds 0–2); dense 2-hop reasoning at matched compute",
        ("first-block", "recipeA-compute"), 33),
    Run("v4_fix_b", "polonez", "2026-10-13 17:59", "finished 2026-10-14 01:35",
        "E31 2-hop reasoning at matched compute; the loop without answer exits, seed 0",
        ("recipeA-compute", "recipeA-exits"), 8),
    Run("v4_fix_c", "polonez", "2026-10-14 01:55", "finished 2026-10-14 13:12",
        "the loop without answer exits, seeds 1–2; the loop on the written recipe, seeds 3–4",
        ("recipeA-exits", "recipeA-seeds"), 8),
)


def issues_for(task_id: str, model: str) -> list[str]:
    """Open or running issues that mark this (task, model) cell (effect caution or invalid)."""
    return [i.id for i in ISSUES if i.effect in ("caution", "invalid") and i.status != "resolved"
            and task_id in i.tasks and (not i.models or model in i.models)]


__all__ = ["ISSUES", "ISSUE_BY_ID", "Issue", "RUNS", "Run", "issues_for"]
