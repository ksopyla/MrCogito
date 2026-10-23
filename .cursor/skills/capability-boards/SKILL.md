---
name: capability-boards
description: Build, update, review or redesign a capability board — the one mission-control dashboard per capability check (the text capability board, the DNA/symbolic capability board, later the SOTA-evals board), read top to bottom by one person who wants to know where we stand against the goal, which recipe and which architecture are best, what changed the results and how sure we are. Use when the user asks to update / refresh / redraw / publish a board, review its layout or usefulness, add a section or measure to it, record new results on it, or start a board for a new capability check. Owns the page order, the notes file (status, decisions, goal, recipes, training signals, claims, changes, names), the builder checks and the publish checklist. Not for running the exams (text-checks, capability-checks), interpreting results into the experiment registry (experiment-track), or chart colour theory (dataviz).
---

# Capability boards

**The rulebook:** `docs/engineering_specs/capability_boards.md` (page order, section rules, colour, automation,
checklist). Read it before changing a board's layout. This skill is the procedure.

**Reference implementation (text board):**
- builder `analysis/text_board.py`; page `analysis/text_board_template.html`;
- judgement text `analysis/text_board_notes.py`;
- output `docs/3_Evaluations_and_Baselines/text_capability_board.html`;
- ledgers `docs/2_Experiments_Registry/results/capability/text/*.json`;
- published at https://claude.ai/artifact/AUUgBTgGQwBDjvKiLzgXo9 (always republish to this URL).

## The principle: top to bottom, mission control

The reader is one tired human who must absorb the state in one pass. Each section answers one question, in this
order:

1. where are we against the goal, and what is running;
2. which **recipe** is closest;
3. which **architecture** wins on equal terms;
4. what changed the results;
5. how sure are we;
6. what each task tests.

Daily facts live in the first screens. The record lives in collapsed archive sections. Nothing is deleted; it moves
down and collapses.

## Updating a board after results land

```
- [ ] 1. Pull every finished run's ledger:
        - this session's runs: bash scripts/pull_text_checks_results.sh <host> <run...>
          (REMOTE_REPO=<checkout> when the run lives in another session's checkout; --no-archive if that session archives);
        - other sessions' ledgers: merge their branch (git), never copy files by hand.
- [ ] 2. Quick numbers while full exams run: a 200-item exam at the training length and 4x, read from the rows or a
         quick-exam JSON; say "quick exam" in the run's result text; the ledger replaces it.
- [ ] 3. Notes file, in this order:
        - status (as_of, headline <= 45 words with its caveat, where / next);
        - DECISIONS (<= 3, current);
        - GOAL (if set);
        - RECIPES (new recipe rows, their runs and signals) and SIGNALS (any new kind of training signal);
        - LADDER (each run's standing, result, when in UTC, why);
        - CHANGES (one row per measured change, matched control named, verdict);
        - CLAIMS (states: holds / single run / open / retracted only);
        - NAMES (every arch id in a ledger).
- [ ] 4. Rebuild: uv run python analysis/text_board.py. It refuses unknown claim states and warns on unnamed archs:
         fix both.
- [ ] 5. Check in the browser pane (launch config "board", http.server on docs/3_Evaluations_and_Baselines):
         no console errors; the headline and goal tracker read right; the recipes table shows plain names; light,
         dark and 375 px phone; no horizontal page scroll.
- [ ] 6. Publish with the Artifact tool to the same URL (file_path = the board HTML, url = the board's URL).
- [ ] 7. Commit (notes + ledgers + page) on the branch and fast-forward dev (git merge-base --is-ancestor first).
```

## Reviewing a board (when asked "is this usable / what is missing")

Measure, then judge against the rulebook:
- **Size:** page height and words per section, from the browser (sections' getBoundingClientRect, innerText).
  Budget: Now about one screen; no section over 2 screens except Runs and Archive.
- **Staleness:**
  - does the headline mention the newest best result?
  - are the decisions still open?
  - is as_of within a working session of the newest ledger?
- **Coverage:**
  - every finished run is in recipes or architectures, the curves and the run list;
  - nothing important lives only in a run's result text.
- **Honesty:**
  - single runs say so;
  - privileged training signals are marked;
  - held-out columns and the notebook-off control appear next to any headline score.
- **Readability:** no raw ids, flags or paths above the archive; every number has its unit and direction.

Report as a short list of issues ranked by how much they block the reader. Then fix the important ones in the same
turn, unless the user asked only for the review.

## Starting a board for a new capability check

- Copy the text board's three files.
- Keep the page order, the tokens, the claim states and the checks.
- Replace the task guide, examples, families and goal. For DNA or SOTA evals, the "recipe" is the training or prompting
  setup and the signals list says what each model was given.
- Add the new board to the rulebook's reference list.

## Pitfalls seen so far

- **A table that answers yesterday's question.** On 18 Oct the from-scratch architecture table was still the main table
  after warm-start recipes had met the goal. Recipes get their own section.
- **Hidden claims.** A claim state the page does not draw ("disproved", "needs another run") silently disappears.
  The builder now refuses it.
- **Raw arch ids in the goal tracker.** Add NAMES with every new arch.
- **Exam queues hog GPUs.** The 64k–128k cells take hours. Plan warm-start ladders with
  --eval_lengths 1024 4096 16384 and pause long exams (they resume from their rows) when GPUs are needed.
- **Watchers that mistake a heat pause for a crash.** Exit 75 means paused for cooldown. Match only real failures.
- **Two runs on one host need different MAIN_PROCESS_PORT values.**
