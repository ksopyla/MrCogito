# Capability boards: how we build them

A capability board is the **one dashboard** for a capability check: text checks, the symbolic/DNA capability suite,
and later the state-of-the-art evals. One person reads it top to bottom, like mission control. It must answer, in this
order and fast:

1. **Where are we against the goal, and what is happening now?** (status)
2. **Which recipe is closest to the goal?** (leaderboard)
3. **Which architecture is best when trained the same way?** (fair comparison)
4. **What changed the results?** (cause and effect)
5. **How sure are we?** (claims)
6. **What does each task test, and in which exam version?** (reference)

The skill `capability-boards` is the procedure; this file is the rulebook it follows.

- **Reference implementation:**
  - `analysis/text_board.py` (builder);
  - `analysis/text_board_template.html` (page);
  - `analysis/text_board_notes.py` (all judgement text).
- **First built:** 2026-10-15. **Revised:** 2026-10-18 (goal tracker, recipe leaderboard, training-signal disclosure,
  stale-content and naming checks).
- The DNA board (`docs/3_Evaluations_and_Baselines/capability_board.html`) and any SOTA-eval board follow the same rules.

## Why these rules exist

- **15 Oct 2026.**
  - The text board had grown to 11,100 px and 7,700 words.
  - Its status box was 1,080 words and its claims 1,700. Run details silently missed new runs.
  - Chart colours followed the seed, not the architecture.
  - The author could not find "which architecture is best" without reading most of the page.
- **18 Oct 2026, after a night of warm-start experiments.**
  - The comparison table still showed only the from-scratch families. The models that had just met the goal
    (warm-started recipes with extra supervision) appeared only as text in the run list, so the board's main table
    answered yesterday's question.
  - The headline and the decisions were a day old.
  - Five architectures had no plain name, so raw ids showed.
  - Claims in states the page does not draw ("disproved", "needs another run") were silently hidden.
  - Nothing told the reader that the 90 % models were told, in training, which fact to read: privileged labels that
    normal text does not have.

## Page order (fixed)

| # | section | answers | budget |
|---|---|---|---|
| 0 | sticky top bar: title + section tabs | where am I | one line |
| 1 | **Now**: dated status line; one-sentence headline; **goal tracker**; cards *Running now*, *Up next*, *Needs your decision*; "latest updates" collapsed | where we stand, what is happening | headline ≤ 45 words; ≤ 3 decisions |
| 2 | **Recipes**: one row per trained recipe, columns = tasks + held-out + controls + cost; training signals shown as chips | which recipe is closest to the goal | one screen on desktop |
| 3 | **Architectures**: one row per architecture family trained the same way (from scratch, equal budget) | which architecture is best when the recipe is fixed | one screen |
| 4 | **What changed the results**: change → before/after → verdict; newest 8 shown, older collapsed | what helps | one row per measured change |
| 5 | **What we know**: claims grouped by state, evidence collapsed | how sure are we | claim text ≤ 25 words |
| 6 | **Tasks and exam versions**: version timeline + task table with an example | what is tested, since when | one row per task |
| 7 | **Curves**: length and learning charts, one tab each | how it behaves | small multiples |
| 8 | **Runs**: compact list, newest first, results clamped to 2 lines, row expands | what was run | 5 columns |
| 9 | **Archive** (collapsed): protocol log, per-run details, calibration, sources | the record | unlimited |

- Above the fold on a laptop: sections 0–1 and the top of 2.
- Everything a reader needs daily lives in 0–6; everything kept for the record lives in 9.
- Nothing is deleted to make the page short: it moves down and collapses.

## Rules

### Status (section 1)

- **The headline:**
  - one sentence, outcome first, ≤ 45 words;
  - it names the best result against the goal, and its caveat in the same sentence (single run, in-distribution
    only, privileged signals);
  - it is rewritten whenever a result changes the answer to question 1. A headline older than the newest finished
    run is a bug.
- **The goal tracker** sits under the headline whenever the author has set a goal:
  - per task and per length, the best model of the candidate kind and the best reference model;
  - the target as a line, the guessing floor as a tick;
  - plain model names, never arch ids.
- **"Running now" and "Up next"** are generated from the run list (standing `running` / `queued`), never written twice.
- **"Needs your decision"** is a hand-written list:
  - at most 3 items, each a question the author can answer with one word;
  - remove an item when it is decided or overtaken;
  - re-read the whole list every time the headline changes.
- **The status line** shows the `as_of` time of the hand-written notes and the build time. If they differ by more than
  a working session, refresh the notes.

### Recipes (section 2)

A *recipe* is architecture + training signals + data + starting point. Warm-started fine-tunes, auxiliary losses and
special data are recipes, not architectures. They are judged here, never mixed into the architecture table.

- One row per recipe:
  - plain name, one-line "what", where it started ("E33r + 0.3B");
  - number of runs, and pending runs (ledger not in yet);
  - **training-signal chips**.
- **Training-signal chips** say what the model was told in training. Each signal is one of three kinds:
  - **architecture / start**: no labels (key stop, order heads, warm start);
  - **data**: a different training mix;
  - **privileged**, drawn in the warning colour: an annotation only the data generator has, such as which fact to read,
    which word each step finds, or how many.

  The legend says what privileged means. A reader must never mistake a supervised routine for an emergent skill.
- **Columns.**
  - Every task at the training length, with a toggle to the long length.
  - **Held-out** columns: what training never showed (new sentence shapes, one more hop, one more step).
  - The **notebook-off control**: the same model with its memory removed, which should fall to guessing.
  - Language cost (story loss).
- **Cells.**
  - The middle of the runs as the big number, the range underneath only when runs differ.
  - Green at or above the goal; grey within 10 points of the effective floor.
  - The run count is always visible: a single run reads as "1 run".
- Numbers come from ledgers only. A recipe whose runs have no ledger yet shows "pending", not hand-typed numbers.

### Architectures (section 3)

- Rows = architecture families trained the same way (from scratch, equal data, compute and parameters, each with its
  own tuned step size). Columns = tasks, in a fixed order.
- **Cells.**
  - The middle run as a large number and the range across runs underneath.
  - A dot strip, one dot per run (filled = learned), with ticks for the guessing rate and the pass mark.
  - Never print every run's number in a row of text.
- **Cell fill** encodes the state: learned in every run / in some runs / never. A ring marks the best architecture in
  the column, only when it learned the task and leads the next by more than the noise margin.
- A **"reliability" column** (k of n runs learned the core skill) sits next to the name: reliability is a result, not
  a footnote.
- One segmented control switches the measure (training length / longer / harder / new shapes), one the recipe. Both
  choices are remembered in localStorage, wrapped in try/catch.
- **The reference model** (dense) is the first row. Where it has no method (e.g. past its training length), the cell
  says so in words instead of showing 0.

### What changed the results (section 4)

- Every change of recipe, architecture, exam or reading rule whose effect was measured gets one row:
  - date and kind;
  - what was compared with what (the matched control);
  - before → after for 1–3 metrics;
  - one verdict chip: *helped / hurt / no effect / mixed / measurement*. "Measurement" means an exam or rule change;
    the models did not change.
- A change that helped one task and failed its own target is **mixed**, not helped.
- Before/after is drawn as a dumbbell on a 0–100 track, with the numbers beside it.
- The newest 8 rows are shown; older rows sit in a collapsed "Earlier changes".
- Numbers come from the ledgers or from a quick exam named in the row. Never write a before/after you have not checked.

### What we know (section 5)

- Claims are grouped by state, each group a card with its count. The states are exactly:
  - **holds**;
  - **single run** (shown as "Needs another run");
  - **open**;
  - **retracted** (shown as "Disproved", struck through, with the date).

  The builder refuses any other state, because the page would hide it.
- Word a claim as narrowly as the evidence: what was tested and what was not (e.g. "on the id split", "with privileged
  read targets").

### Tasks and exam versions (section 6)

- A version timeline: every exam version and reading-rule change, with:
  - date and a one-sentence change;
  - *in use* / *history*;
  - how many runs used it.
- A task table: name, the question in plain words, a short illustrative example, what the exam does, guessing rate,
  what is scored, and core vs frontier.
- When a task or its scoring changes, the table and the timeline change in the same commit.

### Runs (section 8)

- Newest first, 5 columns. Results are clamped to two lines; clicking a row opens the full result, why, tokens, exams
  and time.
- **Filters:** standing groups, exam version, date range. The default shows in use, calibration and active runs.
- Runs from other sessions belong here too, with the session named in the run title.

### Colour and marks (see the `dataviz` skill)

- **One colour per architecture,** fixed across every chart and table on the board. Runs (seeds) of one architecture
  share the colour and differ by line style (solid / dashed / dotted). Never colour by seed or by rank.
- **Status colours** (good / warning / critical / info / muted) are reserved for state chips, goal cells and privileged
  signals, never for a series.
- **Themes and widths:**
  - every colour token is defined for light and dark;
  - check both themes and a 375 px phone width before publishing;
  - no horizontal page scroll: wide tables scroll inside their own box with the first column pinned.
- **Numbers:** tables use tabular figures; a number always carries its unit or meaning ("%", "GPU-hours",
  "lower is better").

### Data and automation

- **Ledgers.** Every ledger with results appears in the recipes, run details and curves automatically. Hand-written
  notes are optional extras, never a gate.
- **Ledgers from other sessions** are merged through git (their branch), never copied by hand. Code still reaches
  servers through git only.
- **Notes.** One notes file holds all judgement text: status, decisions, goal, recipes, signals, claims, changes,
  architecture verdicts, run standings, plain names. Numbers come only from ledgers.
- **Names.** Every arch id in a ledger needs a plain name in `NAMES`, or the board prints raw ids. Check it before
  publishing.
- **Exam budgets.** Warm-start ladders get short exams: the training length, 4× longer and the held-out splits.
  Plan them with `--eval_lengths 1024 4096 16384`. The 64k–128k cells run only for a recipe that becomes a base.
  - A quick exam (200 items per cell) may report a result first; the full ledger replaces it.
  - Long exams that block GPUs are paused (they resume from their rows) and finished when the GPUs are idle.
- Rebuild, check, then publish to the **same** artifact URL.
- **Plain words** throughout (`research-comms`): no run ids, flags or file paths above the archive; an experiment ID
  always travels with a 2–4 word tag.

## Checklist before publishing a board

- [ ] The headline is one sentence, outcome first, ≤ 45 words, current with the newest result, its caveat included.
- [ ] The goal tracker and recipe leaderboard show plain names; the "pending" runs are the ones still in exams.
- [ ] Training signals are listed for every recipe; privileged ones are marked.
- [ ] The running / queued cards match the servers; the decisions list is current (≤ 3 items).
- [ ] Every finished run is in the recipes or architectures, the curves and the run list.
- [ ] Each new change has a "what changed" row with checked numbers and the matched control named.
- [ ] Claims are updated (state, date, narrow wording); the builder ran without a state error.
- [ ] The task table and version timeline match the exam version in use.
- [ ] Light, dark and 375 px phone checked; no console errors; no horizontal scroll.
- [ ] Published to the same URL; committed on the branch and merged to dev.
