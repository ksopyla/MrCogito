# Capability boards: how we build them

A capability board is the **one dashboard** for a capability check (text checks, the symbolic/DNA capability
suite, and later the state-of-the-art evals). It is read top to bottom by one person who wants four answers fast:

1. **Which architecture is best?** (compare)
2. **What changed the results?** (cause and effect)
3. **Which exam version produced this number?** (versions)
4. **What does each task test?** (reference)

Plus one operational answer: **what is running, what is next, what needs my decision?**

Reference implementation: `analysis/text_board.py` + `analysis/text_board_template.html` + the judgement text in
`analysis/text_board_notes.py` (first built to these rules on 2026-10-15). The DNA board
(`docs/3_Evaluations_and_Baselines/capability_board.html`) and any SOTA-eval board follow the same rules.

## Why these rules exist (review of 15 Oct 2026)

The text board had grown to 11,100 px and 7,700 words: 12 screens before the end, a status box of 1,080 words, the
claims table at 1,700 words, a 26-row run table with 7 columns, and run details that silently missed the newest
runs because they had to be listed by hand. Chart colours followed the seed, not the architecture, so one colour meant
two different models. Tasks were explained only in the middle of the page. The author could not find the answer to
"which architecture is best" without reading most of it. The rules below fix each of those.

## Page order (fixed)

| # | section | answers | budget |
|---|---|---|---|
| 0 | sticky top bar: title + section tabs | where am I | one line |
| 1 | **Now**: one-sentence headline; cards *Running now*, *Up next*, *Needs your decision*; "latest updates" collapsed | what is happening | headline ≤ 45 words; ≤ 3 decisions |
| 2 | **Compare**: one row per architecture, one column per task | which is best | one screen on desktop |
| 3 | **What changed the results**: change → before/after → verdict | what helps | one row per measured change |
| 4 | **What we know**: claims grouped by state, evidence collapsed | how sure are we | claim text ≤ 25 words |
| 5 | **Tasks and exam versions**: version timeline + task table with an example | what is tested, since when | one row per task |
| 6 | **Curves**: length and learning charts, one tab each | how it behaves | small multiples |
| 7 | **Runs**: compact list, newest first, row expands | what was run | 5 columns |
| 8 | **Archive** (collapsed): protocol log, per-run details, calibration, sources | the record | unlimited |

Above the fold on a laptop: sections 0–1 and the top of 2. Everything a reader needs daily lives in 0–5;
everything kept for the record lives in 8. Nothing is deleted to make the page short: it moves down and collapses.

## Rules

**Hierarchy and length**
- One idea per section; each section opens with a question as its title ("Which architecture is best?") and one
  sentence that says how to read it. The reading rules go in a collapsed "How it is scored".
- Long text is collapsed by default (evidence, protocol log, run details). A first screen never shows a paragraph
  over ~60 words.
- Status lists (`where`, `next`) are history, not the headline: the headline is one sentence that a person outside
  the project understands; the lists sit under "Latest updates".
- "Running now" and "Up next" are generated from the run list (standing `running` / `queued`), never hand-written
  twice. Decisions are a separate hand-written list (`DECISIONS`); remove an item when it is decided.

**Compare table (the heart of the board)**
- Rows = architectures (families), columns = tasks, in a fixed order. One cell = the middle run (median) as a large
  number, the range across runs underneath, and a dot strip (one dot per run, filled = learned) with the guessing
  rate and the pass mark as ticks. Never print every run's number in a row of text.
- Cell fill encodes the state: learned in every run / in some runs / never. A ring marks the best architecture in
  the column only when it learned the task and leads the next by more than the noise margin.
- A column "reliability" (k of n runs learned the core skill) sits next to the name: reliability is a result, not a
  footnote.
- Cost and no-harm columns at the right: language score vs the reference, compute per run.
- One segmented control switches the measure (training length / longer / harder variants); one switches the recipe
  or setup. The choice is remembered (localStorage, wrapped in try/catch).
- The reference model (dense) is the first row; where it has no method (e.g. past its training length) the cell
  says so in words instead of showing 0.
- Each architecture has a one-line plain description and a one-line current verdict (`ARCHS` in the notes).

**What changed the results**
- Every change of recipe, architecture, exam or reading rule whose effect was measured gets one row: date, kind,
  what was compared with what, before → after for 1–3 metrics, one verdict chip: *helped / hurt / no effect / mixed /
  measurement* (an exam or rule change; the models did not change).
- Before/after is drawn as a dumbbell on a 0–100 track (hollow = before, filled = after) with the numbers beside
  it; run counts ("0 of 3 → 3 of 3") use the same track as a fraction.
- Numbers come from the ledgers or from an existing claim; never write a before/after you have not checked.

**What we know**
- Claims are grouped by state (holds / needs another run / open / disproved), each group a card with its count.
  The claim sentence is the summary; evidence and "what would settle it" open on click. Disproved claims stay,
  struck through, with the date.
- Word a claim as narrowly as the evidence (state what was tested, and what was not).

**Tasks and exam versions**
- A version timeline: every exam version and reading-rule change with date, one-sentence change, *in use* /
  *history*, and how many runs used it.
- A task table: name, the question in plain words, a short illustrative example (document → Q → A, marked
  illustrative), what the exam does, guessing rate, what is scored (e.g. far facts only), core vs frontier.
- When a task or its scoring changes, the table and the timeline change in the same commit.

**Colour and marks** (see the `dataviz` skill)
- One colour per architecture, fixed across every chart and table on the board; runs (seeds) of one architecture
  share the colour and differ by line style (solid / dashed / dotted). Never colour by seed or by rank.
- Status colours (good / warning / critical / info / muted) are reserved for state chips and never used for a series.
- Every colour token is defined for light and dark; check both themes and a 375 px phone width before publishing
  (no horizontal page scroll; wide tables scroll inside their own box with the first column pinned).
- Numbers in tables use tabular figures; a number always carries its unit or meaning ("%", "GPU-hours",
  "lower is better").

**Data and automation**
- Every ledger with results appears in run details and charts automatically; hand-written notes (verdicts,
  observations) are optional extras, never a gate. A run that is on the compare table must also be in the curves.
- Judgement text (status, decisions, claims, changes, architecture verdicts, run standings) lives in one notes file;
  numbers come only from ledgers. Rebuild, then publish to the **same** artifact URL.
- Plain words throughout (`research-comms`): no run ids, flags or file paths above the archive; an experiment ID always
  travels with a 2–4 word tag.

## Checklist before publishing a board

- [ ] Headline is one sentence, outcome first, ≤ 45 words.
- [ ] Running / queued cards match the servers; decisions list is current.
- [ ] Every finished run is in the compare table, the curves and the run list.
- [ ] Each new change has a "what changed" row with checked before/after numbers.
- [ ] Claims updated (state, date, narrow wording).
- [ ] Task table and version timeline match the exam version in use.
- [ ] Light, dark and 375 px phone checked; no console errors.
- [ ] Published to the same URL; commit on the branch and merged to dev.
