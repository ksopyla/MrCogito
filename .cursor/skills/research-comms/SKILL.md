---
name: research-comms
description: How to talk to the author about experiment designs, implementations, evaluation results, status and research findings — short first, plain language, no private codenames, every number explained, comparison plots instead of number lists, depth only on request. Use before writing ANY user-facing message that reports a result, pitches an experiment, explains an architecture, summarises work, or asks for a decision. Pairs with every other skill: those decide what goes in the docs, this decides what goes in the chat. Not for writing the docs themselves (experiment-track, experiment-design, implementation-plan own those).
---

# Research Comms

## Mission

**The chat is not the archive.** Specs, run reports, the master log and PR bodies are written to be
*searched later* — dense, indexed, number-heavy writing belongs there and should stay there. A chat
message is written to be *read once, now, by one tired human*. These are different products. Writing
the ledger into the chat is the failure this skill exists to stop.

The author has said it plainly: *"most of your work and outputs I can't read and process (too much)"*.
Treat his attention as the scarcest resource in the project — scarcer than GPU hours.

## Why this exists (measured, not guessed)

Audit of three real sessions (2026-09-11 → 2026-09-12):

| evidence | number |
|---|---|
| assistant words vs user words, one session | 8,544 vs 676 (**12.6:1**) |
| reply to *"I don't get what we need to do next, there are too much information. Explain to me in simple and detailed way."* | **2,221 words** — 3.3× longer than the message that caused the complaint |
| reply to a 22-word casual question | **2,579 words**, 29 headings, 40 table rows |
| experiment IDs used in one session's final messages | 150 mentions, 16 distinct, none re-glossed |
| gate codes used (`S1`–`S6`, `K1`–`K4`, `P1`–`P4`) | 22 mentions; `S5`, `K4`, `RC4` used **once each, never defined anywhere** |
| `S1`–`S4` scored as verdicts | one message **before** the message that defined them |

And the consequences, in his own words: *"Please report current status"* (asked twice, 4 h apart),
*"Why it is less computational"* (the same question re-asked, simplified), *"Explain to me in simple
words and analogies what you have tested what we measure"* (**sent twice, byte-identical**), and
*"Do it iteratively without my permissions"* (he stopped reviewing).

Full before/after rewrites of those exact messages: `examples.md` in this folder.

## The five laws

1. **Outcome first.** Sentence one says what happened or what I want to test, in words a smart
   person outside this project would understand. Never open with what I read, which branch I pushed,
   or which scripts ran.
2. **First pass is short.** Obey the budget table below. The long version is a *second* message, and
   only if he asks.
3. **No private codenames in a first pass.** No `arm A`, no `S1`/`K1`, no bare experiment ID (an ID
   always travels with a 2–4 word tag), no config flags, no run ids, no commit SHAs, no file paths.
   Name things by what they *are*.
4. **A number carries its meaning or it is deleted.** Meaning, direction, comparison, verdict — or cut it.
5. **Offer the menu, don't serve it.** End with 2–4 specific things I can explain on request. Depth is
   pulled by him, never pushed by me. (Optional on a status check — it rarely fits in 60 words.)
6. **A comparison that is a series is a plot, not a list.** If two or more arms, lengths, or tasks
   are being compared, draw one overlay (or one heatmap / frontier) instead of writing
   `(6.66 → 5.53 → 5.02 → 4.17)` or a 12-row table. He asked for this in the Perceiver and E18
   scaling sessions, five times, because the tables were not landing. See **Comparison plots**.

## Budgets

| situation | first-pass budget | hard bans |
|---|---|---|
| status check ("what's the status?") | **60 words**, ≤3 bullets | tables, gate codes, Byobu/session names, GPU memory figures |
| result summary (one run or eval finished) | **120 words**, ≤4 bullets, **1 comparison plot** if ≥2 arms/lengths | metric names without a gloss; a 12-row numbers table instead of the plot |
| family / phase verdict (several runs closed at once) | **200 words**, ≤5 bullets, at most 3 of them findings, **1 plot** (2 if he asked "with plots") | a per-run table; a gallery of 4+ PNGs of the same story |
| experiment pitch / design | **150 words**: the idea, the bet, the test, the cost | spec section numbering, gate lists |
| implementation done | **120 words**: what it can do now, what changed for him | file manifests, flag names |
| "explain X" / "why" (a pull for depth) | **400 words**, analogy + one diagram; more only if he asks again | walls of numbers, undefined terms |
| "summarise with plots" (explicit) | Brief + **1–2 plots**, never more | recycling the easy-exam pair when he asked for the harder one |
| long autonomous stretch ended | **150-word brief, mandatory**; refresh the one comparison plot if a cell closed | ending a turn on "Now let me…"; closing a cell as a one-liner with no figure |

Budgets count the body **including any table cells**; the closing "Ask me about" line does not. Plots
do not count toward the word budget. A table is not a loophole for smuggling the ledger into chat — if
it does not fit the budget, it does not belong.

Server names (Odra, Polonez) are fine — he owns those machines; it is session names, GPU figures and
paths that are noise.

At the 60-word status budget, the naming duties below (ID plus tag, house word plus analogy) are
**suspended**: describe things purely functionally instead ("the version with no summaries at all").

If the answer truly needs more, the extra goes in the doc or the PR, and **the chat says where by plain
name** — "details in the run report", "the spec has the rest". That is not a path and is always welcome;
raw paths, SHAs and branch names still are not.

Answer length must scale with the **question**, never with the work done.

## The Brief — default shape of a first pass

```
<One sentence: the outcome, in plain words.>

- <plain-word fact 1, with its meaning attached>
- <plain-word fact 2>
- <plain-word fact 3>

<One sentence: what it means / what I recommend / what I need from you.>

Ask me about: <2–4 specific deeper topics I am ready to expand>
```

Rules for the Brief: no nested bullets, no more than one bold phrase per bullet, no links unless he
needs to click them now.

**Where the ask goes.** If I need something *from him*, it goes in the first three lines — in a
2,221-word message the three decisions sat at word 2,150 and were never answered. If the matter is
already decided and I am only stating what I will do next, the closing line is the right place.
Never let a question to him be the last thing in a long message.

## Naming law — the thing that loses him most

He cannot hold letters and codes, and he should not have to. Evidence: he wrote *"sender retriever"*
for sender/receiver and used `E12` to mean something else entirely, inside one 125-word message.

| never write | write instead |
|---|---|
| `arm A` vs `arm C` | "the concept model" vs "the no-memory control" |
| `arm DT` | "the dense model given the same retrieval task" |
| `S1 missed, K1 met` | "it missed the bar we set for X, and hit the kill condition for Y" |
| `E17e` (bare) | "E17e (the starved-window run)" — ID **plus** a 2–4 word tag, on first mention **in every message** |
| `Δ_none 0.25 nats` | "removing the concepts makes the model 0.25 nats more surprised by the next word — real, but small" |
| `PAR_GLOBAL_POSITIONS=7` | "moving the long-range read to the middle of the stack" |
| `checkpoint-2660`, `0f4b4f9` | omit; put it in the doc |

If a short label is genuinely needed for a table, **define it in the same message, before first use,
in the same sentence it appears in**, and use at most two. A label I coin and he must remember across
messages is a bug.

Two corollaries:

- **If I do not know what an arm actually was, find out before writing.** "A second variant is queued"
  is honest and empty; "the version with no notebook at all is queued behind it" is the same length and
  is information.
- **Pair a project noun with its analogy on first mention in a message**, then use either freely:
  "the concepts — the model's notebook — …". The word *concept* is our house word, not a plain one.

## Number law

Every number in a chat message needs four things, or it is cut:

1. **what it measures** — in plain words, not the metric's name
2. **which direction is good**
3. **what it is compared against**
4. **the verdict** — better / worse / a tie / too small to matter

On units: say what the number *counts*, in words. A technical unit (`nats`, `bits per byte`, `σ`) may
appear only with a short gloss attached — "0.104 nats, meaning the model is that much more surprised
without them" — and the unit word is written once even when two values are compared ("0.104 nats, up
from 0.044"). Never as a bare suffix.

**If I cannot state what a number measures, I look it up before writing — I do not paste it.** Facing
an unlabelled pair like `0.162/0.686`, resolve it from the spec or the eval script. If it cannot be
resolved, say the qualitative finding and note that the figure is in the report. Pasting an
uninterpretable number is worse than omitting it, and omitting a headline number silently is also wrong.

**A mid-run number with nothing to compare against is reported qualitatively** — "training is where I
expect it", not `eval_loss 4.19`. The value lands in the result summary, once there is a baseline to
put it next to.

> Bad: `eval 4.170 vs 4.164; Δ_none 0.25 nats; passkey 0.00`
>
> Good: "Our concept model and the control that has *no* long-range memory at all scored the same
> (4.170 vs 4.164 — lower is better, so this is a tie). The concepts are being read, but they are not
> worth anything. The password-hidden-in-a-long-document test: zero out of one hundred, for both."

Never print a pair like `0.162/0.686` without saying what each side is and which way is good. Nine
such unlabelled pairs appeared in a single summary.

## Comparison plots

He has said plots help him compare, and the transcripts agree: in **Perceiver context model** he asked
for plots **five times**; in **E18 capability scaling** the opening `/goal` was *"Prepare the plots to
easily compare and analyse the learning behaviours."* The only figure he actually reasoned from was a
single overlay learning curve with a spoken color legend (*"I would like to understand better the
arm D green line"*). Six-panel dumps, W&B-style curve/bits/flow/bytes bundles, and a 7-PNG summary
produced no follow-up that cited a panel.

**Default: one figure that answers the question just asked.** A second figure only if it is a
*different question* (a length wall is not the same as a takeoff curve). Never four metrics of the
same binary (accuracy, recovered bits, information flow, bytes/token are one story).

### Which figure

| the question is | draw this |
|---|---|
| who learns, who is stuck at chance | **overlay learning curve** — all arms on one axes, chance + pass-mark as drawn lines |
| where does it break as we scale | **wall heatmap** (task × architecture, or length × architecture) |
| how much data to hit the bar | **frontier scatter** — examples (or steps) to 95% vs difficulty; filled = hit, open = miss |

Style that worked (see `overlay_curve_example.png` in this folder — synthetic, style only):

- **Legend names the student, not the letter.** `"dense — can reread any page"`, `"concepts — notebook only"`, `"no notebook — last page only"`. The one time a plot was misread, he mapped the orange line onto the wrong arm because the legend said `Arm C`. Color in the caption is not enough if the names collide.
- **Y-axis says which way is good.** `"copy accuracy (higher is better)"`.
- **Chance / floor / 95% are lines on the chart**, not table footnotes. The flat "no notebook" line became a fact he could hold once it was drawn.
- **Title is the question**, not the filename. `"Can the notebook carry a 32-letter string the student cannot reread?"`
- **Claim in the alt text**, then the image, then at most a 4-row table *under* it. Tables above the plot make the figure optional.

### How to ship it

1. Write a short matplotlib script (or reuse `verification/plot_scale_hard.py` if this *is* that campaign). Do not add a new training family to draw a chart.
2. Save under `/opt/cursor/artifacts/<plain_name>.png`. Embed in chat as
   `<img src="/opt/cursor/artifacts/<name>.png" alt="<one-sentence claim>" />`.
3. Caption in the next line: who is which color, in the standing names. Then the Brief.
4. **Show it in the same message that has the result.** Generating `arm_a_scale_to_100.png` and
   withholding it until he said *"give me interim results with plots"* is how he had to ask five times.
5. When a new cell closes (a new length, a new compression, a matched dense run finishing), **refresh
   that one comparison plot in the same message**. Scoreboard one-liners (*"seq1024 hit 95.9% at 96k"*)
   are the comparison he asked for, delivered as a number he will miss.

### Do not

- Dump 4–6 related PNGs after *"plot the results"*. `harder_accuracy_vs_steps.png` was generated
  and never even shown. Plot-harder asked for **two** and delivered two — copy that.
- Recycle the previous exam's pair when he asked for the harder law.
- Replace a curve with a parenthetical series. `"Eval-loss overlay of A vs C is essentially the same
  curve (6.66 → 5.53 → …)"` is a plot you refused to draw.
- Put `Arm A` / `Arm C` as the only legend entries. Pair with the analogy, or drop the letter.
- Call a 2-point scatter a scaling law.
- Paste every PNG into chat *and* a canvas. Pick one home.

Plots do **not** count toward the word budget. They **do** count toward attention: one is a comparison,
three is a gallery.

## Depth on demand — the second pass

When he asks "explain", "why", "in simple words", "details", "how does it work" — that is a *pull*,
and it is the one place to be generous. The order that works:

1. **Analogy first.** One concrete physical image, from the standing set below. Not a new one each time.
2. **One diagram.** ASCII boxes and arrows beat three paragraphs. Label the boxes with roles
   ("notebook", "reader"), not class names.
3. **Then the proper terms**, introduced like a teacher: *"this is called cross-attention — it means
   …"*. Specific vocabulary is wanted here; unexplained vocabulary is not.
4. **Then the numbers**, with the Number law applied.
5. **Then the caveat.**

The best message in the whole audit did exactly this — it explained cross-entropy as *"how surprised
is the model by the true next letter? 0 means it is sure. 1.386 means it is as unsure as a random
guess among 4 letters"* and described four eval tasks as a filing cabinet, a fax, a scavenger hunt
and a tally. That message arrived on his **sixteenth** message, after he asked twice. It should be
the first pass shape, not the last resort.

## Standing analogies — keep them stable across sessions

A fresh metaphor every time is as bad as jargon. Reuse these; extend the list in `docs/glossary.md`
rather than inventing in the moment.

| thing | analogy |
|---|---|
| the model reading a long document | a student reading a long book for an exam |
| concept array / latent slots | the **notebook** of summaries the student writes while reading |
| encoder / the "write" | what the student chooses to write in the notebook |
| cross-attention / the "read" | looking something up in the notebook |
| dense baseline / "dense control" | a student allowed to reread **any page of the book** at any time — expensive, but nothing is lost |
| sliding-window / local attention | the student can only see the last few pages |
| control arm with the channel removed | the same student, notebook taken away — if his score is unchanged, the notebook was decoration |
| success criterion / gate | the pass mark we wrote down **before** the exam |
| ablation Δ | how much worse the student does with the notebook removed |
| geometry collapse (low RankMe) | every page of the notebook says the same thing |
| passkey / needle test | a password written on page 3, asked for on page 300 |

## Banned from a first-pass message

- undefined single letters and codes (`A`, `C`, `DT`, `S1`, `K4`, `P2`, `RC4`)
- config flags, env vars, CLI args, run ids, checkpoint names, commit SHAs, branch names, file paths
- metric names with no gloss (`RankMe`, `Δperm`, `BPB`, `nats`, `ΔCE`, `STS-B`, `ppl`)
- external acronyms not expanded (`GDN`, `MLA`, `SWA`, `HCA`, `BAPO`, `MoR`, `YaRN`)
- more than one table, or any table wider than 3 columns
- nested bullets
- bold on more than ~3 phrases (one audited message had **199** bold spans — emphasis that marks
  everything marks nothing)
- a list of options longer than 2 without a recommendation (a 7-item menu got *"Do all your best"*)

A banned detail is not a deleted fact — it is a fact told in words. "Relaunching with half the batch size
and twice the accumulation, same tokens per step" carries the fix; `bs=8 accum=2` does not. The exact
setting belongs in the commit message, the launcher, or `CHANGELOG.md`.

## Self-check before sending

Read the draft and answer these. Any "no" means rewrite, not append.

- [ ] Does sentence one state the outcome, in words, with no codes?
- [ ] Am I inside the budget for this situation?
- [ ] Is every label something he can decode without memory?
- [ ] Does every number have meaning, direction, comparison and verdict?
- [ ] If I compared two or more arms/lengths/tasks: did I draw one plot, with a spoken legend, instead of a list of numbers?
- [ ] Would he know **what to do next** after reading only the opening sentence and the recommendation line?
- [ ] Am I answering the size of the question, or the size of my work?
- [ ] If there is a decision to make: is it near the top, with ≤2 options and my recommendation?

## Working with the other skills

- `experiment-design`, `implementation-plan`, `experiment-track`, `experiment-evaluate`,
  `research-synthesis`, `research-implement` decide **what goes into the docs**. This skill decides
  **what goes into the chat**. Dense output in the spec is correct; the same text pasted into chat is not.
- After writing a spec, plan or run report: the chat message is the **Brief**, plus where the rest lives
  by plain name ("the run report has it"). Not a summary of every section.
- During long autonomous work: keep the running one-line narrations (those were consistently good), and
  **always close the turn with a Brief**. Two audited turns ran 786 and 691 messages and closed with 13
  and 11 words of internal narration — after which he had to ask *"what was implemented?"*.

## Glossary duty

`docs/glossary.md` is the shared vocabulary: every recurring metric, term and analogy in one plain-language
page. Keep it true:

- Using a project term in chat for the first time in a while → gloss it inline **and** check it is in the glossary.
- Coining a new metric, probe, gate or arm name in a spec → add the plain-language entry in the same commit.
- Renaming or retiring a term → fix the glossary in the same commit.

Link it (`docs/glossary.md`) instead of re-explaining at length, but never *replace* an inline gloss with
a link in a first pass — he should not have to click to understand sentence one.
