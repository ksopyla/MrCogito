---
name: research-comms
description: How to communicate research in this repo — chat Briefs (short, plain, no private codenames) and ledger/PR writeups (identity, gates S0/S1/K1/K2, bits/flow/bytes-per-token, one experiment at a time, plots, next-ONE vs STOP extra-steps). Use before writing ANY user-facing message, run report, spec/agenda/master-log update, or PR of a calibrated rung. Triggered as research-comms, research-comm, or research communication. File formats stay with experiment-track / experiment-design; this skill owns voice, identity, gates, and the next-step line.
---

# Research Comms

## Mission

**The chat is not the archive.** Specs, run reports, the master log and PR bodies are written to be
*searched later* — dense, indexed, number-heavy writing belongs there and should stay there. A chat
message is written to be *read once, now, by one tired human*. These are different products. Writing
the ledger into the chat is the failure this skill exists to stop.

The archive still has to be *true*. A dense report that calls an E18 control an E21 result, scores
a gate the dense arm never passed, or proposes seven next rungs is as broken as a 2,000-word chat.
The **Ledger SOP** below is the procedure for those files. File *shape* (section order, log columns)
stays with `experiment-track` / `experiment-design`.

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

## Budgets

| situation | first-pass budget | hard bans |
|---|---|---|
| status check ("what's the status?") | **60 words**, ≤3 bullets | tables, gate codes, Byobu/session names, GPU memory figures |
| result summary (one run or eval finished) | **120 words**, ≤4 bullets, 0–1 tables of ≤4 rows | metric names without a gloss |
| family / phase verdict (several runs closed at once) | **200 words**, ≤5 bullets, at most 3 of them findings (the rest: the cause, what survives) | a per-run table |
| experiment pitch / design | **150 words**: the idea, the bet, the test, the cost | spec section numbering, gate lists |
| implementation done | **120 words**: what it can do now, what changed for him | file manifests, flag names |
| "explain X" / "why" (a pull for depth) | **400 words**, analogy + one diagram; more only if he asks again | walls of numbers, undefined terms |
| long autonomous stretch ended | **150-word brief, mandatory** | ending a turn on "Now let me…" |

Budgets count the body **including any table cells**; the closing "Ask me about" line does not. A table
is not a loophole for smuggling the ledger into chat — if it does not fit the budget, it does not belong.

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
- a control's score under the bet's ID (E18 bits labeled as E21; a skipped arm reported as a pass)
- more than one next experiment in the close (the line is next-ONE or `STOP … extra-steps`)

A banned detail is not a deleted fact — it is a fact told in words. "Relaunching with half the batch size
and twice the accumulation, same tokens per step" carries the fix; `bs=8 accum=2` does not. The exact
setting belongs in the commit message, the launcher, or `CHANGELOG.md`.

## Self-check before sending

Read the draft and answer these. Any "no" means rewrite, not append.

- [ ] Does sentence one state the outcome, in words, with no codes?
- [ ] Am I inside the budget for this situation?
- [ ] Is every label something he can decode without memory?
- [ ] Does every number have meaning, direction, comparison and verdict?
- [ ] Would he know **what to do next** after reading only the opening sentence and the recommendation line?
- [ ] Am I answering the size of the question, or the size of my work?
- [ ] If there is a decision to make: is it near the top, with ≤2 options and my recommendation?
- [ ] Did I keep each architecture's numbers on that architecture's ID (no E18-as-E21)?
- [ ] Did I score the bet only after the solvability gate passed, and never via `0.75 × 0`?
- [ ] Is the next action **one** step, or an explicit **STOP extra-steps** — not a menu?
- [ ] If I compared two or more arms/lengths/tasks: is there **one** plot, not a number list?

## Working with the other skills

| product | format owner | this skill owns |
|---|---|---|
| chat message | this skill (the Brief) | everything in the five laws |
| run report | `experiment-track` | identity, gates, bits/flow, plots, next-ONE |
| spec `Status` / `Result` | `experiment-design` freeze; `experiment-track` close | latest-rung line, not the campaign diary |
| `agenda.md` learnings | `experiment-track` | one line, this ID only |
| master experiment log | `experiment-track` | one index row per ID; one run row per run |
| PR of a calibrated rung | this skill (shape below) | this cell first, then STOP/next-ONE |
| `CHANGELOG.md` | `engineering-change-tracking` | do not dump run metrics here |

- Dense output in the spec/report is correct; the same text pasted into chat is not.
- After writing a spec, plan or run report: the chat message is the **Brief**, plus where the rest lives
  by plain name ("the run report has it"). Not a summary of every section.
- During long autonomous work: keep the running one-line narrations (those were consistently good), and
  **always close the turn with a Brief**. Two audited turns ran 786 and 691 messages and closed with 13
  and 11 words of internal narration — after which he had to ask *"what was implemented?"*.

## Ledger SOP — docs, reports, PRs

Do this in order. Any skip is a rewrite, not a footnote.

### 1. Identity — never relabel a control

- An architecture's numbers stay on **that** architecture's experiment ID.
- **E18** is the raw full-causal read. **E21** is the exclusive compressed read (QUERY boundary + slots).
  Quoting E18 accuracy as an E21 score is a false result. Same for dense, `e18_local`, encoder-decoder.
- Do not reuse a previous cell's bits as this cell's score ("do not relabel 1024 MATCH2 61.81 as 1152").
- If dense missed the solvability bar and the other arms were skipped: write **not scored**. Never
  invent a pass, and never pass the bet via `0.75 × 0` because the control sat at chance.
- A matched replica of a control in this JSON is the comparison. A remembered number from last week
  is not, unless the spec says to cite that frozen ceiling.

### 2. One experiment, one rung, one result

- Chat, spec `Status`, and the top of the PR report **this closed cell**.
- A capability map (INDEX wall, MATCH wall, SELECT wall) is archive: run report and/or PR heading
  `## Capability map`. Not the chat Brief. Not a 200-line spec `Status` dump.
- Mixing two IDs in one verdict sentence is allowed only when both arms are in **this** JSON, each
  labeled (`dense 82% / E18 ~0 bits / E21 11 bits`).
- One-experiment-at-a-time is also a *writing* rule: do not close E18 and E21 in the same Result
  block. Two IDs → two reports, two log rows.

### 3. Gates — codes in the ledger, gloss in chat

Codes are **per spec**. Read that spec's Success/Kill list before scoring. The BAPO ladder's usual
meanings (E24/E25 and copies) are:

| code | what it asks | if it misses |
|---|---|---|
| **S0** | dense ≥ 75% in *this* replica — the exam is solvable | **K1** — do not score E18/E21 |
| **S1** | the bet vs its control, usually ≥ 0.75 × E18 in the same run; if E18 is ~0, vs 0.75 × dense | the rung fails; do not step length/recipe |
| **S2** | the required plots exist and are linked | write them before calling the cell closed |
| **K1** | instrument broken (dense < 75% after `k1_mult ×` budget) | skip other arches; 8k extra-step is off |
| **K2** | leak control (`e18_local`) above chance + 0.15 | stop; fix the generator, do not score retrieval |

Chat still glosses ("it missed the bar we set for the compressed read"). Docs and PRs **use the
codes**, defined on first use in that file (`**S1** (compressed read ≥ 75% of the raw-read control)`).
Do not score a code the spec does not name. Do not score S1 when S0 failed.

### 4. Bits, flow, bytes-per-token

A ladder cell in the ledger quotes this four-tuple, plus the prize:

| name | what it is | good direction |
|---|---|---|
| accuracy | share of answer tokens correct | higher |
| recovered bits | how many bits of the prize the model actually got (`floor − CE`, in bits) | higher; **0 = chance** |
| information_flow | recovered / prize, in `[0, 1]` | higher; 1 = full prize |
| B/tok (bytes per input token) | KV/state cost of that arm | lower for the same bits |

Name the prize once ("64-bit copy"). A pair like `53.82 vs 47.34 bits` must say which arm is which
and which comparison S1 uses. `BPB` (bits per *byte* of text, tokenizer-fair LM loss) is a different
metric — do not mix it with recovered bits. Definitions: `docs/glossary.md` and
`docs/engineering_specs/bapo_capability_ladder.md`.

### 5. Plots and artifacts

- **Ledger:** JSON + plots live with the run (`/opt/cursor/artifacts/<run>/` on Cloud, or `Cache/`).
  The run report links them. S2 is "the figure exists and is linked", not "I will plot later".
- **Chat / PR top:** **one** comparison figure that answers the question just asked — overlay
  learning curve (who learns vs who is stuck), wall heatmap (where it breaks as we scale), or
  frontier scatter (how much data to hit the bar). Legend in standing names
  (`dense — can reread any page`), not `Arm D`. Chance and pass-mark as lines.
- Do not dump a 6-PNG gallery. Do not replace a curve with `(6.66 → 5.53 → 5.02 → 4.17)`.
- Refresh the one figure when a new cell closes. Claim in the alt text.

### 6. Next-ONE vs STOP extra-steps

The last line of the report, PR, and chat Brief is **one** next action, or an explicit stop.

- **Extra-step** = the same recipe, same knobs, longer budget (typically 800 → 8k with
  `--no-dense_first` after dense already passed S0). It is not a new experiment.
- Default policy (override only if the live spec says otherwise):
  - 800 floor at chance → **no** extra-step.
  - Climbing, short of S1, S0 already passed → **one** extra-step, then stop.
  - S1 PASS, K1, or K2 → **do not** extra-step.
- After the cell: write `Next: <one concrete step>` **or** `STOP <recipe> extra-steps`.
  "STOP MATCH2 2-item length extra-steps. Do not 1100/1200." is a valid next action.
- Do not offer the next length *and* the next flag *and* the next recipe in the same close.
  Those are three experiments. Pick one, or stop.
- The Brief's "Ask me about" line is **depth topics**, not extra experiments. The
  recommendation / Next line is one action.

### 7. Spec, agenda, master log (after `experiment-track` format)

- **Spec `Status`:** one latest-rung verdict + report link. Not the campaign diary. Walls belong
  under Result (when closing) or in the latest report.
- **Spec `Result` (close only):** run id, WandB/artifact pointer, report link, one-line verdict
  against *this* spec's gates. Do not paste the table back into the spec.
- **`agenda.md`:** one "what we've explored" line for this ID (neutral, evidence, pointer).
  Move it off Current focus when the ID closes. Do not reprint the PR body.
- **Master log:** one Experiment Index row per ID; one Training/Eval row per run. Key result =
  one metric phrase + outcome, not a paragraph.

### 8. PR writeup of a calibrated rung

PRs are archive, so density is allowed. Shape, top to bottom:

1. **This rung first** — one table: arch · acc · recovered bits · flow · B/tok · step.
2. **Gates** with numbers (`S0 PASS 82%`, `S1 FAIL 11.00 vs 13.24 bits`, `K2 PASS`, `8k not run`).
3. **One plot** for this cell.
4. **Next:** one step or `STOP … extra-steps`.
5. Optional `## Capability map` *below* — walls only, no new numbers that are not in a report.

Title names the ID, the rung, and the verdict (`E25 seq=1152 MATCH2 — dense K1, E21 not scored`).
Do not mark an experiment `/goal` complete from a PR body. Do not open a second experiment's
rung in the same PR unless the spec is a declared family close.

## Glossary duty

`docs/glossary.md` is the shared vocabulary: every recurring metric, term and analogy in one plain-language
page. Keep it true:

- Using a project term in chat for the first time in a while → gloss it inline **and** check it is in the glossary.
- Coining a new metric, probe, gate or arm name in a spec → add the plain-language entry in the same commit.
- Renaming or retiring a term → fix the glossary in the same commit.

Link it (`docs/glossary.md`) instead of re-explaining at length, but never *replace* an inline gloss with
a link in a first pass — he should not have to click to understand sentence one.
