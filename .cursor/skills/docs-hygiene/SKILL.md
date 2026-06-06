---
name: docs-hygiene
description: Periodically clean and reconcile the Concept Encoder `docs/` tree so active files contain only what is currently true. Use when the user asks to clean, prune, tidy, declutter, deduplicate, reconcile, or archive docs, when markdown files have accumulated stale ideas, outdated facts, contradictions, or actions that no longer make sense after a research or direction shift, or for routine docs maintenance. Prunes live context while preserving a compact record of what was abandoned and why. Not for logging a single new run (use experiment-tracking), code-change traceability (use engineering-change-tracking), or choosing research direction (use research-synthesis).
---

# Docs Hygiene

## Why this exists
`docs/` accumulates old ideas, superseded facts, and completed/abandoned actions after research or direction shifts. These contradict current truth and feed the agent **false context**. This skill removes that drift while preserving institutional memory.

Two goals that must both hold:
1. **Active docs = current truth only.** No contradictions, no dead actions, no stale facts.
2. **Nothing important is lost.** Abandoned ideas and failed experiments survive as a *compact, greppable record* ("we tried X, it failed because Y") so we neither repeat mistakes nor re-litigate settled decisions.

The resolution to "prune vs. keep" is: **compress, don't delete.** Replace verbose stale content with a one-line tombstone that points to the archive/ledger. Live context stays small; memory stays intact.

## Boundaries (do not overlap other skills)
- Logging one finished run or benchmark → `experiment-tracking`.
- Code-change traceability / `CHANGELOG.md` → `engineering-change-tracking`.
- Deciding the next experiment or shifting direction → `research-synthesis`.
This skill runs *after* those: it reconciles what they left behind and removes drift across the whole tree. Those skills flag drift and hand off bulk pruning/archiving here.

The project-wide convention "treat `docs/5_Archive/` and `> **OBSOLETE — ...**` content as historical, never as current context" is defined in `.cursor/rules/project-overview.mdc` → Docs Hygiene. This skill is what produces and maintains those markers.

## Safety rules (the user is wary of losing context)
1. **Git is the backstop.** Confirm the repo is committed/clean before bulk edits, so every change is recoverable. Prefer `git mv` when relocating files.
2. **Never delete a fact, result, or decision rationale.** Results and decisions are demoted/archived, never erased. Deletion is reserved for pure duplicates and noise (typos, scratch, redundant restatements) — and git still has the history.
3. **Verify before declaring stale.** Cross-check against ground truth (see below) instead of trusting prose. Prose lies; the ledger and code do not.
4. **Propose before bulk archiving.** For anything beyond small in-place fixes, present the disposition plan and get a go-ahead. Then execute.

## Ground truth (trust order)
When two docs disagree, the later/lower-level source wins:
1. Code + git history + `CHANGELOG.md`
2. `docs/2_Experiments_Registry/master_experiment_log.md` and `run_reports/` (the append-only result ledger)
3. `docs/1_Strategy_and_Plans/agenda.md` + `docs/experiments/<ID>.md` specs, `training_eval_matrix.md` (current plan/state)
4. `docs/1_Strategy_and_Plans/vision_and_goals.md`
Everything else (notes, idea files, literature review, application drafts) is downstream and may be corrected against the above.

## Doc lifecycle classes (rules differ by class)
Classify each file before editing it:

| Class | Files | Cleaning rule |
|---|---|---|
| **Source-of-truth / plan** | `agenda.md`, `docs/experiments/<ID>.md`, `training_eval_matrix.md`, `vision_and_goals.md` | Prune hard to current state. Completed/abandoned detail → compress to a tombstone or move to archive. Resolve every contradiction. Keep `agenda.md` to ~1 screen. |
| **Append-only ledger** | `master_experiment_log.md`, `2_Experiments_Registry/run_reports/*`, `4_Research_Notes/*` | **Never delete or rewrite results.** Only add a dated note when a later result supersedes an earlier *interpretation*. |
| **Idea / proposal** | `experiment_ideas/*` | Mark `ADOPTED` / `REJECTED` / `SUPERSEDED`. Rejected ideas keep a one-line reason; archive the file once it no longer informs decisions. |
| **Notes / review / drafts** | `research-notes/*`, `literature_review/*`, `sprind_frontier_ai/*`, `prompts/*`, `debugging/*` | Correct outdated claims in place; mark obsolete sections. These rarely need archiving unless fully dead. |
| **Archive** | `5_Archive/*` | Terminal. Must carry an OBSOLETE header. Don't re-clean content; only fix the header/pointers. |

## Workflow
Copy and track:

```
Docs hygiene progress:
- [ ] 0. Confirm git clean; pick scope (file, folder, or whole tree)
- [ ] 1. Build ground-truth snapshot of current reality
- [ ] 2. Scan scope; flag stale/contradictory/duplicate/dead items
- [ ] 3. Assign a disposition to each flagged item
- [ ] 4. Present plan; get go-ahead for archives/moves
- [ ] 5. Apply edits with the standard markers
- [ ] 6. Fix cross-references; report what changed
```

**Step 1 — Ground-truth snapshot.** Read the ground-truth sources for the scope. Note the current focus, the active experiment specs, the latest decisions, and which approaches are *set aside / parked* (vs. genuinely closed — prefer the softer framing unless a decision was explicit). This is the yardstick for every staleness call.

**Step 2 — Flag drift.** Within scope, flag:
- **Stale facts** — claims contradicted by ground truth (old "current best", retired interfaces, wrong file paths).
- **Dead actions** — "next experiment / TODO / plan" items already done, abandoned, or invalidated by a closed track.
- **Contradictions** — two docs (or two sections) asserting different current truth.
- **Duplication** — the same fact/plan restated across files; pick the canonical home, drop the rest.
- **Time-bound prose** — "currently", "for now", "this week", dateless status — pin to a date or move to the dated event.
- **Bloat** — long completed/running sections in plan files that belong in the ledger or archive.

**Step 3 — Disposition.** For each flagged item choose exactly one:

| Disposition | When | Action |
|---|---|---|
| **KEEP** | Still true | leave it |
| **UPDATE** | True intent, wrong detail | correct in place; use `~~old~~ new` for a revised claim worth showing |
| **TOMBSTONE** | Abandoned/superseded but worth remembering | replace the verbose block with one line: what + why-dead + gate/date + pointer (see Decision test) |
| **ARCHIVE** | Whole file/section is historical | add OBSOLETE header, `git mv` to `5_Archive/` (or move section under an "Abandoned" heading) |
| **DELETE** | Pure duplicate/noise, zero memory value | remove (git retains history) |

**Decision test (TOMBSTONE vs DELETE).** Keep a tombstone if dropping it risks **repeating a failed experiment or re-opening a settled decision**. Otherwise delete. When unsure, tombstone — it costs one line.

**Step 5 — Apply markers** (match existing repo style).

OBSOLETE header for an archived/superseded file:
```markdown
> **OBSOLETE — YYYY-MM-DD**
> Superseded by **[<file>](<relative-path>)**.
> **Why:** <1–2 lines: what changed and the decision/gate that killed this>
> Kept for historical reference; all current work follows the link above.
```

One-line tombstone inside a live plan file:
```markdown
- ~~<idea/experiment>~~ — **DEAD (YYYY-MM-DD):** <why> (gate: <metric/threshold>). See `<archive-or-report path>`.
```

Revised single claim:
```markdown
**Goal:** ~~<old goal>~~ <new goal> *(revised YYYY-MM-DD — see <pointer>)*
```

**Step 6 — Integrity.** After moves/archives, grep for references to moved files/sections and fix links. Confirm each source-of-truth file now reads as internally consistent and contradiction-free. Report a short summary: files touched, what was archived, what was tombstoned, any links fixed.

## Optional: dead-end ledger
If dead-ends are scattered, consolidating them into one compact file (e.g. `docs/5_Archive/abandoned_and_dead_ends.md`) — one line each: idea, why-dead, gate/date, pointer — gives a single greppable memory and lets plan files stay lean. Propose this before creating it; do not create new files unprompted.

## Anti-patterns
- Mass-deleting "old-looking" content without the ground-truth check.
- Erasing experiment results or decision rationale (demote/archive instead).
- Leaving an archived file without an OBSOLETE header (it will be read as current).
- Rewriting ledger results to "tidy" them.
- Editing many files silently — always end with the change summary.
