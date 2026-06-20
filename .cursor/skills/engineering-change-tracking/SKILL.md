---
name: engineering-change-tracking
description: Maintain project traceability for Concept Encoder code changes. Use after refactors, architecture updates, bug fixes, training or evaluation script edits, implementation improvements, or when updating CHANGELOG.md, docs/1_Strategy_and_Plans/agenda.md, commit messages, or architecture git tags. Not for run metrics, benchmark bookkeeping, or experiment selection.
---

# Engineering Change Tracking

## Scope
Use this skill for code-change traceability:
- refactors and cleanups
- architecture changes
- bug fixes
- training or evaluation script edits
- implementation improvements
- direction shifts that should appear in project docs

Do not use this skill to log run metrics or benchmark results. Use `experiment-track` for training and evaluation records.
Do not use this skill to choose research priorities. Use `research-synthesis` for hypothesis, benchmark, and direction decisions grounded in external evidence.

## Keep aligned
When a change matters, keep these artifacts consistent:
- `CHANGELOG.md`
- `docs/1_Strategy_and_Plans/agenda.md` (and the relevant `docs/experiments_specs/<ID>.md` if the change is tied to an experiment)
- git commit message
- architecture tags such as `arch/{feature}` when justified

## After a meaningful code change
1. Update `CHANGELOG.md` with a dated entry.
2. Describe why the change happened and what it changes at a high level.
3. Mark related items in `docs/1_Strategy_and_Plans/agenda.md` (or the relevant `docs/experiments_specs/<ID>.md`). If a commit already exists, include the commit hash.
4. Create an architecture tag if the change establishes a new milestone or architecture variant.
5. Make sure the commit message matches the actual engineering intent.

## Commit prefixes
| Prefix | Use for |
|--------|---------|
| `arch:` | New architecture or significant structural change |
| `refactor:` | Internal cleanup or code simplification |
| `feat:` | New capability |
| `fix:` | Bug fix |
| `train:` | Training pipeline or training script change |
| `eval:` | Evaluation pipeline or benchmark script change |
| `docs:` | Documentation-only change |
| `test:` | Test additions or test fixes |

## CHANGELOG rules
- Group related edits in one entry.
- Capture motivation, impact, and the high-level change summary.
- Avoid file-by-file noise.
- Call out refactors, interface changes, or direction shifts explicitly.
- Do not add a `CHANGELOG.md` entry just because a run finished with no code changes.

## Template
```markdown
## [YYYY-MM-DD] - Short Title

**Why:**
- Hypothesis, motivation, or problem being addressed

**Impact:**
- Expected user, research, or engineering effect

**What changed:**
- [refactored] `path/file.py`, `path/other.py` - grouped summary
- [added] `path/new_file.py` - short purpose

**Git tag:** `arch/feature-name`
**Related:** `docs/1_Strategy_and_Plans/agenda.md` or `docs/experiments_specs/<ID>.md` -> "experiment / item"
```

## Related skills
- Use `experiment-track` when the same change also needs run metadata or benchmark results recorded.
- Use `research-synthesis` when the change reflects a new hypothesis, evaluation plan, or research direction informed by external evidence.
- Use `docs-hygiene` when a change makes existing docs stale or contradictory (retired interfaces, superseded plans) and they need pruning or archiving. Update the directly affected docs here; hand off broader cleanup to `docs-hygiene`. Treat `docs/5_Archive/` and `> **OBSOLETE — ...**` content as historical, not current.
