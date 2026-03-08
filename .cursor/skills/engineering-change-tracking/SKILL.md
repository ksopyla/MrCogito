---
name: engineering-change-tracking
description: Track Concept Encoder code changes and direction shifts. Use after refactors, architecture updates, training or evaluation script changes, bug fixes, implementation improvements, or when updating CHANGELOG.md, docs/1_Strategy_and_Plans/active_todos.md, commit messages, or architecture git tags.
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

Do not use this skill to log run metrics or benchmark results. Use `experiment-tracking` for training and evaluation records.

## Keep aligned
When a change matters, keep these artifacts consistent:
- `CHANGELOG.md`
- `docs/1_Strategy_and_Plans/active_todos.md`
- git commit message
- architecture tags such as `arch/{feature}` when justified

## After a meaningful code change
1. Update `CHANGELOG.md` with a dated entry.
2. Describe why the change happened and what it changes at a high level.
3. Mark related tasks in `docs/1_Strategy_and_Plans/active_todos.md`. If a commit already exists, include the commit hash.
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
**Related TODO:** `docs/1_Strategy_and_Plans/active_todos.md` -> "Task name"
```

## Related skills
- Use `experiment-tracking` when the same change also needs run metadata or benchmark results recorded.
- Use `research-methodology` when the change reflects a new hypothesis, evaluation plan, or research decision.
