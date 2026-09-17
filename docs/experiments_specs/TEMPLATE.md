# <ID> — <short title>

- **Status:** draft | approved | active | done | killed | canceled
- **Serves:** <the Vision / agenda focus this architectural bet serves>
- **Implementation plan:** [<ID>_plan.md](<ID>_plan.md) *(authored by `implementation-plan`; the HOW)*
- **Owner / dates:** <name> · opened YYYY-MM-DD · closed YYYY-MM-DD

> One experiment = one **coherent architectural hypothesis**. A bold composition
> (encode + reason + decode under one claim) can be a single bet — do not split one
> idea into micro-A/Bs. Unrelated changes → separate specs. Implementation is
> args/config over the shared entrypoint (env-var overrides on the bash launcher),
> never a new fork. Prefer material novelty over safe retreads; A/B and ablations
> belong *after* something works. The spec is **frozen once a run starts**; results
> live in the registry and run report, not here. New specs live in `ahead/`. When
> closed, move the spec and plan together to `done_success/`, `done_failed/`, or
> `canceled/` according to the lifecycle rules in README.
>
> **ID scheme:** do not mint this file unless the user asked to design or run.
> The family is `E0NN` (E18, E21, …); a new family number only if the user asked
> for a new family or a different encode/reason/decode object. A small detail
> inside the current family is a flavour: `E{NN}a`, `E{NN}b`, `E{NN}c`, …
> Sequential lowercase letters on **that** family's number; never reuse a letter
> inside a family. The unlettered ID is the parent; the first small change is
> `a` (E21 → `E21a`, not a new E26; E18 → `E18a`). Same compute / memory /
> parameters as the parent unless the user changes scale. A new dataset, probe,
> eval protocol, length ladder, dense control, or optimizer/LR knob is a *run*
> of the current family or flavour ID, not a new ID. When in doubt: flavour
> letter of the current family. Full rules: [README.md](README.md#id-scheme).

## Hypothesis
<One falsifiable sentence: "If we do X, then metric Y will move to Z, because ..." —
bold claim with a reason (mechanism / analogy), not a knob tweak.>

## Builds-on
- **Foundation:** <which existing modules + the single config-driven entrypoint this reuses; NO new fork>
- **Init / checkpoint:** <e.g. random init | prior run id | pretrained — justify; do not default to SVD/pretrained token warm-start out of caution>
- **Baseline to beat:** <named run id + the metric value it scored>
- **Materially new:** <what is novel vs that baseline and vs recent specs — not a cosmetic retune>

## The architectural bet
<The coherent change vs the baseline: mechanism, inductive bias, or objective.
Everything needed for that one claim is in scope; post-hoc ablations are follow-ups.>

## Why this is not a safe retread
<1–3 lines: how this differs from prior ledger ideas (e.g. not "cross-attn without FFN",
not "small embeddings from pretrained via SVD", not optimizer-only A/B). Optional:
maths / physics / biology / info-theory analogy.>

## Success criteria (set BEFORE running)
- <numeric, e.g. "validation perplexity < X on held-out WikiText-103">

## Kill criteria (set BEFORE running)
- <numeric / time, e.g. "if eval loss not below <X> by step <N>, stop">

## Plan
- **Data:** <dataset + size>
- **Compute:** <machine, GPUs, est. GPU-hours>
- **Steps / epochs:** <budget>
- **Launch:** `<exact command + env-var overrides on the shared bash launcher, e.g. HIDDEN_SIZE=768 bash scripts/train_concept_pretraining_multigpu.sh>`
- **New foundation code (if any):** <reusable module added via research-implement, or "none — config only">

## Result
<Filled in AFTER, by experiment-track. Link out; do not paste full results here.>
- Run id: `<run_id>`
- WandB: <link>
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: promising | mixed | regression | killed — <one line>
