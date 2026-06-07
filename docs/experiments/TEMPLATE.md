# E0NN — <short title>

- **Status:** draft | active | done | killed
- **Serves:** <the focus / direction in agenda.md this small increment serves>
- **Implementation plan:** [E0NN_..._plan.md](E0NN_..._plan.md) *(authored by `implementation-plan`; the HOW)*
- **Owner / dates:** <name> · opened YYYY-MM-DD · closed YYYY-MM-DD

> One experiment = one hypothesis = one changed variable. If you need to change
> two things, split it into two specs. Implementation is args/config over the shared
> entrypoint (env-var overrides on the bash launcher), never a new fork. The spec is
> **frozen once a run starts**; results live in the registry and run report, not here.

## Hypothesis
<One falsifiable sentence: "If we do X, then metric Y will move to Z, because ...">

## Builds-on
- **Foundation:** <which existing modules + the single config-driven entrypoint this reuses; NO new fork>
- **Init / checkpoint:** <e.g. SmolLM2-135M warm-start | random init | prior run id>
- **Baseline to beat:** <named run id + the metric value it scored>

## The single change
<The ONE variable changed vs the baseline. Everything else is held fixed.>

## Success criteria (set BEFORE running)
- <numeric, e.g. "validation perplexity < X on held-out WikiText-103">

## Kill criteria (set BEFORE running)
- <numeric / time, e.g. "if eval loss not below <X> by step <N>, stop">

## Plan
- **Data:** <dataset + size>
- **Compute:** <machine, GPUs, est. GPU-hours>
- **Steps / epochs:** <budget>
- **Launch:** `<exact command + env-var overrides on the shared bash launcher, e.g. HIDDEN_SIZE=768 bash scripts/train_perceiver_denoise_multigpu.sh>`
- **New foundation code (if any):** <reusable module added via research-implement, or "none — config only">

## Result
<Filled in AFTER, by experiment-track. Link out; do not paste full results here.>
- Run id: `<run_id>`
- WandB: <link>
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: promising | mixed | regression | killed — <one line>
