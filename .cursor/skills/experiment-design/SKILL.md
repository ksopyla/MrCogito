---
name: experiment-design
description: Turn a research hypothesis into ONE minimal, well-scoped Concept Encoder experiment before any code is written. Use when the user wants to start, define, scope, or implement a new experiment, asks "what should we try next" with a concrete idea, or when an implementation request risks pulling in multiple research-agenda threads at once. Produces a frozen spec in docs/experiments_specs/ahead/<ID>.md that runs as args/config over the shared foundation; never a new training fork. Not for recording finished results (experiment-track), choosing direction from literature (research-synthesis), or implementing the nn modules (research-implement).
---

# Experiment Design

## Mission
Convert a hypothesis into well-scoped runnable experiment with a frozen spec
and a config — *before* writing code. 

The job is to **shrink scope** and **write down the spec** before writing code. 

## Hard rules (non-negotiable)
1. **One hypothesis, one changed variable.** Two changes → two specs.
2. **Spec before code.** No experiment code until
   `docs/experiments_specs/ahead/<ID>.md` exists and the user approves it.
3. **Configs over forks.** An experiment is args/config over the single shared training entrypoint. NEVER create a new `train_*.py` or `nn/concept_encoder_*.py` per experiment. New capability lands as a *reusable, config-selectable* foundation component (implement via `research-implement`).
4. **Numeric success AND kill criteria, set before running.** No open-ended runs.
5. **Builds-on is mandatory.** Name the foundation modules reused, the init/checkpoint, and the baseline run id + score to beat.
See `.cursor/rules/project-overview.mdc` and the hard rules above.

## Read first
1. `docs/1_Strategy_and_Plans/agenda.md` — the current focus (which may be exploratory / undecided) and what is already active. The new experiment should serve the guiding Vision and the current focus, and be a small, single-variable increment.
2. All lifecycle folders under `docs/experiments_specs/` plus `TEMPLATE.md`. Scan
   `ahead/`, `done_success/`, `done_failed/`, and `canceled/` before assigning an ID;
   IDs are globally unique and never reused.
3. `docs/2_Experiments_Registry/master_experiment_log.md` — the nearest fair baseline and the "what we've explored" learnings (don't silently repeat a setup that already underperformed without a materially new ingredient; say what's new).

## Think upfront on architecture and parameters, write down in the spec

1. How many layers do we need, for the encoder, the bottleneck, the reasoning, the decoder?
2. How many concepts? Should we match the previous experiment's concept count?
3. What is the token embedding dimension? Should we match the previous experiment's token embedding dimension? Should we use the asymmetry from the previous experiment?
4. What is the tokenizer? 
5. What dataset we should train on? What are the previous ones, should we reuse or user asks for a new one?
6. Where to train Polonez or Odra, or we should use the cloud provider?
7. How many steps or epochs we should train for?
8. What are the embedding dim, hidden size, concept dim?


## Workflow
```
- [ ] 1. Restate the hypothesis in one falsifiable sentence
- [ ] 2. Check it serves the guiding Vision / current focus and is a small, self-contained increment
- [ ] 3. Reduce to ONE changed variable vs a named baseline
- [ ] 4. Fill Builds-on: foundation modules + init/checkpoint + baseline id & score
- [ ] 5. Set numeric success + kill criteria
- [ ] 6. Assign the next free ID (E0NN_slug); write docs/experiments_specs/ahead/<ID>.md from TEMPLATE.md
- [ ] 7. Specify how it runs: exact command + env-var overrides on the shared bash launcher (list any new FOUNDATION component needed — reusable, not a fork)
- [ ] 8. Update agenda.md (set it as the Current focus); the lifecycle folders are the index — do not add a manual row
- [ ] 9. Get user go-ahead, THEN implement
```

## Scope-check questions (ask if the idea is too big)
- "What single number decides success, and at what threshold?"
- "Which existing run is the baseline this must beat?"
- "What is the ONE thing changing vs that baseline?"
- "Does this need new foundation code, or just a config?" If new code: is it reusable by future experiments, or a one-off fork? (Reusable only.)
- "If it fails, at what step/metric do we stop?"

## Handoffs
- Detailed repo-rooted plan (the HOW: modules, forward pass, data, loss, snippets) → `implementation-plan`; then build → `research-implement`.
- Idea-vs-literature / which family to pursue → `research-synthesis` (+ `research-scout`).
- After the run finishes → `experiment-track` (it writes results and moves the pair from `ahead/` to `done_success/` or `done_failed/`).
- Draft rejected or superseded before any run → `docs-hygiene` moves the spec/plan pair to `canceled/`.
- Pruning/archiving old specs → `docs-hygiene`.

## ID scheme
`E0NN_short_slug`, zero-padded, stable, never reused. The ID joins
spec ↔ launch command/run ↔ run report ↔ checkpoint ↔ WandB run.
