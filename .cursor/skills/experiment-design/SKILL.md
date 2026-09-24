---
name: experiment-design
description: Turn a research hypothesis into ONE bold, falsifiable Concept Encoder experiment before any code is written. Use when the user wants to start, define, scope, or implement a new experiment, asks "what should we try next" with a concrete idea, or when an implementation request risks pulling in multiple research-agenda threads at once. Produces a frozen spec in docs/experiments_specs/ahead/<ID>.md that runs as args/config over the shared foundation; never a new training fork. Prefers novel architectural bets over tiny A/B knobs. Not for recording finished results (experiment-track), choosing direction from literature (research-synthesis), or implementing the nn modules (research-implement).
---

# Experiment Design

## Mission
Convert a **bold, falsifiable architectural hypothesis** into a runnable experiment with a
frozen spec and a config — *before* writing code.

This project is hunting for a **novel architecture**, not optimizing a known recipe. The job
is to **write down a coherent bet** (and make it runnable), not to shrink every idea into the
safest micro-A/B. Discipline (spec, kill criteria, configs-over-forks) is mandatory;
caution-as-default is not.

## Research stance (read with `project-overview.mdc`)
- Prefer material novelty: a new mechanism, inductive bias, or objective — something that
  would be surprising if it worked.
- Connect ideas across papers and across maths / physics / biology / dynamical systems /
  information theory / neuroscience. Good analogies beat nearest-neighbor citations.
- A/B and ablations are for *after* something works, to understand why — not the default
  experiment shape.
- Do **not** re-propose old safe patterns as "new" experiments (e.g. cross-attention without
  FFN; small token embeddings warm-started from pretrained via SVD/PCA; retuning the same
  asymmetry / depth stack we already ran). If the ledger already tried it, say so and add a
  *materially* new ingredient — or pick a different bet.
- Reject specs that only change a knob (optimizer, LR, width) unless the user explicitly
  asked for that ablation.

## Hard rules (non-negotiable)
1. **One coherent hypothesis per experiment.** A bold architectural composition can be one
   idea (encode+reason+decode redesign under a single claim). Do **not** split one bet into
   five micro-A/Bs. Unrelated changes → separate specs.
2. **Spec before code.** No experiment code until
   `docs/experiments_specs/ahead/<ID>.md` exists and the user approves it.
3. **Configs over forks.** An experiment is args/config over the single shared training
   entrypoint. NEVER create a new `train_*.py` or `nn/concept_encoder_*.py` per experiment.
   New capability lands as a *reusable, config-selectable* foundation component (implement
   via `research-implement`).
4. **Numeric success AND kill criteria, set before running.** No open-ended runs. Killing a
   bold idea early is success, not failure of process. **Every absolute-metric gate needs a
   measured ceiling, not an aspiration** — name the control that shows the number is reachable
   at this scale, data and budget, and set the gate as a fraction of it. E22's S1 asked the
   concept array for 0.30 nats of far-context value when E18's own reach ablation had already
   measured a *dense* transformer extracting 0.035–0.10 nats from the same distances at the same
   scale: the gate was ~3× above the physical ceiling, so no architecture could have passed
   (see `docs/4_Research_Notes/e22_root_cause_20260912.md` §7). Gates with a demonstrated
   ceiling look like E18b's ("dense control reaches 99.3% on this task, so ask for 50%").
5. **Builds-on is mandatory.** Name the foundation modules reused, the init/checkpoint, and
   the baseline run id + score to beat — and state **what is materially new** vs that baseline
   (not just a retune).
See `.cursor/rules/project-overview.mdc` and the hard rules above.

## Read first
1. `docs/1_Strategy_and_Plans/agenda.md` — current focus and what is already active. The new
   experiment should serve the guiding Vision; it should be a **self-contained architectural
   increment**, not necessarily a tiny single-knob delta.
2. All lifecycle folders under `docs/experiments_specs/` plus `TEMPLATE.md`. Scan
   `ahead/`, `done_success/`, `done_failed/`, and `canceled/` before assigning an ID;
   IDs are globally unique and never reused. Flag if the draft repeats a prior idea.
3. `docs/2_Experiments_Registry/master_experiment_log.md` — nearest fair baseline and
   "what we've explored". Do not silently repeat a setup that already underperformed without
   a materially new ingredient; say what's new.

## Think upfront on architecture and parameters, write down in the spec

Ask whether the design is a **new inductive bias**, not only whether knobs match the last run:

1. What is the architectural claim? Which of encode / reason / decode / objective / data
   inductive bias is genuinely new?
2. What analogy (maths / physics / biology / information theory / …) motivates the mechanism?
3. How many layers for encoder, bottleneck, reasoning, decoder — and **why that structure**,
   not "match the previous experiment"?
4. How many concepts? Fixed vs length-scaling? Why?
5. Token / concept dims: match prior only when the comparison needs it; do not default to
   proven asymmetry or pretrained-SVD init just because it is familiar.
6. Tokenizer, dataset, train location (Polonez / Odra / cloud), steps/epochs, hidden size —
   choose what the hypothesis needs, document tradeoffs.
7. What would make this experiment **obviously different** from the last three specs in
   `docs/experiments_specs/`?

## Workflow
```
- [ ] 1. Restate the hypothesis in one falsifiable sentence (bold claim, not a knob tweak)
- [ ] 2. Check it serves the Vision / current focus AND is materially novel vs the ledger
- [ ] 3. Frame ONE coherent architectural bet (composition OK if it is one idea); name what
      is NOT in scope (follow-up ablations after a positive signal)
- [ ] 4. Fill Builds-on: foundation modules + init/checkpoint + baseline id & score + delta
- [ ] 5. Set numeric success + kill criteria (aggressive enough to stop a bad bet early);
      for a new architecture, state them on the **capability suite** where possible (target
      frontier level, cells that must beat a recorded reference; see
      `docs/engineering_specs/capability_suite.md`) so the result is comparable;
      for each absolute threshold, cite the control that measured its ceiling — or schedule the
      cheap measurement as a pre-flight step in the Plan
- [ ] 6. Assign the ID (default: next unused flavour letter of the current family,
      not a new family); write docs/experiments_specs/ahead/<ID>.md from
      TEMPLATE.md. If the user did not ask to design or run an experiment,
      write nothing in `ahead/`
- [ ] 7. Specify how it runs: exact command + env-var overrides on the shared bash launcher
      (list any new FOUNDATION component — reusable, not a fork)
- [ ] 8. Update agenda.md (set it as the Current focus); the lifecycle folders are the index
- [ ] 9. **Alignment page (last step before go-ahead):** build an interactive HTML
      information-flow page for the bet (see "Alignment page" below), open it for the author,
      and wait for their confirmation or changes. Fold the changes back into the spec
- [ ] 10. Get user go-ahead, THEN implement
```

## Scope-check questions
- "Did the user ask to design this?" and "flavour of the current family vs new family they named?"
- "Is this a novel architectural bet, or a safe retune of something we already ran?"
- "What single number decides success, and at what threshold?"
- "Which existing run is the baseline this must beat — and what is *materially* new?"
- "If this is an A/B or ablation: did something already work that we are now dissecting,
  or are we using A/B as a substitute for a real idea?"
- "Does this need new foundation code, or just a config?" If new code: is it reusable by
  future experiments, or a one-off fork? (Reusable only.)
- "If it fails, at what step/metric do we stop?"
- "What analogy or theory motivates this — or are we only copying a paper's default block?"

## Alignment page (the last step of every design)
Specs are read once and misread easily; E30's build drifted from its idea note (static
averages instead of addressed latents, a 16-token causal writer) and nobody noticed until
after the runs. Before asking for the go-ahead, make the architecture *visible*, so the
author can check that the spec says what they mean.

- **File:** `docs/3_Evaluations_and_Baselines/<id>_architecture.html` (lowercase ID), one
  self-contained page (inline SVG + vanilla JS, no CDN), light/dark via CSS variables.
  Precedents: `e18_e21_architecture.html`, `e30_architecture.html`, `e31_architecture.html`.
- **Must show:**
  1. **The pipeline, stepwise**: every stage from token ids to the loss as a clickable node
     with prev/next stepping; per stage: what happens in plain words, tensor shapes at the
     claim scale, and *what information survives* this stage.
  2. **Toggles** for each arm of the bet **and** for the baseline being replaced ("as
     built"), so every difference is one click away. Mark each stage *new / unchanged /
     proposed / baseline*.
  3. **A worked toy example** (small sequence) showing where information goes: which
     tokens each component reads or picks. Label hand-built weights as illustrative.
  4. **An information budget** along the path (bits available at each stage vs what the
     answer needs), to show where the bottleneck is expected.
  5. **The masks**: who may attend to what (causality, exclusive cuts, leaks prevented).
  6. **An alignment checklist**: every design decision in the spec as a row with
     agree / change / question + a note field, and a button that collects the answers
     into text the author pastes back into the chat.
- **Check it renders** (open it in the browser pane, screenshot each section, no console
  errors) before showing it.
- **Do not implement** until the author has returned the checklist or said "aligned".
  Changes they request go into the spec (and the page, if it is kept as the reference).

## Pitching the experiment to the author
The spec is the archive; the chat message is a ~150-word pitch: the idea, the bet, how we will know it
failed, the cost, and any decision needed (≤2 options plus a recommendation). Name the arms by what they
are ("the control with no memory"), never by letters, and do not list gate codes in chat. Follow
`research-comms`; add any new metric, probe or arm name to `docs/glossary.md` in the same commit.

## Handoffs
- Detailed repo-rooted plan (the HOW: modules, forward pass, data, loss, snippets) →
  `implementation-plan`; then build → `research-implement`.
- Idea-vs-literature / which family to pursue → `research-synthesis` (+ `research-scout`).
- After the run finishes → `experiment-track` (it writes results and moves the pair from
  `ahead/` to `done_success/` or `done_failed/`).
- Draft rejected or superseded before any run → `docs-hygiene` moves the spec/plan pair to
  `canceled/`.
- Pruning/archiving old specs → `docs-hygiene`.

## ID scheme
The ID joins spec ↔ launch command/run ↔ run report ↔ checkpoint ↔ WandB run.
Never reuse an ID found in any lifecycle folder.

**When to design at all.** Do not create a spec in `ahead/` unless the user asked
to design or run an experiment.

**Family `E0NN`.** The family is the integer ID (E18, E21, …). Assign a new
family number only if the user asked for a new experiment family, or approved a
bet that is a different encode/reason/decode object. Zero-padded, globally
unique, never reused. Do not take the next free integer just because it is
free. E26–E29 were flavours/runs of E21 that were mis-numbered; do not repeat
that.

**Flavours (default for a small detail).** A small architecture change inside
the current family — extra loss head, identity keys, a loop, a mask — is a
flavour of **that** family: `E{NN}a`, `E{NN}b`, `E{NN}c`, … Sequential
lowercase letters on the current family's number. Never reuse a letter inside
a family. The unlettered ID is the parent; the first small change is `a`
(E21 → `E21a`, not a new E26; E18 → `E18a`). Same order of compute / memory /
parameters as the parent unless the user changes scale. Spec path:
`docs/experiments_specs/ahead/E{NN}a_….md`.

**Not a flavour / not a new family.** A new dataset, probe, eval protocol,
length ladder, dense control, or optimizer/LR knob is a *run* of the current
family or flavour ID, logged under that ID — not a new spec ID.

**When in doubt.** Small delta on the current family → next unused flavour
letter of that family. User did not ask → write nothing in `ahead/`.
