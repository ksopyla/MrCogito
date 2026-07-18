---
name: implementation-plan
description: Compile an approved experiment spec, the relevant research analysis, and the existing MrCogito codebase into a detailed, repo-rooted implementation plan that steers research-implement. Use after experiment-design has framed an experiment (and after research-explain / research-synthesis when a paper or repo is involved), before writing code — to decide which modules to reuse or extend, the forward pass with tensor shapes, the inputs and data preprocessing, the loss and training objective, config knobs, the launch command, tests, and risks, with optional code-sketch snippets. Preserves bold architectural bets; does not derisk them into safe retreads. Writes docs/experiments_specs/ahead/<ID>_plan.md. Not for framing the experiment (experiment-design), writing the code (research-implement), recording results (experiment-track), or judging a paper's fit (research-synthesis).
---

# Implementation Plan (research → buildable design)

The bridge between the **frame** and the **code**. `experiment-design` produces the spec
(WHAT / WHY: one coherent architectural bet, success/kill criteria); this skill produces the
**repo-rooted plan** (HOW); `research-implement` executes it. It is the research analog
of a PRD: it turns a synthesis verdict + an experiment frame + our actual modules into a
concrete, buildable design — so implementation is execution, not improvisation.

## Research stance
Preserve the **boldness of the spec**. Plan the novel mechanism faithfully. Do **not**
"derisk" a bold bet into a safe retread (plain cross-attn, pretrained SVD token init, drop
the proposed dynamics "for a first version"). If the bet is too vague to plan, hand back to
`experiment-design` — do not invent a weaker substitute. Infra reuse (config, registry,
collators, launchers) is good; substituting the architectural idea is not.

## When to use
- After `docs/experiments_specs/ahead/<ID>.md` exists and **before** `research-implement` writes code.
- When turning a paper / repo / idea into a concrete build: pull the mechanism from
  `research-explain` (forward pass, shapes, gradients) and the fit/decision from
  `research-synthesis` (Adopt/Adapt + tradeoffs), then ground it in our repo here.

## Read first (plan against reality, not assumptions)
1. The spec `docs/experiments_specs/ahead/<ID>.md` — hypothesis, architectural bet, builds-on, success/kill.
2. The `research-implement` skill — the codebase map, the encode→reason→decode patterns, and the reproducibility / unparking rules. The plan must reuse what already exists for *infra*; new mechanism code is expected when the bet requires it.
3. Research analysis (if external): the `research-explain` walkthrough and/or `research-synthesis` verdict; matching `docs/literature_review/` entries; the actual reference repo/paper.
4. The **real** classes you will touch in `nn/`, `training/`, `data/`, `evaluation/` — open them. Never plan against an imagined API; cite the paths/classes you read.

## Hard rules
1. **Root every decision in real repo paths/classes.** Cite them. If you assume an API, you must have read it.
2. **Reuse infra; invent mechanism when the bet needs it.** Extend existing modules; new code only as a *reusable, config-selectable* component. Never a per-experiment fork. Do not reuse old architectural defaults just because they are familiar.
3. **Scope = the spec's coherent bet.** Plan exactly that; push *post-hoc* ablations to follow-up specs — do not replace the bet with an ablation.
4. **Preserve invariants:** O(C·N) preference (no accidental O(N²) decoder self-attention unless the spec explicitly challenges that tradeoff), backward-compatible config defaults, the checkpoint eval contract, old checkpoints still loadable.
5. **Runnable as config + launcher** (env-var overrides), not a new script.
6. **Snippets are sketches**, not demos — interface signatures, tensor shapes, config fields that pin a decision. Label them `# sketch`.

## Workflow
```
- [ ] 1. Restate the spec's hypothesis + architectural bet (1–2 lines); confirm you are not watering it down
- [ ] 2. Source & fit: origin (prior result / paper / vision / cross-domain analogy) + research-synthesis verdict + mapping onto encode / reason / decode / loss / data
- [ ] 3. Reuse map: read the modules; list reuse-as-is (infra), extend, and any NEW reusable component (and where it lives)
- [ ] 4. Forward pass with shapes: trace encode → (reason) → decode with concrete tensor shapes grounded in our modules (define B, N, C, H, V once)
- [ ] 5. Inputs & data: dataset, collator (which / what changes), preprocessing, masking/splitting
- [ ] 6. Loss & objective: which loss_manager components / new loss (register_loss), training objective, loss weighting
- [ ] 7. Config & launch: new ConceptEncoderConfig/LossConfig fields (backward-compatible defaults) + MODEL_REGISTRY entry + exact launcher command/env overrides
- [ ] 8. Tests & smoke: unit tests to add, local MPS smoke command, what to assert (shapes, loss finite, collapse check via run_concept_analysis)
- [ ] 9. Risks & tradeoffs: failure modes, cheapest early kill signal (tie to the spec's success/kill) — not a substitute safe architecture
- [ ] 10. Write docs/experiments_specs/ahead/<ID>_plan.md from PLAN_TEMPLATE.md; link it from the spec; get user go-ahead → research-implement
```

## Output
`docs/experiments_specs/ahead/<ID>_plan.md` from [../../../docs/experiments_specs/PLAN_TEMPLATE.md](../../../docs/experiments_specs/PLAN_TEMPLATE.md). Keep it concrete and tight — an engineering design doc, not an essay. The spec stays the source of truth for *intent*; the plan is the source of truth for *design*; both are joined by `<ID>`.

## Handoffs
- Paper/repo not yet understood → `research-explain` (+ `research-scout`). Fit/decision unclear → `research-synthesis`.
- Frame missing, too timid, or too vague → `experiment-design`.
- Build it → `research-implement` (reads the spec **and** this plan).
- Results, after the run → `experiment-track`.
