# Synthesis Patterns

## Contents

- Verdict heuristics: Adopt / Adapt / Watch / Reject
- Connection patterns across sources
- Common pitfalls
- Falsifiable-bet principle
- Roadmap and vision synthesis

## Verdict Heuristics

Use these as anchors when assigning a verdict to a paper or repo. Bias rule: when in doubt,
prefer **Adapt** (compose a novel local mechanism) over wholesale Adopt of someone else's
stack, and Watch over Reject — but do **not** prefer timid micro-A/Bs over a coherent bold
bet. Premature full-stack cloning is expensive; so is a research diet of safe retreads.

### Adopt

- Directly addresses an active weakness, with a clear falsifiable claim and kill signal.
- Compatible with the concept bottleneck (no information leakage around it).
- The mechanism is novel enough for MrCogito that shipping it moves the architecture forward
  (not just "we can A/B their hyperparameter").
- Evidence is convincing on at least one diagnostic the project already trusts.

### Adapt

- Mechanism is promising but built for a different setting (autoregressive only, fixed-length
  latent, vision-only, JAX-only, etc.) — or a cross-domain analogy needs translation.
- Port the *inductive bias*, not the nearest safe block we already have.
- Be explicit about which assumptions are being dropped or substituted.
- Prefer a coherent compositional Adapt over stripping the idea down to an optimizer/LR tweak.

### Watch

- Important trend or strong result, but not actionable now.
- Note the trigger condition: what would have to change in our system or the evidence base to move the item to Adopt or Adapt.

### Reject

- Incompatible with the bottleneck (relies on full-sequence self-attention or memory access we will not have) *unless* the recommendation is an explicit, justified challenge to that invariant.
- Result depends on scale or data we cannot reach.
- Reproduction attempts already exist showing the result does not transfer outside the original setting.
- Hype-only: no working implementation and no convincing evidence.
- Safe retread: the idea collapses to something already in the ledger (same cross-attn recipe,
  same SVD/pretrained token init, same asymmetry tweak) with no materially new ingredient.

## Connection Patterns

### Convergence

Multiple independent papers point at the same mechanism. Treat this as stronger evidence than any single paper, even if individual papers are weaker.

Action: surface the convergent claim explicitly, then pick the cleanest implementation among the papers as the porting target — or compose the shared bias into a *new* local design.

### Tension

Two strong papers with incompatible assumptions or opposite results. Do not paper over the tension.

Action: identify the assumption that splits them, decide which side fits MrCogito's invariants (or propose a third path), and justify the choice.

### Composition

Independent techniques that can stack. Common stackable axes: training objective, optimizer or regularization, attention pattern, decoder family, data curation, dynamical / physical analogies.

Action: for **novel architecture search**, composing a coherent multi-axis bet under one hypothesis is encouraged when the pieces serve one claim. Save "one axis at a time" ablations for *after* a positive signal — that is when A/B is worth doing.

### Lineage

A new SoTA paper is a refinement of an older idea. The older paper often clarifies which design choices are essential vs. incidental.

Action: read at least one ancestor paper before adopting a modern variant, especially if the modern variant relies on a non-obvious detail.

### Cross-domain analogy

Maths, physics, biology, dynamical systems, information theory, or neuroscience suggests a mechanism (attractors, compression bounds, sparse coding, Hopfield-like retrieval, renormalization, etc.).

Action: make the analogy explicit, state what maps to tokens / concepts / reasoner / decoder, and name what would falsify the analogy. Do not force a paper citation when the bet is theory-motivated — label it as a working hypothesis.

## Common Pitfalls

- **Hype anchoring**: judging a paper by venue, citation count, or social media volume instead of evidence quality.
- **Benchmark overfitting**: assuming a result on one benchmark transfers to ours without checking that the diagnostic overlaps.
- **Scale-dependent claims**: methods that only work above a parameter or data scale we will not reach. State this explicitly when present.
- **Single-paper extrapolation**: building a multi-experiment plan on one preprint with no replication.
- **Framework barrier**: treating "no PyTorch implementation" as a non-issue. Estimate porting cost honestly when only JAX or TensorFlow code exists.
- **Loss of bottleneck**: silently allowing skip paths around the concept tokens to "make it work" — this is a category error, not a tweak.
- **Ignoring older work**: assuming pre-2023 methods cannot be relevant. Many failed only on compute or data, not on correctness.
- **Safe-bet collapse**: rewriting a bold Adapt into "cross-attn without FFN", "SVD init from pretrained embeddings", or another pattern already tried in the ledger.
- **A/B as research substitute**: proposing optimizer / width / LR sweeps when the question is architectural.

## Falsifiable-Bet Principle

Before recommending the next run, name a **coherent architectural bet**:

- What is the novel claim? What would be surprising if it worked?
- How do papers and/or cross-domain analogies connect into that claim?
- What numeric success and kill criteria stop a bad bet early?
- Which existing diagnostics (effective rank, pairwise similarity, STS-B zero-shot, ViaDecoder
  GLUE, prefix/suffix loss, generation coherence) give an early falsification signal?
- A cheap subset / smoke run is for *killing* bad bets fast — not for replacing the bet with
  a safer retread of `weighted_mlm` / `perceiver_*` / parked families.

If the falsifiable claim and kill signal cannot be specified concretely, the recommendation is not yet ready.

A/B and ablations: worth doing from time to time **when something is working**, to understand
which ingredient mattered. They are not the default output of synthesis.

## Roadmap and Vision Synthesis

When the user wants to evolve `docs/1_Strategy_and_Plans/vision_and_goals.md` or `docs/1_Strategy_and_Plans/agenda.md` based on external evidence:

- Anchor proposed edits to architecture invariants *and* to the bold-research stance, not to recent paper hype alone.
- Quote the evidence that justifies each change, with a citation.
- Prefer minimal, dated edits over rewrites; keep traceability of why the plan changed.
- Distinguish strategic shifts (vision-level) from tactical shifts (next-experiment-level).
- Flag what the change would cost: experiments to redo, benchmarks to re-run, modules to refactor.
