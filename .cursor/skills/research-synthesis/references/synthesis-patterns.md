# Synthesis Patterns

## Contents

- Verdict heuristics: Adopt / Adapt / Watch / Reject
- Connection patterns across sources
- Common pitfalls
- Smallest-test principle
- Roadmap and vision synthesis

## Verdict Heuristics

Use these as anchors when assigning a verdict to a paper or repo. Bias rule: when in doubt, prefer Adapt over Adopt and Watch over Reject — premature adoption is more expensive than a labeled "watch" item.

### Adopt

- Directly addresses an active weakness, with a clear local test path.
- Compatible with the concept bottleneck (no information leakage around it).
- Implementation cost is small relative to the expected gain.
- Evidence is convincing on at least one diagnostic the project already trusts.

### Adapt

- Mechanism is promising but built for a different setting (autoregressive only, fixed-length latent, vision-only, JAX-only, etc.).
- The smallest version of the idea can be ported into our architecture.
- Be explicit about which assumptions are being dropped or substituted.

### Watch

- Important trend or strong result, but not actionable now.
- Note the trigger condition: what would have to change in our system or the evidence base to move the item to Adopt or Adapt.

### Reject

- Incompatible with the bottleneck (relies on full-sequence self-attention or memory access we will not have).
- Result depends on scale or data we cannot reach.
- Reproduction attempts already exist showing the result does not transfer outside the original setting.
- Hype-only: no working implementation and no convincing ablations.

## Connection Patterns

### Convergence

Multiple independent papers point at the same mechanism. Treat this as stronger evidence than any single paper, even if individual papers are weaker.

Action: surface the convergent claim explicitly, then pick the cleanest implementation among the papers as the porting target.

### Tension

Two strong papers with incompatible assumptions or opposite results. Do not paper over the tension.

Action: identify the assumption that splits them, decide which side fits MrCogito's invariants, and justify the choice.

### Composition

Independent techniques that can stack. Common stackable axes: training objective, optimizer or regularization, attention pattern, decoder family, data curation.

Action: pick at most one new axis at a time. Compose only after each part has been validated in isolation.

### Lineage

A new SoTA paper is a refinement of an older idea. The older paper often clarifies which design choices are essential vs. incidental.

Action: read at least one ancestor paper before adopting a modern variant, especially if the modern variant relies on a non-obvious detail.

## Common Pitfalls

- **Hype anchoring**: judging a paper by venue, citation count, or social media volume instead of evidence quality.
- **Benchmark overfitting**: assuming a result on one benchmark transfers to ours without checking that the diagnostic overlaps.
- **Scale-dependent claims**: methods that only work above a parameter or data scale we will not reach. State this explicitly when present.
- **Single-paper extrapolation**: building a multi-experiment plan on one preprint with no replication.
- **Framework barrier**: treating "no PyTorch implementation" as a non-issue. Estimate porting cost honestly when only JAX or TensorFlow code exists.
- **Loss of bottleneck**: silently allowing skip paths around the concept tokens to "make it work" — this is a category error, not a tweak.
- **Ignoring older work**: assuming pre-2023 methods cannot be relevant. Many failed only on compute or data, not on correctness.

## Smallest-Test Principle

Before recommending a full training run, propose a smaller test that uses what the project already has:

- Reuse an existing architecture variant: `weighted_mlm`, `perceiver_mlm`, `recursive_mlm`, `diffusion_mlm`.
- Reuse an existing diagnostic (effective rank, pairwise similarity, STS-B zero-shot, ViaDecoder GLUE, prefix or suffix loss) to pre-empt a costly run.
- Use a smaller dataset (e.g. Minipile or a subset) for fast turnaround.
- Inspect an intermediate checkpoint of an existing run before committing to a new one.

If the smallest test cannot be specified concretely, the recommendation is not yet ready.

## Roadmap and Vision Synthesis

When the user wants to evolve `docs/1_Strategy_and_Plans/vision_and_goals.md` or `docs/1_Strategy_and_Plans/roadmap.md` based on external evidence:

- Anchor proposed edits to architecture invariants, not to recent paper hype.
- Quote the evidence that justifies each change, with a citation.
- Prefer minimal, dated edits over rewrites; keep traceability of why the plan changed.
- Distinguish strategic shifts (vision-level) from tactical shifts (next-experiment-level).
- Flag what the change would cost: experiments to redo, benchmarks to re-run, modules to refactor.
