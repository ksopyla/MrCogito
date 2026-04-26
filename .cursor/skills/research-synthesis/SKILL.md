---
name: research-synthesis
description: Synthesize external AI/ML research into decisions and direction for the Concept reasoning model project. Use this skill whenever the user asks how a paper, technique, repo, or trend relates to MrCogito, what to take from a paper, how to connect ideas across papers, what experiments a finding implies, whether to implement a method, how external work compares with current architecture, or wants to evolve the roadmap, vision, or research direction based on external evidence — even when they do not explicitly say "synthesize". Pair with the `research-scout` agent which gathers source material; this skill turns that material into project-grounded recommendations and concrete next steps.
---

# Research Synthesis for the Concept Reasoning Model

Use this skill to translate external research into decisions for MrCogito's "Concept reasoning model": a concept bottleneck that compresses long token sequences into a small set of semantic concept tokens and reasons over them, text-first with an audio extension.

When fresh external evidence is needed (latest papers, GitHub repos, model cards, benchmarks), spawn the `research-scout` agent and feed its findings into this workflow. This skill is about analysis and synthesis, not search.

## Synthesis Workflow

Copy this checklist and check items off as you progress. The steps are a guide, not a straitjacket: skip what does not apply.

```
Synthesis Progress:
- [ ] Step 1: Capture the question and pick a synthesis mode
- [ ] Step 2: Read project context (vision, roadmap, relevant nn/ files)
- [ ] Step 3: Collect source material (delegate to research-scout when needed)
- [ ] Step 4: Extract thesis, technique, evidence, and code per source
- [ ] Step 5: Map each idea onto the Concept reasoning model
- [ ] Step 6: Connect ideas across sources
- [ ] Step 7: Recommend an action with the smallest local test
- [ ] Step 8: Cite everything; surface what is uncertain
```

### Step 1: Pick a synthesis mode

- **Single-paper deep dive**: extract a paper's method, evidence, and code, then translate it to our architecture.
- **Cross-paper connection**: combine ideas from multiple sources into a coherent design, hypothesis, or research direction.
- **Trend → application**: turn a current SoTA direction into one or two concrete experiments for MrCogito.
- **Roadmap or vision update**: feed external evidence into proposed edits to `docs/1_Strategy_and_Plans/vision_and_goals.md` or `docs/1_Strategy_and_Plans/roadmap.md` so the project's plan stays grounded.

### Step 2: Read project context

Before claiming relevance, read what is currently true about the project:

- `.cursor/rules/project-overview.mdc`
- `docs/1_Strategy_and_Plans/vision_and_goals.md`
- `docs/1_Strategy_and_Plans/roadmap.md`
- The `nn/`, `training/`, or `evaluation/` files that the question actually touches

The roadmap evolves. Anchor recommendations to architecture invariants and currently observed gaps, not to phase numbers or sub-goal labels. Stable invariants are listed in [references/concept-architecture-invariants.md](references/concept-architecture-invariants.md).

### Step 3: Collect source material

Prefer source material the user already provided. When fresh evidence is needed, spawn the `research-scout` agent with a focused brief, for example:

> Look up [topic] across arXiv, OpenReview, ACL Anthology, PMLR, Hugging Face Papers, and GitHub. Return paper thesis, architecture/objective, evidence, official PyTorch code, and notable limitations.

When `research-scout` returns notes, do not stop there — Steps 4–8 are this skill's responsibility.

### Step 4: Extract per source

For each source you intend to use, fill in the synthesis template (below). Keep entries tight. Skip fields that do not apply.

### Step 5: Map onto the Concept reasoning model

For each idea, answer:

- Which part of our system does it touch: encoder, concept bottleneck, decoder/reasoning core, training objective, evaluation, or data?
- Which architecture invariant would change if we adopted it? See `references/concept-architecture-invariants.md`.
- Which currently observed weakness would it plausibly improve, and what evidence would prove that?

### Step 6: Connect ideas across sources

Look for these patterns; details and pitfalls are in [references/synthesis-patterns.md](references/synthesis-patterns.md):

- **Convergence**: multiple independent papers point at the same mechanism. Stronger evidence than any single paper.
- **Tension**: papers with incompatible assumptions or opposite results. Pick a side and justify it.
- **Composition**: independent techniques that stack along different axes (objective, architecture, data).
- **Lineage**: a new SoTA paper refines an older idea; the older paper often clarifies what is essential.

### Step 7: Recommend with the smallest local test

For every recommendation, propose the smallest local test that would falsify or support the idea, using diagnostics already in the project (effective rank, pairwise concept similarity, STS-B, paraphrase or NLI tasks, prefix generation loss, suffix perplexity, generation coherence). Do not jump to full-scale training as a first step.

If you cannot specify a smallest test, the recommendation is not concrete enough yet.

### Step 8: Cite and flag uncertainty

Every external claim must include a URL: arXiv ID, OpenReview page, ACL Anthology page, proceedings page, Hugging Face page, or GitHub repo. If a claim has no source, label it as a working hypothesis. Be explicit about replication status, scale dependence, and benchmark overlap with our diagnostics.

## Paper Synthesis Template

Use this structure per paper that matters. Keep it tight; skip fields that do not apply.

```markdown
**Title** (venue/year, link)
- Thesis: one-sentence claim.
- Technique: architecture, objective, training recipe, inference procedure.
- Evidence: benchmarks, scale, ablation strength; what is convincing or weak.
- Code: official repo + framework, or "no PyTorch implementation found".
- MrCogito mapping: which part of our system it touches and what would change.
- Smallest test: a local experiment or module modification that would prove value.
- Risk: why it might not transfer here.
- Verdict: Adopt / Adapt / Watch / Reject, with one-line justification.
```

`references/synthesis-patterns.md` defines Adopt / Adapt / Watch / Reject in detail.

## Output Patterns

Match the output to the question:

- "Summarize this paper for our project." → one synthesis-template entry plus mapping and verdict.
- "What are the current trends in X?" → 3–7 trend bullets with citations, then a single-paragraph "what matters for MrCogito".
- "Should we implement Y?" → architecture mapping, smallest local test, risks, and a clear Adopt / Adapt / Watch / Reject.
- "Compare X with our `recursive_mlm` / `perceiver_mlm` / `diffusion_mlm` / `weighted_mlm`." → a small comparison table on attention pattern, objective, normalization, and conditioning, then a delta paragraph.
- "Update the roadmap or vision." → propose minimal, dated edits to `docs/1_Strategy_and_Plans/*` referencing the synthesized evidence.

## When to Hand Off

- Need fresh papers, repos, or model cards → spawn the `research-scout` agent.
- Need a deep walkthrough of a paper or repo (architecture, forward pass with shapes, training, gradient flow, decoding) before judging fit → use the `research-explain` skill, then return here for the verdict and smallest local test.
- Need to record a concrete training plan or run → use the `experiment-tracking` skill.
- Need to ship a code change driven by this synthesis → use the `engineering-change-tracking` skill once the change is implemented.

## Reference Files

- [references/concept-architecture-invariants.md](references/concept-architecture-invariants.md) — what stays true across roadmap changes; the substrate for relevance judgments.
- [references/synthesis-patterns.md](references/synthesis-patterns.md) — connection patterns, Adopt/Adapt/Watch/Reject heuristics, common pitfalls, and the smallest-test principle.
