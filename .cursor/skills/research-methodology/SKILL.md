---
name: research-methodology
description: Choose hypotheses, baselines, gates, and next-step experiment priorities for Concept Encoder research. Use when planning what to run next, defining success criteria before training, comparing multiple completed runs at the roadmap level, or discussing research direction. Not for logging a single run, short run reports, or remote evaluation execution.
---

# Research Methodology for Concept Encoder

Use this skill for pre-run design and cross-run strategic decisions. It answers:
- what hypothesis is worth testing next,
- what the fair baseline is,
- what success or failure should mean before compute is spent,
- when a research track should continue, pivot, or stop.

Use `experiment-tracking` to record and interpret a concrete completed run once results exist.

## Do Not Use This Skill For
- updating `docs/2_Experiments_Registry/master_experiment_log.md`
- updating `docs/1_Strategy_and_Plans/active_todos.md` for a single run
- writing short run reports
- remote evaluation, SSH, or report syncing
- code traceability or `CHANGELOG.md` updates

If deep external literature review is required, pair this skill with the `Researcher` agent.

## First Principles Approach
- Reason about the concept bottleneck via information theory: capacity, compression, and entropy.
- Ask: what is the gradient signal, what does the loss surface look like, and what invariances does the model learn?
- Before proposing a new approach, search literature: "Has this been tried? Why did it fail then? What is different now?"
- Do not dismiss older papers (>5 years) - many lacked compute or data, not correctness.

## Research Tools
- Use `user-hf-mcp-server` MCP to search HuggingFace for related models, datasets, and papers.
- Use `WebSearch` for recent papers on concept bottlenecks, Perceiver IO, masked diffusion, latent reasoning, and recursive transformers.
- Ground decisions in `docs/1_Strategy_and_Plans/roadmap.md` and the current experiment history in `docs/2_Experiments_Registry/master_experiment_log.md`.

## Focused Execution
- One experiment at a time. Do not queue multiple new hypotheses before understanding the previous result.
- Define decision gates before training. Example: "If effective rank > 30/128, continue. If < 10/128 with no semantic lift, stop this variant."
- Compare alternatives with the nearest fair baseline first: same family, similar scale, similar data, similar checkpoint maturity.
- Use completed-run evidence already recorded by `experiment-tracking` before making roadmap-level decisions.
- Do not start training on remote servers without user confirmation.

## Evaluation Priorities

Do not optimize for full GLUE average. Priority order for semantic concept quality:

| Priority | Benchmark | Measures |
|----------|-----------|----------|
| 1 | STS-B (Pearson/Spearman) | Semantic similarity |
| 1 | Effective Rank | Concept collapse diagnosis |
| 2 | MRPC, QQP | Paraphrase detection |
| 2 | SICK-Relatedness | Semantic relatedness |
| 3 | PAWS, MNLI | Paraphrase overlap, NLI |
| Skip | CoLA, RTE, SST-2 | Architectural ceiling / noisy / saturated |

- Default classification head: `ViaDecoder`, not CLS-query.
- Effective rank < `10 / 128` is strong evidence of collapse and should outweigh isolated GLUE wins.

## Concept Quality Targets
- Effective rank > 50% of C (for example, > `64 / 128`)
- Mean pairwise similarity < `0.20`
- Max pairwise similarity < `0.60`
- STS-B Pearson > `0.75`
- Zero-shot STS-B cosine > `0.60`

## Decision Rules
- Use gates to decide whether to continue spending compute, not to dismiss informative partial progress automatically.
- A weaker absolute score on a harder objective or smaller model can still justify continuation if the semantic signal moved in the right direction.
- Repeated same-family failures with no geometry or semantic improvement are stronger evidence than one bad run.
- When in doubt, recommend the cheapest decisive next experiment, not the most ambitious one.
