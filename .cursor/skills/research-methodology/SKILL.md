---
name: research-methodology
description: Choose hypotheses, baselines, and decision criteria for Concept Encoder research. Use when deciding what experiment to run next, interpreting results against project goals, prioritizing benchmarks, searching literature, or discussing research direction. Not for run bookkeeping, code traceability, or changelog updates.
---

# Research Methodology for Concept Encoder

Use this skill to decide what to test and how to judge it. Use `experiment-tracking` to record a run once it exists.

## First Principles Approach
- Reason about the concept bottleneck via information theory (capacity, compression, entropy).
- Ask: what is the gradient signal? what does the loss surface look like? what invariances does the model learn?
- Before proposing a new approach, search literature: "Has this been tried? Why did it fail then? What is different now?"
- Do not dismiss older papers (>5 years) — many lacked compute/data, not correctness.

## Research Tools
- Use `user-hf-mcp-server` MCP to search HuggingFace for related models, datasets, and papers (arXiv IDs link models to papers).
- Use `WebSearch` for recent papers on: concept bottlenecks, Perceiver IO, masked diffusion, latent reasoning, TRM/recursive transformers.

## Focused Execution
- One experiment at a time. Analyze concept quality before the next run.
- Define decision gates before training: e.g. "If effective rank > 30/128, proceed. If < 10/128, hypothesis is wrong."
- Do NOT start training on remote servers without user confirmation. Concept analysis is fast and can run automatically.

## Evaluation Priorities

Do not optimize for full GLUE average. Priority order for **semantic concept quality**:

| Priority | Benchmark | Measures |
|----------|-----------|----------|
| 1 | STS-B (Pearson/Spearman) | Semantic similarity |
| 1 | Effective Rank | Concept collapse diagnosis |
| 2 | MRPC, QQP | Paraphrase detection |
| 2 | SICK-Relatedness | Semantic relatedness |
| 3 | PAWS, MNLI | Paraphrase overlap, NLI |
| Skip | CoLA, RTE, SST-2 | Architectural ceiling / noisy / saturated |

- Default classification head: **ViaDecoder** (not CLS-query).
- Effective rank < 10/128 signals collapse and overrides all GLUE scores.

## Concept Quality Targets
- Effective rank > 50% of C (e.g., > 64/128)
- Mean pairwise similarity < 0.20
- Max pairwise similarity < 0.60
- STS-B Pearson > 0.75
- Zero-shot STS-B cosine > 0.60
