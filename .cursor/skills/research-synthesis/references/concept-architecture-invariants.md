# Concept Reasoning Model — Architecture Invariants

## Contents

- Purpose
- Compression principle
- Cross-attention bottleneck
- Concept embeddings as reasoning state
- Decoder and reasoner conditioning
- Representation quality constraints
- Stable diagnostics
- What is not invariant

## Purpose

These are properties that define MrCogito regardless of which roadmap phase is active or which specific objective is being trained right now. Use them as the substrate for relevance judgments: "does this paper push on something the project actually depends on?"

If a paper is incompatible with these invariants, it is at most an "Adapt" candidate — never a drop-in fit.

## Compression Principle

The model compresses long token sequences into a much smaller set of concept tokens.

- Compression ratios in scope: roughly 4:1 to 128:1 (e.g. 128 concepts from 512 tokens, 8192 concepts from ~1M tokens).
- Concept count `C` scales with sequence length `N`, not fixed.
- Compression is lossy by design — the bottleneck is the point. Methods that try to "preserve everything" defeat the architecture.

## Cross-Attention Bottleneck

Concept tokens query input tokens via cross-attention.

- Cost: O(C · N) attention instead of O(N²) self-attention.
- Information flow from input to output passes through concepts.
- Methods that bypass the bottleneck (residual full-token paths, retrieval over raw tokens, attention sinks that re-expose tokens to the decoder) break the architecture's core claim.

## Concept Embeddings as Latent Reasoning State

Concepts are the working memory for downstream reasoning, decoding, or generation.

- Reasoning happens in concept space, not token space.
- Recursive or iterative refinement of concepts is a first-class option (weight-tied or otherwise).
- The concept space is intended to extend across modalities (audio first) via adapters that map into the same space, sharing the reasoning core.

## Decoder and Reasoner Conditioning

Output is produced from concepts via cross-attention.

- Decoders may be reconstructive (MLM, denoising, masked diffusion), generative (diffusion, autoregressive), or task heads (e.g. ViaDecoder for classification).
- The decoder must condition on concepts; "free" output paths defeat the bottleneck.

## Representation Quality Constraints

A "good" concept set is, simultaneously:

- **Semantically rich**: high task transfer (paraphrase, NLI, similarity).
- **Geometrically diverse**: high effective rank, low average pairwise similarity, low max pairwise similarity, no top-1 singular value dominance.
- **Generatively useful**: concepts can support coherent text reconstruction or generation, not just classification.

These constraints rule out common failure modes: concept collapse, single-direction dominance, and overfitting to one benchmark.

## Stable Diagnostics

These metrics have stayed informative across the project and are useful synthesis anchors. Numerical targets move with the roadmap; the diagnostics themselves are stable.

- Effective rank of the concept matrix.
- Mean and max pairwise concept similarity.
- Top-1 singular value dominance ratio.
- STS-B (zero-shot cosine and ViaDecoder).
- Paraphrase tasks: MRPC, QQP, PAWS.
- NLI: MNLI, SICK entailment.
- Generation proxies: prefix generation loss, suffix perplexity, conditional reconstruction loss.

When evaluating a paper's relevance, ask which of these diagnostics it would plausibly move and how that would be measured locally.

## What Is Not Invariant

These can change with the roadmap and should not be used as fixed criteria when synthesizing:

- Specific phase numbers or sub-goal labels.
- Specific compression ratios, layer counts, or hidden sizes.
- The set of training objectives in active rotation (MLM, TSDAE, prefix generation, masked diffusion, contrastive, ...).
- The benchmark chosen as the priority gate at any given time.
- Numerical targets (e.g. STS-B Pearson > 0.7x, effective rank > N/C).

For current values, always re-read `docs/1_Strategy_and_Plans/vision_and_goals.md` and `docs/1_Strategy_and_Plans/agenda.md` instead of caching them in this skill.
