---
name: research-explain
description: Explain how an AI/ML paper or GitHub repository actually works, in plain language with diagrams and tensor-shape walkthroughs. Use this skill whenever the user wants to understand a paper or implementation in depth — covers architecture, forward pass shape by shape, training procedure, loss and gradient flow, and decoding/inference. Trigger on phrases like "explain this paper", "walk me through this repo", "how does X work", "show me the architecture / training / gradient flow / forward pass / decoding of Y", or "I want to learn how this method works", even when the user does not explicitly say "explain". Pair with the `research-scout` agent for fetching sources and with the `research-synthesis` skill when the user wants a project-fit verdict afterward.
---

# Research Explain — Paper and Repository Walkthroughs

Use this skill to produce a faithful, plain-language walkthrough of an AI/ML paper or GitHub repository: what the network is, how the forward pass transforms tensors, how training is set up, where gradients flow, and how decoding/inference works.

Pair with:

- `research-scout` agent — fetch papers, repos, and model cards when the user has not provided them.
- `research-synthesis` skill — when the user wants Adopt / Adapt / Watch / Reject and a smallest local test after the explanation.

This skill is paper-internal by default. Project mapping (to MrCogito's concept reasoning model) is optional and short; deep verdicts belong to `research-synthesis`.

## Walkthrough Workflow

Copy this checklist and tick items as you progress. Skip steps that do not apply.

```
Walkthrough Progress:
- [ ] Step 1: Anchor what is being explained (paper, repo, or component)
- [ ] Step 2: Read primary sources (paper method section + repo core files)
- [ ] Step 3: Plain-language summary (5–10 lines, no jargon)
- [ ] Step 4: Architecture diagram (mermaid)
- [ ] Step 5: Forward pass walkthrough (tensor shapes step by step)
- [ ] Step 6: Training procedure (data, objective, loss, schedule)
- [ ] Step 7: Gradient flow (where gradients live, what is frozen, weight tying, loss weighting)
- [ ] Step 8: Decoding / inference (how outputs are produced)
- [ ] Step 9: Glossary of non-obvious terms
- [ ] Step 10: Optional MrCogito mapping (one paragraph, only if helpful)
```

### Step 1: Anchor scope

Confirm what is actually being explained:

- Whole paper, whole repo, a single component (encoder, decoder, loss, sampler), or a specific code path.
- Match depth to the request. "Explain the loss" gets a focused answer, not a tour.

### Step 2: Read primary sources

For papers:

- Abstract → method/architecture → loss/training → ablations → limitations.
- Re-read the method section after looking at code — many details only become visible once you see the implementation.

For PyTorch repositories, read in this order (more detail in [references/code-reading-checklist.md](references/code-reading-checklist.md)):

1. `README.md` and the linked paper.
2. Training entry point (`train.py`, `pretrain.py`, or `trainer/`).
3. Model class and its `forward` method.
4. Loss / objective.
5. Data loading and collation (input shapes start here).
6. Inference / decoding entry points.
7. Config files (these encode many implicit choices).

### Step 3: Plain-language summary

Write 5–10 lines a colleague who does not work on this method could follow. No jargon without a one-line gloss. Lead with:

1. *Purpose* — what problem the method solves.
2. *Mechanism* — the one core idea.
3. *Why it works* — the intuition.

Heuristic: if a sentence needs four acronyms to parse, rewrite it.

### Step 4: Architecture diagram

Use a **mermaid** diagram for the high-level architecture. Keep it readable: 5–15 nodes, named blocks, and labeled edges where the label clarifies (e.g. `cross-attn (Q=concepts, K=V=tokens)`).

Ready-made recipes are in [references/diagram-patterns.md](references/diagram-patterns.md): encoder–decoder, cross-attention bottleneck, recursive / weight-tied, masked diffusion, autoregressive decoder.

If a graph diagram cannot capture the structure, fall back to ASCII boxes/arrows. Use the `GenerateImage` tool only if the user explicitly asks for a rendered hero diagram — it is the exception, not the default.

### Step 5: Forward pass walkthrough

Trace tensor shapes from input to output, one transformation at a time. Use a fixed concrete example so shapes stay recognizable across the response.

Define your symbols once at the top, e.g. `B = 8 (batch), N = 512 (tokens), C = 128 (concepts), D = 768 (model dim), V = 32k (vocab)`.

Format each step:

```
input shape  →  operation (parameters)  →  output shape   # one-line intent
```

Example:

```
(B, N)               → embed + pos                            → (B, N, D)   # token + positional embedding
(B, N, D)            → cross-attn(Q=concepts, K=V=tokens)     → (B, C, D)   # bottleneck: concepts attend to tokens
(B, C, D) × L blocks → self-attn + FFN + LN                   → (B, C, D)   # per-concept refinement
(B, C, D)            → head                                   → (B, *, V)   # task-specific output
```

Keep one shape per line, one short comment per line.

### Step 6: Training procedure

Cover, briefly:

- Dataset and batch composition (sequence length, masking ratio, special tokens).
- Objective and loss, in markdown math (no LaTeX): e.g. `loss = - Σ_i log p(x_i | context)`, `softmax(Q · Kᵀ / √d)`.
- Optimizer, LR schedule, warmup, weight decay.
- Mixed precision and gradient clipping.
- Distributed wrapping if non-trivial (DDP, FSDP, ZeRO).
- Curriculum or phases if any.

### Step 7: Gradient flow

Explain where gradients live and where they do not:

- Frozen vs trained parameters.
- Weight tying (shared embeddings, recursive / weight-shared layers — gradient through K applications).
- Stop-gradient operations (EMA targets, vector quantization, Gumbel tricks, `detach()`).
- Loss weighting (e.g. diffusion `1/t`, MLM only on masked positions, contrastive temperature).
- Truncated backpropagation through depth or time, if used.
- Auxiliary losses and how they balance against the main objective.

A short list with one line per point is usually enough.

### Step 8: Decoding / inference

Describe how outputs are produced at inference time, focusing on what differs from training:

- AR decoders: sampling strategy, KV caching, length control.
- Diffusion: number of sampling steps, schedule, guidance scale.
- Masked / iterative decoders: which positions are predicted at each step, refinement schedule.
- Classifiers / similarity heads: pooling, head choice, normalization.

Call out test-time tricks explicitly (recursive iterations, beam search, guidance, temperature).

### Step 9: Glossary

Provide a short glossary for any term used that a non-specialist might miss. Two- or three-line definitions, plain English. Skip terms the user clearly already knows from the conversation.

### Step 10: Optional — MrCogito mapping

Add a single short paragraph that maps the method's vocabulary to MrCogito's (concept tokens, cross-attention bottleneck, recursive refinement, decoder conditioning) only when:

- The user asks for it, or
- It would shorten the explanation by reusing project-known terms.

Do not produce a verdict here. If the user wants Adopt / Adapt / Watch / Reject, hand off to `research-synthesis`.

## Faithfulness Rules

- Stay close to what the paper or code actually does. If you simplify, mark the simplification: "(simplified — the paper also does X, see §3.2)".
- If something is unclear, say so explicitly. Do not invent a mechanism to fill a gap.
- Cite each non-obvious claim with a paper section / equation number, or a file path and line range in the repo.
- Prefer concrete numbers and shapes over abstract description.
- No LaTeX (project rule). Use markdown math notation throughout.

## Output Patterns

Match output to the request:

- "Explain this paper to me." → all steps, ~1–2 screenfuls per major section.
- "Walk me through this repo." → repo file tour (Step 2) + forward pass (Step 5) + training (Step 6) + decoding (Step 8). Skip Step 4 if the README already has a diagram.
- "Explain the loss / a single component." → focused answer covering only the relevant steps (often Steps 3, 6, 7).
- "Show me the forward pass with shapes." → Steps 3 and 5 only.

## Common Pitfalls

- **Paraphrasing past the paper.** Rewriting the abstract in different words is not an explanation. Add diagrams, shapes, and gradient flow.
- **Skipping shape annotations.** A walkthrough without tensor shapes hides the bottleneck where most confusion lives.
- **LaTeX leakage.** Anything inside `$...$` or `\\(...\\)` will not render. Keep math in plain markdown.
- **Diagram bloat.** A 30-node mermaid graph helps no one. Split or simplify.
- **Hidden assumptions.** When the paper and the released code disagree, the **config the user runs** wins. Note the discrepancy.
- **Stale repos.** If the repo's last commit is years old or training depends on dead datasets, flag it.

## Reference Files

- [references/diagram-patterns.md](references/diagram-patterns.md) — mermaid recipes for common architectures and an ASCII shape-trace template.
- [references/code-reading-checklist.md](references/code-reading-checklist.md) — what to look for in a PyTorch implementation: tensor shapes, masking, position encodings, normalization, loss weighting, sampling and decoding, mixed precision, distributed wrappers.
