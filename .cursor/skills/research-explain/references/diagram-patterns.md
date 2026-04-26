# Diagram Patterns

## Contents

- Conventions
- Encoder–Decoder
- Cross-attention bottleneck (concept tokens / Perceiver-style)
- Recursive / weight-tied encoder
- Masked diffusion decoder
- Autoregressive decoder
- ASCII shape-trace template

## Conventions

- Use **mermaid** for graph-shaped architectures, **ASCII** for tensor-shape pipelines.
- 5–15 nodes per diagram. Beyond that, split into sub-diagrams.
- Label edges that clarify intent (e.g. `cross-attn (Q=concepts, K=V=tokens)`).
- Keep shape symbols consistent across the response: `B` batch, `N` input tokens, `C` concepts, `D` model dim, `H` heads, `V` vocab, `T` diffusion steps, `K` recursive iterations.
- No LaTeX; use markdown math (`softmax(Q · Kᵀ / √d)`).

## Encoder–Decoder

```mermaid
flowchart LR
  X["Input tokens (B,N)"] --> EMB["Embed + Pos (B,N,D)"]
  EMB --> ENC["Encoder × L"]
  ENC --> H["Hidden (B,N,D)"]
  H --> DEC["Decoder × L (cross-attn to H)"]
  Y["Target tokens (B,M)"] --> DEMB["Embed (B,M,D)"]
  DEMB --> DEC
  DEC --> HEAD["LM head"]
  HEAD --> P["Logits (B,M,V)"]
```

## Cross-Attention Bottleneck (concept tokens / Perceiver-style)

```mermaid
flowchart LR
  X["Input tokens (B,N,D)"] --> CA
  Q["Concept queries (B,C,D)"] --> CA["Cross-attn (Q=concepts, K=V=tokens)"]
  CA --> Z["Concepts (B,C,D)"]
  Z --> SA["Self-attn × L (over C)"]
  SA --> Zr["Refined concepts (B,C,D)"]
  Zr --> DEC["Decoder / head"]
```

Key property: information flows tokens → concepts, not back. Bottleneck width is `C`, scales with `N`.

## Recursive / Weight-Tied Encoder

```mermaid
flowchart LR
  X["Tokens (B,N,D)"] --> Z0["Init concepts (B,C,D)"]
  Z0 --> L["Layer block (shared weights)"]
  L -- "iterate K times" --> L
  L --> Zk["Concepts after K iters (B,C,D)"]
  Zk --> HEAD["Decoder / head"]
```

The same layer block is applied K times. Gradients flow through all K applications, or only the last `k` if the paper uses truncated backpropagation through depth.

## Masked Diffusion Decoder

```mermaid
flowchart LR
  Z["Concepts (B,C,D)"] --> COND
  XT["Noised tokens at step t (B,N)"] --> EMB["Embed + Pos"]
  EMB --> COND["Decoder × L (cross-attn to Z, AdaLN-Zero(t))"]
  T["Step t"] --> COND
  COND --> P["Logits (B,N,V)"]
  P --> X0["Predicted x_0"]
```

Loss is reweighted (e.g. `1/t`) and concepts condition every layer via cross-attention.

## Autoregressive Decoder

```mermaid
flowchart LR
  Y["Target tokens (B,M)"] --> EMB["Embed + Pos"]
  Z["Concepts (B,C,D)"] --> CA
  EMB --> SA["Causal self-attn"]
  SA --> CA["Cross-attn (Q=positions, K=V=Z)"]
  CA --> FFN["FFN + LN"]
  FFN --> P["Next-token logits (B,M,V)"]
```

## ASCII Shape-Trace Template

```
DEFS:  B = 8 (batch), N = 512 (tokens), C = 128 (concepts),
       D = 768 (model dim), V = 32k (vocab)

(B, N)               tokens
  → embed + pos      (B, N, D)            # word + positional
  → cross-attn       (B, C, D)            # concepts query tokens (bottleneck)
  → self-attn × L    (B, C, D)            # per-concept refinement
  → norm             (B, C, D)
  → decoder x-attn   (B, N, D)            # decoder cross-attends to concepts
  → lm head          (B, N, V)            # logits
```

Keep one shape per line, one inline comment per line, max ~10 words.
