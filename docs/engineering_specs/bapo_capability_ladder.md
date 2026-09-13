# BAPO DNA capability ladder — evaluation foundation (engineering spec)

- **Type:** engineering foundation (synthetic data + matched-architecture probe + plots).
  **Not** an `E0NN` experiment by itself; E24 is the first experiment that *uses* it.
- **Status:** implemented 2026-09-13. Extends `docs/engineering_specs/symbolic_long_context_suite.md`.
- **Owner:** Krzysztof Sopyla
- **Serves:** an honest, information-theoretic capability map of E18 vs matched dense
  transformers, as the observation base for later gating / selective-read / compression work.

## What already existed (do not duplicate)

The 2026-09-13 DNA-alphabet suite (`data/symbolic_tasks.py`) already isolates four mechanisms
with closed-form floors: `recall`, `far_copy`, `chain` (shuffled), `count`. A tiny
`perceiver_concept` probe (`verification/symbolic_channel_probe.py`) showed the concept array
*can* carry addressable far content (arm A 42% vs floor 25% / raw 100%), then a sibling agent
mapped the **Arm A → ~100%** exam at seq=128 (`docs/4_Research_Notes/symbolic_arm_a_100pct_limits_20260913.md`).
That work stays theirs. This ladder scales the *same* DNA contract onto **E18** and BAPO-hard
tasks, at tiny → medium (4k–16k) → large (32k–128k), with always-on dense controls.

E18's own ledger (natural text + keyed-recall rows) already says: one global read does exact
*positional* copy @32k (99.9998%) and fails *content* addressing (4.49% vs dense 99.33%). The
ladder is how we turn that into a controlled capability surface rather than a single anecdote.

## BAPO mapping (Schnabel et al., NeurIPS 2025, arXiv:2505.08140)

An `(a, b)`-BAPO caps prefix bandwidth `a` (bits of compressed summary) and attention
bandwidth `b` (raw prefix tokens). Our architectures realise the two channels explicitly:

| Architecture | `b` (raw tokens the answer stream can attend) | `a` (prefix memory) |
|---|---|---|
| `dense` (`perceiver_ar`, `par_mode=dense`) | `seq_len` in every layer | residual stream, every layer |
| `e18` (one global read + SWA stack) | `seq_len` in **one** layer; `local_window` elsewhere | one layer's KV = unbounded cache |
| `e18_local` (`global_layers=0`) | `local_window` only | 0 — must sit at the retrieval floor |
| `encdec` (symmetric encoder-decoder) | **0** on the suffix (self-attn cannot see the prefix) | encoder KV of the whole prefix |

The paper's synthetic suite (INDEX, MATCH2, MATCH3, REACHABILITY, MAJORITY, UNIQUE) is
re-implemented over the DNA alphabet so floors stay `ln(A)` rather than depending on English
tokenisation. We do **not** call the Microsoft API; we train <100M models from scratch.

## Tasks (generators in `data/symbolic_tasks.py`)

| task | user question | BAPO class | floor |
|---|---|---|---|
| `far_copy` | far copy / how many bits? | INDEX / bandwidth (`span_len=1` is INDEX) | `ln(A)` |
| `recall` | recall a fact from far past | MATCH2-easy | `ln(A)` |
| `select` | keep the facts, ignore noise | MATCH2 + decoy blocks | `ln(A)` |
| `chain_ordered` | follow A→B→C→D in order | DFA / easy | `ln(A)` |
| `chain` | same hops, shuffled order | REACHABILITY-hard | `ln(A)` |
| `unique` | the fact that appears once | UNIQUE / Σ-hard | `ln(A)` |
| `match3` | the fact planted three times | MATCH3-hard | `ln(A)` |
| `count` / `majority` | aggregate over the body | MAJORITY-hard | see `floor_nats` |

Retrieval tasks keep the original contract: `gap >= min_gap + 1`, so a raw window of
`min_gap` cannot see the evidence. Aggregation tasks (`count`, `majority`) use the dense arm
as the measured ceiling; their floor is exact only when none of the body is visible.

## Information-theoretic scores (`evaluation/bapo_metrics.py`)

Per held-out evaluation, against the closed-form floor:

- `recovered_nats` = `max(0, floor − CE) × answer_len`
- `recovered_bits` = recovered_nats / ln(2)  — **effective prefix bandwidth `a`**
- `information_flow` = recovered / prize ∈ [0, 1]
- `bits_per_supervised_token`, `bits_per_input_token`, `bytes_per_input_token`
- `nominal_b_tokens` and `nominal_a_bytes` (unbounded KV cache bytes / token, bf16)

The prize on a 32-symbol copy over A=4 is 64 bits. A model at chance recovers 0. A perfect
copy recovers 64 bits ≈ 0.67 bits/input-token at seq=96, or 0.083 bytes/input-token.

## Protocol (the 75% solvability control)

Every rung is **uninterpretable** until:

1. `dense` held-out accuracy ≥ **75%** at this scale, hidden size, and step budget
   (the task is learnable; this is the measured ceiling the experiment-design rule asked for).
2. On retrieval rungs, `e18_local` stays near chance (the task does not leak into the window).

If (1) fails, increase `span_len` / steps / width — do **not** report an E18 failure. Packed
loss (long copies) teaches faster than short answers; that is a documented limit of the
seq=128 Arm-A exam (span=8 stuck at chance, span=32 hit 99%).

## Scales (`data/bapo_ladder.py`)

| scale | seq_len | min_gap | local_window | where |
|---|---|---|---|
| `tiny` | 96 | 16 | 16 | CPU, this probe |
| `tiny_wide` | 256 | 32 | 32 | CPU |
| `medium` | 4096 | 1024 | 256 | GPU, <100M |
| `medium_16k` | 16384 | 4096 | 1024 | GPU |
| `large` | 32768 | 8192 | 1024 | GPU |
| `large_128k` | 131072 | 16384 | 1024 | GPU |

## How to run

```bash
# tiny core (far_copy, recall, select, chain_ordered, chain) + three architectures
uv run python verification/bapo_capability_probe.py \
    --scale tiny --arch dense e18 encdec e18_local \
    --out Cache/bapo_tiny

uv run python analysis/plot_bapo_capability.py \
    --in_dir Cache/bapo_tiny --out_dir Cache/bapo_tiny/plots
```

Plots: learning curves, accuracy heatmap, information flow, recovered bits vs prize,
effective vs nominal bytes/token.

## Non-goals

- Not a substitute for RULER-lite / lm-eval on language checkpoints.
- Not the `perceiver_concept` 100% far_copy map (sibling agent).
- No new training entrypoint: the probe is a verification script over the shared `perceiver_ar`
  family plus a reusable `nn/encdec_lm.py` baseline that is **not** registered in MODEL_REGISTRY.
