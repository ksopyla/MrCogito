# BAPO DNA capability ladder — evaluation foundation (engineering spec)

- **Type:** engineering foundation (synthetic data + matched-architecture probe + plots).
  **Not** an `E0NN` experiment by itself; E24 is the first experiment that *uses* it.
- **Status:** implemented 2026-09-13 (DNA). **2026-09-13 extension:** Family G (Glyph) —
  typed vocab 16/32, structured noise, stack/DFA verifiers — in `data/glyph_tasks.py`.
  DNA is untouched and remains the exact-floor control. Glyph rungs are uncalibrated
  until a dense S0. Spec: [`glyph_capability_ladder.md`](../4_Research_Notes/glyph_capability_ladder.md).
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
copy recovers 64 bits ≈ 0.50 bits/input-token at seq=128, or 0.0625 bytes/input-token.

## Protocol (the 75% solvability control)

Every rung is **uninterpretable** until:

1. `dense` held-out accuracy ≥ **75%** at this scale, hidden size, and step budget
   (the task is learnable; this is the measured ceiling the experiment-design rule asked for).
2. On retrieval rungs, `e18_local` stays near chance (the task does not leak into the window).

Packed loss (long copies / long values / long chain keys) teaches faster than short answers;
that is a documented limit of the seq=128 Arm-A exam (span=8 stuck at chance, span=32 hit 99%).
`config_for` therefore grows the supervised span toward 16 tokens on `tiny`, 24 on `tiny_wide`,
and 32 on medium/large, shrinking decoys only when the packed answer would not fit behind
`min_gap`. Aggregation tasks (`count`, `majority`) stay 1-token by design.

The probe trains **dense first**. Other arches are skipped on a rung whose dense control missed
75% after `--steps * --k1_mult` (K1, default 4×). Compressed models then train for
`max(--steps, dense_steps_used)` so they are not starved relative to the control.

### Named recipes

Default `--task` still generates the harder MATCH2 / shuffled-chain hunts. Those are **not**
tiny-solvable at 0.6M / 4000 steps. Score E18 only on `CALIBRATED_RECIPES`:

| `--recipe` | generator | what it measures | tiny dense (H=128) |
|---|---|---|---|
| `far_copy` | `far_copy` | positional INDEX / bandwidth | **99.4%** |
| `recall_single` | `recall`, `n_distractors=0` | content addressing, one planted key | **99.2%** |
| `select_1decoy` | `select`, 0 distractors + 1 decoy | type-cue (`keymark` vs `decoy`), not MATCH2 | **99.2%** |
| `chain_ordered` | `chain_ordered` | in-order DFA hops | **93.1%** |

Still uncalibrated at tiny (do not score E18): default `recall` / `select` (MATCH2), shuffled
`chain` even with `n_distractors=0` (~34%). `--recipe chain_shuffled` is a hunt, not a gate.

```bash
# score E18 on the calibrated set
uv run python verification/bapo_capability_probe.py \
    --scale tiny --recipe far_copy recall_single select_1decoy chain_ordered \
    --arch dense e18 encdec e18_local --out Cache/bapo_tiny_calibrated
```

## Scales (`data/bapo_ladder.py`)

| scale | seq_len | min_gap | local_window | where |
|---|---|---|---|
| `tiny` | 128 | 16 | 16 | CPU, packed answers ≥16 tokens (chain ~12) |
| `tiny_wide` | 256 | 32 | 32 | CPU, packed ≥24 |
| `bridge` | 512 | 64 | 16 | GPU INDEX; `window < gap` so `e18_local` is a real leak control |
| `bridge_1k` | 1024 | 64 | 16 | GPU INDEX scale-up |
| `medium` | 4096 | 1024 | 256 | GPU, packed 32, <100M; **spread 4k is K1 until dense ≥ 75%** |
| `medium_16k` | 16384 | 4096 | 1024 | GPU |
| `large` | 32768 | 8192 | 1024 | GPU |
| `large_128k` | 131072 | 16384 | 1024 | GPU |

Tiny packed `far_copy` is mostly a **fixed-offset INDEX** (span in a narrow gap band). Spread
placement at 512+ makes the span's position vary widely (content-address the `spanmark`); a
0.6–16M 4-layer dense control does not do that. Use `--evidence_align right` for INDEX S0
(`row.gap == min_gap+1`). Keep `local_window < min_gap` whenever scoring E18 (K2). Do not score
E18 at 4k until a dense control hits 75% on that same recipe.

## How to run

```bash
# calibrated tiny set (the numbers that are allowed to score E18)
uv run python verification/bapo_capability_probe.py \
    --scale tiny --recipe far_copy recall_single select_1decoy chain_ordered \
    --arch dense e18 encdec e18_local \
    --out Cache/bapo_tiny_calibrated
```

uv run python analysis/plot_bapo_capability.py \
    --in_dir Cache/bapo_tiny --out_dir Cache/bapo_tiny/plots
```

Plots: learning curves (accuracy + CE), accuracy heatmap, information flow, recovered bits vs prize,
effective vs nominal bytes/token, plus `capability_table.csv`.

## Medium / large (GPU, still <100M)

Tiny is the solvability proof. Scale up **only the calibrated recipes** (the default MATCH2 and
shuffled-chain generators are still ill-posed at 0.6M). Suggested first GPU rung, AMP on:

```bash
# First GPU INDEX rung (not advertised 4k spread). Right-align, window < gap.
uv run python verification/bapo_capability_probe.py \
    --scale bridge --recipe far_copy --arch dense e18 e18_local \
    --evidence_align right --hidden 128 --steps 800 --k1_mult 4 --amp auto \
    --out Cache/bapo_bridge

# Odra/Polonez. Do not launch this 4k spread recipe until a dense S0 hunt hits 75%.
# uv run python verification/bapo_capability_probe.py \
#     --scale medium --recipe far_copy --arch dense \
#     --hidden 256 --steps 2000 --batch 8 --amp auto --out Cache/bapo_medium
```

`--scale medium_16k` / `large` / `large_128k` need a smaller batch if VRAM is tight;
the architecture factory already caps `--max_params 100000000`. Do not launch shuffled `chain`
or multi-item MATCH2 at those lengths until a dense control hits 75% at the same hidden size.

The first advertised medium recipe (H=256, 4 layers, 1 KV head, `global_logit_scale=none`,
batch 32, 8000 dense steps, **spread** placement) **did not** clear S0: dense stayed at chance
(~25%, CE=ln(4)) for 3800+ steps on `far_copy`, `recall_single`, and `select_1decoy`. Right-align
`far_copy` at seq=512, gap=65, H=128 **does** clear S0 (dense 99.7% @1950) and E18 matches it
(100% / 64 bits). The same recipe at seq=1024, H=256 is dense 99.9% and **E18 0 bits** @4000.
4k right-align at H=256 is chance; one 16M seed reached 74% (47 bits). Do not score E18 at 4k.

```bash
# S0 hunt example. Prefer --scale bridge for 512 INDEX; raise width at 4k until dense ≥ 75%.
uv run python verification/bapo_capability_probe.py \
    --scale bridge --recipe far_copy --arch dense \
    --evidence_align right --hidden 128 --steps 800 --k1_mult 4 --amp auto \
    --out Cache/bapo_bridge_s0
```

## Family G — Glyph (typed vocab, structured noise)

DNA A=4 + iid filler is the exact-floor *bandwidth* instrument. It does not test
“ignore language-like distractors” or stack/filter algorithms. Glyph is a **second
family**, not a replacement: vocab 16/32 is typed (digits, letters, brackets, `plus`,
`cat/dog/red/blue`, role markers), filler is Markov / valid Dyck / well-formed
arithmetic, and each row has a stack/DFA/scan verifier. Same BAPO scores, same 75%
dense gate, DNA generators untouched.

Do **not** duplicate the sibling `perceiver_concept` Arm-A 100% DNA `far_copy` exam.
Do **not** score E18 on Glyph until dense ≥ 75% at the same scale.

| `--recipe` | generator | mechanism | notes |
|---|---|---|---|
| `copy_span` | `copy_span` | INDEX in a Markov haystack | positional control inside Glyph |
| `reverse` | `reverse` | Delétang / Olsson reverse | permutation, not copy |
| `every_k` | `every_k` | selective indexing; `k` in the query | `span[0::k]` |
| `filter_mod` | `filter_mod` | MAD selective copy; `m` in the query | digits ≡ 0 (mod m) in order |
| `dyck_close` | `dyck_close` | bounded Dyck-2 stack | width 32 only |
| `fact_markov_single` | `fact_markov`, 0 distractors | MATCH2-easy in Markov filler | BABILong Adapt |
| `story_fact` | `story_fact` | closed-word key | width 32; TinyStories Adapt |
| `chain_ordered_noise` / `chain_shuffled_noise` | hops in structured filler | DFA vs REACHABILITY | shuffled is a hunt |

```bash
# CPU S0 (dense only). Do not score E18 until 75%. DNA 512/4k GPU hunts take priority.
uv run python verification/bapo_capability_probe.py \
    --scale tiny --recipe copy_span reverse every_k filter_mod fact_markov_single \
    --arch dense --out Cache/bapo_glyph_tiny_s0
```

`--width 16|32` (default 32) and `--noise markov|dyck|arith|mixed|iid` are probe
overrides. Layouts, floors, and kill criteria:
[`docs/4_Research_Notes/glyph_capability_ladder.md`](../4_Research_Notes/glyph_capability_ladder.md).
Papers: [`docs/literature_review/synthetic_capability_exams.md`](../literature_review/synthetic_capability_exams.md).

## Non-goals

- Not a substitute for RULER-lite / lm-eval on language checkpoints.
- Not the `perceiver_concept` 100% far_copy map (sibling agent).
- No new training entrypoint: the probe is a verification script over the shared `perceiver_ar`
  family plus a reusable `nn/encdec_lm.py` baseline that is **not** registered in MODEL_REGISTRY.
