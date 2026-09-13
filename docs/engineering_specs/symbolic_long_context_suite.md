# Symbolic long-context suite — tasks with closed-form information floors (engineering spec)

- **Type:** engineering foundation (synthetic data generator + diagnostic probe). **Not** an `E0NN` experiment.
- **Status:** implemented 2026-09-13. `data/symbolic_tasks.py`, `scripts/build_symbolic_dataset.py`,
  `verification/symbolic_channel_probe.py`, `tests/test_symbolic_tasks.py`.
- **Owner:** Krzysztof Sopyla
- **Serves:** E23 (supplies its two missing dense-label builders) and every later concept-channel
  experiment that needs to know whether its channel carries information *before* spending GPU-days
  on natural text.

## Problem (why this is needed)

E22 spent 0.44B tokens and two GPUs to learn that its concept array carried 0.05 nats of far
content — and that this was close to *all there was to carry*, because E18's reach ablation had
already measured a dense transformer of the same width extracting only 0.035–0.105 nats from
keys 2k–8k back. Three distinct failures were tangled together and the run could not separate
them ([root cause](../4_Research_Notes/e22_root_cause_20260912.md)):

1. the objective did not pay for the channel (natural-text CE barely rewards far context);
2. the read could not address the array;
3. the write blurred the content before it could be addressed.

The pre-registered gate (0.30 nats) was unreachable by construction, and nothing in the run
could tell us which of (1)–(3) was binding. Both problems have the same root: **we were
measuring a channel using data whose long-range information content we did not know.**

## What this provides

Sequences over a small symbolic alphabet (DNA-like, `--n_symbols 4` by default) in which the
supervised tokens are *determined* by evidence at a controlled distance and *statistically
independent* of everything a local window can see. That buys three things natural text cannot:

1. **A large, known prize.** `ln(n_symbols)` nats per supervised token — 1.39 nats at A=4, about
   28× the natural-text far-context prize — so a channel that works is unmissable and a channel
   that does not is unambiguous.
2. **An exact floor, not a measured one.** `floor_nats(cfg, window)` returns the cross-entropy of
   any model whose raw receptive field is `window` and which has no other route to the evidence.
   This is the direct fix for E22's gate mistake: the ceiling is *derived*, so "the channel
   carried information" is a measurement against a proof rather than against a control that may
   itself be weak. The `experiment-design` rule that every absolute gate must cite a measured
   ceiling is satisfied analytically here.
3. **Mechanism separation at negligible cost.** Each task isolates one mechanism, and the models
   are small enough to train on a laptop CPU, so (2) and (3) above can be settled before any
   GPU-day is committed.

### Tasks and the mechanism each isolates

| task | question | evidence | floor (per supervised token) |
|---|---|---|---|
| `recall` | can the read **address** the array by content? | one of several `key -> value` blocks, queried at the end | `ln(A)` |
| `far_copy` | how many **bits** does the channel carry? | a marked span to reproduce verbatim; sweep `span_len` | `ln(A)` |
| `chain` | can slots **compose**? | `a->b`, `b->c`, `c->d` scattered in random order | `ln(A)` |
| `count` | can the array **aggregate**? | count of the queried symbol over the whole body, mod `count_mod` | `H(Binom(n_far, 1/A) mod m)` |

`count` is the one task where a compressive bottleneck should have a *structural advantage*
rather than paying a tax: a running statistic is O(1) state, whereas exact attention must
re-derive it from the whole prefix at every query. Every prior experiment in this family has
measured how much the array *loses* against dense; this is the first task designed so the array
could *win*. If the family never wins anywhere, that is itself a finding.

### The contract that makes the floor exact

Rows guarantee `gap >= min_gap + 1`, where `gap` is the distance from the last evidence token to
the first supervised token. The first supervised token is predicted from `answer_start - 1`,
whose raw window of width `w` covers `[answer_start - w, answer_start - 1]`, so the evidence is
invisible exactly when `w <= min_gap`. **Set `min_gap >= dec_segment`** and the segment-confined
decoder provably cannot reach the evidence, for `dec_local` in both `block` and `swa`.

Nothing is memorisable across rows: keys, values, spans, chains and the counted symbol are drawn
fresh per row, so the mapping cannot be baked into weights. For `chain`, only *sources* are drawn
distinct (so each edge has one unambiguous target) while the terminal node is iid uniform —
a distinct-pool draw would leak a little information and make the floor slightly wrong.

## Validation (what is actually tested, not asserted)

`tests/test_symbolic_tasks.py`, 36 tests, CPU, ~4 s:

- **solvable from the evidence** — each task is solved by following its construction (the queried
  key occurs exactly once; the span matches; every chain hop is followed; the count is recomputed);
- **not solvable locally** — the answer marginal is uniform (chi-square), the answer is independent
  of the query key (chi-square on the contingency table), and a *local-window oracle* that
  memorises `visible window -> answer` over 1000 rows lands at chance on held-out rows;
- **the floor is right** — `_binomial_mod_entropy` agrees with a 400k-draw Monte-Carlo estimate to
  5e-3 nats, `floor_nats` is `ln(A)` at `window == min_gap` and undefined-by-convention above it,
  and the `count` floor decreases monotonically as the window covers more of the body;
- **both label routes agree** — the stored `labels` column and the marker route
  (`labels_from_span_markers` on the `answer`/`end` control ids) select the same supervised span.

## First result from the instrument (2026-09-13, CPU)

`far_copy`, alphabet 4, 32-symbol span, rows of 128, `dec_segment 32`, `min_gap 32`, `r = 8`,
4 decoder layers, 3000 steps, batch 32 — floor **1.3863** nats, chance accuracy **0.250**:

| arm | route to the evidence | CE (nats) | vs floor | accuracy |
|---|---|---|---|---|
| **D** | full raw access (`dec_segment = seq_len`) | **0.0000** | −1.3863 | **1.000** |
| **C** | segment-confined, no array | **1.3863** | +0.0000 | **0.250** |
| **A** | the array only (`exclusive` scope) | **1.1726** | −0.2137 | **0.423** |

Both controls behave, so the middle row means something: arm C is pinned at the floor to four
decimals across 3000 steps (the task does not leak) and arm D solves it outright (the task is
learnable at this size). On arm A's own weights, removing the array raises CE to **1.8458** at
chance accuracy — *worse* than the floor — so the entire gain is attributable to the array, and
`far`-slots-only equals `real` to four decimals, confirming exclusive scope makes the array a
purely long-range channel.

Three things follow that E22 could not establish with 0.44B tokens on two GPUs:

1. **The channel does carry addressable content.** This is the first positive evidence in this
   family; E22 only ever demonstrated a document embedding plus 0.05 nats of far marginal.
2. **It is badly bottlenecked.** It recovers 15% of the information that raw access recovers
   (0.214 of 1.386 nats), at 42% accuracy against 100%.
3. **The failure is in the *read*, not the write or the mask.** Perturbing only the far evidence
   moves the slot array by `max|Δz| = 1.68`, and the mask demonstrably exposes the slots holding
   it (slots 0–11 cover positions 0–191; the evidence ends below 188). The content is present and
   visible; what is missing is the ability to address it.

Arm A also shows a **~2000-step plateau at exactly the floor** before it escapes, which is the
signature the `xattn_wo_init_std` / `pooler_wo_init_std` knobs were added to investigate — see
`docs/4_Research_Notes/concept_channel_cold_start_20260913.md`.

## Interfaces

**`data/symbolic_tasks.py`** — `SymbolicVocab` (id layout: `n_symbols` content symbols at
`sym_lo`, then 8 control ids), `SymbolicTaskConfig`, `generate_row`, `iter_rows`, `floor_nats`,
`chance_accuracy`. No IO, so it is usable from a collator, a probe or a builder.

**`scripts/build_symbolic_dataset.py`** — one manifest source per `--task`, so a single manifest
carries the whole suite. Two schemas:

- default: `input_ids`, `labels` (aligned, `-100` off the answer, matching
  `build_copy_task_dataset.py`), `gap` — for standalone diagnostics and distance bucketing;
- `--lm_columns_only`: `input_ids`, `attention_mask`, `special_tokens_mask` only, i.e. exactly the
  columns `pretokenize_mix.py` writes, so `interleave_datasets` can mix these rows into a text
  corpus. Supervise via `--loss_span_markers` with the `markers` pair recorded in the manifest.
  Use `--sym_lo` to move the alphabet into a reserved slice of the text tokenizer's id space, as
  `build_retrieval_mix_dataset.py` does for its keys.

The manifest records `floor_nats_per_supervised_token`, `floor_window` and `chance_accuracy` per
source, so a spec can cite the ceiling by reference instead of restating an aspiration.

**`verification/symbolic_channel_probe.py`** — trains a tiny arm A (`concept_mode=full`,
`concept_xattn_scope=exclusive`, so the array is the *only* route) against arm C
(`concept_mode=none`) and reports both against the floor, plus arm A under
`concept_override("none")` and `("far")` for a same-weights attribution. Arm C sitting at the
floor is the instrument's self-check: if it drops below, the task leaks and the generator is
wrong, not the model.

## Role in the research program — and the limit of it

**A falsifier, not a success criterion.** A compressive channel that cannot do these tasks cannot
be useful at 1M–10M context, and that can now be established in CPU-minutes instead of GPU-days.
The converse does **not** hold: synthetic retrieval saturates easily (passkey is solved by models
that are poor at real long context), so passing says nothing on its own about language. Record
symbolic results as *mechanism* evidence and keep language claims on language data.

Concretely, the suite is used to (a) gate E23's architecture before launch — if the read cannot
address a 4-symbol alphabet at `min_gap`, exclusive scope on text will not save it; (b) supply
E23's `far_copy` and `chain` dense-label rows, the "two new small builders" its plan still needed;
(c) measure the channel's bit capacity by sweeping `span_len`, which no experiment has done.

## Non-goals

- Not a benchmark to report headline numbers on, and not a substitute for RULER-lite / STS-B.
- Not a language corpus: the alphabet has no morphology, so nothing here speaks to tokenisation,
  subword structure or semantics.
- No new training entrypoint: rows go through the existing `causal_lm` path, either with the
  `labels` column (`--preserve_precomputed_labels`) or via `--loss_span_markers`.
