# Glyph: a typed-vocab, structured-noise capability family

**Date:** 2026-09-13 · **Status:** specified + generators implemented; **not** an E0NN
and **not** a replacement for the DNA ladder. DNA A=4 remains the exact-floor control.
E18 is scored on Glyph only after a matched dense control hits 75%.

Does **not** derail E23 (exclusive concept channel) and does **not** duplicate the
sibling `perceiver_concept` Arm-A 100% `far_copy` exam.

Literature: [`docs/literature_review/synthetic_capability_exams.md`](../literature_review/synthetic_capability_exams.md).
Engineering: [`docs/engineering_specs/bapo_capability_ladder.md`](../engineering_specs/bapo_capability_ladder.md)
(Family G section). Code: `data/glyph_tasks.py`, recipes in `data/bapo_ladder.py`.

---

## Why A=4 (honest answer)

A=4 was chosen so three numbers are *exact*, not estimated:

1. **Floor** `ln(4) = 1.3863` nats per supervised token for any model whose raw window
   cannot see the evidence (`window ≤ min_gap`).
2. **Chance** 25% — a constant predictor cannot hide.
3. **Prize** `answer_len × 2` bits. A 32-symbol copy is 64 bits, full stop.

That is still the right instrument for *channel bandwidth*. E22 died because natural-text
CE paid ~0.05 nats for far context and the gate sat above the physical ceiling. DNA made
“the channel carried information” a proof rather than a comparison to a weak control.

What A=4 does **not** test:

- Ignoring *language-like* distractors. iid uniform filler has no local statistics a
  next-token model would latch onto. RULER’s essay haystack and BABILong’s PG-19
  insertion exist because iid noise is a toy.
- Typed structure: brackets, operators, a closed word list. 32 arbitrary ids would
  only raise chance to 1/32 and destroy the exact floor without buying a mechanism.
- Stack / permutation / predicate-filter algorithms. DNA `far_copy` is INDEX;
  `reverse` and Dyck are different circuits (Olsson; Delétang).

So: **keep DNA**. Add a **second family**, one coherent bet, same BAPO scores.

---

## One coherent bet (not ten micro-A/Bs)

**Claim.** A bounded-attention prefix oracle that can copy a marked DNA span still
fails when (i) the same span must be *permuted or filtered*, (ii) the haystack is a
process a local LM finds plausible, or (iii) the query is a *word/key* rather than a
position. Glyph makes those three failures separately visible, with a stack/DFA
verifier as the solvability proof and the same recovered-bits / information-flow
scores as DNA.

**Not in scope:** A/B of Markov stay probability, k ∈ {2,3,4} as separate experiments,
width 16 vs 32 as a research question (16 is a calibration subset), English TinyStories,
MiniPile, PG-19, SCAN MCD splits. Those are post-signal ablations.

**Kill / gate (same protocol as E24 DNA):**

- **S0 / K1:** dense ≥ 75% on a packed tiny rung before any E18 number is interpreted.
- **K2:** `e18_local` stays within chance+0.15 on retrieval rungs (structured noise
  must not leak the answer into the window).
- **S1:** E18 `copy_span` information_flow ≥ 0.75 × dense (positional INDEX still works
  when the haystack is Markov, not iid). If this fails, the noise model is too hostile
  — fix the generator, do not kill E18.
- **S2:** E18 `fact_markov_single` recovers ~0 bits while dense is ≥ 75% (content
  addressing wall, now with structured distractors).
- **S3:** E18 `reverse` and `every_k` lag `copy_span` (permutation / selective index
  are not free once copy works).
- **S4:** `dyck_close` is the stack rung; skip scoring E18 until dense S0.

---

## Literature → Adopt / Adapt / Watch / Reject

Full reviews: [`synthetic_capability_exams.md`](../literature_review/synthetic_capability_exams.md).

| Source | What they actually used | Verdict for us |
|---|---|---|
| BAPO (Schnabel et al., [arXiv:2505.08140](https://arxiv.org/abs/2505.08140)) | INDEX, MATCH2/3, REACHABILITY, MAJORITY, UNIQUE at n≤200, API models | **Adopt** hardness classes + `(a,b)` scores. DNA already did this. |
| RULER / NIAH ([arXiv:2404.06654](https://arxiv.org/abs/2404.06654)) | needles in *essay / repeat / distractor-needle* haystacks; variable tracking | **Adapt** haystack axis (not Paul Graham essays). |
| BABILong ([arXiv:2406.10149](https://arxiv.org/abs/2406.10149)) | bAbI facts hidden in PG-19 | **Adapt** planted fact + structured narrative noise, closed vocab. |
| Olsson et al. ([arXiv:2209.11895](https://arxiv.org/abs/2209.11895)) | copy / reverse / induction `[A][B]…[A]→[B]`; needs two layers | **Adapt** reverse vs copy as rungs; mechanistic reason E18 may copy and fail reverse. |
| Delétang et al. ([arXiv:2207.02098](https://arxiv.org/abs/2207.02098)) | Reverse, Dyck-ish modular arithmetic, PARITY, duplicate string, `{a,b}` alphabets | **Adapt** reverse + bounded Dyck as in-distribution packed spans. **Watch** PARITY (DNA `count` already). |
| MAD / MQAR ([arXiv:2403.17844](https://arxiv.org/abs/2403.17844), Zoology) | vocab 8–64 KV recall; selective copy; noisy recall | **Adapt** selective copy + noisy recall. |
| TinyStories ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759)) | ~1500-word child English, <10M LMs | **Adapt** closed word list as keys. **Reject** English BPE as the probe. |
| MiniPile ([arXiv:2304.08442](https://arxiv.org/abs/2304.08442)) | 6 GB Pile subset | **Reject** as an exam (not tiny-vocab, no floor). |
| SCAN / CFQ | OOD compound splits | **Watch** as a later split, not a new family. |
| RASP Dyck ([arXiv:2106.06981](https://arxiv.org/abs/2106.06981)) | Dyck-k P/T/F with a stack program | **Adapt** bounded Dyck-2 close-span. |

**Composition (the Adapt):** BAPO classes × MAD selective copy × Delétang reverse/Dyck ×
BABILong structured haystack, on one typed 32-token alphabet, same recovered-bits
instrument. That is one family.

---

## Typed vocab (16 and 32)

Not 32 arbitrary ids. Each id has a *role* a local LM can model.

**Glyph-32 (canonical), ids 0..31**

| ids | class | names |
|---|---|---|
| 0–7 | digits | `d0 … d7` |
| 8–15 | letters | `a … h` |
| 16–19 | brackets | `( ) [ ]` |
| 20 | operator | `plus` |
| 21–24 | closed words | `cat dog red blue` |
| 25–31 | roles | `bos eos query answer end mark hop` |

**Glyph-16 (calibration subset):** digits `d0–d3`, letters `a–d`, `( )`, roles
`bos eos query answer end mark`. No hop / words / Dyck-2. Tasks that need those
(`dyck_close`, `story_fact`, chain) **require 32**.

Answer-class size (`n_symbols` for the DNA floor duck-type) is **not** 16 or 32:

| task | answer class | floor per token |
|---|---|---|
| `copy_span` / `reverse` / `every_k` / chain | 8 letters (4 on width 16) | `ln(8)` or `ln(4)` |
| `filter_mod` (mod 3, 8 digits) | `{0,3,6}` | `ln(3)` |
| `dyck_close` | 2 closers | `ln(2)` |
| `fact_markov` / `story_fact` | 8 digits | `ln(8)` |

Chance is `1/n_symbols` of the *answer class*. Packed spans still target 16/24/32
supervised tokens so CE has a gradient.

---

## Structured noise model

DNA filler: iid uniform over A=4.

Glyph filler: after evidence is placed, each contiguous hole is filled by a **known
process**. Default at tiny: `markov`. `mixed` is the intended medium haystack
(RULER essay-vs-noise Adapt). `iid` exists as a contrast, not a scored recipe.

| `--noise` | Process | Why a local LM likes it | What the answer ignores |
|---|---|---|---|
| `markov` | sticky bigram over letters (width 16) or letters+words (width 32): stay 0.55, step-cyclic 0.30, jump 0.15 | locally predictable closed “language” | everything except the marked evidence |
| `dyck` | valid Dyck-1 (width 16) or Dyck-2 (width 32) fragment; odd length pads one letter | well-formed brackets | the Dyck is *irrelevant* (except on `dyck_close`, where a *marked* unmatched stack is the evidence) |
| `arith` | well-formed `d (plus d)*` (width 32); never starts/ends on `plus` | well-formed expressions, values unused | the values |
| `mixed` | each hole independently picks markov / dyck / arith | RULER-style mixed haystack | same |
| `iid` | uniform over the noise alphabet | DNA-like control inside Glyph | — |

**Contract (unchanged from DNA):** `gap ≥ min_gap + 1`, so a raw window of
`min_gap` cannot see evidence. The query is uniquely determined by `expected_answer`
(the verifier). Structured noise lives in the *holes*, never uses role tokens, and
is independent of the answer given the evidence.

A local model can drive haystack CE far below `ln(|noise alphabet|)` and still sit at
the answer-class floor. That is the point: “ignore language-like distractors”.

---

## Task ladder (layouts)

Every row: `BOS | structured haystack + planted evidence | QUERY [q] ANSWER [y] END EOS`.
Labels cover `[y]` only. Verifier = solvability proof.

### 1. `copy_span` — positional INDEX in a structured haystack

- **Layout:** `[mark] ℓ_1…ℓ_n` planted far; query empty; answer = the span.
- **Keep / ignore:** keep the marked letters; ignore Markov/Dyck/arith.
- **Verifier:** the unique `mark` span equals the answer.
- **Prize:** `n × log2(|letters|)` bits (64 bits at n=16, 8 letters).
- **Why E18:** if this matches DNA `far_copy`, structured noise did not break INDEX.
  If it fails while DNA copy works, the haystack is the cause (fix noise, don’t kill E18).

### 2. `reverse` — positional + permutation

- **Layout:** same marked letter span; answer = span reversed.
- **Keep / ignore:** the span, in reverse order; ignore haystack.
- **Verifier:** `expected = reverse(span)`.
- **Why E18:** Olsson reverse is not the induction copy circuit. Delétang: transformers
  length-generalise poorly on Reverse String. We ask the weaker in-distribution
  question first (packed CE, dense ≥ 75%).

### 3. `every_k` — selective indexing

- **Layout:** `[mark] ℓ_1…ℓ_n`; query = digit `k` (shown to the decoder); answer =
  `span[0::k]` (indices 0, k, 2k, …). `span_len` divisible by `k`.
- **Keep / ignore:** every k-th letter; drop the rest and the haystack.
- **Verifier:** subsequence at those indices.
- **Why E18:** content-free *position arithmetic* on a far span. A one-layer read that
  copies a contiguous block may not stride.

### 4. `filter_mod` — predicate filter (MAD selective copy)

- **Layout:** `[mark]` + a digit span containing **exactly** `n_keep` digits ≡ 0
  (mod `m`) and the rest not; query = digit `m`; answer = keepers **in order**.
- **Keep / ignore:** digits matching the predicate shown to the decoder; ignore
  non-matching digits and the haystack.
- **Verifier:** scan span, keep `d % m == 0`.
- **Prize:** `n_keep × log2(#keeper classes)` (`ln(3)` nats at 8 digits, m=3).
- **Why E18:** must *content-filter* far tokens, not copy a block. DNA `select_1decoy`
  was a type cue (`keymark` vs `decoy`); this has one marker and a predicate.

### 5. `dyck_close` — bounded stack (width 32)

- **Layout:** `[mark]` + `span_len` unmatched openers (a valid Dyck-2 prefix);
  query empty; answer = matching closers, reverse order.
- **Keep / ignore:** the unmatched stack; ignore haystack Dyck *fragments* (those are
  fully balanced and irrelevant).
- **Verifier:** push openers, emit `closer_of[stack[::-1]]`.
- **Prize:** `span_len × 1` bit (two closer types).
- **Why E18:** DCF / stack. BAPO-easy only if the prefix oracle ships the stack
  (`a = O(depth · 1 bit)`). A one-layer KV that stored a document embedding cannot.

### 6. `fact_markov` — keyed recall in Markov haystack (BABILong Adapt)

- **Layout:** `[mark] key letters · value digits`, plus `n_distractors` other
  marked facts; query = key; answer = value. Haystack is Markov, not iid DNA.
- **`fact_markov_single`:** `n_distractors=0` (the calibrated DNA `recall_single` analog).
- **Keep / ignore:** the matching marked fact; ignore Markov filler and other keys.
- **Verifier:** unique key match.
- **Why E18:** DNA `recall_single` recovered **0 bits** vs dense 99%. This asks whether
  that wall is about iid filler (too easy to skip) or about content addressing as such.
  If E18 still recovers 0 and dense ≥ 75%, the wall is the architecture.

### 7. `story_fact` — closed-word key (TinyStories Adapt, width 32)

- **Layout:** `[mark] cat|dog · digit payload`; query = the word; answer = payload.
  Haystack Markov over letters+words so distractors look like a tiny story.
- **Keep / ignore:** the planted entity’s payload; ignore `red/blue/cat/dog` chatter.
- **Verifier:** unique entity match.
- **Why E18:** same MATCH2 mechanism, keys from a *word class* the local LM models.

### 8. `chain_ordered_noise` / `chain_shuffled_noise` — hops in structured filler

- **Layout:** `[hop] src letters · dst letters`. Ordered: true hops occupy the left
  of the body. Shuffled: random order. When `n_distractors ≥ hops`, distractors prefer
  a **second independent chain** (not iid edges).
- **Keep / ignore:** follow the queried source `hops` times; ignore the second chain
  and the haystack.
- **Verifier:** unique outgoing edge per source; walk `hops` steps.
- **Why E18:** DNA already showed ordered DFA easy (97%) and shuffled **not
  dense-solvable** at 0.6M. Glyph asks the same composition question with a haystack
  a local LM likes. **Do not score E18 on shuffled until dense S0.**

### Still paper (specified, not generated)

- **`arith_eval`:** marked `( d + d )` with nested brackets, emit the residue mod 8.
  Delétang Modular Arithmetic (DCF). Needs its own dense S0; easy to make K1.
- **`filter_pattern`:** a pattern shown *in the query* (not a fixed modulus) — the
  decoder-side template Olsson-style. After `filter_mod` calibrates.
- **SCAN-style OOD split:** hold out `k=3` or `hops=4` at test. After in-distribution
  dense ≥ 75%, not as a first recipe.

---

## Implemented vs still paper

**Implemented** (`data/glyph_tasks.py`, `tests/test_glyph_tasks.py`, recipes in
`data/bapo_ladder.py`, probe `--recipe reverse … --width 32 --noise markov`):

- Vocab 16 and 32, typed layout.
- Noise: `markov`, `dyck`, `arith`, `mixed`, `iid`.
- Generators + verifiers: `copy_span`, `reverse`, `every_k`, `filter_mod`,
  `dyck_close`, `fact_markov`, `story_fact`, `chain_ordered_noise`,
  `chain_shuffled_noise`.
- DNA suite **untouched**. Probe default is still the DNA tiny core.
- Same `info_report` recovered-bits / information-flow / bytes/token (duck-typed
  `n_symbols` = answer class).
- Dense ≥ 75% protocol unchanged. Glyph recipes are **uncalibrated** until an S0.

**Still paper:** `arith_eval`, decoder-side `filter_pattern`, SCAN OOD splits,
`noise=mixed` as the default medium haystack (code supports it; recipes still
default `markov` for the first S0).

**Not this family:** sibling Arm-A 100% DNA `far_copy`. English RULER-lite on
language checkpoints (already a different eval layer).

---

## Next calibration step (CPU, not Odra)

Odra medium DNA (`E24_fc` / `E24_rs` / `E24_sel`) stays up. Do not launch Glyph there.

```bash
# CPU S0 — dense only, packed tiny, Markov haystack
uv run python verification/bapo_capability_probe.py \
    --scale tiny --recipe copy_span reverse every_k filter_mod fact_markov_single \
    --arch dense --out Cache/bapo_glyph_tiny_s0
```

Then, only on rungs that hit 75%: add `e18 e18_local encdec`. `dyck_close` and
shuffled chain are hunts. Width 16 is a cheaper S0 if 32 is slow; it cannot run
Dyck-2 / story / hop.

If dense misses 75% on `reverse` after 4× steps and packed span: **K1**, pack harder
or drop noise to `iid` *only as a generator debug*, then put Markov back. Do not
score E18 on an iid Glyph reverse and claim structured-noise results.

---

## Proof the task is solvable

`expected_answer(cfg, ids)` reconstructs `y` from the prefix with no weights.
`tests/test_glyph_tasks.py` asserts it matches the labels on every task, that
`gap ≥ min_gap + 1`, that noise runs emit no role tokens, that Dyck fragments
balance and arith filler is well-formed, and that the answer-class floor is
`ln(|class|)`. That is the proof. A model with a working channel *can* reach
zero loss; whether E18 does is the measurement.
