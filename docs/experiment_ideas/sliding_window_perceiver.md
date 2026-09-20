# Sliding-window Perceiver — architecture + validation (idea, not a spec)

Undated mutable idea. Not frozen. Do **not** launch from this file.
If this becomes a run, `experiment-design` writes a spec in `ahead/` (new family
only if the encode object changes; otherwise an E21 flavour).

Author sketch (2026-09-20) kept in §8. This write-up is the recommended
architecture and the exam that would actually test the Vision.

Related: [vision](../1_Strategy_and_Plans/vision_and_goals.md) ·
[E18/E21 diagnosis](../4_Research_Notes/e18_e21_dense_baseline_diagnosis_20260916.md) ·
[slots vs latents](../4_Research_Notes/slot_write_vs_latent_write_20260920.md) ·
[Perceiver critique](../literature_review/perceiver_io_latent_reasoning_critique.md).

---

## 0. Verdict

The Vision needs a **notebook that can be looked up**, then a **tape to think
on**. E21 gave us the exclusive read (the right *exam condition*) and a mean of
16 tokens (the wrong *write*). Averaging is a gist compressor. It is not a
concept.

**One change to the architecture:** stop averaging a block. Give each window a
small **set of position-owned latents** that *pick* tokens inside that window
(cross-attention with a learned query, not a uniform mean). The number of
windows scales with sequence length. Latent sets of neighbouring windows may
overlap a little. Do not mix those sets with each other until lookup already
works. Do not call that mixer "reasoning".

**One change to validation:** no concept number is scored until an ordinary
dense model clears the same exam. Then probe the *write* (can a latent recover
the item planted in its window?) before you probe reasoning, length, or
language.

---

## 1. What the Vision actually requires (order is the design)

| Vision priority | What it needs from this model | What it is *not* |
|---|---|---|
| 1. Reasoning concept core | Addressable facts in the notebook | Self-attention on a fixed set |
| 2. Long context | Concept count **scales with N**; local windows, not one global array | A 32k language-model loss that a sliding window can satisfy alone |
| 3. Reasoning bandwidth | A **tape**: extra positions (or supervised loops) *after* the notebook holds facts | Extra depth on the same vectors (BAPO: layers at a fixed bottleneck do not add bandwidth) |
| 4. Agent messages | The notebook **is** the message (`prefix_kv` of the Z-sets) | A projector bolted onto a frozen LLM |
| 5. Multimodal | Same notebook object, later adapters | A second architecture |

E21's exclusive boundary is still the right *training condition* for (1) and
(4): if a raw path is open, next-token loss will ignore the notebook. The
object that crosses the boundary was the failure.

Analogy: a student may only use a notebook on the exam (exclusive read). If
every page is "the average of the last 16 paragraphs", they can copy a marked
span and cannot look up a name. We have been grading that student on lookup,
then adding more pens.

---

## 2. Architecture — overlapping window sets that pick, not average

### 2.1 The write (the bet)

```
tokens
  └─ optional cheap local mix inside a short window   (neighbours talk FIRST)
        └─ for each window i, stride S, width W:
              Z_i  =  C_w learned queries, each owning a subspan of W
                      softmax-pick tokens in window i  (not mean)
                      RoPE / position id of the window is on Z_i
  decoder (or the exclusive global read) sees only the Z-sets from completed windows
```

Concrete default (author's numbers, made falsifiable):

| knob | default | why |
|---|---|---|
| token width | 256 | small tokens; not the hypothesis |
| window W | 256 tokens | one page the set can actually see |
| stride S | 192 | overlap 64 — boundary continuity, **not** extra capacity |
| latents per window C_w | 32 | 8 tokens per latent → **8×** compression, the ICAE floor that still reconstructs |
| pick | softmax CA, learned query + position bias, **inside the window only** | replaces mean; this is the write |
| slot width | 4× token width (K and V, not a fatter gist) | a slot is an addressable pair, not one averaged vector |
| Z–Z mixer | **off** | mixing is not thinking; gate it |
| decoder local mix (SWA) | off at first | different question from "mix before the write" |

Coverage: sequence length N produces `⌈(N − W) / S⌉ + 1` windows, so
C_total ≈ 32 × N / 192 ≈ **N / 6** latents. Concept count **scales with N**.
That is Vision priority 2 by construction, unlike a fixed-128 Perceiver array.

### 2.2 Allocation is structural

Do **not** let the 32 queries of Z_i float over the whole 256-token window
(classic Perceiver: unallocated queries collapse or duplicate). Each latent
owns a **subspan** (~8 tokens) and may look a little into its neighbours
inside the same window. Softmax pick is how it *selects* inside that subspan,
not how it *chooses a topic* from the whole page.

Overlap **between** windows (the stride) is only so a fact on a cut is not
orphaned. It does not add addresses. Count of independently addressable items
is still ~ N / 8, not N / 6, because overlap shares coverage.

### 2.3 What this is, vs what we already ran

| object | E21 mean slot | E22 positional array | classic Perceiver | **this** |
|---|---|---|---|---|
| how a block is written | uniform mean of 16 | mean + 1 learned query per 16 | C queries over the **whole** book | 32 position-owned picks per 256, windows slide |
| C vs N | scales (1 per 16) | scales (1 per 16) | **fixed** | scales (windows × 32) |
| pick vs average | average | mostly average | pick, but unallocated | **pick, allocated** |
| mixer | none | 4-layer latent TF | deep Z-Transformer | none until lookup works |
| exclusive read | yes | segment-confined decoder | exclusive decode | **keep E21's exclusive boundary** |

Materially new vs E21: the sufficient statistic. Materially new vs Perceiver:
position-owned windows, C(N), no "reason in Z" story. Materially new vs E22:
many picks per window, not one mean; mixer off.

### 2.4 What we deliberately do **not** put in v1

- Tiny hashed embeddings — skip (author: no difference; not the bet).
- Decoder sliding-window attention as the main mix — **not** the same as the
  cheap local mix *before* the write. Literature (LCLM, DeepSeek) says
  contextualise-then-pool pays; decoder SWA is a later ablation.
- Extra heads as a research program. Heads change where the *read* looks.
  Lookup died because the *write* mixed keys, not because we had too few heads.
- Slot-to-slot attention, extra exclusive hop, or a second unique global layer
  billed as reasoning. Already falsified as the current walls (E25 extra hop
  is hops-only).
- Width / LR / SVD-init sweeps.
- In-place Infini overwrite; query-aware KV eviction.

### 2.5 Local mix before the write (disagreement with the sketch)

The sketch parks "SWA as mixing nearby tokens" as a future ablation. For the
**write**, a 1-layer local mix inside W is load-bearing, not optional: E18's
one-layer keys were not content-discriminative, and LCLM's encoder window is
the ingredient that actually moved language compression. That is *not*
decoder SWA. Keep it tiny (one layer, window = W). If even that is too much
for a first CPU probe, start without it and add it as the first repair when
keys don't separate — do not add a latent transformer instead.

### 2.6 Where reasoning lives (later, gated)

A set is memory. A tape is reasoning. After MATCH-class lookup through Z
passes:

- **Loops with a target on each latent** (LOTUS-shaped), or
- **Appended token / latent steps** (Vision priority 3).

Not: more self-attention on the same 32 vectors. BAPO: extra layers at a
fixed bottleneck do not raise prefix bandwidth.

The agent-message test (Vision priority 4) is then cheap: a second copy loads
only the Z-sets of the prefix and must match the receiver. That was E21's
unrun hook. It stays in the protocol; it is not v1's kill.

---

## 3. Validation protocol — aligned with the Vision, not with LM loss

### 3.1 Laws (non-negotiable)

1. **Dense first.** If a matched dense transformer does not clear the exam
   (accuracy ≥ 75% on packed answers, or the bits prize the spec names), the
   instrument is broken. Do not score the concept model. E28 scored exclusive
   0 bits on an exam whose dense control was 6.5% / 0.21 bits of a 40-bit prize.
2. **Write before read before reason.** (a) linear probe / argmax: can latent
   k recover the planted item in its subspan? (b) exclusive lookup through the
   read. (c) only then hops / loops.
3. **Copy ≠ lookup.** Report INDEX (marked-span copy) and MATCH (keyed recall)
   as different classes. A mean already copies. The bet is MATCH at 8×.
4. **Same prize at two lengths.** Score 1k and 4k on the *same* task. Passing
   1k and dying at 4k is a length story; dying at 512 is not. Do not jump to
   32k until MATCH at 8× passes at the current wall.
5. **Causal channel checks** on every checkpoint: feed the receiver the real
   Z-sets, none, a swapped row's Z-sets, or uncompressed tokens. Kill if
   swapped ≥ none (steering / gist) or if a local-only control solves the prize
   (boundary leak).
6. **The pick must move.** After training, freeze the queries back to a uniform
   mean. If the score does not collapse, the architecture learned nothing new.
7. **Language CE is not the gate.** Next-token loss at this scale pays ~0.05
   nats for far content. A falling loss with position is books getting easier,
   not memory. Do not train 32k under plain CE and call the notebook load-bearing.

### 3.2 Exam stack (cheap → dear)

| tier | what | pass / kill | GPU |
|---|---|---|---|
| 0 | Dense solvability on the exact JSON | dense ≥ 75% else stop | CPU / tiny GPU |
| 0b | Exclusive-mask dump at first answer token | planted span is visible; leftover packing is not a 4-token cliff | CPU |
| 1 | Write probe on packed DNA MATCH @512, 8× | probe accuracy tracks the planted key; mean-ablation collapses it | hours |
| 2 | Exclusive MATCH vs INDEX @512–1024 | MATCH ≥ 0.75 × uncompressed (identity / raw) **and** INDEX does not regress | hours |
| 3 | Same MATCH prize @1024 and @4096 | both, or it is not a length-scaling concept encoder | ~1 GPU-day |
| 4 | CogitoProbe only after dense S0 is real on that family | bits = capacity; bind = (entity, attr, value) not bag | ~1 GPU-day |
| 5 | Message round-trip (second copy, Z-sets only) | logits match; Vision priority 4 | after 2 |
| 6 | Loops / extra hop | only on a rung where 2 already passed; R=1 must *not* equal R=4 | gated |

DNA (4-symbol, exact bits) stays the **bandwidth** instrument. It is not
"semantic richness". Language / CogitoProbe bind is the semantics exam, and
only when its dense control passes.

Do **not** gate on: needle-in-a-haystack, last-256 perplexity, STS-B, GSM8K,
or a 125M LM eval that never saw long rows (E21's packing produced mean length
~3.2k on a 32k manifest).

### 3.3 Packing and data (the silent E21 hole)

If we ever train a language row with a message boundary: pack so the batch
**actually contains** long prefix→suffix dependencies. `length_group` on a
32k manifest that yields ~3k mean length is not a 32k test. Rows whose
answer is determined by tokens after the boundary do not train the notebook.
Pay for tokens whose label lives in the prefix (up-weight or packed retrieval
rows). Exclusive mask without a paying label is E22 again.

### 3.4 Kill the bet, don't retune it

- Pick ≈ mean on the trained checkpoint → 8× is gist-only; record it; do not
  sweep width or add a mixer.
- MATCH stays chance while INDEX is high → the window still writes a mixture;
  next lever is *smaller subspan* (more latents per window) or a raw/indexed
  path for exact retrieval — not reconstruction AE (already missed) and not
  hybrid key anchors (already missed).
- Dense fails the exam → fix the exam, not the model.
- Extra hop / glob=2 / loops lift hops and not MATCH → composition ≠ lookup;
  do not brand that as the reasoner.

---

## 4. How this is a Vision model, not a KV-cache paper

HCA / Beacon / LCLM all compress into KV and keep a raw path for copy. That
is a cache. The Vision is a **concept core** other copies can read.

This design is aligned if and only if:

1. Z-sets are the only far route (exclusive),
2. they are addressable (MATCH, write probe),
3. C grows with N (windows),
4. a second model can load them (message hook),
5. reasoning is added as a tape **on top of that object**, not instead of it.

If we only ever ship (1)+(3) with a mean write, we have a 16× cache and should
name it that way.

---

## 5. Recommended first run (when a spec is wanted)

**Claim.** Position-owned softmax picks inside overlapping windows, exclusive
read, 8×, recover MATCH at ≥ 0.75 × uncompressed where E21's mean, learned
mean, prefix AE, and hybrid anchors all scored ~0–3 bits — *because* the
failure was the mixture, not the container.

**Not in that run:** mixer, loops, 32k LM, hashed embeddings, decoder SWA,
head-count sweep.

**Pre-flight (0 GPU):** dense S0 on the JSON; mask dump; packing check.

ID: new family if we treat "window-set write" as a new encode object; else
`E21a` (the first flavour of the exclusive-read family). Decide at spec time.

---

## 6. Author sketch (source, 2026-09-20)

Do not repeat: slots as the average of r token vectors — we want a pick from
noise in each window.

Use: window latents with positions; latent count depends on sequence length;
sets Z_i of 32 vectors, each attending a sliding window (e.g. 8× coverage,
stride 192, Z_1 sees tokens 0–255, Z_2 sees the next window); latent width on
the order of 4× token embedding, still a real compression ratio; tiny token
embeddings (128/256).

Leave for later: tiny hashed embeddings (no difference seen); decoder SWA as
nearby mixing (unproven).
