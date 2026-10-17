# Writing to the notebook: compression, meaning and their limits (16 Oct 2026)

Research synthesis for the E31 / E33 latent notebook (the writer in `nn/latent_memory.py`). Question from the
author (16 Oct): how to make the writer's output more compressed, more semantic and more meaningful; what the
theoretical limits are; what Cramming and the 2025–2026 literature tell us; every candidate solution with how
practical it is at our compute, ending in something testable at small and mid scale.

Builds on (not repeated here): [information_bottleneck_latent_capacity.md](../literature_review/information_bottleneck_latent_capacity.md)
(ICAE, Gist, 500xCompressor, VIB, Broken ELBO, RankMe, superposition),
[learned_kv_context_compression.md](../literature_review/learned_kv_context_compression.md) (Beacon, Compressive
Transformer, CEPE, KVzip), [concept_bottleneck_collapse_mitigation.md](../literature_review/concept_bottleneck_collapse_mitigation.md),
[recurrent_memory_transformers.md](../literature_review/recurrent_memory_transformers.md), the E35 spec (Cramming
probe on our reader) and the E33 page section 14.

## Brief

- **Three ratios, three goals.** E33 compresses positions 6:1 (good for long context), stores 48× less than a dense
  model's key/value cache, but holds 28× more bits than the token ids. Each goal needs a different fix.
- **The bit gap is mostly not the writer's fault.** Even Cramming's per-text optimized vectors use only 3–7 % of
  their raw bits; a bf16 vector holding 1,568 tokens is about 0.8× the size of the text file. Continuous vectors
  buy fewer positions and a readable format, not smaller files. Smaller messages need quantization.
- **The reader sets the ceiling.** Our 30M reader takes in about 85 bits per vector at best (E35 probe), so a
  note larger than about 64–128 bits is unusable room.
- **What works in the literature converges:** a mixed objective (reconstruction + continuation, never one alone),
  a curriculum on the compression ratio, robust latents (noise / small-β VAE / dropout), and spending more
  latents where the text is surprising. Learned encoders reach 16–32 tokens per vector before quality drops.
- **Recommendation:** first measure what today's notes actually use (no training). Then one bet at 30M on the
  cheaper no-loop base: the **rate-matched notebook** — 4× fewer notes (24 tokens each), robust low-dimensional
  notes with a known bit count, a small note-only decoder that must rebuild and continue the page, and questions
  that can read their document only through the notes. Kill it if far lookup at 16k falls below 60 %.

## 1. Four ratios (recap)

Per page (192 new tokens, 32 notes of 64 + 64 numbers, bf16; vocabulary 4,096 = 12 bits per id):

| ratio | today | what it serves |
|---|---|---|
| positions: tokens per note | 192 / 32 = **6** | attention cost O(C·N); the vision's long-context table |
| memory per token vs dense key/value cache | 1,024 / 21.3 = **48× smaller** | memory at 1M–10M tokens |
| bits stored vs token ids | 65,536 / 2,304 = **28× larger** | the size of a message to another agent |
| bits stored vs measured information | 65,536 / 421 = **156× larger** | how much slack a writer could remove |

A fifth ratio is what this note is about: **usable information per note**, i.e. what a reader that did not
write the note can get out of it. Nothing measures it yet (section 6, step 0).

## 2. The limits, with E33's numbers

Four separate ceilings, from the outside in: what the text contains, what the task needs, what the reader can
take in, and what one attention read can pull out.

### 2.1 A note cannot hold more than its page (data processing)
The writer is a fixed function of the page's tokens, so

```
I(notes ; anything) ≤ H(page)
H(page) ≤ 256 tokens × 12 bits = 3,072 bits                 (token ids, upper bound)
H(page) ≈ 192 new tokens × 2.19 bits = 421 bits             (E33's story loss, 1.52 nats per token)
storage = 32 notes × 128 numbers × 16 bits = 65,536 bits
```

At most 4.7 % of the storage can carry information; with the model's estimate of the content, about 0.6 %.

**This slack is partly intrinsic to continuous vectors.** In Cramming (arXiv 2502.13063) the best vectors,
optimized per text against a frozen model, carry 3–7 % of their raw bits: Pythia-160M 396 of 12,288 bits (3.2 %),
Llama-3.2-1B 2,120 of 32,768 (6.5 %), Llama-3.1-8B 4,866 of 65,536 (7.4 %). The 8B vector holding 1,568 tokens is
about 0.8× the text file's size; arithmetic coding with Pythia-160M compresses the same text 6.8×. So a continuous
note will not beat a good text compressor on bits. It wins on positions and on being directly readable.

### 2.2 The task needs far less than the page (information bottleneck)
The notebook must answer what may be asked, not rebuild the page:

```
one fact   ≈ who (1 of ~16: 4 bits) + relation (1 of ~6: 2.6 bits) + value (1 of 48 places: 5.6 bits) ≈ 12 bits
spread page (one fact)        ≈ 12 bits of facts + the story's gist
crowded page (4–6 facts)      ≈ 50–75 bits of facts
"latest" (10 moves, 5 people) ≈ the current state only: 5 × 5.6 ≈ 28 bits, however many moves
```

The notebook is written before the question is known (query-agnostic), so it must keep every askable fact near
losslessly and may drop filler: rate–distortion with a distortion that counts facts, not tokens. Nagle et al.
(arXiv 2407.15504) compute such a bound for prompt compression: existing methods sit far from it, and
variable-rate compression closes much of the gap.

### 2.3 The reader limits what one vector can deliver (Cramming)

| reader | tokens from one vector | bits per vector | source |
|---|---|---|---|
| **our 30M receiver** | 16 tokens exact 38 % of the time | **≈ 85** | E35 probe (4 vectors: 172–239 bits) |
| Pythia-160M | 80 | 396 | Cramming |
| Llama-3.2-1B | 512 | 2,120 | Cramming |
| Llama-3.1-8B | 1,568 | 4,866 | Cramming |

- Capacity tracks bits, not tokens: random words fit 792 tokens but the same ~4,500 bits as book text.
- It grows nearly linearly with the number of vectors (Pythia-160M: 2,016 tokens in 32 vectors).
- It grows with the reader, faster in newer model families.
- These are **per-sample optimized** vectors. A trained writer falls short: the "amortization gap", which the
  paper puts at about two orders of magnitude for existing encoders. GradMem (arXiv 2603.13875), from the same
  group, closes part of it with up to 5 test-time gradient steps on a reconstruction loss (key-value retrieval with
  8 memory vectors: 99.9 % at 32 pairs, against 45.5 % at 16 pairs for a forward-only recurrent memory).

**For E33:** at about 85 usable bits per vector, a note larger than about 64–128 bits is room the 30M reader cannot
use. The design target "about 85–100 bits per note" is the measured reader capacity, and it rises with model size.

### 2.4 One softmax read limits addressing and mixing
- **Addressing.** To put weight 1 − ε on one note among C visible notes, its score must lead by about
  log C + log(1/ε). At 16k, C ≈ 2,720 notes + 256 raw keys: about 8 + 2 = 10 nats. Easy for distinct facts, hard
  for look-alike names.
- **Mixing.** Several facts in one note share one 64-number value (superposition); the read returns a weighted
  average. This is the crowded-page loss (lookup 44 % crowded vs 83 % spread).
- **One key head.** All 9 query heads score the same 64-number key. Theory on attention as associative memory
  (Adler, arXiv 2509.22840) says many small heads reduce interference between stored relations.
- **Diffuse reads.** The notes-budget exam (16 Oct) shows answers looking at dozens of notes: a sharper read would
  let fewer notes carry an answer.

### 2.5 Summary of the room

| quantity | today | ceiling | room |
|---|---|---|---|
| bits stored per page | 65,536 | ≈ 421 (all the text) | 156× |
| bits a page's facts need | — | 12–75 | — |
| bits one note can deliver to the 30M reader | 2,048 stored | ≈ 85 usable | 24× |
| tokens per note | 6 | 16–32 proven for learned encoders (section 3); ≈ 39 at 85 bits | 3–6× |

## 3. What the literature shows works (convergence and tension)

**Practical ratios reached by learned encoders** (all with 0.5B–8B decoders, far larger than ours):
REFRAG (arXiv 2509.01092): 16 tokens per vector good, 32 acceptable, 64 degrades. ARC-Encoder (2510.20535): 4–8×
with a frozen decoder, within 0–3 points of reading the text. UniGist (2509.15763): ratio 4, 60.4 vs 63.9 full
attention. A hierarchical text autoencoder (2512.03643): reconstruction perplexity 1.00 up to 8×, 1.01 at 15×,
1.24 at 29×, 2.56 at 56×. DeepSeek-OCR (2510.18234): 97 % at about 10×, 60 % at 20×. **The proven frontier is
16–32 tokens per vector; E33 sits at 6.**

**Convergence 1: mix the objectives.** LCLM (2606.09659, Qwen3 4B, ~350B tokens per model): reconstruction-only
generalizes poorly, next-token-only lacks fidelity, the mix wins. REFRAG, ARC-Encoder and PCC all use
reconstruction then continuation (next paragraph). The bad-autoencoding paper: good reconstruction alone did not
give good language modelling. **Our own law agrees:** E26's slots rebuilt keys at 66 % yet carried 1 bit of
lookup; E33's next-token-only notes read far facts but nothing checks what they hold for a stranger.

**Convergence 2: curriculum on the ratio or the chunk count.** REFRAG calls its 9-stage curriculum essential;
LCLM finds staged training (adapter → encoder → end-to-end) better than end-to-end from the start.

**Convergence 3: robust latents.** CALM (2510.27688): a 4-token chunk into 128 numbers rebuilt at > 99.9 %, but a
next-vector model only works once the latent is robust: small-β VAE (β = 0.001), per-dimension KL clipping
(without it 71 of 128 dimensions collapsed), latent dropout 0.15, input masking 0.15. Brittle latents decode
small errors into unrelated tokens. Latent size: 32–256 all rebuild, but downstream quality peaks at 128. For us
this is the route to notes that survive quantization and can be read by a model that did not write them.

**Convergence 4: spend latents where the text is surprising.** Byte Latent Transformer (2412.09871): patches cut
where a small model's entropy is high, matching Llama-3 at 4.5 bytes per patch. H-Net (2507.07955): learned
boundaries, 4.8–7.0 bytes per chunk, better bits per byte than BPE. DLCM (2512.24617): learned "concept"
boundaries at 4 tokens per concept, +2.7 % at matched compute (partly unverified). Our pages are filler with
sparse facts: a fixed 32 notes per page spends most notes on filler.

**Convergence 5: predict ahead, in representation space.** LLM-JEPA (2509.14252): adding an embedding-prediction
term to next-token loss helps 1B–8B models (needs two views of the same content, ~2× compute; without the
next-token loss the model outputs nothing). Future Summary Prediction (2510.14751, 8B): a head that predicts a
learned summary of the future, +4.8 ARC-E, +6.3 HumanEval+. CoCoMix (2502.08524): predicting sparse-autoencoder
concepts saves 21.5 % of tokens at 1.38B.

**Tension: frozen or trained reader.** Cramming, ARC-Encoder, xRAG and CompLLM keep the decoder frozen; LCLM
finds full training clearly better at scale. Our E35 found the channel opens only with a frozen receiver (the
receiver otherwise learns the question template and ignores the message: Broken ELBO). Resolution we adopt: a
trained reader is fine **only if the notes are its only route** to the information (the E33f boundary);
otherwise freeze it or add a loss that only the notes can pay.

**Tension: pooling.** "No Mean Feat" (2510.20797) and LCLM: mean pooling beats learned compression tokens in their
setups. Our E30 (weighted average) lost to E31 (latents) on lookup and decoys to 128k. Keep mean-pooled notes as a
cheap control, not as the design.

**Discrete latents.** Large Concept Models: residual VQ, 832 bits per sentence, rebuilds at about 70 % of the
continuous BLEU. Token Assorted (2502.03275): 10 bits per 16 chain-of-thought tokens, task-specific. No
rate–distortion curve for chunk latents exists: our measurement would be new.

## 4. Our own laws that constrain the writer

1. **The objective must pay for the channel** (E22: natural text pays about 0.05 nats per token for far context;
   text checks: with 1 question per document no notebook switched on, with 8 every E31c did).
2. **Reconstruction is not addressability** (E26: 66 % rebuilt, 1 bit of lookup).
3. **A trained receiver with another route ignores the message** (E35 round 1; Broken ELBO).
4. **Warm output projections on any new read** (cold-start note: two zero-init gates in series give only a
   document embedding).
5. **One fact per page is easy, several are not** (crowded lookup 44 % vs spread 83 %).
6. **Seeds decide switch-on at 30M / 1.5B** (E33 2 of 3): every claim needs two runs.
7. **The writer reads bare token vectors** (no context beyond the page; fine for the text checks, a limit on real
   text).
8. **Writer experiments do not need the loop**: E31c (no loop) switches on 3 of 3, trains in 2.4 h vs 7 h, and the
   loop does no task work. Run writer changes on E31c.

## 5. Every candidate, assessed

Effort: **cfg** = a setting that exists; **S** = < 1 day of code; **M** = 1–3 days; **L** = a new component.
Compute: at 30M on the text checks, E31c base, 0.6B tokens ≈ 1 h on 4 × 3090 (≈ 4 GPU-h), 1.5B ≈ 10 GPU-h.
Targets: **C** compression (fewer positions or bits), **S** semantic / meaningful, **R** readable by another
model, **X** crowding / addressing.

### 5.0 Measurement (do first; no training)
| # | what | targets | evidence | effort | compute | verdict |
|---|---|---|---|---|---|---|
| M1 | **Quantization curve**: notes rounded to 8 / 4 / 2 / 1 bits per number at exam time; far lookup vs bits | C | CALM, Physics of LMs (int8 keeps capacity) | S | < 1 GPU-h | **Adopt** |
| M2 | **Effective rank** of notes (RankMe, PCA 95 % variance) per page and across pages | C, S | RankMe; E05 collapse | S | minutes | **Adopt** |
| M3 | **Read-back probe**: a small decoder trained on frozen notes rebuilds the page / answers questions; bits recovered | R, S | ICAE, E33 page 13.2 | M | ≈ 4 GPU-h | **Adopt** |
| M4 | **Amortization gap**: per-page optimized notes (Cramming) against the frozen reader vs the writer's notes | C | Cramming, GradMem | S | ≈ 2 GPU-h | **Adopt** |
| M5 | Notes-budget exam (exists): how many notes an answer needs | C, X | 16 Oct | done | — | in use |

### 5.1 Objective (losses)
| # | what | targets | evidence | effort | compute | verdict |
|---|---|---|---|---|---|---|
| O1 | **Notes-only questions** (E33f boundary: a question reads its document only through the notes) | C, X | our law 1, 3 | cfg (E33f, queued) | 1 run | **Adopt** (running) |
| O2 | **Note-only decoder: rebuild + continue** the page (a 2-layer decoder reading only the page's notes; loss on the page and the next 64 tokens; weight 0.1–0.3) | S, R | LCLM, REFRAG, ARC, PCC; our law 2 | M | +20–30 % step | **Adapt** |
| O3 | **Predict the next page's notes** (JEPA: a small predictor from notes of pages ≤ p to the notes of page p+1, stop-grad target, cosine loss + SIGReg anti-collapse) | S | LLM-JEPA, LeJEPA, Future Summary Prediction | S–M | +10 % | **Adapt** (after O2) |
| O4 | **Paraphrase invariance**: the same page facts rendered with two sentence shapes and different filler; contrastive loss pulls their notes together | S, R | LLM-JEPA two views; our unseen-shape exam (E33 64 → 30 %, dense 1–4 %) | M (data + loss) | +10 % | **Adapt** |
| O5 | **Rate penalty**: small-β VAE / per-dimension KL clip on notes (CALM recipe) | C, R | CALM, VIB | S | ≈ 0 | **Adapt** (with B3) |
| O6 | **Distillation from dense**: the reader with notes matches the dense model's next-token distribution on far-question answers (KL), where dense can read (≤ 4k) | X | KV-Distill, CompLLM | M | + teacher pass | **Watch** (dense is weak at far facts too) |
| O7 | Reconstruction only (prefix autoencoder alone) | — | E26; LCLM | cfg (E26 head) | — | **Reject** as the sole loss |

### 5.2 Writer architecture
| # | what | targets | evidence | effort | compute | verdict |
|---|---|---|---|---|---|---|
| A1 | **Fewer notes per page** (`lm_latents` 16 / 8; 12 / 24 tokens per note) | C | REFRAG 16–32 frontier; sweep row | cfg | 1 run each | **Adopt** (axis of the bet) |
| A2 | **Longer pages** at the same notes per token (128 / 512) | C | LCLM: encoder window 256 ≫ 16, 1024 slightly better | cfg | 1 run each | sweep |
| A3 | **Null note** (`lm_null_latent`): filler can go nowhere | C, X | Slot Attention | cfg | 1 run | **Adopt** (cheap) |
| A4 | **Writer reads the prelude's states** (context beyond the page) | S, X | sweep row; LCLM (encoder from a strong model) | S | 1 run | **Adapt** |
| A5 | **Low-dimensional robust notes**: key 32 + value 32 numbers, latent dropout 0.15 + Gaussian noise, then **FSQ** (finite scalar quantization, 3 bits per number) → 192 bits per note, exactly countable | C, R | CALM; FSQ; Token Assorted | S–M | ≈ 0 | **Adapt** (core of the bet) |
| A6 | **2–3 key heads on the note read** (keys for "who", "where", "when") | X | Adler 2509.22840; sweep row | M (per-layer kv heads) | 1 run | **Adapt** |
| A7 | **Variable-rate notes**: notes placed by surprise — the reader's own detached per-token loss (or an H-Net router) sets where latents look; budget fixed per document, not per page | C, X | BLT, H-Net, DLCM, AdmTree, Nagle | M–L | +5 % | **Adapt** (stage 2) |
| A8 | **Two-level notebook** (E32): chapter notes written from page notes, 1,024 tokens per coarse note | C | Cramming direction; hierarchical AE 2512.03643 | L | — | **Watch** (after the ratio ladder) |
| A9 | **Overwritable state notes** (entity slots a page can update) | C (latest needs 28 bits, not 10 moves) | EntNet; LeCun's persistent memory | L | — | **Watch** |
| A10 | Mean-pooled notes | C | No Mean Feat | cfg (E30) | 1 run | control only |

### 5.3 Training protocol
| # | what | targets | evidence | effort | compute | verdict |
|---|---|---|---|---|---|---|
| P1 | **Ratio curriculum**: start at 32 notes per page, drop to 16 → 8 during training (latents merged or the stride grown) | C | REFRAG (essential), LCLM staged | S | ≈ 0 | **Adapt** |
| P2 | **Raw-window dropout**: sometimes shrink the 256-token raw window, so recent facts must live in the notes | C, X | notes-budget: late facts do not cross | S | ≈ 0 | **Adapt** |
| P3 | Frozen-reader phase (writer trains against a fixed reader) | R | Cramming, ARC, E35 | cfg (E35 flag) | 1 run | **Watch** (O1 should make it unnecessary) |
| P4 | Test-time writing (a few gradient steps on the notes, GradMem) | C | GradMem | M | per-document cost | **Watch** (inference-time, later) |

### 5.4 Synthetic data
| # | what | targets | effort | verdict |
|---|---|---|---|---|
| D1 | **Crowded pages in training** (a share of documents with 4–6 facts per page) | X | S (`data/text_world.py`) | **Adopt** |
| D2 | **Paraphrase pairs** (for O4): same facts, other sentence shapes and filler | S, R | S | **Adapt** |
| D3 | Gist and count questions ("how many times", "what is the story about") so notes keep more than an index | S | M | **Watch** |
| D4 | Late questions about the last page (with P2) | C | S | **Adapt** |

## 6. Recommended plan

### Step 0 — measure today's notes (no training, about 1 GPU-day, laptop or one server GPU)
On E31c run 2 and E33 run 1 (both switched on), at 4k and 16k:
1. **M1** quantization curve: the smallest bits per number that keep far lookup within 5 points.
2. **M2** effective rank of the 64-number keys and values.
3. **M4** amortization gap on 50 pages: Cramming-optimized notes for the frozen reader vs the writer's notes.
4. **M3** read-back probe (small decoder on frozen notes).

**Decision rule:** if notes survive 4 bits and their rank is ≤ 16, the room is real and A5 is safe; if they need
16 bits, the bet must include O5 (robustness) from step 0 of training.

### Step 1 — the bet at 30M: the rate-matched notebook
One coherent change set on the **E31c** base (no loop), 8-questions recipe, 1.5B tokens, two seeds:

| part | setting | why |
|---|---|---|
| fewer notes | 8 per 256-token page (24 new tokens per note; today 6) | inside the proven 16–32 frontier, 4× fewer positions |
| robust, countable notes | key 32 + value 32 numbers, latent dropout 0.15 + noise, FSQ 3 bits → **192 bits per note** | CALM; the bits become a fact, not an estimate |
| a loss only the notes can pay | E33f boundary (questions read only through notes) | laws 1 and 3 |
| meaningful notes | note-only decoder rebuilds the page and continues 64 tokens (weight 0.2) | the reconstruction + continuation mix |
| null note | `lm_null_latent` | filler has somewhere to go |

What it would give per page: 8 × 192 = 1,536 bits for 192 tokens = **8 bits per token: 1.5× smaller than the
token ids** (today 28× larger) and 4× fewer positions, at about 2.3× the reader's measured per-vector capacity
(192 vs ≈ 85 bits), so the reader, not the format, would be the limit.

- **Pass:** far lookup and keyed at 4k / 16k within 10 points of E31c's typical runs (runs 1 and 3: lookup
  75–77 / 92–93 %, keyed 70–72 / 91 %) on both seeds; read-back probe better than today's notes; story loss ≤ 1.60
  (E31c 1.564).
- **Kill:** far lookup at 16k below 60 % on both seeds → 24 tokens per note at 192 bits is past what a 30M
  writer can learn; fall back to 12 tokens per note.
- **Diagnostics:** FSQ code usage (dead codes), effective rank, notes-budget (notes per answer), switch-on
  token count.
- **Then dissect** (only after a pass): drop one part at a time (FSQ, decoder, boundary) to see which carried it.
- **Cost:** 2 seeds × ≈ 10 GPU-h + controls (E31c at 8 notes, nothing else changed) ≈ 30–40 GPU-h, one day on
  Polonez in bursts.

### Step 2 — mid scale (100M main tier) and the semantic objectives
- The step-1 winner at 100M (and today's E31c at 100M as the reference), to see whether the ratios hold with a
  stronger reader (Cramming: capacity per vector rises with model size).
- Add **O3** (predict the next page's notes) and **O4** (paraphrase invariance) as separate arms, read on the
  unseen-shape exam and the read-back probe.
- Add **A7** variable-rate placement if crowded lookup is still the weak point.
- Cost: about 120 GPU-h per 100M run (tier cap); 3 runs ≈ 1 week of one server.

### What not to do now
- Reconstruction as the only loss (O7), mean-pooled notes as the design (A10), test-time writing (P4), the
  two-level notebook (A8) before the ratio ladder shows where one level breaks.

## Uncertainty
- All learned-encoder numbers come from 0.5B–8B decoders; at 30M the reader's capacity (85 bits per vector) is
  4–5× below Pythia-160M's, so the reachable ratio may be lower.
- The per-fact bit counts in 2.2 assume uniform choices in the generator; real text is not like that.
- LCLM and DLCM numbers were read from abstracts and partial tables; GradMem is a preprint.

## Sources
Cramming [2502.13063](https://arxiv.org/abs/2502.13063) ·
one-step generation [2505.21189](https://arxiv.org/abs/2505.21189) ·
GradMem [2603.13875](https://arxiv.org/abs/2603.13875) ·
LCLM [2606.09659](https://arxiv.org/abs/2606.09659) ·
REFRAG [2509.01092](https://arxiv.org/abs/2509.01092) ·
ARC-Encoder [2510.20535](https://arxiv.org/abs/2510.20535) ·
UniGist [2509.15763](https://arxiv.org/abs/2509.15763) ·
CompLLM [2509.19228](https://arxiv.org/abs/2509.19228) ·
KV-Distill [2503.10337](https://arxiv.org/abs/2503.10337) ·
No Mean Feat [2510.20797](https://arxiv.org/abs/2510.20797) ·
optical compression as autoencoding [2512.03643](https://arxiv.org/abs/2512.03643) ·
CALM [2510.27688](https://arxiv.org/abs/2510.27688) ·
BLT [2412.09871](https://arxiv.org/abs/2412.09871) ·
H-Net [2507.07955](https://arxiv.org/abs/2507.07955) ·
DLCM [2512.24617](https://arxiv.org/abs/2512.24617) ·
AdmTree [2512.04550](https://arxiv.org/abs/2512.04550) ·
DeepSeek-OCR [2510.18234](https://arxiv.org/abs/2510.18234) ·
Large Concept Models [2412.08821](https://arxiv.org/abs/2412.08821) ·
CoCoMix [2502.08524](https://arxiv.org/abs/2502.08524) ·
Token Assorted [2502.03275](https://arxiv.org/abs/2502.03275) ·
LLM-JEPA [2509.14252](https://arxiv.org/abs/2509.14252) ·
Future Summary Prediction [2510.14751](https://arxiv.org/abs/2510.14751) ·
prompt-compression rate–distortion [2407.15504](https://arxiv.org/abs/2407.15504) ·
Physics of LMs 3.3 [2404.05405](https://arxiv.org/abs/2404.05405) ·
attention as associative memory [2509.22840](https://arxiv.org/abs/2509.22840) ·
LLMZip [2306.04050](https://arxiv.org/abs/2306.04050).
Ours: E22 root cause, E26, E35 spec (Cramming probe, round 1), E33f spec (notes-budget exam), the cold-start note,
the text board.
