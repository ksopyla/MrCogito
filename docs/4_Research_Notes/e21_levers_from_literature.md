# E21 levers from the literature — ranked Adapt bets and a 32k protocol

Undated mutable note (research ideas). Does **not** record run numbers; E25 DNA
walls stay in the ledger. Does **not** freeze an experiment spec — that is a
later `experiment-design` step. Companion reviews:

- [`../literature_review/information_bottleneck_latent_capacity.md`](../literature_review/information_bottleneck_latent_capacity.md)
- [`../literature_review/learned_kv_context_compression.md`](../literature_review/learned_kv_context_compression.md)
- [`../literature_review/latent_set_refinement.md`](../literature_review/latent_set_refinement.md)
- exams: [`../literature_review/synthetic_capability_exams.md`](../literature_review/synthetic_capability_exams.md)
- BAPO theory: [`../literature_review/reasoning_bandwidth_information_flow.md`](../literature_review/reasoning_bandwidth_information_flow.md)
- writable memory (Coconut / Huginn / Ouro / Infini):
  [`../literature_review/recurrent_memory_transformers.md`](../literature_review/recurrent_memory_transformers.md)

E21 itself (strategy-branch spec, implemented as config on `perceiver_ar`): a
**message boundary** severs every local path (SWA, n-grams); the global read
sees the prefix only as `KVCompressor` slots (one per `r` tokens, in the
read's K/V space). The original 32k LM bet is in
`docs/experiments_specs/` on `cursor/strategy-sota-review-2026-09-e212`.
The cheap DNA instrument is E25
([`../experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`](../experiments_specs/done_success/E25_e21_bapo_capability_ladder.md)).

---

## 1. Four questions, with literature rather than vibes

### Q1. Is the latent set capacity-starved — count vs width?

**Both, but they starve different tasks. Count first for addressing; width
is usually unspent or overprovisioned.**

- **Reconstruction of unique text is count-starved below ~4×.** ICAE
  (ICLR 2024, [arXiv:2307.06945](https://arxiv.org/abs/2307.06945)):
  512→128 works; `k=64`/`32` fail lossless AE; random text BLEU 0.2 at
  4×. Deng Fine-KV ([arXiv:2412.17483](https://arxiv.org/abs/2412.17483)):
  synthetic recall 93.9 → 40.6 at 4× → 13.8 at 16× → 11.9 at 32×.
- **Instruction gist is count-saturated at 1–few tokens.** Gisting
  ([arXiv:2304.08467](https://arxiv.org/abs/2304.08467)): `k=1` ≈ `k=5`;
  `k=10` can hurt. AutoCompressor `κ=50` best among `{20,50,70,100}`.
- **Per-token KV width is overprovisioned.** MLA
  ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434)) beats MHA at
  ~GQA-2.25 cache. 500xCompressor: at 500→1, **KV-of-slots >> embeddings**;
  16→4 slots is flat, 4→1 drops — bits/slot, not extra slots.
- **Allocated width ≠ used bits.** RankMe ([arXiv:2210.02885](https://arxiv.org/abs/2210.02885));
  Broken ELBO ([arXiv:1711.00464](https://arxiv.org/abs/1711.00464)):
  identical ELBO, rate 0.007–11 nats. Superposition: `D` is a packing
  budget, not `D` independent channels.
- **BAPO splits the two axes.** INDEX is `(a=0, b=1)` — one addressable
  token, not a wide summary. MAJORITY needs `a = Ω(log n)` *bits* with
  `b=0`. MATCH3 needs `a·b = Ω(n)` — **count of independently addressable
  items**, not width of a pooled gist
  ([arXiv:2505.08140](https://arxiv.org/abs/2505.08140)).
- **No published `(K, D)` grid** at matched compute. Do not "try wider
  slots" as the research program.

E25 is consistent: r=16 frozen mean carries INDEX (gist of a marked span);
MATCH dies under pooling even when r drops toward 4 at 1280; identity
`r=1` restores MATCH. That is count/addressability, not H=128 vs 256
(width helped some rungs as optimization, not as the MATCH wall).

**Verdict:** E21 at `r=16` is **count-starved for MATCH-class / exact
rehearsal**, **count-adequate for INDEX-class gist**, and **not
width-starved** until RankMe of the slot matrix saturates. Widening
`head_dim` without a rate target is a safe retread.

### Q2. Does a reasoning / refinement stage over latents pay off, and in which form?

**Not as one extra exclusive attend, and not as one extra unique global
layer. Yes as many weight-tied loops over the set with per-slot targets,
and only after the slots already carry addressable content.**

| Form | Pays on | Dies on | Evidence |
|---|---|---|---|
| Extra attend over frozen K/V (pause / E21 extra hop) | Extractive QA (SQuAD +18 EM) | GSM8k ~+1; MATCH/SELECT at 1280 | Goyal [arXiv:2310.02226](https://arxiv.org/abs/2310.02226); E25 extra hop is hops-only |
| Second unique global layer | PPL / closed-book (memorization) | Multi-hop composition; E25 SELECT | Saunshi [arXiv:2502.17416](https://arxiv.org/abs/2502.17416); E25 `global_layers=2` |
| Rewrite slot K/V (GRU / update-kv) | Object/set binding (`T=1` << `T=3`) | Untested as an LM memory | Slot Attention [arXiv:2006.15055](https://arxiv.org/abs/2006.15055) |
| Many tied loops + per-slot gold CE | Math / p-hop / GSM8K 14.6%→70.0% as `R=2→6` | LM PPL vs iso-FLOP unique depth | LOTUS [arXiv:2606.31779](https://arxiv.org/abs/2606.31779); Saunshi |
| Sequential latent CoT, answer-only | Search/planning at small scale | GSM8K; collapses at 5 latents | Coconut; SIM-CoT rescue [arXiv:2509.20317](https://arxiv.org/abs/2509.20317) |

LOTUS is the existence proof that a **set** (not a Coconut chain) plus
**loops** plus **parallel step CE** matches explicit CoT at 3B. Infer
`R=1` on a `R=6` ckpt is 22.7% — one reasoning step is not the trained
computation. `c=1` vs `c=25` (49.7 vs 70.0) is width-of-block for *step
tokens*, not KV-head width.

Pfau ([arXiv:2404.15758](https://arxiv.org/abs/2404.15758)) + BAPO Thm 8:
hidden delay only works with dense supervision of the delay, and the tape
must be **re-readable**. A Markov `z → z'` extra hop does not satisfy
Thm 8.

**Verdict on "one reasoning step on top of latents": Reject as the
architectural bet.** It is pause tokens / E25 extra hop. Keep extra hop as
a hops-composition tool. The Adapt is LOTUS-shaped loops **after** E21's
exclusive channel is load-bearing on INDEX *and* MATCH at the target
length.

### Q3. What objectives force semantically rich latents vs decoder shortcut?

Convergent list, two families:

**Memory (make the slots hold the prefix):**
- Exclusive raw path cut (E21 boundary; TSDAE decoder confinement
  [arXiv:2104.06979](https://arxiv.org/abs/2104.06979); E02 prefix→suffix).
- Fine-grained autoencoding of the prefix from the slots with a *weak*
  decoder (Deng; ICAE AE+LM > either alone; CT attn-recon 0.973 vs
  conv+BPTT 0.996 vs mean-pool 0.982).
- Up-weight tokens whose target is determined by far content (Deng
  segment-wise importance; NExtLong hard-negative packing
  [arXiv:2501.12766](https://arxiv.org/abs/2501.12766); this repo's E23
  ×8 far-repeat).
- Freeze the pooler until the aux exists (CT; Funnel Top-Attn 75.8 vs
  mean 83.5; E25 learned `u`/`delta` = 0 bits).

**Reasoning (make the slots hold steps):**
- Per-slot CE onto gold steps (LOTUS `L_step`, SIM-CoT). Answer-only is
  63.3 vs 70.0.
- Distillation that **excludes the last answer-copy step** (CODI; keep
  last step 31.7 vs 43.7).
- Generative reconstruction of steps, not L2 onto frozen encodings
  (Chen et al. [arXiv:2606.20075](https://arxiv.org/abs/2606.20075)).

**Known shortcut objectives:** next-token CE with a raw path open
(Broken ELBO; E05; E18 arm C); mean-pool of reasoning embeddings
(CoLaR `-MP`); inference-only pauses; swapped-message that still beats
`none` (StateBridge failure mode, already E21 K3).

### Q4. Compression at 8k–32k, and which evals detect long-range use?

**Degrades with length on exact / high-diffusion tasks; can hold on RAG /
LongQA / NIAH.** Beacon matches Full-FT on LongBench-32k and NIAH-128k
after ≤20k train; AutoCompressor/ICAE collapse at 32k on the same table.
Deng Fine-KV at 16× (E21's `r`) is RAG 55.4 vs 61.8 but synthetic 13.8 vs
93.9. Infini-attention **in-place overwrite** degrades with more compress
steps (HF blog); concat FIFO (Beacon/CT) scales until slot count hits the
window.

**Protocols that catch local shortcutting:**

| Instrument | Catches | Misses |
|---|---|---|
| Vanilla NIAH | Nothing at 32k (saturates) | Everything HELMET cares about |
| PPL last-256 (CEPE) | Fluency | INDEX/MATCH through slots |
| LongBench-v1 ROUGE | Fuzzy use | Exact rehearsal |
| RULER MK-UUID / VT / CWE at 4k **and** 32k | Multi-key, chains, aggregation | Need a dense S0 |
| HELMET RAG + numbered ICL + synthetic recall | Real vs synthetic Spearman <0.8 | English-heavy |
| LongBench v2 no-context ≈ 25% | Parametric cheat | Still English MCQ |
| Retrieval-head knockout | Causal use of lookup heads | Needs a dense/E18 control |
| `message_override` none / swapped / raw | Load-bearing + content-specific | Already in E21 spec |
| Query-agnostic 2nd query after freeze | SnapKV-style cheats | Compactor/KVzip/Beacon 3-turn |
| DNA INDEX vs MATCH vs SELECT vs hops at matched seq | BAPO class walls | Not language |

Levy et al. ([arXiv:2407.00402](https://arxiv.org/abs/2407.00402)): if
all you need is retrieval of one span, it is not a long-context claim.
100-LongBench ([arXiv:2505.19293](https://arxiv.org/abs/2505.19293)):
disentangle task skill vs extra-context skill — score 4k and 32k on the
same prize.

---

## 2. Ranked levers for E21 (bold Adapt compositions)

One coherent bet per lever. Kill criteria are diagnostic, not a spec.
A/B of LR / width / SVD-init is explicitly out.

### Lever 1 — Fine-grained exclusive slots + weak-decoder prefix AE
**Claim:** Interleaved (Beacon / Fine-KV) exclusive slots, with frozen
mean-pool *or* a compressor trained against a **weak reconstruction head
on the prefix**, carry MATCH-class content at r=16 that LM-CE-trained
`u`/`delta` destroy.
**Connects:** Deng Fine-AE +52.7% relative synthetic recall at 4×;
Beacon interleave 40.5 vs end-append 35.2; CT attn-recon vs conv+BPTT;
E25 learned-pool wipe; Funnel mean > Top-Attn.
**Surprising if it works:** a *local* AE on 16-token blocks, not more
global CE, is what makes the exclusive channel content-addressable.
**Kill:** MATCH information_flow still < 0.05 at the DNA length that
identity-r=1 already passes, after the AE saturates. Then r=16 cannot
hold MATCH even with the published repair.
**Reject-as-substitute:** "just train `u` longer"; "cross-attn without
FFN"; SVD/pretrained token init.

### Lever 2 — Count tracks BAPO class; do not spend the budget on width
**Claim:** INDEX stays at r=16 frozen mean; MATCH/SELECT need either
`r→1` identity keys or a *sparse set of raw anchors* (type-marks) plus
pooled gist for the rest — a hybrid `(a, b)` rather than wider slots.
**Connects:** BAPO INDEX `(0,1)` vs MATCH3 `a·b=Ω(n)`; ICAE count
starvation; MLA width overprovision; 500x 16≈4 then 4→1 drop; retrieval
heads <5% (Wu et al.).
**Surprising if it works:** a handful of raw identity keys (the `b`
channel) plus pooled gist (`a`) beats lowering r uniformly.
**Kill:** hybrid anchors leak the exam (`e18_local` analogue) *or* MATCH
still dies when anchors are ablated.
**Reject-as-substitute:** sweep `head_dim` / H at fixed r as the main
experiment.

### Lever 3 — Force the objective to pay for surprise and long-range tokens
**Claim:** Exclusive mask is necessary but not sufficient (E22). Add
(i) reconstruction of high-entropy / marked spans from slots, (ii)
up-weight of tokens whose label is determined by content before P, (iii)
hard-negative packing so 32k rows actually contain prefix→suffix
dependencies.
**Connects:** Deng "lost if surprise" / "lost along the way"; NExtLong;
E02 prefix→suffix as the only de-collapsing CE; E23 far-repeat; E21 K3
swapped-message.
**Surprising if it works:** natural-text CE on boundary rows, *with*
importance + AE, matches DNA MATCH without a synthetic retrieval mix.
**Kill:** `real − none` on early receiver stays < 0.10 nats while arm U
(r=1) is ≥ 0.30 (E21 K1) — compression, not objective, is the failure.

### Lever 4 — Looped set refinement is E19, not a 1-step patch on E21
**Claim:** Once levers 1–3 make slots load-bearing, `R≥4` weight-tied
self-attn over the slot set with per-slot targets lifts hops / MATCH3 /
VT; `R=1` extra exclusive attend does not.
**Connects:** LOTUS `R=2→6`; Saunshi unique vs tied depth; Slot Attention
`T=1` vs `T=3`; Goyal pause vs Quiet-STaR multi-token thoughts; E25 extra
hop / glob=2.
**Surprising if it works:** test-time `R` monotonically raises
information_flow on a BAPO-hard DNA rung that single-pass slots cannot
solve (BAPO Thm 8 in continuous form).
**Kill:** flat flow vs `R`; or RankMe collapse across loops (Ouro/Infini
instability). Then do not ship unstabilized recurrence.
**Reject-as-substitute:** `global_layers=2`; `message_extra_slot_attends=1`
as "the reasoner."

### Not in the top four (Watch)
- MLA-style width compression *of slots* (orthogonal, after count is
  right).
- MoR / ThoughtBubbles (PPL axis).
- Query-aware eviction (SnapKV) — wrong product for a message.
- Infini-attention in-place overwrite — published degrade-with-compress.
- Coconut curriculum at ≥1B — forgetting; LOTUS/CODI avoid it.

---

## 3. Adopt / Adapt / Watch / Reject (sources that move E21)

| Source | Verdict | One-line why |
|---|---|---|
| E21 exclusive boundary (already in code) | **Adopt** | Only published-compatible way to make CE supervise `a`; E02/E05/E18 agree. |
| Frozen mean-pool at init / identity_slots | **Adopt** (until Lever 1 AE exists) | CT mean-pool > LM-trained conv; E25 learned pool = 0 bits. |
| `message_override` none/swapped/raw | **Adopt** | HELMET/LongBench-v2 no-context + StateBridge swap, already specified. |
| Deng Fine-KV + Fine-AE [2412.17483] | **Adapt** | Interleave + weak prefix AE is the missing compressor objective; do not clone their Llama SFT stack. |
| Activation Beacon [2401.03462] | **Adapt** | Interleave placement + concat FIFO + variable `α` at train; skip their ChatGPT NIAH score as a gate. |
| Compressive Transformer attn-recon [1911.05507] | **Adapt** | Local reconstruction aux, not BPTT through the pooler. |
| ICAE count ablations [2307.06945] | **Adapt** (as a capacity probe) | Use 4× as the reconstruction floor; do not adopt LoRA-on-Llama. |
| BAPO [2505.08140] | **Adopt** (lens + DNA classes) | Count vs bits vs raw `b`; INDEX ≠ MATCH. |
| HELMET [2410.02694] | **Adapt** | Length ladder + refuse NIAH-as-success; closed-alphabet the categories. |
| LongBench v2 no-context [2412.15204] | **Adapt** | Chance-floor arm. |
| Retrieval heads [2404.15574] / DuoAttention | **Adapt** | Head-heterogeneous compression / knockout probe. |
| KVzip / Compactor (query-agnostic) | **Watch** | Training-free eviction, not a trained exclusive message; steal reconstruction scoring. |
| LOTUS [2606.31779] | **Adapt** (E19, gated on load-bearing slots) | Loops + parallel step CE over a set; not a 1-step E21 patch. |
| Saunshi looped TF [2502.17416] | **Adapt** (same gate) | Tied depth ≠ unique depth. |
| SIM-CoT / CODI | **Adapt** (methodology) | Per-step pin; block last-step copy. Not Coconut's single vector. |
| Pause tokens [2310.02226] | **Reject** as E21 reasoner | Published extra-attend; QA-selective; E25 already ran the analogue. |
| SnapKV / H2O / LongLLMLingua | **Reject** as E21 compressor | Query-aware; fails reuse; not a message. |
| Infini-attention in-place | **Reject** | HF: quality falls as compress-count grows. |
| MLA [2405.04434] | **Watch** | Width axis, after count is right. |
| Gisting `k=1` [2304.08467] | **Reject** as 32k recipe | Instruction-scale; extra slots overfit. |
| Funnel / Hourglass | **Watch** | Hierarchical count reduction *keeps* full-res tokens; a pure pool is harder. |
| MoR / ThoughtBubbles | **Watch** | PPL, not exclusive memory. |
| Coconut sequential CoT | **Reject** as E21 module | Chain not set; collapses without step CE; already reviewed. |
| SVD / pretrained token init, cross-attn-without-FFN, extra unique layer | **Reject** | Project research stance + Saunshi + ledger retreads. |

---

## 4. What a 32k-capable evaluation protocol should look like

Goal: detect **long-range use of the compressed exclusive prefix**, not
local SWA, not parametric memory, not query-aware eviction, not "PPL went
down."

**Fixed geometry (all arms, all lengths).** Same tokenizer, same
`message_boundary` token, same `r`, same packing. Lengths `{4k, 8k, 32k}`
minimum; 16k optional. Report **4k and 32k on the same prize**
(100-LongBench disentangle). Dense S0 and uncompressed E18 (r=1 / raw)
on every rung; skip E21 if dense < 75% (E25 K1).

**Tier 0 — leak and content (cheap, every ckpt).**
- `message_override ∈ {real, none, swapped, raw}` on early receiver CE
  and on the DNA/RULER prize. Kill if swapped > none (steering) or if
  `e18_local` / keep-SWA solves the prize (boundary leak).
- Exact-floor DNA: INDEX, MATCH (r=1 identity ceiling), SELECT, ordered
  hops. Information_flow vs dense and vs E18. This is BAPO `(a,b)` in
  bits, already built.

**Tier 1 — synthetic long-range at 4k and 32k (must not be NIAH-only).**
- RULER-lite already in the perceiver_ar eval layer: passkey (low
  diffusion — expect Fine-KV-like survival), **multikey UUID**,
  **variable tracking**, FWE. Gate on MK/VT, not passkey.
- Query-agnostic reuse: freeze slots, ask a *second* question (Beacon
  3-turn / Compactor). SnapKV-style methods fail here; E21 must not.

**Tier 2 — application-ish, still controlled.**
- HELMET-shaped: RAG with distractors (gold depth sweep), numbered-label
  ICL, synthetic JSON-KV. Prefer Glyph/closed vocab over English essays
  until a dense 32k control exists.
- LongBench v2-style no-context arm on any MCQ.

**Tier 3 — causal use.**
- Slot permutation / shuffle (existing Δshuffle).
- Retrieval-head analogue: zero the exclusive read vs zero random
  local heads.
- Rate–distortion: recovered bits vs bytes/token across `r ∈ {1,4,8,16}`
  (Delétang unit; E21 spec's scientific deliverable).

**Do not use as the 32k success criterion:** vanilla NIAH, PPL-last-256,
LongBench-v1 ROUGE, MMLU, GSM8K, STS-B. Those can all move while MATCH
through slots is dead (Deng Table 1 vs Table 2; HELMET Spearman; E18
arm C).

**Curriculum to 32k.** Train with a mixture of ratios (Beacon `α ~
Unif{2..32}`) and with packed rows that contain *actual* prefix→suffix
dependencies (NExtLong distractors; E21 q-fraction is not enough if the
document has no long-range signal). Long SFT dumped into the mix can
hurt (ProLong).

---

## 5. Explicitly not recommended (safe retreads the stance forbids)

- Cross-attention without FFN; SVD / PCA / truncated-factor init from a
  pretrained embedding; "small token embeddings warm-started from a
  pretrained model."
- Optimizer / LR / width sweeps as the E21 program.
- One extra exclusive attend or `global_layers=2` billed as latent
  reasoning.
- Learned pooling under plain LM CE without a reconstruction aux.
- Query-aware KV eviction as the message.
- In-place Infini-style overwrite of a fixed matrix.
- Scoring 32k on NIAH / last-256 PPL and calling the channel load-bearing.

---

## 6. What is still open

- No matched `(K, D)` grid, and no matched Beacon/ICAE/Fine-KV vs frozen
  mean vs E21 exclusive slots on **the same** RULER MK/VT + DNA INDEX/MATCH
  suite at 4k and 32k.
- Whether interleaved r=16 mean-pool (Beacon placement) beats E21's
  complete-block / in-place replace on MATCH — Beacon's drop was on
  *learned* beacons, not frozen mean.
- Whether a handful of identity anchors plus pooled gist is a stable
  `(a,b)` or just a leak.
- LOTUS loops over *reasoning pads*, not over *compressed prefix KV*.
  Transfer to E21 slots is a hypothesis, not a result.
- Measured `I(prefix; slots)` (Alemi `R`) vs nominal `K·D` — RankMe is
  the cheap proxy, not the bit count.

A separate agent is diagnosing recent E21/E18 vs dense runs; this note
does not reinterpret those numbers. When that diagnosis lands, the kill
criteria above should be checked against it, not rewritten to fit.
