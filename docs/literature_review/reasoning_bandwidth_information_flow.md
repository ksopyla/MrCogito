# Reasoning bandwidth / information flow — the latent-channel-capacity axis

Reviews of work that frames transformer reasoning limits as a problem of
**information flow / communication bandwidth** through the model's latent
channels (residual stream, attention heads, compressed summaries), rather than
as a parameter-count or depth problem. This is the theoretical lens that says
*why* a concept bottleneck is the right object to engineer and measure.

It is the natural home for: the **BAPO** model and the expressivity-theory
strand it rests on (Edelman, Hahn, Merrill–Sabharwal, Sanford, the "globality
barrier"), plus the latent/hidden-reasoning barrier results (Pfau et al.).

Related reviews already in the repo (do not duplicate):
- the *engineering* counterpart — frozen vs writable concept memory, including
  **Coconut**, **Huginn**, **Ouro**, **RMT**, **Block-Recurrent** — lives in
  [`recurrent_memory_transformers.md`](recurrent_memory_transformers.md);
- collapse / bypass fixes (TSDAE word-dropout, VICReg, JEPA) live in
  [`concept_bottleneck_collapse_mitigation.md`](concept_bottleneck_collapse_mitigation.md).

---

## Lost in Transmission: When and Why LLMs Fail to Reason Globally (BAPO)
NeurIPS 2025 (Spotlight) · **arXiv:2505.08140** ·
[OpenReview MaJ3ASZ0NI](https://openreview.net/forum?id=MaJ3ASZ0NI) ·
[NeurIPS poster 118455](https://neurips.cc/virtual/2025/poster/118455) ·
[proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/cabf611498431ad89a85ace75f790d93-Paper-Conference.pdf) ·
**code: [github.com/microsoft/bapo](https://github.com/microsoft/bapo)** ·
Tobias Schnabel\*, Kiran Tomlinson\* (equal), Adith Swaminathan, Jennifer Neville
(Microsoft Research / Netflix).

### TL;DR
LLMs fail at *global* reasoning (over large parts of the input) not for lack of
parameters, but because the **effective communication bandwidth** from early
tokens to the answer-emitting token is a small constant. The paper formalizes
this as the **Bounded Attention Prefix Oracle (BAPO)** model, proves several
reasoning problems are **BAPO-hard** (need bandwidth that grows with input
size), confirms empirically that GPT-4o/Claude/Gemini fail exactly those (and
**scale does not rescue them**), and proves that **chain-of-thought turns any
BAPO-hard problem BAPO-easy** at constant per-step bandwidth. Crucially, it
concedes it cannot *measure* effective bandwidth inside a model, and names
"optimize for low bandwidth" as an open training objective.

### The problem the authors want to solve
Explain *why* strong LLMs collapse on tasks that require integrating many input
tokens (graph reachability, majority counting, multi-hop matching), in a way
that is principled and predictive rather than anecdotal. Existing expressivity
theory (transformers ⊆ TC⁰, Merrill–Sabharwal) makes MAJORITY look *trivially*
expressible, yet models fail it — so expressive power is the wrong axis. The
paper's bet: the real axis is **information flow across the causal boundary**
between the prefix (early tokens) and the suffix stream that produces the answer.

### The solution (intuition)
Under causal attention, each token's residual stream is processed independently
of future tokens. To answer a global question, prefix information must "cross"
into the suffix stream through one of two channels:
1. **prefix bandwidth `a`** (in **bits**) — attention to *intermediate* prefix
   outputs, i.e. a compressed summary of the prefix;
2. **attention bandwidth `b`** (in **tokens**) — attention to *raw* prefix tokens.

An **(a,b)-BAPO** is the most generous model that still has these two caps: the
prefix oracle `f`, attention selector `g`, and suffix solver `h` get *unbounded*
compute, but the *for-all-prefix/suffix-splits* quantifier (and a worst-case
subset delivery for `g`) is what makes the model hard. A problem is **BAPO-easy**
if constant `(a,b)` suffices, **BAPO-hard** otherwise (bandwidth must grow with
`n`), and **BAPO-Σ-hard** if it must grow with vocabulary size.

### Detailed model / main results
- **BAPO-easy (Thms 1–2,5):** INDEX `(0,1)`, EQUALITY/DISJOINTNESS `(1,1)`,
  MATCH2 `(0,1)`, and any regular language `L` with state complexity `|Q|` via a
  `(⌈log₂|Q|⌉,0)`-BAPO (prefix oracle ships the DFA state after the prefix).
- **BAPO-hard (Thms 3–7):** REACHABILITY needs `(a,b)` growing with graph size;
  MAJORITY needs `a = Ω(log n)` *and* `b = Ω(n^{1−ϵ})` (tight: `⌈log₂ n⌉` bits
  suffice with `b=0`); MATCH3 needs `a·b = Ω(n)`; UNIQUE/SETDIFF are
  BAPO-Σ-hard (scale with vocab). The signature proof technique is a "fooling
  set" + pigeonhole `f`-collision.
- **Theorem 8 (the headline):** any decidable language can be solved by a
  **constant `(2,3)`-BAPO with chain-of-thought** — the BAPO-CoT simulates a
  Turing machine step-by-step; 2 bits carry the symbol under the head, 3
  attended tokens retrieve state + neighbors. Per-step bandwidth is constant;
  only the *number* of CoT steps grows (potentially impractically). Strengthens
  Merrill–Sabharwal ICLR'24 (which needed growing precision).
- **Theorem 10 (the warning):** multi-layer, score-based, *full* attention does
  **not** rescue REACHABILITY/MAJORITY/MATCH3. Adding depth/heads at a fixed
  bottleneck does **not** raise effective bandwidth — you must *decompose* (CoT)
  or add external memory. The three are "fundamentally high-bandwidth."

### Evaluation / results
- Synthetic tasks at `n ∈ {6,50,100,200}` across GPT-4o/4o-mini/o3,
  Claude-3.5-Sonnet/Haiku, Gemini-1.5-Pro/Flash/2.5-Flash
  ([Fig. 3](https://arxiv.org/abs/2505.08140)): models are near-100% on
  BAPO-easy and drop to ~chance (≈50%) on BAPO-hard by `n=200`. Larger models do
  better *overall* but "even with increased scale, no model avoids the
  degradation BAPO predicts."
- **CoT ([Fig. 4](https://arxiv.org/abs/2505.08140)):** "think step by step"
  helps modestly on small `n` for non-reasoning models; reasoning models
  (o3, Gemini-2.5-Flash) with no token cap *succeed* but burn **1k–100k CoT
  tokens** — direct corroboration of Thm 8's "constant bandwidth, many steps."
- Real tasks ([Fig. 5](https://arxiv.org/abs/2505.08140)): FINDNEGATIVEREVIEW
  (INDEX-like, easy) solved; MAJORITYREVIEW (majority over sentiment, hard) and
  VARIABLETRACKING (a special case of REACHABILITY, extends RULER) fail as `n`
  grows — "BAPO-hardness is a good predictor of LLM performance" on real text/code.
- Cost: ~$400 API / ≤1 day compute.

### Limitations (theirs + reviewers')
- BAPO is *not* a faithful transformer: `f,g,h` get unbounded compute; assumes
  perfect positional encoding; single-token outputs only.
- Does **not** capture all failure modes (e.g. tokenization errors);
  "BAPO-easiness is not a guarantee an LLM solves the task."
- The authors **do not know the root cause** of the small effective bandwidth
  (speculate a generalization↔exactness tradeoff) and **never measure it inside
  a model** — they operationalize it only via aggregate task success.
- Several lower bounds are loose; some problems' bandwidth is uncharacterized;
  positional-encoding artifacts can masquerade as BAPO-hardness.

### Related publications (the theory strand — review seeds for this file)
- **Edelman, Goel, Kakade & Zhang, "Inductive Biases and Variable Creation in
  Self-Attention," ICML 2022** — the formal foundation for attention-head
  capacity limits; the bandwidth notion's closest precursor.
  [proceedings.mlr.press/v162/edelman22a](https://proceedings.mlr.press/v162/edelman22a.html).
- **Elhage et al., "A Mathematical Framework for Transformer Circuits," 2021** —
  the residual stream as a finite-width bus, attention heads as read/write ports:
  the architectural bandwidth picture in everything but name.
  [transformer-circuits.pub/2021/framework](https://transformer-circuits.pub/2021/framework/index.html).
- **Hahn, "Theoretical Limitations of Self-Attention," TACL 2020**
  ([arXiv:1906.06755](https://arxiv.org/abs/1906.06755)) — per-token impact → 0
  as `n` grows; one pillar of the "small effective bandwidth" hypothesis.
- **Merrill & Sabharwal, "The Parallelism Tradeoff," TACL 2023** (TC⁰) and
  **"Expressive Power of Transformers with CoT," ICLR 2024**
  ([arXiv:2310.07923](https://arxiv.org/abs/2310.07923)) — CoT+precision ⇒
  Turing-complete; BAPO Thm 8 strengthens this to *constant* bandwidth.
- **Sanford, Hsu & Telgarsky, NeurIPS 2023** (MATCH2 vs MATCH3,
  [arXiv:2306.02896](https://arxiv.org/abs/2306.02896)); **Sanford et al.,
  NeurIPS 2024** (graph-algorithm lower bounds,
  [arXiv:2405.18512](https://arxiv.org/abs/2405.18512)).
- **Feng et al., "Towards Revealing the Mystery behind CoT," NeurIPS 2023**
  ([arXiv:2310.04150](https://arxiv.org/abs/2310.04150)) — CoT enables dynamic
  programming in bounded depth.
- **Pfau, Merrill & Bowman, "Let's Think Dot by Dot," COLM 2024**
  ([arXiv:2404.15758](https://arxiv.org/abs/2404.15758)) — **theoretical
  barriers on hidden/latent CoT.** In *tension* with BAPO Thm 8 (see MrCogito
  relevance): the resolution is that BAPO-CoT may *attend back to earlier latent
  states* (the tape is re-readable), whereas pure hidden dot-by-dot feeds only
  the last state and hits the barrier.
- **Abbe et al., "How Far Can Transformers Reason? The Globality Barrier,"
  NeurIPS 2024** ([arXiv:2406.06467](https://arxiv.org/abs/2406.06467)) —
  "globality degree" sibling concept.
- **Tomlinson et al., "BAPO Bounds on Chain-of-Thought Token Complexity," 2026**
  ([arXiv:2602.02909](https://arxiv.org/abs/2602.02909)) — same authors' follow-up
  on *how long* CoT must be.
- **Coconut** (latent CoT in practice, [arXiv:2412.06769](https://arxiv.org/abs/2412.06769))
  is reviewed in [`recurrent_memory_transformers.md`](recurrent_memory_transformers.md) §B.

### Relevance to MrCogito — the load-bearing mapping
**MrCogito's architecture is a BAPO with the constraint made explicit and
architectural, instead of implicit.** Map the two channels onto our code:

| BAPO channel | MrCogito realization | Code |
|---|---|---|
| **`a`** prefix bandwidth (compressed summary, in bits) | the concept set `[B,C,H]`, read by the decoder cross-attention | `encode_concepts` `nn/concept_encoder_perceiver.py:1526`; `self.cross_attn` `:1078` |
| **`b`** attention bandwidth (raw-token reach, in tokens) | the windowed decoder's local causal self-attention, `context_window = K` | `_chunked_window_causal_attention` `:993`; `context_window` `:1067`; `b`-capacity ≈ `L·(K−1)` `:1481` |
| **effective `a`** (is the channel actually used?) | the Δshuffle_beyond / Δzero_beyond beyond-window CE gap | `_teacher_forced_ce_window` `:1643` |

The code already states the BAPO thesis verbatim at `concept_encoder_perceiver.py:1650`:
*"any dependency further back than the window MUST flow through the concepts …
the intact-vs-ablated CE gap on beyond-window positions is the direct test of
'are concepts used as cross-window memory?'"*. **E05 is a purpose-built BAPO
probe** (it caps `b` at K=128 to force flow through `a`), and our Δshuffle/Δzero
diagnostics are **a direct measurement of effective `a`-bandwidth — exactly the
quantity BAPO theorizes but concedes it cannot measure.**

**What this reframes (our "failures" are the theory's predictions):**
- *Decoder bypass / Δshuffle→0 under E05, E10–E16* = the model routing through
  channel `b`; effective `a` → 0. BAPO predicts exactly this when a local
  raw-token channel suffices for the next-token objective.
- *E05 Muon-long collapsed harder with more compute* = Thm 10: raising nominal
  capacity (or just optimizing longer) does not raise effective bandwidth; it
  improves the `b`-bypass. Explains why "eval_loss ≠ concept semantics."
- *E02-long de-collapses with scale; E01 reconstruction collapses* = objective
  choice selects which channel carries load (prefix→suffix forces `a`;
  reconstruction is `b`-friendly).

**Implications / verdict:**
- **Adopt (as a standing lens + instrument):** report Δshuffle/Δzero as
  "effective concept bandwidth"; use a BAPO-hard/easy probe suite as the
  reasoning benchmark (the project currently has none — STS-B is BAPO-easy/kNN-like).
- **Adapt (primary next bet):** the **Concept-Flow reasoner (E08)** *is* Thm 8 in
  continuous form — constant per-step bandwidth over `C` concepts, `K` steps.
  Sharpen its falsifiable claim to "Δshuffle_beyond rises monotonically with
  inference `K` and closes a BAPO-hard gap single-pass cannot." Decisive kill:
  flat Δshuffle-vs-`K`.
- **Demote (wrong axis):** E11/E12/E13 (more read paths / depth) and E16
  (scale) raise *nominal* `a`, which Thm 10 + our own E10–E16 nulls say does not
  raise *effective* bandwidth. Keep as ablations/fluency, not the reasoning lever.
- **Design constraint from the BAPO↔Pfau tension:** each flow step must keep
  earlier concept states *re-readable* (attend back to the trajectory), not be a
  pure Markov `z_k→z_{k+1)`, or Pfau's hidden-computation barrier bites.

**Bottom line:** BAPO is strong external corroboration that the *latent
information bottleneck is the decisive object* — MrCogito makes it explicit,
controllable, and (uniquely) *measurable*. The theory-mandated fix for reasoning
is decomposition (many small latent steps), not capacity — i.e. E08, instrumented
with the Δshuffle-vs-K curve and a BAPO task suite.

*See also:* [`recurrent_memory_transformers.md`](recurrent_memory_transformers.md)
(the writable-memory / recurrence engineering counterpart to Thm 8),
[`concept_bottleneck_collapse_mitigation.md`](concept_bottleneck_collapse_mitigation.md)
(forcing effective `a` > 0 — E05c), and E08 spec
[`experiments_specs/ahead/E08_concept_flow_reasoner.md`](../experiments_specs/ahead/E08_concept_flow_reasoner.md).
