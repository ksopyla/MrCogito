# Perceiver revisit — what the ledger, E18, and the Sep-2026 frontier actually say (2026-09-12)

**Written:** 2026-09-12 · dated research note (append-only ledger; later notes may supersede the
*interpretation*, never the record) · author: Cursor cloud-agent session on branch
`cursor/perceiver-revisit-synthesis-00f2`, for review by KS.

**Question asked (KS, 2026-09-12).** E18 is an encoder-decoder with a 1-layer encoder and one global
read; that will not reach 10M context. Go back to the original idea — *encoder → reasoning /
refinement → decoder* that builds a Perceiver-style latent concept space — and check five intuitions
against the facts before switching to a new architecture (E21 is implemented and waiting):
(1) 128 latents were too narrow a bandwidth — try 4k / 8k / 16k; (2) evaluation and metrics were
inadequate (retrieval evals were missing); (3) BiXT / cross-attention is needed but should be
upgraded with value embeddings, n-grams, etc.; (4) E18 showed a modernised stack trains faster, so
modernise the old implementation; (5) add simple reasoning by looping the encoder. End goal: 10M
context (interim 1M) that recalls old facts and is transferable to other agents as a high-bandwidth
latent message. Also: what did DeepSeek and Qwen do recently?

**Sources.** `dev` at `e6434e6` (E18 family closed 2026-09-12, `perceiver_ar` eval layer landed);
the strategy branch `cursor/strategy-sota-review-2026-09-e212` (E21 spec + plan + implementation,
[strategy synthesis 2026-09-11](strategy_synthesis_latent_channel_20260911.md), five topical
literature reviews — **not yet merged to `dev`**, referenced here by branch path); primary sources
re-read on 2026-09-12 for the DeepSeek / Qwen claims (URLs inline). Anything not re-read is labelled.

---

## 0. Answer in one page

1. **The five intuitions, scored against the ledger:** (1) *partly wrong* — the 128 slots were not
   the reason the Perceiver line "did not train"; the bypass was, and the same 128 slots carried
   STS-B 0.714 when the objective closed the bypass (E02-long). Where count *does* bite (≥2k tokens,
   exact recall) the fix is **C ∝ N with positional slot allocation**, not 16k free latents.
   (2) *right, and now half-fixed* — RULER-lite, reach ablation and no-channel controls landed on
   `dev` on 2026-09-12; still missing: aggregation / multi-hop tasks, a BAPO-hard suite, MRCR, and an
   embedding-retrieval probe for the slot space. (3) *half right* — contextualised tokens before
   compression is the load-bearing part (BiXT, DeepSeek's 20-layer causal encoder, ECP); value
   embeddings and n-grams are input-side capacity that did **not** move the E18b needle (arm R2:
   +0.2 points). (4) *right about speed, with a sting* — Muon converged ~5× faster on E05 and
   collapsed the bottleneck harder; and the modern stack **already exists** in
   `nn/perceiver_ar_lm.py`; the old `nn/concept_encoder.py` should not be modernised but retired
   behind it. (5) *right in spirit, wrong target* — loop the **slot array** (the Perceiver latent
   transformer, C ≪ N, cheap), not the token encoder; and only where the loop is supervised densely
   (E21's receiver), or it will be a no-op as E16 was.
2. **The frontier moved onto our square, in a form the 2026-09-11 note under-read.**
   DeepSeek-V4's **HCA pools 128 tokens into one KV entry** (softmax pooling with learned per-token
   weights + learnable positional biases), keeps dense attention over the ≈8k pooled entries at 1M,
   and interleaves it with CSA (4:1 pooling + top-512 sparse read). Beyond a 128-token window
   **every layer** sees only compressed entries — there is no raw long-range path. This is
   slot-*count* compression, positional allocation, trained under plain CE at 32T tokens
   ([V4 report](https://arxiv.org/html/2606.19348)). V4.1-Flash then split the model into a 20-layer
   causal encoder that produces one projected global KV for a 20-layer decoder (CED, YOCO lineage;
   [card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)). Qwen3.8-Flash-Next compresses
   the prefix into a **fixed-size** GDN state in 3 of 4 layers and keeps one full/sparse layer in
   four because "no finite-state memory reproduces exact retrieval"
   ([report](https://arxiv.org/html/2608.30320)). Outside the labs, **LCLM** (Jun 2026,
   [arXiv:2606.09659](https://arxiv.org/abs/2606.09659)) ran a from-scratch architecture sweep of
   encoder→latent→decoder compressors at 16× and scaled the winner to 0.6B-enc / 4B-dec at 1:4–1:16:
   one latent per block, mean pooling > CLS pooling, encoder window ≥ 256, loss on uncompressed
   tokens interleaved with compressed spans. So: *positional pooling works at scale*; *encoder depth
   / window matters*; *exact recall needs a raw or indexed path*; *pooled slots never talk to each
   other* in any shipped model (and LCLM's one attempt at that did not move CE).
3. **What is still unoccupied** (unchanged from the 2026-09-11 note, now sharper): (a) a latent
   array whose slots **interact** (slot-to-slot self-attention / a refinement loop) — HCA/CSA entries
   are pooled independently and only ever read; (b) slots trained as a **message** another copy
   acts on without the text (E21); (c) any of it at ≤1B from scratch with published RULER-1M.
4. **Recommendation.** Do not re-open the 2026-Feb Perceiver-MLM line or modernise
   `concept_encoder.py`. **E21 is the Perceiver revisit** in its positionally-allocated form
   (`KVCompressor` = HCA-style pooling at r = 16; boundary = closed bypass) and it is implemented:
   run it (≈25 GPU-h) as the load-bearing gate. Frame the encoder→reasoning→decoder bet on top of
   it as **E22 — hierarchical positional concept core**: structural closure (decoder = SWA-256 +
   slots, DeepSeek-style, not a stochastic boundary), two slot levels (r = 16 and r = 256) so 1M →
   4k coarse slots and 10M → 40k, and a **weight-tied K-step latent transformer over the slot
   array** as the reasoning core. Falsifiable claim: at matched parameters, RULER aggregation /
   multi-hop (`vt`, `fwe`, `cwe`, multikey) and receiver gain rise with K, and beat the
   independent-slots control (K = 0, i.e. HCA) by ≥ 5 points at 32k; kill if gain-vs-K is flat or LM
   cost > 1%. Two cheap pre-checks already staged or trivial: the mid-depth read
   (`PAR_GLOBAL_POSITIONS=7`, 1.4 GPU-h) and an encoder-depth arm (2 → 8 pre-encoder layers), which
   DeepSeek's 20/20 split makes the single most likely missing ingredient of E18b.

---

## 1. The five intuitions against the record

| # | Intuition (KS) | Verdict | Evidence (ledger / literature) |
|---|---|---|---|
| 1 | 128 latents failed because the bandwidth was too narrow; go to 4k–16k | **Partly wrong** | E02 / E02-long: 128 slots over 512 tokens (4:1) reached STS-B **0.714**, slot-rank rose 5.9 → 16.7 with epochs, early-position Δzero **1.43** — the slots carried load when the decoder had no raw prefix. E01 (same slots, reconstruction) collapsed 14.6 → 4.6. E05 (same slots, 2048 tokens, K = 128 window): Δshuffle_beyond 0.39–0.50, STS-B **0.452**, and Muon-long collapsed to RankMe **4.96** while loss improved — more compute made the *bypass* better. E18 arm C: a model with **no** long-range channel matched arm A (4.091 vs 4.090). BAPO Thm 10: raising nominal `a` does not raise effective `a`. Count *does* bite for exact recall at ≥ 2k, but the frontier's answer is a **ratio**, not a number: DeepSeek HCA r = 128 (1M → 8k entries), CSA r = 4; E18c/E21 r = 16. 16k free latents at 512 tokens is more slots than tokens; 16k slots at 1M is r = 64. |
| 2 | We never had proper evaluation, retrieval evals | **Right; half-fixed** | Feb-2026 diagnosis §5 already said GLUE/STS-B are the wrong probes for a compressor. Since 2026-09-12 `dev` has: lm-eval adapter (HellaSwag / ARC / PIQA / SIQA…), RULER-lite teacher-forced (`passkey`, `multikey`, `vt`, `fwe`, `buckets`, `reach`), the paired **reach ablation**, and the rule "always run the no-channel arm" ([eval layer spec](../engineering_specs/long_context_reasoning_eval_layer.md)). **Missing:** aggregation (`cwe`, `fwe` at ≥ 64k), multi-hop QA / HELMET-class, MRCR (the number Qwen and DeepSeek report), BAPO-hard synthetics (majority, reachability, variable tracking — reliable latent signal at ≤ 3B), an embedding-retrieval probe on the slot space (BEIR/MTEB-lite; the Vision's "recall old facts" is retrieval, not CE), and E21's `message` probe (`none` / `swapped` / `raw`). |
| 3 | BiXT + cross-attention needed; upgrade with value embeddings, n-grams | **Half right** | What made the read *work* (P2 copy 99.9998%) was a value embedding on the retrieving layer (tiny study: without it the global layer never learned copy). What made the read *fail* (E18b) was not fixed by it: arm R2 (value embedding on the read) 4.4% vs R 4.2%, LM trajectory identical to 5 dp. Hashed n-grams are in E18 and ratified at 196B (Engram) / 51B (Qwen), but Qwen's ablation: n-gram vocabulary lowers **loss monotonically while downstream saturates** — size by benchmark, not CE. The load-bearing ingredient for a Perceiver-style compressor is **contextualised tokens** (Feb diagnosis §2; BiXT; ECP's propagated history; DeepSeek's 20-layer causal encoder before the projected KV). |
| 4 | E18 shows the modern stack trains faster; modernise the old implementation | **Right on speed, wrong target** | E05 Muon A/B: ~5× faster to a 1.22-nat-lower eval loss than Adam — and RankMe 10.6 → 5.0, STS-B 0.52 → 0.06 over 0.5 → 2 ep (E05b: not the wd). E18 stack (Muon, QK-norm, SwiGLU, value embeddings, U-net skips, softcap, z-loss, FlexAttention, Liger CE) reaches dense parity (3.790 vs 3.786) at 1.02× throughput. The modern stack **is** `nn/perceiver_ar_lm.py`; `nn/concept_encoder.py` still uses `nn.MultiheadAttention`, learned absolute positions on tokens, LayerNorm/GEGLU defaults, no QK-norm (see §6). Modernising it duplicates a foundation that exists; build the latent array on `perceiver_ar_lm.py`. |
| 5 | Add simple reasoning by looping through the encoder | **Right in spirit; pick the right loop** | E16 (shared-depth recurrence, Gemma, 2K, plain CE): 0.0005–0.001 nats — a loop with a bypass is a no-op. Ouro (7.7T tokens) is the only shipped looped LM; Huginn finds no CoT-like mechanism; SIM-CoT / LOTUS: latents collapse unless **each step is supervised**. E18 verdict lists "looped weight-tied encoder" as future work; DeepSeek CED shows encoder depth (20 of 40 layers) is where the projected KV gets its quality. The cheap, principled loop is over the **slot array** (C ≪ N): a Perceiver latent transformer, K steps, weight-tied, with the receiver's CE (E21) supervising every step; metric = gain vs K. |

## 2. What the Perceiver-era runs (Jan–Jul 2026) actually taught

The line is often remembered as "128 concepts collapsed". The ledger says something more specific.

| Period / family | Setup | Outcome | Cause (as diagnosed, with the dated note) |
|---|---|---|---|
| Jan–Feb `perceiver_mlm` / `weighted_mlm` | H512 L2–L6 C128, MiniPile, MLM through a 1-layer position-query decoder | MLM 2.54; slot rank **5/128**; MRPC 81–83, STS-B 0.63–0.65 via decoder | Four structural misalignments ([diagnosis 2026-02-21](mlm_perceiver_diagnosis_20260221.md)): encoder cross-attends `[MASK]` embeddings; tokens **uncontextualised** across all layers (BiXT was optional and off); decoder input-embedding shortcut removes gradient on 85% of positions; GLUE/CLS pooling is the wrong probe. Concept losses (combined / Kendall-Gal) fixed rank (122/128) and destroyed semantics (STS-B 0.34). |
| Feb–Mar diffusion / prefix-diffusion / denoise (BiXT on) | H512 L6 C128 D2–3, MiniPile / WikiText-103 | rank 4–11/128; best zero-shot STS-B **0.607** (denoise) | Unvalidated bottleneck + decoder bypass ([lit scan 2026-06-13](../1_Strategy_and_Plans/agenda.md)); code parked. |
| Jun E01 / E02 / E03 / E04 (`concept_ar`) | H768 L6 C128 D4, BiXT, SwiGLU, RMSNorm, RoPE decoder, seq 512, FineWeb-Edu | E01 recon: collapse 14.6 → 4.6. **E02 prefix→suffix: STS-B 0.702 → 0.714 (5 ep), rank 5.9 → 16.7 with epochs, early Δzero 1.43.** E03 anchor helps relatively. E04 parallel decoder: RankMe 108, STS-B 0.53. | **Collapse is objective-dependent** ([E02-long report](../2_Experiments_Registry/run_reports/e02_long_5epoch_20260618.md)): the objective that closed the raw path de-collapsed with scale. Measurement reframe: batch-mean "effective rank" measured slot redundancy, not representation rank; RankMe + early-Δ became the gates. |
| Jun–Jul E05 / E05b (windowed decoder K = 128, seq 2048) | same model, prefix→suffix, decoder window 128 (reach 508 of 2048) | Adam: RankMe 37.7, Δshuffle_beyond 0.39, **STS-B 0.452 < token-mean floor 0.486**; Muon 2 ep: eval 2.58 (project low), RankMe **4.96**, STS-B **0.062**; E05b: wd innocent. | The within-window bypass is the attractor; Muon's whitened updates find it faster ([E05 Muon-long report](../2_Experiments_Registry/run_reports/e05_muon_long_2ep_collapsed_20260709.md)). Also: 128 free slots for 2048 tokens (16:1) *is* too coarse for exact content — the one place intuition 1 holds. |
| Jul–Aug E10–E17e (Gemma backbone + concept banks) | 1B backbone, 2K–4K, plain CE ± carry dropout / starved window | Δbeyond ≈ 0 under plain CE every time; carry dropout forces use (Δperm 0.59) that does not transfer; free-run degenerate | Same law: a raw path exists, CE routes through it ([five whys](e17c_failure_five_whys_20260815.md)). |

Two things the line never had, which the frontier now shows are decisive: **positional slot
allocation** (one slot per block, so no slot can be starved and rank is structural) and **no raw
long-range path at any layer**. Both are in E18c/E21's `KVCompressor` design and in DeepSeek V4;
neither was in Perceiver-MLM, E01–E05 or E10–E17.

## 3. E18 reread: why one global read is not the 10M answer, and what it did prove

**Parameter count, corrected (KS, 2026-09-12).** The E18 docs call the pilot "125M". That is a
shape label (a GPT-2-small stack: d = 768, 14 layers), not a count. From `analytic_param_count`
on the `scripts/launch_e18.sh` config: 14 layers **88.1M** + embedding MLP 1.0M + token table
(128,256 × 256) 32.8M + **untied `lm_head` (768 × 128,256) 98.5M** = 220.5M "dense" as the
factory logs it, plus lookup tables (2/3-gram 33.6M, three value-embedding tables 24.6M) =
**278.7M total**, which is what W&B's `num_parameters` reports. Only **89M** of it is transformer
compute; 190M is vocabulary-sized tables, and the untied 128k-vocab head alone is 45% of the dense
parameters at this width. No E18 conclusion changes (every arm and the dense control share the
count), but "learnable at 125M" should read "learnable with an 89M stack and a 128k-vocab head",
and every E18/E18b/E21 doc that says 125M should carry the breakdown.

Structurally E18 is `[SWA-1024]×2 → [FULL causal]×1 → [SWA-4096]×20`: a **1-layer encoder**, one
cross-attention, a 12–20-layer decoder ([verdict](../2_Experiments_Registry/run_reports/e18_family_verdict_20260912.md)).
Proven: dense parity (3.790 vs 3.786), throughput parity, exact **positional** retrieval through the
read (copy @32k, offset 16k: 99.9998%; reach cut two tokens short → 0.4%), a 1 KB/token cache, the
extension-protocol fix (≈12% at 32k), and the paired reach-ablation instrument. Falsified: LM loss
from the read (arm C = arm A) and **content**-addressed retrieval (E18b: 4.49% vs dense 99.33% on
the identical task). At 10M the read is O(M²) and 94% of FLOPs, retained activations need ≥ 280 GB,
prefill is ~17 min ([blockers note](e18_10m_context_blockers.md)) — the note's own escape hatches
are "sparse/hierarchical global read" and "learned K/V compression of the prefix", i.e. the concept
idea, installed once because there is exactly one global layer.

Two low-cost E18 questions remain open and both are worth answering *before* any Perceiver v3 run,
because they test the 2026-09-12 hypothesis that the failure was **encoder depth**, not the read:
(i) the staged mid-depth read (`Cache/jobs/e18b_mid_taskonly.sh`, ~1.4 GPU-h) — everything below
layer 7 becomes a 7-layer encoder; (ii) an encoder-depth arm on E18b-T (pre-encoder 2 → 8 SWA
layers, read at the bottom, same cache). DeepSeek CED puts **half** the network in the encoder; ECP
attributes its 28.9 → 18.83 PG-19 gain over Perceiver AR precisely to propagating history through
every layer instead of compressing it after layer 1
([arXiv:2412.06106](https://arxiv.org/abs/2412.06106)). E18's 2-layer pre-encoder is the outlier.

## 4. What DeepSeek and Qwen shipped (re-read 2026-09-12) and what it changes

### 4.1 DeepSeek-V4 (Pro 1.6T-A49B / Flash 284B-A13B; report [arXiv:2606.19348](https://arxiv.org/html/2606.19348))
- **HCA (Heavily Compressed Attention)**: per layer, K/V of every **m′ = 128** tokens are pooled into
  one entry — weights = row-softmax of a learned per-token score `Z = W^Z h` plus a learnable
  positional bias `B ∈ R^{128×c}`; dense attention over all pooled entries (≈ 8k at 1M), no indexer.
  **CSA**: the same compressor at m = 4 with overlap, then DSA's lightning indexer keeps **top-512**
  entries per query. Both add a **128-token uncompressed sliding window**; a query cannot see its own
  unfinished block. V4-Pro even *starts* with two HCA layers. QK-RMSNorm on queries and on the pooled
  KV; partial RoPE (last 64 dims) with a `−i` rotation on the outputs so pooled values carry relative
  positions; learnable attention sinks; MQA over pooled entries. Training: 4K → 16K → 64K → 1M;
  dense attention for the first 1T tokens, then sparse; Muon (RMS 0.18 rescale, wd 0.1) + AdamW for
  embeddings/head/norms; MTP 0.3 → 0.1. MRCR-1M **83.5 MMR%** (Pro-Max; the 8-needle accuracy at 1M
  is ≈ 0.59 per the [HF blog](https://huggingface.co/blog/deepseekv4) — do not conflate the two
  metrics), CorpusQA-1M 62.0; retrieval flat to 128K, degrading beyond. KV cache ≈ 2% of a BF16
  GQA-8 baseline at 1M.
- **Correction to the 2026-09-11 note.** It says the frontier compresses "in bytes, not in count …
  still one slot per token". That is true of **V4.1-Flash's CSA2** (r = 2 in the encoder, r = 1 in
  the decoder) but **false for V4's HCA**: 128:1 in slot count, positionally allocated, per layer,
  trained under CE. The unoccupied cell is therefore *not* "compress in count" — it is (a) slots
  that interact, (b) slots as a message, (c) small scale from scratch. `frontier_open_models_architecture.md`
  on the strategy branch should be amended accordingly (docs-hygiene handoff).
- **Why HCA works under plain CE where E05/E10–E17 did not** (working hypothesis, consistent with
  BAPO and arm C): beyond 128 tokens there is *no raw path in any layer*; the pooled entries are
  contextualised by that layer's depth; allocation is positional (one entry per block, so rank is
  structural and nothing competes for a slot); and CSA supplies fine-grained exact recall so HCA
  is never asked to do it. Scale (284B, 32T) helps but is not the mechanism.

### 4.2 DeepSeek-V4.1-Flash (2026-09-10; [card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [report](https://www.alphaxiv.org/abs/2609.deepseek-v4-1-flash.pdf), [SGLang notes](https://www.lmsys.org/blog/2026-09-10-deepseek-v41/))
- **CED**: 20-layer causal encoder → **one projected global KV** (layer-dependent projections of the
  encoder's last hidden state) → 20-layer decoder that computes only a 128-token window for prompt
  tokens ("SWA bounded replay"). Inspired by YOCO. Prefill nearly halves; 8B active in prefill /
  16B in decode; 552B backbone + 196B **Engram** (hashed n-gram tables at layers 1 and 14, host- or
  GPU-resident); single-pass mHC (4 residual streams, Sinkhorn-projected mixing); head-wise Muon;
  Sinkhorn-balanced momentum instead of Adam for embeddings / Engram / head; no MTP in pretraining
  (DSpark drafting trained after). **CSA2**: r = 2 in encoder layers 3–20, r = 1 in the decoder,
  Full / Reindex / Reuse modes sharing KV and indexer keys across layers, hierarchical indexer
  (first Full layer selects 2,048 blocks × 8 = 16,384 candidates; later layers re-index within),
  top-512, FP4 main KV → **890 B/token**. Sparse from scratch at 64K, extended to 1M at 34T of 45T
  tokens. LongBench-V2 45.2 (base); 1M used in agent evals.
- The encoder's output *is* a transferable object: SGLang's prefix cache stores compressed KV +
  indexer keys and reloads them for the decoder. Same model, same text — not yet a message a
  different reader acts on without the text (E21's gap).

### 4.3 Qwen3.8 / Qwen3.8-Flash-Next (Aug 2026; [report arXiv:2608.30320](https://arxiv.org/html/2608.30320), [card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next))
- 48 layers = 12 × (3 GDN → 1 QSA); GDN "compresses the prefix into a **fixed-size** recurrent state
  at linear cost, while one full-attention layer in every four retains the direct token-level
  retrieval that no finite-state memory reproduces exactly". Ablation at 25B-A3B / 400B + 80B
  tokens: GDN-hybrid > full attention on 8/9, > **SWA-128 hybrid on 7/9**. QSA = compressed
  micro-block indexer (indexer cost O(n²/r)), budget 512 blocks / 2,048 tokens, 4 indexer heads;
  RULER 93.0 beyond 512K, 8-needle MRCR 26.4 @1M. Gated Residual (4 branches, elementwise read gate);
  51B hashed n-gram tables at layer 2, host-prefetched; Muon on 2-D weights (8 Newton–Schulz steps),
  refit scaling law → larger LR/batch, no batch warm-up; **NoPE on full layers**: indistinguishable
  in pretraining, endless generation after post-training. **n-gram vocabulary: loss ↓ monotonically,
  downstream saturates.** 262k native → 1M.
- For us: the local mixer question (SWA vs GDN) is open and cheap to arm; exact recall needs a
  token-level (raw or indexed) path — a pure fixed-state or pure pooled design will fail passkey /
  multikey, which is what E18b already showed for a *learned* channel and what E21's `raw` arm U
  controls for.

### 4.4 The closest published cousin of the revisit: LCLM (Jun 2026)
**End-to-End Context Compression at Scale** — Li, McLeish, … Goldstein, Lotfi, Goldblum, Izmailov
([arXiv:2606.09659](https://arxiv.org/abs/2606.09659)). An encoder maps **each contiguous block of N
tokens to one latent token** (positional allocation again), an MLP adapter projects it, a decoder
consumes latents as context. They ran a **from-scratch architecture sweep** (Qwen3-0.6B-shaped enc +
dec, 38B tokens, 16×) and then continually pre-trained 0.6B-enc / 4B-dec at 1:4, 1:8, 1:16 on
350B tokens; the result sits on a new RULER / LongBench Pareto frontier vs KV-eviction methods.
Findings that transfer directly to a Perceiver v3 design, in their words: **mean pooling beats
CLS/EOS-token pooling** (concat ≈ mean; concat wins at 4×, mean at 16×); **encoder window W matters
a lot** — W = N (16) → 256 is a large gain, 256 → 1024 a smaller one (contextualise before you
pool: the depth law); **causal encoder mask**; an **attention adapter (one self-attention layer over
the latent sequence) did *not* beat an MLP adapter on pre-training loss**; training format =
**interleaved compressed / uncompressed segments with loss only on uncompressed tokens** (E02's
receiver framing, generalised to many boundaries per row — this is E21's objective); an
**auxiliary reconstruction task** to keep fine-grained detail for exact retrieval; staged
training when warm-starting; and an agent that **skims the compressed view and `EXPAND`s a raw
512-token chunk on demand** recovers exact-match NIAH — the "coarse slots + fine raw path" split
that DeepSeek implements with HCA + CSA. Two implications for us: E21's design is independently
validated at 4B scale (positional slots, interleaved boundaries, loss through the latents), and
the E22 "slots talk to each other" claim has one negative data point on *loss* — it must be judged
on aggregation / multi-hop tasks, where loss and downstream diverge (Qwen's lesson), not on CE.
Related 2026 items from the scout (not re-read): Latent Context Compilation
([arXiv:2602.21221](https://arxiv.org/html/2602.21221), inference-time distillation to buffer
tokens, up to 32×), Baseten's STILL (a Perceiver bottleneck cross-attending the full KV into a
compact KV, KL-distilled; blog only), Parcae (stable looped LM + loop scaling laws at ~1.3B,
[arXiv:2604.12946](https://arxiv.org/pdf/2604.12946)).

### 4.5 Convergence table (what to copy, what is still ours)

| Mechanism | DeepSeek V4 / V4.1 | Qwen3.8-Next | Ours (dev / strategy branch) |
|---|---|---|---|
| Slot-count compression of the prefix | **HCA 128:1 dense + CSA 4:1 sparse (V4)**; CSA2 2:1 / 1:1 (V4.1) | GDN fixed state (∞:1) in 3/4 layers | `KVCompressor` r = 16 (E18c/E21, branch) |
| Slot allocation | positional (one entry per block, softmax pooling + positional bias) | recurrent state | positional pooling (E21) — same family; free latents in `concept_encoder.py` |
| Raw long-range path | **none beyond 128 tokens** at any layer | full/sparse layer every 4th | E18: SWA-4096 chain (bypass); E21: none across the boundary on q rows |
| Encoder depth before the shared read | 20 of 40 layers (V4.1 CED) | — | 2 pre-encoder layers (E18) |
| Exact-recall path | CSA top-512 over 4:1 entries | QSA top-2048 tokens | raw current block + retrieval rows (E18b/E21) |
| Slot-to-slot interaction / refinement loop | **none** | none | **open cell** (Perceiver latent transformer) |
| Slots as a message for another reader | prefix-cache reuse only | KV reuse across turns | **E21** (open cell) |
| QK-norm on pooled KV, partial RoPE, sinks | yes / yes / yes | RoPE kept (NoPE hurt) | QK-norm yes; RoPE yes; sinks via `swa_sink` |
| Hashed n-gram tables | Engram 196B | 51B | `TinyHashedEmbedding` 2×2^17×256 |
| Optimiser | head-wise Muon + Sinkhorn momentum | Muon + AdamW split | Muon + AdamW split (wd 0.1) |
| MTP | omitted (V4.1) / 0.3 (V4) | 1-layer MTP | none (hurts < 1B) |

## 5. Synthesis — three laws and one open cell

1. **The bypass law** (ledger + BAPO + arm C): a compressed channel is load-bearing only if the loss
   cannot be satisfied without it. DeepSeek satisfies this *structurally* (no raw path beyond 128
   tokens in any layer); E02 satisfied it by objective (prefix invisible to the decoder); E21
   satisfies it by a mask on q of the rows. E05/E10–E17/E18 violated it and the channel died
   regardless of init, gating, optimizer or depth placement.
2. **The allocation law** (E02 vs E05; HCA): free latents competing for a whole sequence collapse
   under strong optimizers (E05 Muon RankMe 5); positional slots (one per block) cannot collapse in
   count and are contextualised by the encoder depth beneath them. The vision's "C scales with N"
   table is right; the Perceiver's *free* latent array is the wrong way to realise it. Learned
   queries belong **inside a block** (c queries per r tokens — E18c's design), not across the sequence.
3. **The depth law** (ECP, CED, Feb diagnosis §2): what is compressed must be contextualised first.
   One or two layers of context before the read produce keys that support positional copy but not
   content addressing (E18b vs DT). Half the network before the projected KV is the frontier default.
4. **The open cell.** In every shipped design the compressed entries are written once and read
   many times; they never attend to each other, are never refined, and are never handed to another
   model as the sole input. A **slot array that reasons** (latent transformer over C = N/r slots,
   K weight-tied steps, re-readable across steps per BAPO/Pfau) and that is **trained as a message**
   (E21) is the Perceiver vision restated on the frontier's substrate. It is cheap (O(C²) with
   C ≪ N), it is the only place "reasoning bandwidth" can be measured (gain vs K on BAPO-hard and
   RULER aggregation tasks), and no lab has an incentive to re-pretrain for it. **Tension to carry
   into the spec:** LCLM's sweep found one self-attention layer over the latent sequence did not
   lower pre-training *loss* vs an MLP adapter (§4.4). If slot interaction pays, it will show on
   aggregation / multi-hop and on the receiver gain, not on CE — which is exactly where Qwen and
   our own E05 say loss and capability diverge. A spec whose only gate is CE would miss it either way.

## 6. Modernisation: what to port, what to retire

Code audit of `dev` @ `e6434e6` (explore pass, 2026-09-12). The historical concept encoder is
`nn/concept_encoder.py` (`ConceptEncoderConfig` L43–224; classic `ConceptEncoderLayer` L226–332;
`BiXTCrossAttention` L335–444; `BiConceptEncoderLayer` L470–577; `ConceptEncoder` L579+) consumed by
`ConceptEncoderForConditionalLM` (`nn/concept_encoder_perceiver.py` ~L1400–1970, per-layer
cross-attention to the `[B,C,H]` concepts, `ConceptCausalDecoderLayer` L1044–1154). The maintained
platform is `nn/perceiver_ar_lm.py` (family `perceiver_ar`).

| Feature | `nn/concept_encoder.py` + `ConceptEncoderForConditionalLM` (E01–E05) | `nn/perceiver_ar_lm.py` (E18 platform) |
|---|---|---|
| Latent count | fixed `nn.Embedding(concept_num, H)`, free latents expanded per batch (L621–625) | none on `dev` (`concept_num=0`; global read); per-block slots = `KVCompressor` on the strategy branch |
| Concept ↔ token attention | classic: uni-directional C←T via `nn.MultiheadAttention`; BiXT: shared similarity, **manual matmul + softmax** (L335–444) | block-swept causal read via FlexAttention / SDPA / FA (L347–419) |
| Token positions in the encoder | **learned absolute** `token_position_embeddings` (L600–609); concept positions default `none` | RoPE with document-aware positions, NoPE-every-k, optional global NoPE / SSMax (L483–496, L690–698) |
| Norm / FFN | **Pre-LN, LayerNorm + GEGLU defaults**; RMSNorm / SwiGLU opt-in (L30–41, L261–266) | RMSNorm + SwiGLU (L554–562) |
| QK-norm · GQA · softcap · z-loss · value embeddings · U-net skips · zero-init out-proj | none | all present (L512–513, L68, L607–657, L520–545, L565–588, L707–710) |
| Input | full-width token embeddings or `token_embedding_dim` + projection | tiny embed + hashed 2/3-gram tables (`TinyHashedEmbedding`, L455–475) |
| Decoder self-attention | manual QKV + optional RoPE + SDPA; sliding window `decoder_context_window` (E05) | GQA, QK-norm, SWA stack, one global read |
| Long-context path | sequence parallel via the BiXT global-softmax path (1M validated on 3 × 3090, 2026-06-27) | `prefix_kv()` (L943–959), `reach_override()` (L751–773), `write_back_proj` params (zero-init, **not wired into `forward`**, L701–712); `KVCompressor` + `message_boundary` **absent on `dev`**, present on the strategy branch |
| Output head | `ChunkedLMHeadCE` (L1332–1397) | chunked CE / Liger fused CE |
| Optimiser | Muon available (`nn/muon.py`) | Muon + AdamW split, calibrated (wd 0.1, adamw 2e-4) |
| Tests | `tests/test_concept_encoder_layer.py` (default non-BiXT path); BiXT only indirectly | E18/E21 test suites (513 green on the branch) |

Registered families on `dev` (`training/concept_pretraining_factories.py`): `perceiver_ar`,
`backbone_concept`, `concept_ar` (+`_prefix` / `_bixt`), `perceiver_denoise` (+`_bixt` /
`_contrastive`); objectives `reconstruction`, `reconstruction+contrastive`, `prefix_suffix`,
`causal_lm`. The `concept_ar` / `perceiver_denoise` families remain loadable for E01–E05 checkpoints.

What the frontier speedrun stack looks like now, for calibration (modded-nanogpt record #89,
2026-07-17, 1.23 min to 3.28 on 8×H100; [README](https://github.com/KellerJordan/modded-nanogpt)):
RoPE + QK-norm + ReLU², **NorMuon with Polar Express**, value embeddings + U-net/MUDD skips, a
GPU-resident **bigram hash embedding**, logit softcap + FP8 head, FA3 with long–short sliding
windows and dynamic YaRN, MTP, simplified hyper-connections, cautious weight decay. Relative to that,
`perceiver_ar_lm.py` lacks NorMuon/Polar Express, FP8 head, long–short SWA warm-up and
hyper-connections; none of these touches the load-bearing question.

**Recommendation:** do not modernise `concept_encoder.py` in place (checkpoint loadability for
E01–E05 must be preserved anyway). Implement the slot array as a config-selectable component of
`perceiver_ar_lm.py` (`KVCompressor` already is the per-block pooler; add `c` learned queries per
block and an optional latent transformer over the slot sequence). Ranked modern ingredients by
evidence: (1) Muon with the E05 caveat — always watch RankMe and the no-channel control, since it
finds bypasses faster; (2) QK-RMSNorm on the pooled K and on queries (DeepSeek); (3) partial RoPE
at block-end positions with the `−i` output correction so pooled values carry relative position
(DeepSeek), or block-end RoPE as in E18c; (4) learnable attention sinks; (5) a 128–256-token raw
sliding window as the *only* raw path; (6) hashed n-grams — keep, size by benchmarks; (7) value
embeddings on the retrieving layer — keep for copy circuits, do not expect content addressing from
them; (8) head-wise Muon / NorMuon — free; (9) GDN local mixer — one arm, not a default; (10) MTP —
skip below 1B.

## 7. Where E21 sits, and the bet to frame next

**E21 is the Perceiver revisit, not a departure from it.** Its `KVCompressor` is HCA-style softmax
pooling at r = 16 (per-kv-head learned weights, zero-init to mean pooling; DeepSeek adds a positional
bias); its boundary closes the bypass the way DeepSeek's 128-token window does; its `prefix_kv()` is
the slot array; its `none` / `swapped` / `raw` probes are the causal controls the field lacks. What it
does not have — and what the encoder → reasoning → decoder vision needs — is: slot-to-slot
interaction (no latent transformer), a refinement loop, a deeper encoder, and structural (rather
than stochastic, q = 0.5) closure. Those are exactly the arms of the next spec, and they are only
interpretable **after** E21 answers "can pooled slots carry suffix-relevant content at all at
the pilot scale / 32k" (K1 / K2). So: run E21 first (≈ 25 GPU-h), with the two E18 pre-checks in §3 alongside.

**E22 (to be framed by `experiment-design`) — hierarchical positional concept core.** One coherent
bet: on the E18/E21 platform, make closure structural (decoder stack = SWA-256 + slot reads only, no
SWA chain across blocks), pool level-1 slots at r = 16 with c = 1–2 learned queries per block, pool
level-2 slots at r = 256 from level-1, and run a **weight-tied latent transformer over the slot
array for K ∈ {0, 1, 2, 4} steps** (causal over slots, re-readable across steps) before the decoder
reads them. Encoder depth 8. Claim: at matched parameters and tokens, (a) RULER aggregation /
multi-hop (`vt`, `fwe`, `cwe`, multikey) at 32k–128k and the E21 receiver gain **increase
monotonically with K**, (b) K = 2 beats the K = 0 control (independent pooled slots = HCA) by ≥ 5
points on aggregation while passkey through the raw+CSA-style path stays ≥ 0.9 × E18b-DT's, (c) LM
loss on ordinary rows within 1% of the K = 0 control. Kill: flat gain-vs-K after one iteration
(K = 8), or aggregation < K = 0. Diagnostics: RankMe of the slot array per level, reach ablation,
`none` / `swapped`, cache bytes/token (target 64 B at level 1, 4 B at level 2 → 10M = 640 MB + 40 MB).
Ingredients to lift from the frontier into the plan, not to re-derive: mean/softmax pooling with a
learnable positional bias (HCA; LCLM found mean > CLS-token pooling), encoder window ≥ 256 tokens
before pooling (LCLM), a small raw window as the only uncompressed path (DeepSeek 128), QK-norm on
pooled KV and block-end partial RoPE (DeepSeek), an auxiliary reconstruction / retrieval row mix so
slots keep fine-grained detail (LCLM; E18b rows), and dense attention on the coarse level with an
indexed fine level (HCA + CSA; LCLM's `EXPAND` agent).
Cross-domain hook: hierarchical pooling + a recurrent core over the coarse level is a cortical
column / thalamocortical loop — fast local state, slow global refinement — and BAPO Thm 8 in
continuous form (constant per-step bandwidth over C slots, K steps).

## 8. Evaluation gaps to close before claiming "recall old facts at 1M"

Present on `dev`: RULER-lite (`passkey`, `multikey`, `vt`, `fwe`, `buckets`, `reach`), lm-eval
tiers, health check, two-GPU suite runner. Add, in this order: (1) `cwe` and `fwe` at 64k–256k
(aggregation is where pooled slots should win over sparse retrieval); (2) MRCR 2–8 needle at
128k–1M (comparable to Qwen/DeepSeek); (3) a BAPO-hard synthetic set (majority, reachability,
variable tracking) with a gain-vs-K sweep; (4) an embedding-retrieval probe on `prefix_kv()` slots
(mean-pooled slot → document retrieval, BEIR-lite) — the Vision's "recall" is retrieval, and the
Feb diagnosis already asked for it; (5) the E21 `message` probe with `none` / `swapped` / `raw`.

## 9. Docs to amend (docs-hygiene handoff, not done here)

- `docs/literature_review/frontier_open_models_architecture.md` (strategy branch): HCA is 128:1
  slot-count compression; the "one slot per token" claim applies to V4.1-Flash CSA2 only.
- `docs/1_Strategy_and_Plans/vision_and_goals.md` scaling table: express C as a ratio ladder
  (r = 16 / 128 / 256 / 16k two-level) and cite HCA as the shipped anchor at r = 128.
- `agenda.md` current focus: E18 pre-checks (mid-depth read, encoder-depth arm) + E21 run, then E22.
