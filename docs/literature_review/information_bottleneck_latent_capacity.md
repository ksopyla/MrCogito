# Information bottleneck and latent capacity — count vs width

Reviews of work that asks *how much information a fixed set of latents can
carry*, and whether that budget is spent by **adding slots** (count `K`) or by
**widening each slot** (width `D`). This is the capacity axis for E21's
`KVCompressor`: one slot per `r` prefix tokens, slot width = the global read's
K/V (`g` kv-heads × `head_dim`), currently `r = 16` and `D = 2 × 128` at the
tiny/H=256 DNA scale.

Related reviews (do not duplicate):
- BAPO / prefix bandwidth `a` vs raw-token bandwidth `b`:
  [`reasoning_bandwidth_information_flow.md`](reasoning_bandwidth_information_flow.md)
- AutoCompressor / Perceiver / Infini-attention (engineering of the memory):
  [`recurrent_memory_transformers.md`](recurrent_memory_transformers.md)
- Collapse / decoder bypass (why allocated width is unused):
  [`concept_bottleneck_collapse_mitigation.md`](concept_bottleneck_collapse_mitigation.md)
- Learned KV compressors at 8k–32k:
  [`learned_kv_context_compression.md`](learned_kv_context_compression.md)

**Standing claim from this file (evidence, not a project verdict):**
reconstruction of *unique* long text is typically **count-starved** below ~4×
(ICAE). Instruction/gist compression is **count-saturated at 1–few tokens**.
Per-token attention cache is often **width-overprovisioned** (MLA). Decoder
softmax is **width-rank-starved**. RankMe / superposition say extra `D` is
frequently unspent or overpacked, not linearly more bits. Almost no paper
jointly sweeps `(K, D)` at matched compute.

---

## The Information Bottleneck Method

[arXiv:physics/0004057](https://arxiv.org/abs/physics/0004057) · Tishby, Pereira, Bialek (1999).

### TL;DR
The optimal compressor `Z` of `X` for predicting `Y` solves
`min I(X;Z) − β I(Z;Y)`. Rate is what you are allowed to keep; relevance is
what you must keep. A concept set is a `Z`.

### The problem the authors want to solve
Extract a compressed representation that is as small as possible while remaining
predictive of a downstream variable.

### The solution (intuition)
Treat compression as a trade-off, not a reconstruction mandate. `β` traces the
IB curve: small `β` throws information away; large `β` keeps almost all of `X`.

### Detailed solution / training
Blahut–Arimoto on discrete `p(z|x)`. No neural nets, no slot count.

### Evaluation / results
Classic clustering / speech examples. Theoretical, not LM-scale.

### Related
Deep VIB ([arXiv:1612.00410](https://arxiv.org/abs/1612.00410)); *Fixing a Broken ELBO*.

---

## Deep Variational Information Bottleneck

ICLR 2017 workshop · [arXiv:1612.00410](https://arxiv.org/abs/1612.00410) ·
Alemi, Fischer, Dillon, Murphy (Google). Code (TF):
[github.com/alexalemi/vib_demo](https://github.com/alexalemi/vib_demo);
PyTorch port [github.com/1Konny/VIB-pytorch](https://github.com/1Konny/VIB-pytorch).

### TL;DR
IB with a stochastic encoder `q(z|x)`: `E[log q(y|z)] − β KL[q(z|x) || p(z)]`.
The **width of one Gaussian latent is the rate**. Collapse when `β` is too large.

### The problem the authors want to solve
Make IB trainable with backprop on deep nets.

### The solution (intuition)
Reparameterized Gaussian `z ∈ R^d`; the KL term is the rate penalty. Same family
as β-VAE.

### Evaluation / results
Image classifiers: better generalization and adversarial robustness vs
deterministic regularizers. `β` controls rate. **No `K` vs `d` sweep** — a
single vector, not a set of slots.

### Related
β-VAE (Higgins 2017); Tishby 1999.

---

## Fixing a Broken ELBO

ICML 2018 · [arXiv:1711.00464](https://arxiv.org/abs/1711.00464) ·
[PMLR](https://proceedings.mlr.press/v80/alemi18a.html) · Alemi, Poole,
Fischer, Dillon, Saurous, Murphy.

### TL;DR
ELBO is one point on a **rate–distortion** curve. Models with identical ELBO
can sit at `R ≈ 0` (ignored latents) or at useful `I(X;Z)`. Unused width is a
rate-collapse failure, not a count failure.

### The problem the authors want to solve
Stop ranking VAEs by ELBO alone; ELBO hides whether the latent is used.

### The solution (intuition)
Report variational bounds `R` (rate = `E KL[q(z|x)||p(z)]`) and `D`
(distortion) and sweep `β` / a target rate. A powerful decoder can satisfy the
likelihood with `R → 0`.

### Evaluation / results
Omniglot: 14 models with ELBO < 91.2 nats spanning **R = 0.0074 to 10.92 nats**.
Powerful (autoregressive) decoders sit at near-zero rate unless rate is
targeted. This is the same decoder-bypass the project measured as
Δshuffle → 0 (E05, E10–E18 arm C).

### Related
Dai et al., "Usual Suspects" posterior collapse
([ICML 2020](https://proceedings.mlr.press/v119/dai20c.html)) — collapse can
also be a bad local min that truncates latent dimensions.

---

## Breaking the Softmax Bottleneck

ICLR 2018 · [arXiv:1711.03953](https://arxiv.org/abs/1711.03953) ·
Yang, Dai, Salakhutdinov, Cohen. Code:
[github.com/zihangdai/mos](https://github.com/zihangdai/mos).

### TL;DR
`P(x|c) ∝ softmax(h_c W^T)` factorizes a context-by-vocab matrix of rank ≤ `d`.
Natural language is high-rank, so **output width `d` is a hard capacity cap**.
Mixture-of-Softmaxes raises rank to `K·d`.

### The problem the authors want to solve
LM softmax cannot represent high-rank context–word dependencies at typical `d`.

### The solution (intuition)
A mixture of `K` softmaxes lifts the rank of the logit matrix.

### Evaluation / results
PTB 47.69 PPL, WikiText-2 40.68; +5.6 PPL on 1B Word vs baseline. Naively
increasing `d` overfits more than adding mixture components. Later: Ganea et
al. 2019 argue monotonic nonlinearities also break the bottleneck
([PMLR](https://proceedings.mlr.press/v97/ganea19a.html)).

### Why it matters here
The *decoder* of a concept LM can be width-rank-starved even when the concept
set has spare count. Adding slots does not lift softmax rank.

---

## RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank

ICML 2023 · [arXiv:2210.02885](https://arxiv.org/abs/2210.02885) ·
Garrido, Balestriero, Najman, LeCun.

### TL;DR
**Effective rank** of unlabeled embeddings predicts JE-SSL downstream accuracy.
Allocated width is not used bits. High rank is necessary, not sufficient.

### The problem the authors want to solve
Pick SSL hyperparameters without labels.

### The solution (intuition)
`RankMe(Z) = exp(−Σ p_k log p_k)` with `p_k ∝ σ_k(Z)`. Entropy of the
normalized singular-value spectrum.

### Evaluation / results
Recovers ImageNet linear-probe accuracy within ~0.5 pt of a labeled oracle
(e.g. DINO student-temp 72.4 vs 72.4). Source↔target rank Pearson > 0.99.
Full-rank degenerates exist. The project's per-sample RankMe on concept
matrices is this estimator applied to `[B, C, H]`.

### Related
Jing et al. dimensional collapse ([ICLR 2022](https://openreview.net/forum?id=YevsQ05DEN7)).

---

## Toy Models of Superposition

Anthropic, Sep 2022 (not peer-reviewed) ·
[transformer-circuits.pub/2022/toy_model](https://transformer-circuits.pub/2022/toy_model/index.html)
· Elhage, Hume, Olsson, Schiefer, Henighan, Olah et al.

### TL;DR
When features are sparse, a `d`-dim residual stream can represent **more than
`d` features** (superposition), paying interference that nonlinearities must
filter. Width is a packing budget, not `d` independent channels.

### The problem the authors want to solve
Explain how toy ReLU autoencoders pack more features than dimensions.

### The solution (intuition)
Phase diagrams over sparsity × importance. Superposition appears in a
well-defined regime.

### Evaluation / results
Qualitative geometry (feature polytopes). No bit-capacity formula for a *set*
of concept tokens. Follow-up: privileged bases / outlier dims
([transformer-circuits.pub/2023/privileged-basis](https://transformer-circuits.pub/2023/privileged-basis)).

### Why it matters here
BAPO counts bits as if they were independent. Superposition says a slot of
width `D` can pack more than `D` sparse features *or* fewer than `D`
independent bits, depending on interference. Measured `I(text; slots)` beats
nominal `K·D`.

---

## In-context Autoencoder (ICAE)

ICLR 2024 · [arXiv:2307.06945](https://arxiv.org/abs/2307.06945) ·
Ge, Hu, Wang, Wang, Chen, Wei (Microsoft). Code:
[github.com/getao/icae](https://github.com/getao/icae). HF:
[huggingface.co/sggetao/icae](https://huggingface.co/sggetao/icae).

A one-line pointer already lives in
[`concept_modeling_encoding.md`](concept_modeling_encoding.md); this is the
**count / capacity** deep-dive.

### TL;DR
LoRA encoder + frozen Llama decoder compresses `L` tokens into `k` memory
slots. **512 → 128 (4×) is the working point**; `k=64` and `32` fail lossless
AE of 500-token text. Instruction quality keeps improving through `k=256`.
Slot *width* is never ablated (slots are full Llama hidden states).

### Forward pass (shapes)
Symbols: `L` context tokens, `k` memory slots, `d` Llama hidden size (4096 at
7B), LoRA rank 128.

```
c [B, L]                 → embed + Llama+LoRA encoder          → H [B, L, d]
append k memory tokens   → encoder attends over c ‖ mem        → M [B, k, d]
frozen Llama decoder     → condition on M instead of c         → logits
AE loss: restore c from M; LM loss: continue after M
```

Default: `L=512`, `k=128` (4×). Decoder never sees raw `c`.

### Evaluation / results (count)
- Fig. 5: `k=128` still >95% BLEU at context 500; `k=64` and `32` "much less
  satisfactory"; >4× is "rather challenging."
- Continuation PPL Δ vs original: 128→128 +0.16; 256→128 +0.32; **512→128
  +0.49**.
- Entropy stress, 512→128: normal text BLEU 99.3 / loss 0.01; patterned-random
  3.5 / 1.63; fully random 0.2 / 4.55 — slots lack capacity for high-entropy
  strings.
- Llama-2-7b-chat vs original ~512-token context (win+tie): `k=64` 48.4%;
  `k=128` 54.6%; `k=256` 77.8%.
- Pairwise GPT-4: pretrained `k=128` vs `k=64` 57.6 / 19.5 / 22.9 (ratio 3.0);
  pretrained `k=64` matches non-pretrained `k=128`.
- Stronger LLM ⇒ better compression at fixed `k` (Llama-7b ΔPPL +0.49;
  Llama-2-13b +0.30). Opposite of Broken-ELBO's "stronger decoder collapses
  rate" — here the decoder is *frozen* and cannot learn a bypass.

### Limitations
No slot-dimension sweep. Train cap 512. Later Activation Beacon LongBench-32k
shows ICAE collapsing vs uncompressed (see
[`learned_kv_context_compression.md`](learned_kv_context_compression.md)).

---

## Learning to Compress Prompts with Gist Tokens

NeurIPS 2023 · [arXiv:2304.08467](https://arxiv.org/abs/2304.08467) ·
Mu, Li, Goodman (Stanford). Code:
[github.com/jayelm/gisting](https://github.com/jayelm/gisting).

### TL;DR
Instruction prompts compress into `k` gist tokens. **Insensitive to `k`**: a
single token did not substantially underperform larger prefixes; **`k=10` can
hurt** (overfit). This is count *saturation*, not starvation — but the prompts
are ~20–26 tokens, not 512-token documents.

### The solution (intuition)
Attention mask: later tokens attend only to gist KV, not to the instruction.
The LM itself is the gist predictor.

### Evaluation / results
LLaMA-7B ChatGPT win vs uncompressed: Seen 48.6%, Unseen 49.7%, Human OOD
45.8% (50% = parity). Up to 26× prompt compression. Main results use `k=1`.

### Related
ICAE (explicitly: Gist does not solve long context). Deng et al. 2025
("Silver Bullet") stress-tests gist compression on synthetic recall — see
[`learned_kv_context_compression.md`](learned_kv_context_compression.md).

---

## DeepSeek-V2 / Multi-head Latent Attention (MLA)

Technical report · [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) ·
DeepSeek-AI. Code: [github.com/deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2).

### TL;DR
**Per-token width** of KV can be jointly compressed to a latent `c^{KV}` with
**no token-count reduction**, beating MHA at ~GQA-2.25 cache (93.3% KV cut vs
DeepSeek 67B). Strongest evidence that **attention width is overprovisioned**
at fixed count.

### Forward pass (shapes)
```
h_t [d]  → W^{DKV}  → c_t^{KV} [d_c]     # cached
c_t^{KV} → up-project → K, V
plus decoupled RoPE key k^R [d_h^R]       # also cached
cache per layer per token: (d_c + d_h^R) vs MHA 2 n_h d_h
```

V2: `d=5120`, `n_h=128`, `d_h=128`, `d_c=512`, `d_h^R=64`.

### Evaluation / results
Matched-scale MoE: MLA better than MHA on BBH/MMLU/C-Eval/CMMLU while KV is
14% (small MoE) / 4% (large MoE) of MHA. 7B dense: MHA > GQA-8 > MQA on hard
benches (MMLU 45.2 / 41.2 / 37.9) — naive width cuts *without* MLA's joint
latent hurt.

### Why it matters here
Orthogonal to E21's *time-axis* pooling (`r` tokens → 1 slot). MLA would
compress each slot's K/V *width*. The paper never varies token count; E21
never varies slot width. They are different axes.

---

## 500xCompressor

ACL 2025 · [arXiv:2408.03094](https://arxiv.org/abs/2408.03094) ·
[ACL](https://aclanthology.org/2025.acl-long.1219/) · Li, Su, Collier
(Cambridge). Code:
[github.com/ZongqianLi/500xCompressor](https://github.com/ZongqianLi/500xCompressor).

### TL;DR
ICAE-style encoder, but the decoder conditions on **per-layer KV of compressed
tokens**, not last-layer embeddings. At extreme ratios, **bits per slot (KV vs
embed) matter more than adding slots from 4 to 16**. Contexts are 96–480
tokens, not 32k.

### Evaluation / results
- Regeneration: +12–19 Rouge-L-F vs ICAE across ratios.
- **16→4 tokens: similar Rouge** (not all slots used); **4→1: clear drop**.
- QA vs ICAE, 500→1: average F1 38.98 vs 25.66. At 500→16 the two methods are
  close (45.32 vs 44.48).
- Normalized to instruct-full-context: 500→16 retains 72.89% F1; 500→1 62.26%.

### Why it matters here
E21 already stores slots in the global read's K/V space (the 500x bet), not as
decoder embeddings. That is the right carrier. Extra slots past a small
working set can be unused if the task is gist-like.

---

## Funnel-Transformer

NeurIPS 2020 · [arXiv:2006.03236](https://arxiv.org/abs/2006.03236) ·
Dai, Lai, Yang, Le (CMU / Google). Code:
[github.com/laiguokun/Funnel-Transformer](https://github.com/laiguokun/Funnel-Transformer).

### TL;DR
Pool sequence length, reinvest FLOPs in depth/width. **Moderate count
reduction + reinvested width/depth beats aggressive count cut.** Mean pool
beats learned Top-Attn pool (75.8 GLUE-AVG vs B6-6-6 83.5).

### Evaluation / results
B6-6-6 (two stride-2 pools) 83.5 GLUE-AVG; B8-8 (less pooling) 83.4; B5-5-5-5
(more pooling) **82.9**. Compressed-length layer capacity is upper-bounded by
a full-length layer; loss is compensated by stacking more cheap short layers
or widening.

### Related
Hourglass ([arXiv:2110.13711](https://arxiv.org/abs/2110.13711)): shorten by
≥3 in the middle, but **keep 2–3 vanilla full-res layers** or local
expressivity dies. `(pre,post)=(0,0)` → 1.460 BPC (broken); (2,2) → 1.128.
MEGABYTE ([arXiv:2305.07185](https://arxiv.org/abs/2305.07185)): patch size
robust over a wide range (Image256 48/192/768 → 3.178/3.158/3.186 bpb).

---

## Language Modeling Is Compression

ICLR 2024 · [arXiv:2309.10668](https://arxiv.org/abs/2309.10668) ·
Delétang, Ruoss, Duquenne, Catt, Genewein, Mattern, Grau-Moya, Wenliang,
Aitchison, Orseau, Hutter, Veness (DeepMind).

### TL;DR
A predictor is a lossless compressor via arithmetic coding, and conversely.
Cross-entropy *is* the codelength. Scaling laws still hold, but
**adjusted** compression (counting model parameters) ties optimal size to
dataset size. In-context learning is additional compression of the prompt.

### Why it matters here
E21's scientific deliverable is a **rate–distortion curve**: Δ-nats vs
bytes/token across `r`. That is this paper's unit. Do not report only
eval_loss (one ELBO-like number); report recovered bits / information_flow
against a floor, as E25 already does on DNA.

---

## Slot Attention (count vs object count)

NeurIPS 2020 · [arXiv:2006.15055](https://arxiv.org/abs/2006.15055) ·
Locatello et al. (Google). TF:
[google-research/slot_attention](https://github.com/google-research/google-research/tree/master/slot_attention);
PyTorch: [lucidrains/slot-attention](https://github.com/lucidrains/slot-attention).

### TL;DR
`K` is set to the number of objects (`K=7` CLEVR6, `Dslots=64`). Too few
slots merge objects; too many split. **`D` is not swept.** Extra slots at
train are "generally robust." Competitive softmax *over slots* is what
prevents collapse.

### Related
Iterative `T=1` vs `T=3` is reviewed in
[`latent_set_refinement.md`](latent_set_refinement.md).

---

## Already reviewed (pointers, not re-summarized)

- **AutoCompressor** `κ=50` best among `{20,50,70,100}` — count *saturation*
  for semantic summaries
  ([`recurrent_memory_transformers.md`](recurrent_memory_transformers.md)).
- **Perceiver / Perceiver-AR**: compute scales with latent count `N`, not
  input `M`; `N` traded for quality/speed.
- **Compressive Transformer / Infini-attention**: recursive *count* of
  compressed memories; Infini's HF reproduction degrades as the number of
  compressions grows.
- **BAPO**: INDEX is `(a=0, b=1)` — one raw addressable token, not a wide
  summary; MAJORITY needs `a = Ω(log n)` bits with `b=0`; MATCH3 needs
  `a·b = Ω(n)`
  ([`reasoning_bandwidth_information_flow.md`](reasoning_bandwidth_information_flow.md)).

---

## Cross-cutting (capacity, not verdicts)

1. **Task class decides the starvation mode.** Lossless AE of unique text
   (ICAE random-text BLEU 0.2) is count-starved. Gist-of-instruction is
   count-saturated at `k=1`. Extractive QA (500xCompressor 16≈4) is in
   between. E21's DNA INDEX is gist-like (a marked span); MATCH/SELECT is
   ICAE-like (many addressable items).
2. **There is no published `(K, D)` grid** on one reconstruction/LM
   objective. ICAE never varies slot width; MLA never varies token count;
   Gisting/AutoCompressor never vary `d`.
3. **A stronger *frozen* decoder improves compression at fixed `K` (ICAE);
   a stronger *trained* decoder collapses rate (Broken ELBO / E05).** E21's
   receiver is trained, so the Broken-ELBO failure is the default unless the
   raw path is cut (the message boundary) *and* the objective pays for the
   slots.
4. **Mean pool can beat a learned pooler** when the learned pooler is trained
   with the LM loss (Funnel Top-Attn 75.8 vs mean 83.5; Compressive
   Transformer conv+BPTT worse than mean-pool — see
   [`learned_kv_context_compression.md`](learned_kv_context_compression.md)).
5. **Hierarchical count reduction keeps some full-resolution tokens**
   (Hourglass vanilla layers). A pure pooled set with no skip is a harder
   bottleneck than Funnel/Hourglass/MegaByte.
