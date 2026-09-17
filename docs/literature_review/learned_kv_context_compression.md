# Learned KV / context compression — memory tokens, pooling, and 8k–32k behaviour

Reviews of methods that replace a long token prefix with **fewer keys and
values** the rest of the model attends to: gist / memory tokens, in-context
autoencoders, activation beacons, compressive caches, and training-free KV
eviction. Home for the question *how compression architectures behave as
sequence length grows to 8k–32k*, and which evals detect long-range use of
the compressed object rather than local shortcutting.

E21's object is closest to **Activation Beacon / Fine-KV gists**: slots live
in the global-read K/V space, one per `r` tokens, and the receiver cannot
see raw prefix keys. It is **query-agnostic** (one message, many future
queries), unlike SnapKV/H2O.

Related reviews (do not duplicate):
- AutoCompressor, Perceiver-AR, Landmark, Infini-attention, Compressive
  Transformer (writable-memory axis):
  [`recurrent_memory_transformers.md`](recurrent_memory_transformers.md)
- Count vs width / rate–distortion:
  [`information_bottleneck_latent_capacity.md`](information_bottleneck_latent_capacity.md)
- RULER / BABILong / BAPO exams:
  [`synthetic_capability_exams.md`](synthetic_capability_exams.md)
- CEPE (encoder-chunk + decoder cross-attn) one-liner:
  [`concept_modeling_encoding.md`](concept_modeling_encoding.md)

---

## Learning to Compress Prompts with Gist Tokens

NeurIPS 2023 · [arXiv:2304.08467](https://arxiv.org/abs/2304.08467) ·
Mu, Li, Goodman (Stanford). Code:
[github.com/jayelm/gisting](https://github.com/jayelm/gisting).

### TL;DR
A few gist tokens cache an *instruction* prefix. Works at `k=1` for ~20-token
prompts. Not a 8k–32k compressor; included because later gist-KV architectures
inherit the mask.

### The solution (intuition)
Insert `k` gist tokens between task and input; mask so the input cannot attend
to the instruction, only to gist KV. At inference, cache gist KV and drop
prompt tokens.

### Evaluation / results
LLaMA-7B ≈ uncompressed on seen/unseen Alpaca+ (ChatGPT win ~49%). `k=10` can
hurt. **No 8k–32k eval.**

---

## In-context Autoencoder (ICAE)

ICLR 2024 · [arXiv:2307.06945](https://arxiv.org/abs/2307.06945) ·
Ge et al. (Microsoft). Code: [github.com/getao/icae](https://github.com/getao/icae).

Count/capacity numbers: [`information_bottleneck_latent_capacity.md`](information_bottleneck_latent_capacity.md).
This entry is the **length / objective** reading.

### TL;DR
LoRA encoder produces `k` last-layer memory slots a *frozen* LLM conditions
on. Default 512→128. AE + continuation pretrain, then PwC instruction FT.
**AE-only or LM-only each lose to AE+LM.** Concatenating slots needs a few
concat examples in training. Train length 512; Beacon's LongBench-32k later
shows ICAE collapsing vs uncompressed.

### Why learning can help here
The decoder is frozen, so LM gradients cannot rewrite a bypass around the
slots. Pretrained `k=64` ≈ untrained `k=128`. That is the opposite of E21's
*trained* receiver + learned `u`/`delta` under CE, where E25 saw learned
pooling destroy INDEX that frozen mean carried.

---

## Long Context Compression with Activation Beacon

ICLR 2025 · [arXiv:2401.03462](https://arxiv.org/abs/2401.03462) ·
Zhang, Liu, Xiao, Shao, Ye, Dou (BAAI / Renmin). OpenReview:
[1eQT9OzfNQ](https://openreview.net/forum?id=1eQT9OzfNQ). Code lives in
[FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding).

### TL;DR
Compress **per-layer K/V** into interleaved *beacon* tokens, progressively
over chunks. Discard raw KV after each chunk; concat beacon KV as the growing
coarse cache. Trained ≤20k, still hits NIAH at 128k. Soft-prompt compressors
(AutoCompressor / ICAE) collapse at 32k on LongBench; beacons match full FT.
**Interleaving beats dumping all beacons at the chunk end** (40.5 → 35.2
single-doc 32k ×4).

### Forward pass (shapes)
Symbols: chunk size `w` (1024 Llama-2 / 2048 Qwen-2), ratio `α`, beacons per
chunk `k = w/α`.

```
X [n] split into chunks X_i of length w
for each chunk:
  split into units of α tokens; interleave one beacon after each unit
  encode chunk with extra W_Q/K/V^b; beacons attend with differentiated scope
  discard raw KV of X_i; append beacon KV to cache
next chunk attends to accumulated beacon KV as proxy for X_≤i
```

Train `α ~ Unif{2,4,8,16,32}`. Loss: next-token on raw tokens from chunk 2
onward (no stop-grad across chunks). Frozen LLM + ~1B RedPajama + LongAlpaca /
BookSum / synthetic QA.

### Evaluation / results
- LongBench 32k Llama-2 adaptive ×2/×4/×8: Beacon single-doc **34.9** vs
  Full-FT 34.8 vs AutoCompressor **12.9** vs ICAE **19.5** vs LongLLMLingua
  21.5 vs SnapKV (capped 4k window) 24.2.
- Multi-needle 32k/128k, ×8, 1–3 turns: Llama-2 32k Acc 9.75/9.40/9.05 vs
  Full-FT 9.75/9.45/9.10; AutoCompressor ~1.5; ICAE ~2.1; **SnapKV 1.00**
  (query-dependent, fails reuse).
- Ablation Single-Doc 32k ×4: default 40.5; beacons at chunk **end 35.2**;
  instance-wise `α` 37.7; no PT 34.9; no FT 35.5.

### Limitations
Still grows linearly with `n/α`. NIAH scored by ChatGPT 1–10. No RULER table.
LongBench-v1 can be locally shortcut.

### Mapping onto E21
E21 pools *complete homogeneous blocks* and (by default) writes slots at block
ends / in-place replace positions — closer to Beacon's *end-append* ablation
than to interleaving. Beacon's 5-point drop is the published reason to prefer
**interleaved slot placement** over "one slot at the end of `r` tokens" if
the goal is long, detailed context rather than a span gist.

---

## 500xCompressor

ACL 2025 · [arXiv:2408.03094](https://arxiv.org/abs/2408.03094) ·
Li, Su, Collier. Code:
[github.com/ZongqianLi/500xCompressor](https://github.com/ZongqianLi/500xCompressor).

### TL;DR
Same as ICAE but the frozen decoder reads **per-layer KV of the compressed
tokens**. KV carriers beat embeddings at 500→1. Not a 32k paper (96–480 token
passages). Capacity reading:
[`information_bottleneck_latent_capacity.md`](information_bottleneck_latent_capacity.md).

### Why it matters here
E21 already uses the KV carrier. The paper's lesson is "do not store the
message as a last-layer embedding."

---

## A Silver Bullet or a Compromise for Full Attention? (gist-token failure analysis)

ACL 2025 · [arXiv:2412.17483](https://arxiv.org/abs/2412.17483) ·
[Anthology](https://aclanthology.org/2025.acl-long.241/) · Deng, Zhang, Mao,
Li, Huang, Yu, Dou (Renmin / Tencent).

### TL;DR
Fine-grained KV gists are **near-lossless on RAG / LongQA / summarization** at
4× vs full attention, and **not a substitute on rerank or synthetic recall**.
Three failure patterns: **lost by the boundary**, **lost if surprise**,
**lost along the way**. Fine-grained autoencoding (weak decoder reconstructing
the segment) and segment-wise token-importance weighting recover **+52.7% /
+33.7% relative** on synthetic recall at 4×. Coarse (append-all-gists) wastes
the memory budget at every ratio.

### Architectures compared
- Coarse-Recurrent: gist tokens appended after the segment (AutoCompressor-like).
- Coarse-KV: same placement, but condition on gist **KV**.
- Fine-KV: gists **interleaved** (Beacon-style); condition on KV.

### Evaluation / results (Table 2, Llama-scale long-context suite)
Full attention: RAG 61.8, Rerank 39.9, LongQA 41.6, ICL 62.3, **Synthetic
93.9**, Summ 23.8, Code 66.1, avg 55.6.

| ratio | Fine-KV RAG | Rerank | Synthetic | avg |
|---|---|---|---|---|
| 4× | 60.6 | 23.4 | 40.6 | 46.2 |
| 8× | 57.6 | 14.5 | 26.9 | 40.7 |
| 16× | 55.4 | 10.0 | 13.8 | 34.9 |
| 32× | 53.1 | 3.1 | 11.9 | 31.0 |

Coarse-KV at 4× is already Synthetic **14.2** / Rerank **5.2**. Short-context
MMLU-Pro / GSM8K barely move — gist compression does not show up on local
exams.

### Failure patterns
1. **Lost by the boundary** — generation degrades near the start of a
   compressed segment (the slot has not yet accumulated the span).
2. **Lost if surprise** — unexpected / high-entropy details are dropped when
   the budget is tight (ICAE's random-text BLEU 0.2 is the same phenomenon).
3. **Lost along the way** — exact-copy / long rehearsal accumulates errors
   midway.

### Mitigations
Fine-grained AE (reconstruct original tokens from gists with a *weak*
decoder) + segment-wise importance. At 4×, synthetic recall 40.6 → 62.0
(+21.4 abs, +52.7% relative) with AE. Gains shrink at 16×–32× — the
information is gone, not just poorly read. Long-range SFT (LongAlpaca +
BookSum + synthetic) specifically lifts the previously weakest task
(synthetic recall) without hurting RAG.

### Why it matters here
This is the closest published autopsy of E21's DNA walls: INDEX (marked
span) can survive 16-token means; MATCH/SELECT (surprise + exact
rehearsal) die under pooling. The fix they measure is **not** "learn the
pooler harder under CE" — it is **reconstruction of the prefix from the
slots** plus **up-weighting long-range-dependent tokens**.

---

## Compressive Transformers for Long-Range Sequence Modelling

ICLR 2020 · [arXiv:1911.05507](https://arxiv.org/abs/1911.05507) ·
Rae, Potapenko, Jayakumar, Hillier, Lillicrap (DeepMind). PyTorch port:
[lucidrains/compressive-transformer-pytorch](https://github.com/lucidrains/compressive-transformer-pytorch).

Family placement: [`recurrent_memory_transformers.md`](recurrent_memory_transformers.md)
§C. This entry is the **learned-vs-frozen-pool** ablation.

### TL;DR
Don't discard Transformer-XL memories; compress them into a second FIFO.
**Conv + BPTT through the LM loss is worse than frozen mean-pool** (0.996 vs
0.982 Enwik8 BPC). Conv + attention-reconstruction aux is best (0.973).
Learning the compressor with the LM objective can *hurt*; a local
reconstruction aux can help.

### Evaluation / results (their Table 5)

| compressor | loss | Enwik8 BPC |
|---|---|---|
| Conv + BPTT (LM grads) | BPTT | **0.996** |
| Max pool | none | 0.986 |
| Conv + AE | AE | 0.984 |
| **Mean pool** | none | **0.982** |
| Most-used | none | 0.980 |
| Dilated conv + attn recon | attn | 0.977 |
| Conv + attn recon | attn | **0.973** |

Attention mass *rises* at the fine→compressed boundary. Rare-word PPL gain
~20% vs frequent ~2.6%.

### Why it matters here
E25's "learned pooling wrecks INDEX that frozen mean carries" is this table
on DNA. The published repair is an **attention-match / AE aux**, not more LM
steps through `u`/`delta`.

---

## SnapKV

[arXiv:2404.14469](https://arxiv.org/abs/2404.14469) · Li, Huang et al.
(UIUC / Cohere / Princeton). Code:
[github.com/FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV).

### TL;DR
**Training-free, query-aware** eviction: the prompt tail votes for clustered
important prefix KV per head. Strong on single-query NIAH (LWM-1M to 380k
with 1024-token cache). **Fails when the cache must be reused for a later
unseen query** (Beacon 3-turn Acc=1.00; Compactor query-agnostic RULER).

### Why it is not E21
E21's message is produced *before* the receiver's query exists (agent
handoff / suffix after a boundary). Query-aware eviction is a different
product. Use SnapKV as the foil that looks good on NIAH and dies on
query-agnostic exams.

---

## Compactor (query-agnostic leverage scores)

[arXiv:2507.08143](https://arxiv.org/abs/2507.08143).

### TL;DR
Training-free **query-agnostic** eviction via approximate leverage scores, so
one compressed cache serves future queries. They evaluate **RULER-4k on
purpose** (full-cache is already strong; they want retention vs quality, not
length). 75% retention → 93.8% of baseline; 50% → 87.6%; 10% → 59.5%.
SnapKV/H2O/PyramidKV degrade ~linearly in the query-agnostic regime. NIAH vs
QA support very different compression rates.

### Limitations
Not a 32k RULER table — authors note models are already poor at higher
lengths. Still heuristic.

---

## KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

NeurIPS 2025 Oral · [arXiv:2505.23416](https://arxiv.org/abs/2505.23416) ·
Code: [github.com/snu-mllab/KVzip](https://github.com/snu-mllab/KVzip).

### TL;DR
Score KV by usefulness for **reconstructing the context**, then evict.
Training-free. 3–4× KV cut, ~2× decode, eval includes SCBench (RULER +
∞Bench) at 100–170k with **multi-query per context**. Reconstruction scoring
is the query-agnostic cousin of Deng's fine-grained AE.

---

## DuoAttention and retrieval heads

DuoAttention, ICLR 2025 · [arXiv:2410.10819](https://arxiv.org/abs/2410.10819)
· Xiao et al. (MIT HAN Lab). Code:
[github.com/mit-han-lab/duo-attention](https://github.com/mit-han-lab/duo-attention).

Retrieval Head Mechanistically Explains Long-Context Factuality ·
[arXiv:2404.15574](https://arxiv.org/abs/2404.15574) · Wu et al.

### TL;DR
A sparse set of **retrieval heads** (<5%, DuoAttention: 25% of Llama-2-7B-32K
heads, 50% of Llama-3-8B-1M) implement long-context lookup. Prune them →
NIAH/CoT fail; prune random heads → little effect. Same heads persist after
32–128k continued pretrain. DuoAttention keeps full KV only on retrieval
heads; streaming heads keep sink+recent.

### Why it matters here
Uniform `r=16` pooling across *all* heads can destroy the retrieval-head
keys while streaming heads never needed them. Head-heterogeneous compression
is an Adapt; uniform pooling is what E21 does today.

---

## CEPE — Long-Context Language Modeling with Parallel Context Encoding

ACL 2024 · [arXiv:2402.16617](https://arxiv.org/abs/2402.16617) ·
Yen, Gao, Chen (Princeton). Code:
[github.com/princeton-nlp/CEPE](https://github.com/princeton-nlp/CEPE).

Existing one-liner: [`concept_modeling_encoding.md`](concept_modeling_encoding.md).

### TL;DR
Small encoder processes extra context in 256-token chunks; frozen decoder
reads them via added cross-attention. Train at 8k, infer to 128k. **PPL on
the last 256 tokens continues to improve 8k→32k→128k** while position-
extended Llama-2-32K / YaRN-64K break. ~10× throughput, ~1/6 memory. **10
sequences at 128k** — noisy. No RULER/NIAH in the LM table.

### Why PPL-last-256 is a bad 32k gate
The metric can look "good" while INDEX/MATCH through compressed slots is
dead. CEPE is a useful *engineering* existence proof that chunked encoding
extrapolates; it is not an exclusive-channel exam.

---

## Native Sparse Attention (NSA)

[arXiv:2502.11089](https://arxiv.org/abs/2502.11089) · DeepSeek.

### TL;DR
Train sparse attention that mixes **compressed global tokens**, dynamically
selected fine blocks, and a sliding window. Pretrain then continue at 32k +
YaRN. Claims match full attention with 9×/6×/11.6× fwd/bwd/decode at 64k.
This is **trained sparse attention**, not a drop-in compressor.

---

## StreamingLLM (sink + window)

[arXiv:2309.17453](https://arxiv.org/abs/2309.17453) · Xiao et al.
Code: [github.com/mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm).

### TL;DR
4 attention sinks + a local window. PPL stable to 4M. Stabilizes **local LM**,
not random-access long-range use. Llama-2-13B cache 1024: window PPL 5158 vs
Streaming 5.40. **Reject as a test of compressed-context addressing.**

---

## LongLLMLingua (hard token deletion)

[arXiv:2310.06839](https://arxiv.org/abs/2310.06839) ·
[github.com/microsoft/LLMLingua](https://github.com/microsoft/LLMLingua).

### TL;DR
Query-aware *hard* token deletion. LongBench GPT-3.5 can *improve* at 3×/6×
because noise is dropped. Beacon 32k: far behind learned KV beacons.
Recompute per question — not a query-agnostic message.

---

## Infini-attention (in-place overwrite) — pointer

[arXiv:2404.07143](https://arxiv.org/abs/2404.07143). HF reproduction:
[huggingface.co/blog/infini-attention](https://huggingface.co/blog/infini-attention).
Full review: [`recurrent_memory_transformers.md`](recurrent_memory_transformers.md) §D.

### TL;DR
In-place linear-attention memory. HF: **long-context performance decreases as
the number of times we compress the memory**. Concat FIFO (Beacon / CT)
scales until slot count hits the window; in-place overwrite is the failure
mode to not copy.

---

## Cross-cutting (compression vs length, not verdicts)

1. **Length scaling is task-class, not a PPL curve.** Vanilla NIAH saturates;
   RULER/HELMET drop 4k→32k→128k on multi-key, variable tracking, aggregation,
   re-ranking, citation. Compressors that look lossless at 32k on NIAH or
   LongBench-v1 often fail query-agnostic RULER or multi-turn reuse.
2. **Soft prompts saturate; per-layer KV carriers scale further** (Beacon vs
   AutoCompressor/ICAE at 32k).
3. **Learned pooling can lose what frozen pooling keeps** when the learning
   signal is LM CE (CT conv+BPTT; E25 learned `u`/`delta`; Funnel Top-Attn).
   Repair: reconstruction / attention-match aux, or freeze the pooler.
4. **Interleave > end-append** at matched slot count (Beacon, Fine-KV).
5. **Query-agnostic vs query-aware** is a product split. E21 is query-agnostic.
6. **In-place overwrite vs concat FIFO:** Infini degrades with more compress
   steps; Beacon/CT concat scales. E21 exclusive in-place slots with a
   message boundary are not tested in this literature.
7. Protocols that actually detect long-range use of compressed context are
   collected in [`synthetic_capability_exams.md`](synthetic_capability_exams.md)
   (HELMET, LongBench v2 no-context, retrieval-head knockout, swapped-message).
