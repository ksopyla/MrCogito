# Perceiver AR, modernized — feasibility assessment for a VC-facing run

**Written:** 2026-09-06 · mutable research note (correct in place) · author: Claude Code session, reviewed by KS

**Goal under assessment.** Re-create DeepMind's Perceiver AR (Hawthorne, Jaegle et al., ICML 2022) with a
2026 training stack and 2025/26 data, at < 600M parameters, trained at very long context
(256k / 512k / 1M), and beat the original paper's numbers convincingly enough to raise a seed round.
Budget: $100K AWS credits + 50k GPU-hours on a French HPC cluster (assumed Jean Zay H100 partition).

---

## 1. What Perceiver AR actually is (paper review)

Source: [arXiv 2202.07765](https://arxiv.org/abs/2202.07765) (full text read: body + results; appendix
only partially).

**Architecture.** Input `X ∈ R^{M×C}` (M tokens, embedded + RoPE on ≤50% of channels). The **last N**
input positions become queries; one **causally-masked cross-attention** (Q from the last N tokens,
K/V from all M tokens, mask aligned so query *i* sees keys ≤ its own position) produces N latents.
Then a stack of **L causally-masked self-attention layers over the N latents only**, LayerNorm, linear
head. Loss is computed on the N latent positions only. Complexity `O(M·N) + O(L·N²)` versus `O(L·M²)`
for a decoder-only transformer. Pre-LN, squared-ReLU MLPs, Adam. Key regularizer: **cross-attend
dropout** = randomly drop a fraction of the prefix K/V positions at train time (0.5–0.97 on PG-19!).

**It is, exactly, an encoder-decoder with 0 encoder layers and 1 decoder cross-attention layer**
(lucidrains' description; the paper's own Section 4.3 says the same). The "history" (M−N tokens) is
read once, from raw token embeddings, by a single attention layer. This is the architecture's core
weakness (see §4).

**Headline results and the configs behind them**

| Benchmark | Context M | Latents N | Layers L | Params | Result | Prior SOTA |
|---|---|---|---|---|---|---|
| PG-19 (word ppl, SentencePiece 32k) | 2048 / 4096 | 1024 | 60 | **974.6M** | test **28.9** / 29.0 (val 45.9) | Compressive Tr. 33.6, Routing Tr. 33.2 |
| WikiText-103 (word ppl) | 1024–8192 | 1024 | — | — | test **18.35** @2048 (1024: 18.52, 4096: 18.25, 8192: 18.37) | Routing Tr. 15.8, Compressive 17.1 |
| Books (internal 4M books) | 1024–16384 | 1024 | 36 / 60 | 750–826M | 14.88 → 14.56 (1k → 16k); 60L@8k: 12.66 vs T-XL 42L 13.25 | — |
| ImageNet 64×64 (bits/dim) | 12,289 | 1024 | 60 | ~1B | **3.40** (750k steps) | Sparse Tr. 3.44, VDM 3.40 |
| Copy task | **131,072** | 1024 | 6 | — | 100% at 65k copy distance | — |
| MAESTRO v1 (NLL) | 4096 | 2048 | 12 | — | 1.82 | Music Tr. 1.84 |

Training details that matter for us: PG-19 was ~200k steps × batch 2048 × 1024 targets ≈ **420B target
tokens**, on TPUv3, embeddings width 4096, 128 cross-attend heads, cross-attend dropout 0.875–0.97.
The paper explicitly reports **no perplexity gain beyond 2k context on PG-19 and beyond 2k on
WikiText-103**; only the Books set showed gains up to 16k, and small ones (14.88 → 14.56).

**The bar has already moved.** [ECP / LLP (arXiv 2412.06106, Dec 2024)](https://arxiv.org/abs/2412.06106)
reports PG-19 **18.83** and WikiText-103 **17.43** with a 214.6M-param segmented Perceiver-style model at
seq 2048, versus 28.9 / 18.35 for the 974.6M Perceiver AR. Block-Recurrent Transformer (1.3B) reports
PG-19 26.5. So "beat 28.9 on PG-19 with < 600M" is very achievable; it is **not** by itself an
impressive claim in 2026. The impressive claim has to be about *context* (see §6).

## 2. Implementations compared

| Repo | Framework | Faithfulness | Kernels / perf | Status | Verdict |
|---|---|---|---|---|---|
| [google-research/perceiver-ar](https://github.com/google-research/perceiver-ar) | JAX + Haiku + jaxline | Reference (copy task config, 131k copy-task checkpoint for TPU v3-8) | TPU-oriented; no Flash kernels | Archived 2022, "not officially supported" | Read for semantics (masking, latent reset, activation caching); do **not** build on it |
| [google/flaxformer perceiver_ar](https://github.com/google/flaxformer/tree/main/flaxformer/architectures/perceiver_ar) | T5X / Flax | Reference, decoder-only + Prefix-LM; gin configs; `train_cropping_method`, `decoding_latent_reset_fill` (refill cache to `num_latents − 128`) | TPU | Maintained as part of flaxformer | Best source for **inference caching semantics** |
| [lucidrains/perceiver-ar-pytorch](https://github.com/lucidrains/perceiver-ar-pytorch) | PyTorch | Faithful & minimal: `CausalPrefixAttention` (prefix K/V + own K/V, bottom-right causal), rotary, cross-attn dropout via top-k keep mask, `perceive_depth`, `max_heads_process` chunking | Naive einsum softmax attention; abs+rotary pos; no Flash | Frozen 2022 | Cleanest reading of the forward pass; ~150 lines; unusable as-is at 256k |
| [krasserm/perceiver-io](https://github.com/krasserm/perceiver-io) | PyTorch Lightning + HF | Perceiver / IO / AR; `PerceiverAR` with `prefix_len`, KV cache, rotary (right-aligned), tied output, cross-attn dropout 0.5 default, activation checkpointing, symbolic-music checkpoints | Naive attention with `max_heads_parallel` chunking; fairscale checkpoint wrapper | Last major refactor 2023 | Most complete PyTorch reference (KV cache + pad handling); still no fused attention |
| HF `transformers` Perceiver | PyTorch | Perceiver **IO** only (no AR) | — | — | Irrelevant |

**Choice:** none is the "best implementation" for our purpose; all three PyTorch-adjacent codebases use
naive `softmax(QKᵀ)` and will not run at M ≥ 64k. The right move is to **write a new ~300-line module in
`nn/`** using lucidrains' forward pass as the spec and krasserm's KV-cache / right-aligned rotary
semantics, on top of FlashAttention-3 / FlexAttention. This repo already has the ingredients:
Perceiver cross-attention (`nn/concept_encoder_perceiver.py`), Muon, windowed-global attention
(E16b), resume-from-any-checkpoint launchers, W&B, eval harness.

**Kernel fact that makes it cheap:** FlashAttention ≥ 2.1 aligns the causal mask to the **bottom-right**
when `seqlen_q ≠ seqlen_k`. That is precisely Perceiver AR's cross-attend mask. The whole
"perceive" step is one call: `flash_attn_func(q=last_N, k=all_M, v=all_M, causal=True)` — no custom
mask, no memory for an N×M score matrix. (PyTorch SDPA `is_causal=True` is top-left aligned — wrong
here; FlexAttention needs an explicit `kv_idx <= q_idx + (KV_LEN − Q_LEN)` offset. Document masks for
packed sequences go through FlexAttention block masks.)

## 3. Compute reality check

**Budget conversion (verify the Jean Zay number).** Jean Zay Dynamic Access is capped at 50k
*normalized* GPU-hours, where **1 normalized hour = 1 h V100 = 0.5 h A100 = 0.25 h H100**
([IDRIS](http://www.idris.fr/docs/dari/demandes-heures/)). If the 50k is a Dynamic Access grant it is
**≈ 12.5k H100-hours**, not 50k. Other constraints on `gpu_p6`: ≤ 48 GPUs simultaneously, ≤ 100 h
wall-time per job (so checkpoint/resume is mandatory), `module load arch/h100`, no internet from
compute nodes (stage data via login nodes; quotas on `$SCRATCH`).

| Source | Unit price (Sep 2026) | What $100K / 50k h buys |
|---|---|---|
| AWS p5.48xlarge (8×H100 80GB) Capacity Block | $41.5/h (US) | ~2,400 node-h ≈ **19k H100-h** |
| AWS p5.48xlarge on-demand | $55.0/h | ~1,800 node-h ≈ 14.5k H100-h |
| AWS p5e.48xlarge (8×H200 141GB) Capacity Block | $47.8/h | ~2,100 node-h ≈ 16.7k H200-h (141 GB is useful for the 1M stage) |
| AWS p4de (8×A100 80GB) Capacity Block | $17.7/h | ~5,600 node-h ≈ 45k A100-h ≈ 15k H100-equiv |
| Jean Zay H100 | 50k normalized h | **12.5k H100-h** (or 50k if the grant is raw H100 hours) |

Realistic total: **~30k H100-hours** (pessimistic) to ~70k (optimistic). Keep ~10% of AWS for storage,
data staging and evaluation.

**Cost of the model itself is small.** Reference config (~540M total, "<600M"):
`d=1280, L=20–24, 10 heads×128, SwiGLU 3456, GQA 4 kv-heads in the latent stack, Llama-3 tokenizer
(128k vocab, tied embeddings 164M)` → ~400–480M non-embedding + 164M embedding. Using 6·N·D:

| Run | Tokens | FLOPs | H100-h @ 35% MFU |
|---|---|---|---|
| 540M, short-context stage (8k, cross-attend negligible) | 1T | 3.2e21 | **~2,600** |
| same | 2T | 6.5e21 | ~5,200 |
| Faithful Perceiver AR reproduction (974M, 420B targets, M=4k) | 420B | 2.5e21 | ~2,000 |
| 256k-context stage, block-swept latents (§5), 6.9 GFLOP/target | 50B | 3.5e20 | ~300 |
| 1M-context stage, block-swept, ~12 GFLOP/target | 20B | 2.4e20 | ~200 |
| Matched 540M vanilla-transformer baseline @ 8k, 1T | 1T | 3.2e21 | ~2,600 |

So **compute is not the constraint** — even the pessimistic 30k H100-h is ~5–10× what one strong
540M run needs. Spend the surplus on what makes the story credible: a scaling ladder, a matched dense
baseline, and 2–4T-token over-training so short-context benchmarks land near SmolLM2-360M /
Qwen3-0.6B (VCs will run those).

**Per-token cost of long context (why it stays cheap).** For one sequence, forward FLOPs ≈
`4·M·d²` (K/V projections of the prefix) + `4·N·M·d` (cross-attend) + `2·N·P` (latent stack) +
`4·L·N²·d` (latent self-attn) + `2·N·d·V` (head). At M=256k, N=4096, d=1280: the prefix K/V projection
is 1.7 TFLOP and the cross-attend 5.5 TFLOP versus 3.9 TFLOP for the whole 24-layer stack. At M=1M the
cross-attend is 22 TFLOP. The single global attention layer is what makes 1M affordable; a dense
24-layer transformer would pay that ×24.

**Memory at M=1M, d=1280, bf16, batch 1/GPU:** embeddings 2.7 GB, prefix K/V 5.4 GB, plus the latent
block — fits an 80 GB H100 with activation checkpointing on the cross-attend; comfortable on H200.
Inference KV cache for 1M context ≈ **5.4 GB** (one layer) versus ~130 GB for a dense 24-layer model of
the same width. That is the demo number.

## 4. The two real risks (architecture, not compute)

1. **Shallow, lossy history.** The prefix is read from *raw token embeddings* by *one* attention layer.
   That is enough for copy/retrieval (the 131k copy task) but is essentially a learned n-gram memory for
   text: the paper found **no gain past 2k on PG-19/WikiText** and only 2% on Books at 16k. A faithful
   256k Perceiver AR trained on web text will show ~0 perplexity improvement over 8k, and will fail
   RULER's multi-hop/aggregation tasks. ECP (2024) names this "Lossy History" and gets its gains by
   carrying history through every layer.
2. **Latent training dependency (sample inefficiency).** Loss is computed on N of M positions per
   sequence (1.6% at N=4096, M=256k). Every training sequence pays `O(M·d²)` to embed/project the
   prefix for a handful of targets.

Both have cheap, well-understood fixes that keep the Perceiver AR identity (§5).

## 5. Proposed modernized design ("Perceiver AR v2")

Same skeleton — long input → one causal cross-attend → deep latent stack — with four changes:

1. **Block-swept latents.** Compute prefix K/V for the full M-token sequence **once**, then run the
   latent stack over consecutive blocks of N tokens, each block cross-attending causally to the whole
   preceding sequence (FA3 bottom-right causal with a per-block key offset, or one FlexAttention block
   mask). Every token gets a loss. Cost per trained token ≈ **6.9 GFLOP at 256k, ~12 GFLOP at 1M**
   versus 3.2 GFLOP for a short-context dense model — and the model sees full 256k/1M causal context
   at every position. Inference is unchanged (latents = last N tokens, KV cache of the prefix).
2. **Local pre-encoder for the history.** 2–4 sliding-window attention layers (window 512–1024, linear
   in M) over the input before the cross-attend, so the K/V being read are contextualized rather than
   raw embeddings. Fixes "lossy history" at `O(M·w·d)` cost. Optional second cross-attend at mid-depth
   (Perceiver IO / ECP V2 style) — treat as an ablation, not the default.
3. **Modern latent stack.** RMSNorm pre-norm, QK-norm, RoPE (full, YaRN-scaled for the 512k/1M stages)
   with a NoPE fraction as in SmolLM3, SwiGLU (paper used ReLU²; keep ReLU² as a cheap ablation), GQA,
   value embeddings + U-net skip lambdas (modded-nanogpt records 14–17), zero-init output projections,
   logit soft-cap, tied embeddings.
4. **Cross-attend dropout repurposed.** The paper needed 0.875–0.97 prefix dropout because PG-19 is
   ~2B words. With trillions of tokens, set it to 0–0.25 and use it as a *compute knob* (drop 50% of
   prefix K/V positions → 2× cheaper cross-attend early in training), not a regularizer.

**Training stack (all off-the-shelf, most already in this repo):**

| Area | Choice | Why |
|---|---|---|
| Optimizer | Muon (NorMuon / Polar-Express orthogonalization) for 2-D weights + AdamW for embeddings/head/norms; cautious weight decay; WSD or cosine schedule | Repo already runs Muon (E16b); 1.3–2× token-efficiency vs AdamW at 100M–1B in speedrun evidence |
| Attention kernels | FlashAttention-3 (Hopper) for cross-attend and latent self-attn; FlexAttention for packed-document masks and the block-sweep mask | FA3 ≈ 75% of H100 peak; the bottom-right causal convention is exactly the Perceiver cross-attend |
| Fused ops | Liger kernels: fused RMSNorm, SwiGLU, RoPE, **chunked fused cross-entropy** (essential with a 128k vocab at d=1280) | Removes the logits tensor from memory; +20–30% throughput at this size |
| Compile / precision | `torch.compile`, bf16 autocast; FP8 for the head only (Gemma-2/speedrun style); skip FP8 everywhere else at d=1280 | FP8 on small matmuls buys little |
| Parallelism | DDP or FSDP2 (540M fits per GPU); no tensor parallel; sequence-parallel only for the 1M stage if H200 unavailable | 48-GPU cap on Jean Zay; keep it simple |
| Data pipeline | Pretokenize to memmap/MDS shards; packing with FlexAttention document masks; length-bucketed curriculum | Repo already has pretokenization; ProLong ships pre-packed 512k MDS |
| Loss | CE + z-loss 1e-4; optional MTP auxiliary head (ablation) | Stability at high LR with Muon |

## 6. Context length: recommendation

**Train at 256k, extend to 512k in a short final stage, demonstrate 1M at inference.**

| | 256k | 512k | 1M |
|---|---|---|---|
| Natural documents at that length | Rare but real: long books (~120–150k tok), repo-level code, arXiv bundles; ProLong-512K is pre-packed | Almost only packed/concatenated sets | Essentially synthetic concatenations |
| Memory / batch per H100 | batch > 1 per GPU | batch 1, checkpointing | needs H200 or seq-parallel |
| Eval coverage | RULER/NIAH/LV-Eval/InfiniteBench all cover it | RULER (NVIDIA extension), BABILong | RULER-1M, NIAH-1M only |
| Vision fit | training signal for real long-range structure | marginal extra signal | the marketing number |

Honest VC claim: "0.5B model, trained with 256k causal context on every token, verified to 1M on
RULER/NIAH, 1M-token KV cache of 5 GB on one GPU." The paper's own copy task (131k, 100%) is a cheap,
dramatic warm-up demo (6 layers, 25k steps) that should be reproduced first on Odra.

## 7. Data (do not spend time here — use these)

| Role | Dataset | Size | License note |
|---|---|---|---|
| General web (bulk of tokens) | [nvidia/Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1): High-Quality 26B + Medium-High 16.9B organic, High-Quality-Synthetic 93.5B, HQ-Translated 39.6B; add Nemotron-CC-v2 HQ buckets if > 300B needed | 2.5T total; use ~300–500B | NVIDIA Data Access Agreement — "ready for commercial use"; synthetic subsets carry Qwen/DeepSeek/Phi-4 pass-through terms |
| Code | Nemotron-CC-Code-v1 (427.9B, phi-4 cleaned) + Nemotron-Pretraining-Code-v2 (340B GitHub files; has `commit_id`/`rel_path` for repo-level regrouping into long sequences) | 100–200B | same agreement |
| STEM / reasoning | Nemotron-Pretraining-Specialized-v1 (RQA 134.6B, Math-Textbooks 25.1B, STEM-SFT 82.5B) | 30–60B | CC-BY-4.0 (Wiki-Rewrite CC-BY-SA) |
| **Long-context stage** | [princeton-nlp/prolong-data-512K](https://huggingface.co/datasets/princeton-nlp/prolong-data-512K) (31B tokens, books + code repos + arXiv + short data, packed to 524,288, Llama-3 tokenizer, MDS) and prolong-data-64K (31B) | 31B + 31B | research release; check terms before commercial use |
| Benchmark-domain | PG-19 train (Apache 2.0) in the mix at ~1–2%; WikiText-103 train | 2B words | decontaminate PG-19 test (100 books) and WikiText test from every CC-derived shard |
| **Avoid** | [nvidia/Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) / ClimbLab | 400B / 1.2T | **CC-BY-NC-4.0** — non-commercial; fine for a scaling-ladder ablation, not for the product model. Also pre-tokenized with GPT-2 |

Tokenizer: Llama-3 (128,256) — matches ProLong, modern code/math coverage; costs 164M embedding params
at d=1280 (tied). If the < 600M line must exclude nothing, drop to L=20 (≈560M total).

Curriculum (ProLong / OLMo-3 evidence: 0.5–5B long tokens suffice to unlock retrieval; short data
quality still dominates downstream scores): stage 1 ≈ 1–2T tokens at 8k (block-swept, cheap);
stage 2 ≈ 50B tokens at 256k (34% long / 66% short mix, document masks); stage 3 ≈ 10–20B at 512k
(ProLong-512K + repo-level code) with YaRN; 1M by extrapolation + a few B tokens on H200 if it
pays off on RULER-1M.

## 8. Evaluation plan (what the deck shows)

1. **Paper parity table:** PG-19 word-ppl (total NLL / word count, so any tokenizer is comparable, as
   MEGABYTE did), WikiText-103 word-ppl, copy task @131k, optional ImageNet-64 bits/dim (byte-level
   run, only if cheap). Target: < 25 on PG-19 and < 17 on WikiText-103 at ~540M, beating both the
   974M Perceiver AR and the 214M LLP.
2. **Long context:** RULER 4k → 1M (NVIDIA's extended protocol), NIAH-1M grid, LV-Eval (≤256k),
   InfiniteBench, BABILong-512k, HELMET subset; plotted against Qwen2.5-1M-7B, Llama-3.2-1B/3B,
   SmolLM3-3B (128k) and our own matched dense 540M baseline.
3. **Short context (table stakes):** lm-eval-harness HellaSwag, ARC-E/C, PIQA, MMLU, GSM8K vs
   SmolLM2-360M, Qwen3-0.6B, Llama-3.2-1B.
4. **Efficiency:** prefill time and KV-cache bytes at 128k/256k/1M vs a dense model on one GPU.
5. **Ablations that prove it is the architecture:** dense 540M at same tokens/data; Perceiver AR
   *faithful* (no pre-encoder, no block sweep) vs v2; pre-encoder depth 0/2/4.

## 9. Implementation checklist (repo-rooted)

- `nn/perceiver_ar_lm.py` — `PerceiverARLM(config)`: token embed (tied), optional local pre-encoder
  (SWA via FA3 `window_size`), one causal cross-attend (FA3 bottom-right), latent stack with GQA /
  QK-norm / value-embeddings / U-net lambdas, block-sweep training path, KV-cache inference path with
  latent reset (flaxformer semantics).
- `data/` — pretokenizer for Nemotron parquet → memmap; ProLong MDS loader; repo-level code
  regrouping; packing + FlexAttention doc-mask builder (cache `create_block_mask` per length bucket).
- `training/` — reuse `train_concept_pretraining.py` family + Muon; add block-sweep loss reduction,
  z-loss, WSD schedule, length-bucket curriculum; Slurm launcher for Jean Zay (100 h chunks,
  `arch/h100`, resume) alongside the existing Odra launchers.
- `evaluation/` — PG-19/WikiText word-ppl, copy-task, RULER/NIAH runner, lm-eval-harness wrapper.
- `tests/` — mask equivalence (naive vs FA3 vs FlexAttention at seqlen_q ≠ seqlen_k), KV-cache vs
  full-recompute equality, block-sweep vs single-window loss equality, 131k copy task smoke test.

**Timeline (2 people):** weeks 1–2 module + tests + copy-task reproduction on Odra (1 GPU);
week 3 scaling ladder 125M @ 8k/32k/128k on AWS spot + dense baselines — **kill criterion:** if
128k context gives < 2% ppl gain over 8k on long-document eval *and* NIAH@128k < 90%, fix the reader
before spending the big budget; weeks 4–6 main run on Jean Zay (48×H100, ~5–8k H100-h);
week 7 long-context stages + 1M demo on H200; week 8 evals + deck. Compute spend ≈ 15–20k H100-h in
total, inside the pessimistic budget.

## 10. Verdict

Feasible, and cheaper than it looks: one strong 540M run is ~3–5k H100-hours, the budget is 30–70k.
The risk is not compute or engineering; it is that a *faithful* Perceiver AR shows no benefit from
256k context on text (the original paper says so). The block-swept, pre-encoded v2 keeps the
Perceiver AR identity, trains on every token, reads a contextualized history, and delivers the two
numbers a VC will remember: full-context training at 256k on a 0.5B model, and a 1M-token KV cache of
~5 GB. Verify the Jean Zay hour normalization first; it changes the plan from "lavish" to "adequate".
