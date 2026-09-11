# Long-context architectures and training — what makes a long-range channel load-bearing

Topical review (2025 → Sep 2026) organised around the question E18 forced on us: **next-token
loss on natural text gives a long-range channel almost no gradient** (E18 arm A vs C: a single
global read is *used* but adds −0.001 nats; the dense control shows a 0.035–0.10 nat prize exists
only when *every* layer has reach). What does the field do about it, and what does it say about
compressing prefix K/V into fewer slots?

Collected 2026-09-11 (research-scout sweep). Each item: mechanism · key number · URL. Items not
re-read directly are the scout's summary. Related: [`frontier_open_models_architecture.md`](frontier_open_models_architecture.md),
[`reasoning_bandwidth_information_flow.md`](reasoning_bandwidth_information_flow.md) (BAPO),
[`recurrent_memory_transformers.md`](recurrent_memory_transformers.md),
[`concept_modeling_encoding.md`](concept_modeling_encoding.md) (ECP/LLP, Memory Transformer).

---

## 1. Sparse / global attention and hybrids

- **NSA** (DeepSeek) — compressed coarse tokens + fine top-k selection + sliding window in one
  kernel; matches full attention; 9×/6×/11.6× fwd/bwd/decode at 64K. https://arxiv.org/abs/2502.11089
- **DSA** (DeepSeek-V3.2) — a *lightning indexer* scores all prior keys, keeps **top-2048** per
  query over the MLA cache; O(L·k); parity with V3.1. https://arxiv.org/abs/2512.02556 — the
  cheapest way to make *addressing* a trained, differentiable sub-module in front of a read.
- **MoBA** — block-level top-k routing (MoE-style over KV blocks); deployed at Kimi; needs
  continued training. https://arxiv.org/abs/2502.13189
- **Kimi Linear / KDA** — 3:1 KDA:MLA; 75% KV reduction, 6× decode at 1M; NoPE on full layers. https://arxiv.org/abs/2510.26692
- **Qwen3-Next / Qwen3.8** — 3:1 Gated DeltaNet:full; 256K native. https://vllm.ai/blog/2025-09-11-qwen3-next ·
  Qwen3.8-Flash-Next report: SWA-hybrid loses to GDN-hybrid on 7/9 benchmarks. https://arxiv.org/html/2608.30320
- **Gemma 3** — 5:1 SWA-1024:global; only globals see 128K; ~5× KV saving. https://arxiv.org/abs/2503.19786
- **Llama 4 Scout iRoPE** — 3:1 RoPE:NoPE layers; 10M claimed, pretrained at 256K. https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- **Command A** — 3:1 SWA-4096-RoPE : full-NoPE; long SFT interleaves 16K/256K 3:1. https://arxiv.org/abs/2504.00698 ·
  design basis **RNoPE-SWA** https://arxiv.org/html/2501.18795v2
- **MiniMax M2** — *rejected* hybrids and returned to full GQA everywhere: SWA variant scored 72
  vs 90 on 128K RULER-CWE; retrieval heads fix early in pretraining. https://arxiv.org/html/2605.26494v2
- **Attention sinks** emerge with context length; removing BOS hurts RULER; Qwen3-Next's gated
  attention is sink-free. https://arxiv.org/html/2504.02732v4

## 2. Recurrent / state memory

- **Titans** — test-time neural memory + local attention; >2M NIAH. https://arxiv.org/abs/2501.00663
- **MIRAS** — unifies attention bias, retention, online update (Moneta/Yaad/Memora). https://arxiv.org/abs/2504.13173
- **Atlas** — non-online memory update over past tokens; 80% BABILong @10M vs Titans ~70%. https://arxiv.org/abs/2505.23735
- **Gated DeltaNet** — gating + delta rule; beats Mamba2 on in-context retrieval. https://arxiv.org/abs/2412.06464
- **TTT-E2E** — SWA base + meta-learned test-time next-token training into the last ¼ of MLPs;
  a 3B model matches full-attention scaling, 2.7× faster at 128K. https://arxiv.org/abs/2512.23675
- **RWKV-7** — constant-state delta rule; passkey to ~35K at 2.9B; 128K FT unreliable. https://arxiv.org/abs/2503.14456 ·
  **RWKV-X** adds sparse attention → near-perfect 64K passkey after 64K CPT. https://arxiv.org/abs/2504.21463

## 3. Prefix / KV → fewer or smaller slots

- **AutoCompressor** — ~50 summary vectors per segment, 6K–30K; weak on synthetic recall
  ("lost by boundary/surprise"). https://arxiv.org/abs/2305.14788
- **Activation Beacon** — per-layer KV "beacon" tokens; **8× KV cut at ≈ baseline @128K**; trained
  with *variable* compression ratios. https://arxiv.org/abs/2401.03462
- **ICAE** — learned memory slots, **4×** compression, ~1% extra params on Llama. https://arxiv.org/abs/2307.06945
- **MLA** — low-rank latent per token; 93.3% KV cut (V2). https://arxiv.org/html/2405.04434
- **KVzip** — query-agnostic eviction via reconstruction; 3–4× cache, ~2× decode. https://arxiv.org/abs/2505.23416
- **Compactor** — training-free leverage-score eviction; 68% KV cut on LongBench. https://arxiv.org/html/2507.08143
- **DeepSeek-V4.1-Flash** — global cache **890 B/token** via FP4 + cross-layer KV/index sharing +
  encoder-projected decoder KV (byte compression, not slot-count compression). https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

## 4. Why next-token prediction fails to train long-range use — and remedies

- **Retrieval heads** — <5% of heads do copy/retrieve; they are intrinsic already in short-context
  pretraining; pruning them causes hallucination. https://arxiv.org/abs/2404.15574
- **RetMask** — DPO on outputs generated with retrieval heads masked; +2.28 HELMET @128K. https://arxiv.org/html/2601.11020v3
- **SEAL** — head/channel scales tuned on **50 synthetic** format-matched samples in <1 h. https://arxiv.org/abs/2501.15225
- **LongFilter** — per-token KL(long-context vs short-context prediction) to *filter* pretraining
  data: many "long" documents contain no long-range dependency. https://arxiv.org/pdf/2510.25804
  → this is the data-side version of our reach ablation.
- **NExtLong** — synthesise long docs by interleaving hard-negative distractors between
  meta-chunks; beats real long-document synthesis on HELMET/RULER. https://proceedings.mlr.press/v267/gao25n.html
- **ProLong** — books + code repos + 37% ShortMix; 60/40 long/short optimal; synthetic
  long-context data in *SFT* hurts even at 1%. https://arxiv.org/abs/2410.02660
- **Token weighting** — non-uniform NTP loss emphasising long-range-dependent tokens. https://arxiv.org/html/2503.09202
- **Nemotron 3 Nano** — CPT at 512K *with synthetic retrieval / multi-hop / aggregation data*;
  RULER **86.3 @1M** (3.2B active MoE). https://arxiv.org/html/2512.20856

## 5. ≤ 3B → 1M recipes and positions

- **Qwen2.5-1M (7B/14B)** — 256K SFT + DCA extrapolation; RULER 95.7% @128K (14B). https://qwenlm.github.io/blog/qwen2.5-1m/
- **Gradient Llama-3-8B-1048k** — progressive RoPE θ to 2.8B; only 1.4B extension tokens. https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k
- **SmolLM3-3B** — 64K pretrain + YaRN → 128K; RULER 61.0 @128K; NoPE every 4th layer; extra
  long-doc upsampling did *not* help. https://huggingface.co/HuggingFaceTB/SmolLM3-3B · https://huggingface.co/blog/smollm3
- **ProLong-8B** — 64K → 512K, RoPE θ 8M → 128M, 40B tokens; SOTA @128K at the time. https://arxiv.org/abs/2410.02660

## 6. Evaluation

- **RULER** (13 synthetic tasks; half of "128K" models fail at 32K; weak downstream predictor) https://arxiv.org/abs/2404.06654 ·
  **RULER-1M** https://github.com/NVIDIA/RULER
- **HELMET** (7 application categories to 128K; NIAH ≠ downstream) https://arxiv.org/abs/2410.02694
- **LongBench v2** (503 MCQ, 8K–2M words; humans 53.7%, best LLM 50.1%) https://arxiv.org/abs/2412.15204
- **BABILong** (20 reasoning tasks to 10M; GPT-4 < 40% @100K) https://arxiv.org/abs/2406.10149
- **MRCR** (multi-round co-reference, the number Qwen/DeepSeek report at 256K–1M).

---

## Synthesis for [REDACTED]

**(a) Convergence.** Hybrids at 3:1–5:1 local:global dominate; the local mixer is drifting from
SWA to gated linear attention (GDN/KDA); global layers are sparse-indexed (top-2048) and
byte-compressed; long training = long books/code + high-quality short mix, train ≥ eval length,
synthetic retrieval in *continued pretraining* (Nemotron) but not in SFT (ProLong); NoPE or partial
RoPE on the global layers with θ/YaRN scaling. Eval = RULER + HELMET + LongBench v2 (+ MRCR).

**(b) Making one read load-bearing — evidence-backed levers**, in order of cost:
1. **Dense retrieval supervision in the mix** (E18b's premise): Nemotron 3 Nano, NExtLong,
   SEAL/RetMask all show retrieval circuits are trainable from small, dense synthetic signal.
2. **A learned indexer/selector in front of the read** (DSA/QSA/MoBA): makes addressing an
   explicit trained module and gives the 10M path a sub-quadratic read. Cheapest "make addressing
   trainable" baseline to compare E18b against.
3. **Loss reweighting toward long-range-dependent tokens** (LongFilter/token weighting): the
   data-side twin of our reach probe — compute per-token Δ(short vs long context) and upweight.
4. **TTT-style write into weights** if the read must stay read-only (TTT-E2E).
5. **A GDN local mixer** so the local channel carries a gist state and the read is freed for exact
   retrieval / message duty (Qwen3.8-Flash-Next ablation).

**(c) Compressing prefix K/V to ~1/16 slots.** Direct 16× evidence is thin but bracketed: ICAE 4×
with good QA; Activation Beacon 8× near baseline @128K *when trained with variable ratios*;
gist/soft slots fail synthetic exact recall; post-hoc eviction gives 3–4×. Expect **copy fidelity
to drop before semantic/RAG quality** at 16× unless slots are trained with copy/retrieval
objectives — exactly E18c's design (warm-start from a retrieval-trained read, retrieval retention
as the metric, variable r as the fallback ladder r=16 → 4).

**(d) Gaps nobody has filled (concrete):** a single-global-read decoder LM at 125M–600M trained
from scratch to 1M with published RULER-1M; learned 16× slots + exact retrieval in one causal
stack; whether one full read at a ~1 KB/token budget substitutes for 3:1 globals at *matched
memory*; any positive example of LM loss alone activating a lone global layer at small scale
(our negative + LongFilter say no); dense ≤3B @1M on LongBench v2 / BABILong with an open recipe;
a unified 1M eval for small models.
