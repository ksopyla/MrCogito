# Depth-wise residual connections

How a layer's input is built from the outputs of the layers below it, beyond the plain residual sum. Reviewed for
E40 (`docs/experiments_specs/done_failed/E40_depth_mixed_read.md`).

## DeepCrossAttention: Supercharging Transformer Residual Connections

[arXiv 2502.06785](https://arxiv.org/abs/2502.06785) · ICML 2025 ([PMLR](https://proceedings.mlr.press/v267/heddes25a.html)) ·
Heddes, Javanmard, Axiotis, Fu, Bateni, Mirrokni (Google Research) ·
[blog, 2026-11-23](https://research.google/blog/supercharging-transformer-residual-connections/) · no official code;
unofficial PyTorch: [lucidrains/deep-cross-attention](https://github.com/lucidrains/deep-cross-attention).

### TL;DR
Replace the residual sum with learned weighted sums of all earlier layer outputs, and give attention three of them,
one each for its query, key and value inputs. Same perplexity in up to a third of the steps at 24 layers; gains
shrink with width; tested only on 128-token sequences.

### Problem
A plain residual stream adds every layer's output with weight 1, so useful early signal (for example the token's own
identity) is diluted by everything added later; the paper proves, for linear low-rank layers, that learned weights
beat a ResNet of equal size when the layers' summed rank is small relative to the width.

### Solution
Stack the earlier outputs G_t = [f_{t−1}, …, f_0] ∈ R^{d×t}.
- GRN-v1: g_t = G_t b_t (one static scalar per layer; DenseFormer-like).
- GRN-v2: per-layer, per-dimension static weights.
- GRN-v3: g_t = (G_t ⊙ (b_t + 1·relu(w_tᵀ G_t)))·1, w_t ∈ R^d at zero: one input-dependent scalar per token and per
  earlier layer.
- DCA: three independent GRN-v3 per block for the query, key and value inputs. k-DCA keeps the model input, the last
  k outputs and the sum of the rest.
- Overhead: +0.16 % parameters (6 layers); 24 layers 8.14 → 5.39 batches/s (2-DCA) and 3.72 (24-DCA).

### Results
- LM1B, 24 layers: 1- and 2-DCA reach the baseline's final perplexity in 0.33 of the time; final 15.14 → 14.41 (2-DCA).
- C4, 8-DCA: 75M 27.88 → 26.44; 179M 21.57 → 20.46; 449M 17.17 → 16.76.
- Width (12 layers): −2.82 perplexity at width 64, −0.39 at 1024.
- Ablation (6 layers): 18.98 → 18.80 (v1) → 18.43 (v2) → 18.41 (v3) → 18.06 (DCA): per-dimension weights and the
  query/key/value split carry the gain, input dependence almost nothing. (The paper's text ranks v1 first; its table
  does not.)
- Retrofit: DCA added to a trained 6-layer model, 60k more steps: −0.19 against −0.02 for the plain model.
- Learned weights lean on the input and the last layers, negative biases on middle layers.
- Beats LAuReL, 1×1-DenseFormer and Hyper-Connections in its own tables (C4 179M: 20.39 against 20.72–21.17).

### Limits
128-token training only; no long-context, retrieval or downstream language tests; largest model 449M; no independent
replication at scale (a 19M toy study found gaps within batch noise); the blog's "up to five times fewer steps" and
"30–40 % fewer parameters" are not in the paper.

### Related publications
- MUDDFormer, [2502.12170](https://arxiv.org/abs/2502.12170): dynamic dense weights per position, separate for query,
  key, value and residual; tested to 2.8B on 300B tokens, claims the quality of 1.8–2.4× the compute. Not compared
  with DCA.
- Hyper-Connections, [2409.19606](https://arxiv.org/abs/2409.19606); manifold-constrained Hyper-Connections,
  [2512.24880](https://arxiv.org/abs/2512.24880).
- DenseFormer, [2402.02622](https://arxiv.org/abs/2402.02622): static depth-weighted averaging.
- Attention Residuals (Kimi), [2603.15031](https://arxiv.org/abs/2603.15031): softmax attention over depth, 48B-total MoE.
  Thinking Machines' Inkling 2 uses per-query/key/value depth mixes on its global layers
  ([HF post](https://huggingface.co/blog/thinkingmachines-inkling)), credited to block attention residuals.

### MrCogito mapping (2026-11-24)
Our U-net skips are GRN-v1 on six fixed pairs; trained, only the reading layer → stack 5 skip is strongly on
(+0.66 / +0.77) and stack 0 → stack 4 is negative, the pattern DCA reports. A generic DCA trunk adds routes around
the concept read rather than through it, and the expected gain at width 896 is small. The piece that maps onto the
concept idea is the query/key/value split for the reading layer: memory keys and values from early, identity-rich
depths, the question from the current pass. Verdict: **Adapt** for the reading layer only, **Watch** for the trunk;
tested as a retrofit in E40.

**E40 result (2026-11-25): Watch for both.** As a 154M-token retrofit of the trained 150M concept-memory model, the
reading-layer query/key/value mixes and 2-DCA on every layer run both tie a paired control on 64k documents (whole
document and far-context gain within ±0.003 nats per token); the mixes stayed near the identity. Two practical notes
for any later use: the paper's zero-initialised gate cannot learn under relu (its gradient at 0 is 0; E40 uses a relu
with gradient 1 at 0), and a gate that sums over the width moves ~d times faster than b under Adam (E40 averages).
