# PyTorch Code-Reading Checklist

## Contents

- File-level reading order
- Per-function reading checklist
- Common implementation details to extract
- Configuration files
- When code disagrees with the paper

## File-Level Reading Order

For most PyTorch research repos, read in this order:

1. `README.md` — claimed contributions, paper link, repro instructions, dataset.
2. Top-level `train.py` / `pretrain.py` / `trainer/` — what the training loop actually runs.
3. `models/<model>.py` — model class and `forward`.
4. Loss / objective — often inside the model file or in a dedicated `losses/` directory.
5. Data loading and collation — `data/`, `datasets/`, `collator.py`. Input shapes start here.
6. Inference / decoding — `generate.py`, `sample.py`, `eval/`.
7. Config files — `configs/*.yaml` or `configs/*.py`.

If the repo has tests or notebooks under `examples/`, they are often the cleanest entry point to a complete forward pass.

## Per-Function Reading Checklist

When reading the model `forward`:

- Input arguments and their shapes (annotated or inferred from callers).
- Where positional encodings are added.
- Pre-norm vs post-norm placement.
- Attention mask construction (causal, padding, bidirectional, sliding-window).
- Heads × dim layout: `(B, H, N, d_head)` vs `(B, N, H, d_head)`.
- Residual connections — count them; missing residuals matter.
- Output shape and what it represents (logits, hidden states, both).

When reading the loss:

- What is being compared (logits vs targets, embeddings vs embeddings, samples vs samples).
- Reduction (`mean`, `sum`, masked mean) and over which axes.
- Label smoothing or class weighting.
- Auxiliary losses and their weights.
- Loss weighting schedule (e.g. diffusion `1/t`, contrastive temperature).

When reading the training loop:

- Optimizer, LR schedule, warmup steps, weight decay.
- Gradient accumulation, gradient clipping value.
- Mixed precision (`autocast`, `GradScaler`, `bf16` vs `fp16`).
- Distributed strategy (`DDP`, `FSDP`, ZeRO stage).
- EMA target networks, stop-gradient operations.
- Curriculum or phase changes during training.

When reading inference:

- Sampling strategy (greedy, top-k, top-p, beam, diffusion sampler).
- KV caching or its absence.
- Test-time tricks (recursion depth, guidance scale, temperature).
- Length control and stop conditions.

## Common Implementation Details to Extract

- **Tensor shape conventions** — `(B, N, D)` vs `(B, D, N)` vs `(N, B, D)`. Mismatches cause silent bugs.
- **Masking semantics** — `True = mask out` or `True = keep`? Boolean mask vs additive `-inf`?
- **Position encodings** — sinusoidal, learned, RoPE, ALiBi, or none.
- **Normalization** — LayerNorm, RMSNorm; pre-norm vs post-norm; AdaLN / AdaLN-Zero conditioning.
- **Initialization** — scaled init, init from pretrained, zero-init for residual gates.
- **Weight tying** — shared embeddings, shared cross-attention layers, shared QKV projections.
- **Frozen modules** — encoders kept frozen during fine-tuning, EMA target nets, vector-quantization codebooks.
- **Gradient routing** — `detach()`, `torch.no_grad()`, `requires_grad_(False)` calls, custom autograd functions.

## Configuration Files

Configs often encode design choices not visible in code. Capture:

- Sequence lengths and special tokens (BOS, EOS, padding, mask).
- Vocabulary size and tokenizer family (BPE, SentencePiece, WordPiece).
- Model dimensions (`d_model`, `n_heads`, `d_head`, `d_ff`, `n_layers`, latent / concept count).
- Dropout, attention dropout, stochastic depth.
- Mixed precision dtype (`bf16`, `fp16`, `fp32`).
- Seed and any deterministic flags.

## When Code Disagrees with the Paper

Three sources can drift apart in any research repo:

- The README's claims.
- The defaults baked into the model code.
- The values in the config file the user actually runs.

Reconcile them explicitly. The **config the user runs** wins; note the discrepancy and explain which value the rest of the walkthrough assumes.
