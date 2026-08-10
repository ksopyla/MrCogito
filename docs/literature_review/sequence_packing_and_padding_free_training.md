# Sequence packing, unpadding, and pad-free training

External notes for reducing padding waste in variable-length LM training. Motivated by Concept Encoder E17b (2026-08-10): larger microbatch filled VRAM but lost tokens/s to pad-to-batch-max.

### TL;DR — ModernBERT (Answer.AI / HF, Dec 2024)

ModernBERT combines **unpadding** (strip pads, concat to ragged mini-batch) with **greedy sequence packing** and FlashAttention-2 so pad tokens are never computed. Claimed ~10–20% over older unpad/repad loops; packing pushes GPU density further. Pattern is the right mental model for pad-free training, but their implementation lives inside an **encoder MLM** FA2 stack — not a drop-in for Gemma-3 causal + custom masks.

Sources: [HF blog](https://huggingface.co/blog/modernbert) · [HF docs](https://huggingface.co/docs/transformers/model_doc/modernbert) · [paper](https://arxiv.org/abs/2412.13663)

### TL;DR — HF padding-free / `DataCollatorWithFlattening`

Canonical HF path for causal LMs: flatten microbatch → `[1, Σℓᵢ]`, supply FlashAttention boundaries via `return_flash_attn_kwargs=True` (`cu_seqlens`), **do not** pass `attention_mask`. Requires a FlashAttention implementation. Inferring boundaries from `position_ids` alone is discouraged (compile graph breaks / host syncs).

Sources: [padding-free guide](https://huggingface.co/docs/transformers/padding_free) · [packing+FA2 blog](https://huggingface.co/blog/packing-with-FA2) · [arXiv:2407.09105](https://arxiv.org/abs/2407.09105)

### TL;DR — Length bucketing (no model change)

Sort/group similar lengths so pad-to-batch-max ≈ true length. Typical **1.3–1.7×** real tokens/s when length variance is high. Safest first step when the model cannot consume FA varlen layouts.

Source: [variable-length DDP throughput notes](https://duoan.github.io/posts/why-variable-sequence-length-breaks-ddp-throughput/)

### Concept Encoder implication

See eng spec: [`docs/engineering_specs/pad_free_variable_length_training.md`](../engineering_specs/pad_free_variable_length_training.md). BackboneConceptLM’s K=512 recurrent concept state forbids naive multi-doc packing without explicit `z` resets; Phase 1 = length bucketing.
