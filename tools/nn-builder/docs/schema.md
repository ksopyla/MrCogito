# The graph file (`nn-builder/graph`, version 1)

A saved design is one JSON file (`designs/<name>.json`, or a downloaded export). It holds everything needed
to rebuild the network in code. The machine-readable schema is [`schema/graph.schema.json`](../schema/graph.schema.json)
(JSON Schema 2020-12); every preset is tested against it (`tests/schema.test.ts`). This page gives the meaning of
each field — what the forward pass does — so a code generator can write a PyTorch `nn.Module` from the file alone.

## Top level

| field | meaning |
|---|---|
| `schema`, `version` | always `"nn-builder/graph"`, `1`. New optional fields keep version 1; anything that changes the meaning of an existing field bumps it. |
| `name`, `description` | free text. |
| `stacks` | the network: one or more stacks, each a list of rows (below). |
| `edges` | wires between rows, cells and stacks. |
| `groups` | loops (shared weights, run R times) and repeats (R independent copies) over neighbouring rows. |
| `run` | the training setup the estimates use (row length, batch, precision, optimizer, GPUs). |
| `ui` | layout only; never changes the model. Optional. |
| `derived` | written on export: the builder's numbers at export time (below). Informational; ignored on load. |

## Stacks

A stack is one stream of vectors of width `width` that flows top to bottom through its rows.

| `seq.kind` | positions per training row (length S) |
|---|---|
| `tokens` | S — the training row. The loss is computed on these. |
| `source` | `length` (0 = S). An encoder input of its own length. |
| `pages` | `count × window` where pages of `window` tokens start every `stride` tokens over stack `of` (the last page is pulled back to end at the row's end; a row shorter than a page is one page). Pages may overlap. |
| `slots` | `count × per_page`: `per_page` slots for every page of the pages stack `of`. |
| `latents` | `count`, a fixed set (Perceiver-style). |

`role` is a label (`decoder`, `encoder`, `writer`, `latent`, `other`) and changes nothing.

A stack's stream starts either from a cell that creates it (`embedding`, `latent_init`) or from an `input` wire.

## Rows and the order of operations

Rows run top to bottom. For each row:

1. if `x0_mix`: `x = α·x + β·x0` with two learned scalars (α starts at 1, β at 0). `x0` is the output of the
   stack's first row (the input row).
2. for each `skip` wire into this row: `x = x + w·out(from_row)`, where `w` is one learned scalar (`weight:
   "scalar"`, starts at 0), a learned vector of the stream width (`"vector"`), or nothing (`"none"`).
3. the row's cells run left to right, each transforming `x`.
4. `out(row) = x`.

## Cells

Every cell has `id`, `type`, an optional `label`, and the settings below. "Width" is the stream width entering
the cell (the stack width, unless a `linear` or `embedding` cell earlier in the stack changed it).

Residual sublayers (`attention`, `ffn`) use `norm` (`rmsnorm`, `layernorm` with bias, `none`) and `residual`:
- `pre`: `x = x + f(norm(x))`
- `post`: `x = norm(x + f(x))`
- `none`: `x = f(norm(x))`

| type | settings → forward pass | parameters |
|---|---|---|
| `embedding` | `e = table[ids] + Σ ngram_k[hash_k(ids)]` (`ngram_tables` hashed n-gram tables of `ngram_buckets` rows, orders 2, 3, …); then `proj`: `none` (stream = e, width = `dim`), `linear` (`W·e`), `gated_mlp` (`up0(e) + up2(silu(up1(e)))`, the MrCogito input layer); then `out_norm`. Port `table` = `e` before the projection. `share_with` = id of another embedding whose token table is reused. | `vocab·dim` (0 if shared) + `ngram_tables·ngram_buckets·dim` + proj (`dim·d` or `2·dim·d + d²`) + norm |
| `positional` | `rope`: rotary positions (base `theta`) for every later attention cell of the stack (unless that cell sets `nope`); `learned`: `x += P[pos]` with `max_len` rows; `sinusoidal`: fixed table added; `alibi`: linear distance bias in later attention; `none`. | `learned`: `max_len·width`; else 0 |
| `attention` | `mode: self` reads its own stream; `cross` reads keys/values from the stack wired into it (`source` wire). `heads` query heads of size `head_dim`, `kv_heads` key/value heads (grouped-query when fewer). `causal`; `window` > 0 = sliding window of that many keys (self only; 0 = the whole row). `qk_norm`: RMSNorm of size `head_dim` on queries and keys. `bias` on q, k, v, o. `kv_norm` (cross): norm on the source states. In a `pages` stack attention stays inside each page; in a `slots` stack inside each page's slots; a slots stack cross-attending to its own pages stack reads only its own page. `memory` (self): the keys also include the **notes** wired in from a notes stack (one ending in `kv_out`), each readable once its page is complete (`page start + window − 1 ≤ t`); notes are already keys/values in this cell's `kv_heads × head_dim` space, so they are not projected again. `dual_key`: each key is scored twice (E33's read and rank keys). `kernel`: `flash` or `eager` (memory only). `extras.order_term` (E33): a per-head order strength and a width → 1 order gate; `extras.count_heads` (E33): count heads over notes and recent keys, coded as a soft one-hot over 0..`count_max` (+ log(1 + n)) and projected back to the width; `extras.pos_prior`: learned bias `[slots per page, heads, page width]` on a slots → page read; `extras.logit_scale`: one learnable length-aware query scale. | `d·h·dh` (q) + `2·d_src·g·dh` (k, v) + `h·dh·d` (o) + biases + `2·dh` (q/k norm) + norms + extras: order `h + d + 1`, count `2·n + d·n·dh + dh + n·(count_max + 2)·d`, prior `slots·h·W`, scale 1 |
| `ffn` | `swiglu`: `down(silu(gate(x)) · up(x))`; `geglu`: same with GELU; `mlp`: `down(act(up(x)))` with `activation`. `hidden` = inner width. `experts` > 1: a mixture of experts with a linear router (width → experts) choosing `top_k` per token. | `3·d·F` (gated) or `2·d·F`, × experts, + biases + router `d·E` + norm |
| `norm` | `x = norm(x)`. | `d` (RMSNorm), `2d` (LayerNorm) |
| `linear` | `x = W·x (+ b)`, width → `out` (0 = keep), then `out_norm`. | `d·out` + bias + norm |
| `latent_init` | replaces the stream with learned queries, one per slot (slots stack: the same `per_page` queries on every page) or per latent, plus `address_feats` > 0: a linear map from page-address features; then `norm`. | `n·d + address_feats·d` + norm |
| `kv_out` | turns slot states into notes a memory read uses directly: `entries` keys/values per slot, each `kv_heads × head_dim`. `projections: 2` = k, v; `4` = k, k-read, v, v-state as in MrCogito's `LatentMemoryWriter` (`nn/latent_memory.py`). `in_norms` norms on the inputs; `key_norm` = RMSNorm on keys. | `in_norms·d + projections·d·entries·kv_heads·head_dim` + key norm `head_dim` |
| `lm_head` | `logits = W·final_norm(x) (+ b)`; `tied` reuses the stack's token table (needs `dim = width`, same vocab). Loss: cross-entropy, `softcap` > 0 → `cap·tanh(logits/cap)`, `z_loss` weight on `logsumexp²`; `chunked` = computed in chunks and recomputed in backward. | final norm + `vocab·d` (0 if tied) + bias |
| `custom` | anything else: the builder only counts `params`, `flops_per_position` (forward) and `act_bytes_per_position`; `out` = output width (0 = keep). A code generator must ask what it is. | `params` |

## Wires (`edges`)

Each end is `{stack, row?, cell?, port?}`: a row means its output; a stack alone means its last row's output; a cell
with `port: "table"` means an embedding's token table output.

| `kind` | meaning |
|---|---|
| `skip` | row → later row of the same stack, added at the start of the target row (see the row order above). `weight`: `scalar`, `vector`, `none`. |
| `source` | anything → an `attention` cell: what a cross-attention reads keys and values from, or (from a stack ending in `kv_out`) the notes a memory read uses. One per cell. |
| `input` | anything → a stack: that stack's stream starts from this output (e.g. the page writer starts from the decoder's token table). One per stack. |

## Groups

| field | meaning |
|---|---|
| `kind: loop` | the rows (neighbours in one stack) run `count` times with **the same weights**. `round_embedding`: a learned vector per round (`count × width`, starts at 0) added to `x` at the start of each round. `feedback: soft` (E33q): after round r < R, the round's state is read through the stack's own final norm and LM head, the softmax distribution is re-embedded through the token table and the embedding's projection, and added to the next round's input through a per-channel gate (width, starts at 0). Activations of every round are kept for backward. |
| `kind: repeat` | `count` independent copies of the rows, one after another (a shorthand for stacking identical layers). Skip wires into a repeated row go to its first copy. |

## The training setup (`run`)

`row_length`, `rows_per_step` (rows per optimizer step), `micro_batch` (rows per GPU per pass; 0 = the largest that
fits and splits the step evenly), `precision` (`amp_bf16` = PyTorch autocast with fp32 weights, `bf16_master`,
`pure_bf16`, `fp32`), `optimizer`, `weight_decay`, `zero_stage` (sharding of optimizer state, gradients, weights),
`checkpointing` (recompute every row in backward), `gpu`, `gpu_count`, `token_budget`, `deadline_hours`, and the
speed calibration overrides `mfu`, `irregular_ratio`, `dp_efficiency` (0 = the built-in calibration).

## `derived` (export only)

```json
"derived": {
  "exported_at": "2026-10-24T09:48:00.000Z",
  "row_length": 4096,
  "params_total": 29946402,
  "params_by_stack": { "main": 25960546, "pages": 1377792, "notes": 2608064 },
  "train_flops_per_token": 349823931,
  "cells": { "main.L2.attn": { "params": 826190, "width_in": 576, "width_out": 576, "runs_per_row": 4 } }
}
```

A code generator should build the model, count its trainable parameters, and check them against `params_total`
and the per-cell numbers. For the MrCogito presets these match the trainer exactly
(`evaluation/text_checks.count_params`).
