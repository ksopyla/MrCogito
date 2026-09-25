"""Perceiver AR v2 — one global causal read, deep window-N stack, every token trained (E18).

Family `perceiver_ar`, selected via `model_family=perceiver_ar` in the shared training
entrypoint. Spec: docs/experiments_specs/ahead/E18_perceiver_ar_v2_baseline.md.

Training-time structure (all layers share one block; only the attention *pattern* differs):

    ids ─► tiny factorized embedding (e=256) + hashed 2/3-gram tables ─► MLP up-proj ─► x0
        ─► [swa(pre_window)] × pre_layers          # local pre-encoder: contextualize the history
        ─► [full causal]     × global_layers        # exclusive (E21) or raw (E18) global read+FFN
        ─► [swa(block)]      × stack_layers         # deep local processing over the last N tokens
        ─► RMSNorm ─► chunked soft-capped lm_head + CE (+ z-loss)

`par_mode="dense"` turns every layer into `full` with zero pre-layers (the matched control).

Attention backends: `sdpa` (explicit boolean mask; reference / tests), `flex`
(torch.nn.attention.flex_attention block masks; default on Ampere), `flash`
(flash-attn varlen; Hopper fast path for packed inputs — wired, optional dependency).

Hooks for the family (config fields only — no parameters unless enabled):
  * `write_back_hook` (E19): a zero-init projection from latent states into the global
    layer's K/V space, exposed via `prefix_kv()` / `global_kv_space` so latent thoughts and
    agent messages (E21) can be appended to the prefix cache.
  * `block_attention_mode` (E20): `causal` (E18) | `bidirectional` (block-diffusion adaptation).
  * `message_boundary_token_id` / `message_compress_ratio` (E21): a reserved token splits a
    row into sender | receiver. Local layers and n-gram tables treat the boundary as a
    document start; the global read lets receivers see the prefix only as `KVCompressor`
    slots (one per `ratio` tokens, in the read's own K/V space). Complete homogeneous
    blocks only by default; `message_pool_remainder` also pools the incomplete last sender
    block. `message_slots_inplace` writes slots into sender prefix positions (KV_LEN=S).
    `message_inplace_raw_kv` (requires inplace) copies token K/V into those positions
    instead of compressor slots; the exclusive `~replace` mask still hides uncompressed
    remainder.     `message_identity_slots` bypasses learned `u`/`delta` so each slot is a
    frozen mean of the r tokens in the block (r=1: hard copy of token K/V after `k_norm`;
    not last-token copy). `message_pack_stride` (default 0 = off) drops exclusive
    leftover sender tokens vs N-token packs tiled to end at QUERY (left leftover
    after BOS). r=1 identity otherwise keeps every sender token, so `--message_pool_remainder`
    is a no-op at r=1; this flag is how incomplete leftover tokens are dropped vs kept
    as identity slots (`message_pool_remainder` keeps them). E18-loadable.
    `message_keep_local_swa` (default off) keeps exclusive slots on
    the global read but does not treat QUERY as a SWA/n-gram document start — the local
    window still sees raw prefix tokens that fall inside the sliding window.
    `message_raw_window` (default 0 = full causal) limits the global read's raw keys to a
    causal window of that many tokens, so the slots are the only long-range path and the
    whole model is linear in length (the long-context length ladder).
    `message_extra_slot_attends` (default 0) re-reads the *same* exclusive slot K/V
    with queries updated by the previous hop — extra attends *inside one* Attention,
    not a second global Block and not DNA `--hops`. Off by
    default (`id=-1`) so E18 checkpoints stay byte-identical.
    `global_layers` (default 1) is the count of sequential full Attention+FFN
    blocks after the pre-encoder. `global_layers=2` is a second exclusive
    (E21) or raw (E18) global read+FFN over that layer's own slot/prefix K/V —
    distinct from `--stack_layers` (SWA) and from extra hops inside one Attention.
    `message_update_slot_kv` (default off) rewrites exclusive slot K/V from the
    post-attend residual before each extra hop (queries *and* slot keys update).
    Extra=0 is unchanged either way; still not the full raw prefix.
    `message_global_anchors` (default `none`) leaks a *sparse* set of extra
    tokens into the exclusive global read. `query_nbhd`: a few sender tokens
    immediately before QUERY (already r=1 replace slots). `query_side`: QUERY
    itself plus a small window *after* the message boundary (receiver-side
    type request; not prefix replace slots). `type_marks`:
    `keymark`/`decoy`/`spanmark`/`hop`/`mark`. `key_spans` (E27): the DNA
    *key* tokens after each sender `keymark` (not the mark, not `*val`).
    Those positions join exclusive slot K/V as **raw** keys; at r>1 the
    compressor does not overwrite them (identity `b` + pooled `a`). The rest
    of the prefix stays exclusive (not E18 raw KV). `prefix_kv(as_message=True)`
    returns those slots.
    `message_prefix_ae` (E26, default off) adds a weak linear head that
    reconstructs each complete sender block's token ids from that block's
    slot. Compressor `u`/`delta` see AE gradients only when
    `message_prefix_ae_stopgrad_answer` (default on) detaches slots on the
    exclusive read. E18 checkpoints stay byte-identical with both flags off.
    `message_write` (E30, default `block_mean`): `block_mean` is E21's
    `KVCompressor`; `sw_perceiver` writes overlapping Perceiver banks
    (`SlidingWindowPerceiverCompressor`, K learned queries per window,
    stride < W, no mean residual). Concat exclusive path only. Off by
    default so E18/E21 checkpoints stay byte-identical.
    `message_write="latent_memory"` (E31): a separate writer branch (`nn/latent_memory.py`)
    reads the token embeddings, writes `lm_latents` addressed latents per overlapping
    window from two-way context (`lm_context`: page encoder | BiXT rounds) and hands them to
    the global read as precomputed slots (`MessageCtx.slots`). The main path is unchanged.
"""
from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from nn.latent_memory import LM_CONTEXTS, LatentMemoryWriter, lm_geometry, window_pick

logger = logging.getLogger(__name__)

# Sparse exclusive-plus-anchors (E21). Default `none` is prior exclusive slots.
# `query_nbhd` = sender tokens immediately before QUERY (prefix; r=1 slots).
# `query_side` = QUERY + window after the boundary (receiver; not prefix slots).
# `key_spans` = DNA key-field tokens after each sender keymark (E27 hybrid b).
MESSAGE_GLOBAL_ANCHORS = (
    "none", "query_nbhd", "query_side", "type_marks", "query_nbhd+type", "key_spans",
)
MESSAGE_QUERY_NBHD_DEFAULT = 4  # sender-before or QUERY-plus-after window
MESSAGE_WRITES = ("block_mean", "sw_perceiver", "latent_memory")
SWP_BANK_DEFAULT = 32
SWP_COVERAGE_DEFAULT = 8
SWP_OVERLAP_DEFAULT = 0.25  # stride = (1 - overlap) * window → 192/256 in the notes

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


class PerceiverARConfig(PretrainedConfig):
    model_type = "perceiver_ar"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 1280,
        intermediate_size: int = 3456,
        token_embedding_dim: int = 256,
        # pattern
        par_mode: str = "perceiver",          # "perceiver" | "dense"
        pre_layers: int = 2,
        pre_window: int = 1024,
        global_layers: int = 1,
        global_positions: Optional[tuple[int, ...]] = None,  # explicit layer indices of the global read(s)
        stack_layers: int = 20,
        block: int = 4096,                    # N — window of the stack layers
        # heads / positions
        num_attention_heads: Optional[int] = None,
        num_kv_heads: int = 2,
        head_dim: int = 128,
        rope_theta: float = 500000.0,
        nope_every: int = 4,                  # every k-th stack layer has no RoPE (SmolLM3); 0 = off
        global_nope: bool = False,            # global read(s) without RoPE: content-only retrieval (iRoPE-style)
        global_logit_scale: str = "none",     # "log": q *= s*log(n_visible) on full layers (SSMax; anti-dilution)
        global_scale_ref: int = 8192,         # n at which the log scale equals 1 at init (s = 1/log(ref))
        # input
        ngram_orders: tuple[int, ...] = (2, 3),
        ngram_buckets: int = 131072,
        value_embed_layers: tuple[int, ...] = (0, 7, 14),
        value_embed_dim: int = 64,
        # head / loss
        logit_softcap: float = 30.0,
        z_loss: float = 1e-4,
        chunked_ce_block_size: int = 2048,
        use_liger: bool = True,
        # backend / hooks
        attn_backend: str = "flex",           # "sdpa" | "flex" | "flash"
        attn_pad_multiple: int = 2048,
        block_attention_mode: str = "causal", # "causal" | "bidirectional" (E20)
        swa_sink: bool = False,               # windowed layers may also attend to the document's first token
        write_back_hook: bool = False,        # E19 — adds write_back_proj params when True
        message_boundary_token_id: int = -1,  # E21 — reserved id that splits sender | receiver (-1 = off)
        message_compress_ratio: int = 16,     # E21 — prefix tokens per message slot (1 = uncompressed, arm U)
        message_pool_remainder: bool = False, # E21 — pool the incomplete last sender block (default: complete blocks only)
        message_slots_inplace: bool = False,  # E21 — write slots into sender prefix positions (KV_LEN=S; default concat)
        message_inplace_raw_kv: bool = False,  # E21 — inplace: token K/V at replace positions (skip compressor values)
        message_identity_slots: bool = False,  # E21 — freeze mean-pool at any r; bypass u/delta
        message_pack_stride: int = 0,  # E21 — exclusive leftover vs N-token packs ending at QUERY (0=off)
        message_keep_local_swa: bool = False,  # E21 — local SWA/n-grams still see across QUERY
        message_raw_window: int = 0,  # >0: the global read's raw keys are a causal window (memory is the only long path)
        message_extra_slot_attends: int = 0,  # E21 — extra exclusive attends over frozen slots
        message_update_slot_kv: bool = False,  # E21 — rewrite exclusive slot K/V between extra hops
        message_global_anchors: str = "none",  # E21 — sparse raw keys joining exclusive slot K/V
        message_anchor_token_ids: tuple[int, ...] = (),  # type-mark / keymark ids
        message_anchor_window: int = MESSAGE_QUERY_NBHD_DEFAULT,  # nbhd width (query_nbhd / query_side)
        message_anchor_key_len: int = 0,  # E27 — key-span length (0 = treat as 2)
        message_prefix_ae: bool = False,  # E26 — weak prefix-block reconstruction from slots
        message_prefix_ae_weight: float = 0.0,
        message_prefix_ae_stopgrad_answer: bool = True,  # compressor sees AE grads only
        message_write: str = "block_mean",  # E30 — "block_mean" (E21) | "sw_perceiver"
        swp_bank_size: int = SWP_BANK_DEFAULT,
        swp_coverage: int = SWP_COVERAGE_DEFAULT,
        swp_window: int = 0,              # 0 → coverage * bank_size
        swp_stride: int = 0,              # 0 → round((1 - overlap) * window)
        swp_n_heads: int = 0,             # 0 → max(4, num_attention_heads)
        swp_query_dim: int = 0,           # 0 → max(head_dim, 4 * token_embedding_dim)
        swp_auto_fit: bool = True,        # shrink K so n_windows≥2 on short seq
        lm_context: str = "page_bidir",   # E31 — "page_bidir" | "bixt"
        lm_window: int = 256,             # E31 — fixed geometry (no auto-shrink)
        lm_stride: int = 192,
        lm_latents: int = 32,
        lm_latent_dim: int = 512,
        lm_heads: int = 8,                # per-latent heads, never averaged
        lm_writer_dim: int = 256,
        lm_enc_layers: int = 2,           # page_bidir only
        lm_rounds: int = 2,               # latent read rounds (bixt: latent⇄token rounds)
        lm_competition: bool = True,      # softmax over latents per token, then renormalise
        lm_null_latent: bool = False,     # one latent the reader never sees
        lm_reader_tokens: int = 5,        # reader K/V entries per latent (m)
        lm_pos_prior: bool = True,        # per-latent sub-page prior at init
        init_std: float = 0.02,
        zero_init_residuals: bool = True,    # False: warm attn.wo / mlp.down (needed at 512+)
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.token_embedding_dim = token_embedding_dim
        self.par_mode = par_mode
        self.pre_layers = pre_layers
        self.pre_window = pre_window
        self.global_layers = global_layers
        self.global_positions = tuple(int(p) for p in global_positions) if global_positions else None
        self.stack_layers = stack_layers
        self.block = block
        self.num_attention_heads = num_attention_heads or (hidden_size // head_dim)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.nope_every = nope_every
        self.global_nope = bool(global_nope)
        self.global_logit_scale = global_logit_scale
        self.global_scale_ref = int(global_scale_ref)
        self.ngram_orders = tuple(int(o) for o in ngram_orders)
        self.ngram_buckets = ngram_buckets
        self.value_embed_layers = tuple(int(l) for l in value_embed_layers)
        self.value_embed_dim = value_embed_dim
        self.logit_softcap = logit_softcap
        self.z_loss = z_loss
        self.chunked_ce_block_size = chunked_ce_block_size
        self.use_liger = use_liger
        self.attn_backend = attn_backend
        self.attn_pad_multiple = attn_pad_multiple
        self.block_attention_mode = block_attention_mode
        self.swa_sink = swa_sink
        self.write_back_hook = write_back_hook
        self.message_boundary_token_id = int(message_boundary_token_id)
        self.message_compress_ratio = int(message_compress_ratio)
        self.message_pool_remainder = bool(message_pool_remainder)
        self.message_slots_inplace = bool(message_slots_inplace)
        self.message_inplace_raw_kv = bool(message_inplace_raw_kv)
        self.message_identity_slots = bool(message_identity_slots)
        self.message_pack_stride = int(message_pack_stride)
        self.message_keep_local_swa = bool(message_keep_local_swa)
        self.message_raw_window = int(message_raw_window)
        self.message_extra_slot_attends = int(message_extra_slot_attends)
        self.message_update_slot_kv = bool(message_update_slot_kv)
        self.message_global_anchors = str(message_global_anchors or "none")
        self.message_anchor_token_ids = tuple(int(x) for x in (message_anchor_token_ids or ()))
        self.message_anchor_window = int(message_anchor_window)
        self.message_anchor_key_len = int(message_anchor_key_len)
        self.message_prefix_ae = bool(message_prefix_ae)
        self.message_prefix_ae_weight = float(message_prefix_ae_weight)
        self.message_prefix_ae_stopgrad_answer = bool(message_prefix_ae_stopgrad_answer)
        self.message_write = str(message_write or "block_mean")
        self.swp_bank_size = int(swp_bank_size)
        self.swp_coverage = int(swp_coverage)
        self.swp_window = int(swp_window)
        self.swp_stride = int(swp_stride)
        self.swp_n_heads = int(swp_n_heads)
        self.swp_query_dim = int(swp_query_dim)
        self.swp_auto_fit = bool(swp_auto_fit)
        self.lm_context = str(lm_context)
        self.lm_window = int(lm_window)
        self.lm_stride = int(lm_stride)
        self.lm_latents = int(lm_latents)
        self.lm_latent_dim = int(lm_latent_dim)
        self.lm_heads = int(lm_heads)
        self.lm_writer_dim = int(lm_writer_dim)
        self.lm_enc_layers = int(lm_enc_layers)
        self.lm_rounds = int(lm_rounds)
        self.lm_competition = bool(lm_competition)
        self.lm_null_latent = bool(lm_null_latent)
        self.lm_reader_tokens = int(lm_reader_tokens)
        self.lm_pos_prior = bool(lm_pos_prior)
        self.init_std = init_std
        self.zero_init_residuals = bool(zero_init_residuals)
        # Bookkeeping consumed by the shared entrypoint / W&B init / eval routing.
        self.checkpoint_family = "perceiver_ar"
        self.pretraining_objective = "causal_lm"
        self.concept_num = 0
        self.hidden_act = "silu"       # SwiGLU (informational; the entrypoint logs it)
        self.norm_type = "rmsnorm"
        self.max_sequence_length = kwargs.get("max_sequence_length", None)
        self._validate()
        self.num_hidden_layers = self.total_layers

    def _validate(self):
        if self.par_mode not in {"perceiver", "dense"}:
            raise ValueError(f"par_mode must be 'perceiver' or 'dense', got {self.par_mode!r}")
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
        if self.num_attention_heads * self.head_dim > 4 * self.hidden_size:
            raise ValueError("num_attention_heads*head_dim is implausibly large vs hidden_size")
        if self.attn_backend not in {"sdpa", "flex", "flash"}:
            raise ValueError(f"unknown attn_backend {self.attn_backend!r}")
        if self.block_attention_mode not in {"causal", "bidirectional"}:
            raise ValueError("block_attention_mode must be 'causal' or 'bidirectional'")
        if self.global_logit_scale not in {"none", "log"}:
            raise ValueError("global_logit_scale must be 'none' or 'log'")
        if self.global_scale_ref < 2:
            raise ValueError("global_scale_ref must be >= 2")
        if self.global_layers < 1 and self.par_mode == "perceiver":
            logger.warning("perceiver mode with global_layers=0: a purely local model (ablation only)")
        if self.global_positions is not None:
            n = self.pre_layers + self.global_layers + self.stack_layers
            if len(self.global_positions) != self.global_layers:
                raise ValueError("global_positions must list exactly global_layers indices")
            if len(set(self.global_positions)) != len(self.global_positions):
                raise ValueError("global_positions must be distinct")
            if any(p < 0 or p >= n for p in self.global_positions):
                raise ValueError(f"global_positions must lie in [0, {n})")
        if self.message_compress_ratio < 1:
            raise ValueError("message_compress_ratio must be >= 1")
        if self.message_pack_stride < 0:
            raise ValueError("message_pack_stride must be >= 0")
        if self.message_extra_slot_attends < 0:
            raise ValueError("message_extra_slot_attends must be >= 0")
        if self.message_global_anchors not in MESSAGE_GLOBAL_ANCHORS:
            raise ValueError(
                f"message_global_anchors must be one of {MESSAGE_GLOBAL_ANCHORS}, "
                f"got {self.message_global_anchors!r}"
            )
        if self.message_anchor_window < 0:
            raise ValueError("message_anchor_window must be >= 0")
        if self.message_anchor_key_len < 0:
            raise ValueError("message_anchor_key_len must be >= 0")
        if self.message_prefix_ae_weight < 0:
            raise ValueError("message_prefix_ae_weight must be >= 0")
        if self.message_prefix_ae and not self.message_enabled:
            raise ValueError("message_prefix_ae needs message_boundary_token_id >= 0")
        if self.message_write not in MESSAGE_WRITES:
            raise ValueError(
                f"message_write must be one of {MESSAGE_WRITES}, got {self.message_write!r}"
            )
        if self.swp_bank_size < 1:
            raise ValueError("swp_bank_size must be >= 1")
        if self.swp_coverage < 1:
            raise ValueError("swp_coverage must be >= 1")
        if self.swp_window < 0 or self.swp_stride < 0:
            raise ValueError("swp_window and swp_stride must be >= 0 (0 = derive)")
        if self.swp_n_heads < 0 or self.swp_query_dim < 0:
            raise ValueError("swp_n_heads and swp_query_dim must be >= 0 (0 = derive)")
        if self.message_write == "sw_perceiver":
            if not self.message_enabled:
                raise ValueError("message_write='sw_perceiver' needs message_boundary_token_id >= 0")
            if self.message_slots_inplace:
                raise ValueError("sw_perceiver uses concat slots; message_slots_inplace is not 1:1")
            if self.message_identity_slots:
                raise ValueError("sw_perceiver has no mean-pool identity; leave message_identity_slots off")
            if self.message_prefix_ae:
                raise ValueError("message_prefix_ae is a block-mean write (E26); not defined for sw_perceiver")
        if self.message_write == "latent_memory":
            if not self.message_enabled:
                raise ValueError("message_write='latent_memory' needs message_boundary_token_id >= 0")
            if self.message_slots_inplace or self.message_identity_slots or self.message_prefix_ae:
                raise ValueError("latent_memory uses concat slots; inplace / identity / prefix_ae do not apply")
            if self.lm_context not in LM_CONTEXTS:
                raise ValueError(f"lm_context must be one of {LM_CONTEXTS}, got {self.lm_context!r}")
            if min(self.lm_window, self.lm_stride, self.lm_latents, self.lm_heads,
                   self.lm_writer_dim, self.lm_reader_tokens) < 1:
                raise ValueError("lm_window/stride/latents/heads/writer_dim/reader_tokens must be >= 1")
            if self.lm_latent_dim % self.lm_heads != 0:
                raise ValueError("lm_latent_dim must be divisible by lm_heads")
            if self.lm_stride > self.lm_window:
                raise ValueError("lm_stride must be <= lm_window (windows must tile the book)")
            if self.message_extra_slot_attends or self.message_update_slot_kv:
                raise ValueError("latent_memory: extra slot attends are not defined (read once; E31a reads per layer)")
        if self.message_raw_window < 0:
            raise ValueError("message_raw_window must be >= 0 (0 = full causal raw keys)")
        if self.message_raw_window and self.message_slots_inplace:
            raise ValueError("message_raw_window applies to the concat read, not inplace slots")
        if self.message_enabled:
            if self.message_boundary_token_id >= self.vocab_size:
                raise ValueError("message_boundary_token_id must be a vocabulary id")
            if self.par_mode != "perceiver" or self.global_layers < 1:
                raise ValueError("the message boundary needs perceiver mode with >= 1 global read layer")
            if self.attn_backend == "flash":
                raise ValueError("message boundary is not expressible with the flash backend; use flex or sdpa")
            for tid in self.message_anchor_token_ids:
                if tid < 0 or tid >= self.vocab_size:
                    raise ValueError("message_anchor_token_ids must be vocabulary ids")

    @property
    def message_enabled(self) -> bool:
        """E21: rows may carry a sender|receiver boundary token; receivers read the prefix only
        through compressed slots of the global read."""
        return self.message_boundary_token_id >= 0

    @property
    def total_layers(self) -> int:
        return self.pre_layers + self.global_layers + self.stack_layers

    @property
    def resolved_global_positions(self) -> tuple[int, ...]:
        """Layer indices of the global read(s). Default: the `global_layers` layers right after
        the pre-encoder (E18). `global_positions` moves them anywhere — e.g. mid-depth, so the
        read's queries are formed by half the stack (E18 reach-ablation follow-up)."""
        if self.par_mode == "dense":
            return ()
        if self.global_positions is not None:
            return self.global_positions
        return tuple(range(self.pre_layers, self.pre_layers + self.global_layers))

    @property
    def stack_indices(self) -> list[int]:
        """Layers governed by the stack rules (NoPE every k-th, E20 bidirectional): every layer in
        dense mode; every non-pre, non-global layer in perceiver mode."""
        if self.par_mode == "dense":
            return list(range(self.total_layers))
        g = set(self.resolved_global_positions)
        return [i for i in range(self.pre_layers, self.total_layers) if i not in g]

    def layer_patterns(self) -> list[tuple[str, int]]:
        """Per-layer (pattern, window). Dense = all full causal with the same layer count."""
        if self.par_mode == "dense":
            return [("full", 0)] * self.total_layers
        pats = [("swa", self.pre_window)] * self.pre_layers + [("swa", self.block)] * (
            self.global_layers + self.stack_layers
        )
        for p in self.resolved_global_positions:
            pats[p] = ("full", 0)
        return pats

    @property
    def global_layer_index(self) -> int:
        """Index of the (first) global read layer — the one-layer prefix cache / message space."""
        if self.par_mode == "dense":
            return 0
        g = self.resolved_global_positions
        return min(g) if g else self.pre_layers


# --------------------------------------------------------------------------------------
# Masks (single source of truth for every backend)
# --------------------------------------------------------------------------------------


def make_mask_pred(
    pattern: str,
    window: int,
    key_valid: Optional[torch.Tensor],
    doc_ids: Optional[torch.Tensor],
    causal: bool = True,
    sink: bool = False,
    sink_pos: Optional[torch.Tensor] = None,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return mask_mod(b, h, q, kv) -> bool with FlexAttention semantics.

    Rules: causal (kv <= q) unless bidirectional; sliding window (q - kv < window) for
    `swa`; padded keys masked except the diagonal (keeps every query row non-empty);
    cross-document attention masked when `doc_ids` is given. With `sink=True` a `swa`
    query may additionally attend to its document's first token (`sink_pos[b, q]`, or
    position 0 when `sink_pos` is None): with RoPE that single extra key gives every
    windowed layer an absolute-position signal (q·k_0 is a function of q's position) that a
    purely relative, windowed stack otherwise never has — needed for position-based
    retrieval such as copy at a fixed offset (E18 gate P2).
    """

    def pred(b, h, q, kv):
        ok = (kv <= q) if causal else (kv >= 0)
        if pattern == "swa":
            dist = q - kv
            in_win = (dist < window) & (dist > -window)
            if sink:
                anchor = sink_pos[b, q] if sink_pos is not None else torch.zeros_like(q)
                in_win = in_win | (kv == anchor)
            ok = ok & in_win
        if key_valid is not None:
            ok = ok & (key_valid[b, kv] | (kv == q))
        if doc_ids is not None:
            ok = ok & (doc_ids[b, q] == doc_ids[b, kv])
        return ok

    return pred


def dense_bool_mask(
    S: int,
    pattern: str,
    window: int,
    key_valid: Optional[torch.Tensor],
    doc_ids: Optional[torch.Tensor],
    device,
    causal: bool = True,
    batch: int = 1,
    sink: bool = False,
    sink_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[B,1,S,S] boolean mask (True = attend) — reference path used by `sdpa`."""
    q = torch.arange(S, device=device)[:, None]
    kv = torch.arange(S, device=device)[None, :]
    ok = (kv <= q) if causal else torch.ones(S, S, dtype=torch.bool, device=device)
    if pattern == "swa":
        dist = q - kv
        in_win = (dist < window) & (dist > -window)
        if sink:
            if sink_pos is None:
                in_win = in_win | (kv == 0)
            else:
                anchor = (kv[None, None] == sink_pos[:, None, :, None])  # [B,1,S,S]
                in_win = in_win[None, None] | anchor
        ok = ok & in_win
    ok = ok[None, None].expand(batch, 1, S, S) if ok.dim() == 2 else ok.expand(batch, 1, S, S)
    if key_valid is not None:
        kv_ok = key_valid.bool()[:, None, None, :] | torch.eye(S, dtype=torch.bool, device=device)[None, None]
        ok = ok & kv_ok
    if doc_ids is not None:
        ok = ok & (doc_ids[:, None, :, None] == doc_ids[:, None, None, :])
    return ok


_FLEX_CACHE: dict = {}
_flex_attention_fn = None


def _get_flex():
    global _flex_attention_fn
    if _flex_attention_fn is None:
        from torch.nn.attention.flex_attention import flex_attention

        if torch.cuda.is_available():
            # Every (pattern, window) closure × grad-mode × batch/sequence shape is a separate
            # dynamo graph. The default cache_size_limit (8) is exceeded as soon as eval runs
            # next to training, and dynamo then silently falls back to EAGER flex_attention,
            # whose dense-math kernel materialises the S×S scores (48 GB at 32k) → OOM.
            import torch._dynamo.config as dyn_cfg

            dyn_cfg.cache_size_limit = max(int(getattr(dyn_cfg, "cache_size_limit", 8)), 128)
            if hasattr(dyn_cfg, "accumulated_cache_size_limit"):
                dyn_cfg.accumulated_cache_size_limit = max(
                    int(dyn_cfg.accumulated_cache_size_limit), 1024
                )
            _flex_attention_fn = torch.compile(flex_attention, dynamic=False)
        else:
            _flex_attention_fn = flex_attention
    return _flex_attention_fn


def _flex_block_mask(S, pattern, window, key_valid, doc_ids, device, causal, batch,
                     sink=False, sink_pos=None):
    from torch.nn.attention.flex_attention import create_block_mask

    batch_dependent = key_valid is not None or doc_ids is not None or sink_pos is not None
    key = None
    if not batch_dependent:
        key = (S, pattern, window, causal, sink, str(device))
        if key in _FLEX_CACHE:
            return _FLEX_CACHE[key]
    pred = make_mask_pred(pattern, window, key_valid, doc_ids, causal=causal, sink=sink, sink_pos=sink_pos)
    if pattern == "swa" and S * S > _CHUNKED_MASK_PAIRS:
        # Long rows: the band's block lists directly (O(S) work, not O(S²) predicate calls).
        extra = None
        if sink:
            nqb = -(-S // 128)
            first_q = torch.arange(nqb, device=device) * 128
            if sink_pos is not None:
                extra = (sink_pos[:, first_q] // 128)[..., None]
            else:
                extra = torch.zeros(batch, nqb, 1, dtype=torch.long, device=device)
        bm = _band_block_mask(pred, B=batch if batch_dependent or extra is not None else 1, Q_LEN=S, KV_LEN=S,
                              window=window, causal=causal, device=device, extra_blocks=extra)
        if key is not None:
            _FLEX_CACHE[key] = bm
        return bm
    # Always compile the mask build on CUDA: the eager path materialises int64 (Q_LEN, KV_LEN)
    # index grids (~8 GB transient at 32k, impossible at 256k); the compiled path works
    # block by block. CPU keeps the eager path (no inductor cost in unit tests).
    bm = create_block_mask(
        pred, B=batch if batch_dependent else None, H=None, Q_LEN=S, KV_LEN=S, device=device,
        _compile=torch.cuda.is_available(),
    )
    if key is not None:
        _FLEX_CACHE[key] = bm
    return bm


# Above this many (query, key) pairs the global-read mask is built in query chunks: the
# message predicate gathers per-token tags, and create_block_mask then materialises the full
# bool (Q_LEN, KV_LEN) grid (29 GiB at 128k tokens + 109k slots) even when compiled.
_CHUNKED_MASK_PAIRS = 1 << 30


def _chunked_block_mask(pred, *, B: int, Q_LEN: int, KV_LEN: int, device, block: int = 128,
                        max_pairs: int = 1 << 29):
    """`create_block_mask` over query chunks of ≤ `max_pairs` (query, key) pairs, stitched with
    `BlockMask.from_kv_blocks`. Same mask; peak transient memory ≈ `max_pairs` bytes."""
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    qc = max(block, (max_pairs // max(KV_LEN, 1)) // block * block)
    q0 = torch.zeros((), dtype=torch.int32, device=device)  # captured tensor: no recompile per chunk

    def pred_c(b, h, q, kv):
        return pred(b, h, q + q0, kv)

    parts = []
    for start in range(0, Q_LEN, qc):
        q0.fill_(start)
        n = min(qc, Q_LEN - start)
        bm = create_block_mask(pred_c, B=B, H=None, Q_LEN=n, KV_LEN=KV_LEN, device=device,
                               BLOCK_SIZE=block, _compile=torch.cuda.is_available())
        parts.append((bm.kv_num_blocks, bm.kv_indices, bm.full_kv_num_blocks, bm.full_kv_indices))
    cat = [None if parts[0][i] is None else torch.cat([p[i] for p in parts], dim=2) for i in range(4)]
    return BlockMask.from_kv_blocks(cat[0], cat[1], cat[2], cat[3], BLOCK_SIZE=block, mask_mod=pred,
                                    seq_lengths=(Q_LEN, KV_LEN))


def _band_block_mask(mask_mod, *, B: int, Q_LEN: int, KV_LEN: int, window: int, causal: bool, device,
                     extra_blocks: Optional[torch.Tensor] = None, block: int = 128):
    """Block mask for a mask that is zero outside a band `|q − kv| < window` (plus optional extra
    KV blocks per query block, e.g. attention sinks or memory slots), built from the band
    geometry in O(blocks) instead of evaluating every (q, kv) pair. Every listed block is
    *partial*, so `mask_mod` still decides each entry: the mask is exact as long as the lists are
    a superset of the true non-zero blocks. `window <= 0` means unbounded (full causal).
    `extra_blocks` [B, n_q_blocks, E] int, −1 = unused."""
    from torch.nn.attention.flex_attention import BlockMask

    nqb = -(-Q_LEN // block)
    nkb = -(-KV_LEN // block)
    qb = torch.arange(nqb, device=device)
    q_lo, q_hi = qb * block, torch.clamp(qb * block + block - 1, max=Q_LEN - 1)
    if window <= 0:
        lo = torch.zeros_like(qb)
        span = nqb
    else:
        lo = torch.clamp(q_lo - (window - 1), min=0) // block
        wb = -(-(window - 1) // block)
        span = wb + 2 if causal else 2 * wb + 3
    hi_kv = q_hi if causal or window <= 0 else q_hi + (window - 1)
    hi = torch.clamp(hi_kv // block, max=nkb - 1)
    band = lo[:, None] + torch.arange(span, device=device)[None, :]
    band = torch.where(band <= hi[:, None], band, torch.full_like(band, -1))
    cand = band[None].expand(B, nqb, span)
    if extra_blocks is not None:
        cand = torch.cat([cand, extra_blocks.to(cand.dtype)], dim=-1)
    big = torch.full_like(cand, nkb)
    cand = torch.where((cand >= 0) & (cand < nkb), cand, big)
    cand, _ = cand.sort(dim=-1)
    dup = torch.zeros_like(cand, dtype=torch.bool)
    dup[..., 1:] = cand[..., 1:] == cand[..., :-1]
    cand = torch.where(dup, big, cand)
    cand, _ = cand.sort(dim=-1)
    num = (cand < nkb).sum(-1).to(torch.int32)
    idx = torch.where(cand < nkb, cand, torch.zeros_like(cand)).to(torch.int32)
    # flex expects one column per KV block; valid entries are sorted first and number ≤ nkb
    idx = F.pad(idx, (0, nkb - idx.shape[-1])) if idx.shape[-1] < nkb else idx[..., :nkb]
    return BlockMask.from_kv_blocks(num[:, None], idx[:, None].contiguous(), None, None, BLOCK_SIZE=block,
                                    mask_mod=mask_mod, seq_lengths=(Q_LEN, KV_LEN))


def _message_extra_blocks(ctx: "MessageCtx", S: int, block: int = 128) -> Optional[torch.Tensor]:
    """Slot KV blocks for every query block that holds a receiver token (superset)."""
    if ctx.override in ("none", "raw") or ctx.n_slots == 0:
        return None
    B = ctx.side.shape[0]
    nqb = -(-S // block)
    pad = nqb * block - S
    side = F.pad(ctx.side, (0, pad), value=0) if pad else ctx.side
    has_recv = (side.view(B, nqb, block) >= 1).any(-1)  # [B, nqb]
    first, last = S // block, (S + ctx.n_slots - 1) // block
    slot_blocks = torch.arange(first, last + 1, device=ctx.side.device)
    ex = slot_blocks[None, None, :].expand(B, nqb, slot_blocks.numel())
    return torch.where(has_recv[..., None], ex, torch.full_like(ex, -1))


def attend(
    q: torch.Tensor,   # [B,S,h,dh]
    k: torch.Tensor,   # [B,S,g,dh]
    v: torch.Tensor,   # [B,S,g,dh]
    *,
    pattern: str,
    window: int,
    key_valid: Optional[torch.Tensor],
    doc_ids: Optional[torch.Tensor],
    backend: str,
    causal: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    block_masks: Optional[dict] = None,
    sink: bool = False,
    sink_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention with the E18 pattern semantics. Returns [B,S,h,dh].

    `block_masks` is an optional per-forward memo for batch-dependent flex masks (padding or
    packed doc_ids): the same (pattern, window, causal) mask is shared by every layer of one
    forward (and its checkpoint recompute) instead of being rebuilt per layer.
    """
    B, S, h, dh = q.shape
    g = k.shape[2]
    if backend == "flash":
        if sink and pattern == "swa":
            raise NotImplementedError("swa_sink is not expressible with flash-attn window_size; use flex")
        return _attend_flash(q, k, v, pattern, window, causal, cu_seqlens, key_valid)
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))  # [B,h/g,S,dh]
    if backend == "flex":
        memo_key = (pattern, window, causal, sink)
        if block_masks is not None and memo_key in block_masks:
            bm = block_masks[memo_key]
        else:
            bm = _flex_block_mask(S, pattern, window, key_valid, doc_ids, q.device, causal, B,
                                  sink=sink, sink_pos=sink_pos)
            if block_masks is not None:
                block_masks[memo_key] = bm
        qt, kt, vt = qt.contiguous(), kt.contiguous(), vt.contiguous()
        out = _get_flex()(qt, kt, vt, block_mask=bm, enable_gqa=(g != h))
        return out.transpose(1, 2)
    # sdpa reference
    if g != h:
        rep = h // g
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)
    if pattern == "full" and key_valid is None and doc_ids is None and causal:
        out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
    else:
        mask = dense_bool_mask(S, pattern, window, key_valid, doc_ids, q.device, causal, B,
                               sink=sink, sink_pos=sink_pos)
        out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask)
    return out.transpose(1, 2)


def _attend_flash(q, k, v, pattern, window, causal, cu_seqlens, key_valid):
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("attn_backend='flash' requires the flash-attn package") from e
    win = (window - 1, 0 if causal else window - 1) if pattern == "swa" else (-1, -1)
    if cu_seqlens is None:
        if key_valid is not None and not bool(key_valid.all()):
            raise RuntimeError("flash backend needs packed inputs (no padding) or cu_seqlens")
        return flash_attn_func(q, k, v, causal=causal, window_size=win)
    B, S = q.shape[:2]
    max_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
    out = flash_attn_varlen_func(
        q.reshape(B * S, *q.shape[2:]),
        k.reshape(B * S, *k.shape[2:]),
        v.reshape(B * S, *v.shape[2:]),
        cu_seqlens, cu_seqlens, max_len, max_len, causal=causal, window_size=win,
    )
    return out.reshape(B, S, *out.shape[1:])


# --------------------------------------------------------------------------------------
# E21 message boundary: sender | receiver split of a row, prefix visible only as slots
# --------------------------------------------------------------------------------------


@dataclass
class MessageCtx:
    """Per-forward geometry of the E21 message boundary (built once in `_run_layers`).

    `side[b, t]` counts boundary tokens seen so far inside t's document (0 = sender, ≥ 1 =
    receiver). Local layers and the n-gram hashes treat a side change as a document start
    (`local_doc_ids`, unless `message_keep_local_swa`). On the global read a query may use raw keys only from its own side and,
    when it is a receiver, the compressed *slots* of every earlier side of its document.
    Slot `j` covers absolute positions [j·r, (j+1)·r) for `message_write=block_mean`.
    For `sw_perceiver`, slots are `K` queries per overlapping window (`C = n_windows·K`);
    a mixed window keeps the earliest side. By default a block-mean slot is addressable
    only when the block is homogeneous in (document, side) — `slot_doc[b, j]` is that
    document (−1 otherwise). With `message_pool_remainder`, a mixed block still emits a
    slot from the first run of (doc, side) in that block (the incomplete last sender
    block next to QUERY).
    `anchor` [B,S] marks a sparse sender subset that joins exclusive slot K/V as raw keys
    (QUERY neighborhood and/or type marks). It is not the full prefix.
    `message_pack_stride>0` (remainder off) invalidates exclusive slots for the
    QUERY-aligned left leftover `[0, P % stride)` so incomplete leftover tokens are
    dropped rather than kept as r=1 identity slots.
    """

    side: torch.Tensor          # [B,S] int64
    doc: torch.Tensor           # [B,S] int64 — original document ids (pad / invalid = -1)
    local_doc_ids: torch.Tensor  # [B,S] int64 — doc ⊕ side, fed to swa layers and the n-gram tables
    slot_doc: torch.Tensor      # [B,nb] int64
    slot_side: torch.Tensor     # [B,nb] int64
    slot_pos: torch.Tensor      # [B,nb] int64 — RoPE position of each slot (its block's last token)
    n_sides: int = 0            # max(side) + 1 over the batch; 0 = derive lazily from `side`
    override: str = "real"      # "real" | "none" | "swapped" | "raw"
    external: Optional[tuple[torch.Tensor, torch.Tensor]] = None  # (k̄, v̄) given by a receiver-only forward
    pool_valid: Optional[torch.Tensor] = None  # [B,S] tokens that enter compressor slots (remainder path)
    ratio: int = 1              # message_compress_ratio (scatter width for inplace)
    inplace: bool = False       # True: slots overwrite sender prefix K/V; KV_LEN stays S
    inplace_raw_kv: bool = False  # True: keep token K/V at replace positions (skip compressor)
    extra_slot_attends: int = 0  # extra exclusive attends over the frozen slot K/V (default 0)
    update_slot_kv: bool = False  # rewrite exclusive slot K/V from post-attend residual
    anchor: Optional[torch.Tensor] = None  # [B,S] bool — sparse raw keys joining exclusive slots
    anchor_mode: str = "none"
    slots: Optional[tuple[torch.Tensor, torch.Tensor]] = None  # E31: precomputed (k̄, v̄), normed + RoPE'd
    raw_window: int = 0         # >0: raw keys only within `q - j < raw_window` (linear-cost global layer)
    window_valid: Optional[torch.Tensor] = None  # [B,n_w,W] per-window pool mask (E30 / E31)

    @property
    def n_slots(self) -> int:
        return int(self.slot_doc.shape[1])

    def tag_stride(self) -> int:
        """Stride M of the (doc, side) → `doc * M + side` tag; M = 2·n_sides keeps the slot range
        test `0 < tag_q − slot_tag < n_sides` false for every other document."""
        k = self.n_sides or int(self.side.max().item()) + 1
        return 2 * k

    def tags(self) -> tuple[torch.Tensor, torch.Tensor]:
        """int32 `tag` [B,S] and `slot_tag` [B,nb]: `doc * M + side`, −1 for pad / invalid slots."""
        m = self.tag_stride()
        tag = torch.where(self.doc < 0, torch.full_like(self.doc, -1), self.doc * m + self.side)
        slot_tag = torch.where(self.slot_doc < 0, torch.full_like(self.slot_doc, -1),
                               self.slot_doc * m + self.slot_side)
        return tag.to(torch.int32), slot_tag.to(torch.int32)


def build_message_anchors(
    input_ids: torch.Tensor,
    side: torch.Tensor,
    doc: torch.Tensor,
    *,
    mode: str,
    token_ids: tuple[int, ...] = (),
    window: int = MESSAGE_QUERY_NBHD_DEFAULT,
    key_len: int = 0,
) -> torch.Tensor:
    """Sparse extra positions that join exclusive slot K/V as raw keys. [B,S] bool.

    `type_marks`: control tokens in `token_ids` on the sender side (DNA keymark/decoy/
    spanmark/hop, Glyph mark). `query_nbhd`: the `window` sender tokens immediately
    before each QUERY (document-start of side ≥ 1) — prefix, already r=1 replace
    slots. `query_side`: QUERY itself plus the next `window-1` receiver tokens
    (the SELECT type request lives here; not prefix replace slots). `key_spans`:
    the `key_len` sender tokens *after* each sender `keymark` (token_ids[0]); the
    mark itself and the value field are not marked. Default `none` is all-False.
    Count is << seq: a handful of marks and/or `window` tokens per QUERY, never
    the full prefix.
    """
    B, S = input_ids.shape
    anchor = torch.zeros(B, S, dtype=torch.bool, device=input_ids.device)
    mode = mode or "none"
    if mode == "none":
        return anchor
    sender = (side == 0) & (doc >= 0)
    is_query = torch.zeros(B, S, dtype=torch.bool, device=input_ids.device)
    is_query[:, 0] = side[:, 0] >= 1
    is_query[:, 1:] = (side[:, 1:] >= 1) & (side[:, :-1] == 0)
    if mode in ("type_marks", "query_nbhd+type") and token_ids:
        ids = torch.tensor(list(token_ids), device=input_ids.device, dtype=input_ids.dtype)
        is_mark = (input_ids.unsqueeze(-1) == ids).any(dim=-1)
        anchor = anchor | (is_mark & sender)
    if mode in ("query_nbhd", "query_nbhd+type") and int(window) > 0:
        w = int(window)
        nbhd = torch.zeros(B, S, dtype=torch.bool, device=input_ids.device)
        for d in range(1, w + 1):
            at_q = is_query[:, d:]
            same = doc[:, :-d] == doc[:, d:]
            nbhd[:, :-d] = nbhd[:, :-d] | (at_q & same)
        anchor = anchor | (nbhd & sender)
    if mode == "query_side" and int(window) > 0:
        w = int(window)
        nbhd = is_query.clone()
        for d in range(1, w):
            at_q = is_query[:, :-d]
            same = doc[:, d:] == doc[:, :-d]
            recv = side[:, d:] >= 1
            nbhd[:, d:] = nbhd[:, d:] | (at_q & same & recv)
        anchor = anchor | nbhd
    if mode == "key_spans" and token_ids:
        n = int(key_len) if int(key_len) > 0 else 2
        keymark = torch.as_tensor(int(token_ids[0]), device=input_ids.device, dtype=input_ids.dtype)
        is_mark = (input_ids == keymark) & sender
        span = torch.zeros(B, S, dtype=torch.bool, device=input_ids.device)
        for d in range(1, n + 1):
            at_mark = is_mark[:, :-d]
            same_doc = doc[:, d:] == doc[:, :-d]
            same_side = side[:, d:] == side[:, :-d]
            span[:, d:] = span[:, d:] | (at_mark & same_doc & same_side & sender[:, d:])
        anchor = anchor | span
    return anchor


def exclusive_visible(replace: torch.Tensor, ctx: MessageCtx) -> torch.Tensor:
    """In-place exclusive keys: slot replace positions ∪ sparse anchors."""
    vis = replace
    if ctx.anchor is not None:
        vis = vis | ctx.anchor
    return vis


def make_message_mask_pred(S: int, ctx: MessageCtx, key_valid: Optional[torch.Tensor]):
    """mask_mod over KV = raw keys [0, S) ‖ slot keys [S, S + nb) for the global read.

    The predicate captures two int32 tag buffers rather than the four int64 buffers
    (`doc`, `side`, `slot_doc`, `slot_side`): every captured tensor is gathered into an integer
    tile inside the Triton template, and with head_dim 128 / bf16 the four int64 tiles push the
    kernel past the 99 KB shared-memory limit of sm86 (RTX 3090). With `tag = doc * M + side`
    (M = 2·n_sides), "same document and same side" is one equality and "same document, earlier
    side" is the range test `0 < tag_q − slot_tag < n_sides`. `dense_message_mask` is the
    reference semantics.
    """
    nb = ctx.n_slots
    m = ctx.tag_stride()
    n_sides = m // 2
    tag, slot_tag = ctx.tags()
    raw_cross = ctx.override == "raw"
    slots_on = ctx.override not in ("none", "raw")
    raw_window = int(ctx.raw_window or 0)
    anchor = ctx.anchor
    if anchor is None:
        anchor = torch.zeros(tag.shape, dtype=torch.bool, device=tag.device)

    def pred(b, h, q, kv):
        is_raw = kv < S
        j = torch.where(is_raw, kv, kv - S)
        js = torch.clamp(j, max=max(nb - 1, 0))
        tq = tag[b, q]
        tj = tag[b, j]
        same_doc = torch.div(tj, m, rounding_mode="floor") == torch.div(tq, m, rounding_mode="floor")
        if raw_cross:
            same = same_doc
        else:
            same = tj == tq
        raw_ok = (j <= q) & same
        if raw_window > 0:
            raw_ok = raw_ok & (q - j < raw_window)
        if not raw_cross:
            dq = torch.div(tq, m, rounding_mode="floor")
            sq = tq - dq * m
            anc = is_raw & (tq >= 0) & (tj >= 0) & (sq >= 1) & anchor[b, j] & same_doc
            raw_ok = raw_ok | anc
        if key_valid is not None:
            raw_ok = raw_ok & (key_valid[b, j] | (j == q))
        if not slots_on:
            return raw_ok & is_raw
        s = slot_tag[b, js]
        d = tq - s
        slot_ok = (j < nb) & (s >= 0) & (d > 0) & (d < n_sides)
        return torch.where(is_raw, raw_ok, slot_ok)

    return pred


def dense_message_mask(S: int, ctx: MessageCtx, key_valid: Optional[torch.Tensor], device) -> torch.Tensor:
    """[B,1,S,S+nb] boolean mask (True = attend) — reference path used by `sdpa`."""
    side, doc = ctx.side, ctx.doc
    B = side.shape[0]
    q = torch.arange(S, device=device)[:, None]
    j = torch.arange(S, device=device)[None, :]
    causal = (j <= q) & (q - j < ctx.raw_window) if ctx.raw_window else (j <= q)
    causal_doc = causal[None, None] & (doc[:, None, :, None] == doc[:, None, None, :])
    if ctx.override == "raw":
        raw = causal_doc
    else:
        same_side = side[:, None, :, None] == side[:, None, None, :]
        raw = causal_doc & same_side
        if ctx.anchor is not None:
            # Receiver queries may read anchored keys as extra exclusive raw
            # (sender type_marks / query_nbhd, or QUERY-side query_side). Still
            # causal + same-document; not the full prefix.
            anc = (side[:, None, :, None] >= 1) & ctx.anchor[:, None, None, :]
            raw = raw | (causal_doc & anc)
    if key_valid is not None:
        raw = raw & (key_valid.bool()[:, None, None, :] | torch.eye(S, dtype=torch.bool, device=device)[None, None])
    slot = (
        (side[:, None, :, None] >= 1)
        & (ctx.slot_doc[:, None, None, :] >= 0)
        & (ctx.slot_doc[:, None, None, :] == doc[:, None, :, None])
        & (ctx.slot_side[:, None, None, :] < side[:, None, :, None])
    )
    if ctx.override in ("none", "raw"):
        slot = torch.zeros_like(slot)
    return torch.cat([raw.expand(B, 1, S, S), slot], dim=-1)


def mix_inplace_kv(k, v, k_bar, v_bar, ctx: MessageCtx):
    """Scatter slot K/V onto sender positions covered by a valid complete block.

    `k`/`v` and `k_bar`/`v_bar` are un-RoPE'd. Returns mixed K/V at length S and `replace`
    [B,S] (True = this prefix position holds a slot, not uncompressed sender KV).
    When `ctx.inplace_raw_kv`, keep token K/V at those positions (skip compressor values)
    and still return the same `replace` mask so receivers cannot see uncompressed remainder.
    """
    B, S, g, dh = k.shape
    r = max(int(ctx.ratio), 1)
    valid_tok = (ctx.slot_doc >= 0).repeat_interleave(r, dim=1)[:, :S]
    replace = valid_tok & (ctx.side == 0) & (ctx.doc >= 0)
    write = replace
    # E27: at r>1 keep raw token K/V on key-span anchors (identity b); r=1 identity
    # slots already are raw keys, so do not punch holes in the replace mask.
    if ctx.anchor is not None and r > 1 and bool(ctx.anchor.any()):
        write = replace & ~ctx.anchor
    if getattr(ctx, "inplace_raw_kv", False):
        return k, v, replace
    k_exp = k_bar.repeat_interleave(r, dim=1)[:, :S]
    v_exp = v_bar.repeat_interleave(r, dim=1)[:, :S]
    k_mix = torch.where(write[..., None, None], k_exp.to(k.dtype), k)
    v_mix = torch.where(write[..., None, None], v_exp.to(v.dtype), v)
    return k_mix, v_mix, write


def dense_inplace_mask(S: int, ctx: MessageCtx, key_valid: Optional[torch.Tensor],
                       replace: torch.Tensor, device) -> torch.Tensor:
    """[B,1,S,S] — causal same-document (raw geometry) minus uncompressed sender leaks."""
    side, doc = ctx.side, ctx.doc
    q = torch.arange(S, device=device)[:, None]
    j = torch.arange(S, device=device)[None, :]
    raw = (j <= q)[None, None] & (doc[:, None, :, None] == doc[:, None, None, :])
    if key_valid is not None:
        raw = raw & (key_valid.bool()[:, None, None, :] | torch.eye(S, dtype=torch.bool, device=device)[None, None])
    recv = (side >= 1)[:, None, :, None]
    send = (side == 0)[:, None, None, :]
    leak = recv & send & (~replace[:, None, None, :])
    return raw & ~leak


def make_inplace_mask_pred(S: int, ctx: MessageCtx, key_valid: Optional[torch.Tensor], replace: torch.Tensor):
    """mask_mod over KV length S for the in-place exclusive read."""
    m = ctx.tag_stride()
    tag, _ = ctx.tags()

    def pred(b, h, q, kv):
        tq = tag[b, q]
        tj = tag[b, kv]
        same_doc = torch.div(tj, m, rounding_mode="floor") == torch.div(tq, m, rounding_mode="floor")
        raw_ok = (kv <= q) & same_doc
        if key_valid is not None:
            raw_ok = raw_ok & (key_valid[b, kv] | (kv == q))
        leak = (ctx.side[b, q] >= 1) & (ctx.side[b, kv] == 0) & (~replace[b, kv])
        return raw_ok & ~leak

    return pred


def attend_inplace(q, k, v, *, ctx: MessageCtx, replace, key_valid, backend, block_masks=None):
    """Global read over in-place mixed K/V (length S). Returns [B,S,h,dh]."""
    B, S, h, dh = q.shape
    g = k.shape[2]
    if backend == "flash":
        raise NotImplementedError("message boundary needs flex or sdpa")
    visible = exclusive_visible(replace, ctx)
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    if backend == "flex":
        from torch.nn.attention.flex_attention import create_block_mask

        memo_key = ("message_inplace", ctx.override, ctx.n_slots, getattr(ctx, "anchor_mode", "none"))
        if block_masks is not None and memo_key in block_masks:
            bm = block_masks[memo_key]
        else:
            pred = make_inplace_mask_pred(S, ctx, key_valid, visible)
            bm = create_block_mask(pred, B=B, H=None, Q_LEN=S, KV_LEN=S, device=q.device,
                                   _compile=torch.cuda.is_available())
            if block_masks is not None:
                block_masks[memo_key] = bm
        qt, kt, vt = qt.contiguous(), kt.contiguous(), vt.contiguous()
        out = _get_flex()(qt, kt, vt, block_mask=bm, enable_gqa=(g != h))
        return out.transpose(1, 2)
    if g != h:
        rep = h // g
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)
    mask = dense_inplace_mask(S, ctx, key_valid, visible, q.device)
    out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask)
    return out.transpose(1, 2)


class KVCompressor(nn.Module):
    """One message slot per `ratio` prefix tokens, in the global read's K/V space (E21 / E18c).

    Slot = attention-pooled block: weights softmax_j(h_j · u_g) per kv-head (u zero-init → uniform =
    mean pool), keys pooled *before* `k_norm`, plus a zero-init linear correction from the block's
    mean hidden state. At init this is exact mean pooling; with `ratio=1` every slot is one token's
    K/V exactly (the uncompressed arm U) *until* `u`/`delta` move. `identity_slots` bypasses
    `u`/`delta` for a frozen mean of valid tokens in each block of `r` (r=1: hard copy of
    `k_norm(k_raw)`, `v`; r>1: not last-token copy). RoPE is applied by the caller at the
    slot's position.
    """

    def __init__(self, cfg: PerceiverARConfig):
        super().__init__()
        self.ratio = int(cfg.message_compress_ratio)
        self.g, self.dh = cfg.num_kv_heads, cfg.head_dim
        self.identity_slots = bool(getattr(cfg, "message_identity_slots", False))
        self.u = nn.Parameter(torch.zeros(self.g, cfg.hidden_size))
        self.delta = nn.Linear(cfg.hidden_size, 2 * self.g * self.dh, bias=False)
        nn.init.zeros_(self.delta.weight)

    def participation(self) -> torch.Tensor:
        """Keep compressor params in the graph when a batch has no QUERY (DDP)."""
        return self.u.sum() + self.delta.weight.sum()

    def n_slots(self, S: int) -> int:
        return -(-S // self.ratio)

    def forward(self, h, k_raw, v, k_norm, valid: Optional[torch.Tensor] = None):
        """h [B,S,d] (attn-normed block input), k_raw/v [B,S,g,dh] (k before k_norm) →
        (k̄, v̄) each [B,nb,g,dh]; `valid` [B,S] excludes tokens from the pooling."""
        B, S, d = h.shape
        r, g, dh = self.ratio, self.g, self.dh
        nb = self.n_slots(S)
        pad = nb * r - S
        if pad:
            h = F.pad(h, (0, 0, 0, pad))
            k_raw = F.pad(k_raw, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
        ok = torch.ones(B, S, dtype=torch.bool, device=h.device) if valid is None else valid.bool()
        if pad:
            ok = F.pad(ok, (0, pad), value=False)
        ok = ok.view(B, nb, r)
        if self.identity_slots:
            w = ok.to(dtype=k_raw.dtype)
            w = w / w.sum(dim=2, keepdim=True).clamp(min=1)
            k_bar = torch.einsum("bnr,bnrgd->bngd", w, k_raw.view(B, nb, r, g, dh))
            v_bar = torch.einsum("bnr,bnrgd->bngd", w, v.view(B, nb, r, g, dh))
            z = (self.u.sum() + self.delta.weight.sum()).to(k_bar.dtype) * 0.0
            return k_norm(k_bar) + z, v_bar + z
        hb = h.view(B, nb, r, d)
        scores = torch.einsum("bnrd,gd->bnrg", hb, self.u.to(hb.dtype))
        scores = scores.masked_fill(~ok[..., None], float("-inf"))
        w = torch.softmax(scores.float(), dim=2)
        w = torch.nan_to_num(w, nan=0.0).to(hb.dtype)
        k_bar = torch.einsum("bnrg,bnrgd->bngd", w, k_raw.view(B, nb, r, g, dh))
        v_bar = torch.einsum("bnrg,bnrgd->bngd", w, v.view(B, nb, r, g, dh))
        cnt = ok.sum(dim=2, keepdim=True).clamp(min=1).to(hb.dtype)
        h_mean = (hb * ok[..., None].to(hb.dtype)).sum(dim=2) / cnt
        dk, dv = self.delta(h_mean).view(B, nb, 2, g, dh).unbind(dim=2)
        return k_norm(k_bar) + dk, v_bar + dv


# --------------------------------------------------------------------------------------
# E30: overlapping Perceiver banks (learned queries, no mean residual)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SWPGeometry:
    """Length-scaling window banks. `n_slots = n_windows * bank_size` (C ∝ N)."""

    bank_size: int
    window: int
    stride: int
    n_windows: int
    starts: tuple[int, ...]
    n_slots: int
    n_heads: int
    query_dim: int
    coverage: float
    compression: float
    sliding: bool  # n_windows >= 2 — the sliding claim is live


def swp_resolved_heads(cfg: PerceiverARConfig) -> tuple[int, int]:
    """(n_heads, query_dim) for the write. Count first; width is scoring bandwidth.

    Do not gcd-drop below 4 heads when `query_dim` is not divisible (H=384, 6 Q
    heads, qdim=128 would otherwise silently become 2 heads). Derived head counts
    snap to a ≥4 divisor of qdim; an explicit `--swp_n_heads` keeps the count and
    rounds qdim up.
    """
    requested = int(getattr(cfg, "swp_n_heads", 0) or 0)
    n_heads = requested if requested > 0 else max(4, int(cfg.num_attention_heads))
    qdim = int(getattr(cfg, "swp_query_dim", 0) or 0)
    if qdim <= 0:
        qdim = max(int(cfg.head_dim), 4 * int(cfg.token_embedding_dim))
    if qdim % n_heads == 0:
        return n_heads, qdim
    if requested <= 0:
        for h in (8, 4):
            if qdim % h == 0:
                return h, qdim
    qdim = ((qdim + n_heads - 1) // n_heads) * n_heads
    return n_heads, qdim


def _swp_window_starts(seq_len: int, window: int, stride: int) -> tuple[int, ...]:
    S = int(seq_len)
    W = min(max(int(window), 1), max(S, 1))
    st = max(1, int(stride))
    if S <= W:
        return (0,)
    starts = list(range(0, S - W + 1, st))
    last = S - W
    if starts[-1] != last:
        starts.append(last)
    # unique, still increasing (last is always ≥ the range's final start)
    return tuple(dict.fromkeys(starts))


def swp_geometry(cfg: PerceiverARConfig, seq_len: int) -> SWPGeometry:
    """Derive (K, W, stride, windows) for this length.

    Canonical: K=32, W=8K, stride=0.75 W (notes: 256 / 192). On short sequences
    auto-fit halves K until at least two windows exist, keeping coverage ≈ 8.
    Parameter tensors stay at `swp_bank_size`; unused queries are dummy-summed.
    """
    bank_max = max(1, int(getattr(cfg, "swp_bank_size", SWP_BANK_DEFAULT) or SWP_BANK_DEFAULT))
    coverage = max(1, int(getattr(cfg, "swp_coverage", SWP_COVERAGE_DEFAULT) or SWP_COVERAGE_DEFAULT))
    pinned_w = int(getattr(cfg, "swp_window", 0) or 0)
    pinned_st = int(getattr(cfg, "swp_stride", 0) or 0)
    auto = bool(getattr(cfg, "swp_auto_fit", True))
    S = max(int(seq_len), 1)
    K = bank_max
    n_heads, qdim = swp_resolved_heads(cfg)

    def _pack(k: int) -> tuple[int, int, tuple[int, ...]]:
        w = pinned_w if pinned_w > 0 else coverage * k
        w = min(max(w, 1), S)
        st = pinned_st if pinned_st > 0 else max(1, int(round(w * (1.0 - SWP_OVERLAP_DEFAULT))))
        st = max(1, min(st, w))
        starts = _swp_window_starts(S, w, st)
        return w, st, starts

    W, stride, starts = _pack(K)
    if auto and pinned_w <= 0:
        while len(starts) < 2 and K > 1:
            K = max(1, K // 2)
            W, stride, starts = _pack(K)
    n_w = len(starts)
    n_slots = n_w * K
    return SWPGeometry(
        bank_size=K,
        window=W,
        stride=stride,
        n_windows=n_w,
        starts=starts,
        n_slots=n_slots,
        n_heads=n_heads,
        query_dim=qdim,
        coverage=W / max(K, 1),
        compression=S / max(n_slots, 1),
        sliding=n_w >= 2,
    )


class SlidingWindowPerceiverCompressor(nn.Module):
    """K learned queries per overlapping window, pooling token K/V (E30).

    Slot = attention-weighted sum of the window's `k_raw`/`v` — a filter, not a
    mean. No residual mean, no slot–slot mixer. Queries that auto-fit does not
    use still participate via `participation()` so DDP stays happy.
    """

    def __init__(self, cfg: PerceiverARConfig):
        super().__init__()
        self.bank_max = max(1, int(getattr(cfg, "swp_bank_size", SWP_BANK_DEFAULT) or SWP_BANK_DEFAULT))
        self.g, self.dh = cfg.num_kv_heads, cfg.head_dim
        self.cfg_ref = cfg
        n_heads, qdim = swp_resolved_heads(cfg)
        self.n_heads, self.query_dim = n_heads, qdim
        self.dh_q = qdim // n_heads
        self.norm = nn.RMSNorm(cfg.hidden_size)
        self.q = nn.Parameter(torch.randn(self.bank_max, n_heads, self.dh_q) * cfg.init_std)
        self.wk = nn.Linear(cfg.hidden_size, n_heads * self.dh_q, bias=False)
        self.q_norm = nn.RMSNorm(self.dh_q)
        self.k_norm_q = nn.RMSNorm(self.dh_q)
        # Flat so Muon treats it as a bias, matching ConceptPooler.pos_bias.
        max_w = max(self.bank_max * max(1, int(getattr(cfg, "swp_coverage", SWP_COVERAGE_DEFAULT) or SWP_COVERAGE_DEFAULT)),
                    int(getattr(cfg, "swp_window", 0) or 0),
                    1)
        self.pos_bias = nn.Parameter(torch.zeros(max_w * n_heads))
        self._last_entropy = None
        self._last_geometry = None

    def geometry(self, S: int) -> SWPGeometry:
        return swp_geometry(self.cfg_ref, S)

    def n_slots(self, S: int) -> int:
        return self.geometry(S).n_slots

    def participation(self) -> torch.Tensor:
        return self.q.sum() + self.wk.weight.sum() + self.pos_bias.sum()

    def slot_positions(self, pos: torch.Tensor) -> torch.Tensor:
        """pos [B,S] → slot RoPE positions [B,C] (page centres inside each window)."""
        B, S = pos.shape
        geo = self.geometry(S)
        device = pos.device
        starts = torch.tensor(geo.starts, device=device, dtype=torch.long)
        page = max(geo.window // max(geo.bank_size, 1), 1)
        qix = torch.arange(geo.bank_size, device=device)
        off = (qix + 1) * page - 1
        idx = (starts[:, None] + off[None, :]).clamp(0, S - 1).reshape(-1)
        return pos.index_select(1, idx)

    def forward(self, h, k_raw, v, k_norm, valid: Optional[torch.Tensor] = None,
                window_valid: Optional[torch.Tensor] = None):
        """h [B,S,d], k_raw/v [B,S,g,dh] → (k̄, v̄) [B,C,g,dh].

        `window_valid` [B,n_w,W] is the per-window pooling rule (own earliest side, one
        document). When given it replaces the global `valid` mask, which let a window pool
        tokens that were only poolable in *another* window (a future-token leak)."""
        B, S, d = h.shape
        geo = self.geometry(S)
        self._last_geometry = geo
        n_w, W, K = geo.n_windows, geo.window, geo.bank_size
        H, dh_q = self.n_heads, self.dh_q
        g, dh = self.g, self.dh
        device = h.device
        starts = torch.tensor(geo.starts, device=device, dtype=torch.long)
        tok = starts[:, None] + torch.arange(W, device=device)
        in_range = tok < S
        tok = tok.clamp(max=max(S - 1, 0))
        ok = in_range[None].expand(B, n_w, W)
        if window_valid is not None:
            ok = ok & window_valid.bool()
        elif valid is not None:
            ok = ok & valid.bool()[:, tok]
        hn = self.norm(h)
        hb = hn[:, tok]  # [B, n_w, W, d]
        k_win = k_raw[:, tok]  # [B, n_w, W, g, dh]
        v_win = v[:, tok]
        k_s = self.k_norm_q(self.wk(hb).view(B, n_w, W, H, dh_q))
        q = self.q_norm(self.q[:K])  # [K, H, dh_q]
        logits = torch.einsum("khd,bnwhd->bnkhw", q, k_s) / math.sqrt(dh_q)
        bias = self.pos_bias[: W * H].view(W, H).t()  # [H, W]
        logits = logits + bias[None, None, None]
        logits = logits.masked_fill(~ok[:, :, None, None, :], float("-inf"))
        w_h = torch.softmax(logits.float(), dim=-1)
        w_h = torch.nan_to_num(w_h, nan=0.0)
        w = w_h.mean(dim=3).to(k_raw.dtype)  # [B, n_w, K, W]
        live = ok.any(dim=-1)  # [B, n_w]
        w = w * live[:, :, None, None].to(w.dtype)
        k_bar = torch.einsum("bnkw,bnwgd->bnkgd", w, k_win)
        v_bar = torch.einsum("bnkw,bnwgd->bnkgd", w, v_win)
        k_bar = k_bar.reshape(B, n_w * K, g, dh)
        v_bar = v_bar.reshape(B, n_w * K, g, dh)
        # unused auto-fit tail (queries, pos_bias beyond W) stays in the graph for DDP
        extra_q = self.q[K:].sum() if K < self.bank_max else self.q.sum() * 0.0
        used_b = W * H
        extra_b = self.pos_bias[used_b:].sum() if used_b < self.pos_bias.numel() else self.pos_bias.sum() * 0.0
        z = (extra_q + extra_b).to(k_bar.dtype) * 0.0
        # entropy of the token distribution (diagnostic: smear ≈ log W)
        w_tok = w.clamp(min=0)
        w_tok = w_tok / w_tok.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        ent = -(w_tok * (w_tok + 1e-12).log()).sum(dim=-1)
        live_f = live[:, :, None].expand_as(ent)
        if bool(live_f.any()):
            self._last_entropy = float(ent[live_f].mean().detach())
        else:
            self._last_entropy = float("nan")
        return k_norm(k_bar) + z, v_bar + z


class PrefixAEHead(nn.Module):
    """Weak linear reconstruction of each r-token block from its slot K (E26)."""

    def __init__(self, g: int, dh: int, r: int, vocab: int):
        super().__init__()
        self.r = int(r)
        self.vocab = int(vocab)
        self.proj = nn.Linear(g * dh, self.r * vocab, bias=False)

    def forward(self, k_bar: torch.Tensor) -> torch.Tensor:
        """k_bar [B,nb,g,dh] → logits [B,nb,r,V]."""
        B, nb, g, dh = k_bar.shape
        return self.proj(k_bar.reshape(B, nb, g * dh)).view(B, nb, self.r, self.vocab)


def attend_message(q, k, v, k_bar, v_bar, *, ctx: MessageCtx, key_valid, backend, block_masks=None):
    """Global read over raw keys ‖ message slots with the E21 mask. Returns [B,S,h,dh]."""
    B, S, h, dh = q.shape
    g = k.shape[2]
    if backend == "flash":
        raise NotImplementedError("message boundary needs flex or sdpa")
    K = torch.cat([k, k_bar.to(k.dtype)], dim=1)
    V = torch.cat([v, v_bar.to(v.dtype)], dim=1)
    qt, kt, vt = (t.transpose(1, 2) for t in (q, K, V))
    if backend == "flex":
        from torch.nn.attention.flex_attention import create_block_mask

        memo_key = ("message", ctx.override, ctx.n_slots, getattr(ctx, "anchor_mode", "none"), ctx.raw_window)
        if block_masks is not None and memo_key in block_masks:
            bm = block_masks[memo_key]
        else:
            pred = make_message_mask_pred(S, ctx, key_valid)
            if S * (S + ctx.n_slots) > _CHUNKED_MASK_PAIRS and getattr(ctx, "anchor_mode", "none") == "none":
                bm = _band_block_mask(pred, B=B, Q_LEN=S, KV_LEN=S + ctx.n_slots, window=int(ctx.raw_window or 0),
                                      causal=True, device=q.device, extra_blocks=_message_extra_blocks(ctx, S))
            elif S * (S + ctx.n_slots) > _CHUNKED_MASK_PAIRS:
                bm = _chunked_block_mask(pred, B=B, Q_LEN=S, KV_LEN=S + ctx.n_slots, device=q.device)
            else:
                bm = create_block_mask(pred, B=B, H=None, Q_LEN=S, KV_LEN=S + ctx.n_slots, device=q.device,
                                       _compile=torch.cuda.is_available())
            if block_masks is not None:
                block_masks[memo_key] = bm
        qt, kt, vt = qt.contiguous(), kt.contiguous(), vt.contiguous()
        out = _get_flex()(qt, kt, vt, block_mask=bm, enable_gqa=(g != h))
        return out.transpose(1, 2)
    if g != h:
        rep = h // g
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)
    mask = dense_message_mask(S, ctx, key_valid, q.device)
    out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask)
    return out.transpose(1, 2)


# --------------------------------------------------------------------------------------
# Input: tiny factorized embedding + hashed n-gram tables
# --------------------------------------------------------------------------------------

_HASH_PRIMES = (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F, 0x165667B1)


def hashed_ngram_ids(
    ids: torch.Tensor, order: int, buckets: int, doc_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Deterministic bucket id of the n-gram ending at each position.

    Positions whose n-gram would cross the sequence start (or a document boundary when
    `doc_ids` is given) use a sentinel for the out-of-range part, so the first token of a
    document always hashes the same way regardless of what precedes it.
    """
    B, S = ids.shape
    sentinel = torch.full_like(ids, -1)
    acc = torch.zeros_like(ids, dtype=torch.int64)
    for j in range(order):
        shifted = ids[:, : S - j] if j == 0 else torch.cat([sentinel[:, :j], ids[:, : S - j]], dim=1)
        if doc_ids is not None and j > 0:
            same = torch.cat(
                [torch.zeros(B, j, dtype=torch.bool, device=ids.device), doc_ids[:, j:] == doc_ids[:, : S - j]],
                dim=1,
            )
            shifted = torch.where(same, shifted, sentinel)
        acc = acc ^ ((shifted.to(torch.int64) + 2) * _HASH_PRIMES[j % len(_HASH_PRIMES)])
        acc = acc & 0x7FFFFFFFFFFFFFFF
    return acc % buckets


class TinyHashedEmbedding(nn.Module):
    """token table [V,e] + Σ n-gram tables [buckets,e] → MLP up-projection to d → RMSNorm."""

    def __init__(self, cfg: PerceiverARConfig):
        super().__init__()
        e, d = cfg.token_embedding_dim, cfg.hidden_size
        self.orders = cfg.ngram_orders
        self.buckets = cfg.ngram_buckets
        self.tok = nn.Embedding(cfg.vocab_size, e)
        self.ngram = nn.ModuleList([nn.Embedding(cfg.ngram_buckets, e) for _ in self.orders])
        self.up0 = nn.Linear(e, d, bias=False)
        self.up1 = nn.Linear(e, d, bias=False)
        self.up2 = nn.Linear(d, d, bias=False)
        self.norm = nn.RMSNorm(d)

    def forward(self, ids: torch.Tensor, doc_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.tok(ids)
        for order, table in zip(self.orders, self.ngram):
            x = x + table(hashed_ngram_ids(ids, order, self.buckets, doc_ids))
        h = self.up0(x) + self.up2(F.silu(self.up1(x)))
        return self.norm(h)


# --------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------


def rope_cos_sin(positions: torch.Tensor, dim: int, theta: float, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """positions [B,S] → cos, sin [B,S,1,dim] (rotate-half convention)."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
    freqs = positions.to(torch.float32)[..., None] * inv  # [B,S,dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype)[:, :, None, :], emb.sin().to(dtype)[:, :, None, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class Attention(nn.Module):
    def __init__(self, cfg: PerceiverARConfig, layer_idx: int, pattern: str, window: int):
        super().__init__()
        d, h, g, dh = cfg.hidden_size, cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
        self.h, self.g, self.dh = h, g, dh
        self.pattern, self.window = pattern, window
        self.layer_idx = layer_idx
        self.backend = cfg.attn_backend
        self.causal = True
        self.sink = bool(getattr(cfg, "swa_sink", False)) and pattern == "swa"
        self.wq = nn.Linear(d, h * dh, bias=False)
        self.wk = nn.Linear(d, g * dh, bias=False)
        self.wv = nn.Linear(d, g * dh, bias=False)
        self.wo = nn.Linear(h * dh, d, bias=False)
        self.q_norm = nn.RMSNorm(dh)
        self.k_norm = nn.RMSNorm(dh)
        self.use_rope = True
        # SSMax-style length-aware query scale on the unbounded read: q *= s * log(n_visible).
        # s is learnable, initialised so the factor is exactly 1 at n = global_scale_ref.
        self.logit_scale = None
        if pattern == "full" and getattr(cfg, "global_logit_scale", "none") == "log":
            self.logit_scale = nn.Parameter(torch.tensor(1.0 / math.log(cfg.global_scale_ref)))
        self.value_embed = None
        if layer_idx in cfg.value_embed_layers:
            self.value_embed = nn.Embedding(cfg.vocab_size, cfg.value_embed_dim)
            self.value_proj = nn.Linear(cfg.value_embed_dim, g * dh, bias=False)
            self.value_lambda = nn.Parameter(torch.tensor(0.5))
        # Exclusive write: E21 block-mean or E30 overlapping Perceiver banks.
        write = str(getattr(cfg, "message_write", "block_mean") or "block_mean")
        if getattr(cfg, "message_enabled", False) and pattern == "full":
            if write == "sw_perceiver":
                self.compressor = SlidingWindowPerceiverCompressor(cfg)
            elif write == "latent_memory":
                self.compressor = None  # E31: slots come from the model-level writer (ctx.slots)
            else:
                self.compressor = KVCompressor(cfg)
        else:
            self.compressor = None
        self.prefix_ae = bool(getattr(cfg, "message_prefix_ae", False))
        self.prefix_ae_stopgrad = bool(getattr(cfg, "message_prefix_ae_stopgrad_answer", True))
        self._last_k_bar = None
        self._last_v_bar = None

    def kv_raw(self, x: torch.Tensor, ids: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(k before k_norm, v incl. the value-embedding term) — the pooling inputs of the compressor."""
        B, S, _ = x.shape
        k_raw = self.wk(x).view(B, S, self.g, self.dh)
        v = self.wv(x).view(B, S, self.g, self.dh)
        if self.value_embed is not None and ids is not None:
            ve = self.value_proj(self.value_embed(ids)).view(B, S, self.g, self.dh)
            v = v + self.value_lambda * ve
        return k_raw, v

    def kv(self, x: torch.Tensor, ids: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        k_raw, v = self.kv_raw(x, ids)
        return self.k_norm(k_raw), v

    def message_slots(self, x, k_raw, v, ctx: MessageCtx, key_valid, cfg_rope_theta: float, *, rope: bool = True):
        """(k̄, v̄) [B,nb,g,dh] for this forward: external slots (receiver-only forward) or the
        compressor over the block input, RoPE'd at each slot's position unless `rope=False`
        (in-place path applies token-position RoPE after scatter)."""
        if ctx.external is not None:
            return ctx.external
        if ctx.slots is not None:
            return ctx.slots
        valid = key_valid
        if ctx.pool_valid is not None:
            pv = ctx.pool_valid.bool()
            valid = pv if valid is None else (valid.bool() & pv)
        if isinstance(self.compressor, SlidingWindowPerceiverCompressor) and ctx.window_valid is not None:
            k_bar, v_bar = self.compressor(x, k_raw, v, self.k_norm, valid, window_valid=ctx.window_valid)
        else:
            k_bar, v_bar = self.compressor(x, k_raw, v, self.k_norm, valid)
        self._last_k_bar = k_bar
        self._last_v_bar = v_bar
        if self.prefix_ae and self.prefix_ae_stopgrad:
            k_bar, v_bar = k_bar.detach(), v_bar.detach()
        if rope and self.use_rope:
            cos_s, sin_s = rope_cos_sin(ctx.slot_pos, self.dh, cfg_rope_theta, k_bar.dtype)
            k_bar = apply_rope(k_bar, cos_s, sin_s)
        return k_bar, v_bar

    def _scale_q(self, q, pos):
        if self.logit_scale is not None:
            if pos is None:
                raise RuntimeError("global_logit_scale='log' needs per-token positions")
            n_vis = (pos + 1).to(q.dtype)
            q = q * (self.logit_scale * torch.log(n_vis))[:, :, None, None]
        return q

    def _extra_exclusive_attends(self, x, out, *, extra, cos, sin, pos, attend_fn, rewrite_kv=None):
        """Re-read exclusive slot K/V with queries updated by the previous hop.

        `attend_fn(q) -> [B,S,h,dh]`. Extra=0 is a no-op (byte-identical first hop).
        Default: K/V stay the snapshot from the first hop — not recomputed, not raw prefix.
        When `rewrite_kv` is set (`message_update_slot_kv`), it is called with the
        post-attend residual `h` before each extra hop and must return a new
        `attend_fn` bound to rewritten exclusive slot K/V. Still not full prefix.
        """
        extra = int(extra or 0)
        if extra <= 0:
            return out
        B, S, _ = x.shape
        h = x
        for _ in range(extra):
            h = h + out
            q = self.q_norm(self.wq(h).view(B, S, self.h, self.dh))
            if self.use_rope:
                q = apply_rope(q, cos, sin)
            q = self._scale_q(q, pos)
            if rewrite_kv is not None:
                attend_fn = rewrite_kv(h)
            o = attend_fn(q)
            out = self.wo(o.reshape(B, S, self.h * self.dh))
        return out

    def forward(self, x, *, ids, cos, sin, key_valid, doc_ids, cu_seqlens, block_masks=None, sink_pos=None, pos=None,
                message: Optional[MessageCtx] = None, rope_theta: float = 500000.0):
        B, S, _ = x.shape
        q = self.q_norm(self.wq(x).view(B, S, self.h, self.dh))
        k_raw, v = self.kv_raw(x, ids)
        k_un = self.k_norm(k_raw)
        extra = int(getattr(message, "extra_slot_attends", 0) or 0) if message is not None else 0
        update_kv = bool(getattr(message, "update_slot_kv", False)) if message is not None else False
        use_inplace = (
            message is not None
            and self.compressor is not None
            and bool(getattr(message, "inplace", False))
            and message.override in ("real", "swapped")
            and message.external is None
        )
        if use_inplace:
            raw_kv = bool(getattr(message, "inplace_raw_kv", False))
            if raw_kv:
                k_mix, v_mix, replace = mix_inplace_kv(k_un, v, k_un, v, message)
            else:
                k_bar, v_bar = self.message_slots(x, k_raw, v, message, key_valid, rope_theta, rope=False)
                if message.override == "swapped":
                    k_bar, v_bar = k_bar.roll(1, dims=0), v_bar.roll(1, dims=0)
                k_mix, v_mix, replace = mix_inplace_kv(k_un, v, k_bar, v_bar, message)
            k_tok, v_tok = k_un, v  # un-RoPE'd first-hop token K/V (non-slot positions stay these)
            if self.use_rope:
                q = apply_rope(q, cos, sin)
                k_mix = apply_rope(k_mix, cos, sin)
            q = self._scale_q(q, pos)
            o = attend_inplace(
                q, k_mix, v_mix, ctx=message, replace=replace, key_valid=key_valid,
                backend=self.backend, block_masks=block_masks,
            )
            out = self.wo(o.reshape(B, S, self.h * self.dh))

            def _rewrite_inplace(h):
                k_raw2, v2 = self.kv_raw(h, ids)
                k_un2 = self.k_norm(k_raw2)
                if raw_kv:
                    k_bar2, v_bar2 = k_un2, v2
                else:
                    k_bar2, v_bar2 = self.message_slots(
                        h, k_raw2, v2, message, key_valid, rope_theta, rope=False,
                    )
                    if message.override == "swapped":
                        k_bar2, v_bar2 = k_bar2.roll(1, dims=0), v_bar2.roll(1, dims=0)
                k_mix2, v_mix2, _ = mix_inplace_kv(k_tok, v_tok, k_bar2, v_bar2, message)
                if self.use_rope:
                    k_mix2 = apply_rope(k_mix2, cos, sin)
                return lambda qq: attend_inplace(
                    qq, k_mix2, v_mix2, ctx=message, replace=replace, key_valid=key_valid,
                    backend=self.backend, block_masks=block_masks,
                )

            out = self._extra_exclusive_attends(
                x, out, extra=extra, cos=cos, sin=sin, pos=pos,
                attend_fn=lambda qq: attend_inplace(
                    qq, k_mix, v_mix, ctx=message, replace=replace, key_valid=key_valid,
                    backend=self.backend, block_masks=block_masks,
                ),
                rewrite_kv=_rewrite_inplace if update_kv else None,
            )
            if raw_kv:
                out = out + 0.0 * self.compressor.participation().to(out.dtype)
            return out
        k = k_un
        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q = self._scale_q(q, pos)
        if message is not None and (self.compressor is not None or message.slots is not None):
            k_bar, v_bar = self.message_slots(x, k_raw, v, message, key_valid, rope_theta)
            if message.override == "swapped":
                k_bar, v_bar = k_bar.roll(1, dims=0), v_bar.roll(1, dims=0)
            o = attend_message(q, k, v, k_bar, v_bar, ctx=message, key_valid=key_valid,
                               backend=self.backend, block_masks=block_masks)
            out = self.wo(o.reshape(B, S, self.h * self.dh))

            def _rewrite_concat(h):
                k_raw2, v2 = self.kv_raw(h, ids)
                k_bar2, v_bar2 = self.message_slots(h, k_raw2, v2, message, key_valid, rope_theta)
                if message.override == "swapped":
                    k_bar2, v_bar2 = k_bar2.roll(1, dims=0), v_bar2.roll(1, dims=0)
                return lambda qq: attend_message(
                    qq, k, v, k_bar2, v_bar2, ctx=message, key_valid=key_valid,
                    backend=self.backend, block_masks=block_masks,
                )

            return self._extra_exclusive_attends(
                x, out, extra=extra, cos=cos, sin=sin, pos=pos,
                attend_fn=lambda qq: attend_message(
                    qq, k, v, k_bar, v_bar, ctx=message, key_valid=key_valid,
                    backend=self.backend, block_masks=block_masks,
                ),
                rewrite_kv=_rewrite_concat if update_kv else None,
            )
        o = attend(
            q, k, v, pattern=self.pattern, window=self.window, key_valid=key_valid,
            doc_ids=doc_ids, backend=self.backend, causal=self.causal, cu_seqlens=cu_seqlens,
            block_masks=block_masks, sink=self.sink, sink_pos=sink_pos,
        )
        o = self.wo(o.reshape(B, S, self.h * self.dh))
        if self.compressor is not None:
            o = o + 0.0 * self.compressor.participation().to(o.dtype)
        return o


class SwiGLU(nn.Module):
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.gate = nn.Linear(d, ff, bias=False)
        self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: PerceiverARConfig, layer_idx: int, pattern: str, window: int, has_skip: bool = False):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.hidden_size)
        self.attn = Attention(cfg, layer_idx, pattern, window)
        self.mlp_norm = nn.RMSNorm(cfg.hidden_size)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)
        # x0 re-injection (α=1, β=0 at init → identity) and, only on layers that consume a
        # U-net skip, its weight σ (0 at init). Layers without a skip get no σ so every
        # parameter participates in every forward (DDP unused-parameter check).
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.sigma = nn.Parameter(torch.tensor(0.0)) if has_skip else None

    def mix(self, x, x0, skip):
        """The residual-stream input this block's attention actually sees: x0 re-injection plus
        the U-net skip. Shared by `forward` and by `PerceiverARLM._run_layers(capture_input_of=)`
        so `prefix_kv` reconstructs the global layer's K/V exactly, wherever that layer sits."""
        x = self.alpha * x + self.beta * x0
        if skip is not None:
            if self.sigma is None:
                raise RuntimeError("skip passed to a layer built without a U-net skip weight")
            x = x + self.sigma * skip
        return x

    def forward(self, x, x0, skip, *, ids, cos, sin, key_valid, doc_ids, cu_seqlens, block_masks=None, sink_pos=None, pos=None,
                message=None, rope_theta=500000.0):
        x = self.mix(x, x0, skip)
        x = x + self.attn(self.attn_norm(x), ids=ids, cos=cos, sin=sin, key_valid=key_valid,
                          doc_ids=doc_ids, cu_seqlens=cu_seqlens, block_masks=block_masks, sink_pos=sink_pos, pos=pos,
                          message=message, rope_theta=rope_theta)
        x = x + self.mlp(self.mlp_norm(x))
        return x


def _call_block(layer, x, x0, skip, **kwargs):
    return layer(x, x0, skip, **kwargs)


# --------------------------------------------------------------------------------------
# Chunked soft-capped CE (+ z-loss) — never materialises [B,S,V]
# --------------------------------------------------------------------------------------


@torch.no_grad()
def per_token_ce_chunked(hidden, weight, labels, block_size, softcap) -> torch.Tensor:
    """Per-position CE [B,S] (0 where label == -100) without materialising [B,S,V]."""
    B, S, H = hidden.shape
    out = hidden.new_zeros((B, S), dtype=torch.float32)
    bs = block_size if block_size and block_size > 0 else S
    for s in range(0, S, bs):
        e = min(s + bs, S)
        logits = F.linear(hidden[:, s:e], weight).float()
        if softcap and softcap > 0:
            logits = softcap * torch.tanh(logits / softcap)
        V = logits.shape[-1]
        ce = F.cross_entropy(
            logits.reshape(-1, V), labels[:, s:e].reshape(-1), ignore_index=-100, reduction="none"
        )
        out[:, s:e] = ce.view(B, e - s)
    return out


def _chunk_ce(hidden, weight, labels, softcap, z_coef):
    logits = F.linear(hidden, weight).float()
    if softcap and softcap > 0:
        logits = softcap * torch.tanh(logits / softcap)
    V = logits.shape[-1]
    flat, lab = logits.reshape(-1, V), labels.reshape(-1)
    ce = F.cross_entropy(flat, lab, ignore_index=-100, reduction="sum")
    valid = lab != -100
    if z_coef and z_coef > 0:
        lse = torch.logsumexp(flat, dim=-1)
        z = (lse.square() * valid).sum() * z_coef
    else:
        z = ce.new_zeros(())
    return ce, z, valid.sum()


def chunked_softcap_ce(hidden, weight, labels, block_size, softcap, z_coef):
    """Sum-CE, sum-z-loss and count over chunks; chunks are re-computed in backward."""
    B, S, H = hidden.shape
    ce_tot = hidden.new_zeros((), dtype=torch.float32)
    z_tot = hidden.new_zeros((), dtype=torch.float32)
    n_tot = torch.zeros((), dtype=torch.long, device=hidden.device)
    bs = block_size if block_size and block_size > 0 else S
    for s in range(0, S, bs):
        e = min(s + bs, S)
        hb, lb = hidden[:, s:e], labels[:, s:e]
        if hidden.requires_grad or weight.requires_grad:
            ce, z, n = torch_checkpoint(_chunk_ce, hb, weight, lb, softcap, z_coef, use_reentrant=False)
        else:
            ce, z, n = _chunk_ce(hb, weight, lb, softcap, z_coef)
        ce_tot, z_tot, n_tot = ce_tot + ce, z_tot + z, n_tot + n
    return ce_tot, z_tot, n_tot


def _liger_flce():
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    except Exception:  # pragma: no cover
        return None
    return LigerFusedLinearCrossEntropyLoss


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class PerceiverARLM(PreTrainedModel):
    config_class = PerceiverARConfig
    base_model_prefix = "perceiver_ar"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]

    def __init__(self, config: PerceiverARConfig):
        super().__init__(config)
        cfg = config
        self.embed = TinyHashedEmbedding(cfg)
        patterns = cfg.layer_patterns()
        n = len(patterns)
        n_skip = n // 2
        # U-net: layers [0, n_skip) push skips; layers [n - n_skip, n) pop them in reverse.
        self.layers = nn.ModuleList(
            [Block(cfg, i, p, w, has_skip=(i >= n - n_skip)) for i, (p, w) in enumerate(patterns)]
        )
        for j, i in enumerate(cfg.stack_indices):
            layer = self.layers[i]
            if cfg.nope_every and ((j + 1) % cfg.nope_every == 0):
                layer.attn.use_rope = False
            if cfg.block_attention_mode == "bidirectional":
                layer.attn.causal = False
        if cfg.global_nope:
            for i in cfg.resolved_global_positions:
                self.layers[i].attn.use_rope = False
        self.final_norm = nn.RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.memory_writer = (
            LatentMemoryWriter(cfg)
            if str(getattr(cfg, "message_write", "block_mean") or "block_mean") == "latent_memory"
            else None
        )
        if cfg.write_back_hook:
            g, dh = cfg.num_kv_heads, cfg.head_dim
            self.write_back_proj = nn.Linear(cfg.hidden_size, 2 * g * dh, bias=False)
        self.gradient_checkpointing = False
        self._flce = None
        self._message_override = "real"
        self._last_message_ctx = None
        self._last_prefix_ae = None
        if cfg.message_prefix_ae:
            self.prefix_ae_head = PrefixAEHead(
                cfg.num_kv_heads, cfg.head_dim, cfg.message_compress_ratio, cfg.vocab_size,
            )
        else:
            self.prefix_ae_head = None
        self.post_init()
        # Zero-init the residual-writing projections (muP-like, modded-nanogpt).
        # At seq≥512 the 1/S attention mass is too small to open a dead `wo`; the probe can
        # disable this (see `zero_init_residuals=False`) without changing E18 checkpoints.
        if cfg.zero_init_residuals:
            for layer in self.layers:
                nn.init.zeros_(layer.attn.wo.weight)
                nn.init.zeros_(layer.mlp.down.weight)
            if cfg.write_back_hook:
                nn.init.zeros_(self.write_back_proj.weight)
        for layer in self.layers:
            if layer.attn.compressor is not None and getattr(layer.attn.compressor, "delta", None) is not None:
                nn.init.zeros_(layer.attn.compressor.delta.weight)   # post_init re-inits Linears
        if self.memory_writer is not None:
            nn.init.zeros_(self.memory_writer.to_v_state.weight)   # E31: values start as content

    # -- HF plumbing ----------------------------------------------------------------
    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_input_embeddings(self):
        return self.embed.tok

    def set_input_embeddings(self, value):
        self.embed.tok = value

    def get_output_embeddings(self):
        return self.lm_head

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    @property
    def global_kv_space(self) -> tuple[int, int]:
        """(kv_heads, head_dim) of the global read layer — the message/prefix format."""
        return self.config.num_kv_heads, self.config.head_dim

    @property
    def full_layer_indices(self) -> list[int]:
        """Layers currently attending with the unbounded `full` pattern (the global read in
        perceiver mode; every layer in dense mode)."""
        return [i for i, l in enumerate(self.layers) if l.attn.pattern == "full"]

    @contextmanager
    def reach_override(self, window: Optional[int]):
        """Eval-time reach ablation (E18 P3 instrument): for the duration of the block every
        `full` layer attends as `swa(window)` instead — the global read(s) in perceiver mode,
        every layer in dense mode. `window=None` is a no-op. Weights, tokens and the local stack
        are untouched, so a paired comparison against the unrestricted model isolates how much
        the loss at position p depends on *direct* access to keys further than `window` back.
        Positions p < window see exactly the same keys and therefore compute exactly the same
        loss — a built-in check for the probe. Yields the list of touched layer indices."""
        if window is None:
            yield []
            return
        if int(window) < 1:
            raise ValueError("reach_override window must be >= 1 or None")
        originals: list[tuple[int, str, int]] = []
        for i in self.full_layer_indices:
            attn = self.layers[i].attn
            originals.append((i, attn.pattern, attn.window))
            attn.pattern, attn.window = "swa", int(window)
        try:
            yield [i for i, _, _ in originals]
        finally:
            for i, pat, win in originals:
                self.layers[i].attn.pattern, self.layers[i].attn.window = pat, win

    @contextmanager
    def message_override(self, mode: Optional[str]):
        """E21 probe control for the message channel: `none` (receivers get no slots — the
        floor; sparse anchors still join if `message_global_anchors` is on — E27
        anchors-only), `swapped` (slots of the neighbouring batch row — a wrong
        message), `raw` (receivers read the uncompressed prefix K/V across the
        boundary — the ceiling), `real` / None (no-op). Local layers stay severed
        in every mode unless `message_keep_local_swa`, so the paired differences
        isolate the channel. Slots-only is `real` with `message_global_anchors=none`.
        """
        if mode in (None, "real"):
            yield
            return
        if mode not in ("none", "swapped", "raw"):
            raise ValueError(f"unknown message_override {mode!r}")
        prev = self._message_override
        self._message_override = mode
        try:
            yield
        finally:
            self._message_override = prev

    # -- E21 message geometry ---------------------------------------------------------
    @staticmethod
    def _doc_starts(doc: torch.Tensor) -> torch.Tensor:
        starts = torch.ones_like(doc, dtype=torch.bool)
        starts[:, 1:] = doc[:, 1:] != doc[:, :-1]
        return starts

    def _swp_slot_tensors(
        self,
        side: torch.Tensor,
        doc: torch.Tensor,
        pos: torch.Tensor,
        key_valid: Optional[torch.Tensor],
    ):
        """Window banks → slot_doc/side/pos, pool_valid [B,S] and window_valid [B,n_w,W].

        Each window pools only its own earliest side of one document (`window_pick`), computed
        per window. `pool_valid` (the union) is kept for diagnostics; the compressor uses
        `window_valid`, which closes the E30 leak where a straddling window pooled receiver
        tokens that were poolable only in a later, receiver-only window.
        """
        B, S = side.shape
        geo = swp_geometry(self.config, S)
        device = side.device
        n_w, W, K = geo.n_windows, geo.window, geo.bank_size
        tok_c, pick, slot_doc_w, slot_side_w, _has = window_pick(side, doc, key_valid, geo.starts, W)
        slot_doc = slot_doc_w.repeat_interleave(K, dim=1)
        slot_side = slot_side_w.repeat_interleave(K, dim=1)
        starts = torch.tensor(geo.starts, device=device, dtype=torch.long)
        page = max(W // max(K, 1), 1)
        qix = torch.arange(K, device=device)
        off = (qix + 1) * page - 1
        pos_idx = (starts[:, None] + off[None, :]).clamp(0, S - 1).reshape(-1)
        slot_pos = pos.index_select(1, pos_idx)
        pool_valid = torch.zeros(B, S, dtype=torch.bool, device=device)
        if bool(pick.any()):
            b_ix = torch.arange(B, device=device)[:, None, None].expand_as(pick)
            tok_exp = tok_c[None].expand(B, n_w, W)
            pool_valid[b_ix[pick], tok_exp[pick]] = True
        return slot_doc, slot_side, slot_pos, pool_valid, pick

    def _lm_slot_tensors(self, side, doc, pos, key_valid):
        """E31 geometry → slot_doc/side (per reader entry) and a placeholder slot_pos
        (overwritten by the writer's expected positions)."""
        B, S = side.shape
        geo = lm_geometry(self.config, S)
        _tok, pick, slot_doc_w, slot_side_w, _has = window_pick(side, doc, key_valid, geo.starts, geo.window)
        rep = geo.latents * geo.reader_tokens
        slot_doc = slot_doc_w.repeat_interleave(rep, dim=1)
        slot_side = slot_side_w.repeat_interleave(rep, dim=1)
        slot_pos = torch.zeros_like(slot_doc)
        return slot_doc, slot_side, slot_pos, pick

    def _message_context(
        self,
        input_ids: torch.Tensor,
        doc_ids: Optional[torch.Tensor],
        key_valid: Optional[torch.Tensor],
        pos: torch.Tensor,
        external: Optional[tuple] = None,
    ) -> Optional[MessageCtx]:
        """Build the boundary geometry, or None when nothing in the batch is a receiver."""
        cfg = self.config
        B, S = input_ids.shape
        dev = input_ids.device
        r = cfg.message_compress_ratio
        self._last_message_ctx = None
        doc = doc_ids.clone() if doc_ids is not None else torch.zeros(B, S, dtype=torch.long, device=dev)
        if key_valid is not None:
            doc = doc.masked_fill(~key_valid.bool(), -1)
        if external is not None:
            k_bar, v_bar, slot_pos = external
            nb = k_bar.shape[1]
            side = torch.ones(B, S, dtype=torch.long, device=dev)
            return MessageCtx(
                side=side, doc=doc, local_doc_ids=doc,
                slot_doc=torch.zeros(B, nb, dtype=torch.long, device=dev),
                slot_side=torch.zeros(B, nb, dtype=torch.long, device=dev),
                slot_pos=slot_pos.to(dev).expand(B, nb) if slot_pos.dim() == 1 else slot_pos.to(dev),
                n_sides=2, override=self._message_override, external=(k_bar, v_bar),
                ratio=r, inplace=cfg.message_slots_inplace,
                inplace_raw_kv=cfg.message_inplace_raw_kv,
                extra_slot_attends=int(getattr(cfg, "message_extra_slot_attends", 0) or 0),
                update_slot_kv=bool(getattr(cfg, "message_update_slot_kv", False)),
                anchor=torch.zeros(B, S, dtype=torch.bool, device=dev),
                anchor_mode="none",
                raw_window=int(getattr(cfg, "message_raw_window", 0) or 0),
            )
        is_b = input_ids == cfg.message_boundary_token_id
        if not bool(is_b.any()):
            return None
        starts = self._doc_starts(doc)
        cum = is_b.long().cumsum(dim=1)
        base = torch.cummax(torch.where(starts, cum - is_b.long(), torch.zeros_like(cum)), dim=1).values
        side = cum - base
        K = int(side.max().item()) + 1
        if getattr(cfg, "message_keep_local_swa", False):
            local = doc
        else:
            local = torch.where(doc < 0, doc, doc * K + side)
        window_valid = None
        write = str(getattr(cfg, "message_write", "block_mean") or "block_mean")
        if write == "sw_perceiver":
            slot_doc, slot_side, slot_pos, pool_valid, window_valid = self._swp_slot_tensors(
                side, doc, pos, key_valid
            )
        elif write == "latent_memory":
            slot_doc, slot_side, slot_pos, window_valid = self._lm_slot_tensors(side, doc, pos, key_valid)
            pool_valid = None
        else:
            nb = -(-S // r)
            pad = nb * r - S
            docp = F.pad(doc, (0, pad), value=-1).view(B, nb, r)
            sidep = F.pad(side, (0, pad), value=-1).view(B, nb, r)
            homog = (docp == docp[..., :1]).all(dim=2) & (sidep == sidep[..., :1]).all(dim=2) & (docp[..., 0] >= 0)
            slot_doc = torch.where(homog, docp[..., 0], torch.full_like(docp[..., 0], -1))
            slot_side = torch.where(homog, sidep[..., 0], torch.zeros_like(sidep[..., 0]))
            end_idx = (torch.arange(nb, device=dev) * r + (r - 1)).clamp(max=S - 1)
            slot_pos = pos[:, end_idx]
            pool_valid = None
            if cfg.message_pool_remainder:
                first_doc, first_side = docp[..., 0], sidep[..., 0]
                same = (
                    (docp == first_doc[..., None])
                    & (sidep == first_side[..., None])
                    & (first_doc[..., None] >= 0)
                )
                live = same.any(dim=2)
                slot_doc = torch.where(live, first_doc, torch.full_like(first_doc, -1))
                slot_side = torch.where(live, first_side.clamp(min=0), torch.zeros_like(first_side))
                pool_ok = same & live[..., None]
                pool_valid = pool_ok.reshape(B, nb * r)[:, :S]
                idx = torch.arange(r, device=dev)
                last = (same.to(torch.long) * (idx + 1)).amax(dim=2).clamp(min=1) - 1
                end_abs = (torch.arange(nb, device=dev) * r + last).clamp(max=S - 1)
                slot_pos = pos.gather(1, end_abs)
            stride = int(getattr(cfg, "message_pack_stride", 0) or 0)
            if stride > 0 and not bool(getattr(cfg, "message_pool_remainder", False)):
                # QUERY-aligned leftover: sender tokens in [0, P % stride) do not fill a
                # complete pack ending at QUERY. r=1 identity would keep them; drop them
                # from exclusive slots unless remainder-on (keep as identity slots).
                is_q = torch.zeros(B, S, dtype=torch.bool, device=dev)
                is_q[:, 0] = side[:, 0] >= 1
                is_q[:, 1:] = (side[:, 1:] >= 1) & (side[:, :-1] == 0)
                qpos = is_q.to(torch.long).argmax(dim=1)
                leftover = torch.where(is_q.any(dim=1), qpos % stride, torch.zeros_like(qpos))
                tok = torch.arange(S, device=dev)[None, :]
                drop_tok = (tok < leftover[:, None]) & (side == 0) & (doc >= 0)
                drop_slot = F.pad(drop_tok, (0, pad), value=False).view(B, nb, r).all(dim=2)
                slot_doc = torch.where(drop_slot, torch.full_like(slot_doc, -1), slot_doc)
        anchor_mode = str(getattr(cfg, "message_global_anchors", "none") or "none")
        token_ids = tuple(int(x) for x in (getattr(cfg, "message_anchor_token_ids", ()) or ()))
        window = int(getattr(cfg, "message_anchor_window", MESSAGE_QUERY_NBHD_DEFAULT) or 0)
        key_len = int(getattr(cfg, "message_anchor_key_len", 0) or 0)
        anchor = build_message_anchors(
            input_ids, side, doc, mode=anchor_mode, token_ids=token_ids, window=window,
            key_len=key_len,
        )
        ctx = MessageCtx(side=side, doc=doc, local_doc_ids=local, slot_doc=slot_doc, slot_side=slot_side,
                          slot_pos=slot_pos, n_sides=K, override=self._message_override,
                          pool_valid=pool_valid, ratio=r, inplace=cfg.message_slots_inplace,
                          inplace_raw_kv=cfg.message_inplace_raw_kv,
                          extra_slot_attends=int(getattr(cfg, "message_extra_slot_attends", 0) or 0),
                          update_slot_kv=bool(getattr(cfg, "message_update_slot_kv", False)),
                          anchor=anchor, anchor_mode=anchor_mode, window_valid=window_valid,
                          raw_window=int(getattr(cfg, "message_raw_window", 0) or 0))
        self._last_message_ctx = ctx
        return ctx

    # -- helpers ----------------------------------------------------------------------
    @staticmethod
    def _positions(S: int, B: int, doc_ids: Optional[torch.Tensor], device) -> torch.Tensor:
        if doc_ids is None:
            return torch.arange(S, device=device)[None].expand(B, S)
        # position within document: reset at every doc boundary
        starts = torch.ones_like(doc_ids, dtype=torch.bool)
        starts[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
        idx = torch.arange(S, device=device)[None].expand(B, S)
        start_idx = torch.where(starts, idx, torch.zeros_like(idx))
        start_idx = torch.cummax(start_idx, dim=1).values
        return idx - start_idx

    def _pad_inputs(self, input_ids, attention_mask, labels, doc_ids):
        m = self.config.attn_pad_multiple
        S = input_ids.shape[1]
        if m <= 1 or S % m == 0:
            return input_ids, attention_mask, labels, doc_ids, S
        pad = m - (S % m)
        pid = self.config.pad_token_id
        input_ids = F.pad(input_ids, (0, pad), value=pid)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            attention_mask[:, S:] = 0
        else:
            attention_mask = F.pad(attention_mask, (0, pad), value=0)
        if labels is not None:
            labels = F.pad(labels, (0, pad), value=-100)
        if doc_ids is not None:
            doc_ids = F.pad(doc_ids, (0, pad), value=-1)
        return input_ids, attention_mask, labels, doc_ids, S

    @staticmethod
    def _needs_key_mask(attention_mask: Optional[torch.Tensor], causal: bool) -> bool:
        """Right-padded batches need no key mask under causal attention: a valid query at
        position t only sees keys <= t, and every pad key sits after the last valid query.
        Pad query rows are garbage but their labels are -100. Left/interior padding or
        bidirectional blocks still need the explicit mask."""
        if attention_mask is None or bool(attention_mask.all()):
            return False
        if not causal:
            return True
        m = attention_mask.to(torch.int64)
        right_padded = bool((m.cumprod(dim=1) == m).all())
        return not right_padded

    def _run_layers(self, input_ids, attention_mask, doc_ids, cu_seqlens, capture_input_of: Optional[int] = None,
                    message_kv: Optional[tuple] = None, position_offset: int = 0):
        cfg = self.config
        B, S = input_ids.shape
        key_valid = None
        if self._needs_key_mask(attention_mask, cfg.block_attention_mode == "causal"):
            key_valid = attention_mask.bool()
        pos0 = self._positions(S, B, doc_ids, input_ids.device)
        pos = pos0 + int(position_offset) if position_offset else pos0
        msg = None
        local_doc = doc_ids
        if cfg.message_enabled:
            msg = self._message_context(input_ids, doc_ids, key_valid, pos, external=message_kv)
            if msg is not None:
                local_doc = msg.local_doc_ids
        x0 = self.embed(input_ids, local_doc)
        if self.memory_writer is not None:
            if msg is not None and msg.external is None:
                out = self.memory_writer(
                    self.embed.tok(input_ids), msg.side, msg.doc, key_valid, pos, cfg.rope_theta
                )
                msg.slots = (out["k"], out["v"])
                msg.slot_pos = out["slot_pos"]
            else:
                x0 = x0 + self.memory_writer.participation().to(x0.dtype)  # DDP: no QUERY in batch
        cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_theta, x0.dtype)
        x = x0
        n = len(self.layers)
        n_skip = n // 2
        skips: list[torch.Tensor] = []
        block_masks: Optional[dict] = {} if cfg.attn_backend == "flex" else None
        sink_pos = None
        if cfg.swa_sink and local_doc is not None:
            local_pos = pos0 if local_doc is doc_ids else self._positions(S, B, local_doc, input_ids.device)
            sink_pos = (torch.arange(S, device=input_ids.device)[None].expand(B, S) - local_pos)
        for i, layer in enumerate(self.layers):
            skip = skips.pop() if (i >= n - n_skip and skips) else None
            if capture_input_of is not None and i == capture_input_of:
                return layer.mix(x, x0, skip)
            is_global = layer.attn.pattern == "full" and (
                layer.attn.compressor is not None or self.memory_writer is not None
            )
            kwargs = dict(ids=input_ids, cos=cos, sin=sin, key_valid=key_valid,
                          doc_ids=(doc_ids if is_global else local_doc),
                          cu_seqlens=cu_seqlens, block_masks=block_masks, sink_pos=sink_pos, pos=pos,
                          message=(msg if is_global else None), rope_theta=cfg.rope_theta)
            if self.gradient_checkpointing and self.training:
                fn = partial(_call_block, layer, **kwargs)
                x = torch_checkpoint(fn, x, x0, skip, use_reentrant=False)
            else:
                x = layer(x, x0, skip, **kwargs)
            if i < n_skip:
                skips.append(x)
        return x

    # -- forward ----------------------------------------------------------------------
    def hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Final-norm hidden states [B, S, d] without materialising logits.

        Probes that only need argmax at a few positions (passkey, copy) project
        `hidden_states(...)[:, positions]` with `lm_head.weight` themselves: full logits at
        S=32k are 16 GB in fp32 (S × V), the hidden states are 50 MB.
        """
        input_ids, attention_mask, _, doc_ids, S_orig = self._pad_inputs(
            input_ids, attention_mask, None, doc_ids
        )
        x = self._run_layers(input_ids, attention_mask, doc_ids, None)
        return self.final_norm(x)[:, :S_orig]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        return_per_token_loss: bool = False,
        return_logits: bool = False,
        message_kv: Optional[tuple] = None,
        position_offset: int = 0,
    ):
        # NOTE: no **kwargs here on purpose. HF Trainer treats any VAR_KEYWORD forward as a
        # model that normalizes its own loss by num_items_in_batch and then skips the division
        # by gradient_accumulation_steps (loss and grad-clip off by accum×). This forward
        # returns a per-microbatch mean; the Trainer must do the accumulation scaling.
        #
        # `message_kv=(k̄, v̄, slot_pos)` + `position_offset=P` is the E21 receiver-only forward:
        # `input_ids` are the tokens from the boundary on, and the prefix is present only as the
        # given slots (what `prefix_kv(..., as_message=True)` returns).
        cfg = self.config
        if message_kv is not None and not cfg.message_enabled:
            raise RuntimeError("message_kv needs message_boundary_token_id >= 0")
        input_ids, attention_mask, labels, doc_ids, S_orig = self._pad_inputs(
            input_ids, attention_mask, labels, doc_ids
        )
        x = self._run_layers(input_ids, attention_mask, doc_ids, cu_seqlens,
                             message_kv=message_kv, position_offset=position_offset)
        h = self.final_norm(x)

        if labels is None or return_logits:
            logits = F.linear(h[:, :S_orig], self.lm_head.weight).float()
            if cfg.logit_softcap:
                logits = cfg.logit_softcap * torch.tanh(logits / cfg.logit_softcap)
            if labels is None:
                return CausalLMOutput(loss=None, logits=logits)

        # next-token targets: position t predicts labels[t+1]
        tgt = labels[:, 1:]
        hid = h[:, :-1]
        if return_per_token_loss:
            per = per_token_ce_chunked(
                hid, self.lm_head.weight, tgt, cfg.chunked_ce_block_size, cfg.logit_softcap
            )[:, : S_orig - 1]
            valid = tgt[:, : S_orig - 1] != -100
            loss = per.sum() / valid.sum().clamp(min=1)
            return CausalLMOutput(loss=loss, logits=None), per, valid

        flce = self._get_flce() if (cfg.use_liger and hid.is_cuda) else None
        if flce is not None:
            n = (tgt != -100).sum().clamp(min=1)
            loss = flce(self.lm_head.weight, hid.reshape(-1, hid.shape[-1]), tgt.reshape(-1)) / n
        else:
            ce, z, n = chunked_softcap_ce(
                hid, self.lm_head.weight, tgt, cfg.chunked_ce_block_size, cfg.logit_softcap, cfg.z_loss
            )
            loss = (ce + z) / n.clamp(min=1)
        loss = self._maybe_add_prefix_ae(loss, input_ids)
        return CausalLMOutput(loss=loss, logits=(logits if return_logits else None))

    def _maybe_add_prefix_ae(self, loss: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Add λ L_AE. Answer-span CE is unchanged; compressor grads come from AE only."""
        cfg = self.config
        self._last_prefix_ae = None
        if not cfg.message_prefix_ae or self.prefix_ae_head is None:
            return loss
        gi = cfg.global_layer_index
        attn = self.layers[gi].attn
        k_bar = getattr(attn, "_last_k_bar", None)
        ctx = self._last_message_ctx
        if k_bar is None or ctx is None:
            return loss + 0.0 * self.prefix_ae_head.proj.weight.sum().to(loss.dtype)
        B, nb, g, dh = k_bar.shape
        r = max(int(ctx.ratio), 1)
        S = input_ids.shape[1]
        pad = nb * r - S
        ids_pad = F.pad(input_ids, (0, pad), value=-100)
        tgt = ids_pad.view(B, nb, r)
        valid = (ctx.slot_doc >= 0) & (ctx.slot_side == 0)
        tgt = tgt.masked_fill(~valid.unsqueeze(-1), -100)
        logits = self.prefix_ae_head(k_bar)
        V = logits.shape[-1]
        n = (tgt != -100).sum().clamp(min=1)
        ae = F.cross_entropy(logits.reshape(-1, V).float(), tgt.reshape(-1), ignore_index=-100, reduction="sum") / n
        pred = logits.argmax(-1)
        tok_ok = tgt != -100
        tok_acc = (pred[tok_ok] == tgt[tok_ok]).float().mean() if bool(tok_ok.any()) else tgt.new_zeros(())
        key_len = int(getattr(cfg, "message_anchor_key_len", 0) or 0)
        token_ids = tuple(int(x) for x in (getattr(cfg, "message_anchor_token_ids", ()) or ()))
        key_acc = tok_acc
        if token_ids:
            key_mask = build_message_anchors(
                input_ids, ctx.side, ctx.doc, mode="key_spans", token_ids=token_ids,
                key_len=key_len,
            )
            key_pad = F.pad(key_mask, (0, pad), value=False).view(B, nb, r)
            key_ok = key_pad & tok_ok
            if bool(key_ok.any()):
                key_acc = (pred[key_ok] == tgt[key_ok]).float().mean()
            else:
                key_acc = tgt.new_zeros(())
        self._last_prefix_ae = {
            "loss": float(ae.detach()),
            "tok_acc": float(tok_acc.detach()) if torch.is_tensor(tok_acc) else float(tok_acc),
            "key_acc": float(key_acc.detach()) if torch.is_tensor(key_acc) else float(key_acc),
            "n": int(n),
        }
        w = float(cfg.message_prefix_ae_weight)
        if w == 0:
            return loss
        return loss + ae * k_bar.new_tensor(w)

    def _get_flce(self):
        if self._flce is None:
            cls = _liger_flce()
            if cls is None:
                self.config.use_liger = False
                return None
            self._flce = cls(
                ignore_index=-100, reduction="sum", softcap=self.config.logit_softcap or None,
                lse_square_scale=self.config.z_loss,
            )
        return self._flce

    # -- family hooks ---------------------------------------------------------------
    @torch.no_grad()
    def prefix_kv(self, input_ids, attention_mask=None, doc_ids=None, as_message: bool = False):
        """K/V of the global read layer for `input_ids` — the one-layer prefix cache and the
        message object for E19/E21. Returns (k, v) each [B,S,g,dh] (RoPE applied to k).

        `as_message=True` (E21) returns the **compressed message** instead: (k̄, v̄, slot_pos) with
        one slot per `message_compress_ratio` tokens. Default is complete blocks only (⌊S/r⌋
        slots — exactly the slots a receiver of the concatenated row would be allowed to read).
        `message_pool_remainder` also keeps the last incomplete block. Feed it to
        `forward(receiver_ids, message_kv=..., position_offset=S)`.
        """
        cfg = self.config
        if cfg.par_mode == "perceiver" and cfg.global_layers < 1:
            raise RuntimeError("prefix_kv needs a global read layer")
        gi = cfg.global_layer_index
        input_ids, attention_mask, _, doc_ids, S = self._pad_inputs(input_ids, attention_mask, None, doc_ids)
        x_in = self._run_layers(input_ids, attention_mask, doc_ids, None, capture_input_of=gi)
        layer = self.layers[gi]
        h = layer.attn_norm(x_in)
        pos = self._positions(input_ids.shape[1], input_ids.shape[0], doc_ids, input_ids.device)
        if as_message:
            if self.memory_writer is not None:
                raise NotImplementedError("prefix_kv(as_message=True) is not wired for latent_memory (E31)")
            if layer.attn.compressor is None:
                raise RuntimeError("as_message needs message_boundary_token_id >= 0")
            k_raw, v = layer.attn.kv_raw(h, input_ids)
            k_bar, v_bar = layer.attn.compressor(h, k_raw, v, layer.attn.k_norm)
            if str(getattr(cfg, "message_write", "block_mean") or "block_mean") == "sw_perceiver":
                slot_pos = layer.attn.compressor.slot_positions(pos[:, :S])
                k_bar, v_bar = k_bar[:, : slot_pos.shape[1]], v_bar[:, : slot_pos.shape[1]]
            else:
                r = cfg.message_compress_ratio
                n_keep = -(-S // r) if cfg.message_pool_remainder else S // r
                k_bar, v_bar = k_bar[:, :n_keep], v_bar[:, :n_keep]
                end_idx = (torch.arange(n_keep, device=input_ids.device) * r + (r - 1)).clamp(max=max(S - 1, 0))
                slot_pos = pos[:, end_idx]
            if layer.attn.use_rope:
                cos_s, sin_s = rope_cos_sin(slot_pos, cfg.head_dim, cfg.rope_theta, k_bar.dtype)
                k_bar = apply_rope(k_bar, cos_s, sin_s)
            return k_bar, v_bar, slot_pos
        k, v = layer.attn.kv(h, input_ids)
        cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_theta, k.dtype)
        if layer.attn.use_rope:
            k = apply_rope(k, cos, sin)
        return k[:, :S], v[:, :S]

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=32, temperature=0.0, top_k=0, eos_token_id=None):
        """v1 sampler without KV cache (full recompute per step) — for probes only."""
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            logits = self(out).logits[:, -1]
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, top_k).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            out = torch.cat([out, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
        return out


# --------------------------------------------------------------------------------------
# Parameter accounting (used by tests and run reports)
# --------------------------------------------------------------------------------------


@dataclass
class ParamBreakdown:
    dense: int
    sparse_tables: int

    @property
    def total(self) -> int:
        return self.dense + self.sparse_tables


def _swp_param_count(cfg: PerceiverARConfig) -> int:
    """Learned query bank + scoring projection (E30). Independent of seq_len."""
    d = cfg.hidden_size
    bank = max(1, int(getattr(cfg, "swp_bank_size", SWP_BANK_DEFAULT) or SWP_BANK_DEFAULT))
    coverage = max(1, int(getattr(cfg, "swp_coverage", SWP_COVERAGE_DEFAULT) or SWP_COVERAGE_DEFAULT))
    n_heads, qdim = swp_resolved_heads(cfg)
    dh_q = qdim // n_heads
    max_w = max(bank * coverage, int(getattr(cfg, "swp_window", 0) or 0), 1)
    return (
        d  # RMSNorm on h
        + bank * n_heads * dh_q  # q
        + d * n_heads * dh_q  # wk
        + dh_q + dh_q  # q_norm, k_norm_q
        + max_w * n_heads  # pos_bias
    )


def analytic_param_count(cfg: PerceiverARConfig) -> ParamBreakdown:
    d, ff, e, V = cfg.hidden_size, cfg.intermediate_size, cfg.token_embedding_dim, cfg.vocab_size
    h, g, dh = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
    L = cfg.total_layers
    per_layer = (d * h * dh) + 2 * (d * g * dh) + (h * dh * d) + 3 * d * ff + 2 * d + 2 * dh + 2
    dense = L * per_layer + (L // 2)                # + one σ per skip-consuming layer
    dense += V * e + 2 * e * d + d * d + d          # tok table + up0/up1/up2 + norm
    dense += d + d * V                              # final norm + head
    if cfg.write_back_hook:
        dense += d * 2 * g * dh
    if getattr(cfg, "global_logit_scale", "none") == "log":
        dense += sum(1 for pat, _ in cfg.layer_patterns() if pat == "full")   # one scalar per full layer
    if getattr(cfg, "message_enabled", False):
        n_full = sum(1 for pat, _ in cfg.layer_patterns() if pat == "full")
        write = str(getattr(cfg, "message_write", "block_mean") or "block_mean")
        if write == "sw_perceiver":
            dense += n_full * _swp_param_count(cfg)
        elif write == "latent_memory":
            with torch.device("meta"):
                dense += sum(p.numel() for p in LatentMemoryWriter(cfg).parameters())
        else:
            dense += n_full * (g * d + d * 2 * g * dh)                             # KVCompressor: u + delta
    sparse = len(cfg.ngram_orders) * cfg.ngram_buckets * e
    n_ve = sum(1 for i in range(L) if i in cfg.value_embed_layers)
    sparse += n_ve * (V * cfg.value_embed_dim)
    dense += n_ve * (cfg.value_embed_dim * g * dh + 1)
    return ParamBreakdown(dense=dense, sparse_tables=sparse)
