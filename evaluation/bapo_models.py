"""Matched-parameter architecture factory for the BAPO capability probe.

Architectures
-------------
- `dense`      Perceiver AR, `par_mode=dense`: full-causal decoder-only, same width/depth as E18.
- `e18`        Perceiver AR, one global read + SWA stack (the E18 architecture).
- `e18_local`  E18 with `global_layers=0`: local window only. Retrieval rungs must sit at the floor.
- `e21`        E18 plus a message boundary at `query` and prefix keys as `KVCompressor` slots (r=16).
               Complete homogeneous blocks by default; `--message_pool_remainder` also pools the
               incomplete last sender block. Probe `--message_override raw|none|swapped` wraps the
               e21 forward in `model.message_override` (default `real` slots).
               `--message_slots_inplace` writes slots into sender prefix positions (KV_LEN=S).
               `--message_inplace_raw_kv` (with inplace) copies token K/V into those positions
               instead of compressor slots; exclusive remainder hiding stays on.
               `--message_identity_slots` bypasses learned compressor `u`/`delta` (frozen
               mean-pool of the r tokens in each block; r=1: hard token K/V copy) through
               the slot/scatter path, not inplace_raw_kv.
               `--message_pack_stride N` (default 0=off) drops exclusive leftover sender
               tokens vs N-token packs tiled to end at QUERY. r=1 identity otherwise
               keeps every sender token (`--message_pool_remainder` is a no-op at r=1).
               Remainder-on keeps the leftover as identity slots. Not the full raw prefix.
               `--message_keep_local_swa` (default off) leaves exclusive slots on the
               global read but does not treat QUERY as a SWA/n-gram document start.
               `--message_extra_slot_attends N` (default 0) re-reads the same exclusive
               slot K/V with updated queries (second hop in slot space; not raw prefix).
               `--message_update_slot_kv` (default off) rewrites exclusive slot K/V from
               the post-attend residual before each extra hop. Extra=0 is unchanged.
               `--global_layers N` (default 1) is sequential full Attention+FFN global
               blocks (E21 exclusive slots each; E18 raw prefix each). Distinct from
               `--stack_layers` (SWA local) and from extra hops inside one Attention.
               `--message_global_anchors {none,query_nbhd,query_side,type_marks,query_nbhd+type,key_spans}`
               (default `none`) leaks a sparse extra subset into exclusive slot K/V:
               `query_nbhd` = 4 sender tokens before QUERY; `query_side` = QUERY plus
               a small window after the boundary (receiver type request); type-mark
               controls; `key_spans` = DNA key-field tokens after each sender keymark
               (E27 hybrid; not the mark, not values). Still not the full raw prefix.
               `--message_prefix_ae` (default off) adds a weak linear reconstruction of
               each complete sender block from its slot (E26 write objective). Compressor
               `u`/`delta` see AE grads only (`--message_prefix_ae_stopgrad_answer`,
               default on). `--message_identity_slots` off so the pooler can move under AE.
- `e30`        E18 + QUERY boundary + overlapping Perceiver banks (`message_write=sw_perceiver`).
               Exclusive concat read of length-scaling window queries (E30). Inplace / identity
               / prefix-AE flags are illegal on this write. Geometry auto-fits `K` so
               `n_windows≥2` on short DNA rows.
- `encdec`     Symmetric encoder-decoder: bidirectional prefix encoder, suffix-only decoder with
               cross-attention. Prefix information cannot take a raw route into the suffix.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch.nn as nn

from evaluation.bapo_metrics import cache_profile
from nn.encdec_lm import EncDecConfig, EncoderDecoderLM
from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM, swp_geometry


ARCHES = ("dense", "e18", "e18_local", "e21", "e30", "encdec", "e30_ctx", "e31_page", "e31_bixt")
# Arches whose global read is exclusive after QUERY (compressed / latent memory only).
EXCLUSIVE_ARCHES = ("e21", "e30", "e30_ctx", "e31_page", "e31_bixt")
SWP_ARCHES = ("e30", "e30_ctx")
E31_ARCHES = ("e31_page", "e31_bixt")


@dataclass
class ArchSpec:
    name: str
    hidden: int = 64
    pre_layers: int = 1
    global_layers: int = 1
    stack_layers: int = 2
    local_window: int = 16
    enc_layers: int = 2
    dec_layers: int = 2
    head_dim: int = 32
    n_kv_heads: int = 1
    # E18 copy-tiny / 32k copy used VE on the retrieving layer. Layer 0 is the SWA pre-encoder;
    # layer 1 is the global read. Dense gets the same two tables so params stay matched.
    value_embed_layers: tuple[int, ...] = (0, 1)
    attn_backend: str = "sdpa"
    global_logit_scale: str = "none"  # "log" = SSMax anti-dilution on full layers
    z_loss: float = 1e-4
    zero_init_residuals: bool = True
    message_compress_ratio: int = 16
    message_boundary_token_id: int = -1
    message_pool_remainder: bool = False
    message_slots_inplace: bool = False
    message_inplace_raw_kv: bool = False
    message_identity_slots: bool = False
    message_pack_stride: int = 0
    message_keep_local_swa: bool = False
    message_extra_slot_attends: int = 0
    message_update_slot_kv: bool = False
    message_global_anchors: str = "none"
    message_anchor_token_ids: tuple[int, ...] = ()
    message_anchor_window: int = 4
    message_anchor_key_len: int = 0
    message_prefix_ae: bool = False
    message_prefix_ae_weight: float = 0.0
    message_prefix_ae_stopgrad_answer: bool = True
    message_write: str = "block_mean"
    swp_bank_size: int = 32
    swp_coverage: int = 8
    swp_window: int = 0
    swp_stride: int = 0
    swp_n_heads: int = 0
    swp_query_dim: int = 0
    swp_auto_fit: bool = True
    # Platform knobs (E31 runs set them for every arch in the job, so the comparison stays fair)
    token_embedding_dim: int = 0          # 0 → min(32, hidden) (ledger default)
    ngram_orders: tuple[int, ...] = (2,)  # () → hashed n-grams off
    # E30 context-only control: causal pre-encoder reach (e30_ctx only)
    ctx_pre_window: int = 64
    # E31 latent memory (nn/latent_memory.py)
    lm_window: int = 256
    lm_stride: int = 192
    lm_latents: int = 32
    lm_latent_dim: int = 512
    lm_heads: int = 8
    lm_writer_dim: int = 256
    lm_enc_layers: int = 2
    lm_rounds: int = 0                    # 0 → 2 for e31_page, 3 for e31_bixt
    lm_competition: bool = True
    lm_null_latent: bool = False
    lm_reader_tokens: int = 5


def _n_heads(hidden: int, head_dim: int) -> int:
    n = max(1, hidden // head_dim)
    return n if n * head_dim <= 4 * hidden else 1


def build_model(arch: str, *, vocab_size: int, seq_len: int, answer_start: int, pad_id: int,
                bos_id: int, eos_id: int, spec: ArchSpec | None = None, seed: int = 0) -> nn.Module:
    spec = spec or ArchSpec(name=arch)
    import torch

    torch.manual_seed(seed)
    if arch == "encdec":
        cfg = EncDecConfig(
            vocab_size=vocab_size,
            hidden_size=spec.hidden,
            intermediate_size=2 * spec.hidden,
            enc_layers=spec.enc_layers,
            dec_layers=spec.dec_layers,
            num_attention_heads=_n_heads(spec.hidden, spec.head_dim),
            head_dim=spec.head_dim,
            answer_start=answer_start,
            pad_token_id=pad_id,
        )
        return EncoderDecoderLM(cfg)

    if arch == "dense":
        par_mode, pre, glob, stack = "dense", spec.pre_layers, spec.global_layers, spec.stack_layers
    elif arch == "e18":
        par_mode, pre, glob, stack = "perceiver", spec.pre_layers, spec.global_layers, spec.stack_layers
    elif arch == "e21":
        if spec.message_boundary_token_id < 0:
            raise ValueError("e21 needs a message_boundary_token_id (DNA/Glyph query control)")
        par_mode, pre, glob, stack = "perceiver", spec.pre_layers, spec.global_layers, spec.stack_layers
    elif arch in SWP_ARCHES or arch in E31_ARCHES:
        if spec.message_boundary_token_id < 0:
            raise ValueError(f"{arch} needs a message_boundary_token_id (DNA/Glyph query control)")
        par_mode, pre, glob, stack = "perceiver", spec.pre_layers, spec.global_layers, spec.stack_layers
    elif arch == "e18_local":
        par_mode, pre, glob, stack = "perceiver", spec.pre_layers, 0, spec.pre_layers + spec.global_layers + spec.stack_layers - spec.pre_layers
        # Keep total depth matched: pre + stack' = e18's pre+global+stack, no full layer.
        stack = spec.global_layers + spec.stack_layers
        glob = 0
        pre = spec.pre_layers
    else:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHES}")

    n_heads = _n_heads(spec.hidden, spec.head_dim)
    n_kv = spec.n_kv_heads
    if n_kv <= 0 or n_heads % n_kv != 0:
        n_kv = n_heads  # full MHA; do not silently drop to 1 KV head
    exclusive = arch in EXCLUSIVE_ARCHES
    write = "sw_perceiver" if arch in SWP_ARCHES else ("latent_memory" if arch in E31_ARCHES else "block_mean")
    pre_window = int(spec.ctx_pre_window) if arch == "e30_ctx" else spec.local_window
    lm_rounds = int(spec.lm_rounds) or (3 if arch == "e31_bixt" else 2)
    cfg = PerceiverARConfig(
        vocab_size=vocab_size,
        hidden_size=spec.hidden,
        intermediate_size=2 * spec.hidden,
        token_embedding_dim=int(spec.token_embedding_dim) or min(32, spec.hidden),
        par_mode=par_mode,
        pre_layers=pre,
        pre_window=pre_window,
        global_layers=glob,
        stack_layers=stack,
        block=spec.local_window,
        num_attention_heads=n_heads,
        num_kv_heads=n_kv,
        head_dim=spec.head_dim,
        ngram_orders=tuple(spec.ngram_orders),
        ngram_buckets=256,
        value_embed_layers=spec.value_embed_layers,
        value_embed_dim=16,
        use_liger=False,
        attn_backend=spec.attn_backend,
        attn_pad_multiple=1,
        chunked_ce_block_size=64,
        swa_sink=True,
        global_logit_scale=spec.global_logit_scale,
        z_loss=spec.z_loss,
        zero_init_residuals=spec.zero_init_residuals,
        message_boundary_token_id=(
            spec.message_boundary_token_id if exclusive else -1
        ),
        message_compress_ratio=spec.message_compress_ratio if arch == "e21" else 16,
        message_pool_remainder=bool(spec.message_pool_remainder) if arch == "e21" else False,
        message_slots_inplace=bool(spec.message_slots_inplace) if arch == "e21" else False,
        message_inplace_raw_kv=bool(spec.message_inplace_raw_kv) if arch == "e21" else False,
        message_identity_slots=bool(spec.message_identity_slots) if arch == "e21" else False,
        message_pack_stride=int(getattr(spec, "message_pack_stride", 0) or 0) if arch == "e21" else 0,
        message_keep_local_swa=bool(spec.message_keep_local_swa) if exclusive else False,
        message_extra_slot_attends=int(getattr(spec, "message_extra_slot_attends", 0) or 0) if exclusive else 0,
        message_update_slot_kv=bool(getattr(spec, "message_update_slot_kv", False)) if exclusive else False,
        message_global_anchors=(
            str(getattr(spec, "message_global_anchors", "none") or "none") if exclusive else "none"
        ),
        message_anchor_token_ids=(
            tuple(int(x) for x in (getattr(spec, "message_anchor_token_ids", ()) or ())) if exclusive else ()
        ),
        message_anchor_window=int(getattr(spec, "message_anchor_window", 4) or 0) if exclusive else 4,
        message_anchor_key_len=int(getattr(spec, "message_anchor_key_len", 0) or 0) if exclusive else 0,
        message_prefix_ae=bool(getattr(spec, "message_prefix_ae", False)) if arch == "e21" else False,
        message_prefix_ae_weight=float(getattr(spec, "message_prefix_ae_weight", 0.0) or 0.0) if arch == "e21" else 0.0,
        message_prefix_ae_stopgrad_answer=bool(getattr(spec, "message_prefix_ae_stopgrad_answer", True)) if arch == "e21" else True,
        message_write=write,
        swp_bank_size=int(getattr(spec, "swp_bank_size", 32) or 32),
        swp_coverage=int(getattr(spec, "swp_coverage", 8) or 8),
        swp_window=int(getattr(spec, "swp_window", 0) or 0),
        swp_stride=int(getattr(spec, "swp_stride", 0) or 0),
        swp_n_heads=int(getattr(spec, "swp_n_heads", 0) or 0),
        swp_query_dim=int(getattr(spec, "swp_query_dim", 0) or 0),
        swp_auto_fit=bool(getattr(spec, "swp_auto_fit", True)),
        lm_context="bixt" if arch == "e31_bixt" else "page_bidir",
        lm_window=int(spec.lm_window),
        lm_stride=int(spec.lm_stride),
        lm_latents=int(spec.lm_latents),
        lm_latent_dim=int(spec.lm_latent_dim),
        lm_heads=int(spec.lm_heads),
        lm_writer_dim=int(spec.lm_writer_dim),
        lm_enc_layers=int(spec.lm_enc_layers),
        lm_rounds=lm_rounds,
        lm_competition=bool(spec.lm_competition),
        lm_null_latent=bool(spec.lm_null_latent),
        lm_reader_tokens=int(spec.lm_reader_tokens),
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )
    return PerceiverARLM(cfg)


def arch_cache(arch: str, spec: ArchSpec, seq_len: int) -> dict:
    n_layers = spec.pre_layers + spec.global_layers + spec.stack_layers
    glob = 0 if arch == "e18_local" else spec.global_layers
    if arch == "e18_local":
        n_layers = spec.pre_layers + spec.global_layers + spec.stack_layers
        glob = 0
    compress_ratio = spec.message_compress_ratio if arch == "e21" else 1
    if arch in E31_ARCHES:
        from nn.latent_memory import lm_geometry

        geo = lm_geometry(spec, seq_len)
        compress_ratio = max(float(seq_len) / max(geo.n_slots, 1), 1e-6)
    if arch in SWP_ARCHES:
        geo = swp_geometry(
            SimpleNamespace(
                swp_bank_size=int(getattr(spec, "swp_bank_size", 32) or 32),
                swp_coverage=int(getattr(spec, "swp_coverage", 8) or 8),
                swp_window=int(getattr(spec, "swp_window", 0) or 0),
                swp_stride=int(getattr(spec, "swp_stride", 0) or 0),
                swp_auto_fit=bool(getattr(spec, "swp_auto_fit", True)),
                swp_n_heads=int(getattr(spec, "swp_n_heads", 0) or 0),
                swp_query_dim=int(getattr(spec, "swp_query_dim", 0) or 0),
                num_attention_heads=max(1, spec.hidden // max(spec.head_dim, 1)),
                head_dim=spec.head_dim,
                token_embedding_dim=int(spec.token_embedding_dim) or min(32, spec.hidden),
            ),
            seq_len,
        )
        compress_ratio = max(float(geo.compression), 1e-6)
    return cache_profile(
        arch,
        n_layers=n_layers,
        global_layers=glob,
        n_kv_heads=spec.n_kv_heads,
        head_dim=spec.head_dim,
        local_window=spec.local_window,
        seq_len=seq_len,
        enc_layers=spec.enc_layers,
        dec_layers=spec.dec_layers,
        compress_ratio=compress_ratio,
    )


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
