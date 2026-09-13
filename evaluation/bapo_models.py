"""Matched-parameter architecture factory for the BAPO capability probe.

Architectures
-------------
- `dense`      Perceiver AR, `par_mode=dense`: full-causal decoder-only, same width/depth as E18.
- `e18`        Perceiver AR, one global read + SWA stack (the E18 architecture).
- `e18_local`  E18 with `global_layers=0`: local window only. Retrieval rungs must sit at the floor.
- `encdec`     Symmetric encoder-decoder: bidirectional prefix encoder, suffix-only decoder with
               cross-attention. Prefix information cannot take a raw route into the suffix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from evaluation.bapo_metrics import cache_profile
from nn.encdec_lm import EncDecConfig, EncoderDecoderLM
from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM


ARCHES = ("dense", "e18", "e18_local", "encdec")


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
    elif arch == "e18_local":
        par_mode, pre, glob, stack = "perceiver", spec.pre_layers, 0, spec.pre_layers + spec.global_layers + spec.stack_layers - spec.pre_layers
        # Keep total depth matched: pre + stack' = e18's pre+global+stack, no full layer.
        stack = spec.global_layers + spec.stack_layers
        glob = 0
        pre = spec.pre_layers
    else:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHES}")

    n_heads = _n_heads(spec.hidden, spec.head_dim)
    cfg = PerceiverARConfig(
        vocab_size=vocab_size,
        hidden_size=spec.hidden,
        intermediate_size=2 * spec.hidden,
        token_embedding_dim=min(32, spec.hidden),
        par_mode=par_mode,
        pre_layers=pre,
        pre_window=spec.local_window,
        global_layers=glob,
        stack_layers=stack,
        block=spec.local_window,
        num_attention_heads=n_heads,
        num_kv_heads=spec.n_kv_heads if n_heads % spec.n_kv_heads == 0 else 1,
        head_dim=spec.head_dim,
        ngram_orders=(2,),
        ngram_buckets=256,
        value_embed_layers=spec.value_embed_layers,
        value_embed_dim=16,
        use_liger=False,
        attn_backend="sdpa",
        attn_pad_multiple=1,
        chunked_ce_block_size=64,
        swa_sink=True,
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
    )


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
