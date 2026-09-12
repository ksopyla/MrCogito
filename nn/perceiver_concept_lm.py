"""Perceiver Concept LM — encoder → positional concept array → latent transformer → segment-confined
decoder, trained from scratch on every token (E22).

Family `perceiver_concept`, selected via `model_family=perceiver_concept` in the shared training
entrypoint. Spec: docs/experiments_specs/ahead/E22_perceiver_concept_lm.md.

    ids ─► TinyHashedEmbedding (e=256 + hashed 2/3-grams → d)                          x0 [B,S,d]
        ─► ENCODER   [swa(enc_window)] × enc_layers, causal, doc-masked                h  [B,S,d]
        ─► POOL      one block of `concept_ratio` tokens → `concept_slots` slots:
                     mean(block) + zero-init learned-query attention over the block       z0 [B,C,d]
        ─► LATENT    [full causal over the concept array] × latent_layers (× repeats)  z  [B,C,d]
        ─► DECODER   from x0: [ self-attn confined to the token's segment
                                → cross-attn to {z_j : pos(z_j) ≤ pos(t), same doc}
                                → SwiGLU ] × dec_layers                                  y  [B,S,d]
        ─► RMSNorm ─► chunked soft-capped lm_head + CE (+ z-loss)

Every arrow is causal, so one forward trains all positions. The decoder never sees a raw token
outside its own segment; everything earlier reaches it only through the concept array — the
bypass that killed E05 / E10–E17 / E18 is closed structurally (BAPO: `b` capped at the segment,
`a` = the array). Slots are allocated by position (one per block), so none can be starved
(the E05 free-latent collapse cannot happen in count), and they are pooled from
`enc_layers`-deep states rather than raw embeddings (the E18b defect).

Primitives (attention, masks, embedding, CE) are imported from `nn/perceiver_ar_lm.py`; that
module is not modified here.
"""
from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from nn.perceiver_ar_lm import (
    Attention,
    Block,
    SwiGLU,
    TinyHashedEmbedding,
    _get_flex,
    _liger_flce,
    apply_rope,
    attend,
    chunked_softcap_ce,
    per_token_ce_chunked,
    rope_cos_sin,
)

logger = logging.getLogger(__name__)

_FLEX_KV_MULTIPLE = 128   # concept array padded to this so flex block masks tile cleanly


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


class PerceiverConceptConfig(PretrainedConfig):
    model_type = "perceiver_concept"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        token_embedding_dim: int = 256,
        # encoder
        enc_layers: int = 6,
        enc_window: int = 512,
        # concept array
        concept_ratio: int = 16,
        concept_slots: int = 1,
        pool_pos_bias: bool = True,
        # latent transformer
        latent_layers: int = 4,
        latent_repeats: int = 1,
        # decoder
        dec_layers: int = 8,
        dec_segment: int = 1024,
        dec_local: str = "block",          # "block" (segment-reset) | "swa" (sliding window dec_segment)
        concept_mode: str = "full",        # "full" | "none" (arm C: decoder never reads the array)
        xattn_kv_heads: int = 2,
        # heads / positions
        num_attention_heads: Optional[int] = None,
        num_kv_heads: int = 2,
        head_dim: int = 128,
        rope_theta: float = 500000.0,
        # input
        ngram_orders: tuple[int, ...] = (2, 3),
        ngram_buckets: int = 65536,
        enc_value_embed_layers: tuple[int, ...] = (0, 3),
        dec_value_embed_layers: tuple[int, ...] = (0,),
        value_embed_dim: int = 64,
        # head / loss
        logit_softcap: float = 30.0,
        z_loss: float = 1e-4,
        chunked_ce_block_size: int = 2048,
        use_liger: bool = True,
        # backend
        attn_backend: str = "flex",
        attn_pad_multiple: int = 2048,
        init_std: float = 0.02,
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
        self.enc_layers = int(enc_layers)
        self.enc_window = int(enc_window)
        self.concept_ratio = int(concept_ratio)
        self.concept_slots = int(concept_slots)
        self.pool_pos_bias = bool(pool_pos_bias)
        self.latent_layers = int(latent_layers)
        self.latent_repeats = int(latent_repeats)
        self.dec_layers = int(dec_layers)
        self.dec_segment = int(dec_segment)
        self.dec_local = dec_local
        self.concept_mode = concept_mode
        self.xattn_kv_heads = int(xattn_kv_heads)
        self.num_attention_heads = num_attention_heads or (hidden_size // head_dim)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.ngram_orders = tuple(int(o) for o in ngram_orders)
        self.ngram_buckets = ngram_buckets
        self.enc_value_embed_layers = tuple(int(l) for l in enc_value_embed_layers)
        self.dec_value_embed_layers = tuple(int(l) for l in dec_value_embed_layers)
        self.value_embed_dim = value_embed_dim
        self.logit_softcap = logit_softcap
        self.z_loss = z_loss
        self.chunked_ce_block_size = chunked_ce_block_size
        self.use_liger = use_liger
        self.attn_backend = attn_backend
        self.attn_pad_multiple = attn_pad_multiple
        self.init_std = init_std
        # Fields the shared `Attention` / `Block` primitives read (E18 knobs held at their off value).
        self.swa_sink = False
        self.global_logit_scale = "none"
        self.global_scale_ref = 8192
        self.value_embed_layers = ()      # per-stack views override this (see _CfgView)
        # Bookkeeping consumed by the shared entrypoint / W&B init / eval routing.
        self.checkpoint_family = "perceiver_concept"
        self.pretraining_objective = "causal_lm"
        self.concept_num = 0                # length-proportional; not a fixed latent count
        self.hidden_act = "silu"
        self.norm_type = "rmsnorm"
        self.max_sequence_length = kwargs.get("max_sequence_length", None)
        self._validate()
        self.num_hidden_layers = self.total_layers

    def _validate(self):
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
        if self.num_attention_heads % self.xattn_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by xattn_kv_heads")
        if self.attn_backend not in {"sdpa", "flex"}:
            raise ValueError(f"perceiver_concept supports attn_backend sdpa|flex, got {self.attn_backend!r}")
        if self.dec_local not in {"block", "swa"}:
            raise ValueError("dec_local must be 'block' or 'swa'")
        if self.concept_mode not in {"full", "none"}:
            raise ValueError("concept_mode must be 'full' or 'none'")
        if self.concept_ratio < 1 or self.concept_slots < 1:
            raise ValueError("concept_ratio and concept_slots must be >= 1")
        if self.latent_repeats < 1:
            raise ValueError("latent_repeats must be >= 1")
        if self.dec_segment < 1:
            raise ValueError("dec_segment must be >= 1")

    @property
    def total_layers(self) -> int:
        return self.enc_layers + self.latent_layers + self.dec_layers

    def n_concepts(self, S: int) -> int:
        return self.concept_slots * (-(-S // self.concept_ratio))


class _CfgView:
    """Attribute view over the config with a few fields overridden — lets the shared
    `Attention`/`Block` primitives see the value-embedding list of the stack they belong to."""

    def __init__(self, base, **over):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_over", over)

    def __getattr__(self, name):
        over = object.__getattribute__(self, "_over")
        if name in over:
            return over[name]
        return getattr(object.__getattribute__(self, "_base"), name)


# --------------------------------------------------------------------------------------
# Concept pooling: one block of r tokens → c slots
# --------------------------------------------------------------------------------------


class ConceptPooler(nn.Module):
    """[B,S,d] encoder states → [B,C,d] slots, C = c·⌈S/r⌉.

    slot_{j,i} = mean_ok(h[block j]) + W_o( softmax_t( q_i·k_t/√dh + bias[t,head] ; mask ok ) · v_t )

    `q_i` are `c` learned queries (h heads), k/v come from the block's states (g kv-heads,
    QK-norm), `bias` is a learnable within-block positional bias. `W_o` is zero-initialised so
    step 0 is exact mean pooling (LCLM's best pooler); the attention term learns what mean
    pooling loses. Tokens with `ok=False` (padding, or an earlier document sharing the block)
    are excluded from both terms; an all-masked block yields a zero, invalid slot.
    """

    def __init__(self, cfg: PerceiverConceptConfig):
        super().__init__()
        d, h, g, dh = cfg.hidden_size, cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
        self.r, self.c, self.h, self.g, self.dh = cfg.concept_ratio, cfg.concept_slots, h, g, dh
        self.norm = nn.RMSNorm(d)
        self.q = nn.Parameter(torch.randn(self.c, h, dh) * cfg.init_std)
        self.wk = nn.Linear(d, g * dh, bias=False)
        self.wv = nn.Linear(d, g * dh, bias=False)
        self.wo = nn.Linear(h * dh, d, bias=False)
        self.q_norm = nn.RMSNorm(dh)
        self.k_norm = nn.RMSNorm(dh)
        self.pos_bias = nn.Parameter(torch.zeros(self.r, h)) if cfg.pool_pos_bias else None

    def forward(self, h: torch.Tensor, ok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h [B,S,d], ok [B,S] bool → (slots [B,C,d], slot_valid [B,C]). S must be a multiple of r."""
        B, S, d = h.shape
        r, c, H, g, dh = self.r, self.c, self.h, self.g, self.dh
        nb = S // r
        hb = h.view(B, nb, r, d)
        okb = ok.view(B, nb, r)
        okf = okb.to(h.dtype)[..., None]
        cnt = okf.sum(dim=2).clamp(min=1.0)                                 # [B,nb,1]
        mean = (hb * okf).sum(dim=2) / cnt                                  # [B,nb,d]
        hn = self.norm(hb)
        k = self.k_norm(self.wk(hn).view(B, nb, r, g, dh))
        v = self.wv(hn).view(B, nb, r, g, dh)
        rep = H // g
        k = k.repeat_interleave(rep, dim=3)                                 # [B,nb,r,H,dh]
        v = v.repeat_interleave(rep, dim=3)
        q = self.q_norm(self.q)                                             # [c,H,dh]
        logits = torch.einsum("chd,bnrhd->bnchr", q, k) / math.sqrt(dh)    # [B,nb,c,H,r]
        if self.pos_bias is not None:
            logits = logits + self.pos_bias.t()[None, None, None]          # [H,r]
        logits = logits.masked_fill(~okb[:, :, None, None, :], float("-inf"))
        w = torch.softmax(logits.float(), dim=-1)
        w = torch.nan_to_num(w, nan=0.0).to(v.dtype)                        # all-masked block → 0
        o = torch.einsum("bnchr,bnrhd->bnchd", w, v).reshape(B, nb, c, H * dh)
        slots = mean[:, :, None, :] + self.wo(o)                            # [B,nb,c,d]
        valid = okb.any(dim=2)[:, :, None].expand(B, nb, c)
        slots = slots * valid[..., None].to(slots.dtype)
        return slots.reshape(B, nb * c, d), valid.reshape(B, nb * c)


# --------------------------------------------------------------------------------------
# Cross-attention: token queries → concept K/V
# --------------------------------------------------------------------------------------


def _cross_mask_pred(cpos, cdoc, cvalid, pos, doc):
    """Slot 0 is the learned null slot, visible to every query (no query row is ever empty —
    a sink the decoder can attend to when nothing earlier is relevant, and a NaN guard for
    both backends). Every other slot is visible iff it ended at or before the query's position
    inside the same document."""

    def pred(b, hh, q, kv):
        content = (cpos[b, kv] <= pos[b, q]) & (cdoc[b, kv] == doc[b, q]) & cvalid[b, kv]
        return (kv == 0) | content

    return pred


def attend_cross(
    q: torch.Tensor,        # [B,S,h,dh]
    k: torch.Tensor,        # [B,C,g,dh]   slot 0 = null slot
    v: torch.Tensor,        # [B,C,g,dh]
    *,
    cpos: torch.Tensor,     # [B,C]  position of each slot (block end, doc-relative)
    cdoc: torch.Tensor,     # [B,C]  document id of each slot
    cvalid: torch.Tensor,   # [B,C]  slot exists
    pos: torch.Tensor,      # [B,S]
    doc: torch.Tensor,      # [B,S]
    backend: str,
    memo: Optional[dict] = None,
) -> torch.Tensor:
    """Token t attends to the null slot plus slots with pos(slot) ≤ pos(t) in its document.
    Returns [B,S,h,dh]."""
    B, S, h, dh = q.shape
    C, g = k.shape[1], k.shape[2]
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    if backend == "flex":
        from torch.nn.attention.flex_attention import create_block_mask

        key = ("xattn", S, C)
        if memo is not None and key in memo:
            bm = memo[key]
        else:
            bm = create_block_mask(
                _cross_mask_pred(cpos, cdoc, cvalid, pos, doc), B=B, H=None, Q_LEN=S, KV_LEN=C,
                device=q.device, _compile=torch.cuda.is_available(),
            )
            if memo is not None:
                memo[key] = bm
        out = _get_flex()(qt.contiguous(), kt.contiguous(), vt.contiguous(), block_mask=bm, enable_gqa=(g != h))
        return out.transpose(1, 2)
    if g != h:
        kt = kt.repeat_interleave(h // g, dim=1)
        vt = vt.repeat_interleave(h // g, dim=1)
    mask = (cpos[:, None, :] <= pos[:, :, None]) & (cdoc[:, None, :] == doc[:, :, None]) & cvalid[:, None, :]
    mask[:, :, 0] = True
    out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask[:, None])   # [B,1,S,C]
    return out.transpose(1, 2)


class ConceptCrossAttention(nn.Module):
    def __init__(self, cfg: PerceiverConceptConfig):
        super().__init__()
        d, h, dh = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim
        g = cfg.xattn_kv_heads
        self.h, self.g, self.dh = h, g, dh
        self.backend = cfg.attn_backend
        self.wq = nn.Linear(d, h * dh, bias=False)
        self.wk = nn.Linear(d, g * dh, bias=False)
        self.wv = nn.Linear(d, g * dh, bias=False)
        self.wo = nn.Linear(h * dh, d, bias=False)
        self.q_norm = nn.RMSNorm(dh)
        self.k_norm = nn.RMSNorm(dh)

    def kv(self, z: torch.Tensor, ccos, csin):
        B, C, _ = z.shape
        k = self.k_norm(self.wk(z).view(B, C, self.g, self.dh))
        v = self.wv(z).view(B, C, self.g, self.dh)
        return apply_rope(k, ccos, csin), v

    def forward(self, x, kv, *, cos, sin, cpos, cdoc, cvalid, pos, doc, memo):
        B, S, _ = x.shape
        q = apply_rope(self.q_norm(self.wq(x).view(B, S, self.h, self.dh)), cos, sin)
        k, v = kv
        o = attend_cross(q, k, v, cpos=cpos, cdoc=cdoc, cvalid=cvalid, pos=pos, doc=doc,
                         backend=self.backend, memo=memo)
        return self.wo(o.reshape(B, S, self.h * self.dh))


# --------------------------------------------------------------------------------------
# Decoder block: segment-confined self-attn → concept cross-attn → SwiGLU
# --------------------------------------------------------------------------------------


class DecoderBlock(nn.Module):
    def __init__(self, cfg: PerceiverConceptConfig, layer_idx: int):
        super().__init__()
        view = _CfgView(cfg, value_embed_layers=cfg.dec_value_embed_layers)
        self.attn_norm = nn.RMSNorm(cfg.hidden_size)
        self.attn = Attention(view, layer_idx, "swa", cfg.dec_segment)
        self.has_xattn = cfg.concept_mode == "full"
        if self.has_xattn:
            self.xattn_norm = nn.RMSNorm(cfg.hidden_size)
            self.xattn = ConceptCrossAttention(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.hidden_size)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, x0, kv, *, ids, cos, sin, key_valid, seg_ids, block_masks, xmemo,
                cpos, cdoc, cvalid, pos, doc, concepts_on: bool):
        x = self.alpha * x + self.beta * x0
        x = x + self.attn(self.attn_norm(x), ids=ids, cos=cos, sin=sin, key_valid=key_valid,
                          doc_ids=seg_ids, cu_seqlens=None, block_masks=block_masks, sink_pos=None, pos=pos)
        if self.has_xattn and concepts_on:
            x = x + self.xattn(self.xattn_norm(x), kv, cos=cos, sin=sin, cpos=cpos, cdoc=cdoc,
                               cvalid=cvalid, pos=pos, doc=doc, memo=xmemo)
        x = x + self.mlp(self.mlp_norm(x))
        return x


def _call(layer, *args, **kwargs):
    return layer(*args, **kwargs)


def _pad_slots(z, cvalid, cpos, cdoc, n: int):
    """Append `n` invalid slots (never visible, matched by no document)."""
    return (
        F.pad(z, (0, 0, 0, n)),
        F.pad(cvalid, (0, n), value=False),
        F.pad(cpos, (0, n), value=0),
        F.pad(cdoc, (0, n), value=-2),
    )


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class PerceiverConceptLM(PreTrainedModel):
    config_class = PerceiverConceptConfig
    base_model_prefix = "perceiver_concept"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block", "DecoderBlock"]

    def __init__(self, config: PerceiverConceptConfig):
        super().__init__(config)
        cfg = config
        self.embed = TinyHashedEmbedding(cfg)
        enc_view = _CfgView(cfg, value_embed_layers=cfg.enc_value_embed_layers)
        lat_view = _CfgView(cfg, value_embed_layers=())
        self.enc_layers = nn.ModuleList(
            [Block(enc_view, i, "swa", cfg.enc_window, has_skip=False) for i in range(cfg.enc_layers)]
        )
        self.pooler = ConceptPooler(cfg)
        self.latent_layers = nn.ModuleList(
            [Block(lat_view, 1000 + i, "full", 0, has_skip=False) for i in range(cfg.latent_layers)]
        )
        self.concept_norm = nn.RMSNorm(cfg.hidden_size)
        # Learned null slot, prepended to the array the decoder reads (see _cross_mask_pred).
        self.null_slot = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.dec_layers = nn.ModuleList([DecoderBlock(cfg, i) for i in range(cfg.dec_layers)])
        self.final_norm = nn.RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self._flce = None
        self._concept_override: Optional[str] = None
        self.post_init()
        # Zero-init the residual-writing projections (muP-like, modded-nanogpt).
        for layer in list(self.enc_layers) + list(self.latent_layers):
            nn.init.zeros_(layer.attn.wo.weight)
            nn.init.zeros_(layer.mlp.down.weight)
        for layer in self.dec_layers:
            nn.init.zeros_(layer.attn.wo.weight)
            nn.init.zeros_(layer.mlp.down.weight)
            if layer.has_xattn:
                nn.init.zeros_(layer.xattn.wo.weight)
        nn.init.zeros_(self.pooler.wo.weight)

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

    # -- probes ---------------------------------------------------------------------
    @contextmanager
    def concept_override(self, mode: Optional[str]):
        """Eval-time concept ablation (E22 S1 instrument). `none`: the decoder's cross-attention
        output is dropped (the array is invisible; segment-local raw context only). `shuffled`:
        every row reads the concept array of the *next* row in the batch (content check — a
        generic prior would survive, real content would not). `real` / None: no-op."""
        if mode in (None, "real"):
            yield
            return
        if mode not in {"none", "shuffled"}:
            raise ValueError("concept_override must be one of real|none|shuffled")
        prev = self._concept_override
        self._concept_override = mode
        try:
            yield
        finally:
            self._concept_override = prev

    @contextmanager
    def reach_override(self, window):
        """Compatibility with the perceiver_ar probe runner: this family has no unbounded read to
        restrict. Only `None` is accepted."""
        if window is not None:
            raise NotImplementedError("perceiver_concept has no reach_override; use concept_override")
        yield []

    # -- helpers ----------------------------------------------------------------------
    @staticmethod
    def _positions(S: int, B: int, doc: torch.Tensor) -> torch.Tensor:
        starts = torch.ones_like(doc, dtype=torch.bool)
        starts[:, 1:] = doc[:, 1:] != doc[:, :-1]
        idx = torch.arange(S, device=doc.device)[None].expand(B, S)
        start_idx = torch.where(starts, idx, torch.zeros_like(idx))
        start_idx = torch.cummax(start_idx, dim=1).values
        return idx - start_idx

    def _pad_inputs(self, input_ids, attention_mask, labels, doc_ids):
        # Pad so that S is a multiple of both the kernel pad multiple and the concept block size.
        m = math.lcm(max(int(self.config.attn_pad_multiple), 1), self.config.concept_ratio)
        S = input_ids.shape[1]
        if S % m == 0:
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

    def _run_layers(self, input_ids, attention_mask, doc_ids, return_concepts: bool = False):
        cfg = self.config
        B, S = input_ids.shape
        dev = input_ids.device
        r = cfg.concept_ratio
        # Effective document ids: given doc_ids, or one document per row; padding → -1.
        doc = doc_ids.clone() if doc_ids is not None else torch.zeros(B, S, dtype=torch.long, device=dev)
        if attention_mask is not None:
            doc = doc.masked_fill(attention_mask == 0, -1)
        valid = doc >= 0                                                   # [B,S]
        key_valid = None if bool(valid.all()) else valid
        pos = self._positions(S, B, doc)
        cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_theta, torch.float32)
        x0 = self.embed(input_ids, doc)
        cos, sin = cos.to(x0.dtype), sin.to(x0.dtype)
        block_masks: Optional[dict] = {} if cfg.attn_backend == "flex" else None
        ckpt = self.gradient_checkpointing and self.training

        # ---- encoder -------------------------------------------------------------
        # Unpacked, unpadded rows need no document mask: keep the token masks batch-independent
        # (cacheable) in that case, exactly as the E18 stack does.
        doc_attn = doc if (doc_ids is not None or key_valid is not None) else None
        h = x0
        enc_kwargs = dict(ids=input_ids, cos=cos, sin=sin, key_valid=key_valid, doc_ids=doc_attn,
                          cu_seqlens=None, block_masks=block_masks, sink_pos=None, pos=pos)
        for layer in self.enc_layers:
            if ckpt:
                h = torch_checkpoint(partial(_call, layer, **enc_kwargs), h, x0, None, use_reentrant=False)
            else:
                h = layer(h, x0, None, **enc_kwargs)

        # ---- pool ----------------------------------------------------------------
        nb = S // r
        last = (torch.arange(nb, device=dev) + 1) * r - 1                  # block-end token index
        cdoc_b = doc[:, last]                                              # [B,nb]
        cpos_b = pos[:, last]
        ok = valid & (doc == cdoc_b.repeat_interleave(r, dim=1))          # exclude earlier docs inside the block
        slots, cvalid = self.pooler(h, ok)                                 # [B,C,d], [B,C]
        c = cfg.concept_slots
        cdoc = cdoc_b.repeat_interleave(c, dim=1)
        cpos = cpos_b.repeat_interleave(c, dim=1)
        cdoc = cdoc.masked_fill(~cvalid, -2)                               # invalid slots match no token
        # The latent stack works on the array padded to the flex tile so its block mask tiles.
        C = slots.shape[1]
        if cfg.attn_backend == "flex" and C % _FLEX_KV_MULTIPLE:
            slots, cvalid, cpos, cdoc = _pad_slots(slots, cvalid, cpos, cdoc, _FLEX_KV_MULTIPLE - C % _FLEX_KV_MULTIPLE)

        # ---- latent transformer ----------------------------------------------------
        ccos, csin = rope_cos_sin(cpos, cfg.head_dim, cfg.rope_theta, torch.float32)
        ccos, csin = ccos.to(x0.dtype), csin.to(x0.dtype)
        z = slots
        lat_masks: Optional[dict] = {} if cfg.attn_backend == "flex" else None
        lat_kwargs = dict(ids=None, cos=ccos, sin=csin, key_valid=(None if bool(cvalid.all()) else cvalid),
                          doc_ids=cdoc, cu_seqlens=None, block_masks=lat_masks, sink_pos=None, pos=cpos)
        for _ in range(cfg.latent_repeats):
            for layer in self.latent_layers:
                if ckpt:
                    z = torch_checkpoint(partial(_call, layer, **lat_kwargs), z, slots, None, use_reentrant=False)
                else:
                    z = layer(z, slots, None, **lat_kwargs)
        z = self.concept_norm(z)
        if return_concepts:
            return z, cvalid, cpos, cdoc

        # ---- decoder ---------------------------------------------------------------
        mode = self._concept_override
        concepts_on = cfg.concept_mode == "full" and mode != "none"
        if concepts_on and mode == "shuffled":
            z, cvalid, cpos, cdoc = (t.roll(1, dims=0) for t in (z, cvalid, cpos, cdoc))
        # Prepend the null slot (index 0, always visible), then re-tile for flex.
        z = torch.cat([self.null_slot.to(z.dtype)[None, None].expand(B, 1, -1), z], dim=1)
        cvalid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), cvalid], dim=1)
        cpos = torch.cat([torch.zeros(B, 1, dtype=cpos.dtype, device=dev), cpos], dim=1)
        cdoc = torch.cat([torch.full((B, 1), -3, dtype=cdoc.dtype, device=dev), cdoc], dim=1)
        C = z.shape[1]
        if cfg.attn_backend == "flex" and C % _FLEX_KV_MULTIPLE:
            z, cvalid, cpos, cdoc = _pad_slots(z, cvalid, cpos, cdoc, _FLEX_KV_MULTIPLE - C % _FLEX_KV_MULTIPLE)
        ccos, csin = rope_cos_sin(cpos, cfg.head_dim, cfg.rope_theta, torch.float32)
        ccos, csin = ccos.to(x0.dtype), csin.to(x0.dtype)
        if cfg.dec_local == "block":
            seg = torch.arange(S, device=dev)[None] // cfg.dec_segment
            seg_ids = doc * (S // cfg.dec_segment + 2) + seg
        else:
            seg_ids = doc
        xmemo: Optional[dict] = {} if cfg.attn_backend == "flex" else None
        y = x0
        dec_kwargs = dict(ids=input_ids, cos=cos, sin=sin, key_valid=key_valid, seg_ids=seg_ids,
                          block_masks=({} if cfg.attn_backend == "flex" else None), xmemo=xmemo,
                          cpos=cpos, cdoc=cdoc, cvalid=cvalid, pos=pos, doc=doc, concepts_on=concepts_on)
        for layer in self.dec_layers:
            kv = layer.xattn.kv(z, ccos, csin) if (layer.has_xattn and concepts_on) else None
            if ckpt:
                y = torch_checkpoint(partial(_call, layer, **dec_kwargs), y, x0, kv, use_reentrant=False)
            else:
                y = layer(y, x0, kv, **dec_kwargs)
        return y

    # -- forward ----------------------------------------------------------------------
    @torch.no_grad()
    def concepts(self, input_ids, attention_mask=None, doc_ids=None):
        """The concept array for `input_ids` — the compressed state / message object.
        Returns (z [B,C,d], valid [B,C], pos [B,C], doc [B,C])."""
        input_ids, attention_mask, _, doc_ids, _ = self._pad_inputs(input_ids, attention_mask, None, doc_ids)
        return self._run_layers(input_ids, attention_mask, doc_ids, return_concepts=True)

    def hidden_states(self, input_ids, attention_mask=None, doc_ids=None):
        """Final-norm hidden states [B,S,d] without materialising logits (probe contract)."""
        input_ids, attention_mask, _, doc_ids, S_orig = self._pad_inputs(input_ids, attention_mask, None, doc_ids)
        y = self._run_layers(input_ids, attention_mask, doc_ids)
        return self.final_norm(y)[:, :S_orig]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        return_per_token_loss: bool = False,
        return_logits: bool = False,
    ):
        # No **kwargs on purpose (HF Trainer loss-scaling contract; see PerceiverARLM.forward).
        cfg = self.config
        input_ids, attention_mask, labels, doc_ids, S_orig = self._pad_inputs(
            input_ids, attention_mask, labels, doc_ids
        )
        y = self._run_layers(input_ids, attention_mask, doc_ids)
        h = self.final_norm(y)

        if labels is None or return_logits:
            logits = F.linear(h[:, :S_orig], self.lm_head.weight).float()
            if cfg.logit_softcap:
                logits = cfg.logit_softcap * torch.tanh(logits / cfg.logit_softcap)
            if labels is None:
                return CausalLMOutput(loss=None, logits=logits)

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
        return CausalLMOutput(loss=loss, logits=(logits if return_logits else None))

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
# Parameter accounting
# --------------------------------------------------------------------------------------


@dataclass
class ParamBreakdown:
    compute: int          # transformer stacks + pooler + embedding MLP (what does the work)
    dense_tables: int     # token table + lm_head (dense, vocabulary-sized)
    sparse_tables: int    # hashed n-gram + value-embedding tables

    @property
    def dense(self) -> int:
        return self.compute + self.dense_tables

    @property
    def total(self) -> int:
        return self.compute + self.dense_tables + self.sparse_tables


def analytic_param_count(cfg: PerceiverConceptConfig) -> ParamBreakdown:
    d, ff, e, V = cfg.hidden_size, cfg.intermediate_size, cfg.token_embedding_dim, cfg.vocab_size
    h, g, dh = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
    gx = cfg.xattn_kv_heads
    attn = (d * h * dh) + 2 * (d * g * dh) + (h * dh * d) + 2 * dh        # wq wk wv wo + q/k norms
    mlp = 3 * d * ff
    block = attn + mlp + 2 * d + 2                                        # two norms + alpha, beta
    compute = (cfg.enc_layers + cfg.latent_layers) * block
    xattn = (d * h * dh) + 2 * (d * gx * dh) + (h * dh * d) + 2 * dh + d  # + its norm
    dec_block = block + (xattn if cfg.concept_mode == "full" else 0)
    compute += cfg.dec_layers * dec_block
    # pooler: norm + q + wk wv wo + q/k norms + pos bias
    compute += d + cfg.concept_slots * h * dh + 2 * (d * g * dh) + (h * dh * d) + 2 * dh
    compute += (cfg.concept_ratio * h if cfg.pool_pos_bias else 0)
    compute += 2 * e * d + d * d + d                                      # embed MLP + norm
    compute += 3 * d                                                      # concept_norm + final_norm + null slot
    n_ve = len(cfg.enc_value_embed_layers) + len(cfg.dec_value_embed_layers)
    compute += n_ve * (cfg.value_embed_dim * g * dh + 1)                  # value_proj + lambda
    dense_tables = V * e + d * V
    sparse = len(cfg.ngram_orders) * cfg.ngram_buckets * e + n_ve * V * cfg.value_embed_dim
    return ParamBreakdown(compute=compute, dense_tables=dense_tables, sparse_tables=sparse)
