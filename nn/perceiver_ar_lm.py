"""Perceiver AR v2 — one global causal read, deep window-N stack, every token trained (E18).

Family `perceiver_ar`, selected via `model_family=perceiver_ar` in the shared training
entrypoint. Spec: docs/experiments_specs/ahead/E18_perceiver_ar_v2_baseline.md.

Training-time structure (all layers share one block; only the attention *pattern* differs):

    ids ─► tiny factorized embedding (e=256) + hashed 2/3-gram tables ─► MLP up-proj ─► x0
        ─► [swa(pre_window)] × pre_layers          # local pre-encoder: contextualize the history
        ─► [full causal]     × global_layers        # the ONE global read (the only unbounded KV cache)
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
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

logger = logging.getLogger(__name__)

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
        stack_layers: int = 20,
        block: int = 4096,                    # N — window of the stack layers
        # heads / positions
        num_attention_heads: Optional[int] = None,
        num_kv_heads: int = 2,
        head_dim: int = 128,
        rope_theta: float = 500000.0,
        nope_every: int = 4,                  # every k-th stack layer has no RoPE (SmolLM3); 0 = off
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
        write_back_hook: bool = False,        # E19 — adds write_back_proj params when True
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
        self.par_mode = par_mode
        self.pre_layers = pre_layers
        self.pre_window = pre_window
        self.global_layers = global_layers
        self.stack_layers = stack_layers
        self.block = block
        self.num_attention_heads = num_attention_heads or (hidden_size // head_dim)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.nope_every = nope_every
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
        self.write_back_hook = write_back_hook
        self.init_std = init_std
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
        if self.global_layers < 1 and self.par_mode == "perceiver":
            logger.warning("perceiver mode with global_layers=0: a purely local model (ablation only)")

    @property
    def total_layers(self) -> int:
        if self.par_mode == "dense":
            return self.pre_layers + self.global_layers + self.stack_layers
        return self.pre_layers + self.global_layers + self.stack_layers

    def layer_patterns(self) -> list[tuple[str, int]]:
        """Per-layer (pattern, window). Dense = all full causal with the same layer count."""
        if self.par_mode == "dense":
            return [("full", 0)] * self.total_layers
        return (
            [("swa", self.pre_window)] * self.pre_layers
            + [("full", 0)] * self.global_layers
            + [("swa", self.block)] * self.stack_layers
        )

    @property
    def global_layer_index(self) -> int:
        """Index of the (first) global read layer — the one-layer prefix cache / message space."""
        return self.pre_layers if self.par_mode == "perceiver" else 0


# --------------------------------------------------------------------------------------
# Masks (single source of truth for every backend)
# --------------------------------------------------------------------------------------


def make_mask_pred(
    pattern: str,
    window: int,
    key_valid: Optional[torch.Tensor],
    doc_ids: Optional[torch.Tensor],
    causal: bool = True,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return mask_mod(b, h, q, kv) -> bool with FlexAttention semantics.

    Rules: causal (kv <= q) unless bidirectional; sliding window (q - kv < window) for
    `swa`; padded keys masked except the diagonal (keeps every query row non-empty);
    cross-document attention masked when `doc_ids` is given.
    """

    def pred(b, h, q, kv):
        ok = (kv <= q) if causal else (kv >= 0)
        if pattern == "swa":
            dist = q - kv
            ok = ok & (dist < window) & (dist > -window)
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
) -> torch.Tensor:
    """[B,1,S,S] boolean mask (True = attend) — reference path used by `sdpa`."""
    q = torch.arange(S, device=device)[:, None]
    kv = torch.arange(S, device=device)[None, :]
    ok = (kv <= q) if causal else torch.ones(S, S, dtype=torch.bool, device=device)
    if pattern == "swa":
        dist = q - kv
        ok = ok & (dist < window) & (dist > -window)
    ok = ok[None, None].expand(batch, 1, S, S)
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
            _flex_attention_fn = torch.compile(flex_attention, dynamic=False)
        else:
            _flex_attention_fn = flex_attention
    return _flex_attention_fn


def _flex_block_mask(S, pattern, window, key_valid, doc_ids, device, causal, batch):
    from torch.nn.attention.flex_attention import create_block_mask

    batch_dependent = key_valid is not None or doc_ids is not None
    key = None
    if not batch_dependent:
        key = (S, pattern, window, causal, str(device))
        if key in _FLEX_CACHE:
            return _FLEX_CACHE[key]
    pred = make_mask_pred(pattern, window, key_valid, doc_ids, causal=causal)
    bm = create_block_mask(
        pred, B=batch if batch_dependent else None, H=None, Q_LEN=S, KV_LEN=S, device=device,
        _compile=bool(batch_dependent and torch.cuda.is_available()),
    )
    if key is not None:
        _FLEX_CACHE[key] = bm
    return bm


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
) -> torch.Tensor:
    """Attention with the E18 pattern semantics. Returns [B,S,h,dh]."""
    B, S, h, dh = q.shape
    g = k.shape[2]
    if backend == "flash":
        return _attend_flash(q, k, v, pattern, window, causal, cu_seqlens, key_valid)
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))  # [B,h/g,S,dh]
    if backend == "flex":
        bm = _flex_block_mask(S, pattern, window, key_valid, doc_ids, q.device, causal, B)
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
        mask = dense_bool_mask(S, pattern, window, key_valid, doc_ids, q.device, causal, B)
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
        self.wq = nn.Linear(d, h * dh, bias=False)
        self.wk = nn.Linear(d, g * dh, bias=False)
        self.wv = nn.Linear(d, g * dh, bias=False)
        self.wo = nn.Linear(h * dh, d, bias=False)
        self.q_norm = nn.RMSNorm(dh)
        self.k_norm = nn.RMSNorm(dh)
        self.use_rope = True
        self.value_embed = None
        if layer_idx in cfg.value_embed_layers:
            self.value_embed = nn.Embedding(cfg.vocab_size, cfg.value_embed_dim)
            self.value_proj = nn.Linear(cfg.value_embed_dim, g * dh, bias=False)
            self.value_lambda = nn.Parameter(torch.tensor(0.5))

    def kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape
        k = self.k_norm(self.wk(x).view(B, S, self.g, self.dh))
        v = self.wv(x).view(B, S, self.g, self.dh)
        return k, v

    def forward(self, x, *, ids, cos, sin, key_valid, doc_ids, cu_seqlens):
        B, S, _ = x.shape
        q = self.q_norm(self.wq(x).view(B, S, self.h, self.dh))
        k, v = self.kv(x)
        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.value_embed is not None:
            ve = self.value_proj(self.value_embed(ids)).view(B, S, self.g, self.dh)
            v = v + self.value_lambda * ve
        o = attend(
            q, k, v, pattern=self.pattern, window=self.window, key_valid=key_valid,
            doc_ids=doc_ids, backend=self.backend, causal=self.causal, cu_seqlens=cu_seqlens,
        )
        return self.wo(o.reshape(B, S, self.h * self.dh))


class SwiGLU(nn.Module):
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.gate = nn.Linear(d, ff, bias=False)
        self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: PerceiverARConfig, layer_idx: int, pattern: str, window: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.hidden_size)
        self.attn = Attention(cfg, layer_idx, pattern, window)
        self.mlp_norm = nn.RMSNorm(cfg.hidden_size)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)
        # x0 re-injection (α=1, β=0 at init → identity) and U-net skip weight (σ=0 at init).
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.sigma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, x0, skip, *, ids, cos, sin, key_valid, doc_ids, cu_seqlens):
        x = self.alpha * x + self.beta * x0
        if skip is not None:
            x = x + self.sigma * skip
        x = x + self.attn(self.attn_norm(x), ids=ids, cos=cos, sin=sin, key_valid=key_valid,
                          doc_ids=doc_ids, cu_seqlens=cu_seqlens)
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
        self.layers = nn.ModuleList(
            [Block(cfg, i, p, w) for i, (p, w) in enumerate(patterns)]
        )
        stack_start = cfg.pre_layers + cfg.global_layers if cfg.par_mode == "perceiver" else 0
        for i, layer in enumerate(self.layers):
            if cfg.nope_every and i >= stack_start and ((i - stack_start + 1) % cfg.nope_every == 0):
                layer.attn.use_rope = False
            if cfg.block_attention_mode == "bidirectional" and i >= stack_start:
                layer.attn.causal = False
        self.final_norm = nn.RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.write_back_hook:
            g, dh = cfg.num_kv_heads, cfg.head_dim
            self.write_back_proj = nn.Linear(cfg.hidden_size, 2 * g * dh, bias=False)
        self.gradient_checkpointing = False
        self._flce = None
        self.post_init()
        # Zero-init the residual-writing projections (muP-like, modded-nanogpt).
        for layer in self.layers:
            nn.init.zeros_(layer.attn.wo.weight)
            nn.init.zeros_(layer.mlp.down.weight)
        if cfg.write_back_hook:
            nn.init.zeros_(self.write_back_proj.weight)

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

    def _run_layers(self, input_ids, attention_mask, doc_ids, cu_seqlens, stop_after: Optional[int] = None):
        cfg = self.config
        B, S = input_ids.shape
        key_valid = None
        if self._needs_key_mask(attention_mask, cfg.block_attention_mode == "causal"):
            key_valid = attention_mask.bool()
        x0 = self.embed(input_ids, doc_ids)
        pos = self._positions(S, B, doc_ids, input_ids.device)
        cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_theta, x0.dtype)
        x = x0
        n = len(self.layers)
        n_skip = n // 2
        skips: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            skip = skips.pop() if (i >= n - n_skip and skips) else None
            kwargs = dict(ids=input_ids, cos=cos, sin=sin, key_valid=key_valid, doc_ids=doc_ids,
                          cu_seqlens=cu_seqlens)
            if self.gradient_checkpointing and self.training:
                # partial binds THIS layer/kwargs (a lambda would capture the loop variables
                # by reference and re-run the last layer in backward).
                fn = partial(_call_block, layer, **kwargs)
                x = torch_checkpoint(fn, x, x0, skip, use_reentrant=False)
            else:
                x = layer(x, x0, skip, **kwargs)
            if i < n_skip:
                skips.append(x)
            if stop_after is not None and i == stop_after:
                return x
        return x

    # -- forward ----------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        return_per_token_loss: bool = False,
        return_logits: bool = False,
        **unused,
    ):
        cfg = self.config
        input_ids, attention_mask, labels, doc_ids, S_orig = self._pad_inputs(
            input_ids, attention_mask, labels, doc_ids
        )
        x = self._run_layers(input_ids, attention_mask, doc_ids, cu_seqlens)
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

    # -- family hooks ---------------------------------------------------------------
    @torch.no_grad()
    def prefix_kv(self, input_ids, attention_mask=None, doc_ids=None):
        """K/V of the global read layer for `input_ids` — the one-layer prefix cache and the
        message object for E19/E21. Returns (k, v) each [B,S,g,dh] (RoPE applied to k)."""
        cfg = self.config
        if cfg.par_mode == "perceiver" and cfg.global_layers < 1:
            raise RuntimeError("prefix_kv needs a global read layer")
        gi = cfg.global_layer_index
        input_ids, attention_mask, _, doc_ids, S = self._pad_inputs(input_ids, attention_mask, None, doc_ids)
        x = self._run_layers(input_ids, attention_mask, doc_ids, None, stop_after=gi - 1) if gi > 0 else None
        if x is None:
            x = self.embed(input_ids, doc_ids)
        layer = self.layers[gi]
        h = layer.attn_norm(layer.alpha * x + layer.beta * self.embed(input_ids, doc_ids))
        k, v = layer.attn.kv(h)
        pos = self._positions(input_ids.shape[1], input_ids.shape[0], doc_ids, input_ids.device)
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


def analytic_param_count(cfg: PerceiverARConfig) -> ParamBreakdown:
    d, ff, e, V = cfg.hidden_size, cfg.intermediate_size, cfg.token_embedding_dim, cfg.vocab_size
    h, g, dh = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
    L = cfg.total_layers
    per_layer = (d * h * dh) + 2 * (d * g * dh) + (h * dh * d) + 3 * d * ff + 2 * d + 2 * dh + 3
    dense = L * per_layer
    dense += V * e + 2 * e * d + d * d + d          # tok table + up0/up1/up2 + norm
    dense += d + d * V                              # final norm + head
    if cfg.write_back_hook:
        dense += d * 2 * g * dh
    sparse = len(cfg.ngram_orders) * cfg.ngram_buckets * e
    n_ve = sum(1 for i in range(L) if i in cfg.value_embed_layers)
    sparse += n_ve * (V * cfg.value_embed_dim)
    dense += n_ve * (cfg.value_embed_dim * g * dh + 1)
    return ParamBreakdown(dense=dense, sparse_tables=sparse)
