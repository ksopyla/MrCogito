"""Sliding-window latent memory writer (E31).

Each overlapping window of the *book* (the tokens before QUERY, or before the window
closes in text) is written by `K` addressed latent vectors that read the window **in
context**. The latents are projected into the global read's key/value space and appended
to it, like E21/E30 slots, but a slot here is a latent with its own state (residual + FFN),
not a weighted average of token K/V.

Spec: docs/experiments_specs/ahead/E31_sliding_window_latent_memory.md
Plan: docs/experiments_specs/ahead/E31_sliding_window_latent_memory_plan.md

Shapes (claim scale): tok_emb [B,S,e] → windows [B·n_w, W, d_w] → latents [B·n_w, K(+1), D]
→ reader slots k̄, v̄ [B, n_w·K·m, g, dh] (normed + RoPE'd at the latent's expected
position), slot_doc / slot_side / slot_pos [B, n_w·K·m].

Two ways of giving tokens context before (or while) the latents read them:
  * `page_bidir` — `lm_enc_layers` bidirectional attention layers inside each window,
    then `lm_rounds` latent cross-attention rounds;
  * `bixt`       — no encoder; `lm_rounds` rounds of latents ⇄ tokens (BiXT), the tokens
    reading the latents back after every latent update.

`lm_competition` makes latents compete for tokens (softmax over latents per token and
head, then renormalise over tokens, Slot-Attention style). `lm_null_latent` adds one
latent that the reader never sees. Heads are never averaged: each head writes its own
slice of the latent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LM_CONTEXTS = ("page_bidir", "bixt")
_ADDR_FEATS = 64


def _rope():
    # lazy: nn.perceiver_ar_lm imports this module
    from nn.perceiver_ar_lm import apply_rope, rope_cos_sin

    return rope_cos_sin, apply_rope


def window_starts(seq_len: int, window: int, stride: int) -> tuple[int, ...]:
    from nn.perceiver_ar_lm import _swp_window_starts

    return _swp_window_starts(seq_len, window, stride)


@dataclass(frozen=True)
class LMGeometry:
    window: int
    stride: int
    latents: int
    reader_tokens: int
    starts: tuple[int, ...]

    @property
    def n_windows(self) -> int:
        return len(self.starts)

    @property
    def n_slots(self) -> int:
        """Reader entries (latents × reader tokens × windows)."""
        return self.n_windows * self.latents * self.reader_tokens

    @property
    def tokens_per_latent(self) -> float:
        s = self.starts[-1] + self.window if self.starts else 0
        return s / max(self.n_windows * self.latents, 1)


def lm_geometry(cfg, seq_len: int) -> LMGeometry:
    """Fixed geometry (no auto-shrink). A row shorter than the window is one window."""
    S = max(int(seq_len), 1)
    W = min(int(cfg.lm_window), S)
    st = max(1, min(int(cfg.lm_stride), W))
    return LMGeometry(
        window=W,
        stride=st,
        latents=int(cfg.lm_latents),
        reader_tokens=int(cfg.lm_reader_tokens),
        starts=window_starts(S, W, st),
    )


def window_pick(
    side: torch.Tensor,
    doc: torch.Tensor,
    key_valid: Optional[torch.Tensor],
    starts: tuple[int, ...],
    window: int,
):
    """Per-window pooling rule shared by E30 (`sw_perceiver`) and E31.

    A window writes only from its **own earliest side** and from **one document** (the
    document of its last token on that side). Computed per window, so a token that is
    poolable in one window never leaks into another (the E30 `pool_valid` leak).

    Returns tok_idx [n_w, W] (clamped), pick [B, n_w, W], slot_doc_w [B, n_w] (−1 = empty),
    slot_side_w [B, n_w], has [B, n_w].
    """
    B, S = side.shape
    device = side.device
    st = torch.tensor(starts, device=device, dtype=torch.long)
    tok = st[:, None] + torch.arange(window, device=device)
    in_range = tok < S
    tok_c = tok.clamp(max=max(S - 1, 0))
    side_w = side[:, tok_c]
    doc_w = doc[:, tok_c]
    ok = in_range[None] & (doc_w >= 0)
    if key_valid is not None:
        ok = ok & key_valid.bool()[:, tok_c]
    big = torch.full((), 10**6, device=device, dtype=side.dtype)
    min_side = torch.where(ok, side_w, big).min(dim=-1).values
    has = ok.any(dim=-1)
    min_side = torch.where(has, min_side, torch.zeros_like(min_side))
    on_side = ok & (side_w == min_side[..., None])
    idx = torch.arange(window, device=device)
    last = (on_side.to(torch.long) * (idx + 1)).amax(dim=-1).clamp(min=1) - 1
    pick_doc = doc_w.gather(-1, last[..., None]).squeeze(-1)
    pick = on_side & (doc_w == pick_doc[..., None])
    slot_doc_w = torch.where(has, pick_doc, torch.full_like(pick_doc, -1))
    slot_side_w = torch.where(has, min_side, torch.zeros_like(min_side))
    return tok_c, pick, slot_doc_w, slot_side_w, has


def _sinusoid(pos: torch.Tensor, dim: int = _ADDR_FEATS) -> torch.Tensor:
    """pos [...] → [..., dim] fixed sinusoid features (window-start address)."""
    half = dim // 2
    inv = 1.0 / (10000 ** (torch.arange(half, device=pos.device, dtype=torch.float32) / half))
    ang = pos.to(torch.float32)[..., None] * inv
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class _SwiGLU(nn.Module):
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.gate = nn.Linear(d, ff, bias=False)
        self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class PageEncoderLayer(nn.Module):
    """Bidirectional self-attention inside one window (+ SwiGLU). Keys outside `mask` are
    invisible; masked query rows still produce (ignored) outputs."""

    def __init__(self, d: int, head_dim: int = 64):
        super().__init__()
        self.h = max(1, d // head_dim)
        self.dh = d // self.h
        self.norm1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.qn = nn.RMSNorm(self.dh)
        self.kn = nn.RMSNorm(self.dh)
        self.o = nn.Linear(d, d, bias=False)
        self.norm2 = nn.RMSNorm(d)
        self.ff = _SwiGLU(d, 2 * d)

    def forward(self, x, mask, cos, sin):
        _, apply_rope = _rope()
        Bn, W, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(Bn, W, 3, self.h, self.dh).unbind(dim=2)
        q, k = apply_rope(self.qn(q), cos, sin), apply_rope(self.kn(k), cos, sin)
        attn_mask = mask[:, None, None, :].expand(Bn, 1, W, W)
        # rows whose window has no valid key would be all -inf → let them see themselves
        attn_mask = attn_mask | torch.eye(W, dtype=torch.bool, device=x.device)[None, None]
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask
        ).transpose(1, 2)
        x = x + self.o(o.reshape(Bn, W, d))
        return x + self.ff(self.norm2(x))


class LatentMemoryWriter(nn.Module):
    """K addressed latents per window, written from two-way context (E31)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg_ref = cfg
        e = int(cfg.token_embedding_dim)
        dw = int(cfg.lm_writer_dim)
        D = int(cfg.lm_latent_dim)
        hl = int(cfg.lm_heads)
        self.context = str(cfg.lm_context)
        self.K = int(cfg.lm_latents)
        self.n_null = 1 if bool(cfg.lm_null_latent) else 0
        Kp = self.K + self.n_null
        self.hl, self.dhl, self.D, self.dw = hl, D // hl, D, dw
        self.W = int(cfg.lm_window)
        self.m = int(cfg.lm_reader_tokens)
        self.g, self.dh = int(cfg.num_kv_heads), int(cfg.head_dim)
        self.rounds = int(cfg.lm_rounds)
        self.competition = bool(cfg.lm_competition)
        std = float(cfg.init_std)

        self.in_proj = nn.Linear(e, dw, bias=False)
        # unit-scale token states: without it tok (0.02) × in_proj (0.02) ≈ 0.002, and the first
        # residual update (≈ window average, ~0.3) wipes each token's identity before any pick
        self.in_norm = nn.RMSNorm(dw)
        n_enc = int(cfg.lm_enc_layers) if self.context == "page_bidir" else 0
        self.enc = nn.ModuleList([PageEncoderLayer(dw) for _ in range(n_enc)])
        self.enc_dh = self.enc[0].dh if n_enc else 64

        # latents: identity (q) + address (window start) — warm, never zero; z0 is RMS-normed so
        # each latent starts at unit scale (its identity is not swamped by the first read)
        self.q = nn.Parameter(torch.randn(Kp, D))
        self.addr = nn.Linear(_ADDR_FEATS, D, bias=False)
        self.z0_norm = nn.RMSNorm(D)
        self.z_norm = nn.RMSNorm(D)
        self.x_norm = nn.RMSNorm(dw)
        self.wq = nn.Linear(D, D, bias=False)
        self.wk = nn.Linear(dw, D, bias=False)
        self.wv = nn.Linear(dw, D, bias=False)
        self.wo = nn.Linear(D, D, bias=False)
        self.qn = nn.RMSNorm(self.dhl)
        self.kn = nn.RMSNorm(self.dhl)
        # per-latent, per-head position prior over the window: soft sub-pages at init
        prior = torch.zeros(Kp, hl, self.W)
        if bool(getattr(cfg, "lm_pos_prior", True)) and self.K > 0:
            t = torch.arange(self.W, dtype=torch.float32)
            sigma = max(self.W / self.K, 1.0)
            for k in range(self.K):
                c = (k + 0.5) * self.W / self.K
                prior[k] = (-0.5 * ((t - c) / sigma) ** 2).clamp(min=-4.0)
        self.prior = nn.Parameter(prior)
        self.z_ffn_norm = nn.RMSNorm(D)
        self.z_ffn = _SwiGLU(D, 2 * D)

        if self.context == "bixt":
            hx = max(1, dw // 64)
            self.hx, self.dhx = hx, dw // hx
            self.xq = nn.Linear(dw, dw, bias=False)
            self.zk = nn.Linear(D, dw, bias=False)
            self.zv = nn.Linear(D, dw, bias=False)
            self.xo = nn.Linear(dw, dw, bias=False)
            self.xz_norm = nn.RMSNorm(D)
            self.x_ffn_norm = nn.RMSNorm(dw)
            self.x_ffn = _SwiGLU(dw, 2 * dw)

        # Reader entries. Values carry *what the latent read* (the heads' read-out of the last
        # round); keys combine that content with the latent's state (identity + address).
        # `to_v_state` (latent state → value) starts at zero: at init the value is content, not
        # the latent's identity, which otherwise makes every row's slots nearly identical.
        self.out_norm = nn.RMSNorm(D)
        self.read_norm = nn.RMSNorm(D)
        self.to_k = nn.Linear(D, self.m * self.g * self.dh, bias=False)
        self.to_k_read = nn.Linear(D, self.m * self.g * self.dh, bias=False)
        self.to_v = nn.Linear(D, self.m * self.g * self.dh, bias=False)
        self.to_v_state = nn.Linear(D, self.m * self.g * self.dh, bias=False)
        self.k_out_norm = nn.RMSNorm(self.dh)
        self.last_diag: dict = {}
        self._last_k_bar = None

    # ------------------------------------------------------------------ helpers
    def participation(self) -> torch.Tensor:
        return sum(p.sum() for p in self.parameters()) * 0.0

    def geometry(self, S: int) -> LMGeometry:
        return lm_geometry(self.cfg_ref, S)

    def _latent_read(self, z, x, mask, W):
        """Latents read tokens. z [Bn,Kp,D], x [Bn,W,dw], mask [Bn,W] → (w [Bn,hl,Kp,W], out [Bn,Kp,D])."""
        Bn, Kp, _ = z.shape
        q = self.qn(self.wq(self.z_norm(z)).view(Bn, Kp, self.hl, self.dhl))
        xn = self.x_norm(x)
        k = self.kn(self.wk(xn).view(Bn, W, self.hl, self.dhl))
        v = self.wv(xn).view(Bn, W, self.hl, self.dhl)
        logits = torch.einsum("bkhd,bwhd->bhkw", q, k) / math.sqrt(self.dhl)
        logits = logits + self.prior[:, :, :W].permute(1, 0, 2)[None].to(logits.dtype)
        m = mask[:, None, None, :]
        if self.competition:
            a = torch.softmax(logits.float(), dim=2)  # compete over latents, per (head, token)
            a = a * m
            w = a / (a.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            w = torch.softmax(logits.float().masked_fill(~m, float("-inf")), dim=-1)
            w = torch.nan_to_num(w, nan=0.0)
        w = w.to(v.dtype)
        read = torch.einsum("bhkw,bwhd->bkhd", w, v).reshape(Bn, Kp, self.D)
        self._last_read = read  # per-head read-out (content), used for the reader's values/keys
        return w, self.wo(read)

    def _token_read(self, x, z):
        """BiXT back-step: tokens read the latents. x [Bn,W,dw], z [Bn,Kp,D] → Δx."""
        Bn, W, _ = x.shape
        Kp = z.shape[1]
        q = self.xq(self.x_norm(x)).view(Bn, W, self.hx, self.dhx).transpose(1, 2)
        zn = self.xz_norm(z)
        k = self.zk(zn).view(Bn, Kp, self.hx, self.dhx).transpose(1, 2)
        v = self.zv(zn).view(Bn, Kp, self.hx, self.dhx).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(Bn, W, self.dw)
        return self.xo(o)

    # ------------------------------------------------------------------ forward
    def forward(self, tok_emb, side, doc, key_valid, pos, rope_theta: float):
        rope_cos_sin, apply_rope = _rope()
        B, S, _ = tok_emb.shape
        geo = self.geometry(S)
        W, n_w, K, m = geo.window, geo.n_windows, self.K, self.m
        tok_c, pick, slot_doc_w, slot_side_w, has = window_pick(side, doc, key_valid, geo.starts, W)
        Bn = B * n_w
        x = self.in_norm(self.in_proj(tok_emb))[:, tok_c].reshape(Bn, W, self.dw)
        mask = pick.reshape(Bn, W)
        if len(self.enc):
            rel = torch.arange(W, device=x.device)[None]
            cos, sin = rope_cos_sin(rel, self.enc_dh, 10000.0, x.dtype)
            for layer in self.enc:
                x = layer(x, mask, cos, sin)
        starts = torch.tensor(geo.starts, device=x.device, dtype=torch.long)
        start_pos = pos[:, starts]  # [B, n_w] position of each window start (per document)
        z = self.q[None].to(x.dtype) + self.addr(_sinusoid(start_pos).to(x.dtype)).reshape(Bn, 1, self.D)
        z = self.z0_norm(z)
        w = None
        for _ in range(max(self.rounds, 1)):
            w, dz = self._latent_read(z, x, mask, W)
            z = z + dz
            z = z + self.z_ffn(self.z_ffn_norm(z))
            if self.context == "bixt":
                x = x + self._token_read(x, z)
                x = x + self.x_ffn(self.x_ffn_norm(x))
        z = z[:, :K]
        read = self.read_norm(self._last_read[:, :K])
        w_real = w[:, :, :K].float()  # [Bn, hl, K, W]

        # address: expected position of what each latent read (mean over heads), detached
        w_avg = w_real.mean(dim=1)
        rel_idx = torch.arange(W, device=x.device, dtype=torch.float32)
        mass = w_avg.sum(-1).clamp(min=1e-6)
        p_rel = ((w_avg * rel_idx).sum(-1) / mass).detach().round().long().clamp(0, W - 1)
        p_abs = start_pos.reshape(Bn, 1) + p_rel  # [Bn, K]

        zo = self.out_norm(z)
        kk = self.k_out_norm((self.to_k(zo) + self.to_k_read(read)).view(Bn, K, m, self.g, self.dh))
        vv = (self.to_v(read) + self.to_v_state(zo)).view(Bn, K, m, self.g, self.dh)
        C = n_w * K * m
        kk = kk.reshape(B, C, self.g, self.dh)
        vv = vv.reshape(B, C, self.g, self.dh)
        slot_pos = p_abs.reshape(B, n_w, K, 1).expand(B, n_w, K, m).reshape(B, C)
        cos_s, sin_s = rope_cos_sin(slot_pos, self.dh, rope_theta, kk.dtype)
        kk = apply_rope(kk, cos_s, sin_s)
        slot_doc = slot_doc_w.repeat_interleave(K * m, dim=1)
        slot_side = slot_side_w.repeat_interleave(K * m, dim=1)
        self._last_k_bar = kk
        self.last_diag = self._diagnostics(w, mask, geo)
        return {
            "k": kk, "v": vv, "slot_doc": slot_doc, "slot_side": slot_side, "slot_pos": slot_pos,
            "window_pick": pick,
        }

    @torch.no_grad()
    def _diagnostics(self, w, mask, geo) -> dict:
        """Per-head entropy / log(valid tokens), head diversity, latent usage (last round)."""
        wf = w.detach().float()[:, :, : self.K]  # [Bn,hl,K,W]
        live = mask.any(-1)
        if not bool(live.any()):
            return {}
        wf, maskf = wf[live], mask[live]
        n_valid = maskf.sum(-1).clamp(min=2).float()  # [Bl]
        p = wf / wf.sum(-1, keepdim=True).clamp(min=1e-12)
        ent = -(p * (p + 1e-12).log()).sum(-1)  # [Bl,hl,K]
        ent_ratio = (ent / n_valid.log()[:, None, None]).mean()
        # head diversity: 1 − mean pairwise cosine between heads' distributions per latent
        pn = F.normalize(p, dim=-1)
        sim = torch.einsum("bhkw,bgkw->bkhg", pn, pn)
        hl = sim.shape[-1]
        off = (sim.sum((-1, -2)) - sim.diagonal(dim1=-2, dim2=-1).sum(-1)) / max(hl * (hl - 1), 1)
        diversity = float(1.0 - off.mean())
        usage = wf.sum(-1).mean((0, 1))  # mass each latent takes (competition: share of tokens)
        return {
            "entropy_over_log_valid": float(ent_ratio),
            "entropy_over_logW": float(ent_ratio),  # probe print alias
            "head_diversity": diversity,
            "latent_usage_min": float(usage.min()),
            "latent_usage_max": float(usage.max()),
            "n_windows": geo.n_windows,
            "n_slots": geo.n_slots,
            "window": geo.window,
            "stride": geo.stride,
            "latents": geo.latents,
            "reader_tokens": geo.reader_tokens,
            "tokens_per_latent": geo.tokens_per_latent,
        }
