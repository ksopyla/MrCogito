"""Symmetric encoder-decoder baseline for the BAPO capability ladder.

The decoder's self-attention is confined to the suffix (from the `answer` marker onward).
Everything the prefix contains must cross as encoder memory via cross-attention. That is a
BAPO with attention bandwidth `b = 0` on the answer-emitting stream and prefix bandwidth `a`
equal to the encoder's KV (no pooling: full prefix memory).

Used as a matched-parameter control next to `perceiver_ar` dense decoder-only. Not a training
family — the shared entrypoint does not register it. Selected from
`verification/bapo_capability_probe.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput


@dataclass
class EncDecConfig:
    vocab_size: int
    hidden_size: int = 64
    intermediate_size: int = 128
    enc_layers: int = 2
    dec_layers: int = 2
    num_attention_heads: int = 2
    head_dim: int = 32
    answer_start: int = 0
    pad_token_id: int = 0
    init_std: float = 0.02


class _MHA(nn.Module):
    def __init__(self, d: int, n_heads: int, head_dim: int):
        super().__init__()
        self.h, self.dh = n_heads, head_dim
        self.wq = nn.Linear(d, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(d, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # attn_mask: True = keep, shape [B, 1, Q, K] or broadcastable.
        B, Q, _ = q_in.shape
        K = kv_in.shape[1]
        q = self.q_norm(self.wq(q_in).view(B, Q, self.h, self.dh)).transpose(1, 2)
        k = self.k_norm(self.wk(kv_in).view(B, K, self.h, self.dh)).transpose(1, 2)
        v = self.wv(kv_in).view(B, K, self.h, self.dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.wo(out.transpose(1, 2).contiguous().view(B, Q, self.h * self.dh))


class _SwiGLU(nn.Module):
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.gate = nn.Linear(d, ff, bias=False)
        self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _EncBlock(nn.Module):
    def __init__(self, cfg: EncDecConfig):
        super().__init__()
        self.n1 = nn.RMSNorm(cfg.hidden_size)
        self.attn = _MHA(cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim)
        self.n2 = nn.RMSNorm(cfg.hidden_size)
        self.mlp = _SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x), self.n1(x), mask)
        return x + self.mlp(self.n2(x))


class _DecBlock(nn.Module):
    def __init__(self, cfg: EncDecConfig):
        super().__init__()
        d = cfg.hidden_size
        self.n_self = nn.RMSNorm(d)
        self.self_attn = _MHA(d, cfg.num_attention_heads, cfg.head_dim)
        self.n_cross = nn.RMSNorm(d)
        self.cross_attn = _MHA(d, cfg.num_attention_heads, cfg.head_dim)
        self.n_mlp = nn.RMSNorm(d)
        self.mlp = _SwiGLU(d, cfg.intermediate_size)

    def forward(self, x: torch.Tensor, enc: torch.Tensor, self_mask: torch.Tensor, cross_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.n_self(x), self.n_self(x), self_mask)
        x = x + self.cross_attn(self.n_cross(x), enc, cross_mask)
        return x + self.mlp(self.n_mlp(x))


class EncoderDecoderLM(nn.Module):
    def __init__(self, cfg: EncDecConfig):
        super().__init__()
        self.config = cfg
        d = cfg.hidden_size
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.enc = nn.ModuleList([_EncBlock(cfg) for _ in range(cfg.enc_layers)])
        self.dec = nn.ModuleList([_DecBlock(cfg) for _ in range(cfg.dec_layers)])
        self.final = nn.RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.config.init_std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.config.init_std)

    @staticmethod
    def _sinusoidal(S: int, d: int, device, dtype) -> torch.Tensor:
        """Absolute sinusoidal positions. Without these the encoder is a bag of tokens
        and positional far_copy is unlearnable (the first tiny run sat at ~31%)."""
        pe = torch.zeros(S, d, device=device, dtype=dtype)
        pos = torch.arange(S, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_per_token_loss: bool = False,
        return_logits: bool = False,
    ):
        cfg = self.config
        B, S = input_ids.shape
        a0 = cfg.answer_start  # first supervised token
        marker = a0 - 1        # answer marker; decoder starts here
        pe = self._sinusoidal(S, cfg.hidden_size, input_ids.device, self.tok.weight.dtype)
        x = self.tok(input_ids) + pe.unsqueeze(0)
        # Encoder: bidirectional on the prefix (everything before the answer marker).
        prefix = x[:, :marker]
        P = prefix.shape[1]
        enc_mask = torch.ones(B, 1, P, P, dtype=torch.bool, device=x.device)
        enc = prefix
        for blk in self.enc:
            enc = blk(enc, enc_mask)
        # Decoder: causal on the suffix, no self-attn into the prefix.
        suffix = x[:, marker:]
        Q = suffix.shape[1]
        q = torch.arange(Q, device=x.device)
        self_mask = (q[:, None] >= q[None, :])[None, None]
        cross_mask = torch.ones(B, 1, Q, P, dtype=torch.bool, device=x.device)
        h = suffix
        for blk in self.dec:
            h = blk(h, enc, self_mask, cross_mask)
        h = self.final(h)
        logits_suf = self.lm_head(h).float()

        # Stitch suffix logits into a full-sequence tensor so the probe's evaluate() can
        # argmax at the same label positions as the causal LMs.
        logits = x.new_zeros(B, S, cfg.vocab_size)
        logits[:, marker:] = logits_suf

        if labels is None:
            return CausalLMOutput(loss=None, logits=logits)

        # Position t predicts labels[t+1], matching PerceiverARLM.
        tgt = labels[:, 1:]
        hid_full = x.new_zeros(B, S, cfg.hidden_size)
        hid_full[:, marker:] = h
        hid = hid_full[:, :-1]
        per = F.cross_entropy(
            self.lm_head(hid).float().reshape(-1, cfg.vocab_size),
            tgt.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, S - 1)
        valid = tgt != -100
        loss = per[valid].sum() / valid.sum().clamp(min=1)
        if return_per_token_loss:
            out = CausalLMOutput(loss=loss, logits=(logits if return_logits else None))
            return out, per, valid
        if return_logits:
            return CausalLMOutput(loss=loss, logits=logits)
        return CausalLMOutput(loss=loss, logits=None)
