"""Sequence-parallel collectives for long-context Concept Encoder training.

Shards the token axis (N) across GPUs within one node so per-GPU activation memory
scales as O(N / P). The concept bottleneck (C concepts, replicated) makes the
communication tiny:

- Encoder (BiXT, concepts <-> tokens): only the lat<-tok direction needs a GLOBAL
  softmax over the sharded token axis. ``DistLatTokAttention`` does it with two
  all-reduces of ``[B,h,C,1]`` (max, sum) for the numerically-stable normaliser and
  one all-reduce of ``[B,h,C,d]`` for the output; backward adds one all-reduce for the
  softmax g-term. The tok<-lat direction, concept self-attn and all FFNs are LOCAL
  (concepts replicated, tokens sharded).
- Concepts are reproduced on every rank by the output all-reduce, so the decoder sees
  the full ``[B,C,H]`` with no extra sync.

The math is backend-agnostic: NCCL (GPU) for real runs, gloo/CPU for unit tests.
``pg`` defaults to ``dist.group.WORLD``; pass a subgroup to decouple sequence-parallel
groups from data-parallel groups.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def seq_parallel_pg():
    """Process group for sequence-parallel collectives (default: world)."""
    return dist.group.WORLD


def seq_parallel_world_size(pg=None) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    return dist.get_world_size(group=pg or seq_parallel_pg())


def seq_parallel_rank(pg=None) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    return dist.get_rank(group=pg or seq_parallel_pg())


def maybe_split_tokens(t: torch.Tensor, dim: int, rank: int, world: int) -> torch.Tensor:
    """Return this rank's contiguous shard of ``t`` along ``dim`` (token axis)."""
    if world == 1:
        return t
    n = t.size(dim)
    assert n % world == 0, f"token axis {n} not divisible by world {world}"
    s = n // world
    return t.narrow(dim, rank * s, s).contiguous()


def maybe_pad_tokens(t: torch.Tensor, dim: int, world: int) -> torch.Tensor:
    """Right-pad ``t`` along ``dim`` so it divides ``world`` (returns ``t`` if already)."""
    if world == 1:
        return t
    n = t.size(dim)
    rem = n % world
    if rem == 0:
        return t
    pad = world - rem
    pads = [0, 0] * t.dim()
    # F.pad takes dims in reverse; index for `dim`
    pads[(t.dim() - 1 - dim) * 2 + 1] = pad
    return torch.nn.functional.pad(t, pads)


class DistLatTokAttention(torch.autograd.Function):
    """Global-softmax cross-attention for the lat<-tok direction with sharded KEYS.

    Concepts (queries, ``r_lat``) are replicated on every rank; tokens (keys ``r_tok``,
    values ``v_tok``) are sharded along N. The score ``S = (r_lat @ r_tok^T) * scale`` is
    computed here so the backward can route grads to BOTH sides:

      forward:  global softmax over the sharded token axis (all_reduce MAX then SUM for a
                numerically-stable normaliser); out = all_reduce(A @ v_tok, SUM) -> replicated.
      backward: dA = dout @ v_tok^T; dv_tok = A^T @ dout; global g = all_reduce(sum(A*dA));
                dS = A*(dA-g); then d(r_lat) = scale*(dS @ r_tok) and d(r_tok) = scale*(dS^T @ r_lat).
                d(r_lat) is the CONCEPT-side grad and is rank-dependent (it mixes the local
                r_tok shard) -> all_reduce(SUM) here so the replicated concepts receive the FULL
                cross-shard contribution. d(r_tok) / dv_tok are token-side (local shard), no sync.

    This self-summation of the concept-side grad means the residual chain through the encoder
    stays "full" with a SINGLE output-side concepts-grad barrier (no per-layer barriers), and
    every concept-side parameter (incl. rv_lat) becomes correctly averageable across ranks.
    """

    @staticmethod
    def forward(ctx, r_lat, r_tok, v_tok, scale, key_padding_mask, pg):
        S = (torch.matmul(r_lat, r_tok.transpose(-2, -1))) * scale      # [B,h,C,n]
        if key_padding_mask is not None:
            S = S.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        m = S.amax(dim=-1, keepdim=True)                                # [B,h,C,1]
        if pg is not None:
            dist.all_reduce(m, op=dist.ReduceOp.MAX, group=pg)
        e = torch.exp(S - m)                                            # [B,h,C,n]
        z = e.sum(dim=-1, keepdim=True)                                 # [B,h,C,1]
        if pg is not None:
            dist.all_reduce(z, op=dist.ReduceOp.SUM, group=pg)
        A = e / z                                                       # global softmax weights
        out = torch.matmul(A, v_tok)                                    # [B,h,C,d] partial
        if pg is not None:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=pg)
        ctx.save_for_backward(A, r_lat, r_tok, v_tok)
        ctx.scale = scale
        ctx.pg = pg
        return out

    @staticmethod
    def backward(ctx, dout):
        A, r_lat, r_tok, v_tok = ctx.saved_tensors
        pg = ctx.pg
        scale = ctx.scale
        dA = torch.matmul(dout, v_tok.transpose(-2, -1))                # [B,h,C,n]
        dv_tok = torch.matmul(A.transpose(-2, -1), dout)                # [B,h,n,d]
        g = (A * dA).sum(dim=-1, keepdim=True)                          # [B,h,C,1]
        if pg is not None:
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=pg)
        dS = A * (dA - g)                                               # [B,h,C,n]
        d_r_lat = torch.matmul(dS, r_tok) * scale                       # [B,h,C,d] concept-side
        if pg is not None:
            dist.all_reduce(d_r_lat, op=dist.ReduceOp.SUM, group=pg)    # full cross-shard grad
        d_r_tok = torch.matmul(dS.transpose(-2, -1), r_lat) * scale     # [B,h,n,d] token-side
        return d_r_lat, d_r_tok, dv_tok, None, None, None


def local_tok_lat_attention(S: torch.Tensor, v_lat: torch.Tensor) -> torch.Tensor:
    """tok<-lat direction: tokens (sharded) attend to concepts (replicated, full C).

    Local softmax over the concept axis (no collective): A_tok = softmax(S^T, dim=C);
    tok_out = A_tok @ v_lat. ``S`` is the SAME local score shard used for lat<-tok
    ([B,h,C,n]); ``v_lat`` is the replicated value ([B,h,C,d]). Returns [B,h,n,d].
    Plain (non-distributed) helper; the sequence-parallel path uses LocalTokLatAttention
    so the concept-side grads are all-reduced.
    """
    S_T = S.transpose(-2, -1)                            # [B,h,n,C]
    A_tok = torch.softmax(S_T, dim=-1)
    return torch.matmul(A_tok, v_lat)                    # [B,h,n,d]


class LocalTokLatAttention(torch.autograd.Function):
    """tok<-lat direction in sequence-parallel mode.

    Tokens (sharded) attend to the replicated concepts via a LOCAL softmax over C. The
    concept-side gradients (``d r_lat`` from S = r_lat @ r_tok^T, and ``d v_lat`` from the
    value) both mix the local token shard, so they are rank-dependent and are all-reduced
    (SUM) here — otherwise concept-side parameters that feed rv_lat (pre_cross_norm_lat,
    rv_lat itself, concept_embeddings, ...) would receive a grad that is neither a clean
    per-shard sum nor a clean replicated average. The token-side grads (d r_tok) stay local.
    """

    @staticmethod
    def forward(ctx, r_lat, r_tok, v_lat, scale, pg):
        S = (torch.matmul(r_lat, r_tok.transpose(-2, -1))) * scale      # [B,h,C,n]
        A_tok = torch.softmax(S.transpose(-2, -1), dim=-1)              # [B,h,n,C]
        tok_out = torch.matmul(A_tok, v_lat)                            # [B,h,n,d]
        ctx.save_for_backward(A_tok, r_lat, v_lat, r_tok)
        ctx.scale = scale
        ctx.pg = pg
        return tok_out

    @staticmethod
    def backward(ctx, d_tok_out):
        A_tok, r_lat, v_lat, r_tok = ctx.saved_tensors
        pg, scale = ctx.pg, ctx.scale
        dA_tok = torch.matmul(d_tok_out, v_lat.transpose(-2, -1))       # [B,h,n,C]
        dv_lat = torch.matmul(A_tok.transpose(-2, -1), d_tok_out)       # [B,h,C,d] concept-side
        if pg is not None:
            dist.all_reduce(dv_lat, op=dist.ReduceOp.SUM, group=pg)
        g_tok = (A_tok * dA_tok).sum(dim=-1, keepdim=True)             # over C (local)
        dS = (A_tok * (dA_tok - g_tok)).transpose(-2, -1)              # [B,h,C,n]
        d_r_lat = torch.matmul(dS, r_tok) * scale                      # [B,h,C,d] concept-side
        if pg is not None:
            dist.all_reduce(d_r_lat, op=dist.ReduceOp.SUM, group=pg)
        d_r_tok = torch.matmul(dS.transpose(-2, -1), r_lat) * scale    # [B,h,n,d] token-side
        return d_r_lat, d_r_tok, dv_lat, None, None


class AllReduceGrad(torch.autograd.Function):
    """Identity in forward; all-reduce(SUM) the gradient in backward.

    Insert on the REPLICATED concepts (between encoder and decoder) so the encoder
    sees the FULL concepts gradient — the SUM of every decoder shard's partial
    d(loss)/d(concepts). Without this, token-side encoder params miss the cross-shard
    terms (rank s's decoder loss depends on rank r's tokens via the shared concepts),
    which is the classic sequence-parallel + replicated-bottleneck gradient trap.
    """

    @staticmethod
    def forward(ctx, x, pg):
        ctx.pg = pg
        return x

    @staticmethod
    def backward(ctx, grad):
        if ctx.pg is not None:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.pg)
        return grad, None


# Encoder parameters whose forward is REPLICATED in sequence-parallel mode (they operate
# on the replicated concepts, identical on every rank). Their gradient is already correct
# per-rank once the concepts-grad barrier is in place, so they must be AVERAGED (not
# summed) across ranks — summing would over-count by world_size. Everything else
# (tokens / suffix / decoder / lm_head) is SHARDED and must be SUMMED.
_CONCEPT_SIDE_MARKERS = (
    "concept_embeddings", "output_layer_norm",
    "rv_lat", "proj_lat", "Wi_lat", "Wo_lat", "concept_self_attn",
    "pre_cross_norm_lat", "pre_self_attn_norm", "pre_ff_norm_lat",
)


def is_concept_side_param(name: str) -> bool:
    """True for encoder parameters that run replicated (on the concepts) in SP mode."""
    return any(m in name for m in _CONCEPT_SIDE_MARKERS)


def sync_seq_parallel_grads(model, pg=None):
    """All-reduce sequence-parallel gradients with the split rule:

    - token / suffix / decoder / lm_head params (sharded forward) -> SUM over ranks.
    - concept-side encoder params (replicated forward)            -> AVG over ranks
      (they are identical post-barrier, so averaging is a sync no-op that avoids the
      world_size over-count a plain SUM would introduce).

    Call after ``loss.backward()`` (replaces DDP's gradient sync for the SP path).
    """
    if pg is None:
        pg = seq_parallel_pg()
    world = dist.get_world_size(group=pg)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=pg)
        if is_concept_side_param(name):
            p.grad.div_(world)
