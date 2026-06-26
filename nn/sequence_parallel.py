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

    Concepts (queries, derived from replicated r_lat) are present on every rank; tokens
    (keys/values) are sharded along N. The caller computes the local score shard
    ``S = (r_lat @ r_tok_shard^T) * scale`` ([B,h,C,n]) and passes it with the local
    value shard ``v`` ([B,h,n,d]). This Function:

      forward:  m = all_reduce(amax(S), MAX);  e = exp(S - m);  z = all_reduce(sum(e), SUM);
                A = e / z (GLOBAL softmax over tokens);  out = all_reduce(A @ v, SUM)
                -> out ([B,h,C,d]) is replicated.
      backward: dout is identical on every rank (out feeds only replicated concept ops),
                so dA = dout @ v^T and dv = A^T @ dout are locally correct; the softmax
                derivative needs the GLOBAL g = sum(A*dA) -> one all_reduce(SUM); dS = A*(dA-g).

    Numerically equivalent to single-GPU softmax(S)@v (verified, gloo, CPU). When ``pg``
    is None (single rank) it collapses to plain local softmax attention with no collectives.
    """

    @staticmethod
    def forward(ctx, S, v, pg):
        m = S.amax(dim=-1, keepdim=True)                 # [B,h,C,1] local max
        if pg is not None:
            dist.all_reduce(m, op=dist.ReduceOp.MAX, group=pg)
        e = torch.exp(S - m)                             # [B,h,C,n]
        z = e.sum(dim=-1, keepdim=True)                  # [B,h,C,1]
        if pg is not None:
            dist.all_reduce(z, op=dist.ReduceOp.SUM, group=pg)
        A = e / z                                        # global softmax weights
        out = torch.matmul(A, v)                         # [B,h,C,d] local partial
        if pg is not None:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=pg)
        ctx.save_for_backward(A, v)
        ctx.pg = pg
        return out

    @staticmethod
    def backward(ctx, dout):
        A, v = ctx.saved_tensors
        pg = ctx.pg
        dA = torch.matmul(dout, v.transpose(-2, -1))     # [B,h,C,n]
        dv = torch.matmul(A.transpose(-2, -1), dout)     # [B,h,n,d]
        g = (A * dA).sum(dim=-1, keepdim=True)           # [B,h,C,1] local
        if pg is not None:
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=pg)
        dS = A * (dA - g)                                 # global softmax derivative
        return dS, dv, None


def local_tok_lat_attention(S: torch.Tensor, v_lat: torch.Tensor) -> torch.Tensor:
    """tok<-lat direction: tokens (sharded) attend to concepts (replicated, full C).

    Local softmax over the concept axis (no collective): A_tok = softmax(S^T, dim=C);
    tok_out = A_tok @ v_lat. ``S`` is the SAME local score shard used for lat<-tok
    ([B,h,C,n]); ``v_lat`` is the replicated value ([B,h,C,d]). Returns [B,h,n,d].
    """
    S_T = S.transpose(-2, -1)                            # [B,h,n,C]
    A_tok = torch.softmax(S_T, dim=-1)
    return torch.matmul(A_tok, v_lat)                    # [B,h,n,d]
